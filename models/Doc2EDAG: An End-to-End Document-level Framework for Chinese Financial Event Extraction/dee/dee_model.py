# -*- coding: utf-8 -*-
# AUTHOR: Shun Zheng
# DATE: 19-9-19

import torch
from torch import nn
import torch.nn.functional as F
import math
from collections import OrderedDict, namedtuple, defaultdict
import random

from . import transformer
from .ner_model import NERModel


DocSpanInfo = namedtuple(
    'DocSpanInfo', (
        'span_token_tup_list',  # [(span_token_id, ...), ...], num_spans
        'span_dranges_list',  # [[(sent_idx, char_s, char_e), ...], ...], num_spans
        'span_mention_range_list',  # [(mention_idx_s, mention_idx_e), ...], num_spans
        'mention_drange_list',  # [(sent_idx, char_s, char_e), ...], num_mentions
        'mention_type_list',  # [mention_type_id, ...], num_mentions
        'event_dag_info',  # event_idx -> field_idx -> pre_path -> cur_span_idx_set
        'missed_sent_idx_list',  # index list of sentences where gold spans are not extracted
    )
)


def get_doc_span_info_list(doc_token_types_list, doc_fea_list, use_gold_span=False):
    assert len(doc_token_types_list) == len(doc_fea_list)
    doc_span_info_list = []
    for doc_token_types, doc_fea in zip(doc_token_types_list, doc_fea_list):
        doc_token_type_mat = doc_token_types.tolist()  # [[token_type, ...], ...]

        # using extracted results is also ok
        # span_token_tup_list, span_dranges_list = extract_doc_valid_span_info(doc_token_type_mat, doc_fea)
        if use_gold_span:
            span_token_tup_list = doc_fea.span_token_ids_list
            span_dranges_list = doc_fea.span_dranges_list
        else:
            span_token_tup_list, span_dranges_list = extract_doc_valid_span_info(doc_token_type_mat, doc_fea)

            if len(span_token_tup_list) == 0:
                # do not get valid entity span results,
                # just use gold spans to avoid crashing at earlier iterations
                # TODO: consider generate random negative spans
                span_token_tup_list = doc_fea.span_token_ids_list
                span_dranges_list = doc_fea.span_dranges_list

        # one span may have multiple mentions
        span_mention_range_list, mention_drange_list, mention_type_list = get_span_mention_info(
            span_dranges_list, doc_token_type_mat
        )

        # generate event decoding dag graph for model training
        event_dag_info, _, missed_sent_idx_list = doc_fea.generate_dag_info_for(span_token_tup_list, return_miss=True)

        # doc_span_info will incorporate all span-level information needed for the event extraction
        doc_span_info = DocSpanInfo(
            span_token_tup_list, span_dranges_list, span_mention_range_list,
            mention_drange_list, mention_type_list,
            event_dag_info, missed_sent_idx_list,
        )

        doc_span_info_list.append(doc_span_info)

    return doc_span_info_list


class Doc2EDAGModel(nn.Module):
    """Document-level Event Extraction Model"""

    def __init__(self, config, event_type_fields_pairs, ner_model=None):
        super(Doc2EDAGModel, self).__init__()
        # Note that for distributed training, you must ensure that
        # for any batch, all parameters need to be used

        self.config = config
        self.event_type_fields_pairs = event_type_fields_pairs

        if ner_model is None:
            self.ner_model = NERModel(config)
        else:
            self.ner_model = ner_model

        # all event tables
        self.event_tables = nn.ModuleList([
            EventTable(event_type, field_types, config.hidden_size)
            for event_type, field_types in self.event_type_fields_pairs
        ])

        # sentence position indicator
        self.sent_pos_encoder = SentencePosEncoder(
            config.hidden_size, max_sent_num=config.max_sent_num, dropout=config.dropout
        )

        if self.config.use_token_role:
            self.ment_type_encoder = MentionTypeEncoder(
                config.hidden_size, config.num_entity_labels, dropout=config.dropout
            )

        # various attentive reducer
        if self.config.seq_reduce_type == 'AWA':
            self.doc_token_reducer = AttentiveReducer(config.hidden_size, dropout=config.dropout)
            self.span_token_reducer = AttentiveReducer(config.hidden_size, dropout=config.dropout)
            self.span_mention_reducer = AttentiveReducer(config.hidden_size, dropout=config.dropout)
        else:
            assert self.config.seq_reduce_type in {'MaxPooling', 'MeanPooling'}

        if self.config.use_doc_enc:
            # get doc-level context information for every mention and sentence
            self.doc_context_encoder = transformer.make_transformer_encoder(
                config.num_tf_layers, config.hidden_size, ff_size=config.ff_size, dropout=config.dropout
            )

        if self.config.use_path_mem:
            # get field-specific and history-aware information for every span
            self.field_context_encoder = transformer.make_transformer_encoder(
                config.num_tf_layers, config.hidden_size, ff_size=config.ff_size, dropout=config.dropout
            )

    def get_doc_span_mention_emb(self, doc_token_emb, doc_span_info):
        if len(doc_span_info.mention_drange_list) == 0:
            doc_mention_emb = None
        else:
            # get mention context embeding
            # doc_mention_emb = torch.cat([
            #     # doc_token_emb[sent_idx, char_s:char_e, :].sum(dim=0, keepdim=True)
            #     doc_token_emb[sent_idx, char_s:char_e, :].max(dim=0, keepdim=True)[0]
            #     for sent_idx, char_s, char_e in doc_span_info.mention_drange_list
            # ])
            mention_emb_list = []
            for sent_idx, char_s, char_e in doc_span_info.mention_drange_list:
                mention_token_emb = doc_token_emb[sent_idx, char_s: char_e, :]  # [num_mention_tokens, hidden_size]
                if self.config.seq_reduce_type == 'AWA':
                    mention_emb = self.span_token_reducer(mention_token_emb)  # [hidden_size]
                elif self.config.seq_reduce_type == 'MaxPooling':
                    mention_emb = mention_token_emb.max(dim=0)[0]
                elif self.config.seq_reduce_type == 'MeanPooling':
                    mention_emb = mention_token_emb.mean(dim=0)
                else:
                    raise Exception('Unknown seq_reduce_type {}'.format(self.config.seq_reduce_type))
                mention_emb_list.append(mention_emb)
            doc_mention_emb = torch.stack(mention_emb_list, dim=0)

            # add sentence position embedding
            mention_sent_id_list = [drange[0] for drange in doc_span_info.mention_drange_list]
            doc_mention_emb = self.sent_pos_encoder(doc_mention_emb, sent_pos_ids=mention_sent_id_list)

            if self.config.use_token_role:
                # get mention type embedding
                doc_mention_emb = self.ment_type_encoder(doc_mention_emb, doc_span_info.mention_type_list)

        return doc_mention_emb

    def get_batch_sent_emb(self, ner_token_emb, ner_token_masks, valid_sent_num_list):
        # From [ner_batch_size, sent_len, hidden_size] to [ner_batch_size, hidden_size]
        if self.config.seq_reduce_type == 'AWA':
            total_sent_emb = self.doc_token_reducer(ner_token_emb, masks=ner_token_masks)
        elif self.config.seq_reduce_type == 'MaxPooling':
            total_sent_emb = ner_token_emb.max(dim=1)[0]
        elif self.config.seq_reduce_type == 'MeanPooling':
            total_sent_emb = ner_token_emb.mean(dim=1)
        else:
            raise Exception('Unknown seq_reduce_type {}'.format(self.config.seq_reduce_type))

        total_sent_pos_ids = []
        for valid_sent_num in valid_sent_num_list:
            total_sent_pos_ids += list(range(valid_sent_num))
        total_sent_emb = self.sent_pos_encoder(total_sent_emb, sent_pos_ids=total_sent_pos_ids)

        return total_sent_emb

    def get_doc_span_sent_context(self, doc_token_emb, doc_sent_emb, doc_fea, doc_span_info):
        doc_mention_emb = self.get_doc_span_mention_emb(doc_token_emb, doc_span_info)

        # only consider actual sentences
        if doc_sent_emb.size(0) > doc_fea.valid_sent_num:
            doc_sent_emb = doc_sent_emb[:doc_fea.valid_sent_num, :]

        span_context_list = []

        if doc_mention_emb is None:
            if self.config.use_doc_enc:
                doc_sent_context = self.doc_context_encoder(doc_sent_emb.unsqueeze(0), None).squeeze(0)
            else:
                doc_sent_context = doc_sent_emb
        else:
            num_mentions = doc_mention_emb.size(0)

            if self.config.use_doc_enc:
                # Size([1, num_mentions + num_valid_sents, hidden_size])
                total_ment_sent_emb = torch.cat([doc_mention_emb, doc_sent_emb], dim=0).unsqueeze(0)

                # size = [num_mentions+num_valid_sents, hidden_size]
                # here we do not need mask
                total_ment_sent_context = self.doc_context_encoder(total_ment_sent_emb, None).squeeze(0)

                # collect span context
                for mid_s, mid_e in doc_span_info.span_mention_range_list:
                    assert mid_e <= num_mentions
                    multi_ment_context = total_ment_sent_context[mid_s:mid_e]  # [num_mentions, hidden_size]

                    # span_context.size [1, hidden_size]
                    if self.config.seq_reduce_type == 'AWA':
                        span_context = self.span_mention_reducer(multi_ment_context, keepdim=True)
                    elif self.config.seq_reduce_type == 'MaxPooling':
                        span_context = multi_ment_context.max(dim=0, keepdim=True)[0]
                    elif self.config.seq_reduce_type == 'MeanPooling':
                        span_context = multi_ment_context.mean(dim=0, keepdim=True)
                    else:
                        raise Exception('Unknown seq_reduce_type {}'.format(self.config.seq_reduce_type))

                    span_context_list.append(span_context)

                # collect sent context
                doc_sent_context = total_ment_sent_context[num_mentions:, :]
            else:
                # collect span context
                for mid_s, mid_e in doc_span_info.span_mention_range_list:
                    assert mid_e <= num_mentions
                    multi_ment_emb = doc_mention_emb[mid_s:mid_e]  # [num_mentions, hidden_size]

                    # span_context.size is [1, hidden_size]
                    if self.config.seq_reduce_type == 'AWA':
                        span_context = self.span_mention_reducer(multi_ment_emb, keepdim=True)
                    elif self.config.seq_reduce_type == 'MaxPooling':
                        span_context = multi_ment_emb.max(dim=0, keepdim=True)[0]
                    elif self.config.seq_reduce_type == 'MeanPooling':
                        span_context = multi_ment_emb.mean(dim=0, keepdim=True)
                    else:
                        raise Exception('Unknown seq_reduce_type {}'.format(self.config.seq_reduce_type))
                    span_context_list.append(span_context)

                # collect sent context
                doc_sent_context = doc_sent_emb

        return span_context_list, doc_sent_context

    def get_event_cls_info(self, sent_context_emb, doc_fea, train_flag=True):
        doc_event_logps = []
        for event_idx, event_label in enumerate(doc_fea.event_type_labels):
            event_table = self.event_tables[event_idx]
            cur_event_logp = event_table(sent_context_emb=sent_context_emb)  # [1, hidden_size]
            doc_event_logps.append(cur_event_logp)
        doc_event_logps = torch.cat(doc_event_logps, dim=0)  # [num_event_types, 2]

        if train_flag:
            device = doc_event_logps.device
            doc_event_labels = torch.tensor(
                doc_fea.event_type_labels, device=device, dtype=torch.long, requires_grad=False
            )  # [num_event_types]
            doc_event_cls_loss = F.nll_loss(doc_event_logps, doc_event_labels, reduction='sum')
            return doc_event_cls_loss
        else:
            doc_event_pred_list = doc_event_logps.argmax(dim=-1).tolist()
            return doc_event_pred_list

    def get_field_cls_info(self, event_idx, field_idx, batch_span_emb,
                           batch_span_label=None, train_flag=True):
        batch_span_logp = self.get_field_pred_logp(event_idx, field_idx, batch_span_emb)

        if train_flag:
            assert batch_span_label is not None
            device = batch_span_logp.device
            data_type = batch_span_logp.dtype
            # to prevent too many FPs
            class_weight = torch.tensor(
                [self.config.neg_field_loss_scaling, 1.0], device=device, dtype=data_type, requires_grad=False
            )
            field_cls_loss = F.nll_loss(batch_span_logp, batch_span_label, weight=class_weight, reduction='sum')
            return field_cls_loss, batch_span_logp
        else:
            span_pred_list = batch_span_logp.argmax(dim=-1).tolist()
            return span_pred_list, batch_span_logp

    def get_field_pred_logp(self, event_idx, field_idx, batch_span_emb, include_prob=False):
        event_table = self.event_tables[event_idx]
        batch_span_logp = event_table(batch_span_emb=batch_span_emb, field_idx=field_idx)

        if include_prob:
            # used for decision sampling, is not inside the computation graph
            batch_span_prob = batch_span_logp.detach().exp()
            return batch_span_logp, batch_span_prob
        else:
            return batch_span_logp

    def get_none_span_context(self, init_tensor):
        none_span_context = torch.zeros(
            1, self.config.hidden_size,
            device=init_tensor.device, dtype=init_tensor.dtype, requires_grad=False
        )
        return none_span_context

    def conduct_field_level_reasoning(self, event_idx, field_idx, prev_decode_context, batch_span_context):
        event_table = self.event_tables[event_idx]
        field_query = event_table.field_queries[field_idx]
        num_spans = batch_span_context.size(0)
        # make the model to be aware of which field
        batch_cand_emb = batch_span_context + field_query
        if self.config.use_path_mem:
            # [1, num_spans + valid_sent_num, hidden_size]
            total_cand_emb = torch.cat([batch_cand_emb, prev_decode_context], dim=0).unsqueeze(0)
            # use transformer to do the reasoning
            total_cand_emb = self.field_context_encoder(total_cand_emb, None).squeeze(0)
            batch_cand_emb = total_cand_emb[:num_spans, :]
        # TODO: what if reasoning over reasoning context
        return batch_cand_emb, prev_decode_context

    def get_field_mle_loss_list(self, doc_sent_context, batch_span_context,
                                event_idx, field_idx2pre_path2cur_span_idx_set):
        field_mle_loss_list = []
        num_fields = self.event_tables[event_idx].num_fields
        num_spans = batch_span_context.size(0)
        prev_path2prev_decode_context = {
            (): doc_sent_context
        }

        for field_idx in range(num_fields):
            prev_path2cur_span_idx_set = field_idx2pre_path2cur_span_idx_set[field_idx]
            for prev_path, cur_span_idx_set in prev_path2cur_span_idx_set.items():
                if prev_path not in prev_path2prev_decode_context:
                    # note that when None and valid_span co-exists, ignore None paths during training
                    continue
                # get decoding context
                prev_decode_context = prev_path2prev_decode_context[prev_path]
                # conduct reasoning on this field
                batch_cand_emb, prev_decode_context = self.conduct_field_level_reasoning(
                    event_idx, field_idx, prev_decode_context, batch_span_context
                )
                # prepare label for candidate spans
                batch_span_label = get_batch_span_label(
                    num_spans, cur_span_idx_set, batch_span_context.device
                )
                # calculate loss
                cur_field_cls_loss, batch_span_logp = self.get_field_cls_info(
                    event_idx, field_idx, batch_cand_emb,
                    batch_span_label=batch_span_label, train_flag=True
                )

                field_mle_loss_list.append(cur_field_cls_loss)

                # cur_span_idx_set needs to ensure at least one element, None
                for span_idx in cur_span_idx_set:
                    # Teacher-forcing Style Training
                    if span_idx is None:
                        span_context = self.event_tables[event_idx].field_queries[field_idx]
                    else:
                        # TODO: add either batch_cand_emb or batch_span_context to the memory tensor
                        span_context = batch_cand_emb[span_idx].unsqueeze(0)

                    cur_path = prev_path + (span_idx, )
                    if self.config.use_path_mem:
                        cur_decode_context = torch.cat([prev_decode_context, span_context], dim=0)
                        prev_path2prev_decode_context[cur_path] = cur_decode_context
                    else:
                        prev_path2prev_decode_context[cur_path] = prev_decode_context

        return field_mle_loss_list

    def get_loss_on_doc(self, doc_token_emb, doc_sent_emb, doc_fea, doc_span_info):
        span_context_list, doc_sent_context = self.get_doc_span_sent_context(
            doc_token_emb, doc_sent_emb, doc_fea, doc_span_info,
        )
        if len(span_context_list) == 0:
            raise Exception('Error: doc_fea.ex_idx {} does not have valid span'.format(doc_fea.ex_idx))

        batch_span_context = torch.cat(span_context_list, dim=0)
        num_spans = len(span_context_list)
        event_idx2field_idx2pre_path2cur_span_idx_set = doc_span_info.event_dag_info

        # 1. get event type classification loss
        event_cls_loss = self.get_event_cls_info(doc_sent_context, doc_fea, train_flag=True)

        # 2. for each event type, get field classification loss
        # Note that including the memory tensor into the computing graph can boost the performance (>1 F1)
        all_field_loss_list = []
        for event_idx, event_label in enumerate(doc_fea.event_type_labels):
            if event_label == 0:
                # treat all spans as invalid arguments for that event,
                # because we need to use all parameters to support distributed training
                prev_decode_context = doc_sent_context
                num_fields = self.event_tables[event_idx].num_fields
                for field_idx in range(num_fields):
                    # conduct reasoning on this field
                    batch_cand_emb, prev_decode_context = self.conduct_field_level_reasoning(
                        event_idx, field_idx, prev_decode_context, batch_span_context
                    )
                    # prepare label for candidate spans
                    batch_span_label = get_batch_span_label(
                        num_spans, set(), batch_span_context.device
                    )
                    # calculate the field loss
                    cur_field_cls_loss, batch_span_logp = self.get_field_cls_info(
                        event_idx, field_idx, batch_cand_emb,
                        batch_span_label=batch_span_label, train_flag=True
                    )
                    # update the memory tensor
                    span_context = self.event_tables[event_idx].field_queries[field_idx]
                    if self.config.use_path_mem:
                        prev_decode_context = torch.cat([prev_decode_context, span_context], dim=0)

                    all_field_loss_list.append(cur_field_cls_loss)
            else:
                field_idx2pre_path2cur_span_idx_set = event_idx2field_idx2pre_path2cur_span_idx_set[event_idx]
                field_loss_list = self.get_field_mle_loss_list(
                    doc_sent_context, batch_span_context, event_idx, field_idx2pre_path2cur_span_idx_set,
                )
                all_field_loss_list += field_loss_list

        total_event_loss = event_cls_loss + sum(all_field_loss_list)
        return total_event_loss

    def get_mix_loss(self, doc_sent_loss_list, doc_event_loss_list, doc_span_info_list):
        batch_size = len(doc_span_info_list)
        loss_batch_avg = 1.0 / batch_size
        lambda_1 = self.config.loss_lambda
        lambda_2 = 1 - lambda_1

        doc_ner_loss_list = []
        for doc_sent_loss, doc_span_info in zip(doc_sent_loss_list, doc_span_info_list):
            # doc_sent_loss: Size([num_valid_sents])
            doc_ner_loss_list.append(doc_sent_loss.sum())

        return loss_batch_avg * (lambda_1 * sum(doc_ner_loss_list) + lambda_2 * sum(doc_event_loss_list))

    def get_eval_on_doc(self, doc_token_emb, doc_sent_emb, doc_fea, doc_span_info):
        span_context_list, doc_sent_context = self.get_doc_span_sent_context(
            doc_token_emb, doc_sent_emb, doc_fea, doc_span_info
        )
        if len(span_context_list) == 0:
            event_pred_list = []
            event_idx2obj_idx2field_idx2token_tup = []
            event_idx2event_decode_paths = []
            for event_idx in range(len(self.event_type_fields_pairs)):
                event_pred_list.append(0)
                event_idx2obj_idx2field_idx2token_tup.append(None)
                event_idx2event_decode_paths.append(None)

            return doc_fea.ex_idx, event_pred_list, event_idx2obj_idx2field_idx2token_tup, \
                doc_span_info, event_idx2event_decode_paths

        batch_span_context = torch.cat(span_context_list, dim=0)

        # 1. get event type prediction
        event_pred_list = self.get_event_cls_info(doc_sent_context, doc_fea, train_flag=False)

        # 2. for each event type, get field prediction
        # the following mappings are all implemented using list index
        event_idx2event_decode_paths = []
        event_idx2obj_idx2field_idx2token_tup = []
        for event_idx, event_pred in enumerate(event_pred_list):
            if event_pred == 0:
                event_idx2event_decode_paths.append(None)
                event_idx2obj_idx2field_idx2token_tup.append(None)
                continue

            num_fields = self.event_tables[event_idx].num_fields

            prev_path2prev_decode_context = {(): doc_sent_context}
            last_field_paths = [()]  # only record paths of the last field
            for field_idx in range(num_fields):
                cur_paths = []
                for prev_path in last_field_paths:  # traverse all previous decoding paths
                    # get decoding context
                    prev_decode_context = prev_path2prev_decode_context[prev_path]
                    # conduct reasoning on this field
                    batch_cand_emb, prev_decode_context = self.conduct_field_level_reasoning(
                        event_idx, field_idx, prev_decode_context, batch_span_context
                    )

                    # get field prediction for all spans
                    span_pred_list, _ = self.get_field_cls_info(
                        event_idx, field_idx, batch_cand_emb, train_flag=False
                    )

                    # prepare span_idx to be used for the next field
                    cur_span_idx_list = []
                    for span_idx, span_pred in enumerate(span_pred_list):
                        if span_pred == 1:
                            cur_span_idx_list.append(span_idx)
                    if len(cur_span_idx_list) == 0:
                        # all span is invalid for this field, just choose 'Unknown' token
                        cur_span_idx_list.append(None)

                    for span_idx in cur_span_idx_list:
                        if span_idx is None:
                            span_context = self.event_tables[event_idx].field_queries[field_idx]
                            # span_context = none_span_context
                        else:
                            span_context = batch_cand_emb[span_idx].unsqueeze(0)

                        cur_path = prev_path + (span_idx, )
                        cur_decode_context = torch.cat([prev_decode_context, span_context], dim=0)
                        cur_paths.append(cur_path)
                        prev_path2prev_decode_context[cur_path] = cur_decode_context

                # update decoding paths
                last_field_paths = cur_paths

            obj_idx2field_idx2token_tup = []
            for decode_path in last_field_paths:
                assert len(decode_path) == num_fields
                field_idx2token_tup = []
                for span_idx in decode_path:
                    if span_idx is None:
                        token_tup = None
                    else:
                        token_tup = doc_span_info.span_token_tup_list[span_idx]

                    field_idx2token_tup.append(token_tup)
                obj_idx2field_idx2token_tup.append(field_idx2token_tup)

            event_idx2event_decode_paths.append(last_field_paths)
            event_idx2obj_idx2field_idx2token_tup.append(obj_idx2field_idx2token_tup)

        # the first three terms are for metric calculation, the last two are for case studies
        return doc_fea.ex_idx, event_pred_list, event_idx2obj_idx2field_idx2token_tup, \
            doc_span_info, event_idx2event_decode_paths

    def adjust_token_label(self, doc_token_labels_list):
        if self.config.use_token_role:  # do not use detailed token
            return doc_token_labels_list
        else:
            adj_doc_token_labels_list = []
            for doc_token_labels in doc_token_labels_list:
                entity_begin_mask = doc_token_labels % 2 == 1
                entity_inside_mask = (doc_token_labels != 0) & (doc_token_labels % 2 == 0)
                adj_doc_token_labels = doc_token_labels.masked_fill(entity_begin_mask, 1)
                adj_doc_token_labels = adj_doc_token_labels.masked_fill(entity_inside_mask, 2)

                adj_doc_token_labels_list.append(adj_doc_token_labels)
            return adj_doc_token_labels_list

    def get_local_context_info(self, doc_batch_dict, train_flag=False, use_gold_span=False):
        label_key = 'doc_token_labels'
        if train_flag or use_gold_span:
            assert label_key in doc_batch_dict
            need_label_flag = True
        else:
            need_label_flag = False

        if need_label_flag:
            doc_token_labels_list = self.adjust_token_label(doc_batch_dict[label_key])
        else:
            doc_token_labels_list = None

        batch_size = len(doc_batch_dict['ex_idx'])
        doc_token_ids_list = doc_batch_dict['doc_token_ids']
        doc_token_masks_list = doc_batch_dict['doc_token_masks']
        valid_sent_num_list = doc_batch_dict['valid_sent_num']

        # transform doc_batch into sent_batch
        ner_batch_idx_start_list = [0]
        ner_token_ids = []
        ner_token_masks = []
        ner_token_labels = [] if need_label_flag else None
        for batch_idx, valid_sent_num in enumerate(valid_sent_num_list):
            idx_start = ner_batch_idx_start_list[-1]
            idx_end = idx_start + valid_sent_num
            ner_batch_idx_start_list.append(idx_end)

            ner_token_ids.append(doc_token_ids_list[batch_idx])
            ner_token_masks.append(doc_token_masks_list[batch_idx])
            if need_label_flag:
                ner_token_labels.append(doc_token_labels_list[batch_idx])

        # [ner_batch_size, norm_sent_len]
        ner_token_ids = torch.cat(ner_token_ids, dim=0)
        ner_token_masks = torch.cat(ner_token_masks, dim=0)
        if need_label_flag:
            ner_token_labels = torch.cat(ner_token_labels, dim=0)

        # get ner output
        ner_token_emb, ner_loss, ner_token_preds = self.ner_model(
            ner_token_ids, ner_token_masks, label_ids=ner_token_labels,
            train_flag=train_flag, decode_flag=not use_gold_span,
        )

        if use_gold_span:  # definitely use gold span info
            ner_token_types = ner_token_labels
        else:
            ner_token_types = ner_token_preds

        # get sentence embedding
        ner_sent_emb = self.get_batch_sent_emb(ner_token_emb, ner_token_masks, valid_sent_num_list)

        assert sum(valid_sent_num_list) == ner_token_emb.size(0) == ner_sent_emb.size(0)

        # followings are all lists of tensors
        doc_token_emb_list = []
        doc_token_masks_list = []
        doc_token_types_list = []
        doc_sent_emb_list = []
        doc_sent_loss_list = []
        for batch_idx in range(batch_size):
            idx_start = ner_batch_idx_start_list[batch_idx]
            idx_end = ner_batch_idx_start_list[batch_idx+1]
            doc_token_emb_list.append(ner_token_emb[idx_start:idx_end, :, :])
            doc_token_masks_list.append(ner_token_masks[idx_start:idx_end, :])
            doc_token_types_list.append(ner_token_types[idx_start:idx_end, :])
            doc_sent_emb_list.append(ner_sent_emb[idx_start:idx_end, :])
            if ner_loss is not None:
                # every doc_sent_loss.size is [valid_sent_num]
                doc_sent_loss_list.append(ner_loss[idx_start:idx_end])

        return doc_token_emb_list, doc_token_masks_list, doc_token_types_list, doc_sent_emb_list, doc_sent_loss_list

    def forward(self, doc_batch_dict, doc_features,
                train_flag=True, use_gold_span=False, teacher_prob=1,
                event_idx2entity_idx2field_idx=None, heuristic_type=None):
        # Using scheduled sampling to gradually transit to predicted entity spans
        if train_flag and self.config.use_scheduled_sampling:
            # teacher_prob will gradually decrease outside
            if random.random() < teacher_prob:
                use_gold_span = True
            else:
                use_gold_span = False

        # get doc token-level local context
        doc_token_emb_list, doc_token_masks_list, doc_token_types_list, doc_sent_emb_list, doc_sent_loss_list = \
            self.get_local_context_info(
                doc_batch_dict, train_flag=train_flag, use_gold_span=use_gold_span,
            )

        # get doc feature objects
        ex_idx_list = doc_batch_dict['ex_idx']
        doc_fea_list = [doc_features[ex_idx] for ex_idx in ex_idx_list]

        # get doc span-level info for event extraction
        doc_span_info_list = get_doc_span_info_list(doc_token_types_list, doc_fea_list, use_gold_span=use_gold_span)

        if train_flag:
            doc_event_loss_list = []
            for batch_idx, ex_idx in enumerate(ex_idx_list):
                doc_event_loss_list.append(
                    self.get_loss_on_doc(
                        doc_token_emb_list[batch_idx],
                        doc_sent_emb_list[batch_idx],
                        doc_fea_list[batch_idx],
                        doc_span_info_list[batch_idx],
                    )
                )

            mix_loss = self.get_mix_loss(doc_sent_loss_list, doc_event_loss_list, doc_span_info_list)

            return mix_loss
        else:
            # return a list object may not be supported by torch.nn.parallel.DataParallel
            # ensure to run it under the single-gpu mode
            eval_results = []

            if heuristic_type is None:
                for batch_idx, ex_idx in enumerate(ex_idx_list):
                    eval_results.append(
                        self.get_eval_on_doc(
                            doc_token_emb_list[batch_idx],
                            doc_sent_emb_list[batch_idx],
                            doc_fea_list[batch_idx],
                            doc_span_info_list[batch_idx],
                        )
                    )
            else:
                assert event_idx2entity_idx2field_idx is not None
                for batch_idx, ex_idx in enumerate(ex_idx_list):
                    eval_results.append(
                        self.heuristic_decode_on_doc(
                            doc_token_emb_list[batch_idx],
                            doc_sent_emb_list[batch_idx],
                            doc_fea_list[batch_idx],
                            doc_span_info_list[batch_idx],
                            event_idx2entity_idx2field_idx,
                            heuristic_type=heuristic_type,
                        )
                    )

            return eval_results

    def heuristic_decode_on_doc(self, doc_token_emb, doc_sent_emb, doc_fea, doc_span_info,
                                event_idx2entity_idx2field_idx, heuristic_type='GreedyDec'):
        support_heuristic_types = ['GreedyDec', 'ProductDec']
        if heuristic_type not in support_heuristic_types:
            raise Exception('Unsupported heuristic type {}, pleasure choose from {}'.format(
                heuristic_type, str(support_heuristic_types)
            ))

        span_context_list, doc_sent_context = self.get_doc_span_sent_context(
            doc_token_emb, doc_sent_emb, doc_fea, doc_span_info
        )

        span_token_tup_list = doc_span_info.span_token_tup_list
        span_mention_range_list = doc_span_info.span_mention_range_list
        mention_drange_list = doc_span_info.mention_drange_list
        mention_type_list = doc_span_info.mention_type_list
        # heuristic decoding strategies will work on these span candidates
        event_idx2field_idx2span_token_tup2dranges = self.get_event_field_span_candidates(
            span_token_tup_list, span_mention_range_list, mention_drange_list,
            mention_type_list, event_idx2entity_idx2field_idx,
        )

        # if there is no extracted span, just directly return
        if len(span_token_tup_list) == 0:
            event_pred_list = []
            event_idx2obj_idx2field_idx2token_tup = []  # this term will be compared with ground-truth table contents
            for event_idx in range(len(self.event_type_fields_pairs)):
                event_pred_list.append(0)
                event_idx2obj_idx2field_idx2token_tup.append(None)

            return doc_fea.ex_idx, event_pred_list, event_idx2obj_idx2field_idx2token_tup, \
                doc_span_info, event_idx2field_idx2span_token_tup2dranges

        # 1. get event type prediction as model-based approach
        event_pred_list = self.get_event_cls_info(doc_sent_context, doc_fea, train_flag=False)

        # 2. for each event type, get field prediction
        # From now on, use heuristic inference to get the token for the field
        # the following mappings are all implemented using list index
        event_idx2obj_idx2field_idx2token_tup = []
        for event_idx, event_pred in enumerate(event_pred_list):
            if event_pred == 0:
                event_idx2obj_idx2field_idx2token_tup.append(None)
                continue

            num_fields = self.event_tables[event_idx].num_fields
            field_idx2span_token_tup2dranges = event_idx2field_idx2span_token_tup2dranges[event_idx]

            obj_idx2field_idx2token_tup = [[]]  # at least one decode path will be appended
            for field_idx in range(num_fields):
                if heuristic_type == support_heuristic_types[0]:
                    obj_idx2field_idx2token_tup = append_top_span_only(
                        obj_idx2field_idx2token_tup, field_idx, field_idx2span_token_tup2dranges
                    )
                elif heuristic_type == support_heuristic_types[1]:
                    obj_idx2field_idx2token_tup = append_all_spans(
                        obj_idx2field_idx2token_tup, field_idx, field_idx2span_token_tup2dranges
                    )
                else:
                    raise Exception('Unsupported heuristic type {}, pleasure choose from {}'.format(
                        heuristic_type, str(support_heuristic_types)
                    ))

            event_idx2obj_idx2field_idx2token_tup.append(obj_idx2field_idx2token_tup)

        return doc_fea.ex_idx, event_pred_list, event_idx2obj_idx2field_idx2token_tup, \
            doc_span_info, event_idx2field_idx2span_token_tup2dranges

    def get_event_field_span_candidates(self, span_token_tup_list, span_mention_range_list,
                                        mention_drange_list, mention_type_list, event_idx2entity_idx2field_idx):
        # get mention idx -> span idx
        mention_span_idx_list = []
        for span_idx, (ment_idx_s, ment_idx_e) in enumerate(span_mention_range_list):
            mention_span_idx_list.extend([span_idx] * (ment_idx_e - ment_idx_s))
        assert len(mention_span_idx_list) == len(mention_drange_list)

        event_idx2field_idx2span_token_tup2dranges = {}
        for event_idx, (event_type, field_types) in enumerate(self.event_type_fields_pairs):
            # get the predefined entity idx to field idx mapping
            gold_entity_idx2field_idx = event_idx2entity_idx2field_idx[event_idx]

            # store field candidates for this doc
            field_idx2span_token_tup2dranges = {}
            for field_idx, _ in enumerate(field_types):
                field_idx2span_token_tup2dranges[field_idx] = {}

            # aggregate field candidates according to mention types
            for ment_idx, (ment_drange, ment_entity_idx) in enumerate(zip(mention_drange_list, mention_type_list)):
                if ment_entity_idx not in gold_entity_idx2field_idx:
                    continue
                ment_field_idx = gold_entity_idx2field_idx[ment_entity_idx]
                if ment_field_idx is None:
                    continue

                ment_span_idx = mention_span_idx_list[ment_idx]
                span_token_tup = span_token_tup_list[ment_span_idx]

                # because it is dict, so all modifications to the key will take effect in raw dict
                cur_span_token_tup2dranges = field_idx2span_token_tup2dranges[ment_field_idx]
                if span_token_tup not in cur_span_token_tup2dranges:
                    cur_span_token_tup2dranges[span_token_tup] = []
                cur_span_token_tup2dranges[span_token_tup].append(ment_drange)

            event_idx2field_idx2span_token_tup2dranges[event_idx] = field_idx2span_token_tup2dranges

        return event_idx2field_idx2span_token_tup2dranges


def append_top_span_only(last_token_path_list, field_idx, field_idx2span_token_tup2dranges):
    new_token_path_list = []
    span_token_tup2dranges = field_idx2span_token_tup2dranges[field_idx]
    token_min_drange_list = [
        (token_tup, dranges[0]) for token_tup, dranges in span_token_tup2dranges.items()
    ]
    token_min_drange_list.sort(key=lambda x: x[1])

    for last_token_path in last_token_path_list:
        new_token_path = list(last_token_path)
        if len(token_min_drange_list) == 0:
            new_token_path.append(None)
        else:
            token_tup = token_min_drange_list[0][0]
            new_token_path.append(token_tup)

        new_token_path_list.append(new_token_path)

    return new_token_path_list


def append_all_spans(last_token_path_list, field_idx, field_idx2span_token_tup2dranges):
    new_token_path_list = []
    span_token_tup2dranges = field_idx2span_token_tup2dranges[field_idx]

    for last_token_path in last_token_path_list:
        for token_tup in span_token_tup2dranges.keys():
            new_token_path = list(last_token_path)
            new_token_path.append(token_tup)
            new_token_path_list.append(new_token_path)

        if len(span_token_tup2dranges) == 0:  # ensure every last path will be extended
            new_token_path = list(last_token_path)
            new_token_path.append(None)
            new_token_path_list.append(new_token_path)

    return new_token_path_list


class AttentiveReducer(nn.Module):
    def __init__(self, hidden_size, dropout=0.1):
        super(AttentiveReducer, self).__init__()

        self.hidden_size = hidden_size
        self.att_norm = math.sqrt(self.hidden_size)

        self.fc = nn.Linear(hidden_size, 1, bias=False)
        self.att = None

        self.layer_norm = transformer.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, batch_token_emb, masks=None, keepdim=False):
        # batch_token_emb: Size([*, seq_len, hidden_size])
        # masks: Size([*, seq_len]), 1: normal, 0: pad

        query = self.fc.weight
        if masks is None:
            att_mask = None
        else:
            att_mask = masks.unsqueeze(-2)  # [*, 1, seq_len]

        # batch_att_emb: Size([*, 1, hidden_size])
        # self.att: Size([*, 1, seq_len])
        batch_att_emb, self.att = transformer.attention(
            query, batch_token_emb, batch_token_emb, mask=att_mask
        )

        batch_att_emb = self.dropout(self.layer_norm(batch_att_emb))

        if keepdim:
            return batch_att_emb
        else:
            return batch_att_emb.squeeze(-2)

    def extra_repr(self):
        return 'hidden_size={}, att_norm={}'.format(self.hidden_size, self.att_norm)


class SentencePosEncoder(nn.Module):
    def __init__(self, hidden_size, max_sent_num=100, dropout=0.1):
        super(SentencePosEncoder, self).__init__()

        self.embedding = nn.Embedding(max_sent_num, hidden_size)
        self.layer_norm = transformer.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, batch_elem_emb, sent_pos_ids=None):
        if sent_pos_ids is None:
            num_elem = batch_elem_emb.size(-2)
            sent_pos_ids = torch.arange(
                num_elem, dtype=torch.long, device=batch_elem_emb.device, requires_grad=False
            )
        elif not isinstance(sent_pos_ids, torch.Tensor):
            sent_pos_ids = torch.tensor(
                sent_pos_ids, dtype=torch.long, device=batch_elem_emb.device, requires_grad=False
            )

        batch_pos_emb = self.embedding(sent_pos_ids)
        out = batch_elem_emb + batch_pos_emb
        out = self.dropout(self.layer_norm(out))

        return out


class MentionTypeEncoder(nn.Module):
    def __init__(self, hidden_size, num_ment_types, dropout=0.1):
        super(MentionTypeEncoder, self).__init__()

        self.embedding = nn.Embedding(num_ment_types, hidden_size)
        self.layer_norm = transformer.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, batch_mention_emb, mention_type_ids):
        if not isinstance(mention_type_ids, torch.Tensor):
            mention_type_ids = torch.tensor(
                mention_type_ids, dtype=torch.long, device=batch_mention_emb.device, requires_grad=False
            )

        batch_mention_type_emb = self.embedding(mention_type_ids)
        out = batch_mention_emb + batch_mention_type_emb
        out = self.dropout(self.layer_norm(out))

        return out


class EventTable(nn.Module):
    def __init__(self, event_type, field_types, hidden_size):
        super(EventTable, self).__init__()

        self.event_type = event_type
        self.field_types = field_types
        self.num_fields = len(field_types)
        self.hidden_size = hidden_size

        self.event_cls = nn.Linear(hidden_size, 2)  # 0: NA, 1: trigger this event
        self.field_cls_list = nn.ModuleList(
            # 0: NA, 1: trigger this field
            [nn.Linear(hidden_size, 2) for _ in range(self.num_fields)]
        )

        # used to aggregate sentence and span embedding
        self.event_query = nn.Parameter(torch.Tensor(1, self.hidden_size))
        # used for fields that do not contain any valid span
        # self.none_span_emb = nn.Parameter(torch.Tensor(1, self.hidden_size))
        # used for aggregating history filled span info
        self.field_queries = nn.ParameterList(
            [nn.Parameter(torch.Tensor(1, self.hidden_size)) for _ in range(self.num_fields)]
        )

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.hidden_size)
        self.event_query.data.uniform_(-stdv, stdv)
        # self.none_span_emb.data.uniform_(-stdv, stdv)
        for fq in self.field_queries:
            fq.data.uniform_(-stdv, stdv)

    def forward(self, sent_context_emb=None, batch_span_emb=None, field_idx=None):
        assert (sent_context_emb is None) ^ (batch_span_emb is None)

        if sent_context_emb is not None:  # [num_spans+num_sents, hidden_size]
            # doc_emb.size = [1, hidden_size]
            doc_emb, _ = transformer.attention(self.event_query, sent_context_emb, sent_context_emb)
            doc_pred_logits = self.event_cls(doc_emb)
            doc_pred_logp = F.log_softmax(doc_pred_logits, dim=-1)

            return doc_pred_logp

        if batch_span_emb is not None:
            assert field_idx is not None
            # span_context_emb: [batch_size, hidden_size] or [hidden_size]
            if batch_span_emb.dim() == 1:
                batch_span_emb = batch_span_emb.unsqueeze(0)
            span_pred_logits = self.field_cls_list[field_idx](batch_span_emb)
            span_pred_logp = F.log_softmax(span_pred_logits, dim=-1)

            return span_pred_logp

    def extra_repr(self):
        return 'event_type={}, num_fields={}, hidden_size={}'.format(
            self.event_type, self.num_fields, self.hidden_size
        )


class MLP(nn.Module):
    """Implements Multi-layer Perception."""

    def __init__(self, input_size, output_size, mid_size=None, num_mid_layer=1, dropout=0.1):
        super(MLP, self).__init__()

        assert num_mid_layer >= 1
        if mid_size is None:
            mid_size = input_size

        self.input_fc = nn.Linear(input_size, mid_size)
        self.out_fc = nn.Linear(mid_size, output_size)
        if num_mid_layer > 1:
            self.mid_fcs = nn.ModuleList(
                nn.Linear(mid_size, mid_size) for _ in range(num_mid_layer-1)
            )
        else:
            self.mid_fcs = []
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(F.relu(self.input_fc(x)))
        for mid_fc in self.mid_fcs:
            x = self.dropout(F.relu(mid_fc(x)))
        x = self.out_fc(x)
        return x


def get_span_mention_info(span_dranges_list, doc_token_type_list):
    span_mention_range_list = []
    mention_drange_list = []
    mention_type_list = []
    for span_dranges in span_dranges_list:
        ment_idx_s = len(mention_drange_list)
        for drange in span_dranges:
            mention_drange_list.append(drange)
            sent_idx, char_s, char_e = drange
            mention_type_list.append(doc_token_type_list[sent_idx][char_s])
        ment_idx_e = len(mention_drange_list)
        span_mention_range_list.append((ment_idx_s, ment_idx_e))

    return span_mention_range_list, mention_drange_list, mention_type_list


def extract_doc_valid_span_info(doc_token_type_mat, doc_fea):
    doc_token_id_mat = doc_fea.doc_token_ids.tolist()
    doc_token_mask_mat = doc_fea.doc_token_masks.tolist()

    # [(token_id_tuple, (sent_idx, char_s, char_e)), ...]
    span_token_drange_list = []

    valid_sent_num = doc_fea.valid_sent_num
    for sent_idx in range(valid_sent_num):
        seq_token_id_list = doc_token_id_mat[sent_idx]
        seq_token_mask_list = doc_token_mask_mat[sent_idx]
        seq_token_type_list = doc_token_type_mat[sent_idx]
        seq_len = len(seq_token_id_list)

        char_s = 0
        while char_s < seq_len:
            if seq_token_mask_list[char_s] == 0:
                break

            entity_idx = seq_token_type_list[char_s]

            if entity_idx % 2 == 1:
                char_e = char_s + 1
                while char_e < seq_len and seq_token_mask_list[char_e] == 1 and \
                        seq_token_type_list[char_e] == entity_idx + 1:
                    char_e += 1

                token_tup = tuple(seq_token_id_list[char_s:char_e])
                drange = (sent_idx, char_s, char_e)

                span_token_drange_list.append((token_tup, drange))

                char_s = char_e
            else:
                char_s += 1

    span_token_drange_list.sort(key=lambda x: x[-1])  # sorted by drange = (sent_idx, char_s, char_e)
    # drange is exclusive and sorted
    token_tup2dranges = OrderedDict()
    for token_tup, drange in span_token_drange_list:
        if token_tup not in token_tup2dranges:
            token_tup2dranges[token_tup] = []
        token_tup2dranges[token_tup].append(drange)

    span_token_tup_list = list(token_tup2dranges.keys())
    span_dranges_list = list(token_tup2dranges.values())

    return span_token_tup_list, span_dranges_list


def get_batch_span_label(num_spans, cur_span_idx_set, device):
    # prepare span labels for this field and this path
    span_field_labels = [
        1 if span_idx in cur_span_idx_set else 0 for span_idx in range(num_spans)
    ]

    batch_field_label = torch.tensor(
        span_field_labels, dtype=torch.long, device=device, requires_grad=False
    )  # [num_spans], val \in {0, 1}

    return batch_field_label


class DCFEEModel(nn.Module):
    """
    This module implements the baseline model described in http://www.aclweb.org/anthology/P18-4009:
        "DCFEE: A Document-level Chinese Financial Event Extraction System
        based on Automatically Labeled Training Data"
    """

    def __init__(self, config, event_type_fields_pairs, ner_model=None):
        super(DCFEEModel, self).__init__()
        # Note that for distributed training, you must ensure that
        # for any batch, all parameters need to be used

        self.config = config
        self.event_type_fields_pairs = event_type_fields_pairs

        if ner_model is None:
            self.ner_model = NERModel(config)
        else:
            self.ner_model = ner_model

        # attentively reduce token embedding into sentence embedding
        self.doc_token_reducer = AttentiveReducer(config.hidden_size, dropout=config.dropout)
        # map sentence embedding to event prediction logits
        self.event_cls_layers = nn.ModuleList([
            nn.Linear(config.hidden_size, 2) for _ in self.event_type_fields_pairs
        ])

    def get_batch_sent_emb(self, ner_token_emb, ner_token_masks, valid_sent_num_list):
        # From [ner_batch_size, sent_len, hidden_size] to [ner_batch_size, hidden_size]
        total_sent_emb = self.doc_token_reducer(ner_token_emb, ner_token_masks)
        total_sent_pos_ids = []
        for valid_sent_num in valid_sent_num_list:
            total_sent_pos_ids += list(range(valid_sent_num))

        return total_sent_emb

    def get_loss_on_doc(self, doc_sent_emb, doc_fea):
        doc_sent_label_mat = torch.tensor(
            doc_fea.doc_sent_labels, dtype=torch.long, device=doc_sent_emb.device, requires_grad=False
        )
        event_cls_loss_list = []
        for event_idx, event_cls in enumerate(self.event_cls_layers):
            doc_sent_logits = event_cls(doc_sent_emb)  # [sent_num, 2]
            doc_sent_labels = doc_sent_label_mat[:, event_idx]  # [sent_num]
            event_cls_loss = F.cross_entropy(doc_sent_logits, doc_sent_labels, reduction='sum')
            event_cls_loss_list.append(event_cls_loss)

        final_loss = sum(event_cls_loss_list)
        return final_loss

    def get_mix_loss(self, doc_sent_loss_list, doc_event_loss_list, doc_span_info_list):
        batch_size = len(doc_span_info_list)
        loss_batch_avg = 1.0 / batch_size
        lambda_1 = self.config.loss_lambda
        lambda_2 = 1 - lambda_1

        doc_ner_loss_list = []
        for doc_sent_loss, doc_span_info in zip(doc_sent_loss_list, doc_span_info_list):
            # doc_sent_loss: Size([num_valid_sents])
            sent_loss_scaling = doc_sent_loss.new_full(
                doc_sent_loss.size(), 1, requires_grad=False
            )
            sent_loss_scaling[doc_span_info.missed_sent_idx_list] = self.config.loss_gamma
            doc_ner_loss = (doc_sent_loss * sent_loss_scaling).sum()
            doc_ner_loss_list.append(doc_ner_loss)

        return loss_batch_avg * (lambda_1 * sum(doc_ner_loss_list) + lambda_2 * sum(doc_event_loss_list))

    def get_local_context_info(self, doc_batch_dict, train_flag=False, use_gold_span=False):
        label_key = 'doc_token_labels'
        if train_flag or use_gold_span:
            assert label_key in doc_batch_dict
            need_label_flag = True
        else:
            need_label_flag = False

        if need_label_flag:
            doc_token_labels_list = doc_batch_dict[label_key]
        else:
            doc_token_labels_list = None

        batch_size = len(doc_batch_dict['ex_idx'])
        doc_token_ids_list = doc_batch_dict['doc_token_ids']
        doc_token_masks_list = doc_batch_dict['doc_token_masks']
        valid_sent_num_list = doc_batch_dict['valid_sent_num']

        # transform doc_batch into sent_batch
        ner_batch_idx_start_list = [0]
        ner_token_ids = []
        ner_token_masks = []
        ner_token_labels = [] if need_label_flag else None
        for batch_idx, valid_sent_num in enumerate(valid_sent_num_list):
            idx_start = ner_batch_idx_start_list[-1]
            idx_end = idx_start + valid_sent_num
            ner_batch_idx_start_list.append(idx_end)

            ner_token_ids.append(doc_token_ids_list[batch_idx])
            ner_token_masks.append(doc_token_masks_list[batch_idx])
            if need_label_flag:
                ner_token_labels.append(doc_token_labels_list[batch_idx])

        # [ner_batch_size, norm_sent_len]
        ner_token_ids = torch.cat(ner_token_ids, dim=0)
        ner_token_masks = torch.cat(ner_token_masks, dim=0)
        if need_label_flag:
            ner_token_labels = torch.cat(ner_token_labels, dim=0)

        # get ner output
        ner_token_emb, ner_loss, ner_token_preds = self.ner_model(
            ner_token_ids, ner_token_masks, label_ids=ner_token_labels,
            train_flag=train_flag, decode_flag=not use_gold_span,
        )

        if use_gold_span:  # definitely use gold span info
            ner_token_types = ner_token_labels
        else:
            ner_token_types = ner_token_preds

        # get sentence embedding
        ner_sent_emb = self.get_batch_sent_emb(ner_token_emb, ner_token_masks, valid_sent_num_list)

        assert sum(valid_sent_num_list) == ner_token_emb.size(0) == ner_sent_emb.size(0)

        # followings are all lists of tensors
        doc_token_emb_list = []
        doc_token_masks_list = []
        doc_token_types_list = []
        doc_sent_emb_list = []
        doc_sent_loss_list = []
        for batch_idx in range(batch_size):
            idx_start = ner_batch_idx_start_list[batch_idx]
            idx_end = ner_batch_idx_start_list[batch_idx+1]
            doc_token_emb_list.append(ner_token_emb[idx_start:idx_end, :, :])
            doc_token_masks_list.append(ner_token_masks[idx_start:idx_end, :])
            doc_token_types_list.append(ner_token_types[idx_start:idx_end, :])
            doc_sent_emb_list.append(ner_sent_emb[idx_start:idx_end, :])
            if ner_loss is not None:
                # every doc_sent_loss.size is [valid_sent_num]
                doc_sent_loss_list.append(ner_loss[idx_start:idx_end])

        return doc_token_emb_list, doc_token_masks_list, doc_token_types_list, doc_sent_emb_list, doc_sent_loss_list

    def forward(self, doc_batch_dict, doc_features,
                use_gold_span=False, train_flag=True, heuristic_type='DCFEE-O',
                event_idx2entity_idx2field_idx=None, **kwargs):
        # DCFEE does not need scheduled sampling
        # get doc token-level local context
        doc_token_emb_list, doc_token_masks_list, doc_token_types_list, doc_sent_emb_list, doc_sent_loss_list = \
            self.get_local_context_info(
                doc_batch_dict, train_flag=train_flag, use_gold_span=use_gold_span,
            )

        # get doc feature objects
        ex_idx_list = doc_batch_dict['ex_idx']
        doc_fea_list = [doc_features[ex_idx] for ex_idx in ex_idx_list]

        # get doc span-level info for event extraction
        doc_span_info_list = get_doc_span_info_list(doc_token_types_list, doc_fea_list, use_gold_span=use_gold_span)

        if train_flag:
            doc_event_loss_list = []
            for batch_idx, ex_idx in enumerate(ex_idx_list):
                doc_event_loss_list.append(
                    self.get_loss_on_doc(
                        doc_sent_emb_list[batch_idx],
                        doc_fea_list[batch_idx],
                    )
                )

            mix_loss = self.get_mix_loss(doc_sent_loss_list, doc_event_loss_list, doc_span_info_list)

            return mix_loss
        else:
            # return a list object may not be supported by torch.nn.parallel.DataParallel
            # ensure to run it under the single-gpu mode
            eval_results = []

            assert event_idx2entity_idx2field_idx is not None
            for batch_idx, ex_idx in enumerate(ex_idx_list):
                eval_results.append(
                    self.heuristic_decode_on_doc(
                        doc_sent_emb_list[batch_idx],
                        doc_fea_list[batch_idx],
                        doc_span_info_list[batch_idx],
                        event_idx2entity_idx2field_idx,
                        heuristic_type=heuristic_type,
                    )
                )

            return eval_results

    def heuristic_decode_on_doc(self, doc_sent_emb, doc_fea, doc_span_info,
                                event_idx2entity_idx2field_idx, heuristic_type='DCFEE-O'):
        # DCFEE-O: just produce One event per triggered sentence
        # DCFEE-M: produce Multiple potential events per triggered sentence
        support_heuristic_types = ['DCFEE-O', 'DCFEE-M']
        if heuristic_type not in support_heuristic_types:
            raise Exception('Unsupported heuristic type {}, pleasure choose from {}'.format(
                heuristic_type, str(support_heuristic_types)
            ))

        span_token_tup_list = doc_span_info.span_token_tup_list
        span_mention_range_list = doc_span_info.span_mention_range_list
        mention_drange_list = doc_span_info.mention_drange_list
        mention_type_list = doc_span_info.mention_type_list
        # heuristic decoding strategies will work on these span candidates
        event_idx2field_idx2span_token_tup2dranges = self.get_event_field_span_candidates(
            span_token_tup_list, span_mention_range_list, mention_drange_list,
            mention_type_list, event_idx2entity_idx2field_idx,
        )

        # if there is no extracted span, just directly return
        if len(span_token_tup_list) == 0:
            event_pred_list = []
            event_idx2obj_idx2field_idx2token_tup = []  # this term will be compared with ground-truth table contents
            for event_idx in range(len(self.event_type_fields_pairs)):
                event_pred_list.append(0)
                event_idx2obj_idx2field_idx2token_tup.append(None)

            return doc_fea.ex_idx, event_pred_list, event_idx2obj_idx2field_idx2token_tup, \
                doc_span_info, event_idx2field_idx2span_token_tup2dranges

        event_idx2key_sent_idx_list = []
        event_pred_list = []
        event_idx2obj_idx2field_idx2token_tup = []
        for event_idx, event_cls in enumerate(self.event_cls_layers):
            event_type, field_types = self.event_type_fields_pairs[event_idx]
            num_fields = len(field_types)
            field_idx2span_token_tup2dranges = event_idx2field_idx2span_token_tup2dranges[event_idx]

            # get key event sentence prediction
            doc_sent_logits = event_cls(doc_sent_emb)  # [sent_num, 2]
            doc_sent_logp = F.log_softmax(doc_sent_logits, dim=-1)  # [sent_num, 2]
            doc_sent_pred_list = doc_sent_logp.argmax(dim=-1).tolist()
            key_sent_idx_list = [
                sent_idx for sent_idx, sent_pred in enumerate(doc_sent_pred_list) if sent_pred == 1
            ]
            event_idx2key_sent_idx_list.append(key_sent_idx_list)

            if len(key_sent_idx_list) == 0:
                event_pred_list.append(0)
                event_idx2obj_idx2field_idx2token_tup.append(None)
            else:
                obj_idx2field_idx2token_tup = []
                for key_sent_idx in key_sent_idx_list:
                    if heuristic_type == support_heuristic_types[0]:
                        field_idx2token_tup = get_one_key_sent_event(
                            key_sent_idx, num_fields, field_idx2span_token_tup2dranges
                        )
                        obj_idx2field_idx2token_tup.append(field_idx2token_tup)
                    elif heuristic_type == support_heuristic_types[1]:
                        field_idx2token_tup_list = get_many_key_sent_event(
                            key_sent_idx, num_fields, field_idx2span_token_tup2dranges
                        )
                        obj_idx2field_idx2token_tup.extend(field_idx2token_tup_list)
                    else:
                        raise Exception('Unsupported heuristic type {}, pleasure choose from {}'.format(
                            heuristic_type, str(support_heuristic_types)
                        ))
                event_pred_list.append(1)
                event_idx2obj_idx2field_idx2token_tup.append(obj_idx2field_idx2token_tup)

        return doc_fea.ex_idx, event_pred_list, event_idx2obj_idx2field_idx2token_tup, \
            doc_span_info, event_idx2field_idx2span_token_tup2dranges, event_idx2key_sent_idx_list

    def get_event_field_span_candidates(self, span_token_tup_list, span_mention_range_list,
                                        mention_drange_list, mention_type_list, event_idx2entity_idx2field_idx):
        # get mention idx -> span idx
        mention_span_idx_list = []
        for span_idx, (ment_idx_s, ment_idx_e) in enumerate(span_mention_range_list):
            mention_span_idx_list.extend([span_idx] * (ment_idx_e - ment_idx_s))
        assert len(mention_span_idx_list) == len(mention_drange_list)

        event_idx2field_idx2span_token_tup2dranges = {}
        for event_idx, (event_type, field_types) in enumerate(self.event_type_fields_pairs):
            # get the predefined entity idx to field idx mapping
            gold_entity_idx2field_idx = event_idx2entity_idx2field_idx[event_idx]

            # store field candidates for this doc
            field_idx2span_token_tup2dranges = {}
            for field_idx, _ in enumerate(field_types):
                field_idx2span_token_tup2dranges[field_idx] = {}

            # aggregate field candidates according to mention types
            for ment_idx, (ment_drange, ment_entity_idx) in enumerate(zip(mention_drange_list, mention_type_list)):
                if ment_entity_idx not in gold_entity_idx2field_idx:
                    continue
                ment_field_idx = gold_entity_idx2field_idx[ment_entity_idx]
                if ment_field_idx is None:
                    continue

                ment_span_idx = mention_span_idx_list[ment_idx]
                span_token_tup = span_token_tup_list[ment_span_idx]

                # because it is dict, so all modifications to the key will take effect in raw dict
                cur_span_token_tup2dranges = field_idx2span_token_tup2dranges[ment_field_idx]
                if span_token_tup not in cur_span_token_tup2dranges:
                    cur_span_token_tup2dranges[span_token_tup] = []
                cur_span_token_tup2dranges[span_token_tup].append(ment_drange)

            event_idx2field_idx2span_token_tup2dranges[event_idx] = field_idx2span_token_tup2dranges

        return event_idx2field_idx2span_token_tup2dranges


def get_one_key_sent_event(key_sent_idx, num_fields, field_idx2span_token_tup2dranges):
    field_idx2token_tup = []
    for field_idx in range(num_fields):
        token_tup2dranges = field_idx2span_token_tup2dranges[field_idx]

        # find the closest token_tup to the key sentence
        best_token_tup = None
        best_dist = 10000
        for token_tup, dranges in token_tup2dranges.items():
            for sent_idx, _, _ in dranges:
                cur_dist = abs(sent_idx - key_sent_idx)
                if cur_dist < best_dist:
                    best_token_tup = token_tup
                    best_dist = cur_dist

        field_idx2token_tup.append(best_token_tup)
    return field_idx2token_tup


def get_many_key_sent_event(key_sent_idx, num_fields, field_idx2span_token_tup2dranges):
    # get key_field_idx contained in key event sentence
    key_field_idx2token_tup_set = defaultdict(lambda: set())
    for field_idx, token_tup2dranges in field_idx2span_token_tup2dranges.items():
        assert field_idx < num_fields
        for token_tup, dranges in token_tup2dranges.items():
            for sent_idx, _, _ in dranges:
                if sent_idx == key_sent_idx:
                    key_field_idx2token_tup_set[field_idx].add(token_tup)

    field_idx2token_tup_list = []
    while len(key_field_idx2token_tup_set) > 0:
        # get key token tup candidates according to the distance in the sentence
        prev_field_idx = None
        prev_token_cand = None
        key_field_idx2token_cand = {}
        for key_field_idx, token_tup_set in key_field_idx2token_tup_set.items():
            assert len(token_tup_set) > 0

            if prev_token_cand is None:
                best_token_tup = token_tup_set.pop()
            else:
                prev_char_range = field_idx2span_token_tup2dranges[prev_field_idx][prev_token_cand][0][1:]
                best_dist = 10000
                best_token_tup = None
                for token_tup in token_tup_set:
                    cur_char_range = field_idx2span_token_tup2dranges[key_field_idx][token_tup][0][1:]
                    cur_dist = min(
                        abs(cur_char_range[1] - prev_char_range[0]),
                        abs(cur_char_range[0] - prev_char_range[1])
                    )
                    if cur_dist < best_dist:
                        best_dist = cur_dist
                        best_token_tup = token_tup
                token_tup_set.remove(best_token_tup)

            key_field_idx2token_cand[key_field_idx] = best_token_tup
            prev_field_idx = key_field_idx
            prev_token_cand = best_token_tup

        field_idx2token_tup = []
        for field_idx in range(num_fields):
            token_tup2dranges = field_idx2span_token_tup2dranges[field_idx]

            if field_idx in key_field_idx2token_tup_set:
                token_tup_set = key_field_idx2token_tup_set[field_idx]
                if len(token_tup_set) == 0:
                    del key_field_idx2token_tup_set[field_idx]
                token_tup = key_field_idx2token_cand[field_idx]
                field_idx2token_tup.append(token_tup)
            else:
                # find the closest token_tup to the key sentence
                best_token_tup = None
                best_dist = 10000
                for token_tup, dranges in token_tup2dranges.items():
                    for sent_idx, _, _ in dranges:
                        cur_dist = abs(sent_idx - key_sent_idx)
                        if cur_dist < best_dist:
                            best_token_tup = token_tup
                            best_dist = cur_dist

                field_idx2token_tup.append(best_token_tup)

        field_idx2token_tup_list.append(field_idx2token_tup)

    return field_idx2token_tup_list



