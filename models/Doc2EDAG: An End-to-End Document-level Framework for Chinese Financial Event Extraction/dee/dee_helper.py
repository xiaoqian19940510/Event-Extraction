# -*- coding: utf-8 -*-
# AUTHOR: Shun Zheng
# DATE: 19-9-19

import logging
import os
import re
from collections import defaultdict, Counter
import numpy as np
import torch

from .dee_metric import measure_event_table_filling
from .event_type import event_type2event_class, BaseEvent, event_type_fields_list, common_fields
from .ner_task import NERExample, NERFeatureConverter
from .utils import default_load_json, default_dump_json, default_dump_pkl, default_load_pkl


logger = logging.getLogger(__name__)


class DEEExample(object):
    def __init__(self, annguid, detail_align_dict, only_inference=False):
        self.guid = annguid
        # [sent_text, ...]
        self.sentences = detail_align_dict['sentences']
        self.num_sentences = len(self.sentences)

        if only_inference:
            # set empty entity/event information
            self.only_inference = True
            self.ann_valid_mspans = []
            self.ann_mspan2dranges = {}
            self.ann_mspan2guess_field = {}
            self.recguid_eventname_eventdict_list = []
            self.num_events = 0
            self.sent_idx2srange_mspan_mtype_tuples = {}
            self.event_type2event_objs = {}
        else:
            # set event information accordingly
            self.only_inference = False

            # [span_text, ...]
            self.ann_valid_mspans = detail_align_dict['ann_valid_mspans']
            # span_text -> [drange_tuple, ...]
            self.ann_mspan2dranges = detail_align_dict['ann_mspan2dranges']
            # span_text -> guessed_field_name
            self.ann_mspan2guess_field = detail_align_dict['ann_mspan2guess_field']
            # [(recguid, event_name, event_dict), ...]
            self.recguid_eventname_eventdict_list = detail_align_dict['recguid_eventname_eventdict_list']
            self.num_events = len(self.recguid_eventname_eventdict_list)

            # for create ner examples
            # sentence_index -> [(sent_match_range, match_span, match_type), ...]
            self.sent_idx2srange_mspan_mtype_tuples = {}
            for sent_idx in range(self.num_sentences):
                self.sent_idx2srange_mspan_mtype_tuples[sent_idx] = []

            for mspan in self.ann_valid_mspans:
                for drange in self.ann_mspan2dranges[mspan]:
                    sent_idx, char_s, char_e = drange
                    sent_mrange = (char_s, char_e)

                    sent_text = self.sentences[sent_idx]
                    if sent_text[char_s: char_e] != mspan:
                        raise Exception('GUID: {} span range is not correct, span={}, range={}, sent={}'.format(
                            annguid, mspan, str(sent_mrange), sent_text
                        ))

                    guess_field = self.ann_mspan2guess_field[mspan]

                    self.sent_idx2srange_mspan_mtype_tuples[sent_idx].append(
                        (sent_mrange, mspan, guess_field)
                    )

            # for create event objects
            # the length of event_objs should >= 1
            self.event_type2event_objs = {}
            for mrecguid, event_name, event_dict in self.recguid_eventname_eventdict_list:
                event_class = event_type2event_class[event_name]
                event_obj = event_class()
                assert isinstance(event_obj, BaseEvent)
                event_obj.update_by_dict(event_dict, recguid=mrecguid)

                if event_obj.name in self.event_type2event_objs:
                    self.event_type2event_objs[event_obj.name].append(event_obj)
                else:
                    self.event_type2event_objs[event_name] = [event_obj]

    def __repr__(self):
        dee_str = 'DEEExample (\n'
        dee_str += '  guid: {},\n'.format(repr(self.guid))

        if not self.only_inference:
            dee_str += '  span info: (\n'
            for span_idx, span in enumerate(self.ann_valid_mspans):
                gfield = self.ann_mspan2guess_field[span]
                dranges = self.ann_mspan2dranges[span]
                dee_str += '    {:2} {:20} {:30} {}\n'.format(span_idx, span, gfield, str(dranges))
            dee_str += '  ),\n'

            dee_str += '  event info: (\n'
            event_str_list = repr(self.event_type2event_objs).split('\n')
            for event_str in event_str_list:
                dee_str += '    {}\n'.format(event_str)
            dee_str += '  ),\n'

        dee_str += '  sentences: (\n'
        for sent_idx, sent in enumerate(self.sentences):
            dee_str += '    {:2} {}\n'.format(sent_idx, sent)
        dee_str += '  ),\n'

        dee_str += ')\n'

        return dee_str

    @staticmethod
    def get_event_type_fields_pairs():
        return list(event_type_fields_list)

    @staticmethod
    def get_entity_label_list():
        visit_set = set()
        entity_label_list = [NERExample.basic_entity_label]

        for field in common_fields:
            if field not in visit_set:
                visit_set.add(field)
                entity_label_list.extend(['B-' + field, 'I-' + field])

        for event_name, fields in event_type_fields_list:
            for field in fields:
                if field not in visit_set:
                    visit_set.add(field)
                    entity_label_list.extend(['B-' + field, 'I-' + field])

        return entity_label_list


class DEEExampleLoader(object):
    def __init__(self, rearrange_sent_flag, max_sent_len):
        self.rearrange_sent_flag = rearrange_sent_flag
        self.max_sent_len = max_sent_len

    def rearrange_sent_info(self, detail_align_info):
        if 'ann_valid_dranges' not in detail_align_info:
            detail_align_info['ann_valid_dranges'] = []
        if 'ann_mspan2dranges' not in detail_align_info:
            detail_align_info['ann_mspan2dranges'] = {}

        detail_align_info = dict(detail_align_info)
        split_rgx = re.compile('[，：:；;）)]')

        raw_sents = detail_align_info['sentences']
        doc_text = ''.join(raw_sents)
        raw_dranges = detail_align_info['ann_valid_dranges']
        raw_sid2span_char_set = defaultdict(lambda: set())
        for raw_sid, char_s, char_e in raw_dranges:
            span_char_set = raw_sid2span_char_set[raw_sid]
            span_char_set.update(range(char_s, char_e))

        # try to split long sentences into short ones by comma, colon, semi-colon, bracket
        short_sents = []
        for raw_sid, sent in enumerate(raw_sents):
            span_char_set = raw_sid2span_char_set[raw_sid]
            if len(sent) > self.max_sent_len:
                cur_char_s = 0
                for mobj in split_rgx.finditer(sent):
                    m_char_s, m_char_e = mobj.span()
                    if m_char_s in span_char_set:
                        continue
                    short_sents.append(sent[cur_char_s:m_char_e])
                    cur_char_s = m_char_e
                short_sents.append(sent[cur_char_s:])
            else:
                short_sents.append(sent)

        # merge adjacent short sentences to compact ones that match max_sent_len
        comp_sents = ['']
        for sent in short_sents:
            prev_sent = comp_sents[-1]
            if len(prev_sent + sent) <= self.max_sent_len:
                comp_sents[-1] = prev_sent + sent
            else:
                comp_sents.append(sent)

        # get global sentence character base indexes
        raw_char_bases = [0]
        for sent in raw_sents:
            raw_char_bases.append(raw_char_bases[-1] + len(sent))
        comp_char_bases = [0]
        for sent in comp_sents:
            comp_char_bases.append(comp_char_bases[-1] + len(sent))

        assert raw_char_bases[-1] == comp_char_bases[-1] == len(doc_text)

        # calculate compact doc ranges
        raw_dranges.sort()
        raw_drange2comp_drange = {}
        prev_comp_sid = 0
        for raw_drange in raw_dranges:
            raw_drange = tuple(raw_drange)  # important when json dump change tuple to list
            raw_sid, raw_char_s, raw_char_e = raw_drange
            raw_char_base = raw_char_bases[raw_sid]
            doc_char_s = raw_char_base + raw_char_s
            doc_char_e = raw_char_base + raw_char_e
            assert doc_char_s >= comp_char_bases[prev_comp_sid]

            cur_comp_sid = prev_comp_sid
            for cur_comp_sid in range(prev_comp_sid, len(comp_sents)):
                if doc_char_e <= comp_char_bases[cur_comp_sid+1]:
                    prev_comp_sid = cur_comp_sid
                    break
            comp_char_base = comp_char_bases[cur_comp_sid]
            assert comp_char_base <= doc_char_s < doc_char_e <= comp_char_bases[cur_comp_sid+1]
            comp_char_s = doc_char_s - comp_char_base
            comp_char_e = doc_char_e - comp_char_base
            comp_drange = (cur_comp_sid, comp_char_s, comp_char_e)

            raw_drange2comp_drange[raw_drange] = comp_drange
            assert raw_sents[raw_drange[0]][raw_drange[1]:raw_drange[2]] == \
                comp_sents[comp_drange[0]][comp_drange[1]:comp_drange[2]]

        # update detailed align info with rearranged sentences
        detail_align_info['sentences'] = comp_sents
        detail_align_info['ann_valid_dranges'] = [
            raw_drange2comp_drange[tuple(raw_drange)] for raw_drange in detail_align_info['ann_valid_dranges']
        ]
        ann_mspan2comp_dranges = {}
        for ann_mspan, mspan_raw_dranges in detail_align_info['ann_mspan2dranges'].items():
            comp_dranges = [
                raw_drange2comp_drange[tuple(raw_drange)] for raw_drange in mspan_raw_dranges
            ]
            ann_mspan2comp_dranges[ann_mspan] = comp_dranges
        detail_align_info['ann_mspan2dranges'] = ann_mspan2comp_dranges

        return detail_align_info

    def convert_dict_to_example(self, annguid, detail_align_info, only_inference=False):
        if self.rearrange_sent_flag:
            detail_align_info = self.rearrange_sent_info(detail_align_info)
        dee_example = DEEExample(annguid, detail_align_info, only_inference=only_inference)

        return dee_example

    def __call__(self, dataset_json_path):
        total_dee_examples = []
        annguid_aligninfo_list = default_load_json(dataset_json_path)
        for annguid, detail_align_info in annguid_aligninfo_list:
            # if self.rearrange_sent_flag:
            #     detail_align_info = self.rearrange_sent_info(detail_align_info)
            # dee_example = DEEExample(annguid, detail_align_info)
            dee_example = self.convert_dict_to_example(annguid, detail_align_info)
            total_dee_examples.append(dee_example)

        return total_dee_examples


class DEEFeature(object):
    def __init__(self, guid, ex_idx, doc_token_id_mat, doc_token_mask_mat, doc_token_label_mat,
                 span_token_ids_list, span_dranges_list, event_type_labels, event_arg_idxs_objs_list,
                 valid_sent_num=None):
        self.guid = guid
        self.ex_idx = ex_idx  # example row index, used for backtracking
        self.valid_sent_num = valid_sent_num

        # directly set tensor for dee feature to save memory
        # self.doc_token_id_mat = doc_token_id_mat
        # self.doc_token_mask_mat = doc_token_mask_mat
        # self.doc_token_label_mat = doc_token_label_mat
        self.doc_token_ids = torch.tensor(doc_token_id_mat, dtype=torch.long)
        self.doc_token_masks = torch.tensor(doc_token_mask_mat, dtype=torch.uint8)  # uint8 for mask
        self.doc_token_labels = torch.tensor(doc_token_label_mat, dtype=torch.long)

        # sorted by the first drange tuple
        # [(token_id, ...), ...]
        # span_idx -> span_token_id tuple
        self.span_token_ids_list = span_token_ids_list
        # [[(sent_idx, char_s, char_e), ...], ...]
        # span_idx -> [drange tuple, ...]
        self.span_dranges_list = span_dranges_list

        # [event_type_label, ...]
        # length = the total number of events to be considered
        # event_type_label \in {0, 1}, 0: no 1: yes
        self.event_type_labels = event_type_labels
        # event_type is denoted by the index of event_type_labels
        # event_type_idx -> event_obj_idx -> event_arg_idx -> span_idx
        # if no event objects, event_type_idx -> None
        self.event_arg_idxs_objs_list = event_arg_idxs_objs_list

        # event_type_idx -> event_field_idx -> pre_path -> {span_idx, ...}
        # pre_path is tuple of span_idx
        self.event_idx2field_idx2pre_path2cur_span_idx_set = self.build_dag_info(self.event_arg_idxs_objs_list)

        # event_type_idx -> key_sent_idx_set, used for key-event sentence detection
        self.event_idx2key_sent_idx_set, self.doc_sent_labels = self.build_key_event_sent_info()

    def generate_dag_info_for(self, pred_span_token_tup_list, return_miss=False):
        token_tup2pred_span_idx = {
            token_tup: pred_span_idx for pred_span_idx, token_tup in enumerate(pred_span_token_tup_list)
        }
        gold_span_idx2pred_span_idx = {}
        # pred_span_idx2gold_span_idx = {}
        missed_span_idx_list = []  # in terms of self
        missed_sent_idx_list = []  # in terms of self
        for gold_span_idx, token_tup in enumerate(self.span_token_ids_list):
            if token_tup in token_tup2pred_span_idx:
                pred_span_idx = token_tup2pred_span_idx[token_tup]
                gold_span_idx2pred_span_idx[gold_span_idx] = pred_span_idx
                # pred_span_idx2gold_span_idx[pred_span_idx] = gold_span_idx
            else:
                missed_span_idx_list.append(gold_span_idx)
                for gold_drange in self.span_dranges_list[gold_span_idx]:
                    missed_sent_idx_list.append(gold_drange[0])
        missed_sent_idx_list = list(set(missed_sent_idx_list))

        pred_event_arg_idxs_objs_list = []
        for event_arg_idxs_objs in self.event_arg_idxs_objs_list:
            if event_arg_idxs_objs is None:
                pred_event_arg_idxs_objs_list.append(None)
            else:
                pred_event_arg_idxs_objs = []
                for event_arg_idxs in event_arg_idxs_objs:
                    pred_event_arg_idxs = []
                    for gold_span_idx in event_arg_idxs:
                        if gold_span_idx in gold_span_idx2pred_span_idx:
                            pred_event_arg_idxs.append(
                                gold_span_idx2pred_span_idx[gold_span_idx]
                            )
                        else:
                            pred_event_arg_idxs.append(None)

                    pred_event_arg_idxs_objs.append(tuple(pred_event_arg_idxs))
                pred_event_arg_idxs_objs_list.append(pred_event_arg_idxs_objs)

        # event_idx -> field_idx -> pre_path -> cur_span_idx_set
        pred_dag_info = self.build_dag_info(pred_event_arg_idxs_objs_list)

        if return_miss:
            return pred_dag_info, missed_span_idx_list, missed_sent_idx_list
        else:
            return pred_dag_info

    def get_event_args_objs_list(self):
        event_args_objs_list = []
        for event_arg_idxs_objs in self.event_arg_idxs_objs_list:
            if event_arg_idxs_objs is None:
                event_args_objs_list.append(None)
            else:
                event_args_objs = []
                for event_arg_idxs in event_arg_idxs_objs:
                    event_args = []
                    for arg_idx in event_arg_idxs:
                        if arg_idx is None:
                            token_tup = None
                        else:
                            token_tup = self.span_token_ids_list[arg_idx]
                        event_args.append(token_tup)
                    event_args_objs.append(event_args)
                event_args_objs_list.append(event_args_objs)

        return event_args_objs_list

    def build_key_event_sent_info(self):
        assert len(self.event_type_labels) == len(self.event_arg_idxs_objs_list)
        # event_idx -> key_event_sent_index_set
        event_idx2key_sent_idx_set = [set() for _ in self.event_type_labels]
        for key_sent_idx_set, event_label, event_arg_idxs_objs in zip(
            event_idx2key_sent_idx_set, self.event_type_labels, self.event_arg_idxs_objs_list
        ):
            if event_label == 0:
                assert event_arg_idxs_objs is None
            else:
                for event_arg_idxs_obj in event_arg_idxs_objs:
                    sent_idx_cands = []
                    for span_idx in event_arg_idxs_obj:
                        if span_idx is None:
                            continue
                        span_dranges = self.span_dranges_list[span_idx]
                        for sent_idx, _, _ in span_dranges:
                            sent_idx_cands.append(sent_idx)
                    if len(sent_idx_cands) == 0:
                        raise Exception('Event {} has no valid spans'.format(str(event_arg_idxs_obj)))
                    sent_idx_cnter = Counter(sent_idx_cands)
                    key_sent_idx = sent_idx_cnter.most_common()[0][0]
                    key_sent_idx_set.add(key_sent_idx)

        doc_sent_labels = []  # 1: key event sentence, 0: otherwise
        for sent_idx in range(self.valid_sent_num):  # masked sents will be truncated at the model part
            sent_labels = []
            for key_sent_idx_set in event_idx2key_sent_idx_set:  # this mapping is a list
                if sent_idx in key_sent_idx_set:
                    sent_labels.append(1)
                else:
                    sent_labels.append(0)
            doc_sent_labels.append(sent_labels)

        return event_idx2key_sent_idx_set, doc_sent_labels

    @staticmethod
    def build_dag_info(event_arg_idxs_objs_list):
        # event_idx -> field_idx -> pre_path -> {span_idx, ...}
        # pre_path is tuple of span_idx
        event_idx2field_idx2pre_path2cur_span_idx_set = []
        for event_idx, event_arg_idxs_list in enumerate(event_arg_idxs_objs_list):
            if event_arg_idxs_list is None:
                event_idx2field_idx2pre_path2cur_span_idx_set.append(None)
            else:
                num_fields = len(event_arg_idxs_list[0])
                # field_idx -> pre_path -> {span_idx, ...}
                field_idx2pre_path2cur_span_idx_set = []
                for field_idx in range(num_fields):
                    pre_path2cur_span_idx_set = {}
                    for event_arg_idxs in event_arg_idxs_list:
                        pre_path = event_arg_idxs[:field_idx]
                        span_idx = event_arg_idxs[field_idx]
                        if pre_path not in pre_path2cur_span_idx_set:
                            pre_path2cur_span_idx_set[pre_path] = set()
                        pre_path2cur_span_idx_set[pre_path].add(span_idx)
                    field_idx2pre_path2cur_span_idx_set.append(pre_path2cur_span_idx_set)
                event_idx2field_idx2pre_path2cur_span_idx_set.append(field_idx2pre_path2cur_span_idx_set)

        return event_idx2field_idx2pre_path2cur_span_idx_set

    def is_multi_event(self):
        event_cnt = 0
        for event_objs in self.event_arg_idxs_objs_list:
            if event_objs is not None:
                event_cnt += len(event_objs)
                if event_cnt > 1:
                    return True

        return False


class DEEFeatureConverter(object):
    def __init__(self, entity_label_list, event_type_fields_pairs,
                 max_sent_len, max_sent_num, tokenizer,
                 ner_fea_converter=None, include_cls=True, include_sep=True):
        self.entity_label_list = entity_label_list
        self.event_type_fields_pairs = event_type_fields_pairs
        self.max_sent_len = max_sent_len
        self.max_sent_num = max_sent_num
        self.tokenizer = tokenizer
        self.truncate_doc_count = 0  # track how many docs have been truncated due to max_sent_num
        self.truncate_span_count = 0  # track how may spans have been truncated

        # label not in entity_label_list will be default 'O'
        # sent_len > max_sent_len will be truncated, and increase ner_fea_converter.truncate_freq
        if ner_fea_converter is None:
            self.ner_fea_converter = NERFeatureConverter(entity_label_list, self.max_sent_len, tokenizer,
                                                         include_cls=include_cls, include_sep=include_sep)
        else:
            self.ner_fea_converter = ner_fea_converter

        self.include_cls = include_cls
        self.include_sep = include_sep

        # prepare entity_label -> entity_index mapping
        self.entity_label2index = {}
        for entity_idx, entity_label in enumerate(self.entity_label_list):
            self.entity_label2index[entity_label] = entity_idx

        # prepare event_type -> event_index and event_index -> event_fields mapping
        self.event_type2index = {}
        self.event_type_list = []
        self.event_fields_list = []
        for event_idx, (event_type, event_fields) in enumerate(self.event_type_fields_pairs):
            self.event_type2index[event_type] = event_idx
            self.event_type_list.append(event_type)
            self.event_fields_list.append(event_fields)

    def convert_example_to_feature(self, ex_idx, dee_example, log_flag=False):
        annguid = dee_example.guid
        assert isinstance(dee_example, DEEExample)

        # 1. prepare doc token-level feature

        # Size(num_sent_num, num_sent_len)
        doc_token_id_mat = []  # [[token_idx, ...], ...]
        doc_token_mask_mat = []  # [[token_mask, ...], ...]
        doc_token_label_mat = []  # [[token_label_id, ...], ...]

        for sent_idx, sent_text in enumerate(dee_example.sentences):
            if sent_idx >= self.max_sent_num:
                # truncate doc whose number of sentences is longer than self.max_sent_num
                self.truncate_doc_count += 1
                break

            if sent_idx in dee_example.sent_idx2srange_mspan_mtype_tuples:
                srange_mspan_mtype_tuples = dee_example.sent_idx2srange_mspan_mtype_tuples[sent_idx]
            else:
                srange_mspan_mtype_tuples = []

            ner_example = NERExample(
                '{}-{}'.format(annguid, sent_idx), sent_text, srange_mspan_mtype_tuples
            )
            # sentence truncated count will be recorded incrementally
            ner_feature = self.ner_fea_converter.convert_example_to_feature(ner_example, log_flag=log_flag)

            doc_token_id_mat.append(ner_feature.input_ids)
            doc_token_mask_mat.append(ner_feature.input_masks)
            doc_token_label_mat.append(ner_feature.label_ids)

        assert len(doc_token_id_mat) == len(doc_token_mask_mat) == len(doc_token_label_mat) <= self.max_sent_num
        valid_sent_num = len(doc_token_id_mat)

        # 2. prepare span feature
        # spans are sorted by the first drange
        span_token_ids_list = []
        span_dranges_list = []
        mspan2span_idx = {}
        for mspan in dee_example.ann_valid_mspans:
            if mspan in mspan2span_idx:
                continue

            raw_dranges = dee_example.ann_mspan2dranges[mspan]
            char_base_s = 1 if self.include_cls else 0
            char_max_end = self.max_sent_len - 1 if self.include_sep else self.max_sent_len
            span_dranges = []
            for sent_idx, char_s, char_e in raw_dranges:
                if char_base_s + char_e <= char_max_end and sent_idx < self.max_sent_num:
                    span_dranges.append((sent_idx, char_base_s + char_s, char_base_s + char_e))
                else:
                    self.truncate_span_count += 1
            if len(span_dranges) == 0:
                # span does not have any valid location in truncated sequences
                continue

            span_tokens = self.tokenizer.char_tokenize(mspan)
            span_token_ids = tuple(self.tokenizer.convert_tokens_to_ids(span_tokens))

            mspan2span_idx[mspan] = len(span_token_ids_list)
            span_token_ids_list.append(span_token_ids)
            span_dranges_list.append(span_dranges)
        assert len(span_token_ids_list) == len(span_dranges_list) == len(mspan2span_idx)

        if len(span_token_ids_list) == 0 and not dee_example.only_inference:
            logger.warning('Neglect example {}'.format(ex_idx))
            return None

        # 3. prepare doc-level event feature
        # event_type_labels: event_type_index -> event_type_exist_sign (1: exist, 0: no)
        # event_arg_idxs_objs_list: event_type_index -> event_obj_index -> event_arg_index -> arg_span_token_ids

        event_type_labels = []  # event_type_idx -> event_type_exist_sign (1 or 0)
        event_arg_idxs_objs_list = []  # event_type_idx -> event_obj_idx -> event_arg_idx -> span_idx
        for event_idx, event_type in enumerate(self.event_type_list):
            event_fields = self.event_fields_list[event_idx]

            if event_type not in dee_example.event_type2event_objs:
                event_type_labels.append(0)
                event_arg_idxs_objs_list.append(None)
            else:
                event_objs = dee_example.event_type2event_objs[event_type]

                event_arg_idxs_objs = []
                for event_obj in event_objs:
                    assert isinstance(event_obj, BaseEvent)

                    event_arg_idxs = []
                    any_valid_flag = False
                    for field in event_fields:
                        arg_span = event_obj.field2content[field]

                        if arg_span is None or arg_span not in mspan2span_idx:
                            # arg_span can be none or valid span is truncated
                            arg_span_idx = None
                        else:
                            # when constructing data files,
                            # must ensure event arg span is covered by the total span collections
                            arg_span_idx = mspan2span_idx[arg_span]
                            any_valid_flag = True

                        event_arg_idxs.append(arg_span_idx)

                    if any_valid_flag:
                        event_arg_idxs_objs.append(tuple(event_arg_idxs))

                if event_arg_idxs_objs:
                    event_type_labels.append(1)
                    event_arg_idxs_objs_list.append(event_arg_idxs_objs)
                else:
                    event_type_labels.append(0)
                    event_arg_idxs_objs_list.append(None)

        dee_feature = DEEFeature(
            annguid, ex_idx, doc_token_id_mat, doc_token_mask_mat, doc_token_label_mat,
            span_token_ids_list, span_dranges_list, event_type_labels, event_arg_idxs_objs_list,
            valid_sent_num=valid_sent_num
        )

        return dee_feature

    def __call__(self, dee_examples, log_example_num=0):
        """Convert examples to features suitable for document-level event extraction"""
        dee_features = []
        self.truncate_doc_count = 0
        self.truncate_span_count = 0
        self.ner_fea_converter.truncate_count = 0

        remove_ex_cnt = 0
        for ex_idx, dee_example in enumerate(dee_examples):
            if ex_idx < log_example_num:
                dee_feature = self.convert_example_to_feature(ex_idx-remove_ex_cnt, dee_example, log_flag=True)
            else:
                dee_feature = self.convert_example_to_feature(ex_idx-remove_ex_cnt, dee_example, log_flag=False)

            if dee_feature is None:
                remove_ex_cnt += 1
                continue

            dee_features.append(dee_feature)

        logger.info('{} documents, ignore {} examples, truncate {} docs, {} sents, {} spans'.format(
            len(dee_examples), remove_ex_cnt,
            self.truncate_doc_count, self.ner_fea_converter.truncate_count, self.truncate_span_count
        ))

        return dee_features


def convert_dee_features_to_dataset(dee_features):
    # just view a list of doc_fea as the dataset, that only requires __len__, __getitem__
    assert len(dee_features) > 0 and isinstance(dee_features[0], DEEFeature)

    return dee_features


def prepare_doc_batch_dict(doc_fea_list):
    doc_batch_keys = ['ex_idx', 'doc_token_ids', 'doc_token_masks', 'doc_token_labels', 'valid_sent_num']
    doc_batch_dict = {}
    for key in doc_batch_keys:
        doc_batch_dict[key] = [getattr(doc_fea, key) for doc_fea in doc_fea_list]

    return doc_batch_dict


def measure_dee_prediction(event_type_fields_pairs, features, event_decode_results,
                           dump_json_path=None):
    pred_record_mat_list = []
    gold_record_mat_list = []
    for term in event_decode_results:
        ex_idx, pred_event_type_labels, pred_record_mat = term[:3]
        pred_record_mat = [
            [
                [
                    tuple(arg_tup) if arg_tup is not None else None
                    for arg_tup in pred_record
                ] for pred_record in pred_records
            ] if pred_records is not None else None
            for pred_records in pred_record_mat
        ]
        doc_fea = features[ex_idx]
        assert isinstance(doc_fea, DEEFeature)
        gold_record_mat = [
            [
                [
                    tuple(doc_fea.span_token_ids_list[arg_idx]) if arg_idx is not None else None
                    for arg_idx in event_arg_idxs
                ] for event_arg_idxs in event_arg_idxs_objs
            ] if event_arg_idxs_objs is not None else None
            for event_arg_idxs_objs in doc_fea.event_arg_idxs_objs_list
        ]

        pred_record_mat_list.append(pred_record_mat)
        gold_record_mat_list.append(gold_record_mat)

    g_eval_res = measure_event_table_filling(
        pred_record_mat_list, gold_record_mat_list, event_type_fields_pairs, dict_return=True
    )

    if dump_json_path is not None:
        default_dump_json(g_eval_res, dump_json_path)

    return g_eval_res


def aggregate_task_eval_info(eval_dir_path, target_file_pre='dee_eval', target_file_suffix='.json',
                             dump_name='total_task_eval.pkl', dump_flag=False):
    """Enumerate the evaluation directory to collect all dumped evaluation results"""
    logger.info('Aggregate task evaluation info from {}'.format(eval_dir_path))
    data_span_type2model_str2epoch_res_list = {}
    for fn in os.listdir(eval_dir_path):
        fn_splits = fn.split('.')
        if fn.startswith(target_file_pre) and fn.endswith(target_file_suffix) and len(fn_splits) == 6:
            _, data_type, span_type, model_str, epoch, _ = fn_splits

            data_span_type = (data_type, span_type)
            if data_span_type not in data_span_type2model_str2epoch_res_list:
                data_span_type2model_str2epoch_res_list[data_span_type] = {}
            model_str2epoch_res_list = data_span_type2model_str2epoch_res_list[data_span_type]

            if model_str not in model_str2epoch_res_list:
                model_str2epoch_res_list[model_str] = []
            epoch_res_list = model_str2epoch_res_list[model_str]

            epoch = int(epoch)
            fp = os.path.join(eval_dir_path, fn)
            eval_res = default_load_json(fp)

            epoch_res_list.append((epoch, eval_res))

    for data_span_type, model_str2epoch_res_list in data_span_type2model_str2epoch_res_list.items():
        for model_str, epoch_res_list in model_str2epoch_res_list.items():
            epoch_res_list.sort(key=lambda x: x[0])

    if dump_flag:
        dump_fp = os.path.join(eval_dir_path, dump_name)
        logger.info('Dumping {} into {}'.format(dump_name, eval_dir_path))
        default_dump_pkl(data_span_type2model_str2epoch_res_list, dump_fp)

    return data_span_type2model_str2epoch_res_list


def print_total_eval_info(data_span_type2model_str2epoch_res_list,
                          metric_type='micro',
                          span_type='pred_span',
                          model_strs=('DCFEE-O', 'DCFEE-M', 'GreedyDec', 'Doc2EDAG'),
                          target_set='test'):
    """Print the final performance by selecting the best epoch on dev set and emitting performance on test set"""
    dev_type = 'dev'
    test_type = 'test'
    avg_type2prf1_keys = {
        'macro': ('MacroPrecision', 'MacroRecall', 'MacroF1'),
        'micro': ('MicroPrecision', 'MicroRecall', 'MicroF1'),
    }

    name_key = 'EventType'
    p_key, r_key, f_key = avg_type2prf1_keys[metric_type]

    def get_avg_event_score(epoch_res):
        eval_res = epoch_res[1]
        avg_event_score = eval_res[-1][f_key]

        return avg_event_score

    dev_model_str2epoch_res_list = data_span_type2model_str2epoch_res_list[(dev_type, span_type)]
    test_model_str2epoch_res_list = data_span_type2model_str2epoch_res_list[(test_type, span_type)]

    has_header = False
    mstr_bepoch_list = []
    print('=' * 15, 'Final Performance (%) (avg_type={})'.format(metric_type), '=' * 15)
    for model_str in model_strs:
        if model_str not in dev_model_str2epoch_res_list or model_str not in test_model_str2epoch_res_list:
            continue

        # get the best epoch on dev set
        dev_epoch_res_list = dev_model_str2epoch_res_list[model_str]
        best_dev_epoch, best_dev_res = max(dev_epoch_res_list, key=get_avg_event_score)

        test_epoch_res_list = test_model_str2epoch_res_list[model_str]
        best_test_epoch = None
        best_test_res = None
        for test_epoch, test_res in test_epoch_res_list:
            if test_epoch == best_dev_epoch:
                best_test_epoch = test_epoch
                best_test_res = test_res
        assert best_test_epoch is not None
        mstr_bepoch_list.append((model_str, best_test_epoch))

        if target_set == 'test':
            target_eval_res = best_test_res
        else:
            target_eval_res = best_dev_res

        align_temp = '{:20}'
        head_str = align_temp.format('ModelType')
        eval_str = align_temp.format(model_str)
        head_temp = ' \t {}'
        eval_temp = ' \t & {:.1f} & {:.1f} & {:.1f}'
        ps = []
        rs = []
        fs = []
        for tgt_event_res in target_eval_res[:-1]:
            head_str += align_temp.format(head_temp.format(tgt_event_res[0][name_key]))
            p, r, f1 = (100 * tgt_event_res[0][key] for key in [p_key, r_key, f_key])
            eval_str += align_temp.format(eval_temp.format(p, r, f1))
            ps.append(p)
            rs.append(r)
            fs.append(f1)

        head_str += align_temp.format(head_temp.format('Average'))
        ap, ar, af1 = (x for x in [np.mean(ps), np.mean(rs), np.mean(fs)])
        eval_str += align_temp.format(eval_temp.format(ap, ar, af1))

        head_str += align_temp.format(head_temp.format('Total ({})'.format(metric_type)))
        g_avg_res = target_eval_res[-1]
        ap, ar, af1 = (100 * g_avg_res[key] for key in [p_key, r_key, f_key])
        eval_str += align_temp.format(eval_temp.format(ap, ar, af1))

        if not has_header:
            print(head_str)
            has_header = True
        print(eval_str)

    return mstr_bepoch_list


# evaluation dump file name template
# dee_eval.[DataType].[SpanType].[ModelStr].[Epoch].(pkl|json)
decode_dump_template = 'dee_eval.{}.{}.{}.{}.pkl'
eval_dump_template = 'dee_eval.{}.{}.{}.{}.json'


def resume_decode_results(base_dir, data_type, span_type, model_str, epoch):
    decode_fn = decode_dump_template.format(data_type, span_type, model_str, epoch)
    decode_fp = os.path.join(base_dir, decode_fn)
    logger.info('Resume decoded results from {}'.format(decode_fp))
    decode_results = default_load_pkl(decode_fp)

    return decode_results


def resume_eval_results(base_dir, data_type, span_type, model_str, epoch):
    eval_fn = eval_dump_template.format(data_type, span_type, model_str, epoch)
    eval_fp = os.path.join(base_dir, eval_fn)
    logger.info('Resume eval results from {}'.format(eval_fp))
    eval_results = default_load_json(eval_fp)

    return eval_results


def print_single_vs_multi_performance(mstr_bepoch_list, base_dir, features,
                                      metric_type='micro', data_type='test', span_type='pred_span'):
    model_str2decode_results = {}
    for model_str, best_epoch in mstr_bepoch_list:
        model_str2decode_results[model_str] = resume_decode_results(
            base_dir, data_type, span_type, model_str, best_epoch
        )

    single_eid_set = set([doc_fea.ex_idx for doc_fea in features if not doc_fea.is_multi_event()])
    multi_eid_set = set([doc_fea.ex_idx for doc_fea in features if doc_fea.is_multi_event()])
    event_type_fields_pairs = DEEExample.get_event_type_fields_pairs()
    event_type_list = [x for x, y in event_type_fields_pairs]

    name_key = 'EventType'
    avg_type2f1_key = {
        'micro': 'MicroF1',
        'macro': 'MacroF1',
    }
    f1_key = avg_type2f1_key[metric_type]

    model_str2etype_sf1_mf1_list = {}
    for model_str, _ in mstr_bepoch_list:
        total_decode_results = model_str2decode_results[model_str]

        single_decode_results = [dec_res for dec_res in total_decode_results if dec_res[0] in single_eid_set]
        assert len(single_decode_results) == len(single_eid_set)
        single_eval_res = measure_dee_prediction(
            event_type_fields_pairs, features, single_decode_results
        )

        multi_decode_results = [dec_res for dec_res in total_decode_results if dec_res[0] in multi_eid_set]
        assert len(multi_decode_results) == len(multi_eid_set)
        multi_eval_res = measure_dee_prediction(
            event_type_fields_pairs, features, multi_decode_results
        )

        etype_sf1_mf1_list = []
        for event_idx, (se_res, me_res) in enumerate(zip(single_eval_res[:-1], multi_eval_res[:-1])):
            assert se_res[0][name_key] == me_res[0][name_key] == event_type_list[event_idx]
            event_type = event_type_list[event_idx]
            single_f1 = se_res[0][f1_key]
            multi_f1 = me_res[0][f1_key]

            etype_sf1_mf1_list.append((event_type, single_f1, multi_f1))
        g_avg_se_res = single_eval_res[-1]
        g_avg_me_res = multi_eval_res[-1]
        etype_sf1_mf1_list.append(
            ('Total ({})'.format(metric_type), g_avg_se_res[f1_key], g_avg_me_res[f1_key])
        )
        model_str2etype_sf1_mf1_list[model_str] = etype_sf1_mf1_list

    print('=' * 15, 'Single vs. Multi (%) (avg_type={})'.format(metric_type), '=' * 15)
    align_temp = '{:20}'
    head_str = align_temp.format('ModelType')
    head_temp = ' \t {}'
    eval_temp = ' \t & {:.1f} & {:.1f} '
    for event_type in event_type_list:
        head_str += align_temp.format(head_temp.format(event_type))
    head_str += align_temp.format(head_temp.format('Total ({})'.format(metric_type)))
    head_str += align_temp.format(head_temp.format('Average'))
    print(head_str)

    for model_str, _ in mstr_bepoch_list:
        eval_str = align_temp.format(model_str)
        sf1s = []
        mf1s = []
        for _, single_f1, multi_f1 in model_str2etype_sf1_mf1_list[model_str]:
            eval_str += align_temp.format(eval_temp.format(single_f1*100, multi_f1*100))
            sf1s.append(single_f1)
            mf1s.append(multi_f1)
        avg_sf1 = np.mean(sf1s[:-1])
        avg_mf1 = np.mean(mf1s[:-1])
        eval_str += align_temp.format(eval_temp.format(avg_sf1*100, avg_mf1*100))
        print(eval_str)


def print_ablation_study(mstr_bepoch_list, base_dir, base_mstr, other_mstrs,
                         metric_type='micro', data_type='test', span_type='pred_span'):
    model_str2best_epoch = dict(mstr_bepoch_list)
    if base_mstr not in model_str2best_epoch:
        print('No base model type {}'.format(base_mstr))
        return

    base_eval = resume_eval_results(base_dir, data_type, span_type, base_mstr, model_str2best_epoch[base_mstr])
    model_str2eval_res = {
        model_str: resume_eval_results(base_dir, data_type, span_type, model_str, model_str2best_epoch[model_str])
        for model_str in other_mstrs if model_str in model_str2best_epoch
    }

    event_type_fields_pairs = DEEExample.get_event_type_fields_pairs()
    event_type_list = [x for x, y in event_type_fields_pairs]
    # name_key = 'EventType'
    # f1_key = 'AvgFieldF1'
    avg_type2f1_key = {
        'micro': 'MicroF1',
        'macro': 'MacroF1'
    }
    f1_key = avg_type2f1_key[metric_type]

    print('=' * 15, 'Ablation Study (avg_type={})'.format(metric_type), '=' * 15)
    align_temp = '{:20}'
    head_str = align_temp.format('ModelType')
    head_temp = ' \t {}'
    for event_type in event_type_list:
        head_str += align_temp.format(head_temp.format(event_type))
    head_str += align_temp.format(head_temp.format('Average ({})'.format(metric_type)))
    head_str += align_temp.format(head_temp.format('Average'))
    print(head_str)

    eval_temp = ' \t & {:.1f}'
    eval_str = align_temp.format(base_mstr)
    bf1s = []
    for base_event_res in base_eval[:-1]:
        base_f1 = base_event_res[0][f1_key]
        eval_str += align_temp.format(eval_temp.format(base_f1*100))
        bf1s.append(base_f1)
    g_avg_bf1 = base_eval[-1][f1_key]
    eval_str += align_temp.format(eval_temp.format(g_avg_bf1*100))
    avg_bf1 = np.mean(bf1s)
    eval_str += align_temp.format(eval_temp.format(avg_bf1*100))
    print(eval_str)

    inc_temp = ' \t & +{:.1f}'
    dec_temp = ' \t & -{:.1f}'
    for model_str in other_mstrs:
        if model_str in model_str2eval_res:
            eval_str = align_temp.format(model_str)
            cur_eval = model_str2eval_res[model_str]
            f1ds = []
            for base_event_res, cur_event_res in zip(base_eval[:-1], cur_eval[:-1]):
                base_f1 = base_event_res[0][f1_key]
                cur_f1 = cur_event_res[0][f1_key]
                f1_diff = cur_f1 - base_f1
                f1ds.append(f1_diff)
                f1_abs = abs(f1_diff)
                if f1_diff >= 0:
                    eval_str += align_temp.format(inc_temp.format(f1_abs*100))
                else:
                    eval_str += align_temp.format(dec_temp.format(f1_abs*100))

            g_avg_f1_diff = cur_eval[-1][f1_key] - base_eval[-1][f1_key]
            g_avg_f1_abs = abs(g_avg_f1_diff)
            if g_avg_f1_diff >= 0:
                eval_str += align_temp.format(inc_temp.format(g_avg_f1_abs*100))
            else:
                eval_str += align_temp.format(dec_temp.format(g_avg_f1_abs*100))

            avg_f1_diff = np.mean(f1ds)
            avg_f1_abs = abs(avg_f1_diff)
            if avg_f1_diff >= 0:
                eval_str += align_temp.format(inc_temp.format(avg_f1_abs*100))
            else:
                eval_str += align_temp.format(dec_temp.format(avg_f1_abs*100))

            print(eval_str)

