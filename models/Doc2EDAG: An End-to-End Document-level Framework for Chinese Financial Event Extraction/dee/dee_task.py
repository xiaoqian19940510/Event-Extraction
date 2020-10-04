# -*- coding: utf-8 -*-
# AUTHOR: Shun Zheng
# DATE: 19-9-19

import logging
import os
import torch.optim as optim
import torch.distributed as dist
from itertools import product

from .dee_helper import logger, DEEExample, DEEExampleLoader, DEEFeatureConverter, \
    convert_dee_features_to_dataset, prepare_doc_batch_dict, measure_dee_prediction, \
    decode_dump_template, eval_dump_template
from .utils import BERTChineseCharacterTokenizer, default_dump_json, default_load_pkl
from .ner_model import BertForBasicNER
from .base_task import TaskSetting, BasePytorchTask
from .event_type import event_type_fields_list
from .dee_model import Doc2EDAGModel, DCFEEModel


class DEETaskSetting(TaskSetting):
    base_key_attrs = TaskSetting.base_key_attrs
    base_attr_default_pairs = [
        ('train_file_name', 'train.json'),
        ('dev_file_name', 'dev.json'),
        ('test_file_name', 'test.json'),
        ('summary_dir_name', '/tmp/Summary'),
        ('max_sent_len', 128),
        ('max_sent_num', 64),
        ('train_batch_size', 64),
        ('gradient_accumulation_steps', 8),
        ('eval_batch_size', 2),
        ('learning_rate', 1e-4),
        ('num_train_epochs', 100),
        ('no_cuda', False),
        ('local_rank', -1),
        ('seed', 99),
        ('optimize_on_cpu', False),
        ('fp16', False),
        ('use_bert', False),  # whether to use bert as the encoder
        ('bert_model', 'bert-base-chinese'),  # use which pretrained bert model
        ('only_master_logging', True),  # whether to print logs from multiple processes
        ('resume_latest_cpt', True),  # whether to resume latest checkpoints when training for fault tolerance
        ('cpt_file_name', 'Doc2EDAG'),  # decide the identity of checkpoints, evaluation results, etc.
        ('model_type', 'Doc2EDAG'),  # decide the model class used
        ('rearrange_sent', False),  # whether to rearrange sentences
        ('use_crf_layer', True),  # whether to use CRF Layer
        ('min_teacher_prob', 0.1),  # the minimum prob to use gold spans
        ('schedule_epoch_start', 10),  # from which epoch the scheduled sampling starts
        ('schedule_epoch_length', 10),  # the number of epochs to linearly transit to the min_teacher_prob
        ('loss_lambda', 0.05),  # the proportion of ner loss
        ('loss_gamma', 1.0),  # the scaling proportion of missed span sentence ner loss
        ('add_greedy_dec', True),  # whether to add additional greedy decoding
        ('use_token_role', True),  # whether to use detailed token role
        ('seq_reduce_type', 'MaxPooling'),   # use 'MaxPooling', 'MeanPooling' or 'AWA' to reduce a tensor sequence
        # network parameters (follow Bert Base)
        ('hidden_size', 768),
        ('dropout', 0.1),
        ('ff_size', 1024),  # feed-forward mid layer size
        ('num_tf_layers', 4),  # transformer layer number
        # ablation study parameters,
        ('use_path_mem', True),  # whether to use the memory module when expanding paths
        ('use_scheduled_sampling', True),  # whether to use the scheduled sampling
        ('use_doc_enc', True),  # whether to use document-level entity encoding
        ('neg_field_loss_scaling', 3.0),  # prefer FNs over FPs
    ]

    def __init__(self, **kwargs):
        super(DEETaskSetting, self).__init__(
            self.base_key_attrs, self.base_attr_default_pairs, **kwargs
        )


class DEETask(BasePytorchTask):
    """Doc-level Event Extraction Task"""

    def __init__(self, dee_setting, load_train=True, load_dev=True, load_test=True,
                 parallel_decorate=True):
        super(DEETask, self).__init__(dee_setting, only_master_logging=dee_setting.only_master_logging)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logging('Initializing {}'.format(self.__class__.__name__))

        self.tokenizer = BERTChineseCharacterTokenizer.from_pretrained(self.setting.bert_model)
        self.setting.vocab_size = len(self.tokenizer.vocab)

        # get entity and event label name
        self.entity_label_list = DEEExample.get_entity_label_list()
        self.event_type_fields_pairs = DEEExample.get_event_type_fields_pairs()

        # build example loader
        self.example_loader_func = DEEExampleLoader(self.setting.rearrange_sent, self.setting.max_sent_len)

        # build feature converter
        if self.setting.use_bert:
            self.feature_converter_func = DEEFeatureConverter(
                self.entity_label_list, self.event_type_fields_pairs,
                self.setting.max_sent_len, self.setting.max_sent_num, self.tokenizer,
                include_cls=True, include_sep=True,
            )
        else:
            self.feature_converter_func = DEEFeatureConverter(
                self.entity_label_list, self.event_type_fields_pairs,
                self.setting.max_sent_len, self.setting.max_sent_num, self.tokenizer,
                include_cls=False, include_sep=False,
            )


        # load data
        self._load_data(
            self.example_loader_func, self.feature_converter_func, convert_dee_features_to_dataset,
            load_train=load_train, load_dev=load_dev, load_test=load_test,
        )
        # customized mini-batch producer
        self.custom_collate_fn = prepare_doc_batch_dict

        if not self.setting.use_token_role:
            # no token role conflicts with some settings
            assert self.setting.model_type == 'Doc2EDAG'
            assert self.setting.add_greedy_dec is False
            self.setting.num_entity_labels = 3  # 0: 'O', 1: 'Begin', 2: 'Inside'
        else:
            self.setting.num_entity_labels = len(self.entity_label_list)

        if self.setting.use_bert:
            ner_model = BertForBasicNER.from_pretrained(
                self.setting.bert_model, num_entity_labels = self.setting.num_entity_labels
            )
            self.setting.update_by_dict(ner_model.config.__dict__)  # BertConfig dictionary

            # substitute pooler in bert to support distributed training
            # because unused parameters will cause errors when conducting distributed all_reduce
            class PseudoPooler(object):
                def __init__(self):
                    pass

                def __call__(self, *x):
                    return x
            del ner_model.bert.pooler
            ner_model.bert.pooler = PseudoPooler()
        else:
            ner_model = None

        if self.setting.model_type == 'Doc2EDAG':
            self.model = Doc2EDAGModel(
                self.setting, self.event_type_fields_pairs, ner_model=ner_model,
            )
        elif self.setting.model_type == 'DCFEE':
            self.model = DCFEEModel(
                self.setting, self.event_type_fields_pairs, ner_model=ner_model
            )
        else:
            raise Exception('Unsupported model type {}'.format(self.setting.model_type))

        self._decorate_model(parallel_decorate=parallel_decorate)

        # prepare optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.setting.learning_rate)

        # # resume option
        # if resume_model or resume_optimizer:
        #     self.resume_checkpoint(resume_model=resume_model, resume_optimizer=resume_optimizer)

        self.min_teacher_prob = None
        self.teacher_norm = None
        self.teacher_cnt = None
        self.teacher_base = None
        self.reset_teacher_prob()

        self.logging('Successfully initialize {}'.format(self.__class__.__name__))

    def reset_teacher_prob(self):
        self.min_teacher_prob = self.setting.min_teacher_prob
        if self.train_dataset is None:
            # avoid crashing when not loading training data
            num_step_per_epoch = 500
        else:
            num_step_per_epoch = int(len(self.train_dataset) / self.setting.train_batch_size)
        self.teacher_norm = num_step_per_epoch * self.setting.schedule_epoch_length
        self.teacher_base = num_step_per_epoch * self.setting.schedule_epoch_start
        self.teacher_cnt = 0

    def get_teacher_prob(self, batch_inc_flag=True):
        if self.teacher_cnt < self.teacher_base:
            prob = 1
        else:
            prob = max(
                self.min_teacher_prob, (self.teacher_norm - self.teacher_cnt + self.teacher_base) / self.teacher_norm
            )

        if batch_inc_flag:
            self.teacher_cnt += 1

        return prob

    def get_event_idx2entity_idx2field_idx(self):
        entity_idx2entity_type = {}
        for entity_idx, entity_label in enumerate(self.entity_label_list):
            if entity_label == 'O':
                entity_type = entity_label
            else:
                entity_type = entity_label[2:]

            entity_idx2entity_type[entity_idx] = entity_type

        event_idx2entity_idx2field_idx = {}
        for event_idx, (event_name, field_types) in enumerate(self.event_type_fields_pairs):
            field_type2field_idx = {}
            for field_idx, field_type in enumerate(field_types):
                field_type2field_idx[field_type] = field_idx

            entity_idx2field_idx = {}
            for entity_idx, entity_type in entity_idx2entity_type.items():
                if entity_type in field_type2field_idx:
                    entity_idx2field_idx[entity_idx] = field_type2field_idx[entity_type]
                else:
                    entity_idx2field_idx[entity_idx] = None

            event_idx2entity_idx2field_idx[event_idx] = entity_idx2field_idx

        return event_idx2entity_idx2field_idx

    def get_loss_on_batch(self, doc_batch_dict, features=None):
        if features is None:
            features = self.train_features

        # teacher_prob = 1
        # if use_gold_span, gold spans will be used every time
        # else, teacher_prob will ensure the proportion of using gold spans
        if self.setting.use_scheduled_sampling:
            use_gold_span = False
            teacher_prob = self.get_teacher_prob()
        else:
            use_gold_span = True
            teacher_prob = 1

        try:
            loss = self.model(
                doc_batch_dict, features, use_gold_span=use_gold_span, train_flag=True, teacher_prob=teacher_prob
            )
        except Exception as e:
            print('-'*30)
            print('Exception occurs when processing ' +
                  ','.join([features[ex_idx].guid for ex_idx in doc_batch_dict['ex_idx']]))
            raise Exception('Cannot get the loss')

        return loss

    def get_event_decode_result_on_batch(self, doc_batch_dict, features=None, use_gold_span=False, heuristic_type=None):
        if features is None:
            raise Exception('Features mush be provided')

        if heuristic_type is None:
            event_idx2entity_idx2field_idx = None
        else:
            # this mapping is used to get span candidates for each event field
            event_idx2entity_idx2field_idx = self.get_event_idx2entity_idx2field_idx()

        batch_eval_results = self.model(
            doc_batch_dict, features, use_gold_span=use_gold_span, train_flag=False,
            event_idx2entity_idx2field_idx=event_idx2entity_idx2field_idx, heuristic_type=heuristic_type,
        )

        return batch_eval_results

    def train(self, save_cpt_flag=True, resume_base_epoch=None):
        self.logging('=' * 20 + 'Start Training' + '=' * 20)
        self.reset_teacher_prob()

        # resume_base_epoch arguments have higher priority over settings
        if resume_base_epoch is None:
            # whether to resume latest cpt when restarting, very useful for preemptive scheduling clusters
            if self.setting.resume_latest_cpt:
                resume_base_epoch = self.get_latest_cpt_epoch()
            else:
                resume_base_epoch = 0

        # resume cpt if possible
        if resume_base_epoch > 0:
            self.logging('Training starts from epoch {}'.format(resume_base_epoch))
            for _ in range(resume_base_epoch):
                self.get_teacher_prob()
            self.resume_cpt_at(resume_base_epoch, resume_model=True, resume_optimizer=True)
        else:
            self.logging('Training starts from scratch')

        self.base_train(
            DEETask.get_loss_on_batch,
            kwargs_dict1={},
            epoch_eval_func=DEETask.resume_save_eval_at,
            kwargs_dict2={
                'save_cpt_flag': save_cpt_flag,
                'resume_cpt_flag': False,
            },
            base_epoch_idx=resume_base_epoch,
        )

    def resume_save_eval_at(self, epoch, resume_cpt_flag=False, save_cpt_flag=True):
        if self.is_master_node():
            print('\nPROGRESS: {:.2f}%\n'.format(epoch / self.setting.num_train_epochs * 100))
        self.logging('Current teacher prob {}'.format(self.get_teacher_prob(batch_inc_flag=False)))

        if resume_cpt_flag:
            self.resume_cpt_at(epoch)

        if self.is_master_node() and save_cpt_flag:
            self.save_cpt_at(epoch)

        if self.setting.model_type == 'DCFEE':
            eval_tasks = product(['dev', 'test'], [False, True], ['DCFEE-O', 'DCFEE-M'])
        else:
            if self.setting.add_greedy_dec:
                eval_tasks = product(['dev', 'test'], [False, True], ['GreedyDec', None])
            else:
                eval_tasks = product(['dev', 'test'], [False, True], [None])

        for task_idx, (data_type, gold_span_flag, heuristic_type) in enumerate(eval_tasks):
            if self.in_distributed_mode() and task_idx % dist.get_world_size() != dist.get_rank():
                continue

            if data_type == 'test':
                features = self.test_features
                dataset = self.test_dataset
            elif data_type == 'dev':
                features = self.dev_features
                dataset = self.dev_dataset
            else:
                raise Exception('Unsupported data type {}'.format(data_type))

            if gold_span_flag:
                span_str = 'gold_span'
            else:
                span_str = 'pred_span'

            if heuristic_type is None:
                # store user-provided name
                model_str = self.setting.cpt_file_name.replace('.', '~')
            else:
                model_str = heuristic_type

            decode_dump_name = decode_dump_template.format(data_type, span_str, model_str, epoch)
            eval_dump_name = eval_dump_template.format(data_type, span_str, model_str, epoch)
            self.eval(features, dataset, use_gold_span=gold_span_flag, heuristic_type=heuristic_type,
                      dump_decode_pkl_name=decode_dump_name, dump_eval_json_name=eval_dump_name)

    def save_cpt_at(self, epoch):
        self.save_checkpoint(cpt_file_name='{}.cpt.{}'.format(self.setting.cpt_file_name, epoch), epoch=epoch)

    def resume_cpt_at(self, epoch, resume_model=True, resume_optimizer=False):
        self.resume_checkpoint(cpt_file_name='{}.cpt.{}'.format(self.setting.cpt_file_name, epoch),
                               resume_model=resume_model, resume_optimizer=resume_optimizer)

    def get_latest_cpt_epoch(self):
        prev_epochs = []
        for fn in os.listdir(self.setting.model_dir):
            if fn.startswith('{}.cpt'.format(self.setting.cpt_file_name)):
                try:
                    epoch = int(fn.split('.')[-1])
                    prev_epochs.append(epoch)
                except Exception as e:
                    continue
        prev_epochs.sort()

        if len(prev_epochs) > 0:
            latest_epoch = prev_epochs[-1]
            self.logging('Pick latest epoch {} from {}'.format(latest_epoch, str(prev_epochs)))
        else:
            latest_epoch = 0
            self.logging('No previous epoch checkpoints, just start from scratch')

        return latest_epoch

    def eval(self, features, dataset, use_gold_span=False, heuristic_type=None,
             dump_decode_pkl_name=None, dump_eval_json_name=None):
        self.logging('=' * 20 + 'Start Evaluation' + '=' * 20)

        if dump_decode_pkl_name is not None:
            dump_decode_pkl_path = os.path.join(self.setting.output_dir, dump_decode_pkl_name)
            self.logging('Dumping decode results into {}'.format(dump_decode_pkl_name))
        else:
            dump_decode_pkl_path = None

        total_event_decode_results = self.base_eval(
            dataset, DEETask.get_event_decode_result_on_batch,
            reduce_info_type='none', dump_pkl_path=dump_decode_pkl_path,
            features=features, use_gold_span=use_gold_span, heuristic_type=heuristic_type,
        )

        self.logging('Measure DEE Prediction')

        if dump_eval_json_name is not None:
            dump_eval_json_path = os.path.join(self.setting.output_dir, dump_eval_json_name)
            self.logging('Dumping eval results into {}'.format(dump_eval_json_name))
        else:
            dump_eval_json_path = None

        total_eval_res = measure_dee_prediction(
            self.event_type_fields_pairs, features, total_event_decode_results,
            dump_json_path=dump_eval_json_path
        )

        return total_event_decode_results, total_eval_res

    def reevaluate_dee_prediction(self, target_file_pre='dee_eval', target_file_suffix='.pkl',
                                  dump_flag=False):
        """Enumerate the evaluation directory to collect all dumped evaluation results"""
        eval_dir_path = self.setting.output_dir
        logger.info('Re-evaluate dee predictions from {}'.format(eval_dir_path))
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

                if data_type == 'dev':
                    features = self.dev_features
                elif data_type == 'test':
                    features = self.test_features
                else:
                    raise Exception('Unsupported data type {}'.format(data_type))

                epoch = int(epoch)
                fp = os.path.join(eval_dir_path, fn)
                self.logging('Re-evaluating {}'.format(fp))
                event_decode_results = default_load_pkl(fp)
                total_eval_res = measure_dee_prediction(
                    event_type_fields_list, features, event_decode_results
                )

                if dump_flag:
                    fp = fp.rstrip('.pkl') + '.json'
                    self.logging('Dumping {}'.format(fp))
                    default_dump_json(total_eval_res, fp)

                epoch_res_list.append((epoch, total_eval_res))

        for data_span_type, model_str2epoch_res_list in data_span_type2model_str2epoch_res_list.items():
            for model_str, epoch_res_list in model_str2epoch_res_list.items():
                epoch_res_list.sort(key=lambda x: x[0])

        return data_span_type2model_str2epoch_res_list


