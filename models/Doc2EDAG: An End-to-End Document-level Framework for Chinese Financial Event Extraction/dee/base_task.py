# -*- coding: utf-8 -*-
# AUTHOR: Shun Zheng
# DATE: 19-9-19
# Code Reference: pytorch-pretrained-bert (https://github.com/huggingface/pytorch-transformers)

import logging
import random
import os
import json
import sys
import numpy as np
from datetime import datetime
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.nn.parallel as para
from pytorch_pretrained_bert.optimization import BertAdam
from tqdm import trange, tqdm
from tensorboardX import SummaryWriter

from .utils import default_dump_pkl, default_dump_json


PY2 = sys.version_info[0] == 2
PY3 = sys.version_info[0] == 3
if PY2:
    import collections
    container_abcs = collections
elif PY3:
    import collections.abc
    container_abcs = collections.abc


logger = logging.getLogger(__name__)


class TaskSetting(object):
    """Base task setting that can be initialized with a dictionary"""
    base_key_attrs = ['data_dir', 'model_dir', 'output_dir']
    base_attr_default_pairs = [
        ('bert_model', 'bert-base-chinese'),
        ('train_file_name', 'train.json'),
        ('dev_file_name', 'dev.json'),
        ('test_file_name', 'test.json'),
        ('max_seq_len', 128),
        ('train_batch_size', 32),
        ('eval_batch_size', 256),
        ('learning_rate', 1e-4),
        ('num_train_epochs', 3.0),
        ('warmup_proportion', 0.1),
        ('no_cuda', False),
        ('local_rank', -1),
        ('seed', 99),
        ('gradient_accumulation_steps', 1),
        ('optimize_on_cpu', False),
        ('fp16', False),
        ('loss_scale', 128),
        ('cpt_file_name', 'task.cpt'),
        ('summary_dir_name', '/root/summary'),
    ]

    def __init__(self, key_attrs, attr_default_pairs, **kwargs):
        for key_attr in TaskSetting.base_key_attrs:
            setattr(self, key_attr, kwargs[key_attr])

        for attr, val in TaskSetting.base_attr_default_pairs:
            setattr(self, attr, val)

        for key_attr in key_attrs:
            setattr(self, key_attr, kwargs[key_attr])

        for attr, val in attr_default_pairs:
            if attr in kwargs:
                setattr(self, attr, kwargs[attr])
            else:
                setattr(self, attr, val)

    def update_by_dict(self, config_dict):
        for key, val in config_dict.items():
            setattr(self, key, val)

    def dump_to(self, dir_path, file_name='task_setting.json'):
        dump_fp = os.path.join(dir_path, file_name)
        default_dump_json(self.__dict__, dump_fp)


def set_optimizer_params_grad(named_params_optimizer, named_params_model, test_nan=False):
    """
        Utility function for optimize_on_cpu and 16-bits training.
        Copy the gradient of the GPU parameters to the CPU/RAMM copy of the model
    """
    is_nan = False
    for (name_opti, param_opti), (name_model, param_model) in zip(named_params_optimizer, named_params_model):
        if name_opti != name_model:
            logger.error("name_opti != name_model: {} {}".format(name_opti, name_model))
            raise ValueError
        if param_model.grad is not None:
            if test_nan and torch.isnan(param_model.grad).sum() > 0:
                is_nan = True
            if param_opti.grad is None:
                param_opti.grad = torch.nn.Parameter(param_opti.data.new().resize_(*param_opti.data.size()))
            param_opti.grad.data.copy_(param_model.grad.data)
        else:
            param_opti.grad = None
    return is_nan


def copy_optimizer_params_to_model(named_params_model, named_params_optimizer):
    """
        Utility function for optimize_on_cpu and 16-bits training.
        Copy the parameters optimized on CPU/RAM back to the model on GPU
    """
    for (name_opti, param_opti), (name_model, param_model) in zip(named_params_optimizer, named_params_model):
        if name_opti != name_model:
            logger.error("name_opti != name_model: {} {}".format(name_opti, name_model))
            raise ValueError
        param_model.data.copy_(param_opti.data)


class BasePytorchTask(object):
    """Basic task to support deep learning models on Pytorch"""

    def __init__(self, setting, only_master_logging=False):
        self.setting = setting
        self.logger = logging.getLogger(self.__class__.__name__)
        self.only_master_logging = only_master_logging

        if self.in_distributed_mode() and not dist.is_initialized():
            # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
            dist.init_process_group(backend='nccl')
            # dist.init_process_group(backend='gloo')  # 3 times slower than nccl for gpu training
            torch.cuda.set_device(self.setting.local_rank)
            self.logging('World Size {} Rank {}, Local Rank {}, Device Num {}, Device {}'.format(
                dist.get_world_size(), dist.get_rank(), self.setting.local_rank,
                torch.cuda.device_count(), torch.cuda.current_device()
            ))
            dist.barrier()

        self._check_setting_validity()
        self._init_device()
        self.reset_random_seed()
        self.summary_writer = None

        # ==> task-specific initialization
        # The following functions should be called specifically in inherited classes

        self.custom_collate_fn = None
        self.train_examples = None
        self.train_features = None
        self.train_dataset = None
        self.dev_examples = None
        self.dev_features = None
        self.dev_dataset = None
        self.test_examples = None
        self.test_features = None
        self.test_dataset = None
        # self._load_data()

        self.model = None
        # self._decorate_model()

        self.optimizer = None
        self.num_train_steps = None
        self.model_named_parameters = None
        # self._init_bert_optimizer()
        # (option) self.resume_checkpoint()

    def logging(self, msg, level=logging.INFO):
        if self.in_distributed_mode():
            msg = 'Rank {} {}'.format(dist.get_rank(), msg)
        if self.only_master_logging:
            if self.is_master_node():
                self.logger.log(level, msg)
        else:
            self.logger.log(level, msg)

    def _check_setting_validity(self):
        self.logging('='*20 + 'Check Setting Validity' + '='*20)
        self.logging('Setting: {}'.format(
            json.dumps(self.setting.__dict__, ensure_ascii=False, indent=2)
        ))

        # check valid grad accumulate step
        if self.setting.gradient_accumulation_steps < 1:
            raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                self.setting.gradient_accumulation_steps))
        # reset train batch size
        self.setting.train_batch_size = int(self.setting.train_batch_size
                                            / self.setting.gradient_accumulation_steps)

        # check output dir
        if os.path.exists(self.setting.output_dir) and os.listdir(self.setting.output_dir):
            self.logging("Output directory ({}) already exists and is not empty.".format(self.setting.output_dir),
                         level=logging.WARNING)
        os.makedirs(self.setting.output_dir, exist_ok=True)

        # check model dir
        if os.path.exists(self.setting.model_dir) and os.listdir(self.setting.model_dir):
            self.logging("Model directory ({}) already exists and is not empty.".format(self.setting.model_dir),
                         level=logging.WARNING)
        os.makedirs(self.setting.model_dir, exist_ok=True)

    def _init_device(self):
        self.logging('='*20 + 'Init Device' + '='*20)

        # set device
        if self.setting.local_rank == -1 or self.setting.no_cuda:
            self.device = torch.device("cuda" if torch.cuda.is_available() and not self.setting.no_cuda else "cpu")
            self.n_gpu = torch.cuda.device_count()
        else:
            self.device = torch.device("cuda", self.setting.local_rank)
            self.n_gpu = 1
            if self.setting.fp16:
                self.logging("16-bits training currently not supported in distributed training")
                self.setting.fp16 = False  # (see https://github.com/pytorch/pytorch/pull/13496)
        self.logging("device {} n_gpu {} distributed training {}".format(
            self.device, self.n_gpu,self.in_distributed_mode()
        ))

    def reset_random_seed(self, seed=None):
        if seed is None:
            seed = self.setting.seed
        self.logging('='*20 + 'Reset Random Seed to {}'.format(seed) + '='*20)

        # set random seeds
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if self.n_gpu > 0:
            torch.cuda.manual_seed_all(seed)

    def is_master_node(self):
        if self.in_distributed_mode():
            if dist.get_rank() == 0:
                return True
            else:
                return False
        else:
            return True

    def in_distributed_mode(self):
        return self.setting.local_rank >= 0

    def _init_summary_writer(self):
        if self.is_master_node():
            self.logging('Init Summary Writer')
            current_time = datetime.now().strftime('%b%d_%H-%M-%S')
            sum_dir = '{}-{}'.format(self.setting.summary_dir_name, current_time)
            self.summary_writer = SummaryWriter(sum_dir)
            self.logging('Writing summary into {}'.format(sum_dir))

        if self.in_distributed_mode():
            # TODO: maybe this can be removed
            dist.barrier()

    def load_example_feature_dataset(self, load_example_func, convert_to_feature_func, convert_to_dataset_func,
                                     file_name=None, file_path=None):
        if file_name is None and file_path is None:
            raise Exception('Either file name or file path should be provided')

        if file_path is None:
            file_path = os.path.join(self.setting.data_dir, file_name)

        if os.path.exists(file_path):
            self.logging('Load example feature dataset from {}'.format(file_path))
            examples = load_example_func(file_path)
            features = convert_to_feature_func(examples)
            dataset = convert_to_dataset_func(features)
        else:
            self.logging('Warning: file does not exists, {}'.format(file_path))
            examples = None
            features = None
            dataset = None

        return examples, features, dataset

    def _load_data(self, load_example_func, convert_to_feature_func, convert_to_dataset_func,
                   load_train=True, load_dev=True, load_test=True):
        self.logging('='*20 + 'Load Task Data' + '='*20)
        # prepare data
        if load_train:
            self.logging('Load train portion')
            self.train_examples, self.train_features, self.train_dataset = self.load_example_feature_dataset(
                load_example_func, convert_to_feature_func, convert_to_dataset_func,
                file_name=self.setting.train_file_name
            )
        else:
            self.logging('Do not load train portion')

        if load_dev:
            self.logging('Load dev portion')
            self.dev_examples, self.dev_features, self.dev_dataset = self.load_example_feature_dataset(
                load_example_func, convert_to_feature_func, convert_to_dataset_func,
                file_name=self.setting.dev_file_name
            )
        else:
            self.logging('Do not load dev portion')

        if load_test:
            self.logging('Load test portion')
            self.test_examples, self.test_features, self.test_dataset = self.load_example_feature_dataset(
                load_example_func, convert_to_feature_func, convert_to_dataset_func,
                file_name=self.setting.test_file_name
            )
        else:
            self.logging('Do not load test portion')

    def reload_data(self, load_example_func, convert_to_feature_func, convert_to_dataset_func,
                    data_type='return', file_name=None, file_path=None):
        """Subclass should inherit this function to omit function arguments"""
        if data_type.lower() == 'train':
            self.train_examples, self.train_features, self.train_dataset = \
                self.load_example_feature_dataset(
                    load_example_func, convert_to_feature_func, convert_to_dataset_func,
                    file_name=file_name, file_path=file_path
                )
        elif data_type.lower() == 'dev':
            self.dev_examples, self.dev_features, self.dev_dataset = \
                self.load_example_feature_dataset(
                    load_example_func, convert_to_feature_func, convert_to_dataset_func,
                    file_name=file_name, file_path=file_path
                )
        elif data_type.lower() == 'test':
            self.test_examples, self.test_features, self.test_dataset = \
                self.load_example_feature_dataset(
                    load_example_func, convert_to_feature_func, convert_to_dataset_func,
                    file_name=file_name, file_path=file_path
                )
        elif data_type.lower() == 'return':
            examples, features, dataset = self.load_example_feature_dataset(
                load_example_func, convert_to_feature_func, convert_to_dataset_func,
                file_name=file_name, file_path=file_path,
            )

            return examples, features, dataset
        else:
            raise Exception('Unexpected data type {}'.format(data_type))

    def _decorate_model(self, parallel_decorate=True):
        self.logging('='*20 + 'Decorate Model' + '='*20)

        if self.setting.fp16:
            self.model.half()

        self.model.to(self.device)
        self.logging('Set model device to {}'.format(str(self.device)))

        if parallel_decorate:
            if self.in_distributed_mode():
                self.model = para.DistributedDataParallel(self.model,
                                                          device_ids=[self.setting.local_rank],
                                                          output_device=self.setting.local_rank)
                self.logging('Wrap distributed data parallel')
                # self.logging('In Distributed Mode, but do not use DistributedDataParallel Wrapper')
            elif self.n_gpu > 1:
                self.model = para.DataParallel(self.model)
                self.logging('Wrap data parallel')
        else:
            self.logging('Do not wrap parallel layers')

    def _init_bert_optimizer(self):
        self.logging('='*20 + 'Init Bert Optimizer' + '='*20)
        self.optimizer, self.num_train_steps, self.model_named_parameters = \
            self.reset_bert_optimizer()

    def reset_bert_optimizer(self):
        # Prepare optimizer
        if self.setting.fp16:
            model_named_parameters = [(n, param.clone().detach().to('cpu').float().requires_grad_())
                                      for n, param in self.model.named_parameters()]
        elif self.setting.optimize_on_cpu:
            model_named_parameters = [(n, param.clone().detach().to('cpu').requires_grad_())
                                      for n, param in self.model.named_parameters()]
        else:
            model_named_parameters = list(self.model.named_parameters())

        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in model_named_parameters if n not in no_decay],
                'weight_decay_rate': 0.01
            },
            {
                'params': [p for n, p in model_named_parameters if n in no_decay],
                'weight_decay_rate': 0.0
            }
        ]

        num_train_steps = int(len(self.train_examples)
                              / self.setting.train_batch_size
                              / self.setting.gradient_accumulation_steps
                              * self.setting.num_train_epochs)

        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=self.setting.learning_rate,
                             warmup=self.setting.warmup_proportion,
                             t_total=num_train_steps)

        return optimizer, num_train_steps, model_named_parameters

    def prepare_data_loader(self, dataset, batch_size, rand_flag=True):
        # prepare data loader
        if rand_flag:
            data_sampler = RandomSampler(dataset)
        else:
            data_sampler = SequentialSampler(dataset)

        if self.custom_collate_fn is None:
            dataloader = DataLoader(dataset,
                                    batch_size=batch_size,
                                    sampler=data_sampler)
        else:
            dataloader = DataLoader(dataset,
                                    batch_size=batch_size,
                                    sampler=data_sampler,
                                    collate_fn=self.custom_collate_fn)

        return dataloader

    def prepare_dist_data_loader(self, dataset, batch_size, epoch=0):
        # prepare distributed data loader
        data_sampler = DistributedSampler(dataset)
        data_sampler.set_epoch(epoch)

        if self.custom_collate_fn is None:
            dataloader = DataLoader(dataset,
                                    batch_size=batch_size,
                                    sampler=data_sampler)
        else:
            dataloader = DataLoader(dataset,
                                    batch_size=batch_size,
                                    sampler=data_sampler,
                                    collate_fn=self.custom_collate_fn)
        return dataloader

    def get_current_train_batch_size(self):
        if self.in_distributed_mode():
            train_batch_size = max(self.setting.train_batch_size // dist.get_world_size(), 1)
        else:
            train_batch_size = self.setting.train_batch_size

        return train_batch_size

    def set_batch_to_device(self, batch):
        # move mini-batch data to the proper device
        if isinstance(batch, torch.Tensor):
            batch = batch.to(self.device)

            return batch
        elif isinstance(batch, dict):
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(self.device)
                elif isinstance(value, dict) or isinstance(value, container_abcs.Sequence):
                    batch[key] = self.set_batch_to_device(value)

            return batch
        elif isinstance(batch, container_abcs.Sequence):
            # batch = [
            #     t.to(self.device) if isinstance(t, torch.Tensor) else t for t in batch
            # ]
            new_batch = []
            for value in batch:
                if isinstance(value, torch.Tensor):
                    new_batch.append(value.to(self.device))
                elif isinstance(value, dict) or isinstance(value, container_abcs.Sequence):
                    new_batch.append(self.set_batch_to_device(value))
                else:
                    new_batch.append(value)

            return new_batch
        else:
            raise Exception('Unsupported batch type {}'.format(type(batch)))

    def base_train(self, get_loss_func, kwargs_dict1={},
                   epoch_eval_func=None, kwargs_dict2={}, base_epoch_idx=0):
        assert self.model is not None

        if self.num_train_steps is None:
            self.num_train_steps = round(
                self.setting.num_train_epochs * len(self.train_examples) / self.setting.train_batch_size
            )

        train_batch_size = self.get_current_train_batch_size()

        self.logging('='*20 + 'Start Base Training' + '='*20)
        self.logging("\tTotal examples Num = {}".format(len(self.train_examples)))
        self.logging("\tBatch size = {}".format(self.setting.train_batch_size))
        self.logging("\tNum steps = {}".format(self.num_train_steps))
        if self.in_distributed_mode():
            self.logging("\tWorker Batch Size = {}".format(train_batch_size))
        self._init_summary_writer()

        # prepare data loader
        train_dataloader = self.prepare_data_loader(
            self.train_dataset, self.setting.train_batch_size, rand_flag=True
        )

        # enter train mode
        global_step = 0
        self.model.train()

        self.logging('Reach the epoch beginning')
        for epoch_idx in trange(base_epoch_idx, int(self.setting.num_train_epochs), desc="Epoch"):
            iter_desc = 'Iteration'
            if self.in_distributed_mode():
                train_dataloader = self.prepare_dist_data_loader(
                    self.train_dataset, train_batch_size, epoch=epoch_idx
                )
                iter_desc = 'Rank {} {}'.format(dist.get_rank(), iter_desc)

            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0

            if self.only_master_logging:
                if self.is_master_node():
                    step_batch_iter = enumerate(tqdm(train_dataloader, desc=iter_desc))
                else:
                    step_batch_iter = enumerate(train_dataloader)
            else:
                step_batch_iter = enumerate(tqdm(train_dataloader, desc=iter_desc))

            for step, batch in step_batch_iter:
                batch = self.set_batch_to_device(batch)

                # forward
                loss = get_loss_func(self, batch, **kwargs_dict1)

                if self.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if self.setting.fp16 and self.setting.loss_scale != 1.0:
                    # rescale loss for fp16 training
                    # see https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html
                    loss = loss * self.setting.loss_scale
                if self.setting.gradient_accumulation_steps > 1:
                    loss = loss / self.setting.gradient_accumulation_steps

                # backward
                loss.backward()

                loss_scalar = loss.item()
                tr_loss += loss_scalar
                if self.is_master_node():
                    self.summary_writer.add_scalar('Loss', loss_scalar, global_step=global_step)
                nb_tr_examples += self.setting.train_batch_size  # may not be very accurate due to incomplete batch
                nb_tr_steps += 1
                if (step + 1) % self.setting.gradient_accumulation_steps == 0:
                    if self.setting.fp16 or self.setting.optimize_on_cpu:
                        if self.setting.fp16 and self.setting.loss_scale != 1.0:
                            # scale down gradients for fp16 training
                            for param in self.model.parameters():
                                param.grad.data = param.grad.data / self.setting.loss_scale
                        is_nan = set_optimizer_params_grad(
                            self.model_named_parameters, self.model.named_parameters(), test_nan=True
                        )
                        if is_nan:
                            self.logging("FP16 TRAINING: Nan in gradients, reducing loss scaling")
                            self.setting.loss_scale = self.setting.loss_scale / 2
                            self.model.zero_grad()
                            continue
                        self.optimizer.step()
                        copy_optimizer_params_to_model(
                            self.model.named_parameters(), self.model_named_parameters
                        )
                    else:
                        self.optimizer.step()

                    self.model.zero_grad()
                    global_step += 1

            if epoch_eval_func is not None:
                epoch_eval_func(self, epoch_idx + 1, **kwargs_dict2)

    def base_eval(self, eval_dataset, get_info_on_batch, reduce_info_type='mean', dump_pkl_path=None, **func_kwargs):
        self.logging('='*20 + 'Start Base Evaluation' + '='*20)
        self.logging("\tNum examples = {}".format(len(eval_dataset)))
        self.logging("\tBatch size = {}".format(self.setting.eval_batch_size))
        self.logging("\tReduce type = {}".format(reduce_info_type))

        # prepare data loader
        eval_dataloader = self.prepare_data_loader(
            eval_dataset, self.setting.eval_batch_size, rand_flag=False
        )

        # enter eval mode
        total_info = []
        if self.model is not None:
            self.model.eval()

        iter_desc = 'Iteration'
        if self.in_distributed_mode():
            iter_desc = 'Rank {} {}'.format(dist.get_rank(), iter_desc)

        for step, batch in enumerate(tqdm(eval_dataloader, desc=iter_desc)):
            batch = self.set_batch_to_device(batch)

            with torch.no_grad():
                # this func must run batch_info = model(batch_input)
                # and metrics is an instance of torch.Tensor with Size([batch_size, ...])
                # to fit the DataParallel and DistributedParallel functionality
                batch_info = get_info_on_batch(self, batch, **func_kwargs)
            # append metrics from this batch to event_info
            if isinstance(batch_info, torch.Tensor):
                total_info.append(
                    batch_info.to(torch.device('cpu'))  # collect results in cpu memory
                )
            else:
                # batch_info is a list of some info on each example
                total_info.extend(batch_info)

        if isinstance(total_info[0], torch.Tensor):
            # transform event_info to torch.Tensor
            total_info = torch.cat(total_info, dim=0)

        # [batch_size, ...] -> [...]
        if reduce_info_type.lower() == 'sum':
            reduced_info = total_info.sum(dim=0)
        elif reduce_info_type.lower() == 'mean':
            reduced_info = total_info.mean(dim=0)
        elif reduce_info_type.lower() == 'none':
            reduced_info = total_info
        else:
            raise Exception('Unsupported reduce metric type {}'.format(reduce_info_type))

        if dump_pkl_path is not None:
            default_dump_pkl(reduced_info, dump_pkl_path)

        return reduced_info

    def save_checkpoint(self, cpt_file_name=None, epoch=None):
        self.logging('='*20 + 'Dump Checkpoint' + '='*20)
        if cpt_file_name is None:
            cpt_file_name = self.setting.cpt_file_name
        cpt_file_path = os.path.join(self.setting.model_dir, cpt_file_name)
        self.logging('Dump checkpoint into {}'.format(cpt_file_path))

        store_dict = {
            'setting': self.setting.__dict__,
        }

        if self.model:
            if isinstance(self.model, para.DataParallel) or \
                    isinstance(self.model, para.DistributedDataParallel):
                model_state = self.model.module.state_dict()
            else:
                model_state = self.model.state_dict()
            store_dict['model_state'] = model_state
        else:
            self.logging('No model state is dumped', level=logging.WARNING)

        if self.optimizer:
            store_dict['optimizer_state'] = self.optimizer.state_dict()
        else:
            self.logging('No optimizer state is dumped', level=logging.WARNING)

        if epoch:
            store_dict['epoch'] = epoch

        torch.save(store_dict, cpt_file_path)

    def resume_checkpoint(self, cpt_file_path=None, cpt_file_name=None,
                          resume_model=True, resume_optimizer=False, strict=False):
        self.logging('='*20 + 'Resume Checkpoint' + '='*20)
        # decide cpt_file_path to resume
        if cpt_file_path is None:  # use provided path with highest priority
            if cpt_file_name is None:  # no path and no name will resort to the default cpt name
                cpt_file_name = self.setting.cpt_file_name
            cpt_file_path = os.path.join(self.setting.model_dir, cpt_file_name)
        elif cpt_file_name is not None:  # error when path and name are both provided
            raise Exception('Confused about path {} or file name {} to resume'.format(
                cpt_file_path, cpt_file_name
            ))

        if os.path.exists(cpt_file_path):
            self.logging('Resume checkpoint from {}'.format(cpt_file_path))
        elif strict:
            raise Exception('Checkpoint does not exist, {}'.format(cpt_file_path))
        else:
            self.logging('Checkpoint does not exist, {}'.format(cpt_file_path), level=logging.WARNING)
            return

        if torch.cuda.device_count() == 0:
            store_dict = torch.load(cpt_file_path, map_location='cpu')
        else:
            store_dict = torch.load(cpt_file_path, map_location=self.device)

        self.logging('Setting: {}'.format(
            json.dumps(store_dict['setting'], ensure_ascii=False, indent=2)
        ))

        if resume_model:
            if self.model and 'model_state' in store_dict:
                if isinstance(self.model, para.DataParallel) or \
                        isinstance(self.model, para.DistributedDataParallel):
                    self.model.module.load_state_dict(store_dict['model_state'])
                else:
                    self.model.load_state_dict(store_dict['model_state'])
                self.logging('Resume model successfully')
            elif strict:
                raise Exception('Resume model failed, dict.keys = {}'.format(store_dict.keys()))
        else:
            self.logging('Do not resume model')

        if resume_optimizer:
            if self.optimizer and 'optimizer_state' in store_dict:
                self.optimizer.load_state_dict(store_dict['optimizer_state'])
                self.logging('Resume optimizer successfully')
            elif strict:
                raise Exception('Resume optimizer failed, dict.keys = {}'.format(store_dict.keys()))
        else:
            self.logging('Do not resume optimizer')


def average_gradients(model):
    """ Gradient averaging. """
    size = float(dist.get_world_size())
    for name, param in model.named_parameters():
        try:
            dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
            param.grad.data /= size
        except Exception as e:
            logger.error('Error when all_reduce parameter {}, size={}, grad_type={}, error message {}'.format(
                name, param.size(), param.grad.data.dtype, repr(e)
            ))









