#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for token classification.
"""
# You can also adapt this script on your own token classification task and datasets. Pointers for this are left as
# comments.

import csv
import json
import logging
import os
import random
import sys
sys.dont_write_bytecode = True
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from re import I
from types import MethodDescriptorType
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from datasets import ClassLabel, concatenate_datasets, load_dataset, load_metric

import transformers
from transformers import (
    AutoTokenizer,
    GPT2TokenizerFast,
    HfArgumentParser,
    PreTrainedTokenizerFast,
    TrainingArguments,
    Trainer,
    set_seed,
)
from transformers.file_utils import (
    PaddingStrategy,
    add_end_docstrings,
    is_datasets_available,
    is_sagemaker_mp_enabled,
    is_tf_available,
    is_torch_available,
    is_torch_tpu_available,
)
from transformers.modeling_utils import unwrap_model
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version

from models.modeling_gpt2 import GPT2ForTokenClassification
from models.configuration_gpt2 import GPT2Config
from trainer import JointTrainer

if is_datasets_available():
    import datasets

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl

if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from transformers.trainer_pt_utils import smp_forward_backward, smp_forward_only, smp_gather, smp_nested_concat

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.5.0")

logger = logging.getLogger(__name__)

# import atexit
# import line_profiler
# profile = line_profiler.LineProfiler()
# atexit.register(profile.print_stats)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    classifier_type: str = field(
        default="linear",
        metadata={"help": "NER classifier head type: linear|crf|partial-crf"},
    )
    beta: Optional[float] = field(
        default=1,
        metadata={"help": "coefficient for attention regularization loss "},
    )
    alpha: Optional[float] = field(
        default=0.5,
        metadata={"help": "coefficient for logic regularization loss "},
    )
    n_embd: Optional[int] = field(default=768, metadata={"help": "hidden or embedding dimensionality "})
    use_subtoken_mask: bool = field(
        default=False,
        metadata={"help": "Wether use subtoken mask when calculating loss for GPT2TokenClassification model."},
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    task_name: Optional[str] = field(default="ner", metadata={"help": "The name of the task (ner, pos...)."})
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a csv or JSON file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate on (a csv or JSON file)."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input test data file to predict on (a csv or JSON file)."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
            "efficient on GPU but very bad for TPU."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_val_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of validation examples to this "
            "value if set."
        },
    )
    max_test_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of test examples to this "
            "value if set."
        },
    )
    label_all_tokens: bool = field(
        default=False,
        metadata={
            "help": "Whether to put the label for one word on all tokens of generated by that word or just on the "
            "one (in which case the other tokens will have a padding index)."
        },
    )
    return_entity_level_metrics: bool = field(
        default=False,
        metadata={"help": "Whether to return all the entity levels during evaluation or just the overall ones."},
    )
    threshold: Optional[float] = field(
        default=0.5, metadata={"help": "threshold for instance selection in bootstrap training."}
    )
    sample_rate: Optional[float] = field(default=0.1, metadata={"help": "negative sampling rate."})
    use_negative_sampling: bool = field(
        default=False,
        metadata={"help": "Wether use negative sampling before feeding the data into model."},
    )
    baseline: bool = field(
        default=False,
        metadata={"help": "Wether use bootstrap or not."},
    )
    boot_start_epoch: Optional[int] = field(
        default=5, metadata={"help": "If baseline is False, the start epoch to do instance selection."}
    )
    max_new_patterns: Optional[int] = field(
        default=5, metadata={"help": "The number of new patterns add to the pattern set for each epoch."}
    )
    max_ent_range: Optional[int] = field(
        default=None,
        metadata={"help": "The number of tokens arround the query entity to be considered as entity patterns."},
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
        self.task_name = self.task_name.lower()


@dataclass
class DataCollatorForJointClassification:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (:obj:`int`, `optional`, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignore by PyTorch loss functions).
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    attention_pad_token_id: float = 0.0

    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature[label_name] for feature in features] if label_name in features[0].keys() else None
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            # Conversion to tensors will fail if we have labels as they are not of the same length yet.
            return_tensors="pt" if labels is None else None,
        )

        if labels is None:
            return batch

        # label padding (list of T -> B X T)
        seq_len = torch.tensor(batch["input_ids"]).shape[1]
        padding_side = self.tokenizer.padding_side
        if padding_side == "right":
            batch["labels"] = [label + [self.label_pad_token_id] * (seq_len - len(label)) for label in labels]
        else:
            batch["labels"] = [[self.label_pad_token_id] * (seq_len - len(label)) + label for label in labels]

        # target attention padding (list of (T) -> B X T)
        target_att = [feature["target_att"] for feature in features] if "target_att" in features[0].keys() else None
        if padding_side == "right":
            batch["target_att"] = [att + [self.attention_pad_token_id] * (seq_len - len(att)) for att in target_att]
        else:
            batch["target_att"] = [[self.attention_pad_token_id] * (seq_len - len(att)) + att for att in target_att]

        batch = {
            k: torch.tensor(v, dtype=torch.int64) if k != "target_att" else torch.tensor(v, dtype=torch.float)
            for k, v in batch.items()
        }

        return batch


class Training_Pipeline:
    def __init__(self, model_args, data_args, training_args):
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args

        # KL divergence loss
        self.kl_loss = nn.KLDivLoss(reduction="none")
        self.threshold = data_args.threshold
        self.use_negative_sampling = data_args.use_negative_sampling
        self.use_bootstrap = not data_args.baseline
        self.boot_start_epoch = data_args.boot_start_epoch
        self.max_new_patterns = data_args.max_new_patterns
        self.max_ent_range = data_args.max_ent_range
        self.ent_group = None
        self.ent_label = None
        self.ent_pred = None

        # Detecting last checkpoint.
        self.last_checkpoint = None
        if (
            os.path.isdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
        ):
            self.last_checkpoint = get_last_checkpoint(training_args.output_dir)
            if self.last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
                raise ValueError(
                    f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                    "Use --overwrite_output_dir to overcome."
                )
            elif self.last_checkpoint is not None:
                logger.info(
                    f"Checkpoint detected, resuming training at {self.last_checkpoint}. To avoid this behavior, change "
                    "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
                )
        # Setup logging
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            handlers=[logging.StreamHandler(sys.stdout)],
        )
        logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

        # Log on each process the small summary:
        logger.warning(
            f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
            + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
        )
        # Set the verbosity to info of the Transformers logger (on main process only):
        if is_main_process(training_args.local_rank):
            transformers.utils.logging.set_verbosity_info()
            transformers.utils.logging.enable_default_handler()
            transformers.utils.logging.enable_explicit_format()
        logger.info(f"Training/evaluation parameters {training_args}")

        self.deepspeed = None
        # Set seed before initializing model.
        set_seed(training_args.seed)

        # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
        # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
        # (the dataset will be downloaded automatically from the datasets Hub).
        #
        # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
        # 'text' is found. You can easily tweak this behavior (see below).
        #
        # In distributed training, the load_dataset function guarantee that only one local process can concurrently
        # download the dataset.
        if data_args.dataset_name is not None:
            # Downloading and loading a dataset from the hub.
            datasets = load_dataset(data_args.dataset_name, data_args.dataset_config_name)
        else:
            data_files = {}
            datasets = None
            if data_args.train_file is not None:
                data_files["train"] = data_args.train_file
            if data_args.validation_file is not None:
                data_files["validation"] = data_args.validation_file
            if data_args.test_file is not None:
                data_files["test"] = data_args.test_file
            extension = data_args.train_file.split(".")[-1]
            datasets = load_dataset(extension, data_files=data_files)

        # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
        # https://huggingface.co/docs/datasets/loading_datasets.html.

        if training_args.do_train:
            self.column_names = datasets["train"].column_names
            features = datasets["train"].features
        else:
            self.column_names = datasets["validation"].column_names
            features = datasets["validation"].features
        self.text_column_name = "tokens" if "tokens" in self.column_names else self.column_names[0]
        self.label_column_name = (
            f"{data_args.task_name}_tags"
            if f"{data_args.task_name}_tags" in self.column_names
            else self.column_names[1]
        )

        if isinstance(features[self.label_column_name].feature, ClassLabel):
            self.label_list = features[self.label_column_name].feature.names
            # No need to convert the labels since they are already ints.
            self.label_to_id = {i: i for i in range(len(self.label_list))}
        else:
            self.label_list = self.get_label_list(datasets["train"][self.label_column_name])
            self.label_to_id = {l: i for i, l in enumerate(self.label_list)}
        num_labels = len(self.label_list)

        # Load pretrained model and tokenizer
        #
        # Distributed training:
        # The .from_pretrained methods guarantee that only one local process can concurrently
        # download model & vocab.
        label2id = {l: i for i, l in enumerate(self.label_list)}
        id2label = {i: l for i, l in enumerate(self.label_list)}
        config = GPT2Config.from_pretrained(
            model_args.config_name if model_args.config_name else model_args.model_name_or_path,
            num_labels=num_labels,
            label2id=label2id,  # Workaround for GPT2 w/o predefined labels
            id2label=id2label,  # Workaround for GPT2 w/o predefined labels
            token_classifier_o_label_id=label2id["O"],  # GPT2TokenClassificaton specific
            token_classifier_type=model_args.classifier_type,  # GPT2TokenClassificaton specific
            finetuning_task=data_args.task_name,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            beta=model_args.beta,  # attention loss parameter
            alpha=model_args.alpha,  # logic loss parameter
            use_subtoken_mask=model_args.use_subtoken_mask,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=True,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            add_prefix_space=True,  # Workaround for GPT2
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token  # Workaround for GPT2
        self.model = GPT2ForTokenClassification.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        # torch.save(self.tokenizer, 'examples/tok_cls_result/tokenizer.pt')
        # Tokenizer check: this script requires a fast tokenizer.
        if not isinstance(self.tokenizer, PreTrainedTokenizerFast):
            raise ValueError(
                "This example script only works for models that have a fast tokenizer. Checkout the big table of models "
                "at https://huggingface.co/transformers/index.html#bigtable to find the model types that meet this "
                "requirement"
            )

        # Preprocessing the dataset
        # Padding strategy
        self.padding = "max_length" if data_args.pad_to_max_length else False

        # NOTE Behaviors of label_all_tokens
        # - When label_all_tokens is False, tokens generated by a word are labeled in the following way:
        # (1) the designated label for the first token, (2) -100 for the rest of the tokens;
        # Tokens labeled by -100 are converted to 'O' but ignored via label_masks in GPT2TokenClassification;
        # When doing viterbi_decode in the CRF/PartialCRF setting, it is important to apply masks on such subwords as well.
        # Additional note: label_all_tokens=False + label masks is consistent with 4.3 in BERT paper https://arxiv.org/pdf/1810.04805v1.pdf
        #
        # - When label_all_tokens is True, tokens generated by a word are labeled in the following way:
        # (1) the designated label for the first token, (2) the designated label for the rest of the tokens;
        # However, "B-" labels will all be distributed to all tokens following the first token, which may confuse the model;
        #
        # - Proposed changes to the behavior when label_all_token is True:
        # (1) the designated label for the first token, (2) the designated label converted to "I-" if it is "B-" for the rest of the tokens.
        # Need to be careful in cases that no corresponding "I-" label for a "B-" label; corresponding changes in get_label_list to avoid such cases.
        # Additional note: label_all_tokens=True + the proposed changes seems diiverge from the method BERT paper used,
        # but it may be more friendly to CRF/PartialCRF.

        # Data collator
        self.data_collator = DataCollatorForJointClassification(
            self.tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None
        )
        # Metrics
        self.metric = load_metric("seqeval")

        if training_args.do_train:
            if "train" not in datasets:
                raise ValueError("--do_train requires a train dataset")

            train_dataset = datasets["train"]
            if data_args.max_train_samples is not None:
                num_max_sample = min(len(train_dataset), data_args.max_train_samples)
                train_dataset = train_dataset.select(range(num_max_sample))

            # filter extremely long instances
            self.train_dataset = train_dataset.filter(
                lambda exp: len(exp["tokens"]) <= 512,
                num_proc=self.data_args.preprocessing_num_workers,
                load_from_cache_file=not self.data_args.overwrite_cache,
            )

            # Get the initial pattern set M from D with E
            self.raw_patterns = self.train_dataset.map(
                self.pattern_extract,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
            )
            _, self.patterns, self.pattern2id, self.pattern2target, self.pattern_count = self.handle_patterns(
                self.raw_patterns
            )
            # build bag of tokens from pattern set
            self.pattern_bow, self.pattern_bow_freq = self.relation_bow()

            self.get_re_ent_instance()

            # If entity range is defined, we also consider entity patterns
            if self.max_ent_range is not None:
                self.raw_ent_patterns = self.train_dataset.map(
                    self.ent_pattern_extract,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    load_from_cache_file=not data_args.overwrite_cache,
                )
                _, self.ent_patterns, self.ent_pattern2id, self.ent_pattern_count = self.handle_ent_patterns(
                    self.raw_ent_patterns
                )
                self.entity_bow, self.entity_bow_freq = self.entity_pattern_bow()  # build bag of tokens

            if self.use_bootstrap:
                # before next training, redistribute training dataset D based on M (and negative sampling)
                self.init_dataset = self.data_redistribute(self.train_dataset, initial=True)

            else:
                if self.use_negative_sampling:
                    init_dataset = self.negative_sampling(self.train_dataset)
                else:
                    init_dataset = self.train_dataset

                self.init_dataset = init_dataset.map(
                    self.tokenize_and_align_labels,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    load_from_cache_file=not data_args.overwrite_cache,
                )

            self.original_dataset = self.train_dataset.map(
                self.tokenize_and_align_labels,
                batched=True,
                num_proc=self.data_args.preprocessing_num_workers,
                load_from_cache_file=not self.data_args.overwrite_cache,
            )

        if training_args.do_eval:
            if "validation" not in datasets:
                raise ValueError("--do_eval requires a validation dataset")

            self.eval_dataset = datasets["validation"]
            if data_args.max_val_samples is not None:
                num_max_sample = min(len(self.eval_dataset), data_args.max_val_samples)
                self.eval_dataset = self.eval_dataset.select(range(num_max_sample))

            # filter extremely long instances
            self.eval_dataset = self.eval_dataset.filter(
                lambda exp: len(exp["tokens"]) <= 512,
                num_proc=self.data_args.preprocessing_num_workers,
                load_from_cache_file=not self.data_args.overwrite_cache,
            )
            self.eval_dataset = self.eval_dataset.map(
                self.tokenize_and_align_labels,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
            )
            # attention_mask, input_ids, instanceID, labels, ner_tags,
            # query_ids, special_tokens_mask, sentID, target_att, tokens
            # self.eval_dataset.to_json("examples/tok_cls_result/attention/test_ent_data.json")

        if training_args.do_predict:
            if "test" not in datasets:
                raise ValueError("--do_predict requires a test dataset")

            self.test_dataset = datasets["test"]
            if data_args.max_test_samples is not None:
                num_max_sample = min(len(self.test_dataset), data_args.max_test_samples)
                self.test_dataset = self.test_dataset.select(range(num_max_sample))
            self.test_dataset = self.test_dataset.map(
                self.tokenize_and_align_labels,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
            )

    def get_re_ent_instance(self):
        self.pos_sents = set(
            [
                sentID
                for sentID, tags in zip(self.train_dataset["sentID"], self.train_dataset["ner_tags"])
                if any("/" in tag for tag in tags)
            ]
        )
        self.re_instances = set(
            [
                insID
                for insID, tags in zip(self.train_dataset["instanceID"], self.train_dataset["ner_tags"])
                if any("/" in tag for tag in tags)
            ]
        )
        self.pos_instances = set(
            [
                insID
                for sentID, insID in zip(self.train_dataset["sentID"], self.train_dataset["instanceID"])
                if sentID in self.pos_sents
            ]
        )
        self.neg_sents = set(self.train_dataset["sentID"]) - self.pos_sents
        self.ent_instances = set(self.train_dataset["instanceID"]) - self.re_instances
        self.neg_instances = set(self.train_dataset["instanceID"]) - self.pos_instances
        print(
            "train data: # sents {}, # pos sents {}, # pos ins {}, # re ins {}, # neg sents {}, # neg ins {}, # ent ins {}".format(
                len(set(self.train_dataset["sentID"])),
                len(self.pos_sents),
                len(self.pos_instances),
                len(self.re_instances),
                len(self.neg_sents),
                len(self.neg_instances),
                len(self.ent_instances),
            )
        )

    # Tokenize all texts and align the labels with them.
    def tokenize_and_align_labels(self, examples):
        tokenized_inputs = self.tokenizer(
            examples[self.text_column_name],
            padding=self.padding,
            truncation=True,
            max_length=512,  # Workaround for GPU memory consumption
            # We use this argument because the texts in our dataset are lists of words (with a label for each word).
            is_split_into_words=True,
            return_special_tokens_mask=True,
        )
        labels = []
        queryID = []
        for i, label in enumerate(examples[self.label_column_name]):
            tokens = tokenized_inputs.tokens(batch_index=i)  # subtokens after GPT2 Tokenizer
            word_ids = tokenized_inputs.word_ids(batch_index=i)  # [0, 1, 1, 2, 3, 3, 3, 4, ...]
            previous_word_idx = None
            label_ids = []
            for j, word_idx in enumerate(word_ids):
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif (
                    word_idx != previous_word_idx
                ):  # and tokens[j].startswith("Ġ"): # ADDED condition, should be held when add_prefix_space=True
                    label_ids.append(self.label_to_id[label[word_idx]])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    # label_ids.append(label_to_id[label[word_idx]] if data_args.label_all_tokens else -100)
                    # NOTE Change behavior of label_all_tokens:
                    # NOTE The word_ids trick does not always work, e.g., a file path /usr/bin/bash will be split into subparts with different word_ids.
                    # To solve this problem, as we specific add_prefix_space=True, we can use the leading Ġ to check word boundary.
                    if self.data_args.label_all_tokens:
                        if label[word_idx].startswith("B-"):
                            ilb = "I" + label[word_idx][1:]
                            label_ids.append(self.label_to_id[ilb])
                        else:  # label starts with "I-" or "O": directly add it to label_ids
                            label_ids.append(self.label_to_id[label[word_idx]])
                    else:
                        label_ids.append(-100)
                previous_word_idx = word_idx

            labels.append(label_ids)
            # add query id
            query_id = examples["query_ids"][i]
            try:
                queryID.append([word_ids.index(query_id)])
            except:
                # if len(word_ids) == 0:
                #     print("special case: word_ids is empty!")
                # else:
                #     print("special case: queryID: {}, len word_ids: {}, last word_ids: {}".format(
                #         query_id, len(word_ids), word_ids[-1]
                #     ))
                queryID.append([0])

        ####################################################################################################
        if self.training_args.do_train:
            target_att = []
            for i, IOB_tags in enumerate(examples[self.label_column_name]):
                word_ids = tokenized_inputs.word_ids(batch_index=i)  # [0, 1, 1, 2, 3, 3, 3, 4, ...]
                if examples["instanceID"][i] < 0:  # all "O" sequence has a negative instanceID
                    atts = [0] * len(word_ids)  # target attention are all zeros
                else:
                    tokens = examples["tokens"][i]
                    # first record all the relations in this instance
                    relation_set = list(set([tag[2:] for tag in IOB_tags if "B-/" in tag]))
                    query_id = examples["query_ids"][i]
                    query_tag = IOB_tags[query_id][2:]
                    # record the entity bag of words
                    all_dist = []
                    for relation in relation_set:
                        b_m = [
                            int(tok in self.pattern_bow[relation] or (tag != "O" and "/" not in tag))
                            for tok, tag in zip(tokens, IOB_tags)
                        ]
                        # b_m = [
                        #     self.pattern_bow_freq[relation].get(tok, 0)
                        #     for tok in tokens
                        # ]
                        all_dist.append(b_m)

                    if not relation_set:  # if no relations in this instance, we only attend to the query entity
                        if self.max_ent_range is None:
                            freq = np.array([int(tag != "O" and "/" not in tag) for tag in IOB_tags])
                        else:
                            freq = [
                                int(tok in self.entity_bow[query_tag] or (tag != "O" and "/" not in tag))
                                for tok, tag in zip(tokens, IOB_tags)
                            ]
                            freq = np.array(freq)
                    else:  # otherwise, we maxpool over all relation token distribution
                        freq = np.array(all_dist).max(axis=0)  # maxpooling (T)
                    # else: # otherwise, we avgpool over all relation token distribution
                    #     freq = np.array(all_dist).mean(axis=0) # avgpooling (T)

                    # normalize and avoid trivial division
                    a_m = freq / freq.sum() if freq.sum() else freq  # normalize by sum (T)

                    # handling subtokens
                    atts = []
                    for word_idx in word_ids:
                        if word_idx is None:
                            atts.append(0)  # special tokens padding
                        else:  # We set the attention score of each subtoken the same as the original token.
                            atts.append(a_m[word_idx])

                target_att.append(atts)
        else:
            target_att = [[] for _ in examples[self.label_column_name]]
        ####################################################################################################

        tokenized_inputs["labels"] = labels
        tokenized_inputs["query_ids"] = queryID
        tokenized_inputs["target_att"] = target_att

        return tokenized_inputs

    # In the event the labels are not a `Sequence[ClassLabel]`, we will need to go through the dataset to get the unique labels.
    def get_label_list(self, labels):
        unique_labels = set()
        for label in labels:
            unique_labels = unique_labels | set(label)
        # NOTE Improvements for GPT2+CRF: check if a B-label has its corresponding I-label.
        # Related to changes of the label_all_tokens behavior.
        ilabels_to_add = set()
        if "O" not in unique_labels:
            ilabels_to_add.add("O")
        if self.data_args.label_all_tokens:
            for ulabel in unique_labels:
                if ulabel.startswith("B-"):
                    ilabel = "I" + ulabel[1:]  # B-XXX -> I-XXX
                    if ilabel not in unique_labels:
                        ilabels_to_add.add(ilabel)
        if ilabels_to_add:
            unique_labels = unique_labels | ilabels_to_add
            logger.info(f"Additional labels added: {ilabels_to_add}")
        ##
        label_list = list(unique_labels)
        label_list.sort()
        return label_list

    # def entities2dict(self, entities, queryid, ent_dict):
    #     """
    #     We build ent_dict iterately for each instance, each item contains:
    #         key: the query entity index tuple,
    #         values: a dict including the query entity tag, query entity index, and related entity info.
    #     Outputs:
    #         ent_dict (dict): {
    #             record_idx1: {"entity_group": Tag1, "related_ent": {idx1: tag1, ...}},
    #             record_idx2: {"entity_group": Tag2, "related_ent": {idx1: tag1, ...}},
    #             ...
    #         }
    #     """
    #     related_ent = dict()
    #     ent_record = None
    #     for entity in entities:
    #         tag, index= entity['entity_group'], sorted(entity['index'])
    #         if index[0] == queryid: # query entity
    #             ent_dict[tuple(index)]['entity_group'] = tag
    #             ent_record = tuple(index)
    #         else: # other related entities
    #             related_ent[tuple(index)] = tag

    #     if ent_record is not None: # if query entity exist we also record its related entities
    #         ent_dict[ent_record]['related_ent'] = related_ent
    #     else:
    #         if related_ent: # no query entity but predict other related entities
    #             ent_dict[tuple([queryid])]['entity_group'] = 'None'
    #             ent_dict[tuple([queryid])]['related_ent'] = related_ent

    # def merge_ent_dict(self, ent_dict, sent_ents):
    #     """
    #     We use the ent_dict to interately extract all triplets in the form:
    #         {"ent1": idx1, "ent1_tag": tag1, "ent2": idx2, "ent2_tag": tag2}.
    #     Each triplet is then added to sent_ents.
    #     """
    #     for ent1, items in ent_dict.items():
    #         ent1_tag = items['entity_group']
    #         if not items['related_ent']: # no related entities (empty dict)
    #             sent_ents.append({
    #                 "ent1": ent1, "ent1_tag": ent1_tag,
    #                 "ent2": tuple(), "ent2_tag": 'None',
    #             })
    #         else: # iterately append each related entity triplet
    #             for ent2, ent2_tag in items['related_ent'].items():
    #                 sent_ents.append({
    #                     "ent1": ent1, "ent1_tag": ent1_tag,
    #                     "ent2": ent2, "ent2_tag": ent2_tag,
    #                 })

    def entities2dict(self, entities, queryid, ent_dict):
        """
        We build ent_dict iterately for each instance, each item contains:
            key: the query entity index tuple,
            values: a dict including the query entity tag, query entity index, and related entity info.
        Outputs:
            ent_dict (dict): {
                record_idx1: {"entity_group": Tag1, "word": word1, "related_ent": {idx1: (tag1, word1), ...}},
                record_idx2: {"entity_group": Tag2, "word": word2, "related_ent": {idx2: (tag2, word1), ...}},
                ...
            }
        """
        related_ent = dict()
        ent_record = None
        for entity in entities:
            tag, index, word = entity["entity_group"], sorted(entity["index"]), entity["word"].strip()
            if index[0] == queryid:  # query entity
                ent_dict[tuple(index)]["entity_group"] = tag
                ent_dict[tuple(index)]["word"] = word
                ent_record = tuple(index)
            else:  # other related entities
                related_ent[tuple(index)] = (tag, word)

        if ent_record is not None:  # if query entity exist we also record its related entities
            ent_dict[ent_record]["related_ent"] = related_ent
        else:
            if related_ent:  # no query entity but predict other related entities
                ent_dict[tuple([queryid])]["entity_group"] = "None"
                ent_dict[tuple([queryid])]["word"] = ""
                ent_dict[tuple([queryid])]["related_ent"] = related_ent

    def merge_ent_dict(self, ent_dict, sent_ents):
        """
        We use the ent_dict to interately extract all triplets in the form:
            {"ent1": idx1, "ent1_tag": tag1, "ent2": idx2, "ent2_tag": tag2}.
        Each triplet is then added to sent_ents.
        """
        for ent1, items in ent_dict.items():
            ent1_tag, ent1_word = items["entity_group"], items["word"]
            if not items["related_ent"]:  # no related entities (empty dict)
                sent_ents.append(
                    {
                        "ent1": ent1_word,
                        "ent1_tag": ent1_tag,
                        "ent2": "",
                        "ent2_tag": "None",
                    }
                )
            else:  # iterately append each related entity triplet
                for ent2, (ent2_tag, ent2_word) in items["related_ent"].items():
                    sent_ents.append(
                        {
                            "ent1": ent1_word,
                            "ent1_tag": ent1_tag,
                            "ent2": ent2_word,
                            "ent2_tag": ent2_tag,
                        }
                    )

    def extract_triplets(self, grouped_entities, dataset_name="eval", is_label=True):
        """
        dataset: self.train_dataset or self.eval_dataset
        """
        if dataset_name == "eval":
            dataset = self.eval_dataset
        else:
            dataset = self.train_dataset
        sentIDs = dataset["sentID"]
        queryIDs = dataset["query_ids"]

        if is_label:  # extract triplets from grouped_labels (N X T)
            label_entities = []
            ID_set = set()
            for i, entities in enumerate(grouped_entities):
                sentid, queryid = sentIDs[i], queryIDs[i][0]  # each instance
                if sentid not in ID_set:  # new sentence
                    if i != 0:  # not the first instance
                        self.merge_ent_dict(ent_dict, sent_ents)  # merge all the entities and relations into triplets
                        label_entities.append(sent_ents)  # append each sentence triplets to output

                    ID_set.add(sentid)
                    sent_ents = []
                    ent_dict = defaultdict(dict)

                self.entities2dict(entities, queryid, ent_dict)  # build entity-relations dict

                if i == len(grouped_entities) - 1:  # last instance
                    self.merge_ent_dict(ent_dict, sent_ents)  # merge all the entities and relations into triplets
                    label_entities.append(sent_ents)  # append each sentence triplets to output

        else:  # extract triplets from grouped_preds (N X T X T')
            label_entities = []
            all_labels = dataset["labels"]
            all_sentIDs = dataset["sentID"]
            unique_pair = []
            id_set = set()
            for Id, tag in zip(all_sentIDs, all_labels):
                if Id not in id_set:
                    id_set.add(Id)
                    unique_pair.append((Id, tag))

            for i, sentence_entities in enumerate(grouped_entities):  # every sentence (T X T')
                sent_ents = []
                label = unique_pair[i][1]  # corresponding labels
                ent_dict = defaultdict(dict)  # record each entities and related entities for each sentence
                for queryid, entities in enumerate(sentence_entities):  # every query instance (T')
                    if label[queryid] != -100:  # we only extract triplets for non-subword positions
                        self.entities2dict(entities, queryid, ent_dict)  # build entity-relations dict

                self.merge_ent_dict(ent_dict, sent_ents)  # merge all the entities and relations into triplets
                label_entities.append(sent_ents)

        return label_entities

    def _common_cal(self, preds, labels):
        """
        Both preds and labels are a list of triplets (dicts).
        """
        n_hyp = len(preds)
        n_ref = len(labels)

        false_tag = 0
        re_fn = 0
        re_fp = 0
        re_tag_f = 0
        re_mention_f = 0
        ent_mention_f = 0
        ent_tag_f = 0

        # consider ent1_tag
        intersection_tag = [ent for ent in preds if ent in labels]
        tp_tag = len(intersection_tag)

        for ent1 in preds:
            for ent2 in labels:
                # if ent1 != ent2 and ent1['ent1'][0] == ent2['ent1'][0]: #  we define a corresponding pair
                if ent1 != ent2 and ent1["ent1"] == ent2["ent1"]:  #  we define a corresponding pair
                    false_tag += 1
                    if ent1["ent1"] != ent2["ent1"]:  # incorrect entity mention prediction
                        ent_mention_f += 1
                    if ent1["ent1_tag"] != ent2["ent1_tag"]:
                        if ent1["ent1_tag"] != "None" and ent2["ent1_tag"] != "None":
                            ent_tag_f += 1
                    if ent1["ent2"] != ent2["ent2"]:
                        if ent1["ent2"] != tuple() and ent2["ent2"] != tuple():
                            re_mention_f += 1
                    if ent1["ent2_tag"] != ent2["ent2_tag"]:  # incorrect relation prediction
                        if ent1["ent2_tag"] == "None" and ent2["ent2_tag"] != "None":  # relation false negative
                            re_fn += 1
                        elif ent1["ent2_tag"] != "None" and ent2["ent2_tag"] == "None":  # relation false positive
                            re_fp += 1
                        else:
                            re_tag_f += 1

        # not considering ingent1_tag
        removed_keys = [
            "ent1_tag",
        ]
        for rm_key in removed_keys:
            for ent1 in preds:
                ent1.pop(rm_key, None)
            for ent2 in labels:
                ent2.pop(rm_key, None)

        intersection_notag = [ent for ent in preds if ent in labels]
        tp_notag = len(intersection_notag)

        return (
            tp_notag,
            tp_tag,
            n_hyp,
            n_ref,
            false_tag,
            ent_mention_f,
            ent_tag_f,
            re_mention_f,
            re_fn,
            re_fp,
            re_tag_f,
        )

    def compute_metrics(self, p):
        """
        predictions logits (np.ndarray: N X T X T X V): # instances X query dimension X token dimension X label dimension.
        labels (np.ndarray: N X T (Dict)): ground truth of label_ids (corresponding to query_ids).
        """
        predictions, labels = p  # N X T X T X V, N X T
        grouped_preds = self.preds_to_grouped_entity(preds=predictions)
        # remove repeated preds for the same sentence
        sent_id_pool = set()
        remove_idx = []
        for i, sent_id in enumerate(self.eval_dataset["sentID"]):
            if sent_id not in sent_id_pool:
                sent_id_pool.add(sent_id)
            else:
                remove_idx.append(i)

        grouped_preds = [preds for i, preds in enumerate(grouped_preds) if i not in remove_idx]  # N' X T X T'
        grouped_labels = self.preds_to_grouped_entity(preds=labels, is_label=True)  # N X T'

        true_predictions = self.extract_triplets(grouped_preds, is_label=False)  # N X T' (quadratic dict)
        true_labels = self.extract_triplets(grouped_labels, is_label=True)  # N X T' (quadratic dict)

        with open("pred_triplets(gpt2).csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(true_predictions)

        with open("label_triplets(gpt2).csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(true_labels)

        self.ent_group = grouped_preds
        self.ent_label = true_labels
        self.ent_pred = true_predictions

        TP_notag, TP_tag, Pos, Neg = 0, 0, 0, 0
        pred_F, ent_mention_F, ent_tag_F = 0, 0, 0
        re_mention_F, re_tag_FN, re_tag_FP, re_tag_F = 0, 0, 0, 0
        # calculate precision, recall, F1 and accuracy
        for hyp, ref in zip(true_predictions, true_labels):
            (
                tp_notag,
                tp_tag,
                n_hyp,
                n_ref,
                false_tag,
                ent_mention_f,
                ent_tag_f,
                re_mention_f,
                re_fn,
                re_fp,
                re_tag_f,
            ) = self._common_cal(hyp, ref)
            TP_notag += tp_notag
            TP_tag += tp_tag
            Pos += n_hyp
            Neg += n_ref
            pred_F += false_tag
            ent_mention_F += ent_mention_f
            ent_tag_F += ent_tag_f
            re_mention_F += re_mention_f
            re_tag_FN += re_fn
            re_tag_FP += re_fp
            re_tag_F += re_tag_f

        pre_notag = TP_notag / Pos if Pos else 0.0
        rec_notag = TP_notag / Neg if Neg else 0.0
        f1_notag = 2.0 * pre_notag * rec_notag / (pre_notag + rec_notag) if (pre_notag or rec_notag) else 0.0

        pre_tag = TP_tag / Pos if Pos else 0.0
        rec_tag = TP_tag / Neg if Neg else 0.0
        f1_tag = 2.0 * pre_tag * rec_tag / (pre_tag + rec_tag) if (pre_tag or rec_tag) else 0.0

        ent_m_fr = ent_mention_F / pred_F if pred_F else 0.0
        ent_tag_fr = ent_tag_F / pred_F if pred_F else 0.0
        re_m_fr = re_mention_F / pred_F if pred_F else 0.0
        re_tag_fnr = re_tag_FN / pred_F if pred_F else 0.0
        re_tag_fpr = re_tag_FP / pred_F if pred_F else 0.0
        re_tag_fr = re_tag_F / pred_F if pred_F else 0.0

        pred_len = [len(pred) for pred in true_predictions]
        avg_pred_len = sum(pred_len) / len(pred_len) if len(pred_len) else 0.0
        label_len = [len(label) for label in true_labels]
        avg_label_len = sum(label_len) / len(label_len) if len(label_len) else 0.0

        return {
            "precision": pre_notag,
            "recall": rec_notag,
            "f1": f1_notag,
            "precision(tag)": pre_tag,
            "recall(tag)": rec_tag,
            "f1(tag)": f1_tag,
            "ent_mention_fr": ent_m_fr,
            "ent_tag_fr": ent_tag_fr,
            "re_mention_fr": re_m_fr,
            "re_fpr": re_tag_fpr,
            "re_fnr": re_tag_fnr,
            "re_tag_fr": re_tag_fr,
            "avg_pred_len": avg_pred_len,
            "avg_true_len": avg_label_len,
        }

    def preds_to_grouped_entity(
        self,
        preds: Union[np.ndarray, Tuple[np.ndarray]] = None,
        is_label: bool = False,
        ignore_labels=["O"],
        grouped_entities: bool = True,
        ignore_subwords: bool = True,
        detect_gpt2_leading_space: bool = False,
    ):
        """
        preds (np.ndarray: N X T X T X V): prediction logits from model outputs.
        """
        if detect_gpt2_leading_space:
            if not isinstance(self.tokenizer, GPT2TokenizerFast):
                raise ValueError("tokenizer must be a GPT2TokenizerFast")
            if not self.tokenizer.add_prefix_space:
                raise ValueError(
                    "tokenizer.add_prefix_space must be set to True when detect_gpt2_leading_space is True."
                )

        if ignore_subwords and not self.tokenizer.is_fast:
            raise ValueError(
                "Slow tokenizers cannot ignore subwords. Please set the `ignore_subwords` option"
                "to `False` or use a fast tokenizer."
            )

        if isinstance(self.model, GPT2ForTokenClassification) and ignore_subwords:
            apply_gpt2_subword_mask = True
        else:
            apply_gpt2_subword_mask = False

        answers = []
        all_input_ids = self.eval_dataset["input_ids"]  # N X T
        all_special_tokens_mask = self.eval_dataset["special_tokens_mask"]  # N X T
        all_labels = self.eval_dataset["labels"]  # N X T

        if not is_label:  # preds is prediction N X T X T X V
            for i, logits in enumerate(preds):  # logits: T X T X V (T is after padding)
                input_ids = all_input_ids[i]  # T (non-padding)
                special_tokens_mask = all_special_tokens_mask[i]  # T (non-padding)
                labels = all_labels[i]  # T (non-padding)
                logits = logits[: len(input_ids), : len(input_ids)]  # remove the padding part of the logit
                sent_res = []

                for query, logit in enumerate(logits):  # T X V
                    score = np.exp(logit) / np.exp(logit).sum(-1, keepdims=True)  # T X V
                    labels_idx = score.argmax(axis=-1)  # T

                    seq_entities = self.handling_score(
                        labels_idx,
                        input_ids,
                        special_tokens_mask,
                        labels,
                        grouped_entities,
                        ignore_subwords,
                        apply_gpt2_subword_mask,
                        detect_gpt2_leading_space,
                        ignore_labels,
                        is_label=False,
                    )  # List[Dict] 1-D
                    sent_res.append(seq_entities)  # List[List[Dict]] 2-D

                answers.append(sent_res)

        else:  # preds are labels N X T
            for i, label_ids in enumerate(preds):  # label_ids: T (T is after padding)
                input_ids = all_input_ids[i]  # T (non-padding)
                special_tokens_mask = all_special_tokens_mask[i]  # T (non-padding)
                labels_idx = label_ids[: len(input_ids)]  # remove the padding part of the labels
                score = np.array([[1.0] * len(self.model.config.id2label)] * len(labels_idx))  # T X V

                seq_entities = self.handling_score(
                    labels_idx,
                    input_ids,
                    special_tokens_mask,
                    labels_idx,
                    grouped_entities,
                    ignore_subwords,
                    apply_gpt2_subword_mask,
                    detect_gpt2_leading_space,
                    ignore_labels,
                    is_label=True,
                )  # List[Dict] 1-D
                answers.append(seq_entities)

        if len(answers) == 1:
            return answers[0]

        return answers

    def handling_score(
        self,
        labels_idx,
        input_ids,
        special_tokens_mask,
        gen_labels,
        grouped_entities,
        ignore_subwords,
        apply_gpt2_subword_mask,
        detect_gpt2_leading_space,
        ignore_labels,
        is_label=False,
    ):
        entities = []
        # Filter to labels not in `self.ignore_labels`
        # Filter special_tokens
        filtered_labels_idx = []
        true_idx = None

        if is_label:  # handling groud truth labels
            for idx, label_idx in enumerate(labels_idx):
                if label_idx == -100:  # subword
                    if (
                        true_idx is not None and idx == true_idx + 1
                    ):  # the current subword belongs to the latest true token
                        filtered_labels_idx.append((idx, label_idx))
                        true_idx += 1  # the true idx is updated as the current idx
                elif self.model.config.id2label[label_idx] not in ignore_labels and not special_tokens_mask[idx]:
                    true_idx = idx  # record the latest true idx
                    filtered_labels_idx.append((idx, label_idx))
        else:  # handling predictions
            for idx, label_idx in enumerate(labels_idx):
                if gen_labels[idx] == -100:  # subword
                    if (
                        true_idx is not None and idx == true_idx + 1
                    ):  # the current subword belongs to the latest true token
                        filtered_labels_idx.append((idx, label_idx))
                        true_idx += 1  # the true idx is updated as the current idx
                elif self.model.config.id2label[label_idx] not in ignore_labels and not special_tokens_mask[idx]:
                    true_idx = idx  # record the latest true idx
                    filtered_labels_idx.append((idx, label_idx))

        for idx, label_idx in filtered_labels_idx:
            word = self.tokenizer.convert_ids_to_tokens([int(input_ids[idx])])[0]  # contains "Ġ"
            is_subword = False
            # NOTE Patch for GPT2 subword detection by using word_ids (ref. line 192)
            if apply_gpt2_subword_mask:
                is_subword = gen_labels[idx] == -100

            # NOTE GPT2 specific subword detection for special words like IP/Email addresses
            # Only be correct when add_prefix_space = True in Tokenizer
            if detect_gpt2_leading_space:
                is_subword = is_subword or not word.startswith("Ġ")

            if int(input_ids[idx]) == self.tokenizer.unk_token_id:
                is_subword = False

            if is_subword:
                entity = {
                    "word": word,
                    "entity": "B-X",
                    "index": idx,
                }
            else:
                entity = {
                    "word": word,
                    "entity": self.model.config.id2label[label_idx],
                    "index": idx,
                }

            if grouped_entities and ignore_subwords:
                entity["is_subword"] = is_subword

            entities += [entity]

        if grouped_entities:
            return self.group_entities(ignore_subwords, entities)  # Append ungrouped entities
        else:
            return entities

    def group_sub_entities(self, entities: List[dict]) -> dict:
        """
        Group together the adjacent tokens with the same entity predicted.

        Args:
            entities (:obj:`dict`): The entities predicted by the pipeline (List of entity dicts).
        """
        # Get the first entity in the entity group
        entity = entities[0]["entity"].split("-")[-1]
        tokens = [entity["word"] for entity in entities]
        index = [entity["index"] for entity in entities]

        entity_group = {
            "entity_group": entity,
            "word": self.tokenizer.convert_tokens_to_string(tokens),
            "index": index,
        }
        return entity_group

    def group_entities(self, ignore_subwords, entities: List[dict]) -> List[dict]:
        """
        Find and group together the adjacent tokens with the same entity predicted.

        Args:
            entities (:obj:`dict`): The entities predicted by the pipeline.
        """

        entity_groups = []
        entity_group_disagg = []

        if entities:
            last_idx = entities[-1]["index"]

        for entity in entities:
            is_last_idx = entity["index"] == last_idx
            is_subword = ignore_subwords and entity["is_subword"]
            if not entity_group_disagg:
                if not is_subword:  # the first entity can never be a subword
                    entity_group_disagg += [entity]
                if is_last_idx and entity_group_disagg:
                    entity_groups += [self.group_sub_entities(entity_group_disagg)]
                # print("entity group disagg: {}".format(entity_group_disagg))
                continue

            # If the current entity is similar and adjacent to the previous entity, append it to the disaggregated entity group
            # The split is meant to account for the "B" and "I" suffixes
            # Shouldn't merge if both entities are B-type
            if (
                (
                    entity["entity"].split("-")[-1] == entity_group_disagg[-1]["entity"].split("-")[-1]
                    and entity["entity"].split("-")[0] != "B"
                )
                and entity["index"] == entity_group_disagg[-1]["index"] + 1
            ) or is_subword:
                # Modify subword type to be previous_type
                if is_subword:
                    entity["entity"] = entity_group_disagg[-1]["entity"].split("-")[-1]
                    # print("entity (after aligning tag): {}".format(entity))

                entity_group_disagg += [entity]
                # Group the entities at the last entity
                if is_last_idx:
                    entity_groups += [self.group_sub_entities(entity_group_disagg)]
            # If the current entity is different from the previous entity, aggregate the disaggregated entity group
            else:
                entity_groups += [self.group_sub_entities(entity_group_disagg)]
                entity_group_disagg = [entity]
                # If it's the last entity, add it to the entity groups
                if is_last_idx:
                    entity_groups += [self.group_sub_entities(entity_group_disagg)]

        return entity_groups

    def _prepare_inputs(self, inputs: Dict[str, Union[torch.Tensor, Any]]) -> Dict[str, Union[torch.Tensor, Any]]:
        """
        Prepare :obj:`inputs` before feeding them to the model, converting them to tensors if they are not already and
        handling potential state.
        """
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.training_args.device)

        if self.training_args.past_index >= 0 and self._past is not None:
            inputs["mems"] = self._past
        return inputs

    def _wrap_model(self, model):
        if is_sagemaker_mp_enabled():
            # Wrapping the base model twice in a DistributedModel will raise an error.
            if isinstance(self.model, smp.model.DistributedModel):
                return self.model
            return smp.DistributedModel(model, backward_passes_per_step=self.training_args.gradient_accumulation_steps)

        # already initialized its own DDP and AMP
        if self.deepspeed:
            return self.deepspeed

        # train/eval could be run multiple-times - if already wrapped, don't re-wrap it again
        if unwrap_model(model) is not model:
            return model

        # Multi-gpu training (should be after apex fp16 initialization)
        if self.training_args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Note: in torch.distributed mode, there's no point in wrapping the model
        # inside a DistributedDataParallel as we'll be under `no_grad` anyways.
        return model

    def relation_bow(self):
        """
        Convert all patterns in each relation type to bag of words.
        """
        pattern_bow = defaultdict(list)
        pattern_bow_freq = defaultdict(dict)
        for re, pattern_list in self.patterns.items():
            tokens = []
            for pat in pattern_list:
                tokens.extend(pat.split())
            pattern_bow[re] = list(set(tokens))  # remove repetition
            pattern_bow_freq[re] = dict(Counter(tokens))  # count token appreance times

        return pattern_bow, pattern_bow_freq

    def entity_pattern_bow(self):
        """
        Convert all patterns in each entity type to bag of words.
        """
        entity_bow = defaultdict(list)
        entity_bow_freq = defaultdict(dict)
        for ent, pattern_list in self.ent_patterns.items():
            tokens = []
            for pat in pattern_list:
                tokens.extend(pat.split())
            entity_bow[ent] = list(set(tokens))  # remove repetition
            entity_bow_freq[ent] = dict(Counter(tokens))  # count token appreance times

        return entity_bow, entity_bow_freq

    def neg_sample_map(self, example):
        """
        Negative sampling for the raw dataset before feed into the Trainer.
        """
        tags = []
        query = []
        instance = []
        for i, tokens in enumerate(example["tokens"]):
            seq_len = len(tokens)
            sentID = example["sentID"][i]
            insID = example["instanceID"][i]
            ner_tags = ["O"] * seq_len
            usable_ids = list(set(range(seq_len)) - self.sent2query[sentID])

            if not usable_ids:
                query_id = -100
            else:
                query_id = random.choice(usable_ids)  # randomly chose the non-queryid as neg sampes queryid
            instanceID = -insID - 1

            # record
            tags.append(ner_tags)
            query.append(query_id)
            instance.append(instanceID)

        example["instanceID"] = instance
        example["query_ids"] = query
        example["ner_tags"] = tags
        return example

    def negative_sampling(self, train):
        """
        Return the merged train and sampled negative instances for training.
        """
        # get query_id list for each sentID
        self.sent2query = defaultdict(set)
        for i, instance in enumerate(train):
            sentID, query_id = instance["sentID"], instance["query_ids"]
            self.sent2query[sentID].add(query_id)

        neg_data = train.map(
            self.neg_sample_map,
            batched=True,
            num_proc=self.data_args.preprocessing_num_workers,
            load_from_cache_file=not self.data_args.overwrite_cache,
            features=train.features,
        )

        neg_usedata = neg_data.filter(
            lambda exp: exp["query_ids"] != -100,
            num_proc=self.data_args.preprocessing_num_workers,
            load_from_cache_file=not self.data_args.overwrite_cache,
        )

        # sample 10% of negative sequences
        sample_idx = np.random.choice(
            len(neg_usedata), size=int(self.data_args.sample_rate * len(neg_usedata)), replace=False
        )

        if len(sample_idx) == 0:  # empty idx
            neg_data_final = neg_usedata
        else:
            neg_data_final = neg_usedata.select(sample_idx)

        # concat both train and negative instances
        final_train = concatenate_datasets([train, neg_data_final], axis=0)
        final_train = final_train.sort(
            column="sentID",
            load_from_cache_file=not self.data_args.overwrite_cache,
        )

        return final_train

    def instance_select(self, model):
        """
        Selecting training instance with higher confidence score w.r.t the position attention distribution.
        """
        print("selecting instances ...")
        # tokenize the raw text to feed into model
        tokenized_dataset = self.train_dataset.map(
            self.tokenize_and_align_labels,
            batched=True,
            num_proc=self.data_args.preprocessing_num_workers,
            load_from_cache_file=not self.data_args.overwrite_cache,
            remove_columns=["tokens", "ner_tags", "instanceID", "sentID"],
        )
        dataloader = self.trainer.get_eval_dataloader(tokenized_dataset)
        # convert each input into tensor and handle GPU stuff
        model = self._wrap_model(model)
        model = model.to(self.training_args.device)

        model.eval()
        if is_torch_tpu_available():
            dataloader = pl.ParallelLoader(dataloader, [self.training_args.device]).per_device_loader(
                self.training_args.device
            )

        matched_idx = []
        kl_set = []
        for step, inputs in enumerate(dataloader):
            inputs = self._prepare_inputs(inputs)  # {"input_ids", "query_ids", "target_att", ...}
            target_att = inputs["target_att"]  # target attention (B X T)
            with torch.no_grad():
                outputs = model(**inputs)
                pa = outputs["position_attentions"]  # position attention (B X T)
                loss = self.kl_loss(pa.log(), target_att).sum(dim=1)  # sum on the T dimension to get (B)
                kl_set.extend(loss.detach().cpu().tolist())
                confidence = 1 / (1 + loss)  # the final confidence score (B)
                batch_idx = (
                    torch.where(confidence > self.threshold)[0] + step * self.training_args.eval_batch_size
                ).tolist()
                matched_idx.extend(batch_idx)

        # exchange matched_idx to instanceIDs
        insIDs = self.train_dataset["instanceID"]
        self.matched_IDs = set([insIDs[idx] for idx in matched_idx])
        # statistic IoU of new selected idx with initial and previous set
        self.init_intersect = len(self.matched_IDs & self.init_idx)  # intersection
        self.IoU_init = self.init_intersect / len(self.matched_IDs | self.init_idx)
        self.prev_intersect = len(self.matched_IDs & self.trust_idx)  # intersection
        self.IoU_prev = self.prev_intersect / len(self.matched_IDs | self.trust_idx)
        # merge trust index set with the new select index set to update self.trust_idx
        self.trust_idx = self.matched_IDs | self.trust_idx

        if matched_idx:
            res_record = []
            re_count, ent_count = 0, 0
            pos_count, neg_count = 0, 0
            # save predicted kl loss
            for idx in matched_idx:
                loss = kl_set[idx]
                insID = insIDs[idx]
                if insID in self.pos_instances:
                    IS_POS = "positive"
                    pos_count += 1
                else:
                    IS_POS = "negative"
                    neg_count += 1
                if insID in self.re_instances:
                    IS_RE = "relation"
                    re_count += 1
                else:
                    IS_RE = "entity"
                    ent_count += 1

                res_record.append({"instanceID": insID, "KLloss": loss, "pos/neg": IS_POS, "re/ent": IS_RE})

            res_record.append(
                {
                    "pos_rate": pos_count / len(matched_idx),
                    "neg_rate": neg_count / len(matched_idx),
                    "re_rate": re_count / len(matched_idx),
                    "ent_rate": ent_count / len(matched_idx),
                }
            )

            # save predicted confidence score
            with open(
                os.path.join(self.training_args.output_dir, "KLloss_{}.json".format(self.epoch)), "w", encoding="utf-8"
            ) as f:
                for value in res_record:
                    f.write(json.dumps(value))
                    f.write("\n")
                f.close()

        # matched training and pattern instances
        matched_dataset = self.train_dataset.filter(
            lambda exp: exp["instanceID"] in self.trust_idx,
            num_proc=self.data_args.preprocessing_num_workers,
            load_from_cache_file=not self.data_args.overwrite_cache,
        )
        matched_patterns = self.raw_patterns.filter(
            lambda exp: exp["instanceID"] in self.trust_idx,
            num_proc=self.data_args.preprocessing_num_workers,
            load_from_cache_file=not self.data_args.overwrite_cache,
        )
        if self.max_ent_range is not None:
            matched_ent_patterns = self.raw_ent_patterns.filter(
                lambda exp: exp["instanceID"] in self.trust_idx,
                num_proc=self.data_args.preprocessing_num_workers,
                load_from_cache_file=not self.data_args.overwrite_cache,
            )
        else:
            matched_ent_patterns = None

        return matched_dataset, matched_patterns, matched_ent_patterns

    def ent_pattern_extract(self, examples):
        patterns = []
        res = {}
        for i, IOB_tags in enumerate(examples["ner_tags"]):
            insID = examples["instanceID"][i]
            sentID = examples["sentID"][i]
            if insID >= 0:  # not "O" sequences
                query_id = examples["query_ids"][i]  # before tokenization
                query_tag = IOB_tags[query_id][2:]
                query_end = examples["query_ids"][i]  # the temp query entity end index
                tokens = examples["tokens"][i]

                ent_prepat_toks = []
                ent_postpat_toks = []
                for index, IOB_tag in enumerate(IOB_tags):
                    if index < query_id and query_id - index <= self.max_ent_range:
                        ent_prepat_toks.append(tokens[index])
                    elif index > query_id:
                        if IOB_tag != "O" and IOB_tag[:2] == "I-" and index == query_end + 1:
                            query_end += 1  # update the query end as the current index
                        elif index - query_end <= self.max_ent_range:
                            ent_postpat_toks.append(tokens[index])

                ent_pattern = [
                    {"entity": IOB_tags[query_id][2:], "pattern": " ".join(pat_toks)}
                    for pat_toks in [ent_prepat_toks, ent_postpat_toks]
                    if pat_toks
                ]
                patterns.append(ent_pattern)

        res["ent_patterns"] = patterns

        return res

    def pattern_extract(self, examples):
        """
        Inputs:
            examples (object: huggingface dataset): any dataset in our setting.
        Outputs:
            raw_patterns (object: huggingface dataset): list of dictionary of patterns in all instances.

        For each query-level instance, we extract all patterns (string of tokens)
        that are between the query entity and each target entity.
        """
        patterns = []
        res = {}

        for i, IOB_tags in enumerate(examples["ner_tags"]):
            insID = examples["instanceID"][i]
            if insID >= 0:  # not "O" sequences
                query_id = examples["query_ids"][i]  # before tokenization
                query_end = examples["query_ids"][i]  # the temp query entity end index
                tokens = examples["tokens"][i]
                instance_dict = defaultdict(defaultdict(list).copy)

                for index, IOB_tag in enumerate(IOB_tags):
                    if IOB_tag != "O":
                        prefix = IOB_tag[:2]
                        # if the cur_index is smaller than the query_id, we need to put the tokens between them into dict
                        # we also need to dynamically pop (delete) the I-RE from the pattern parts
                        if index < query_id:
                            if prefix == "B-":  # find a start of target entity
                                instance_dict[index]["patterns"] = tokens[
                                    index + 1 : query_id
                                ]  # store the tokens in between
                                instance_dict[index]["target_id"] = index  # store the target id
                                tmp_index = index  # record the current index of a target entity
                                target_start = index  # record the starting index of a target entity
                            elif index == tmp_index + 1:  # still the same RE
                                instance_dict[target_start]["patterns"].pop(0)  # delete the RE tail from the pattern
                                tmp_index += 1
                        # if the cur_index is larger than the query_id, we only need to record the query ending index
                        elif index > query_id:
                            if prefix == "I-" and index == query_end + 1:
                                query_end += 1  # update the query end as the current index
                            elif prefix == "B-":  # starting index of target entity
                                instance_dict[index]["patterns"] = tokens[query_end + 1 : index]
                                instance_dict[index]["target_id"] = index
                # Join the pattern tokens into strings and save
                # Format [{re1, pattern1, target_id1}, {re2, pattern2, target_id2}, ...]
                re_pattern_pair = [
                    {"relation": IOB_tags[k][2:], "pattern": " ".join(v["patterns"]), "target_id": v["target_id"]}
                    if "patterns" in v
                    else {"relation": IOB_tags[k][2:], "pattern": "", "target_id": 0}
                    for k, v in instance_dict.items()
                ]
                patterns.append(re_pattern_pair)

        res["re_patterns"] = patterns

        return res

    def handle_ent_patterns(self, patterns, max_size=20):
        """
        This function Counter each pattern's frequency, and choose the topk patterns for each label as final pattern set.
        Outputs:
            ent_pattern_set (Dict[List]): key is each entity type, value is a list of ordered (in terms of occurance) patterns (no repetition).
            ent_pattern2id (Dict[List]): key is each pattern, value is a list of all corresponding instanceIDs.
            res_ent_pat (Dictp[List]): key is each relation type, value is a list of top10% frequent patterns (no repetition).
        """
        ent_pattern_set = defaultdict(list)
        ent_pattern2id = defaultdict(defaultdict(list).copy)
        for instance in patterns:
            for pat_dict in instance["ent_patterns"]:  # list of {ent, pattern} dicts
                ent, pt = pat_dict["entity"], pat_dict["pattern"]
                if pt:
                    ent_pattern_set[ent].append(pt)
                    ent_pattern2id[ent][pt].append(instance["instanceID"])

        # most common (for building the initial pattern set M')
        res_ent_pat = defaultdict(list)
        ent_pattern_count = defaultdict(dict)
        for ent in ent_pattern_set:
            pat = ent_pattern_set[ent]
            pos_size = min(int(0.1 * len(set(pat))), max_size)
            count = Counter(pat)
            ent_pattern_count[ent].update(count)
            common_set = count.most_common(pos_size) if pos_size != 0 else count.most_common()
            res_ent_pat[ent] = [x[0] for x in common_set]

        # remove repetition (for building new patterns)
        for ent, pt_list in ent_pattern_set.items():
            ent_pattern_set[ent] = [x[0] for x in Counter(pt_list).most_common()]

        return ent_pattern_set, res_ent_pat, ent_pattern2id, ent_pattern_count

    def handle_patterns(self, patterns, max_size=20):
        """
        This function Counter each pattern's frequency, and choose the topk patterns for each label as final pattern set.
        Outputs:
            pattern_set (Dict[List]): key is each relation type, value is a list of ordered (in terms of occurance) patterns (no repetition).
            pattern2id (Dict[List]): key is each pattern, value is a list of all corresponding instanceIDs.
            res_pat (Dictp[List]): key is each relation type, value is a list of top10% frequent patterns (no repetition).
        """
        pattern_set = defaultdict(list)
        pattern2id = defaultdict(defaultdict(list).copy)
        pattern2target = defaultdict(defaultdict(tuple).copy)  # (query_id, target_id, sent_id)
        for instance in patterns:
            insID, sentID, query_id = instance["instanceID"], instance["sentID"], instance["query_ids"]
            for pat_dict in instance["re_patterns"]:  # list of {re, pattern, target_id} dicts
                re, pt, target_id = pat_dict["relation"], pat_dict["pattern"], pat_dict["target_id"]
                if pt:
                    pattern_set[re].append(pt)
                    pattern2id[re][pt].append(insID)
                    pattern2target[re][pt] = (query_id, target_id, sentID)

        # most common (for building the initial pattern set M)
        res_pat = defaultdict(list)
        pattern_count = defaultdict(dict)
        for re in pattern_set:
            pat = pattern_set[re]
            pos_size = min(int(0.1 * len(set(pat))), max_size)
            count = Counter(pat)
            pattern_count[re].update(count)
            common_set = count.most_common(pos_size) if pos_size != 0 else count.most_common()
            res_pat[re] = [x[0] for x in common_set]

        # remove repetition (for building new patterns)
        for re, pt_list in pattern_set.items():
            pattern_set[re] = [x[0] for x in Counter(pt_list).most_common()]

        return pattern_set, res_pat, pattern2id, pattern2target, pattern_count

    def pattern_matcher(self, dataset):
        """
        We extract all the instanceIDs from the pattern index dict, and select the matched dataset.
        """
        matched_idx = []
        sent2target = defaultdict(list)
        # Match those with high-quality relations instances
        for re, pat_list in self.patterns.items():
            for pat in pat_list:
                query_id, target_id, sentID = self.pattern2target[re][pat]
                sent2target[sentID].append(query_id)  # each sentID corresponds a list of query_ids.
                sent2target[sentID].append(target_id)  # each sentID corresponds a list of target_ids.
                matched_idx.extend(self.pattern2id[re][pat])  # merge all the instanceIDs.

        print("# of matched idx (re patterns): {}".format(len(set(matched_idx))))

        # Match those instances with its query entity as the tail entity of selected patterns
        for instance in dataset:
            sentID, queryID, insID = instance["sentID"], instance["query_ids"], instance["instanceID"]
            if sentID in sent2target and queryID in sent2target[sentID]:  # same sentence tail entity
                if insID in self.ent_instances:
                    matched_idx.append(insID)

        print("# of matched idx (after adding tail entities): {}".format(len(set(matched_idx))))

        if self.max_ent_range is not None:
            # Match those with high-quality entity instances
            for ent, pat_list in self.ent_patterns.items():
                for pat in pat_list:
                    matched_idx.extend(self.ent_pattern2id[ent][pat])  # merge all the instanceIDs.

        matched_idx = set(matched_idx)
        print("# of matched idx (final): {}".format(len(matched_idx)))

        matched_dataset = dataset.filter(
            lambda exp: exp["instanceID"] in matched_idx,
            num_proc=self.data_args.preprocessing_num_workers,
            load_from_cache_file=not self.data_args.overwrite_cache,
        )
        # unmatched_dataset = dataset.filter(
        #     lambda exp: exp['instanceID'] not in matched_idx,
        #     num_proc=self.data_args.preprocessing_num_workers,
        #     load_from_cache_file=not self.data_args.overwrite_cache,
        # )

        return matched_dataset, matched_idx

    def data_redistribute(self, dataset, initial=False):
        """
        Use the pattern set to select positive and negative dataset.
        Modify the NER and RE labels for both parts.
        The positive dataset are used first for negative sampling and then tokenization, and finally feed into the Trainer.
        """
        if initial:
            matched_dataset, self.init_idx = self.pattern_matcher(dataset)
            self.trust_idx = self.init_idx  # temp index set
            if self.use_negative_sampling:
                matched_dataset = self.negative_sampling(matched_dataset)
        else:
            sent2target = defaultdict(list)
            # Match those with high-quality relations instances
            for re, pat_list in self.new_patterns.items():
                for pat in pat_list:
                    query_id, target_id, sentID = self.pattern2target[re][pat]
                    sent2target[sentID].append(query_id)  # each sentID corresponds a list of query_ids.
                    sent2target[sentID].append(target_id)  # each sentID corresponds a list of target_ids.
            # Match those instances with its query entity as the tail entity of selected patterns
            considered_idx = self.ent_instances - set(dataset["instanceID"])
            considered_data = self.train_dataset.filter(
                lambda exp: exp["instanceID"] in considered_idx,
                num_proc=self.data_args.preprocessing_num_workers,
                load_from_cache_file=not self.data_args.overwrite_cache,
            )
            print("getting tailed entites ...")
            tail_insIDs = []
            for instance in considered_data:
                sentID, queryID, insID = instance["sentID"], instance["query_ids"], instance["instanceID"]
                # Same sentence with tail entity and query entity in those entity instances
                if sentID in sent2target and queryID in sent2target[sentID]:
                    tail_insIDs.append(insID)

            ultimate_IDs = set(tail_insIDs + dataset["instanceID"])
            matched_dataset = self.train_dataset.filter(
                lambda exp: exp["instanceID"] in ultimate_IDs,
                num_proc=self.data_args.preprocessing_num_workers,
                load_from_cache_file=not self.data_args.overwrite_cache,
            )
            # We directly use all dataset for next training
            if self.use_negative_sampling:
                matched_dataset = self.negative_sampling(matched_dataset)

            print("tail size: {}, data size (merge tail): {}".format(len(tail_insIDs), len(dataset)))

        # tokenization
        matched_dataset = matched_dataset.map(
            self.tokenize_and_align_labels,
            batched=True,
            num_proc=self.data_args.preprocessing_num_workers,
            load_from_cache_file=not self.data_args.overwrite_cache,
        )

        return matched_dataset

    def bootstrap(self, model, epoch):
        """
        The whole architecture of boostrap training procedure.
        """
        self.epoch = epoch
        # Select trustable instances based on current model
        trust_dataset, trust_patterns, trust_ent_patterns = self.instance_select(model)

        # Call handling function to extract unique patterns as our pattern set
        self.new_patterns, _, _, _, _ = self.handle_patterns(trust_patterns)

        # Union patterns (max N new patterns for each relation type)
        added_patterns = defaultdict(list)
        for re_label, pt_list in self.new_patterns.items():
            max_num_pt = self.max_new_patterns
            for pat in pt_list:
                if max_num_pt == 0:
                    break
                if pat not in self.patterns[re_label]:
                    self.patterns[re_label].append(pat)
                    max_num_pt = max_num_pt - 1
                    added_patterns[re_label].append(pat)

        self.pattern_bow, self.pattern_bow_freq = self.relation_bow()  # build bag of tokens

        if self.max_ent_range is not None:
            new_ent_patterns, _, _, _ = self.handle_ent_patterns(trust_ent_patterns)

            added_ent_patterns = defaultdict(list)
            for ent_label, pt_list in new_ent_patterns.items():
                max_num_pt = self.max_new_patterns
                for pat in pt_list:
                    if max_num_pt == 0:
                        break
                    if pat not in self.ent_patterns[ent_label]:
                        self.ent_patterns[ent_label].append(pat)
                        max_num_pt = max_num_pt - 1
                        added_ent_patterns[ent_label].append(pat)

            self.entity_bow, self.entity_bow_freq = self.entity_pattern_bow()  # build bag of tokens

        # Redistribute training dataset D based on M (negative sampling & tokenize for pos_dataset)
        pos_dataset = self.data_redistribute(trust_dataset)

        # Logging info to be saved
        logging_dict = {}
        logging_dict["epoch"] = epoch
        logging_dict["original data size"] = len(self.train_dataset)
        logging_dict["initial data size"] = len(self.init_idx)
        logging_dict["selected data size"] = len(self.matched_IDs)
        logging_dict["selected data size (after union)"] = len(trust_dataset)
        logging_dict["selected data size (after neg sample)"] = len(pos_dataset)
        logging_dict["intersection size (with initial data)"] = self.init_intersect
        logging_dict["IoU (with initial data)"] = self.IoU_init
        logging_dict["intersection size (with previous data)"] = self.prev_intersect
        logging_dict["IoU (with previous data)"] = self.IoU_prev
        logging_dict["pattern size"] = {k: len(v) for k, v in self.patterns.items()}
        logging_dict["new pattern size"] = {k: len(v) for k, v in added_patterns.items()}
        if self.max_ent_range is not None:
            logging_dict["entity pattern size"] = {k: len(v) for k, v in self.ent_patterns.items()}
            logging_dict["new entity pattern size"] = {k: len(v) for k, v in added_ent_patterns.items()}
        logging_dict["new patterns"] = {
            k: [{"pattern": v, "count": self.pattern_count[k][v]} for v in v_list]
            for k, v_list in added_patterns.items()
        }
        logging_dict["patterns"] = {
            k: [{"pattern": v, "count": self.pattern_count[k][v]} for v in v_list]
            for k, v_list in self.patterns.items()
        }
        logging_dict["pattern_bag_of_words"] = self.pattern_bow_freq
        if self.max_ent_range is not None:
            logging_dict["new entity patterns"] = {
                k: [{"pattern": v, "count": self.ent_pattern_count[k][v]} for v in v_list]
                for k, v_list in added_ent_patterns.items()
            }
            logging_dict["entity patterns"] = {
                k: [{"pattern": v, "count": self.ent_pattern_count[k][v]} for v in v_list]
                for k, v_list in self.ent_patterns.items()
            }
        # logging_dict["pred entities"] = self.ent_pred
        # logging_dict["label entities"] = self.ent_label

        # Saving dataset size and other logging info
        with open(
            os.path.join(
                self.training_args.output_dir, "bootstrap_logging{}({}).json".format(epoch, len(pos_dataset))
            ),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(logging_dict, f, ensure_ascii=False, indent=2)
            f.close()

        # with open(os.path.join(self.training_args.output_dir, "pattern2id.json"), 'w', encoding='utf-8') as f:
        #     json.dump(self.pattern2id, f, ensure_ascii=False, indent=2)
        #     f.close()

        return pos_dataset

    def training(self):
        trainer = JointTrainer(
            model=self.model,
            args=self.training_args,
            data_collator=self.data_collator,
            train_dataset=self.init_dataset
            if self.training_args.do_train
            else self.eval_dataset,  # we feed the intial training data to the Trainer
            eval_dataset=self.eval_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
            bootstrap=self.bootstrap,
            original_dataset=self.original_dataset if self.training_args.do_train else self.eval_dataset,
            use_bootstrap=self.use_bootstrap,
            boot_start_epoch=self.boot_start_epoch,
        )
        self.trainer = trainer
        # Training
        if self.training_args.do_train:
            if self.last_checkpoint is not None:
                checkpoint = self.last_checkpoint
            elif os.path.isdir(self.model_args.model_name_or_path):
                checkpoint = self.model_args.model_name_or_path
            else:
                checkpoint = None
            # train_result = trainer.train(resume_from_checkpoint=checkpoint)
            # Workaround for GPT2 when labels are not present in config.json
            train_result = trainer.train()
            metrics = train_result.metrics
            trainer.save_model()  # Saves the tokenizer too for easy upload

            max_train_samples = (
                self.data_args.max_train_samples
                if self.data_args.max_train_samples is not None
                else len(self.train_dataset)
            )
            metrics["train_samples"] = min(max_train_samples, len(self.train_dataset))

            trainer.log_metrics("train", metrics)
            trainer.save_metrics("train", metrics)
            trainer.save_state()

        # Evaluation
        if self.training_args.do_eval:
            logger.info("*** Evaluate ***")

            metrics = trainer.evaluate(ignore_keys=["position_attentions"])

            max_val_samples = (
                self.data_args.max_val_samples
                if self.data_args.max_val_samples is not None
                else len(self.eval_dataset)
            )
            metrics["eval_samples"] = min(max_val_samples, len(self.eval_dataset))

            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)

        # Predict
        if self.training_args.do_predict:
            logger.info("*** Predict ***")

            predictions, labels, metrics = trainer.predict(self.test_dataset, ignore_keys=["position_attentions"])
            predictions = np.argmax(predictions, axis=2)

            # Remove ignored index (special tokens)
            true_predictions = [
                [self.label_list[p] for (p, l) in zip(prediction, label) if l != -100]
                for prediction, label in zip(predictions, labels)
            ]

            trainer.log_metrics("test", metrics)
            trainer.save_metrics("test", metrics)

            # Save predictions
            output_test_predictions_file = os.path.join(self.training_args.output_dir, "test_predictions.txt")
            if trainer.is_world_process_zero():
                with open(output_test_predictions_file, "w") as writer:
                    for prediction in true_predictions:
                        writer.write(" ".join(prediction) + "\n")


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    training_pipeline = Training_Pipeline(model_args, data_args, training_args)
    training_pipeline.training()


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
