"""
Fine-tuning the library models for token classification.
"""
import csv
import json
import logging
import math
import os
import random
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from os.path import exists
from re import I
from types import MethodDescriptorType
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils import data
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from tqdm import tqdm

from gensim.models import KeyedVectors
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)
from transformers.file_utils import (
    WEIGHTS_NAME,
    is_apex_available,
    is_datasets_available,
    is_in_notebook,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
    is_torch_tpu_available,
    is_training_run_on_sagemaker,
)
from models.modeling_gpt2 import GPT2ForTokenClassification
from models.configuration_gpt2 import GPT2Config
from trainer import JointTrainer


if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl

from transformers.modeling_utils import PreTrainedModel, unwrap_model


logger = logging.getLogger(__name__)

import atexit

import line_profiler


profile = line_profiler.LineProfiler()
atexit.register(profile.print_stats)


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
    n_embd: Optional[int] = field(default=300, metadata={"help": "hidden or embedding dimensionality "})
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
    wordvec_mode: Optional[str] = field(
        default="word2vec", metadata={"help": "Embedding used for lstm case. [word2vec, glove, None]"}
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
    Inputs: ['input_ids', 'labels', 'query_ids', 'sentID', 'instanceID', 'target_attention']
    Outputs: ['input_ids', 'input_embeds', 'labels', 'query_ids', 'target_att']
    """

    padding: bool = True
    max_length: Optional[int] = None
    embedding: object = None  # (V X E)
    pad_to_multiple_of: Optional[int] = None
    pad_token_id: int = 0
    pad_label_id: int = 0
    attention_pad_token_id: float = 0.0

    def __call__(self, features):
        # features: list of tuples
        input_ids = [feature["input_ids"] for feature in features]  # list of token_ids (T: int)
        labels = [feature["labels"] for feature in features]  # list of label_ids (T: int)
        query_ids = [feature["query_ids"] for feature in features]  # list of label_ids [int]
        target_att = [feature["target_att"] for feature in features]  # list of target_att (T: float)

        # input_ids, label padding (list of T -> B X T)
        seq_len = self.max_length or max(len(s) for s in input_ids)

        padded_ids = [input_id + [self.pad_token_id] * (seq_len - len(input_id)) for input_id in input_ids]
        padded_labels = [label + [self.pad_label_id] * (seq_len - len(label)) for label in labels]
        padded_target_att = [att + [self.attention_pad_token_id] * (seq_len - len(att)) for att in target_att]

        # padded_ids (B X T) -> embedding (B X T X E)
        padded_embed = [[self.embedding[idx] for idx in instance] for instance in padded_ids]

        return {
            "input_ids": torch.tensor(padded_ids, dtype=torch.int64),
            "input_embeds": torch.tensor(padded_embed, dtype=torch.float),
            "labels": torch.tensor(padded_labels, dtype=torch.int64),
            "query_ids": torch.tensor(query_ids, dtype=torch.int64),
            "target_att": torch.tensor(padded_target_att, dtype=torch.float),
        }


class NYT_dataset(Dataset):
    def __init__(self, dataset, word_dict):
        """
        dataset: list of dicts {'input_ids', 'labels', 'query_ids', 'sentID', 'instanceID', 'target_att'}
        """
        self.dataset = dataset
        # word embedding
        self.word_dict = word_dict

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        instance = self.dataset[index]
        input_ids = [self.word_dict._encode(token) for token in instance["tokens"]]  # list (T)
        labels = [self.word_dict.label2id[tag] for tag in instance["ner_tags"]]  # list (T)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "query_ids": instance["query_ids"],
            "sentID": instance["sentID"],
            "instanceID": instance["instanceID"],
            "target_att": instance["target_att"],
        }


class WordDictionary:
    """
    input: text data, wordvec_file (word2vec/glove)
    output: word2dix, weights
    """

    def __init__(self, args):
        # self.max_vocab_size = args.max_vocab_size
        # self.vector_size = args.vector_size # embedding dim
        # self.wordvec_mode = args.wordvec_mode # word2vec, glove, or dictionary idx
        self.max_vocab_size = 1000000
        self.vector_size = 300  # embedding dim
        self.wordvec_mode = args.wordvec_mode  # word2vec, glove, or dictionary idx
        self.PAD_TOKEN = "<PAD>"
        self.UNK_TOKEN = "<UNK>"
        self.SOS_TOKEN = "<SOS>"
        self.EOS_TOKEN = "<EOS>"

    def build_dictionary(self, data):
        self.vocab_words, self.word2idx, self.idx2word = self._build_vocab(data)
        self.label_list, self.label2id, self.id2label = self._build_labels(data)
        self.vocab_size = len(self.vocab_words)

        if self.wordvec_mode is None:
            self.embedding = np.random.randn(self.vocab_size, self.vector_size)  # V X H
        elif self.wordvec_mode == "word2vec":
            self.embedding = self._load_word2vec()
        elif self.wordvec_mode == "glove":
            self.embedding = self._load_glove()

    def _build_vocab(self, data):
        counter = Counter([token for instance in data for token in instance["tokens"]])
        if self.max_vocab_size:
            counter = {token: freq for token, freq in counter.most_common(self.max_vocab_size)}

        # build vocabulary
        vocab_words = [self.PAD_TOKEN, self.UNK_TOKEN, self.SOS_TOKEN, self.EOS_TOKEN]  # special tokens
        vocab_words += sorted(counter.keys())

        word2idx = {token: idx for idx, token in enumerate(vocab_words)}
        idx2word = {idx: token for idx, token in enumerate(vocab_words)}

        return vocab_words, word2idx, idx2word

    def _encode(self, word):
        try:
            return self.word2idx[word]
        except:
            return self.word2idx[self.UNK_TOKEN]

    def _decode(self, idx):
        try:
            return self.idx2word[idx]
        except:
            return self.UNK_TOKEN

    def _build_labels(self, data):
        counter = Counter([label for instance in data for label in instance["ner_tags"]])
        labels = sorted(counter.keys())

        label2id = {label: idx for idx, label in enumerate(labels)}
        id2label = {idx: label for idx, label in enumerate(labels)}

        return labels, label2id, id2label

    def _load_word2vec(self):
        wordvec_file = "GoogleNews-vectors-negative300.bin"
        if not exists(wordvec_file):
            raise Exception("You must download word vectors through `download_wordvec.py` first")
            # wget -c "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz"
            # gzip -d GoogleNews-vectors-negative300.bin.gz
        word2vec = KeyedVectors.load_word2vec_format(wordvec_file, binary=True)
        self.vector_size = word2vec.vector_size

        word_vectors = []
        for word in self.vocab_words:
            if word in word2vec.key_to_index:
                vector = word2vec[word]
            else:
                vector = np.random.normal(scale=0.2, size=self.vector_size)  # random vector
            word_vectors.append(vector)

        weight = np.stack(word_vectors)
        return weight

    def _load_glove(self):
        wordvec_file = "glove.840B.300d.txt"
        self.vector_size = 300
        if not exists(wordvec_file):
            raise Exception("You must download word vectors through `download_wordvec.py` first")

        glove_model = {}
        with open(wordvec_file) as file:
            for line in file:
                line_split = line.split()
                word = " ".join(line_split[: -self.vector_size])
                numbers = line_split[-self.vector_size :]
                glove_model[word] = numbers
        glove_vocab = glove_model.keys()

        word_vectors = []
        for word in self.vocab_words:

            if word in glove_vocab:
                vector = np.array(glove_model[word], dtype=float)
            else:
                vector = np.random.normal(scale=0.2, size=self.vector_size)  # random vector

            word_vectors.append(vector)

        weight = np.stack(word_vectors)
        return weight


class Position_Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_dim = 2 * config.n_embd
        self.hidden_dim = hidden_dim
        self.W_H = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_p = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_h = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, 1, bias=False)
        self.softmax = nn.Softmax(dim=1)

    # @profile
    def _pa(self, hidden_states, h_p, h_t):
        """
        h_p (B X H): batch hidden states for query tokens.
        h_t (B X H): batch hidden states for target tokens.
        hidden_states (B X T X H): hidden states of transformer output for position attention.
        """
        tail = (self.W_p(h_p) + self.W_h(h_t)).unsqueeze(1).expand_as(hidden_states)  # B X H -> B X T X H
        s_t = self.v(torch.tanh(self.W_H(hidden_states) + tail))  # B X T X 1
        a_t = self.softmax(s_t)  # B X T X 1
        c_t = torch.sum(a_t.expand_as(hidden_states) * hidden_states, dim=1)  # B X H
        return a_t.squeeze(-1), c_t

    # @profile
    def forward(self, hidden_states, query_ids):
        all_at = []
        all_ut = []
        seq_len = hidden_states.size(1)
        if isinstance(query_ids, int):
            h_p = hidden_states[:, query_ids, :]  # B X H
        else:
            h_p = torch.gather(
                hidden_states, dim=1, index=query_ids.unsqueeze(-1).expand(-1, -1, self.hidden_dim)
            ).squeeze(
                1
            )  # B X H

        for target_id in range(seq_len):
            h_t = hidden_states[:, target_id, :]  # B X H
            a_t, c_t = self._pa(hidden_states, h_p, h_t)  # B X T, B X H
            u_t = torch.cat((h_t, c_t), dim=1)  # B X 2H
            all_at.append(a_t)
            all_ut.append(u_t)  # a list of T times (B X 2H)

        all_at = torch.stack(all_at).permute(1, 0, 2)  # B X T X T
        # at_pool, at_index = torch.max(all_at, dim=1) # maxpool over target token dimension (B X T)
        at_pool = torch.mean(all_at, dim=1)  # avgpool over target token dimension (B X T)

        # normalize attention
        normalized_at = at_pool / at_pool.sum(dim=1).unsqueeze(1).expand_as(at_pool)
        normalized_at[normalized_at != normalized_at] = 0  # handle Nan by zero divide

        all_ut = torch.stack(all_ut).permute(1, 0, 2)  # B X T X 2H

        return normalized_at, all_ut


class JointBiLSTM(nn.Module):
    def __init__(self, config, num_labels):
        super().__init__()
        self.embed_dim = config.n_embd
        self.num_labels = num_labels
        self.rnn = nn.LSTM(
            self.embed_dim,
            self.embed_dim,
            num_layers=3,
            batch_first=True,
            dropout=0.1,
            bidirectional=True,
        )
        self.position_attention = Position_Attention(config)  # position attention
        self.classifier = nn.Linear(4 * config.n_embd, self.num_labels, bias=False)  # z_t = W_u * u_t
        self.crf = CRF(self.num_labels)
        # KL divergence loss
        self.kl_loss = nn.KLDivLoss(reduction="sum")
        self.beta = config.beta

    def forward(
        self,
        input_ids=None,  # input token2id (B X T)
        input_embeds=None,  # embeddings of input tokens (B X T X E)
        labels=None,  # input tag index (B X T)
        return_dict=None,
        target_att=None,  # corresponding attention guidance (B X T)
        query_ids=None,  # query index (B, 1)
    ):
        hidden_states, _ = self.rnn(input_embeds)  # B X T X 2*E, 2*num_layers X B X E
        # Training
        if query_ids is not None:
            a_t, query_hiddens = self.position_attention(hidden_states, query_ids)  # a_t: (B X T), u_t: (B X T X 4H)
            # print("a_t: {}, u_t: {}".format(a_t, query_hiddens))
            logits = self.classifier(query_hiddens)  # (B X T X 4H) -> (B X T X num_labels)

            loss = None
            if labels is not None:
                # loss (crf_loss + attention_loss)
                crf_loss = self.crf(logits, labels)
                kl_loss = self.kl_loss(a_t.log(), target_att)
                if self.beta == 0:
                    loss = crf_loss  # loss of (batch_size)
                else:
                    loss = crf_loss + self.beta * kl_loss
                    # print('crf_loss: {}, kl_loss: {}, logic_loss: {}'.format(crf_loss, kl_loss, logic_loss))

            # (B X T X num_labels) -> (B X T)
            viterbi_labels = self.crf.viterbi_decode(logits)
            viterbi_labels = torch.tensor(viterbi_labels, device=logits.device)
            # (B X T) -> (B X T X num_labels)
            logits = self._viterbi_to_logits(viterbi_labels)

            if not return_dict:
                output = (logits,) + (hidden_states,) + (a_t,)
                return ((loss,) + output) if loss is not None else output

        # Evaluating & Predicting
        else:
            all_logits = []
            # attention_stack = []
            for query_id in range(input_ids.size(-1)):  # inference for each query_id in the input sentence
                a_t, query_hiddens = self.position_attention(
                    hidden_states, query_id
                )  # a_t: (B X T), u_t: (B X T X 4H)
                # attention_stack.append(a_t)
                logits = self.classifier(query_hiddens)  # (B X T X 4H) -> (B X T X num_labels)
                # (B X T X num_labels) -> (B X T)
                viterbi_labels = self.crf.viterbi_decode(logits)
                viterbi_labels = torch.tensor(viterbi_labels, device=logits.device)
                # (B X T) -> (B X T X num_labels)
                logits = self._viterbi_to_logits(viterbi_labels)
                all_logits.append(logits)

            all_logits = torch.stack(all_logits)  # T X B X T X num_labels
            loss = (torch.tensor(0, dtype=torch.float32, device=logits.device),)

            if not return_dict:
                output = (loss, all_logits, hidden_states)
                return output

    def _viterbi_to_logits(self, viterbi_labels):
        """
        Parameters
            viterbi_labels: generated by CRF/PartialCRF.viterbi_decode of shape (B X T)
        Returns
            logits: (B X T X num_labels)
        """
        # (B X T) -> (B X T X 1)
        tags_ = torch.unsqueeze(viterbi_labels, 2)
        # (B X T X num_labels)
        # logits = torch.zeros(tags_.size(0), tags_.size(1), self.num_labels, dtype=torch.uint8, device=tags_.device)
        logits = torch.empty(
            tags_.size(0), tags_.size(1), self.num_labels, dtype=torch.int8, device=tags_.device
        ).fill_(-1)
        # Write 1 to places indexed by tags_
        logits.scatter_(2, tags_, 10)
        return logits


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

        # Setup logging
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            handlers=[logging.StreamHandler(sys.stdout)],
        )

        # Log on each process the small summary:
        logger.warning(
            f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
            + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
        )
        self.deepspeed = None

        if training_args.do_train:
            # load dataset
            train_dataset = []
            f = open(data_args.train_file, "r")
            for line in f.readlines():
                train_dataset.append(json.loads(line))

            # filter long sentence
            self.train_dataset = [instance for instance in train_dataset if len(instance["tokens"]) <= 512]

            # get the vocabulary and embedding [using the train_dataset]
            self.word_dict = WordDictionary(data_args)
            self.word_dict.build_dictionary(self.train_dataset)
            self.label2id = self.word_dict.label2id
            self.id2label = self.word_dict.id2label
            self.label_list = self.word_dict.label_list
            self.num_labels = len(self.label_list)
            self.embedding = self.word_dict.embedding

            if data_args.max_train_samples is not None:
                num_max_sample = min(len(self.train_dataset), data_args.max_train_samples)
                self.train_dataset = random.sample(self.train_dataset, num_max_sample)

        self.model = JointBiLSTM(model_args, self.num_labels)
        # torch.save(self.tokenizer, 'examples/tok_cls_result/tokenizer.pt')
        # Preprocessing the dataset
        # Padding strategy
        self.padding = "max_length" if data_args.pad_to_max_length else False

        # Data collator
        self.data_collator = DataCollatorForJointClassification(
            embedding=self.embedding,
            pad_to_multiple_of=8 if training_args.fp16 else None,
            pad_token_id=self.word_dict._encode("<PAD>"),
            pad_label_id=self.label2id["O"],
        )

        if training_args.do_train:
            # Get the initial pattern set M from D with E
            self.raw_patterns = self.pattern_extract(self.train_dataset)  # list of dicts
            _, self.patterns, self.pattern2id, self.pattern2target, self.pattern_count = self.handle_patterns(
                self.raw_patterns
            )
            # build bag of tokens from pattern set
            self.pattern_bow, self.pattern_bow_freq = self.relation_bow()
            self.get_re_ent_instance()  # statistics instance info

            if self.use_bootstrap:
                # before next training, redistribute training dataset D based on M (and negative sampling)
                init_dataset = self.data_redistribute(self.train_dataset, initial=True)
            else:
                if self.use_negative_sampling:
                    init_dataset = self.negative_sampling(self.train_dataset)
                else:
                    init_dataset = self.train_dataset
                init_dataset = self.gen_target_att(init_dataset)

            self.init_dataset = NYT_dataset(init_dataset, self.word_dict)
            # print("init dataset[0] (after word2id): ", self.init_dataset[0])
            original_dataset = self.gen_target_att(self.train_dataset)
            # print("get original train data info:")
            self.original_dataset = NYT_dataset(original_dataset, self.word_dict)

        if training_args.do_eval:
            # load dataset
            eval_dataset = []
            f = open(data_args.validation_file, "r")
            for line in f.readlines():
                eval_dataset.append(json.loads(line))

            # filter long sentence
            eval_dataset = [instance for instance in eval_dataset if len(instance["tokens"]) <= 512]
            if data_args.max_val_samples is not None:
                num_max_sample = min(len(eval_dataset), data_args.max_val_samples)
                eval_dataset = random.sample(eval_dataset, num_max_sample)

            eval_dataset = self.gen_target_att(eval_dataset)
            # print("get eval data info:")
            self.eval_dataset = NYT_dataset(eval_dataset, self.word_dict)
            # attention_mask, input_ids, instanceID, labels, ner_tags,
            # query_ids, special_tokens_mask, sentID, target_att, tokens
            # self.eval_dataset.to_json("examples/tok_cls_result/attention/test_ent_data.json")

    def get_re_ent_instance(self):
        self.all_train_sents = set([ins["sentID"] for ins in self.train_dataset])
        self.all_train_instances = set([ins["instanceID"] for ins in self.train_dataset])
        self.pos_sents = set(
            [ins["sentID"] for ins in self.train_dataset if any("/" in tag for tag in ins["ner_tags"])]
        )
        self.re_instances = set(
            [ins["instanceID"] for ins in self.train_dataset if any("/" in tag for tag in ins["ner_tags"])]
        )
        self.pos_instances = set([ins["instanceID"] for ins in self.train_dataset if ins["sentID"] in self.pos_sents])
        self.neg_sents = self.all_train_sents - self.pos_sents
        self.ent_instances = self.all_train_instances - self.re_instances
        self.neg_instances = self.all_train_instances - self.pos_instances
        print(
            "train data: # sents {}, # pos sents {}, # pos ins {}, # re ins {}, # neg sents {}, # neg ins {}, # ent ins {}".format(
                len(self.all_train_sents),
                len(self.pos_sents),
                len(self.pos_instances),
                len(self.re_instances),
                len(self.neg_sents),
                len(self.neg_instances),
                len(self.ent_instances),
            )
        )

    # Tokenize all texts and align the labels with them.
    def gen_target_att(self, dataset):
        gen_data = deepcopy(dataset)
        if self.training_args.do_train:
            for instance in gen_data:
                IOB_tags = instance["ner_tags"]
                if instance["instanceID"] < 0:  # all "O" sequence has a negative instanceID
                    atts = [0] * len(IOB_tags)  # target attention are all zeros
                else:
                    tokens = instance["tokens"]
                    # first record all the relations in this instance
                    relation_set = list(set([tag[2:] for tag in IOB_tags if "B-/" in tag]))
                    # record the entity bag of words
                    all_dist = []
                    for relation in relation_set:
                        b_m = [
                            int(tok in self.pattern_bow[relation] or (tag != "O" and "/" not in tag))
                            for tok, tag in zip(tokens, IOB_tags)
                        ]
                        all_dist.append(b_m)

                    if not relation_set:  # if no relations in this instance, we only attend to the query entity
                        freq = np.array([int(tag != "O" and "/" not in tag) for tag in IOB_tags])
                    else:  # otherwise, we maxpool over all relation token distribution
                        freq = np.array(all_dist).max(axis=0)  # maxpooling (T)

                    # normalize and avoid trivial division
                    atts = list(freq / freq.sum() if freq.sum() else freq)  # normalize by sum (T)

                instance["target_att"] = atts
                instance["query_ids"] = [instance["query_ids"]]  # int -> list
        else:
            for instance in gen_data:
                instance["query_ids"] = [instance["query_ids"]]  # int -> list
                instance["target_att"] = []

        return gen_data

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

        sentIDs = [ins["sentID"] for ins in dataset]
        queryIDs = [ins["query_ids"] for ins in dataset]
        labels = [ins["labels"] for ins in dataset]

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
            unique_pair = []
            id_set = set()
            for Id, tag in zip(sentIDs, labels):
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
        for i, instance in enumerate(self.eval_dataset):
            sent_id = instance["sentID"]
            if sent_id not in sent_id_pool:
                sent_id_pool.add(sent_id)
            else:
                remove_idx.append(i)

        grouped_preds = [preds for i, preds in enumerate(grouped_preds) if i not in remove_idx]  # N' X T X T'
        grouped_labels = self.preds_to_grouped_entity(preds=labels, is_label=True)  # N X T'

        true_predictions = self.extract_triplets(grouped_preds, is_label=False)  # N X T' (quadratic dict)
        true_labels = self.extract_triplets(grouped_labels, is_label=True)  # N X T' (quadratic dict)

        with open("pred_triplets(lstm).csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(true_predictions)

        with open("label_triplets(lstm).csv", "w", newline="") as f:
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
    ):
        """
        preds (np.ndarray: N X T X T X V): prediction logits from model outputs.
        """
        answers = []
        all_input_ids = [ins["input_ids"] for ins in self.eval_dataset]  # N X T
        all_labels = [ins["labels"] for ins in self.eval_dataset]  # N X T

        if not is_label:  # preds is prediction N X T X T X V
            for i, logits in enumerate(preds):  # logits: T X T X V (T is after padding)
                input_ids = all_input_ids[i]  # T (non-padding)
                labels = all_labels[i]  # T (non-padding)
                logits = logits[: len(input_ids), : len(input_ids)]  # remove the padding part of the logit
                sent_res = []

                for query, logit in enumerate(logits):  # T X V
                    score = np.exp(logit) / np.exp(logit).sum(-1, keepdims=True)  # T X V
                    labels_idx = score.argmax(axis=-1)  # T

                    seq_entities = self.handling_score(
                        labels_idx,
                        input_ids,
                        labels,
                        grouped_entities,
                        ignore_labels,
                        is_label=False,
                    )  # List[Dict] 1-D
                    sent_res.append(seq_entities)  # List[List[Dict]] 2-D

                answers.append(sent_res)

        else:  # preds are labels N X T
            for i, label_ids in enumerate(preds):  # label_ids: T (T is after padding)
                input_ids = all_input_ids[i]  # T (non-padding)
                labels_idx = label_ids[: len(input_ids)]  # remove the padding part of the labels
                score = np.array([[1.0] * self.num_labels] * len(labels_idx))  # T X V

                seq_entities = self.handling_score(
                    labels_idx,
                    input_ids,
                    labels_idx,
                    grouped_entities,
                    ignore_labels,
                    is_label=True,
                )  # List[Dict] 1-D
                answers.append(seq_entities)

        if len(answers) == 1:
            return answers[0]

        return answers

    def handling_score(self, labels_idx, input_ids, gen_labels, grouped_entities, ignore_labels, is_label=False):
        entities = []
        # Filter to labels not in `self.ignore_labels`
        # Filter special_tokens
        filtered_labels_idx = []

        for idx, label_idx in enumerate(labels_idx):
            if self.id2label[label_idx] not in ignore_labels:
                filtered_labels_idx.append((idx, label_idx))

        for idx, label_idx in filtered_labels_idx:
            word = self.word_dict._decode(input_ids[idx])
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
                    "entity": self.id2label[label_idx],
                    "index": idx,
                }

            if grouped_entities:
                entity["is_subword"] = is_subword

            entities += [entity]

        if grouped_entities:
            return self.group_entities(entities)  # Append ungrouped entities
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
            "word": " ".join(tokens),
            "index": index,
        }
        return entity_group

    def group_entities(self, entities: List[dict]) -> List[dict]:
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
            is_subword = False
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

    def neg_sample_map(self, dataset):
        """
        Negative sampling for the raw dataset before feed into the Trainer.
        """
        gen_data = deepcopy(dataset)
        for instance in gen_data:
            seq_len = len(instance["tokens"])
            sentID = instance["sentID"]
            insID = instance["instanceID"]
            usable_ids = list(set(range(seq_len)) - self.sent2query[sentID])

            if not usable_ids:
                query_id = -100
            else:
                query_id = random.choice(usable_ids)  # randomly chose the non-queryid as neg sampes queryid

            instance["instanceID"] = -insID - 1
            instance["query_ids"] = query_id
            instance["ner_tags"] = ["O"] * seq_len

        return gen_data

    def negative_sampling(self, dataset):
        """
        Return the merged dataset and sampled negative instances for training.
        """
        # get query_id list for each sentID
        self.sent2query = defaultdict(set)
        for instance in dataset:
            sentID, query_id = instance["sentID"], instance["query_ids"]
            self.sent2query[sentID].add(query_id)

        neg_data = self.neg_sample_map(dataset)
        neg_usedata = [ins for ins in neg_data if ins["query_ids"] != -100]

        # sample 10% of negative sequences
        sample_size = int(self.data_args.sample_rate * len(neg_usedata))
        if sample_size == 0:  # empty idx
            neg_data_final = neg_usedata
        else:
            neg_data_final = random.sample(neg_usedata, sample_size)

        # concat both train and negative instances
        final_train = dataset + neg_data_final
        final_train.sort(key=lambda x: x["sentID"])
        print(
            "dataset size: {}, neg size: {}, merged size: {}".format(
                len(dataset), len(neg_data_final), len(final_train)
            )
        )

        return final_train

    def instance_select(self, model):
        """
        Selecting training instance with higher confidence score w.r.t the position attention distribution.
        """
        print("selecting instances ...")
        # tokenize the raw text to feed into model
        data = self.gen_target_att(self.train_dataset)  # add target att (tokenize)
        data = NYT_dataset(data, self.word_dict)  # word2id
        dataloader = self.trainer.get_eval_dataloader(data)
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
                # pa = outputs["position_attentions"] # position attention (B X T)
                pa = outputs[-1]  # position attention (B X T)
                loss = self.kl_loss(pa.log(), target_att).sum(dim=1)  # sum on the T dimension to get (B)
                kl_set.extend(loss.detach().cpu().tolist())
                confidence = 1 / (1 + loss)  # the final confidence score (B)
                batch_idx = (
                    torch.where(confidence > self.threshold)[0] + step * self.training_args.eval_batch_size
                ).tolist()
                matched_idx.extend(batch_idx)

        # exchange matched_idx to instanceIDs
        insIDs = [ins["instanceID"] for ins in self.train_dataset]
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
        matched_dataset = [exp for exp in self.train_dataset if exp["instanceID"] in self.trust_idx]
        matched_patterns = [exp for exp in self.raw_patterns if exp["instanceID"] in self.trust_idx]

        return matched_dataset, matched_patterns

    def pattern_extract(self, dataset):
        """
        Inputs:
            dataset (object: huggingface dataset): raw dataset (before embedding and word2id).
        Outputs:
            raw_patterns (object: huggingface dataset): list of dictionary of patterns in all instances.

        For each query-level instance, we extract all patterns (string of tokens)
        that are between the query entity and each target entity.
        """
        patterns = deepcopy(dataset)
        for i, instance in enumerate(patterns):
            IOB_tags = instance["ner_tags"]
            insID = instance["instanceID"]
            if insID >= 0:  # not "O" sequences
                query_id = instance["query_ids"]  # before tokenization
                query_end = instance["query_ids"]  # the temp query entity end index
                tokens = instance["tokens"]
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
                instance["re_patterns"] = re_pattern_pair

        return patterns

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
        matched_idx = set(matched_idx)
        matched_dataset = [ins for ins in dataset if ins["instanceID"] in matched_idx]

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
            return self.gen_target_att(matched_dataset)

        else:
            sent2target = defaultdict(list)
            # Match those with high-quality relations instances
            for re, pat_list in self.new_patterns.items():
                for pat in pat_list:
                    query_id, target_id, sentID = self.pattern2target[re][pat]
                    sent2target[sentID].append(query_id)  # each sentID corresponds a list of query_ids.
                    sent2target[sentID].append(target_id)  # each sentID corresponds a list of target_ids.
            # Match those instances with its query entity as the tail entity of selected patterns
            data_instances = [ins["instanceID"] for ins in dataset]
            considered_idx = self.ent_instances - set(data_instances)
            considered_data = [ins for ins in self.train_dataset if ins["instanceID"] in considered_idx]
            print("getting tailed entites ...")
            tail_insIDs = []
            for instance in considered_data:
                sentID, queryID, insID = instance["sentID"], instance["query_ids"], instance["instanceID"]
                # Same sentence with tail entity and query entity in those entity instances
                if sentID in sent2target and queryID in sent2target[sentID]:
                    tail_insIDs.append(insID)

            ultimate_IDs = set(tail_insIDs + data_instances)
            matched_dataset = [ins for ins in self.train_dataset if ins["instanceID"] in ultimate_IDs]
            # We directly use all dataset for next training
            if self.use_negative_sampling:
                matched_dataset = self.negative_sampling(matched_dataset)

            print("tail size: {}, data size (merge tail): {}".format(len(tail_insIDs), len(dataset)))

            # add target attention
            matched_dataset = self.gen_target_att(matched_dataset)
            return NYT_dataset(matched_dataset, self.word_dict)

    def bootstrap(self, model, epoch):
        """
        The whole architecture of boostrap training procedure.
        """
        self.epoch = epoch
        # Select trustable instances based on current model
        trust_dataset, trust_patterns = self.instance_select(model)
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
        logging_dict["new patterns"] = {
            k: [{"pattern": v, "count": self.pattern_count[k][v]} for v in v_list]
            for k, v_list in added_patterns.items()
        }
        logging_dict["patterns"] = {
            k: [{"pattern": v, "count": self.pattern_count[k][v]} for v in v_list]
            for k, v_list in self.patterns.items()
        }
        logging_dict["pattern_bag_of_words"] = self.pattern_bow_freq
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
            compute_metrics=self.compute_metrics,
            bootstrap=self.bootstrap,
            original_dataset=self.original_dataset if self.training_args.do_train else self.eval_dataset,
            use_bootstrap=self.use_bootstrap,
            boot_start_epoch=self.boot_start_epoch,
        )
        self.trainer = trainer
        # Training
        if self.training_args.do_train:
            if os.path.isdir(self.model_args.model_name_or_path):
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
