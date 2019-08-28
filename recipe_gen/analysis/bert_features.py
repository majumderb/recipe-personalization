'''
-*- coding: utf-8 -*-

Script for obtaining BERT pretrained features.

@inproceedings{majumder2019emnlp,
  title={Generating Personalized Recipes from Historical User Preferences},
  author={Majumder, Bodhisattwa Prasad* and Li, Shuyang* and Ni, Jianmo and McAuley, Julian},
  booktitle={EMNLP},
  year={2019}
}

Copyright Shuyang Li & Bodhisattwa Majumder
License: GNU GPLv3
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import collections
import logging
import json
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
import os

import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel

from recipe_gen.utils import get_device

USE_CUDA, DEVICE = get_device()

class InputExample(object):

    def __init__(self, unique_id, text_a, text_b):
        self.unique_id = unique_id
        self.text_a = text_a
        self.text_b = text_b

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids):
        self.unique_id = unique_id
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids

def convert_examples_to_features(examples, seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > seq_length - 2:
                tokens_a = tokens_a[0:(seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.

        tokens = [] #final sequence
        input_type_ids = []
        tokens.append("[CLS]")
        input_type_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            input_type_ids.append(0)
        tokens.append("[SEP]")
        input_type_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                input_type_ids.append(1)
            tokens.append("[SEP]")
            input_type_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens) # tokeniser is an angument of this function

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)

        assert len(input_ids) == seq_length
        assert len(input_mask) == seq_length
        assert len(input_type_ids) == seq_length

        features.append(
            InputFeatures(
                unique_id=example.unique_id,
                tokens=tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                input_type_ids=input_type_ids))
    return features

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def bert_input_prepare(sentences):
    # takes a list of sentences
    examples = []
    unique_id = 0
    for sent in sentences:
        text_a = None
        text_b = None
        m = re.match(r"^(.*) \|\|\| (.*)$", sent)
        if m is None:
            text_a = sent
        else:
            text_a = m.group(1)
            text_b = m.group(2)
        examples.append(
            InputExample(unique_id=unique_id, text_a=text_a, text_b=text_b))
        unique_id += 1
    return examples

def get_bert_hidden(examples, max_sent_length, tokenizer, model, device, max_recipe_steps):
    '''
    Input -
    a list of examples in bert input format
    maximum sentence length MAX_SENT_LENGTH
    tokenize as bert pretrained tokenier
    model as bert pretrained model
    device cuda with number of gpus

    Returns -
    A list of numpy arrays of each sentence encodings
    as proxied by the hidden states for [CLS] label
    '''

    # creating features
    features = convert_examples_to_features(
        examples=examples, seq_length=max_sent_length, tokenizer=tokenizer)

    # creating tensors - IDs, Masks, Indices
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long).to(device)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long).to(device)
    all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)

    # last_encoder_layer, _ = model(all_input_ids, token_type_ids=None, attention_mask=all_input_mask, output_all_encoded_layers=False)
    all_encoder_layers, pooled_output = model(all_input_ids, token_type_ids=None, attention_mask=all_input_mask)
    last_6_encoder_layer = all_encoder_layers[0]
    for i in range(1,6):
        last_6_encoder_layer += all_encoder_layers[i]

    all_sentence_encodings = []
    for b, example_index in enumerate(all_example_index):
        last_layer_output = last_6_encoder_layer.detach().cpu().numpy()
        last_layer_output = last_layer_output[b]
        last_layer_output = [round(x.item(), 6) for x in last_layer_output[0]]
        all_sentence_encodings.append(np.array(last_layer_output))
    if len(all_sentence_encodings) < max_recipe_steps:
        all_sentence_encodings.append([np.zeros_like(np.array(last_layer_output))]*(max_recipe_steps - len(all_sentence_encodings)))

    return np.array(all_sentence_encodings)    

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_path",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")

    args = parser.parse_args()

    tokenizer = BertTokenizer.from_pretrained(args.bert_model)
    model = BertModel.from_pretrained(args.bert_model)
    model.to(DEVICE)

    # read dataframe
    recipe_df = pd.read_csv(args.data_path)
    recipe_df['pred_steps'] = recipe_df['pred_steps'].agg(eval)

    bert_features_for_all_recipe = []
    for index, recipe in tqdm(recipe_df.iterrows(), total=len(recipe_df)):
        if len(recipe['pred_steps']) > 15:
            steps = recipe['pred_steps'][:15]
        else:
            steps = recipe['pred_steps']
        recipe_bert_like_input = bert_input_prepare(steps)
        recipe_bert_features = get_bert_hidden(
            recipe_bert_like_input,
            max_sent_length=256,
            tokenizer=tokenizer,
            model=model,
            device=DEVICE,
            max_recipe_steps=15)
        bert_features_for_all_recipe.append(recipe_bert_features)
    
    recipe_df['bert_features_steps'] = bert_features_for_all_recipe

    recipe_df[['i', 'bert_features_steps']].to_msgpack(os.path.join(args.output_dir, 'absolute_order_model_baseline_e9_pred.msgpack'))
    
if __name__ == "__main__":
    main()