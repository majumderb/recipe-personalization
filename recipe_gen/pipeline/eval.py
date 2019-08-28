'''
-*- coding: utf-8 -*-

Utilities for getting evaluation scores

@inproceedings{majumder2019emnlp,
  title={Generating Personalized Recipes from Historical User Preferences},
  author={Majumder, Bodhisattwa Prasad* and Li, Shuyang* and Ni, Jianmo and McAuley, Julian},
  booktitle={EMNLP},
  year={2019}
}

Copyright Shuyang Li & Bodhisattwa Majumder
License: GNU GPLv3
'''

from datetime import datetime
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu
from pyrouge import Rouge155
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import json
import spacy
import pickle
import argparse
import pandas as pd
import numpy as np

def top_k_logits(logits, k):
    """
    Masks everything but the k top entries as -infinity (1e10).
    Used to mask logits such that e^-infinity -> 0 won't contribute to the
    sum of the denominator.
    """
    if k == 0:
        return logits
    else:
        values = torch.topk(logits, k)[0]
        batch_mins = values[:, :, -1].expand_as(logits.squeeze(1)).unsqueeze(1)
        return torch.where(logits < batch_mins, torch.ones_like(logits) * -1e10, logits)

def top_p_logits(logits, p=0.9):
    """
    Masks everything but the top probability entries as -infinity.
    """
    if p == 1:
        return logits
    else:
        probs = torch.softmax(logits, dim=-1)
        sorted_probs, _ = torch.sort(probs, descending=True, dim=-1)

        cumprobs = sorted_probs.cumsum(dim=-1)
        # Create mask for all cumulative probabilities less than p
        mask = cumprobs < p
        # First mask must always be pickable
        mask = F.pad(mask[:, :, :-1], (1, 0, 0, 0), value=1)

        masked_probs = torch.where(mask, sorted_probs, torch.tensor(float('inf')).to(probs))

        batch_mins = masked_probs.min(dim=-1, keepdim=True)[0].expand_as(logits)

        # Mask out all logits (tail) that are too small
        return torch.where(probs < batch_mins, torch.tensor(float('-inf')).to(logits), logits)

def sample_next_token(logits, logit_modifier_fxn=partial(top_k_logits, k=0), sampler='greedy'):
    """
    Samples a token index from the token vocabulary 
    
    Arguments:
        logits {torch.Tensor} -- logits (in batch), size B x 1 x vocab_size
    
    Keyword Arguments:
        logit_modifier {partial fxn} -- modifies the logits, (default: top_k_logits(k = 0)), options: top_k_logits, top_p_logits (nucleus)
        sampler {str} -- string indicating sampler (default: {'greedy'}), options: 'multinomial'
    """
    logits = logit_modifier_fxn(logits)
    probs = F.softmax(logits, dim=-1)

    if sampler == 'greedy':
        return torch.argmax(probs, dim=-1)
    
    elif sampler == 'multinomial':
        return torch.multinomial(probs.squeeze(1), num_samples=1)

def calculate_rouge(system_dir, model_dir, 
                    system_pattern='item.(\d+).txt', model_pattern='item.A.#ID#.txt'):
    """
    Calculates ROUGE
    
    Arguments:
        system_dir {string} -- folder path for generated outputs
        model_dir {[type]} -- forlder path for reference or gold outputs
    
    Keyword Arguments:
        system_pattern {str} -- filename pattern for generated outputs (default: {'item.(\d+).txt'})
        model_pattern {str} -- filename pattern for reference or gold outputs (default: {'item.A.#ID#.txt'})
    
    Returns:
        dict -- dictionary with ROUGE scores per file
    """

    r = Rouge155()
    r.system_dir = system_dir
    r.model_dir = model_dir
    r.system_filename_pattern = system_pattern
    r.model_filename_pattern = model_pattern

    output = r.convert_and_evaluate()
    output_dict = r.output_to_dict(output)

    return output_dict

def calculate_bleu(system_dir, model_dir, tokenizer):
    """
    Calculates BLEU
    
    Arguments:
        system_dir {string} -- folder path for generated outputs
        model_dir {[type]} -- forlder path for reference or gold outputs
        tokenizer {spacy} -- tokenizer (spacy preferred)
    
    Returns:
        list -- list of bleu scores for all examples
    """

    ref_files = os.listdir(model_dir)
    gen_files = os.listdir(system_dir)

    bleu_1 = []
    bleu_2 = []
    bleu_3 = []
    bleu_4 = []

    for ref, gen in zip(ref_files, gen_files):
        ref_text = open(os.path.join(model_dir, ref), 'r').readlines()[0]
        gen_text = open(os.path.join(system_dir, gen), 'r').readlines()[0]
        
        reference_tokens = [t.text for t in nlp(ref_text.strip())]
        generated_tokens = [t.text for t in nlp(gen_text.strip())]

        bleu_1.append(sentence_bleu([reference_tokens], generated_tokens, weights=(1, 0, 0, 0)))
        bleu_2.append(sentence_bleu([reference_tokens], generated_tokens, weights=(0.5, 0.5, 0, 0)))
        bleu_3.append(sentence_bleu([reference_tokens], generated_tokens, weights=(0.33, 0.33, 0.33, 0)))
        bleu_4.append(sentence_bleu([reference_tokens], generated_tokens, weights=(0.25, 0.25, 0.25, 0.25)))
    
    return bleu_1, bleu_2, bleu_3, bleu_4

def document_lengths(system_dir, model_dir):
    """
    Compute document lengths (for lenght-wise score plotting)
    
    Arguments:
        system_dir {string} -- folder path for generated outputs
        model_dir {[type]} -- forlder path for reference or gold outputs
    """
    ref_files = os.listdir(model_dir)
    gen_files = os.listdir(system_dir)

    reference_lengths = []
    generated_lengths = []
    for ref, gen in zip(ref_files, gen_files):
        reference_token_length = len([str(t) for t in nlp(ref.readlines()[0].strip())])
        generated_token_length = len([str(t) for t in nlp(gen.readlines()[0].strip())])

        reference_lengths.append(reference_token_length)
        generated_lengths.append(generated_token_length)
    
    return reference_lengths, generated_lengths

'''
python3 -m recipe_gen.pipeline.eval --generated_dir results/model_no_user/system_outputs/ --gold_dir results/model_no_user/reference_outputs/

python3 -m recipe_gen.pipeline.eval --generated_dir results/model_user_dyn_20/system_outputs/ --gold_dir results/model_user_dyn_20/reference_outputs/

python3 -m recipe_gen.pipeline.eval --generated_dir results/model_nouser_mlp/system_outputs/ --gold_dir results/model_nouser_mlp/reference_outputs/

python3 -m recipe_gen.pipeline.eval --generated_dir results/model_user_tech/system_outputs/ --gold_dir results/model_user_tech/reference_outputs/
'''
if __name__ == "__main__":

    # define nlp pipeline for spacy
    nlp = spacy.load("en", disable=["parser", "tagger", "textcat", "ner", "vectors"])

    parser = argparse.ArgumentParser(description='Evaluation')
    parser.add_argument('--generated_dir', type=str, required=True, help='location of the generated data')
    parser.add_argument('--gold_dir', type=str, required=True, help='location of the gold data')
    
    args = parser.parse_args()

    generated_dir = args.generated_dir
    gold_dir = args.gold_dir
    
    start = datetime.now()

    rouge_scores = calculate_rouge(generated_dir, gold_dir)
    print('{} - ROUGE scores:\n{}'.format(
        datetime.now() - start, json.dumps(rouge_scores, indent=4)
    ))

    bleu_1, bleu_2, bleu_3, bleu_4 = calculate_bleu(generated_dir, gold_dir, nlp)
    print('{} - \nMean BLEU-1 score: {:.3f}\nMean BLEU-2 score: {:.3f}\nMean BLEU-3 score: {:.3f}\nMean BLEU-4 score: {:.3f}'.format(
        datetime.now() - start, np.array(bleu_1).mean()*100, np.array(bleu_2).mean()*100, np.array(bleu_3).mean()*100, np.array(bleu_4).mean()*100
    ))

