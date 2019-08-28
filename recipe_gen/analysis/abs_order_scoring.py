''''
-*- coding: utf-8 -*-

Scoring script for recipe level coherence score via absolute ordering

@inproceedings{majumder2019emnlp,
  title={Generating Personalized Recipes from Historical User Preferences},
  author={Majumder, Bodhisattwa Prasad* and Li, Shuyang* and Ni, Jianmo and McAuley, Julian},
  booktitle={EMNLP},
  year={2019}
}

Copyright Shuyang Li & Bodhisattwa Majumder
License: GNU GPLv3
'''

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

from functools import partial
from datetime import datetime
from tqdm import tqdm

from .absolute_order_teacher import AbsoluteOrderTeacher

def generate_step_batches(df, randomize=False, batch_size=32):
    """
    Generate batches of recipes. Each batch is of the shape:
    B : T : D
        B = batch size
        T = # steps (15)
        D = sentence embedding dimension
    
    Args:
        df (pd.DataFrame): pandas DataFrame of steps, with columns:
            i - recipe index
            steps - stores T : D step numpy arrays
        randomize (bool, optional): Whether to randomize the batch order. Defaults to False.
        batch_size (int, optional): Number of examples in each batch. Defaults to 32.
    """
    # Total # examples
    n_rows = len(df)

    # Order of indices
    ix_order = df.index.values
    if randomize:
        np.random.shuffle(ix_order)

    # Generate batches (last batch may be uneven sized)
    for i in range(0, n_rows, batch_size):
        yield (
            torch.FloatTensor(df.loc[ix_order[i:(i+batch_size)]]['bert_features_step_lists'].values.tolist()), 
            torch.FloatTensor(df.loc[ix_order[i:(i+batch_size)]]['bert_features_pred_step_lists'].values.tolist())
        )

def test_run_epoch(device, df, model, batch_size=24):
    start = datetime.now()
    total_score = 0.0

    # Iterate over batches in the epoch
    for i, batch in enumerate(tqdm(
        generate_step_batches(df, randomize=False, batch_size=batch_size),
        total=int(len(df) / batch_size)
    )):
        
        gold_batch = batch[0]
        pred_batch = batch[1].to(device)

        # Batch size
        this_batch_size = gold_batch.size()[0]

        # Fill out batch information
        gold_batch_fwd = gold_batch.to(device)

        # Flip steps
        gold_batch_bck = torch.flip(gold_batch_fwd, [1]).to(device)

        # Calculate cosine similarity of fwd -> bwd (loss is sum)
        cos_sim_1 = model(gold_batch_fwd, pred_batch)
        cos_sim_2 = model(gold_batch_bck, pred_batch)
        score = cos_sim_1 - cos_sim_2
        batch_score = torch.sum(score)
        total_score += batch_score

    # Average fwd-bwd cosine similarity over epoch    
    return total_score / len(df)

def main():

    import os
    import torch
    import argparse
    import torch.nn.init as init

    from recipe_gen.utils import get_device, count_parameters

    parser = argparse.ArgumentParser(description='Baseline for recipe generation (dynamic attn)')
    parser.add_argument('--gold-path', type=str, required=True, help='location of the data corpus')
    parser.add_argument('--pred-path', type=str, required=True, help='location of the data corpus')
    parser.add_argument('--model-path', type=str, required=True, help='location of the data corpus')

    args = parser.parse_args()

    start = datetime.now()
    USE_CUDA, DEVICE = get_device()

    args = parser.parse_args()

    gold_path = args.gold_path
    pred_path = args.pred_path
    model_path = args.model_path

    model = AbsoluteOrderTeacher(embedding_dim=768, hidden_size=256, gru_layers=2)
    gold_test_df = pd.read_msgpack(gold_path)
    pred_test_df = pd.read_msgpack(pred_path)

    gold_bert_feature_lists = []
    for index, row in tqdm(gold_test_df.iterrows(), total=len(gold_test_df)):
        row = row['bert_features_steps'].tolist()
        if type(row[-1]) == tuple:
            pads = [x for x in row[-1]] 
            new_row = np.array(row[:-1] + pads)
            if new_row.shape[0] > 15:
                new_row = new_row[:15]
            gold_bert_feature_lists.append(new_row)
        else:
            new_row = np.array(row)
            if new_row.shape[0] > 15:
                new_row = new_row[:15]
            gold_bert_feature_lists.append(np.array(new_row))
    gold_test_df['bert_features_steps'] = gold_bert_feature_lists

    print(gold_test_df.head(2))

    pred_bert_feature_lists = []
    for index, row in tqdm(pred_test_df.iterrows(), total=len(pred_test_df)):
        row = row['bert_features_steps'].tolist()
        if type(row[-1]) == tuple:
            pads = [x for x in row[-1]]
            new_row = np.array(row[:-1] + pads)
            if new_row.shape[0] > 15:
                new_row = new_row[:15]
            pred_bert_feature_lists.append(new_row)
        else:
            new_row = np.array(row)
            if new_row.shape[0] > 15:
                new_row = new_row[:15]
            pred_bert_feature_lists.append(np.array(new_row))
    pred_test_df['bert_features_steps'] = pred_bert_feature_lists


    eval_df = pd.merge(gold_test_df, pred_test_df, left_on='i', right_on='i')
    eval_df.dropna()
    eval_df.columns = ['i', 'bert_features_step_lists', 'bert_features_pred_step_lists']
    print(eval_df.head(2))
    eval_df = eval_df.sample(frac=0.5)
    print(len(eval_df))

    
    # Load state dict
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict, strict=True)
    model.to(DEVICE)
    print('Model loaded')

    model.eval()
    with torch.no_grad():
        eval_loss = test_run_epoch(
            device=DEVICE,
            df=eval_df,
            model=model,
            batch_size=16
            )
    
    print('The absolute oder score is: {}'.format(eval_loss))

if __name__ == "__main__":
    main()