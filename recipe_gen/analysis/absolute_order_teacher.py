'''
-*- coding: utf-8 -*-

Teacher moder for absolute order model for recipe level coherence.

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

class AbsoluteOrderTeacher(nn.Module):
    """
    Teacher that learns ordered recipe embeddings to inform future work
    """

    def __init__(self, embedding_dim=300, hidden_size=256, gru_layers=2, dropout=0.2):
        """[summary]
        
        Args:
            embedding_dim (int, optional): [description]. Defaults to 300.
            hidden_size (int, optional): [description]. Defaults to 256.
            gru_layers (int, optional): [description]. Defaults to 2.
            dropout (float, optional): [description]. Defaults to 0.2.
        """
        super().__init__()

        # Params
        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim
        self.gru_layers = gru_layers
        self.dropout = dropout

        # GRU learner
        self.gru = nn.GRU(
            self.embedding_dim,
            self.hidden_size,
            self.gru_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout
        )
        print('Generated Absolute Order Teacher. Base GRU: {:,} x {}, embedding dim {:,}'.format(
            hidden_size, gru_layers, embedding_dim
        ))

    def forward(self, x1, x2):
        # Generate representations of each ordered step set
        # L : B : H
        _, repr_1 = self.gru(x1)
        _, repr_2 = self.gru(x2)

        # The representation is the final layer final output
        # B : H
        if self.gru_layers > 1:
            repr_1 = repr_1[-1]
            repr_2 = repr_2[-1]

        # Cosine similarity between the step sets (to be minimized or used)
        # B
        cos_similarity = F.cosine_similarity(repr_1, repr_2, dim=1)

        return cos_similarity

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
        yield torch.FloatTensor(
            df.loc[ix_order[i:(i+batch_size)]]['bert_features_step_lists'].values.tolist()
        )

def _sample_df(size=300, n_steps=3, dim=5):
    """
from recipe_gen.new_eval.absolute_order_teacher import _sample_df, generate_step_batches

df = _sample_df(size=100)
for b in generate_step_batches(df, randomize=False, batch_size=8):
    break

b
b.size()
    """
    return pd.DataFrame({
        'i': np.arange(size),
        'bert_features_step_lists': [np.random.rand(n_steps, dim) for _ in range(size)]
    })

def run_epoch(device, df, model, opt=None, batch_size=24, randomize_batch=True, print_every=100):
    start = datetime.now()
    epoch_loss = 0.0

    # Iterate over batches in the epoch
    for i, batch in enumerate(tqdm(
        generate_step_batches(df, randomize=randomize_batch, batch_size=batch_size),
        total=int(len(df) / batch_size)
    )):
        # Batch size
        this_batch_size = batch.size()[0]

        # Fill out batch information
        batch_fwd = batch.to(device)

        # Flip steps
        batch_bck = torch.flip(batch_fwd, [1])

        # Calculate cosine similarity of fwd -> bwd (loss is sum)
        cos_sim = model(batch_fwd, batch_bck)
        loss = torch.sum(cos_sim)
        epoch_loss += loss

        # Backprop
        if opt is not None:
            opt.zero_grad()
            loss.backward()
            opt.step()

        if model.training and i % print_every == 0:
            elapsed = datetime.now() - start
            print("Epoch Step: {} Loss: {:.5f}".format(
                i,
                loss / this_batch_size
            ))
            start = datetime.now()
            print_tokens = 0

    # Average fwd-bwd cosine similarity over epoch    
    return epoch_loss / len(df)

def train_model(device, model, train_df, eval_df, lr=0.001, batch_size=24, print_every=100,
                n_epochs=200, save_folder='', exp_name='abs_order_teacher'):
    start_train = datetime.now()

    # Making sure the folder exists
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    # Model serialization
    model_loc = os.path.join(save_folder, 'model_{}'.format(exp_name))
    print('{} - Training model with LR {}, saving to {}'.format(
        datetime.now() - start_train,
        lr,
        model_loc
    ))

    # Optimizer
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    # For each epoch, train and evaluate
    eval_losses = []
    best_eval_loss = 1e10
    best_model_save_loc = ''
    for epoch in range(n_epochs):
        # Train model
        model.train()
        train_loss = run_epoch(
            device=device,
            df=train_df,
            model=model,
            opt=opt,
            batch_size=batch_size,
            randomize_batch=True,
            print_every=print_every
        )
        print('[{} - Epoch {}] Train loss {:.5f}'.format(
            datetime.now() - start_train, epoch, train_loss
        ))

        # Save improved model
        candidate_loc = model_loc + 'CANDIDATE.pt'
        torch.save(model.state_dict(), candidate_loc)
        print('[{} - Epoch {}] Saved candidate model to {}'.format(
            datetime.now() - start_train, epoch, candidate_loc
        ))

        # Validation
        model.eval()
        with torch.no_grad():
            eval_loss = run_epoch(
                device=device,
                df=eval_df,
                model=model,
                opt=None,
                batch_size=batch_size,
                randomize_batch=False,
                print_every=print_every
            )
            print('[{} - Epoch {}] Train loss {:.5f}'.format(
                datetime.now() - start_train, epoch, train_loss
            ))

            # Early stopping - compare with prior perplexity
            prior_best_loss = eval_losses[-1] if eval_losses else 9.9e12
            print("Validation loss: %f" % eval_loss)
            eval_losses.append(eval_loss)

            if prior_best_loss < eval_loss:
                print('[{} - Epoch {}] EARLY STOPPAGE'.format(
                    datetime.now() - start_train, epoch
                ))
                break

            # Save improved model
            best_eval_loss = min(best_eval_loss, eval_loss)
            best_model_save_loc = model_loc + '_e{}.pt'.format(epoch)
            torch.save(model.state_dict(), best_model_save_loc)
            print('[{} - Epoch {}] Saved model to {}'.format(
                datetime.now() - start_train, epoch, best_model_save_loc
            ))

    return eval_losses

def main():
    import os
    import torch
    import argparse
    import torch.nn.init as init

    from recipe_gen.utils import get_device, count_parameters

    parser = argparse.ArgumentParser(description='Baseline for recipe generation (dynamic attn)')
    parser.add_argument('--data-dir', type=str, required=True, help='location of the data corpus')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N', help='batch size')
    parser.add_argument('--sent-emb-size', type=int, default=300, help='size of word embeddings')
    
    parser.add_argument('--nhid', type=int, default=256, help='number of hidden units per layer')
    parser.add_argument('--nlayers', type=int, default=1, help='number of layers')
    parser.add_argument('--lr', type=float, default=1e-2, help='initial learning rate')
    parser.add_argument('--epochs', type=int, default=200, help='upper epoch limit')
    
    parser.add_argument('--dropout', type=float, default=0.2,
        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--log-interval', type=int, default=500, metavar='N', help='report interval')
    parser.add_argument('--save', type=str, default='/home/shuyang/data/torch-models',
        help='path to save the final model')

    parser.add_argument('--exp-name', type=str, required=True, default='abs_order_teacher', help='exp name')

    parser.add_argument('--load-checkpoint', type=str, default=None,
        help='Load from state dict checkpoint')
    args = parser.parse_args()

    start = datetime.now()
    USE_CUDA, DEVICE = get_device()

    args = parser.parse_args()

    data_dir = args.data_dir
    batch_size = args.batch_size
    sent_emb_dim = args.sent_emb_size
    hidden_size = args.nhid
    n_layers = args.nlayers
    dropout = args.dropout
    num_epochs = args.epochs
    lr = args.lr
    print_every = args.log_interval
    exp_name = args.exp_name
    save_folder = args.save
    checkpoint_loc = args.load_checkpoint
    if checkpoint_loc is not None:
        print('Loading state dict from {}'.format(checkpoint_loc))

    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    train_df = pd.read_msgpack(os.path.join(data_dir, 'absolute_order_model_train.msgpack'))
    eval_df = pd.read_msgpack(os.path.join(data_dir, 'absolute_order_model_val.msgpack'))

    start_bad_stuff = datetime.now()
    train_bert_feature_lists = []
    for index, row in tqdm(train_df.iterrows(), total=len(train_df)):
        row = row['bert_features_steps'].tolist()
        if type(row[-1]) == tuple:
            pads = [x for x in row[-1]] 
            train_bert_feature_lists.append(np.array(row[:-1] + pads))
        else:
            train_bert_feature_lists.append(np.array(row))
    train_df['bert_features_step_lists'] = train_bert_feature_lists
    print(train_df.head(2))
    print(train_df['bert_features_step_lists'].iloc[0].shape)

    eval_bert_feature_lists = []
    for index, row in tqdm(eval_df.iterrows(), total=len(eval_df)):
        row = row['bert_features_steps'].tolist()
        if type(row[-1]) == tuple:
            pads = [x for x in row[-1]] 
            eval_bert_feature_lists.append(np.array(row[:-1] + pads))
        else:
            eval_bert_feature_lists.append(np.array(row))
    eval_df['bert_features_step_lists'] = eval_bert_feature_lists

    print('Time taken: {}'.format(datetime.now() - start_bad_stuff))
    

    model = AbsoluteOrderTeacher(embedding_dim=sent_emb_dim, hidden_size=hidden_size, gru_layers=n_layers, dropout=dropout)
    model.to(DEVICE)

    train_model(DEVICE, model, train_df, eval_df, lr=lr, batch_size=batch_size, print_every=print_every,
                n_epochs=num_epochs, save_folder=save_folder, exp_name=exp_name)

if __name__ == "__main__":
    main()

