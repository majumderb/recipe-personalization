'''
-*- coding: utf-8 -*-

Training script for the prior technique model

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
import numpy as np
import pickle
import torch.utils.data as data
import torch.nn as nn

from functools import partial
from tqdm import tqdm
from itertools import chain
from datetime import datetime

from recipe_gen.language import START_INDEX, PAD_INDEX, VOCAB_SIZE
from recipe_gen.pipeline.train import train_model
from recipe_gen.pipeline.batch import get_batch_information_general, get_user_prior_techniques_mask
from recipe_gen.pipeline.eval import top_k_logits, sample_next_token

def run_epoch(device, model, sampler, loss_compute, print_every, max_len,
              user_items_df, clip=None, teacher_forcing=False, max_name_len=15, **tensor_kwargs):
    """
    Run a single epoch

    Arguments:
        device {torch.device} -- Torch device on which to store/process data
        model {nn.Module} -- Model to be trained/run
        sampler {BatchSampler} -- Data sampler
        loss_compute {funct} -- Function to compute loss for each batch
        print_every {int} -- Log loss every k iterations
        max_len {int} -- Maximum length / number of steps to unroll and predict
        user_items_df {pd.DataFrame} -- DataFrame with 'items' column containing historical items
            a user has interacted with

    Keyword Arguments:
        clip {float} -- Clip gradients to a maximum (default: {None})
        teacher_forcing {bool} -- Whether to do teacher-forcing in training (default: {False})
        max_name_len {int} -- Maximum # timesteps to unroll to predict name (default: {15})
        **tensor_kwargs {torch.Tensor} -- Assorted tensors for fun and profit

    Returns:
        float -- Average loss across the epoch
    """
    start = datetime.now()
    total_tokens = 0
    total_name_tokens = 0
    total_loss = 0.0
    total_name_loss = 0.0
    print_tokens = 0

    # Extract into tuples and list
    tensor_names, base_tensors = zip(*tensor_kwargs.items())

    # Iterate through batches in the epoch
    for i, batch in enumerate(tqdm(sampler.epoch_batches(), total=sampler.n_batches), 1):
        batch_users, items = [t.to(device) for t in batch]

        # Fill out batch information
        batch_map = dict(zip(
            tensor_names,
            get_batch_information_general(items, *base_tensors)
        ))

        # Get user prior techniques
        user_prior_technique_masks = torch.stack([get_user_prior_techniques_mask(
            user_ix=uix.item(), item_ix=iix.item(),
            user_items_df=user_items_df, tech_mask_tensor=tensor_kwargs['tech_mask_tensor'],
            device=device, normalize=True
        ) for uix, iix in zip(batch_users, items)], dim=0)

        # Logistics
        this_batch_size = batch_map['steps_tensor'].size(0)
        this_batch_num_tokens = (batch_map['steps_tensor'] != PAD_INDEX).data.sum().item()
        this_batch_num_name_tokens = 0
        this_batch_num_name_tokens = (batch_map['name_tensor'] != PAD_INDEX).data.sum().item()
        name_targets = batch_map['name_tensor']

        # Batch first
        # Comparing out(token[t-1]) to token[t]
        (log_probs, _), (name_log_probs, _) = model.forward(
            device=device, inputs=(
                batch_map['calorie_level_tensor'],
                batch_map['name_tensor'],
                batch_map['ingr_tensor']
            ),
            targets=batch_map['steps_tensor'][:, :-1],
            ingr_masks=batch_map['ingr_mask_tensor'],
            user_prior_technique_masks=user_prior_technique_masks,
            max_len=max_len-1,
            start_token=START_INDEX,
            teacher_forcing=teacher_forcing,
            name_targets=name_targets[:, :-1],
            max_name_len=max_name_len-1,
            visualize=False
        )
        loss, name_loss = loss_compute(
            log_probs, batch_map['steps_tensor'][:, 1:],
            name_outputs=name_log_probs,
            name_targets=name_targets[:, 1:],
            norm=this_batch_size,
            model=model,
            clip=clip
        )

        total_loss += loss
        total_name_loss += name_loss

        # Logging
        total_tokens += this_batch_num_tokens
        total_name_tokens += this_batch_num_name_tokens
        print_tokens += this_batch_num_tokens

        if model.training and i % print_every == 0:
            elapsed = datetime.now() - start
            print("Epoch Step: {} LM Loss: {:.5f}; Name Loss: {:.5f}; Tokens/s: {:.3f}".format(
                i, loss / this_batch_size, name_loss / this_batch_size, print_tokens / elapsed.seconds
            ))
            start = datetime.now()
            print_tokens = 0

        del log_probs, name_log_probs

    # Reshuffle the sampler
    sampler.renew_indices()

    if total_name_tokens > 0:
        print('\nName Perplexity: {}'.format(np.exp(total_name_loss / float(total_name_tokens))))

    return np.exp(total_loss / float(total_tokens))

'''
==== RUN
nohup python -u -m recipe_gen.models.user_technique_pref.train --data-dir <DATA FOLDER> --batch-size 38 --vocab-emb-size 300 --tech-emb-size 50 --calorie-emb-size 5 --nhid 256 --nlayers 2 --lr 1e-3 --epochs 50 --annealing-rate 0.9 --save <MODEL FOLDER> --ingr-emb --ingr-gru --exp-name <EXP> > <EXP>.out &
tail -f <EXP>.out
'''
if __name__ == "__main__":
    import torch
    import argparse
    import torch.nn.init as init

    from functools import partial

    from recipe_gen.utils import get_device, count_parameters
    from recipe_gen.language import N_TECHNIQUES
    from recipe_gen.pipeline import DataFrameDataset, BatchSampler
    from recipe_gen.pipeline.batch import load_full_data, pad_recipe_info, load_recipe_tensors

    # Module imports
    from . import create_model
    from .generate import decode_single

    parser = argparse.ArgumentParser(description='Baseline for recipe generation (dynamic attn)')
    parser.add_argument('--data-dir', type=str, required=True, help='location of the data corpus')
    parser.add_argument('--batch-size', type=int, default=48, metavar='N', help='batch size')
    parser.add_argument('--vocab-emb-size', type=int, default=50, help='size of word embeddings')
    parser.add_argument('--tech-emb-size', type=int, default=50, help='size of technique embeddings')
    parser.add_argument('--calorie-emb-size', type=int, default=50, help='size of calorie embeddings')
    parser.add_argument('--ingr-emb-size', type=int, default=10, help='size of ingr embeddings')

    parser.add_argument('--nhid', type=int, default=256, help='number of hidden units per layer')
    parser.add_argument('--nlayers', type=int, default=1, help='number of layers')
    parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate')
    parser.add_argument('--clip', type=float, default=None, help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=5, help='upper epoch limit')

    parser.add_argument('--dropout', type=float, default=0.2,
        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--log-interval', type=int, default=500, metavar='N', help='report interval')
    parser.add_argument('--annealing-rate', type=float, default=1.0, metavar='N',
        help='learning rate annealing (default 1.0 - no annealing, 0.0 - early stoppage)')
    parser.add_argument('--teacher-forcing', default=None, type=int,
        help='number of epochs to teacher-force when training (default ALL epochs)')
    parser.add_argument('--save', type=str, default='<MODEL FOLDER>',
        help='path to save the final model')
    parser.add_argument('--exp-name', type=str, required=True, default='base', help='exp name')

    parser.add_argument('--ingr-gru', action='store_true', default=False,
        help='Use BiGRU for ingredient encoding')
    parser.add_argument('--decode-name', action='store_true', default=False,
        help='Multi-task learn to decode name along with recipe')
    parser.add_argument('--ingr-emb', action='store_true', default=False,
        help='Use Ingr embedding in encoder')
    parser.add_argument('--shared-proj', action='store_true', default=False,
        help='Share projection layers for name and steps')

    parser.add_argument('--load-checkpoint', type=str, default=None,
        help='Load from state dict checkpoint')
    args = parser.parse_args()

    start = datetime.now()
    USE_CUDA, DEVICE = get_device()

    # Filters
    MAX_NAME = 15
    MAX_INGR = 5
    MAX_INGR_TOK = 20
    MAX_STEP_TOK = 256

    # Reproducibility
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Args
    data_dir = args.data_dir
    batch_size = args.batch_size
    vocab_emb_dim = args.vocab_emb_size
    tech_emb_dim = args.tech_emb_size
    calorie_emb_dim = args.calorie_emb_size
    ingr_emb_dim = args.ingr_emb_size
    hidden_size = args.nhid
    n_layers = args.nlayers
    dropout = args.dropout
    num_epochs = args.epochs
    lr = args.lr
    print_every = args.log_interval
    exp_name = args.exp_name
    save_folder = args.save
    lr_annealing_rate = args.annealing_rate
    clip = args.clip
    ingr_gru = args.ingr_gru
    ingr_emb = args.ingr_emb
    decode_name = args.decode_name
    shared_proj = args.shared_proj
    n_teacher_forcing = args.teacher_forcing
    checkpoint_loc = args.load_checkpoint
    if checkpoint_loc is not None:
        print('Loading state dict from {}'.format(checkpoint_loc))
    if n_teacher_forcing is None:
        n_teacher_forcing = num_epochs

    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    # Get the DFs
    train_df, valid_df, test_df, user_items_df, df_r, ingr_map = load_full_data(data_dir)
    n_items = len(df_r)
    print('{} - Data loaded.'.format(datetime.now() - start))

    N_INGREDIENTS = 0
    if ingr_emb:
        print('INGR EMBEDDING')
        n_ingredients_og = max(chain.from_iterable(df_r['ingredient_ids'].values)) + 1
        PAD_INGR = n_ingredients_og
        N_INGREDIENTS = n_ingredients_og + 1

    # Pad recipe information
    df_r = pad_recipe_info(
        df_r, max_name_tokens=MAX_NAME, min_ingredients=3, max_ingredients=MAX_INGR,
        max_ingr_tokens=MAX_INGR_TOK, max_step_tokens=MAX_STEP_TOK
    )

    tensors_to_load = [
        ('name_tensor', 'name_tokens'),
        ('calorie_level_tensor', 'calorie_level'),
        ('technique_tensor', 'techniques'),
        ('ingr_tensor', 'ingredient_ids' if ingr_emb else 'ingredient_tokens'),
        ('steps_tensor', 'steps_tokens'),
        ('ingr_mask_tensor', 'ingredient_id_mask' if ingr_emb else 'ingredient_mask'),
        ('tech_mask_tensor', 'techniques_mask'),
    ]
    tensor_names, tensor_cols = zip(*tensors_to_load)

    # Load tensors into memory
    memory_tensors = load_recipe_tensors(
        df_r, DEVICE, cols=tensor_cols, types=[torch.LongTensor] * len(tensors_to_load)
    )
    memory_tensor_map = dict(zip(tensor_names, memory_tensors))
    print('{} - Tensors loaded in memory.'.format(datetime.now() - start))

    # Samplers
    train_data = DataFrameDataset(train_df, ['u', 'i'])
    train_sampler = BatchSampler(train_data, batch_size, random=True)
    valid_data = DataFrameDataset(valid_df, ['u', 'i'])
    valid_sampler = BatchSampler(valid_data, batch_size)
    test_data = DataFrameDataset(test_df, ['u', 'i'])
    test_sampler = BatchSampler(test_data, batch_size)

    '''
    Create model
    '''
    model = create_model(
        vocab_emb_dim=vocab_emb_dim, tech_emb_dim=tech_emb_dim, calorie_emb_dim=calorie_emb_dim,
        hidden_size=hidden_size, n_layers=n_layers, dropout=dropout, max_ingr=MAX_INGR,
        max_ingr_tok=MAX_INGR_TOK, use_cuda=USE_CUDA, state_dict_path=checkpoint_loc,
        ingr_gru=ingr_gru, decode_name=decode_name, ingr_emb=ingr_emb, num_ingr=N_INGREDIENTS,
        ingr_emb_dim=ingr_emb_dim, shared_projection=shared_proj,
    )
    print('{} - {} Model defined with {:,} parameters'.format(
        datetime.now() - start, exp_name, count_parameters(model)
    ))

    '''
    TRAIN MODEL
    '''
    partial_run_epoch = partial(
        run_epoch,
        user_items_df=user_items_df,
        print_every=print_every,
        max_len=MAX_STEP_TOK,
        max_name_len=MAX_NAME,
        clip=clip,
        **memory_tensor_map
    )
    partial_decode_single = partial(
        decode_single,
        user_items_df=user_items_df,
        max_len=MAX_STEP_TOK,
        max_name_len=MAX_NAME,
        ingr_map=ingr_map,
        max_ingr=MAX_INGR,
        max_ingr_tok=MAX_INGR_TOK,
        **memory_tensor_map
    )

    dev_perplexities, test_perplexity = train_model(
        DEVICE, model, train_sampler, valid_sampler, test_sampler,
        num_epochs=num_epochs, lr=lr, exp_name=exp_name,
        partial_run_epoch=partial_run_epoch, partial_decode_single=partial_decode_single,
        lr_annealing_rate=lr_annealing_rate, n_teacher_forcing=n_teacher_forcing,
        save_folder=save_folder)

    # Save perplexities
    stats_loc = os.path.join(args.save, 'model_stats_{}.pkl'.format(args.exp_name))
    with open(stats_loc, 'wb') as stats_file:
        pickle.dump([dev_perplexities, test_perplexity], stats_file, protocol=pickle.HIGHEST_PROTOCOL)
    print('{} - Saved stats to {}'.format(
        datetime.now() - start, stats_loc
    ))
