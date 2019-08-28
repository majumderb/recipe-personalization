'''
-*- coding: utf-8 -*-

Evaluation for the prior item/name model

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
from recipe_gen.language import START_INDEX, PAD_INDEX, TECHNIQUES_LIST, pretty_decode, decode_ids, END_INDEX
from recipe_gen.pipeline.train import train_model
from recipe_gen.pipeline.batch import get_batch_information_general, get_user_prior_item_mask
from recipe_gen.pipeline.eval import top_k_logits, sample_next_token

# Filters
MAX_NAME = 15
MAX_INGR = 5
MAX_INGR_TOK = 20
MAX_STEP_TOK = 256

def eval_model(device, model, sampler, loss_compute, logit_modifier_fxn, token_sampler,
               print_every, max_len,
               user_items_df, top_k, pad_item_ix,
               max_name_len=15, ingr_map=None, 
               base_save_dir='', pad_ingr=None, ppx_only=False, **tensor_kwargs):
    """
    Run a single epoch

    Arguments:
        device {torch.device} -- Torch device on which to store/process data
        model {nn.Module} -- Model to be trained/run
        sampler {BatchSampler} -- Data sampler
        loss_compute {func} -- Function to compute loss for each batch
        logit_modifier_fxn {func} -- Function to modify a logit distr. and return a prob. distro
        token_sampler {str} -- "greedy" or "multinomial"
        print_every {int} -- Log loss every k iterations
        max_len {int} -- Maximum length / number of steps to unroll and predict

    Keyword Arguments:
        max_name_len {int} -- Maximum # timesteps to unroll to predict name (default: {15})
        ingr_map {dict} -- Map of ingredient ID -> ingredient raw name.
        pad_ingr {int} -- Index of pad item (default: {None})
        base_save_dir {str} -- Base folder in which to save experiments
        ppx_only {bool} -- Only calculate test perplexity (default: {False})
        **tensor_kwargs {torch.Tensor} -- Assorted tensors for fun and profit

    Returns:
        float -- Average loss across the epoch
    """
    start = datetime.now()
    results_dicts = []

    # Extract into tuples and list
    tensor_names, base_tensors = zip(*tensor_kwargs.items())

    # Iterate through batches in the epoch
    model.eval()
    with torch.no_grad():
        total_tokens = 0
        total_name_tokens = 0
        total_loss = 0.0
        total_name_loss = 0.0
        print_tokens = 0

        for i, batch in enumerate(tqdm(sampler.epoch_batches(), total=sampler.n_batches), 1):
            batch_users, items = [t.to(device) for t in batch]

            # Get prior items and their masks
            tuple_user_item_masks = [get_user_prior_item_mask(
                        user_ix=uix.item(), 
                        item_ix=iix.item(), 
                        user_items_df=user_items_df, 
                        top_k=top_k, 
                        pad_item_ix=pad_item_ix)
            for uix, iix in zip(batch_users, items)]
            user_items, user_item_masks = [
                torch.LongTensor(t).to(device) for t in zip(*tuple_user_item_masks)
            ]
            user_prior_names = torch.stack([
                torch.index_select(tensor_kwargs['name_tensor'], 0, u_i) for u_i in user_items
            ], dim=0).to(device)

            # Fill out batch information
            batch_map = dict(zip(
                tensor_names,
                get_batch_information_general(items, *base_tensors)
            ))
            use_ingr_embedding = batch_map['ingr_tensor'].size(-1) != MAX_INGR * MAX_INGR_TOK

            # Logistics
            this_batch_size = batch_map['steps_tensor'].size(0)
            this_batch_num_tokens = (batch_map['steps_tensor'] != PAD_INDEX).data.sum().item()
            this_batch_num_name_tokens = 0
            this_batch_num_name_tokens = (batch_map['name_tensor'] != PAD_INDEX).data.sum().item()
            name_targets = batch_map['name_tensor'][:, :-1]

            '''
            Teacher forcing - evaluate
            '''
            # Comparing out(token[t-1]) to token[t]
            (log_probs, _), (name_log_probs, _) = model.forward(
                device=device, inputs=(
                    batch_map['calorie_level_tensor'],
                    batch_map['name_tensor'],
                    batch_map['ingr_tensor']
                ),
                ingr_masks=batch_map['ingr_mask_tensor'],
                user_item_names=user_prior_names,
                user_items=user_items, user_item_masks=user_item_masks,
                targets=batch_map['steps_tensor'][:, :-1], max_len=max_len-1,
                start_token=START_INDEX, teacher_forcing=True,
                name_targets=name_targets,
                max_name_len=max_name_len-1,
                visualize=False
            )
            loss, name_loss = loss_compute(
                log_probs, batch_map['steps_tensor'][:, 1:],
                name_outputs=name_log_probs,
                name_targets=name_targets,
                norm=this_batch_size,
                model=model,
                clip=None
            )

            total_loss += loss
            total_name_loss += name_loss

            # Logging
            total_tokens += this_batch_num_tokens
            total_name_tokens += this_batch_num_name_tokens
            print_tokens += this_batch_num_tokens

            del log_probs, name_log_probs

            # Short-circuit if we only want to calculate test perplexity
            if ppx_only:
                if i % print_every == 0:
                    elapsed = datetime.now() - start
                    print("Epoch Step: {} LM Loss: {:.5f}; Name Loss: {:.5f}; Tok/s: {:.3f}".format(
                        i, loss / this_batch_size, name_loss / this_batch_size,
                        print_tokens / elapsed.seconds
                    ))
                    start = datetime.now()
                    print_tokens = 0
                continue

            '''
            Non-teacher-forcing - Generate!
            '''
            # Generates probabilities
            (log_probs, output_tokens, ingr_attns, prior_item_attns), \
            (name_log_probs, name_output_tokens) = model.forward(
                device=device, inputs=(
                    batch_map['calorie_level_tensor'],
                    batch_map['name_tensor'],
                    batch_map['ingr_tensor']
                ),
                ingr_masks=batch_map['ingr_mask_tensor'],
                user_item_names=user_prior_names,
                user_items=user_items, user_item_masks=user_item_masks,
                targets=batch_map['steps_tensor'][:, :-1], max_len=max_len-1,
                start_token=START_INDEX, teacher_forcing=False,
                logit_modifier_fxn=logit_modifier_fxn, token_sampler=token_sampler,
                visualize=True, max_name_len=max_name_len-1, name_targets=name_targets,
            )

            del log_probs, name_log_probs

            # Generated recipe
            calorie_levels, technique_strs, ingredient_strs, gold_strs, generated_strs, \
                prior_items, recipe_reprs = get_batch_generated_recipes(
                    batch_users=batch_users, batch_generated=output_tokens,
                    max_ingr=MAX_INGR, max_ingr_tok=MAX_INGR_TOK,
                    names_generated=name_output_tokens, ingr_map=ingr_map,
                    user_items_df=user_items_df, **batch_map
                )

            for ix in range(len(generated_strs)):
                # Create save location: test_i<item>_u<user>
                ii = items[ix].data.item()
                uu = batch_users[ix].data.item()
                sample_id = 'test_i{}_u{}'.format(ii, uu)
                trial_save_dir = os.path.join(base_save_dir, sample_id)
                if not os.path.exists(trial_save_dir):
                    os.mkdir(trial_save_dir)

                # Output tokens for heatmap axes
                out_indices = output_tokens[ix].detach().cpu().numpy().tolist()
                out_tokens = decode_ids(out_indices)
                trunc_indices = out_indices[:out_indices.index(END_INDEX)] \
                    if END_INDEX in out_indices else out_indices
                output_len = len(trunc_indices)
                output_techniques = [t for t in TECHNIQUES_LIST if t in generated_strs[ix]]
                results_dicts.append({
                    'u': uu,
                    'i': ii,
                    'generated': generated_strs[ix],
                    'n_tokens': output_len,
                    'generated_techniques': output_techniques,
                    'n_techniques': len(output_techniques)
                })

                # Save output
                with open(os.path.join(trial_save_dir, 'output.txt'), 'w+', encoding='utf-8') as wf:
                    wf.write(recipe_reprs[ix])

                # Ingredient Attention
                ingr_attentions = np.matrix([
                    a.squeeze().detach().cpu().numpy().tolist() for a in ingr_attns[ix]
                ]).T
                ingr_attn_df = pd.DataFrame(
                    ingr_attentions[:len(ingredient_strs[ix])],
                    index=ingredient_strs[ix], columns=out_tokens
                )
                ingr_attn_df = ingr_attn_df[ingr_attn_df.index != '']
                ingr_attn_df.to_pickle(
                    os.path.join(trial_save_dir, 'ingredient_attention.pkl')
                )

                # Prior Item Attention
                prior_item_attentions = np.matrix([
                    a.squeeze().detach().cpu().numpy().tolist() for a in prior_item_attns[ix]
                ]).T
                user_item_ids = user_items[ix].detach().cpu().numpy().tolist()
                user_item_names = [
                    pretty_decode(df_r.loc[iid, 'name_tokens']) if iid in df_r.index else ''
                    for iid in user_item_ids
                ]
                prior_item_attn_df = pd.DataFrame(
                    prior_item_attentions,
                    index=user_item_names, columns=out_tokens
                )
                prior_item_attn_df = prior_item_attn_df[prior_item_attn_df.index != '']
                prior_item_attn_df.to_pickle(
                    os.path.join(trial_save_dir, 'prior_item_attention.pkl')
                )

            if i % print_every == 0:
                elapsed = datetime.now() - start
                print("Epoch Step: {} LM Loss: {:.5f}; Name Loss: {:.5f}; Tok/s: {:.3f}".format(
                    i, loss / this_batch_size, name_loss / this_batch_size,
                    print_tokens / elapsed.seconds
                ))
                print('SAMPLE DECODED RECIPE:\n\n{}\n\n'.format(recipe_reprs[0]))
                start = datetime.now()
                print_tokens = 0

        # Reshuffle the sampler
        sampler.renew_indices()

        if total_name_tokens > 0:
            print('\nName Perplexity: {}'.format(
                np.exp(total_name_loss / float(total_name_tokens))
            ))

        # Store perplexity
        ppx = np.exp(total_loss / float(total_tokens))
        with open(os.path.join(base_save_dir, 'ppx.pkl'), 'wb') as wf:
            pickle.dump(ppx, wf)
        print('PERPLEXITY: {:.5f}'.format(
            ppx
        ))

        if not ppx_only:
            # Store recipe information -- generated string, # tokens (length), tech, # tech
            gen_df = pd.DataFrame(results_dicts)[[
                'u', 'i', 'generated', 'n_tokens', 'generated_techniques', 'n_techniques'
            ]]
            df_loc = os.path.join(base_save_dir, 'generated_df.pkl')
            gen_df.to_pickle(df_loc)
            print('Saved generation DF to {}'.format(
                df_loc
            ))
            print(gen_df.head(3))

'''
==== RUN CLUSTER ALL (PRIOR NAME ATTN)
python -m recipe_gen.models.user_item_pref.test --data-dir <DATA FOLDER> --model-path <MODEL PATH> --vocab-emb-size 300 --calorie-emb-size 5 --top-k 20 --nhid 256 --nlayers 2 --save-dir <OUTPUT FOLDER> --overwrite --batch-size 48 --ingr-emb-size 10  --ingr-gru --ingr-emb

==== RUN CLUSTER ALL (PRIOR ITEM ATTN)
python -m recipe_gen.models.user_item_pref.test --data-dir <DATA FOLDER> --model-path <MODEL PATH> --vocab-emb-size 300 --calorie-emb-size 5 --top-k 20 --nhid 256 --nlayers 2 --save-dir <OUTPUT FOLDER> --overwrite --batch-size 48 --ingr-emb-size 10  --ingr-gru --ingr-emb --item-emb --item-emb-size 50
'''
if __name__ == "__main__":
    import os
    import torch
    import argparse
    import torch.nn as nn
    import numpy as np
    import pandas as pd
    import pickle

    from datetime import datetime
    from functools import partial
    from itertools import chain

    from recipe_gen.utils import get_device, count_parameters
    from recipe_gen.language import N_TECHNIQUES, VOCAB_SIZE, START_INDEX, TECHNIQUES_LIST, decode_ids, PAD_INDEX
    from recipe_gen.pipeline import DataFrameDataset, BatchSampler
    from recipe_gen.pipeline.train import SimpleLossCompute
    from recipe_gen.pipeline.batch import load_full_data, pad_recipe_info, load_recipe_tensors

    from . import create_model

    from recipe_gen.pipeline.visualization import get_tag_ix_batches_from_data, get_batch_generated_recipes
    from recipe_gen.pipeline.eval import top_k_logits, sample_next_token, top_p_logits

    start = datetime.now()
    USE_CUDA, DEVICE = get_device()

    parser = argparse.ArgumentParser(description='Baseline for recipe generation (dynamic attn)')
    parser.add_argument('--data-dir', type=str, required=True, help='location of the data corpus')
    parser.add_argument('--vocab-emb-size', type=int, default=50, help='size of word embeddings')
    parser.add_argument('--calorie-emb-size', type=int, default=50, help='size of calorie embeddings')
    parser.add_argument('--ingr-emb-size', type=int, default=10, help='size of ingr embeddings')
    parser.add_argument('--item-emb-size', type=int, default=20, help='size of item embeddings')
    parser.add_argument('--top-k', type=int, default=20, help='top k prior item to attend on')

    parser.add_argument('--nhid', type=int, default=256, help='number of hidden units per layer')
    parser.add_argument('--nlayers', type=int, default=1, help='number of layers')
    parser.add_argument('--batch-size', '-b', type=int, default=24, help='batch size')

    parser.add_argument('--model-path', type=str, required=True,
        help='Path from which to retrieve saved model dict')
    parser.add_argument('--save-dir', type=str, required=True,
        help='Where to save model outputs, graphs, etc.')
    parser.add_argument('--overwrite', '-o', action='store_true', default=False,
        help='Overwrite existing outputs')
    parser.add_argument('--ingr-gru', action='store_true', default=False,
        help='Use BiGRU for ingredient encoding')
    parser.add_argument('--decode-name', action='store_true', default=False,
        help='Multi-task learn to decode name along with recipe')
    parser.add_argument('--ingr-emb', action='store_true', default=False,
        help='Use Ingr embedding in encoder')
    parser.add_argument('--shared-proj', action='store_true', default=False,
        help='Share projection layers for name and steps')
    parser.add_argument('--item-emb', action='store_true', default=False,
        help='Use Ingr embedding in encoder')

    parser.add_argument('--ppx-only', action='store_true', default=False,
        help='Only calculate perplexity (on full test set)')
    parser.add_argument('--n-samples', '-n', type=int, default=1e9, help='sample test items')
    args = parser.parse_args()

    # Reproducibility
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Args
    data_dir = args.data_dir
    vocab_emb_dim = args.vocab_emb_size
    calorie_emb_dim = args.calorie_emb_size
    ingr_emb_dim = args.ingr_emb_size
    item_emb_dim = args.item_emb_size
    top_k = args.top_k
    hidden_size = args.nhid
    n_layers = args.nlayers
    model_path = args.model_path
    save_dir = args.save_dir
    overwrite = args.overwrite
    ingr_gru = args.ingr_gru
    ingr_emb = args.ingr_emb
    item_emb = args.item_emb
    decode_name = args.decode_name
    batch_size = args.batch_size
    n_samples = args.n_samples
    shared_proj = args.shared_proj
    ppx_only = args.ppx_only

    '''
    Load data
    '''
    # Get the DFs
    train_df, valid_df, test_df, user_items_df, df_r, ingr_map = load_full_data(data_dir)
    n_items = len(df_r)
    print('{} - Data loaded.'.format(datetime.now() - start))

    # Pad recipe information
    N_INGREDIENTS = 0
    if ingr_emb:
        print('INGR EMBEDDING')
        n_ingredients_og = max(chain.from_iterable(df_r['ingredient_ids'].values)) + 1
        PAD_INGR = n_ingredients_og
        N_INGREDIENTS = n_ingredients_og + 1

    df_r = pad_recipe_info(
        df_r, max_name_tokens=MAX_NAME, max_ingredients=MAX_INGR,
        max_ingr_tokens=MAX_INGR_TOK, max_step_tokens=MAX_STEP_TOK
    )
    
    # Number of item
    N_ITEMS = len(df_r)
    # Num Item embedding with pad item
    NUM_ITEM_EMBEDDING = N_ITEMS + 1
    PAD_ITEM_INDEX = N_ITEMS

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

    # Name padding for item
    memory_tensor_map['name_tensor'] = torch.cat(
        [memory_tensor_map['name_tensor'],
        torch.LongTensor([[PAD_INDEX] * MAX_NAME]).to(DEVICE)]
    )

    if n_samples < len(test_df) and not ppx_only:
        sampled_test = test_df.sample(n=n_samples)
    else:
        sampled_test = test_df
    test_data = DataFrameDataset(sampled_test, ['u', 'i'])
    test_sampler = BatchSampler(test_data, batch_size)

    '''
    Create model
    '''
    model = create_model(
        vocab_emb_dim=vocab_emb_dim, calorie_emb_dim=calorie_emb_dim,
        item_emb_dim=item_emb_dim, n_items_w_pad=NUM_ITEM_EMBEDDING, hidden_size=hidden_size,
        n_layers=n_layers, dropout=0.0, max_ingr=MAX_INGR, max_ingr_tok=MAX_INGR_TOK,
        use_cuda=USE_CUDA, state_dict_path=model_path,
        ingr_gru=ingr_gru, decode_name=decode_name, ingr_emb=ingr_emb,
        num_ingr=N_INGREDIENTS, ingr_emb_dim=ingr_emb_dim, shared_projection=shared_proj,
        item_emb=item_emb
    )

    model_id = os.path.basename(model_path)[:-3]
    model_save_dir = os.path.join(save_dir, model_id)
    if not os.path.exists(model_save_dir):
        os.mkdir(model_save_dir)

    # Calculate loss
    criterion = nn.NLLLoss(reduction="sum", ignore_index=PAD_INDEX)
    loss_compute = SimpleLossCompute(criterion, None)
    
    # Sample via top-3
    logit_mod = partial(top_k_logits, k=3)
    sample_method = 'multinomial'

    eval_model(
        device=DEVICE,
        model=model,
        sampler=test_sampler,
        loss_compute=loss_compute,
        logit_modifier_fxn=logit_mod,
        token_sampler=sample_method,
        print_every=20,
        max_len=MAX_STEP_TOK,
        user_items_df=user_items_df,
        top_k=top_k,
        pad_item_ix=PAD_ITEM_INDEX,
        max_name_len=MAX_NAME,
        ingr_map=ingr_map,
        pad_ingr=PAD_INGR,
        base_save_dir=model_save_dir,
        ppx_only=ppx_only,
        **memory_tensor_map
    )
