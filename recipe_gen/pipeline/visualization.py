'''
-*- coding: utf-8 -*-

Visualization utilities

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
import gc
import torch
import numpy as np
import seaborn as sns

from datetime import datetime
from itertools import chain

from recipe_gen.pipeline import DataFrameDataset
from recipe_gen.pipeline.batch import get_batch_information_general, pad_name, pad_ingredients, pad_steps, get_ingr_mask
from recipe_gen.language import TOKENIZER, END_INDEX, TECHNIQUES_LIST, recipe_spec_repr

def get_state_dict(saved_model_loc):
    start = datetime.now()

    # Load the model
    models_folder = os.path.dirname(saved_model_loc)      # Where saved models lie
    model_name = os.path.basename(saved_model_loc)[:-3]   # Models saved as .pt
    state_dict_loc = os.path.join(models_folder, '{}.state_dict'.format(model_name))

    if os.path.exists(state_dict_loc):
        state_dict = torch.load(state_dict_loc)
        print('{} - Loaded {} state dict from {}'.format(
            datetime.now() - start, model_name, state_dict_loc
        ))
    else:
        model = torch.load(saved_model_loc)
        print('{} - Loaded {} model from {}'.format(
            datetime.now() - start, repr(model.__class__), saved_model_loc
        ))
        state_dict = model.state_dict()
        torch.save(state_dict, state_dict_loc)
        print('{} - Saved state dict to {} ({:,.3f} MB on disk)'.format(
            datetime.now() - start, state_dict_loc, os.path.getsize(state_dict_loc) / 1024 / 1024
        ))
        del model
        gc.collect()

    return state_dict, model_name

def get_tag_ix_batches_from_data(device, dfs, df_names, n_samples, user_items_df=None,
                                 get_prior_items_fn=None, get_prior_techniques_fn=None,
                                 *tensors):
    """
    Get full batches from training/testing data

    Arguments:
        device {torch.device} -- Device on which we store/process data
        dfs {list of pd.DataFrame} -- Source DFs
        df_names {list of str} -- Name/tag for each DF
        n_samples {list of int} -- Number of samples per DF
    
    Keyword Arguments:
        user_items_df {pd.DataFrame} -- DataFrame (default: {None})
        get_prior_items_fn {func} -- partial call to `get_user_prior_item_mask` (default: {None})
        get_prior_techniques_fn {func} -- partial call to `get_user_prior_techniques_mask`
            (default: {None})
        **tensors {torch.Tensor} -- Tensors to retrieve information for a batch
    """
    for df_name, df, n_samp in zip(df_names, dfs, n_samples):
        data = DataFrameDataset(df, ['u', 'i'])
        sample_indices = np.random.choice(len(data), n_samp, replace=False)
        for ix in sample_indices:
            # Get batch tensors from item indices
            users, items = [t.to(device) for t in data.get_tensor_batch([ix])]
            batch_yield = get_batch_information_general(
                items, *tensors
            )

            # Append the proper tensors
            batch_yield = [df_name, ix, users, items] + batch_yield

            # Construct prior items
            if get_prior_items_fn is not None:
                user_items, user_item_masks = [torch.LongTensor(t).to(device) for t in 
                    zip(*[get_prior_items_fn(
                        user_ix=uix.item(), 
                        item_ix=iix.item(), 
                        user_items_df=user_items_df)
                    for uix, iix in zip(users, items)])]
                batch_yield.extend([user_items, user_item_masks])
            
            # Construct prior techniques per user
            if get_prior_techniques_fn is not None:
                user_prior_technique_masks = torch.stack([get_prior_techniques_fn(
                    user_ix=uix.item(), item_ix=iix.item(),
                    user_items_df=user_items_df,
                    device=device
                ) for uix, iix in zip(users, items)], dim=0)
                batch_yield.append(user_prior_technique_masks)

            yield batch_yield

def get_custom_tag_ix_batches(device, custom_spec_dir, max_name, max_ingr, max_ingr_tok, max_steps):
    for custom_r in [f for f in os.listdir(custom_spec_dir) if f.endswith('.txt')]:
        with open(os.path.join(custom_spec_dir, custom_r), 'r+') as cr_file:
            user, name, ingrs = cr_file.readlines()[:3]
        
        # Process inputs directly
        ix = int(custom_r[:-4])

        # User
        user_ix = int(user.strip())

        # Item
        item_ix = -1

        # Name tokens
        name_ix = pad_name(TOKENIZER.convert_tokens_to_ids(
            TOKENIZER.tokenize(name.strip().lower())
        ), max_name)

        # Ingredient tokens
        ingr_ix = pad_ingredients([
            TOKENIZER.convert_tokens_to_ids(TOKENIZER.tokenize(ingr.strip())) for ingr in
            ingrs.strip().lower().split(',') if ingr != ''
        ], max_ingr, max_ingr_tok)

        # Ingredient mask
        ingr_mask = get_ingr_mask(ingr_ix, max_ingr)

        # Flatten ingredients
        ingr_ix = list(chain.from_iterable(ingr_ix))

        # Gold steps not provided in the custom case
        steps_ix = pad_steps([], max_steps)

        # Get batch tensors
        users, items, batch_names, batch_ingr, batch_ingr_mask, batch_steps = [
            torch.LongTensor([batch_list]).to(device) for batch_list in 
            [user_ix, item_ix, name_ix, ingr_ix, ingr_mask, steps_ix]
        ]
        yield 'custom', ix, users, items, \
                    batch_names, batch_ingr, batch_ingr_mask, batch_steps

def get_batch_generated_recipes(batch_users, batch_generated, max_ingr=5, max_ingr_tok=20,
                                user_items_df=None, names_generated=None, ingr_map=None,
                                **tensor_kwargs):
    # Inputs
    calorie_batch = tensor_kwargs['calorie_level_tensor'].cpu().data.numpy().tolist()  # B, N
    techniques_batch = tensor_kwargs['technique_tensor'].cpu().data.numpy().tolist()  # B, N
    use_ingredient_embeddings = tensor_kwargs['ingr_tensor'].size(-1) != max_ingr * max_ingr_tok
    if use_ingredient_embeddings:
        ingr_batch = tensor_kwargs['ingr_tensor'].cpu().data.numpy().tolist()
    else:
        ingr_batch = tensor_kwargs['ingr_tensor'].view(-1, max_ingr, max_ingr_tok) \
            .cpu().data.numpy().tolist()  # B, N, N

    # Gold steps
    gold_steps_batch = tensor_kwargs['steps_tensor'].cpu().data.numpy().tolist()  # B, N

    # Generated output
    generated_steps_batch = batch_generated.cpu().data.numpy().tolist()  # B, T

    # User stuff
    users_batch = batch_users.cpu().data.numpy().tolist()  # B, 1

    # Names
    name_input_batch = tensor_kwargs['name_tensor'].cpu().data.numpy().tolist()  # B, N

    # Generated names
    names_gen_batch = None
    if names_generated is not None:
        names_gen_batch = names_generated.cpu().data.numpy().tolist()  # B, max_name

    # Cut off at END token
    calorie_levels = []
    technique_strs = []
    ingredient_strs = []
    gold_strs = []
    generated_strs = []
    prior_items = []
    recipe_reprs = []

    for i, generated in enumerate(generated_steps_batch):
        # Clean up generated text once it hits END_INDEX
        generated = generated[:generated.index(END_INDEX)] if END_INDEX in generated else generated

        # Decode into strings
        calorie_levels.append(calorie_batch[i])
        techniques = [
            TECHNIQUES_LIST[it] for it in techniques_batch[i] if it < len(TECHNIQUES_LIST)
        ]
        technique_strs.append(techniques)

        # User prior items
        if user_items_df is not None:
            try:
                user_prior_items = user_items_df.loc[users_batch[i], 'items']
            except KeyError:
                user_prior_items = []
            prior_items.append(user_prior_items)

        # Get recipe representation
        if use_ingredient_embeddings:
            batch_ingr_input = [ingr_map[x] for x in ingr_batch[i]]
        else:
            batch_ingr_input = ingr_batch[i]
        ingr_strs, steps_str, output_str, full_recipe_str = recipe_spec_repr(
            calorie_level=calorie_batch[i],
            techniques=techniques,
            ingr_tokens=batch_ingr_input,
            input_tokens=gold_steps_batch[i],
            output_tokens=generated,
            name_tokens=names_gen_batch[i] if names_gen_batch is not None else None,
            original_name_tokens=name_input_batch[i]
        )

        ingredient_strs.append(ingr_strs)
        gold_strs.append(steps_str)
        generated_strs.append(output_str)
        recipe_reprs.append(full_recipe_str)

    return calorie_levels, technique_strs, ingredient_strs, gold_strs, generated_strs, prior_items, \
        recipe_reprs

def plot_df_heatmap(plot_df, title, save_loc=None, dpi=300, tight=True, label_rotation=40,
                    transparent=True, font_scale=2, **kwargs):
    from matplotlib import rcParams
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = ['Arial']
    import matplotlib.pyplot as plt

    # Get plot dimensions
    x_pixels = dpi * len(plot_df.columns)
    y_pixels = dpi * len(plot_df)
    w = x_pixels / dpi
    h = y_pixels / dpi

    # Plot heatmap
    sns.set(font_scale=font_scale)
    sns.set_style('ticks')
    kwargs.setdefault('xticklabels', list(plot_df.columns))
    kwargs.setdefault('yticklabels', list(plot_df.index))
    kwargs.setdefault('cbar', False)
    kwargs.setdefault('cbar_kws', dict(use_gridspec=False, location="top"))
    g = sns.heatmap(
        plot_df, **kwargs
    )
    g.figure.set_size_inches(w, h)
    g.set_title(title)
    plt.yticks(rotation=0)
    g.set_xticklabels(g.get_xticklabels(), rotation=label_rotation)
    g.set_aspect('equal', 'box')
    if tight:
        plt.tight_layout()

    # SAVE
    if save_loc is None:
        plt.show()
    else:
        g.figure.savefig(
            save_loc,
            transparent=transparent,
            dpi=dpi,
            format=save_loc.split('.')[-1]
        )

    # Cleanup
    g.clear()
    plt.clf()
    plt.close('all')

def truncate_attention_map(attn_df):
    cols = attn_df.columns
    end_token_index = list(cols).index('</R>') if '</R>' in cols else len(cols)
    return attn_df.iloc[:, :end_token_index]
