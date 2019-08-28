'''
-*- coding: utf-8 -*-

Utilities for loading data and preprocessing

@inproceedings{majumder2019emnlp,
  title={Generating Personalized Recipes from Historical User Preferences},
  author={Majumder, Bodhisattwa Prasad* and Li, Shuyang* and Ni, Jianmo and McAuley, Julian},
  booktitle={EMNLP},
  year={2019}
}

Copyright Shuyang Li & Bodhisattwa Majumder
License: GNU GPLv3
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import pandas as pd
import numpy as np
import os

from datetime import datetime
from itertools import chain
from recipe_gen.pipeline import DataFrameDataset
from recipe_gen.language import PAD_INDEX, pretty_decode, pretty_decode_tokens, N_TECHNIQUES, PAD_TECHNIQUE_INDEX

def load_full_data(dataset_folder,
        base_splits=['train', 'valid_new', 'test_new']):
    """
    Load full data (including recipe information, user information)

    Arguments:
        dataset_folder {str} -- Location of data

    Keyword Arguments:
        base_splits {list} -- Train/Validation/Test file base strs

    Returns:
        pd.DataFrame -- Training interactions DataFrame
        pd.DataFrame -- Validation interactions DataFrame
        pd.DataFrame -- Test interactions DataFrame
        pd.DataFrame -- Items per user DataFrame
        pd.DataFrame -- DataFrame containing recipe information
    """
    start = datetime.now()

    # Interactions
    train_df, valid_df, test_df = [
        pd.read_pickle(os.path.join(dataset_folder, 'interactions_{}.pkl'.format(split)))
        for split in base_splits
    ]
    total_memory = sum([df.memory_usage(deep=True).sum() for df in [train_df, valid_df, test_df]])
    print('{} - Loaded {:,} training interactions, {:,} validation, {:,} test ({:,.3f} MB total memory)'.format(
        datetime.now() - start, len(train_df), len(valid_df), len(test_df), total_memory / 1024 / 1024
    ))

    # User items
    user_items = pd.read_pickle(os.path.join(dataset_folder, 'user_rep.pkl'))
    max_ints = user_items['n_items'].max()
    print('{} - Loaded items for {:,} users, {:,} maximum interactions/user ({:,.3f} MB total memory)'.format(
        datetime.now() - start, len(user_items), max_ints, user_items.memory_usage(deep=True).sum() / 1024 / 1024
    ))

    # Recipes
    df_r = pd.read_pickle(os.path.join(dataset_folder, 'recipes.pkl'))
    print('{} - Loaded {:,} recipes ({:,.3f} MB total memory)'.format(
        datetime.now() - start, len(df_r), df_r.memory_usage(deep=True).sum() / 1024 / 1024
    ))

    # Ingredient map
    df_ingr = pd.read_pickle(os.path.join(dataset_folder, 'ingr_map.pkl'))
    ingr_ids, ingr_names = zip(*df_ingr.groupby(['id'], as_index=False)['replaced'].first().values)
    ingr_map = dict(zip(ingr_ids, ingr_names))
    ingr_map[max(ingr_ids) + 1] = ''
    print('{} - Loaded map for {:,} unique ingredients'.format(
        datetime.now() - start, len(ingr_map)
    ))

    return train_df, valid_df, test_df, user_items, df_r, ingr_map

def pad_name(name_tokens, max_name_tokens=15):
    return name_tokens + [PAD_INDEX]*(max_name_tokens - len(name_tokens))

def pad_steps(step_tokens, max_step_tokens=256):
    # Pad steps to maximum step length
    return step_tokens + [PAD_INDEX]*(max_step_tokens - len(step_tokens))

def pad_ingredients(ingredient_tokens, max_ingredients=20, max_ingr_tokens=20):
    # Pad ingredients to maximum ingredient length
    new_tokens = [
        i[:max_ingr_tokens] + [PAD_INDEX]*(max_ingr_tokens - len(i[:max_ingr_tokens])) for
        i in ingredient_tokens[:max_ingredients]
    ]

    # Pad with empty ingredients
    new_tokens += [[PAD_INDEX]*max_ingr_tokens] *\
        (max_ingredients - len(ingredient_tokens[:max_ingredients]))
    return new_tokens

def pad_ingredient_ids(ingredient_ids, max_ingredients, pad_ingredient):
    # Pad ingredients to maximum ingredient length
    return ingredient_ids[:max_ingredients] + \
        [pad_ingredient]*(max_ingredients - len(ingredient_ids[:max_ingredients]))

def get_ingr_mask(ingredient_tokens, max_ingredients=20):
    return [1]*len(ingredient_tokens[:max_ingredients]) + \
        [0]*(max_ingredients - len(ingredient_tokens[:max_ingredients]))

def get_technique_embedding_indices(technique_mask):
    technique_onehot = np.array(technique_mask)
    return np.arange(N_TECHNIQUES) * technique_onehot + \
        (technique_onehot == 0) * PAD_TECHNIQUE_INDEX

def pad_recipe_info(df_r, max_name_tokens=15, min_ingredients=3, max_ingredients=20,
                    max_ingr_tokens=20, max_step_tokens=256):
    """
    Pads relevant recipe tokenized representations

    Arguments:
        df_r {pd.DataFrame} -- Recipe information DataFrame, containing the columns:
                                name_tokens
                                step_tokens
                                ingredient_tokens

    Keyword Arguments:
        max_name_tokens {int} -- Maximum tokens in a recipe name (default: {15})
        min_ingredients {int} -- Minimum # ingredients in a recipe (default: {3})
        max_ingredients {int} -- Maximum # ingredients in a recipe (default: {20})
        max_ingr_tokens {int} -- Maximum # tokens in an ingredient (default: {20})
        max_step_tokens {int} -- Maximum # steps in a recipe (default: {256})
    
    Returns:
        pd.DataFrame -- Padded recipe information DataFrame
    """
    start = datetime.now()

    # Pad name
    df_r['name_tokens'] = df_r['name_tokens'].agg(lambda n: pad_name(n, max_name_tokens))
    print('{} - Padded names to maximum {} tokens'.format(
        datetime.now() - start, max_name_tokens
    ))

    # Pad steps
    df_r['steps_tokens'] = df_r['steps_tokens'].agg(lambda s: pad_steps(s, max_step_tokens))
    print('{} - Padded steps to maximum {} tokens'.format(
        datetime.now() - start, max_step_tokens
    ))

    # Clip ingredients
    df_r['ingredient_tokens'] = df_r['ingredient_tokens'].agg(lambda i:
        i[:np.random.randint(min_ingredients, max_ingredients+1)]
    )
    df_r['ingredient_ids'] = df_r['ingredient_ids'].agg(lambda i:
        i[:np.random.randint(min_ingredients, max_ingredients+1)]
    )
    print('{} - Clipped ingredients randomly between {} and {} incidents'.format(
        datetime.now() - start, min_ingredients, max_ingredients
    ))

    # Pad ingredients + generate mask
    df_r['ingredient_mask'] = df_r['ingredient_tokens'].agg(lambda i: get_ingr_mask(i, max_ingredients))
    df_r['ingredient_tokens'] = df_r['ingredient_tokens'].agg(
        lambda i: pad_ingredients(i, max_ingredients, max_ingr_tokens)
    )
    print('{} - Padded ingredients to maximum {} tokens and {} ingredients'.format(
        datetime.now() - start, max_ingr_tokens, max_ingredients
    ))

    # Pad ingredient IDs
    df_r['ingredient_id_mask'] = df_r['ingredient_ids'].agg(lambda i: get_ingr_mask(i, max_ingredients))
    n_ingredients_og = max(chain.from_iterable(df_r['ingredient_ids'].values)) + 1
    pad_ingr = n_ingredients_og
    n_ingredients = n_ingredients_og + 1
    df_r['ingredient_ids'] = df_r['ingredient_ids'].agg(
        lambda i: pad_ingredient_ids(i, max_ingredients, pad_ingr)
    )
    print('{} - Padded ingredient IDs to maximum {} ingredients w pad ingredient {}'.format(
        datetime.now() - start, max_ingredients, pad_ingr
    ))

    # Pad techniques mask - we will never attend on the pad technique
    df_r['techniques_mask'] = df_r['techniques'].agg(lambda x: x + [sum(x) == 0])
    df_r['techniques'] = df_r['techniques_mask'].agg(get_technique_embedding_indices)
    print('{} - Processed techniques and the associated masks'.format(
        datetime.now() - start
    ))

    return df_r

def load_recipe_tensors(df_r, device,
                        cols=['name_tokens', 'ingredient_tokens', 'ingredient_mask', 'steps_tokens'],
                        types=[torch.LongTensor, torch.LongTensor, torch.LongTensor, torch.LongTensor]):
    """
    Loads recipe component data into the GPU as tensors

    Arguments:
        df_r {pd.DataFrame} -- Recipe information DataFrame
        device {torch.Device} -- Device onto which to load the data
        cols {list} -- List of columns from which to retrieve tensors
        types {list} -- List of types of tensors to create

    Returns:
        list -- Tensor with tokens for each column specified
    """
    start = datetime.now()

    df_r = df_r.set_index('i').sort_index()
    print('{} - Sorted recipes DF by recipe ID'.format(datetime.now() - start))

    created_tensors = []
    max_col_len = max(len(k) for k in cols if k is not None)

    # Load only relevant tensors to GPU
    total_size = 0
    for col, tens_type in zip(cols, types):
        # Short-circuit
        if col is None:
            created_tensors.append(None)
            continue

        # Load column tensor
        if col == 'ingredient_tokens':
            flattened_ingredients = df_r['ingredient_tokens'].agg(
                lambda i_list: list(chain.from_iterable(i_list))
            ).values.tolist()
            col_tensor = tens_type(flattened_ingredients).to(device)
        else:
            col_tensor = tens_type(df_r[col].values.tolist()).to(device)
        created_tensors.append(col_tensor)
        col_size = col_tensor.element_size() * col_tensor.nelement()
        total_size += col_size
        print('{} {} ({:,.3f} MB)'.format(
            '{}:'.format(col).ljust(max_col_len + 1),
            ' x '.join(['{:,}'.format(s) for s in col_tensor.size()]).ljust(15),
            col_size / 1024 / 1024
        ))

    print('{} - Loaded tensors to GPU - TOTAL {:,.3f} MB'.format(
        datetime.now() - start, total_size / 1024 / 1024
    ))
    return created_tensors

def get_batch_information(i, name_tensor, ingredients_tensor, ingredients_mask_tensor,
                          steps_tensor, device):
    """
    Get information about a recipe and user for each row in a batch
    
    Arguments:
        i {torch.LongTensor} -- Item IDs in batch [n_batch]
        name_tensor {torch.LongTensor} -- Name tokens per recipe [n_items x max_name_tokens]
        ingredients_tensor {torch.LongTensor} -- Flattened ingredients
                                                [n_items x (max_ingrs x max_ingr_tokens)]
        ingredients_mask_tensor {torch.LongTensor} -- Which ingredients are present positionally
                                                [n_items x max_ingredients]
        steps_tensor {torch.LongTensor} -- Steps per recipe [n_items x max_step_tokens]
        device {torch.Device} -- Device on which to load data

    Returns:
        torch.LongTensor -- Name [n_batch x max_name_tokens]
        torch.LongTensor -- Ingredients (flattened) [n_batch x (max_ingr x max_ingr_tokens)]
        torch.LongTensor -- Ingredient mask [n_batch x max_ingr]
        torch.LongTensor -- Steps per recipe [n_batch x max_step_tokens]
    """
    # RECIPE INFORMATION - NAMES, INGREDIENTS, STEPS
    batch_names = torch.index_select(name_tensor, 0, i)
    batch_ingredients = torch.index_select(ingredients_tensor, 0, i)
    batch_ingr_mask = torch.index_select(ingredients_mask_tensor, 0, i)
    batch_steps = torch.index_select(steps_tensor, 0, i)

    return batch_names, batch_ingredients, batch_ingr_mask, batch_steps

def get_recipe_repr(r_name_tensor, r_ingr_tensor, r_steps_tensor):
    """
    Gets recipe representation
    
    Arguments:
        r_name_tensor {torch.LongTensor} -- Name token indices for one recipe
        r_ingr_tensor {torch.LongTensor} -- Ingredient token indices for one recipe
        r_steps_tensor {torch.LongTensor} -- Step token indices for one recipe
    
    Returns:
        str -- Recipe name
        list -- Recipe ingredient strings
        str -- Recipe steps
        str -- Formatted recipe representation
    """
    # Decode name
    name_str = pretty_decode(r_name_tensor.cpu().numpy().tolist())
    recipe_str = 'RECIPE: `{}\n\n`'.format(name_str)

    # Decode ingredients
    ingr_strings = [
        pretty_decode(l.tolist()) for l in np.array_split(r_ingr_tensor.cpu().numpy().tolist(), 20)
    ]
    recipe_str += 'INGREDIENTS:\n--{}\n\n'.format('\n--'.join(filter(None, ingr_strings)))

    # Input steps
    steps_str = ''
    if r_steps_tensor is not None:
        steps_str = pretty_decode(r_steps_tensor.cpu().numpy().tolist())
        recipe_str += 'STEPS:\n{}'.format(steps_str)

    return name_str, ingr_strings, steps_str, recipe_str

def get_batch_information_general(i, *tensors):
    """
    Get information about a recipe and user for each row in a batch
    
    Arguments:
        i {torch.LongTensor} -- Item IDs in batch [n_batch]
    
    Positional Arguments:
        *tensors {torch.LongTensor} -- Tokens per recipe [n_items x max_t_tokens]

    Returns:
        list of torch.Tensors
    """
    return [torch.index_select(t, 0, i) if t is not None else None for t in tensors]

def get_user_prior_items(user_ix, item_ix, user_items_df, top_k=None, get_ratings=False):
    # Get indices of prior items
    try:
        # Wrap in a list because .loc returns a reference. We do NOT want to mutate it!! -shu
        user_items = list(user_items_df.loc[user_ix, 'items'])
        if get_ratings:
            user_item_ratings = list(user_items_df.loc[user_ix, 'ratings'])
    except KeyError:
        user_items = []
    
    # Remove current item
    try:
        pop_idx = user_items.index(item_ix)
        user_items.remove(item_ix)
        if get_ratings:
            user_item_ratings.pop(pop_idx)
            assert len(user_items) == len(user_item_ratings)
    except:
        ...

    if top_k is not None:
        return user_items[-top_k:], (user_item_ratings[-top_k:] if get_ratings else None)
    return user_items, (user_item_ratings if get_ratings else None)

def get_user_prior_item_mask(user_ix, item_ix, user_items_df, top_k, pad_item_ix, get_ratings=False):
    user_items, user_item_ratings = get_user_prior_items(
        user_ix, item_ix, user_items_df, top_k, get_ratings
    )

    # Pad to top k
    user_items_padded = user_items + (top_k - len(user_items)) * [pad_item_ix]
    if get_ratings:
        user_item_ratings_padded = user_item_ratings + (top_k - len(user_items)) * [0]

    # Mask
    user_items_mask = [1] * len(user_items) + (top_k - len(user_items)) * [0]

    if get_ratings:
        return user_items_padded, user_items_mask, user_item_ratings_padded
    else:
        return user_items_padded, user_items_mask

def get_user_prior_techniques_mask(user_ix, item_ix, user_items_df, tech_mask_tensor, device,
                                   normalize=True):
    """
    Gets a mask indicating relative frequency of techniques encountered by a user in their history

    Arguments:
        user_ix {int} -- User index
        item_ix {int} -- Current item index
        user_items_df {pd.DataFrame} -- DataFrame with 'items' column containing historical items
            a user has interacted with
        tech_mask_tensor {torch.Tensor} -- Tensor of technique masks (techniques present in a recipe)
        device {torch.device} -- Torch device

    Returns:
        torch.FloatTensor -- Relative frequency history
    """
    # Get prior item indices
    user_items, _ = get_user_prior_items(user_ix, item_ix, user_items_df, top_k=None)

    # Get technique profiles for all prior recipes user has interacted with
    user_prior_techniques_by_item = torch.index_select(
        tech_mask_tensor, 0, torch.LongTensor(user_items).to(device)
    )

    # Sum to get technique frequency for user
    user_prior_techniques = torch.sum(user_prior_techniques_by_item, dim=0).to(dtype=torch.float32)
    if normalize:  # Normalize to 1
        user_prior_techniques = F.normalize(user_prior_techniques, p=1, dim=0)
    return user_prior_techniques
