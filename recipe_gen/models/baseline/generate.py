'''
-*- coding: utf-8 -*-

Utilities for generating text from the baseline model.

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

from functools import partial
from recipe_gen.language import TECHNIQUES_LIST, START_INDEX, END_INDEX, pretty_decode, recipe_spec_repr
from recipe_gen.pipeline.batch import get_batch_information_general
from recipe_gen.pipeline.visualization import get_batch_generated_recipes
from recipe_gen.pipeline.eval import top_k_logits, sample_next_token

def decode_single(device, model, sampler, **kwargs):
    # Get single example from test set
    batch = sampler.dataset.get_tensor_batch([0])
    batch_reprs = \
        decode_batch(
            device=device, model=model, batch=batch, **kwargs
        )

    # Return the decoded values out of list
    return [item[0] for item in batch_reprs if item]

def decode_batch(device, model, batch,
                 max_len, max_name_len=None, teacher_forcing=False,
                 logit_modifier_fxn=partial(top_k_logits, k=0), token_sampler='greedy',
                 ingr_map=None, max_ingr=5, max_ingr_tok=20,
                 **tensor_kwargs):
    # Extract into tuples and list
    tensor_names, base_tensors = zip(*tensor_kwargs.items())

    # Send tensors in batch to device
    batch_users, items = [t.to(device) for t in batch]

    # Fill out batch information
    batch_map = dict(zip(
        tensor_names,
        get_batch_information_general(items, *base_tensors)
    ))

    max_name = max_name_len - 1
    name_targets = batch_map['name_tensor'][:, :-1]

    # Run the model in eval mode
    model.eval()
    with torch.no_grad():
        # Generates probabilities
        (log_probs, output_tokens, ingr_attns_for_plot), \
        (name_log_probs, name_output_tokens) = model.forward(
            device=device, inputs=(
                batch_map['calorie_level_tensor'],
                batch_map['name_tensor'],
                batch_map['ingr_tensor']
            ),
            ingr_masks=batch_map['ingr_mask_tensor'],
            targets=batch_map['steps_tensor'][:, :-1],
            max_len=max_len-1,
            start_token=START_INDEX,
            teacher_forcing=teacher_forcing,
            logit_modifier_fxn=logit_modifier_fxn,
            token_sampler=token_sampler,
            visualize=True,
            max_name_len=max_name,
            name_targets=name_targets,
        )

    return get_batch_generated_recipes(
        batch_users=batch_users, batch_generated=output_tokens,
        max_ingr=max_ingr, max_ingr_tok=max_ingr_tok, user_items_df=None,
        names_generated=name_output_tokens, ingr_map=ingr_map, **batch_map
    )
