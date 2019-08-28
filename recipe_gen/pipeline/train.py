'''
-*- coding: utf-8 -*-

General training pipeline execution

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
import torch.nn as nn
import numpy as np
import pandas as pd

from datetime import datetime

from recipe_gen.utils import count_parameters
from recipe_gen.language import PAD_INDEX, recipe_repr

# define utilities for training
class SimpleLossCompute:
    """A simple loss compute and train function."""

    def __init__(self, criterion, opt=None):
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm, model, clip, **kwargs):
        lm_loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1))
        
        name_x = kwargs.get('name_outputs', None)
        name_y = kwargs.get('name_targets', None)
        if name_x is not None:
            name_loss = self.criterion(name_x.contiguous().view(-1, name_x.size(-1)),
                              name_y.contiguous().view(-1))
        else:
            name_loss = 0.0

        loss = lm_loss + name_loss
        loss = loss / norm
        if self.opt is not None:
            self.opt.zero_grad()

            loss.backward()    
            
            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs
            if clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), clip)

            self.opt.step()

        return lm_loss.data.item(), name_loss.data.item() if name_x is not None else 0.0

# BASE FUNCTION SIGNATURE: run_epoch
def _run_epoch(device, model, sampler, loss_compute, print_every, max_len, clip, teacher_forcing,
               **kwargs):
    ...

# BASE FUNCTION SIGNATURE: greedy_decode_single
def _greedy_decode_single(model, dataloader, device, **kwargs):
    ...

def train_model(device, model, train_sampler, val_sampler, test_sampler,
                num_epochs, lr, exp_name,
                partial_run_epoch, partial_decode_single,
                lr_annealing_rate=1.0, n_teacher_forcing=1, save_folder=''):
    """
    Training pipeline for model

    Arguments:
        device {torch.device} -- Device on which data is stored/operated
        model {nn.Module} -- Model (with `forward` function)
        train_sampler {BatchSampler} -- Sampler to produce tensor batches of training data
        val_sampler {BatchSampler} -- Sampler to produce tensor batches of validation data
        test_sampler {BatchSampler} -- Sampler to produce tensor batches of testing data
        num_epochs {int} -- Maximum # epochs
        lr {float} -- Starting learning rate
        exp_name {str} -- Experiment name
        partial_run_epoch {func} -- `run_epoch` function, with train/epoch-invariant arguments already 
            filled in via `functools.partial`
        partial_decode_single {func} -- `decode_single` function, w/ train/epoch-invariant
            arguments already filled in via `functools.partial`

    Keyword Arguments:
        lr_annealing_rate {float} -- Scale learning rate upon validation ppx increasing (default: {1.0})
        n_teacher_forcing {int} -- Number of epochs to conduct teacher forcing (default: {1})
        save_folder {str} -- Location to which to save models (default: {''})

    Returns:
        list -- validation perplexities across epochs
        float -- final test perplexity
    """
    start_train = datetime.now()
    current_lr = lr

    # Making sure the folder exists
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    # Model serialization
    model_loc = os.path.join(save_folder, 'model_{}'.format(exp_name))
    print('{} - Training model with {} epochs of teacher forcing, starting LR {}, saving to {}'.format(
        datetime.now() - start_train,
        n_teacher_forcing,
        current_lr,
        model_loc
    ))

    # optionally add label smoothing; see the Annotated Transformer
    criterion = nn.NLLLoss(reduction="sum", ignore_index=PAD_INDEX)
    optim = torch.optim.Adam(model.parameters(), lr=current_lr, weight_decay=1e-5)

    dev_perplexities = []
    best_perplexity_thus_far = 1e56
    best_model_save_loc = ''
    for epoch in range(num_epochs):
        model.train()
        # curriculum learning with teacher forcing
        force_teaching = epoch <= n_teacher_forcing
        print('[{} - Epoch {}] START EPOCH{} | {} model has {:,} parameters'.format(
            datetime.now() - start_train, epoch,
            ' (Teacher Forcing for {} epochs)'.format(n_teacher_forcing) if force_teaching else '',
            exp_name, count_parameters(model)
        ))
        train_perplexity = partial_run_epoch(
            device=device,
            model=model,
            sampler=train_sampler,
            loss_compute=SimpleLossCompute(criterion, optim),
            teacher_forcing=force_teaching
        )

        print('Train perplexity: %f' % train_perplexity)
        
        # Save improved model
        candidate_loc = model_loc + 'CANDIDATE.pt'
        torch.save(model.state_dict(), candidate_loc)
        print('[{} - Epoch {}] Saved candidate model to {}'.format(
            datetime.now() - start_train, epoch, candidate_loc
        ))

        # VALIDATION
        model.eval()
        with torch.no_grad():     
            dev_perplexity = partial_run_epoch(
                device=device,
                model=model,
                sampler=val_sampler,
                loss_compute=SimpleLossCompute(criterion, None),
                teacher_forcing=True
            )

            # Early stopping - compare with prior perplexity
            prior_perplexity = dev_perplexities[-1] if dev_perplexities else 9.9e12
            print("Validation perplexity: %f" % dev_perplexity)
            dev_perplexities.append(dev_perplexity)

            # Pick the first recipe and decode it as an example
            decode_output = partial_decode_single(
                device=device, model=model, sampler=val_sampler
            )
            recipe_str = decode_output[-1]
            print('[{} - Epoch {}] Decoded recipe from validation:'.format(
                datetime.now() - start_train, epoch
            ))
            print(recipe_str)

            # If validation perplexity doesn't go down, we either anneal or stop
            if dev_perplexity > prior_perplexity:
                if lr_annealing_rate == 0.0 or current_lr < 1e-12:  # Early stoppage
                    print('[{} - Epoch {}] EARLY STOPPAGE'.format(
                        datetime.now() - start_train, epoch
                    ))
                    break
                elif lr_annealing_rate != 1.0:                      # No annealing if 1.0
                    new_lr = current_lr * lr_annealing_rate
                    print('[{} - Epoch {}] Annealing: changed LR from {:.5f} to {:.5f}'.format(
                        datetime.now() - start_train, epoch, current_lr, new_lr
                    ))
                    current_lr = new_lr
                    for param_group in optim.param_groups:
                        param_group['lr'] = current_lr
                    continue

            # Save improved model
            if dev_perplexity < best_perplexity_thus_far:
                best_perplexity_thus_far = min(best_perplexity_thus_far, dev_perplexity)
                best_model_save_loc = model_loc + '_e{}.pt'.format(epoch)
                torch.save(model.state_dict(), best_model_save_loc)
                print('[{} - Epoch {}] Saved model to {}'.format(
                    datetime.now() - start_train, epoch, best_model_save_loc
                ))

    # TESTING
    model.load_state_dict(torch.load(best_model_save_loc))
    model = model.to(device)
    print('{} - Loaded best model from {}'.format(
        datetime.now() - start_train, best_model_save_loc
    ))
    model.eval()
    with torch.no_grad():
        test_perplexity = partial_run_epoch(
            device=device,
            model=model,
            sampler=test_sampler,
            loss_compute=SimpleLossCompute(criterion, None),
            teacher_forcing=True
        )
        print("Test perplexity: {:.4f}".format(test_perplexity))

    # Pick the first recipe and decode it as an example
    decode_output = partial_decode_single(
        device=device, model=model, sampler=test_sampler
    )
    recipe_str = decode_output[-1]
    print('[{} - Epoch {}] Decoded recipe from test set:'.format(
        datetime.now() - start_train, epoch
    ))
    print(recipe_str)

    return dev_perplexities, test_perplexity
