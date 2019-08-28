'''
-*- coding: utf-8 -*-

Model classes for baseline generative model, with attention over ingredients and names

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

from functools import partial
from datetime import datetime

from recipe_gen.pipeline.eval import top_k_logits, sample_next_token
from recipe_gen.models import Encoder, Decoder, EncoderDecoder, BahdanauAttention, NameDecoder
from recipe_gen.language import START_INDEX, PAD_INDEX, VOCAB_SIZE, N_TECHNIQUES
from recipe_gen.utils import count_parameters

class BaselineEncoderDecoder(EncoderDecoder):
    """
    Encoder-Decoder with dynamic attention from input
    """
    def forward(self, device, inputs, targets, ingr_masks, 
                max_len, start_token, teacher_forcing,
                logit_modifier_fxn=partial(top_k_logits, k=0), token_sampler='greedy',
                visualize=False, name_targets=None, max_name_len=None):
        """
        Encode target recipe and use it to return 

        Arguments:
            device {torch.device} -- Device on which to hold/process data
            inputs {tuple} -- Batch input tensors
            targets {torch.Tensor} -- Gold label step tokens
            ingr_masks {torch.Tensor} -- Onehot positional ingredient mask for current recipe
            max_len {int} -- Maximum number of steps to unroll/predict
            start_token {int} -- Index of the start-recipe token
            teacher_forcing {bool} -- Whether to teacher-force (predict each step from gold prior)

        Keyword Arguments:
            logit_modifier_fxn {func} -- Function to modify the logits
                Default: no-op
            token_sampler {str} -- Function to create a probability distribution from logits and
                sample a token therefrom. Default: 'greedy' (softmax -> argmax)
            name_targets {torch.Tensor} -- Gold label name tokens (for name decoding)
            max_name_len {int} -- Maximum number of name tokens to unroll/predict

        Returns:
            torch.Tensor -- Vocabulary probability distribution for each step in a batch
                [B, T, vocab_size]
            {Additional tensors for visualization}
        """
        # Get recipe encoding
        batch_size = targets.size(0)
        recipe_encoding, calorie_encoding, name_encoding, ingr_encodings = self.encoder(inputs)

        # Project recipe encoding to the proper size
        recipe_encoding_projected = self.bridge_layer(recipe_encoding)

        # Get logit probability quashed with Softmax
        decoder_outputs = self.decoder(
            device=device,
            initial_hidden=recipe_encoding_projected,
            calorie_encoding=calorie_encoding,
            name_encoding=name_encoding,
            ingr_encodings=ingr_encodings,
            ingr_masks=ingr_masks,
            targets=targets if teacher_forcing else None,
            max_len=max_len, start_token=start_token, batch_size=batch_size,
            logit_modifier_fxn=logit_modifier_fxn,
            token_sampler=token_sampler,
            visualize=visualize
        )

        # Decode name for multi task
        name_decoder_outputs = (None, None)
        if self.name_decoder is not None:
            name_decoder_outputs = self.name_decoder(
                device=device,
                initial_hidden=recipe_encoding_projected,
                name_targets=name_targets if teacher_forcing else None,
                max_name_len=max_name_len, start_token=start_token, batch_size=batch_size,
                logit_modifier_fxn=logit_modifier_fxn,
                token_sampler=token_sampler
            )

        return decoder_outputs, name_decoder_outputs

class BaselineDecoder(Decoder):
    """A conditional RNN decoder"""
    def __init__(self, **kwargs):
        """
        See `recipe_gen.models.__init__` for required default arguments.

        Additional Arguments:
            N/A
        """
        super().__init__(**kwargs)

        '''Current item ingredient Attention'''
        self.ingr_attention = BahdanauAttention(
            self.hidden_size,
            key_size=self.ingr_encoded_size
        )

        '''Pre Output Layer'''
        # Input: [GRU output; input embedding; ingr context; calorie encoding; name encoding]
        self.attn_fusion_input_size = self.hidden_size + \
            self.vocab_emb_dim + \
            self.ingr_encoded_size + \
            self.calorie_encoded_size + \
            self.name_encoded_size

        self.attn_fusion_layer = nn.Sequential(
            nn.Linear(
                self.attn_fusion_input_size,
                self.hidden_size
            ),
            nn.ReLU(),
            nn.Dropout(p=self.dropout)
        )

        '''Decoder RNN'''
        # Input: [input embedding; ingr context]
        self.rnn_input_size = self.vocab_emb_dim + \
            self.ingr_encoded_size
        self.rnn = nn.GRU(
            self.rnn_input_size,
            self.hidden_size,
            self.gru_layers,
            batch_first=True,
            dropout=self.dropout
        )

    def forward(self, device, initial_hidden, calorie_encoding, name_encoding,
                ingr_encodings, ingr_masks, 
                targets, max_len, batch_size, start_token,
                logit_modifier_fxn, token_sampler,
                visualize=False):
        """
        Forward pass over a batch, unrolled over all timesteps

        Arguments:
            device {torch.device} -- Torch device
            initial_hidden {torch.Tensor} -- Initial hidden state for the decoder RNN [L; B; H]
            calorie_encoding {torch.Tensor} -- Calorie level encoding [B; H]
            name_encoding {torch.Tensor} -- Recipe encoding for final name [L; B; H]
            ingr_encodings {torch.Tensor} -- MLP-encodings for each ingredient in recipe [B; Ni; H]
            ingr_masks {torch.Tensor} -- Positional binary mask of non-pad ingredients in recipe
                [B; Ni]
            targets {torch.Tensor} -- Target (gold) token indices. If provided, will teacher-force
                [B; T; V]
            max_len {int} -- Unroll to a maximum of this many timesteps
            batch_size {int} -- Number of examples in a batch
            start_token {int} -- Start token index to use as initial input for non-teacher-forcing

        Keyword Arguments:
            sample_next_token {func} -- Function to select the next token from a set of logit probs.
                Only used if not teacher-forcing. (default: partial(top_k_logits, k=0) with sampler='greedy')
            visualize {bool} -- Whether to accumulate items for visualization (default: {False})

        Returns:
            torch.Tensor -- Logit probabilities for each step in the batch [B; T; V]
            torch.Tensor -- Output tokens /step /batch [B; T]
            {Optional tensors if visualizing}
                torch.Tensor -- Positional ingredient attention weights /step /batch [B; T; Ni]
        """
        # Initialize variables
        logit_probs = []
        use_teacher_forcing = targets is not None
        input_token = None
        decoder_hidden = initial_hidden

        # Accumulation of attention weights
        ingr_attns_for_plot = []
        output_tokens = []

        # Key projections
        ingr_proj_key = self.ingr_attention.key_layer(ingr_encodings)

        # Unroll the decoder RNN for max_len steps
        for i in range(max_len):
            # Teacher forcing - use prior target token
            if use_teacher_forcing:
                input_token = targets[:, i].unsqueeze(1)
            # Non-teacher forcing - initialize with START; otherwise use previous input
            elif i == 0:
                input_token = torch.LongTensor([start_token] * batch_size).unsqueeze(1).to(device)

            # Project input to vocab space
            input_embed = self.vocab_embedding(input_token)

            # Query -> decoder hidden state
            query = decoder_hidden[-1].unsqueeze(1)  # [#layers, B, D] -> [B, 1, D]

            # Current item ingredient attention
            ingr_context, ingr_alpha = self.ingr_attention(
                query=query,
                proj_key=ingr_proj_key,
                value=ingr_encodings,
                mask=ingr_masks
            )
            if visualize:
                ingr_attns_for_plot.append(ingr_alpha)

            # Take a single step
            _, decoder_hidden, pre_output = self.forward_step(
                input_embed=input_embed,
                decoder_hidden=decoder_hidden,
                calorie_encoding=calorie_encoding.unsqueeze(1),
                name_encoding=name_encoding[-1].unsqueeze(1),
                context=[ingr_context]
            )

            # Project output to vocabulary space
            logits = self.proj(pre_output)
            logit_prob = F.log_softmax(logits, dim=-1)

            # Debug segment
            if torch.sum(torch.isnan(logit_prob)) > 0:
                print('NAN DETECTED')
                print('INPUT:\n{}'.format(input_token))
                print('NAME ENCODING:\n{}'.format(name_encoding))
                print('INGR CONTEXT:\n{}'.format(ingr_context))
                print('CALORIE ENCODING:\n{}'.format(calorie_encoding))
                raise Exception('NAN DETECTED')

            logit_probs.append(logit_prob)

            # Save input token for next iteration
            if not use_teacher_forcing:
                input_token = sample_next_token(
                    logits, logit_modifier_fxn=logit_modifier_fxn, sampler=token_sampler
                )
                output_tokens.append(input_token)

        # Return logit probabilities in tensor form
        logit_probs = torch.cat(logit_probs, dim=1)

        # Concatenate along step dimension for visualizations
        if not use_teacher_forcing:
            output_tokens = torch.cat(output_tokens, dim=1)
        if visualize:
            ingr_attns_for_plot = torch.cat(ingr_attns_for_plot, dim=1)
            return logit_probs, output_tokens, ingr_attns_for_plot

        return logit_probs, output_tokens

def create_model(vocab_emb_dim, calorie_emb_dim, hidden_size, n_layers,
                 dropout=0.0, max_ingr=20, max_ingr_tok=20, use_cuda=True,
                 state_dict_path=None, decode_name=False, ingr_gru=False,
                 ingr_emb=False, num_ingr=None, ingr_emb_dim=None, shared_projection=False):
    """
    Instantiates a model

    Arguments:
        vocab_emb_dim {int} -- Embedding dimension for vocabulary (ingredients, steps)
        calorie_emb_dim {int} -- Embedding dimension for calorie levels
        hidden_size {int} -- Size of hidden layers
        n_layers {int} -- Number of decoder RNN layers

    Keyword Arguments:
        dropout {float} -- Dropout rate (default: {0.0})
        max_ingr {int} -- Maximum # ingredients/recipe (default: {20})
        max_ingr_tok {int} -- Maximum # tokens/ingredient (default: {20})
        use_cuda {bool} -- Whether to use CUDA (default: {True})
        state_dict_path {str} -- If provided, loads pretrained model weights from here (default: {None})
        decode_name {bool} -- Decode the name as a separate output (default: {False})
        ingr_gru {bool} -- Use a GRU for ingredient encoding
        shared_projection {bool} -- Use the same projection layer for name & steps

    Returns:
        nn.Module -- Loaded Encoder-Decoder model
    """
    start = datetime.now()

    # Create embedding layers
    proj_layer = nn.Linear(hidden_size, VOCAB_SIZE, bias=False)
    vocab_emb_layer = nn.Embedding(VOCAB_SIZE, embedding_dim=vocab_emb_dim)
    calorie_emb_layer = nn.Embedding(3, embedding_dim=calorie_emb_dim)
    if ingr_emb:
        ingr_emb_layer = nn.Embedding(num_ingr, embedding_dim=ingr_emb_dim)
    else:
        ingr_emb_layer = None

    # Encoder
    encoder = Encoder(
        vocab_embedding_layer=vocab_emb_layer,
        calorie_embedding_layer=calorie_emb_layer,
        ingr_embedding_layer=ingr_emb_layer,
        hidden_size=hidden_size, max_ingrs=max_ingr, max_ingr_tokens=max_ingr_tok,
        dropout=dropout,
        ingr_gru=ingr_gru,
        gru_layers=n_layers,
    )

    # Decoder
    decoder = BaselineDecoder(
        vocab_embedding_layer=vocab_emb_layer,
        hidden_size=hidden_size,
        gru_layers=n_layers,
        dropout=dropout,
        ingr_encoded_size=encoder.ingr_encoded_size,
        calorie_encoded_size=encoder.calorie_encoded_size,
        name_encoded_size=encoder.name_encoded_size,
        proj_layer=proj_layer,
    )

    # Total model
    if decode_name:
        name_decoder = NameDecoder(
            vocab_embedding_layer=vocab_emb_layer,
            hidden_size=hidden_size,
            gru_layers=n_layers,
            dropout=dropout,
            proj_layer=proj_layer if shared_projection else nn.Linear(
                hidden_size, VOCAB_SIZE, bias=False
            ),
        )
        print('Parameters in name decoder {}'.format(count_parameters(name_decoder)))
        model = BaselineEncoderDecoder(encoder, decoder, name_decoder=name_decoder)
    else:
        model = BaselineEncoderDecoder(encoder, decoder)
    if use_cuda:
        model = model.cuda()
    
    print('{} - Constructed model skeleton'.format(
        datetime.now() - start
    ))

    if state_dict_path is not None:
        # Load model state dictionary
        state_dict = torch.load(state_dict_path)

        # Load state dict
        model.load_state_dict(state_dict, strict=True)

    print('{} - Created {} model with {:,} parameters'.format(
        datetime.now() - start, model.__class__.__name__, count_parameters(model)
    ))

    print(model)
    print('\n\n')

    return model
