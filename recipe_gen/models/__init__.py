'''
-*- coding: utf-8 -*-

Skeleton/parent classes for models.

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

from recipe_gen.pipeline.eval import top_k_logits, sample_next_token

class EncoderDecoder(nn.Module):
    """ Base Encoder Decoder Wrapper """

    def __init__(self, encoder, decoder, **kwargs):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.name_decoder = kwargs.get('name_decoder', None)

        # Bridge layer - encoder output -> decoder hidden
        self.bridge_layer = nn.Sequential(
            nn.Linear(self.encoder.output_size, self.decoder.hidden_size),
            nn.Tanh()
        )

    def forward(self, **kwargs):
        raise NotImplementedError(
            '{} is a parent class. Please subclass this with module-specific logic.'.format(
                self.__class__.__name__
            )
        )

class Encoder(nn.Module):
    """Encodes a sequence of word embeddings"""
    def __init__(self, vocab_embedding_layer, calorie_embedding_layer,
                 hidden_size, max_ingrs, max_ingr_tokens, dropout=0.,
                 ingr_gru=False, gru_layers=1, **kwargs):
        """
        Recipe Encoder model, takes in a model specified by name, a caloric level,
        and ingredients, and returns a recipe representation
        
        Arguments:
            vocab_embedding_layer {nn.Embedding} -- Vocabulary embedding (for ingredients & name)
            calorie_embedding_layer {nn.Embedding} -- Caloric level embedding
            hidden_size {int} -- Size of each hidden layer
            max_ingrs {int} -- Maximum # ingredients
            max_ingr_tokens {int} -- Maximum # tokens in each ingredient
        
        Keyword Arguments:
            dropout {float} -- Drop out this proportion of weights in each train iter (default: {0.})
            ingr_gru {bool} -- Whether to use a GRU for ingredient encoding (default: {False})
            gru_layers {int} -- Number of layers in ingredient encoder GRU (default: {1})
        """
        super().__init__()

        self.vocab_embedding = vocab_embedding_layer
        self.calorie_embedding = calorie_embedding_layer
        self.ingr_embedding = kwargs.get('ingr_embedding_layer', None)

        # embedding sizes
        self.vocab_embedding_dim = self.vocab_embedding.embedding_dim
        self.calorie_embedding_dim = self.calorie_embedding.embedding_dim
        if self.ingr_embedding is not None:
            self.ingr_embedding_dim = self.ingr_embedding.embedding_dim
        self.hidden_size = hidden_size

        # Sizes
        self.max_ingr = max_ingrs
        self.max_ingr_tokens = max_ingr_tokens
        self.gru_layers = gru_layers

        # Base output size
        self.output_size = 0

        # Name encoder BiGRU
        self.name_encoder = nn.GRU(
            self.vocab_embedding_dim,
            self.hidden_size,
            self.gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.name_encoded_size = self.name_encoder.hidden_size * \
            (1 + self.name_encoder.bidirectional)
        self.output_size += self.name_encoded_size

        # Calorie level encoder
        self.calorie_encoder = nn.Sequential(
            nn.Linear(self.calorie_embedding_dim, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )
        self.calorie_encoded_size = self.calorie_encoder[0].out_features
        self.output_size += self.calorie_encoded_size

        # Ingredient encoder - GRU or MLP
        self.ingr_input_size = self.vocab_embedding_dim if self.ingr_embedding is None \
            else self.ingr_embedding_dim
        self.ingr_gru = ingr_gru
        if self.ingr_gru:  # BiGRU for encoder
            self.ingr_encoder = nn.GRU(
                self.ingr_input_size,
                self.hidden_size,
                self.gru_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout,
            )
            self.ingr_encoded_size = self.ingr_encoder.hidden_size * \
                (1 + self.ingr_encoder.bidirectional)
        else:  # MLP encoder for ingredients
            self.ingr_encoder = nn.Sequential(
                nn.Linear(self.ingr_input_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(p=dropout)
            )
            self.ingr_encoded_size = self.ingr_encoder[0].out_features
        self.output_size += self.ingr_encoded_size

        print('Created encoder with {:,} hidden size, {} ingredient encoding, {:,} output size'.format(
            self.hidden_size,
            '{}-layer BiGRU'.format(self.gru_layers) if self.ingr_gru else 'MLP',
            self.output_size
        ))

    def forward(self, x):
        """
        Encodes a recipe
        
        Arguments:
            x {tuple} -- Input batch tensors. Contains the following tensors:
                batch_calories: Calorie level for each recipe
                batch_names: Name indices in each recipe
                batch_ingredients: Ingredients (padded & flattened) for each recipe
        
        Returns:
            torch.FloatTensor -- Full recipe encoding [B, 3 * hidden_size]
            torch.FloatTensor -- Calorie level encoding [B, hidden_size]
            torch.FloatTensor -- Name encodings [B, hidden_size]
            torch.FloatTensor -- Ingredient encodings [B, n_ingredients, hidden_size]
        """
        batch_calories, batch_names, batch_ingr = x
        recipe_encoding_components = []

        ''' Encode Calorie information (initial: B, 1) '''
        # Embedded: B, calorie_embedding_dim
        calorie_embed = self.calorie_embedding(batch_calories)

        # Encoded: B, hidden_size
        calorie_level_encoding = self.calorie_encoder(calorie_embed)
        recipe_encoding_components.append(
            torch.stack([calorie_level_encoding] * self.gru_layers)
        )

        ''' Encode Name (initial: B, max_name_tokens) '''
        # Embedded: B, max_name_tokens, vocab_embedding_dim
        name_embed = self.vocab_embedding(batch_names)

        # Encoded: L, B, hidden_size (Bidirectional GRU)
        _, name_final = self.name_encoder(name_embed)
        name_fwd = name_final[0:name_final.size(0):2]
        name_bwd = name_final[1:name_final.size(0):2]
        recipe_name_encoding = torch.cat([name_fwd, name_bwd], dim=2)
        recipe_encoding_components.append(recipe_name_encoding)

        if self.ingr_embedding is None:
            ''' Encode Ingr (initial: B, max_ingr * max_ingr_tok) '''
            # Reshape flattened ingredients & embed: B, max_ingr, max_ingr_tok, vocab_embedding_dim
            ingr_reshaped = self.vocab_embedding(batch_ingr).view(
                -1, self.max_ingr, self.max_ingr_tokens, self.vocab_embedding_dim
            )
            # Average embeddings across tokens: B, max_ingr, vocab_embedding_dim
            ingr_embed = ingr_reshaped.mean(dim=-2)
        else:
            ''' Encode Ingr (initial: B, max_ingr) '''
            ingr_embed = self.ingr_embedding(batch_ingr) # B, max_ingr, ingr_embedding_dim

        # Encoded: B, max_ingr, hidden_size
        if self.ingr_gru:
            # Bidirectional GRU
            ingr_encodings, ingr_final = self.ingr_encoder(ingr_embed)
            ingr_fwd = ingr_final[0:ingr_final.size(0):2]
            ingr_bwd = ingr_final[1:ingr_final.size(0):2]
            recipe_ingr_encoding = torch.cat([ingr_fwd, ingr_bwd], dim=2)
            recipe_encoding_components.append(recipe_ingr_encoding)
        else:
            ingr_encodings = self.ingr_encoder(ingr_embed)

            # Mean to represent recipe: B, hidden_size
            recipe_ingr_encoding = ingr_encodings.mean(dim=1)
            recipe_encoding_components.append(
                torch.stack([recipe_ingr_encoding] * self.gru_layers)
            )

        ''' Recipe Encoding - Concat All Encodings '''
        recipe_encoding = torch.cat(recipe_encoding_components, dim=2)

        # Output shape: n_layers, B, 3 * hidden_size
        return recipe_encoding, calorie_level_encoding, recipe_name_encoding, ingr_encodings

class BahdanauAttention(nn.Module):
    """
    Implements Bahdanau (Additive/MLP) attention

    score(s, h) = v^T tanh(W[s; h])
    
    Energy layer: v^T [something]
    Query layer: projects query into the same space as the projection key
    """
    def __init__(self, hidden_size, key_size=None, query_size=None):
        super().__init__()

        key_size = key_size or hidden_size
        query_size = query_size or hidden_size

        # Projections into the hidden space
        self.key_layer = nn.Linear(key_size, hidden_size, bias=False)
        self.query_layer = nn.Linear(query_size, hidden_size, bias=False)
        self.energy_layer = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, query=None, proj_key=None, value=None, mask=None, copy=None, ratings=None):
        # We first project the query (the decoder state)
        # The projected keys (the encoder states) were already pre-computated
        query = self.query_layer(query)

        # Calculate scores
        scores = self.energy_layer(torch.tanh(query + proj_key))
        scores = scores.squeeze(2).unsqueeze(1)

        # Mask out invalid positions
        # The mask marks valid positions so we invert it using `mask & 0`
        if mask is not None:
            scores.data.masked_fill_(
                mask.unsqueeze(1) == 0, float('-inf')
            )

        # Turn scores to probabilities
        alphas = F.softmax(scores, dim=-1)

        # Copy over existing probability distribution to modify attention weights
        if copy is not None:
            # Can't use += since it's in-place
            alphas = alphas + copy.unsqueeze(1)

            # Normalize to perform a convex combination
            alphas = F.normalize(alphas, p=1, dim=-1)
        
        if ratings is not None:
            # add shifted ratings with alphas
            alphas = alphas + ratings.unsqueeze(1)

            # softmax to supress negative ratings
            alphas = F.softmax(alphas, dim=-1)

        # The context vector is the weighted sum of the values.
        context = torch.bmm(alphas, value)

        # context shape: [B, 1, 2D], alphas shape: [B, 1, M]
        return context, alphas

class Decoder(nn.Module):
    """
    Conditional RNN Decoder
    """
    def __init__(self, **kwargs):
        """
        Recipe Decoder model, takes in a recipe representation and optional context to predict
        output tokens in a sequence.

        Required Arguments:
            vocab_embedding_layer {nn.Embedding} -- Vocabulary embedding (for ingredients & steps)
            hidden_size {int} -- Size of each hidden layer
            gru_layers {int} -- Number of layers in decoder language model RNN
            dropout {float} -- Drop out this proportion of weights in each fully-connected layer
        """
        super().__init__()

        # Vocabulary space
        self.vocab_embedding = kwargs['vocab_embedding_layer']
        self.vocab_emb_dim = self.vocab_embedding.embedding_dim
        self.vocab_size = self.vocab_embedding.num_embeddings

        # Parameters
        self.hidden_size = kwargs['hidden_size']
        self.gru_layers = kwargs['gru_layers']
        self.dropout = kwargs['dropout']
        print('Creating decoder with {} layers of size {:,} with {:.3f} dropout'.format(
            self.hidden_size, self.gru_layers, self.dropout,
        ))

        # Sizes
        self.ingr_encoded_size = kwargs['ingr_encoded_size']
        self.name_encoded_size = kwargs['name_encoded_size']
        self.calorie_encoded_size = kwargs['calorie_encoded_size']
        print('Encoded ingredients size {}, encoded name size {}, encoded calorie size {}'.format(
            self.ingr_encoded_size, self.name_encoded_size, self.calorie_encoded_size
        ))

        '''Vocabulary Projection'''
        self.proj = kwargs['proj_layer']
        self.rnn = None

    def forward_step(self, input_embed, decoder_hidden, calorie_encoding, name_encoding,
                     context=None, **attention_fusion_context):
        """
        Runs a single forward step
        
        Arguments:
            input_embed {torch.Tensor} -- Embedding form of RNN inputs
            decoder_hidden {torch.Tensor} -- Decoder hidden state
            calorie_encoding {torch.Tensor} -- Calorie level encoding to be concatenated in pre-output
            name_encoding {torch.Tensor} -- Name encoding to be concatenated in pre-output

        Keyword Arguments:
            context {torch.Tensor / list} -- Dynamic current item attention context. If concatenative
                context, list of tensors.
            **attention_fusion_context {torch.Tensor} -- Personalized (attention) context tensors
                to add to pre-output layer

        Returns:
            torch.Tensor -- Pure language model (RNN) output
            torch.Tensor -- New RNN hidden state
            torch.Tensor -- Single step output logits
        """
        attn_fusion_context = list(attention_fusion_context.values())

        # Get RNN output
        rnn_input = torch.cat([input_embed, *context], dim=-1)
        rnn_output, decoder_hidden = self.rnn(rnn_input, decoder_hidden)

        # Pre-output: projects RNN output, input embedding, recipe encoding
        # [output; input; calorie encoding; name encoding; ingredient attn]
        attn_fusion_inputs = [rnn_output, input_embed, calorie_encoding, name_encoding]
        attn_fusion_inputs += context
        attn_fusion_inputs += attn_fusion_context

        # Run the pre-output layer
        attn_fusion_inputs = torch.cat(attn_fusion_inputs, dim=-1)
        attn_fusion_output = self.attn_fusion_layer(attn_fusion_inputs)

        return rnn_output, decoder_hidden, attn_fusion_output

    def forward(self, visualize=False, **kwargs):
        """
        Forward pass over a batch, unrolled over all timesteps

        Keyword Arguments:
            visualize {bool} -- Whether to accumulate items for visualization (default: {False})

        Raises:
            NotImplementedError -- Implement 
        """
        raise NotImplementedError(
            '{} is a parent class. Please subclass this with module-specific logic.'.format(
                self.__class__.__name__
            )
        )

class NameDecoder(nn.Module):
    """
    Conditional Name decoder
    """
    def __init__(self, **kwargs):
        """
        Recipe Decoder model, takes in a recipe representation and optional context to predict
        output tokens in a Name sequence.

        Required Arguments:
            vocab_embedding_layer {nn.Embedding} -- Vocabulary embedding (for ingredients & steps)
            hidden_size {int} -- Size of each hidden layer
            gru_layers {int} -- Number of layers in decoder language model RNN
            dropout {float} -- Drop out this proportion of weights in each fully-connected layer
        """
        super().__init__()

        # Vocabulary space
        self.vocab_embedding = kwargs['vocab_embedding_layer']
        self.vocab_emb_dim = self.vocab_embedding.embedding_dim
        self.vocab_size = self.vocab_embedding.num_embeddings

        # Parameters
        self.hidden_size = kwargs['hidden_size']
        self.gru_layers = kwargs['gru_layers']
        self.dropout = kwargs['dropout']
        print('Creating decoder with {} layers of size {:,} with {:.3f} dropout'.format(
            self.hidden_size, self.gru_layers, self.dropout
        ))

        '''Vocabulary Projection'''
        self.proj = kwargs['proj_layer']

        '''Decoder RNN'''
        # Input: [input embedding; ingr context + tech context]
        self.rnn = nn.GRU(
            self.vocab_emb_dim,
            self.hidden_size,
            self.gru_layers,
            batch_first=True,
            dropout=self.dropout
        )
    
    def forward(self, device, initial_hidden, name_targets, 
                    max_name_len, start_token, batch_size,
                    logit_modifier_fxn, token_sampler,
                    **kwargs):

        use_teacher_forcing = name_targets is not None
        decoder_hidden = initial_hidden
        logit_probs = []
        output_tokens = []

        for i in range(max_name_len):

            if use_teacher_forcing:
                input_token = name_targets[:, i].unsqueeze(1)
            # Non-teacher forcing - initialize with START; otherwise use previous input
            elif i == 0:
                input_token = torch.LongTensor([start_token] * batch_size).unsqueeze(1).to(device)

            # Project input to vocab space
            input_embed = self.vocab_embedding(input_token)

            rnn_output, decoder_hidden = self.rnn(input_embed, decoder_hidden)
            
            logits = self.proj(rnn_output)
            logit_prob = F.log_softmax(logits, dim=-1)

            logit_probs.append(logit_prob)

            # by default greedy
            logit_modifier_fxn = partial(top_k_logits, k=0)
            token_sampler = 'greedy'
            input_token = sample_next_token(
                logits, logit_modifier_fxn=logit_modifier_fxn, sampler=token_sampler
            )
            output_tokens.append(input_token)
            
        # Return logit probabilities in tensor form
        logit_probs = torch.cat(logit_probs, dim=1)
        # Concatenate along step dimension for visualizations
        output_tokens = torch.cat(output_tokens, dim=1)

        return logit_probs, output_tokens



