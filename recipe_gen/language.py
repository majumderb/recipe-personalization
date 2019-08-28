'''
-*- coding: utf-8 -*-

Utilities for handling vocabulary, tokenization, and recipe presentation.

@inproceedings{majumder2019emnlp,
  title={Generating Personalized Recipes from Historical User Preferences},
  author={Majumder, Bodhisattwa Prasad* and Li, Shuyang* and Ni, Jianmo and McAuley, Julian},
  booktitle={EMNLP},
  year={2019}
}

Copyright Shuyang Li & Bodhisattwa Majumder
License: GNU GPLv3
'''

import re
import numpy as np

from itertools import chain
from difflib import SequenceMatcher
from string import punctuation
from pytorch_pretrained_bert import OpenAIGPTTokenizer

# https://stackoverflow.com/questions/5286541/how-can-i-flatten-lists-without-splitting-strings
def _flatten(list_of_lists):
    for x in list_of_lists:
        if hasattr(x, '__iter__') and not isinstance(x, str):
            for y in _flatten(x):
                yield y
        else:
            yield x

# Vocabulary
TOKENIZER = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
GPT_VOCAB_SIZE = len(TOKENIZER.encoder)

# Extra tokens
EOS_INDEX = GPT_VOCAB_SIZE          # End of step
PAD_INDEX = GPT_VOCAB_SIZE + 1      # PAD
START_INDEX = GPT_VOCAB_SIZE + 2    # START document
END_INDEX = GPT_VOCAB_SIZE + 3      # END document
SOS_INDEX = GPT_VOCAB_SIZE + 4      # Start of step

# Total vocabulary size
VOCAB_SIZE = GPT_VOCAB_SIZE + 5

# Decoder
DECODER = {v: k for k, v in TOKENIZER.encoder.items()}
DECODER.update({
    SOS_INDEX: '<s>',
    EOS_INDEX: '</s>',
    PAD_INDEX: '</p>',
    START_INDEX: '<R>',
    END_INDEX: '</R>',
})

def tokenize_string(tok_str, separators=True):
    tokens = TOKENIZER.convert_tokens_to_ids(TOKENIZER.tokenize(tok_str))
    if separators:
        return [START_INDEX] + tokens + [END_INDEX]
    return tokens

def tokenize_list(str_list, flatten=False, separators=True):
    tokens_list = [TOKENIZER.convert_tokens_to_ids(TOKENIZER.tokenize(s)) for s in str_list]
    if separators:
        tokens_list = [[SOS_INDEX] + t + [EOS_INDEX] for t in tokens_list]
    if flatten:
        tokens_list = list(_flatten(tokens_list))
    return tokens_list

# Decoding utilities
def decode_ids(ids, tokens_list=True):
    '''
    Decode a list of IDs (from softmax->argmax or beam search) into tokens. e.g. usage:

    >>> from language import decode_ids, TOKENIZER, START_INDEX, END_INDEX
    >>> target_str = 'decode everything! (but why?)'
    >>> ids = [START_INDEX] + TOKENIZER.convert_tokens_to_ids(TOKENIZER.tokenize(target_str)) + [END_INDEX]
    >>> decode_ids(ids)
    ['</S>', 'de', 'code</w>', 'everything</w>', '!</w>', '(</w>', 'but</w>', 'why</w>', '?</w>', ')</w>', '</E>']
    >>> decode_ids(ids, False)
    '</S>decode everything ! ( but why ? ) </E>'

    Arguments:
        ids {list} -- Iterable of token IDs

    Keyword Arguments:
        tokens_list {bool} -- If True, returns list of raw tokens (default: {True})

    Returns:
        list -- List of decoded tokens
            OR
        str -- Decoded string, if tokens_list was False
    '''
    # Decode indices into tokens, ignoring '' and None
    raw_tokens = list(filter(None, [DECODER.get(i) for i in ids]))

    # Return as list of tokens
    if tokens_list:
        return raw_tokens

    # Join and separate by the GPT token separators (</w>)
    str_decoded = ''.join(raw_tokens).replace('</w>', ' ')
    return str_decoded

def pretty_decode_tokens(tokens):
    # Join them into string form
    str_decoded = ''.join(tokens)

    # Kill start/end tokens
    str_decoded = str_decoded.replace('<R>', '').replace('</R>', '|END|')

    # Replace pad tokens with null strings
    str_decoded = str_decoded.replace('</p>', '')

    # Replace word end tokens with spaces
    str_decoded = str_decoded.replace('</w>', ' ')

    # Replace EOS with full stops (step ending), clobber SOS
    str_decoded = str_decoded.replace('<s>', '')
    str_decoded = str_decoded.replace('</s>', '. ')

    return str_decoded
    
def pretty_decode(ids):
    '''
    Decode a list of IDs (from softmax->argmax or beam search) into a human-readable sentence. e.g. usage:

    >>> from language import pretty_decode, TOKENIZER, START_INDEX, END_INDEX, EOS_INDEX
    >>> from itertools import chain
    >>> target_strs = 'decode everything. (but why?)'.split('. ')
    >>> tokenized_steps = [TOKENIZER.convert_tokens_to_ids(TOKENIZER.tokenize(s)) + [EOS_INDEX] for s in target_strs]
    >>> ids = [START_INDEX] + list(chain.from_iterable(tokenized_steps)) + [END_INDEX]
    >>> print(pretty_decode(ids))
    decode everything . ( but why ? ) .

    Arguments:
        ids {list} -- Iterable of token IDs
    
    Returns:
        str -- Decoded string
    ''' 
    # Get raw tokens
    raw_tokens = decode_ids(ids, tokens_list=True)

    return pretty_decode_tokens(raw_tokens)

# Techniques - ordered!
TECHNIQUES_LIST = [
    'bake',
    'barbecue',
    'blanch',
    'blend',
    'boil',
    'braise',
    'brine',
    'broil',
    'caramelize',
    'combine',
    'crock pot',
    'crush',
    'deglaze',
    'devein',
    'dice',
    'distill',
    'drain',
    'emulsify',
    'ferment',
    'freez',
    'fry',
    'grate',
    'griddle',
    'grill',
    'knead',
    'leaven',
    'marinate',
    'mash',
    'melt',
    'microwave',
    'parboil',
    'pickle',
    'poach',
    'pour',
    'pressure cook',
    'puree',
    'refrigerat',
    'roast',
    'saute',
    'scald',
    'scramble',
    'shred',
    'simmer',
    'skillet',
    'slow cook',
    'smoke',
    'smooth',
    'soak',
    'sous-vide',
    'steam',
    'stew',
    'strain',
    'tenderize',
    'thicken',
    'toast',
    'toss',
    'whip',
    'whisk',
]

# Including the padding technique
PAD_TECHNIQUE_INDEX = len(TECHNIQUES_LIST)
N_TECHNIQUES = len(TECHNIQUES_LIST) + 1

TECHNIQUE_TOKENS = [
    TOKENIZER.convert_tokens_to_ids(TOKENIZER.tokenize(t))
    for t in TECHNIQUES_LIST
]

def get_technique_onehot(techniques):
    return [int(t in techniques) for t in TECHNIQUES_LIST]

def get_tokenized_techniques(techniques_onehot, pad_to=15):
    token_indices = list(chain.from_iterable(
        TECHNIQUE_TOKENS[i] for i in techniques_onehot if i == 1
    )) + [END_INDEX]

    if pad_to:
        token_indices += [PAD_INDEX] * (pad_to - len(token_indices))

    return token_indices

def match_line(segment_list, line, threshold=None):
    """
    Returns the likelihood of each segment in a list appearing in a line of text

    Arguments:
        segment_list {list} -- List of strings to check for
        line {str} -- Target line

    Keyword Arguments:
        threshold {float} -- Squash to 0 below threshold and 1 above.

    Returns:
        list -- List of sequence matcher scores
    """
    segment_scores = []
    for seg in segment_list:
        seg_len = len(seg)
        match_len = len(line) - seg_len
        # Find the maximum likelihood substring match
        segment_scores.append(max(
            SequenceMatcher(None, seg, line[i:i+seg_len]).ratio() for i in range(match_len)
        ))

    if threshold:
        return [int(s >= threshold) for s in segment_scores]

    return segment_scores

REMOVE_STAR = re.compile('\*+')
WHITESPACE_SHORTEN = re.compile('\s+')
WEBSITE_REMOVE = re.compile('http(s?)://[^\s]*')
SPACE_PUNCTUATION = [';', '.', ',', '!', '?', '/']
PARENTHETICAL = re.compile("\([^()]*\)")
REMOVE_NOTES = re.compile('(see )?(note|tip)(:| that)?[^\.]*')
SPLIT = re.compile('\s*[;.]+\s*')
ALL_PUNCTUATION = punctuation + '“”’‘'
REMOVE_PUNCTUATION = str.maketrans(ALL_PUNCTUATION, ' ' * len(ALL_PUNCTUATION))

def recipe_repr(name_tokens, ingr_tokens, input_tokens, output_tokens=None):
    """
    Creates representation of recipe

    Arguments:
        name_tokens {list} -- Name tokens
        ingr_tokens {list} -- Ingredient tokens
        input_tokens {list} -- Input tokens
    
    Keyword Arguments:
        output_tokens {list} -- Output tokens (default: {None})
    
    Returns:
        str -- Recipe name
        list -- Recipe ingredient strings
        str -- Recipe steps
        str -- Output steps
        str -- Formatted recipe representation
    """
    # Decode individual segments
    name_str = pretty_decode_tokens(name_tokens)
    ingr_strs = list(filter(None, [
        pretty_decode_tokens(l) for l in np.array_split(ingr_tokens, 20)
    ]))
    steps_str = pretty_decode_tokens(input_tokens)

    full_recipe_str = ''
    full_recipe_str += 'Name: `{}`\n'.format(name_str)
    full_recipe_str += 'Ingredients:\n--{}\n'.format('\n--'.join(ingr_strs))
    full_recipe_str += '\nOriginal Steps:\n{}\n'.format(steps_str)
    
    output_str = None
    if output_tokens is not None:
        output_str = pretty_decode_tokens(output_tokens)
        full_recipe_str += '\nMODEL OUTPUT:\n{}'.format(output_str)

    return name_str, ingr_strs, steps_str, output_str, full_recipe_str

CALORIE_LEVELS = ['Low-Calorie', 'Medium-Calorie', 'High-Calorie']

def recipe_spec_repr(calorie_level, techniques, ingr_tokens, input_tokens,
                     output_tokens=None, name_tokens=None, original_name_tokens=None):
    """
    Creates representation of recipe

    Arguments:
        calorie_level {int} -- Calorie level
        techniques {list} -- Technique list
        ingr_tokens {list} -- Ingredient tokens
        input_tokens {list} -- Input tokens
    
    Keyword Arguments:
        output_tokens {list} -- Output tokens (default: {None})
    
    Returns:
        str -- Recipe name
        list -- Recipe ingredient strings
        str -- Recipe steps
        str -- Output steps
        str -- Formatted recipe representation
    """
    # Decode individual segments
    if len(ingr_tokens) > 0 and isinstance(ingr_tokens[0], str):
        ingr_strs = list(filter(None, ingr_tokens))
    else:
        ingr_strs = list(filter(None, [
            pretty_decode(l) for l in ingr_tokens
        ]))
    steps_str = pretty_decode(input_tokens)

    full_recipe_str = ''
    original_name_str = pretty_decode(original_name_tokens)
    full_recipe_str += '\nORIGINAL NAME: `{}`\n'.format(original_name_str)
    if name_tokens is not None:
        name_str = pretty_decode(name_tokens)
        full_recipe_str += '\nGENERATED NAME: `{}`\n'.format(name_str)
    full_recipe_str += 'Calorie-Level: `{}`\n'.format(CALORIE_LEVELS[calorie_level])
    full_recipe_str += 'Techniques:\n--{}\n'.format('\n--'.join(techniques))
    full_recipe_str += 'Ingredients:\n--{}\n'.format('\n--'.join(ingr_strs))
    full_recipe_str += '\nOriginal Steps:\n{}\n'.format(steps_str)
    
    output_str = None
    if output_tokens is not None:
        output_str = pretty_decode(output_tokens)
        full_recipe_str += '\nMODEL OUTPUT:\n{}'.format(output_str)


    return ingr_strs, steps_str, output_str, full_recipe_str

STEPS_REGEX = re.compile('Original Steps:\n[^\n]*')
CALORIE_REGEX = re.compile('Calorie-Level: `[^\n]*`')
NAME_REGEX = re.compile('ORIGINAL NAME: `[^\n]*`')
INGR_REGEX = re.compile('Ingredients:\n(?:--[^\n]*\n)+')
TECH_REGEX = re.compile('Techniques:\n(?:--[^\n]*\n)+')
OUTPUT_REGEX = re.compile('MODEL OUTPUT:\n[^\n]*')

def parse_output_file(fname, get_name=True, get_calorie=True, get_ingredients=True,
                 get_techniques=True, get_steps=True, get_output=True):
    base_str = open(fname, 'r+').read()
    returns = []
    
    # Name
    if get_name:
        name = NAME_REGEX.findall(base_str)[0]
        name = name[len('ORIGINAL NAME: `'):-len('|END|`')]
        returns.append(name)
    
    # Calorie level
    if get_calorie:
        calorie = CALORIE_REGEX.findall(base_str)[0]
        calorie = calorie[len('Calorie-Level: `'):-1]
        returns.append(calorie)
    
    # Ingredients
    if get_ingredients:
        ingredients = INGR_REGEX.findall(base_str)[0]
        ingredients = ingredients[len('Ingredients:\n'):-1].replace('--', '').split('\n')
        returns.append(ingredients)
    
    # Techniques
    if get_techniques:
        techniques = TECH_REGEX.findall(base_str)[0]
        techniques = techniques[len('Techniques:\n'):-1].replace('--', '').split('\n')
        returns.append(techniques)

    # Original steps
    if get_steps:
        original_steps = STEPS_REGEX.findall(base_str)[0]
        original_steps = original_steps[len('Original Steps:\n'):-len('|END|')].strip()
        returns.append(original_steps)
    
    # Output
    if get_output:
        output = OUTPUT_REGEX.findall(base_str)[0]
        output = output[len('MODEL OUTPUT:\n'):].strip()
        returns.append(output)
    
    return returns
