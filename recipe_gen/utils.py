'''
-*- coding: utf-8 -*-

Basic torch / model utilities

Author: Shuyang Li, Bodhisattwa Majumder
License: GNU GPLv3
'''

import torch

def get_device(use_cuda=True):
    cuda_available = torch.cuda.is_available()
    use_cuda = use_cuda and cuda_available

    # Prompt user to use CUDA if available
    if cuda_available and not use_cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    # Set device
    device = torch.device('cuda:0' if use_cuda else 'cpu')

    print('Device: {}'.format(device))
    if use_cuda:
        print('Using CUDA {}'.format(torch.cuda.current_device()))
    return use_cuda, device

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
