from argparse import ArgumentParser
import argparse
import os
import json
import sys

def train_classifier_args():
    parser = ArgumentParser()
    parser = common_args(parser)
    return parser.parse_args()

def common_args(parser):
    # Dataset
    parser.add_argument('-p', '--datapath', type=str, default='./dataset/PTBRGestures')
    parser.add_argument('-n', '--dataname', type=str, default='ptbr')
    parser.add_argument('-w', '--window_length', type=int, default=120)
    parser.add_argument('-ts', '--trn_step_length', type=int, default=10)
    parser.add_argument('-vs', '--val_step_length', type=int, default=120)

    # OS
    parser.add_argument('-o', '--output_path', type=str, default='./style_encoder/classifier_output/')
    parser.add_argument('-d', '--device', type=str, default='gpu')

    # Training
    parser.add_argument('-bs', '--batch_size', type=int, default=64)
    parser.add_argument('-cr', '--criterion', type=str, default='cross')
    parser.add_argument('-op', '--optimizer', type=str, default='sgd')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001)
    parser.add_argument('-mo', '--momentum', type=float, default=0.9)
    parser.add_argument('-ep', '--epochs', type=int, default=100)

    # Model
    parser.add_argument('-hs', '--hidden_size', type=int, default=128)
    parser.add_argument('-se', '--style_embedding_size', type=int, default=64)
    return parser

def generate_args():
    parser = ArgumentParser()
    parser.add_argument('-cp', '--classifier_path', type=str, required=True)
    parser = common_args(parser)
    load_args(parser)
    return load_args(parser)

def load_args(parser):
    args = parser.parse_args()
    args_path = os.path.join(args.classifier_path, 'args.json')
    assert os.path.exists(args_path), 'args.json not found in {}'.format(args.classifier_path)
    with open(args_path, 'r') as f:
        loaded_args = json.load(f)
    for key in loaded_args:
        if key in args.__dict__.keys():
            setattr(args, key, loaded_args[key])
        else:
            print('Warning: was not able to load [{}], using default value [{}] instead.'.format(key, args.__dict__[key]))
    return args