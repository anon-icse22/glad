'''Implements automated fine-tuning.'''

import os, sys
from pathlib import Path

from models.train_lm import train
from models.data_loader import get_SmallLinear_loader

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from util import *
from java_analyzer.bpe_training import count_tokens

def tokenize(proj, bug_id, data_dir='./data/finetune/'):
    full_tok_file_prefix = data_dir+f'{proj}_{bug_id}'
    count_tokens(repo_path(proj, bug_id), full_tok_file_prefix)
    return full_tok_file_prefix + '_tokenized.txt'

def fine_tune(proj, bug_id, varpair_file, vocab_file, lm, device='cuda'):
    token_file = tokenize(proj, bug_id)
    data_loader = get_SmallLinear_loader(token_file, varpair_file, vocab_file, 32, num_workers=4)
    model_path = f'{parentdir}/models/weights/jmBPELitOn_{lm.model_type}_ft{proj}{bug_id}.pth'
    train(lm, data_loader, model_path, 1, device)
    return model_path

