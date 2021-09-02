#!/usr/bin/python

import torch
import torchvision.datasets as dss
from torch.utils import data
import os, sys
import numpy as np

import re, tqdm
from pickle import load
import random

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from java_analyzer.bpe_tokenizer import BPETokenizer

class SrcMLLinearData(dss.DatasetFolder):
    def __init__(self, data_file, vocab_file):
        self.data = []
        with open(data_file) as f:
            for line in f:
                org_tokens = line.split()
                annotated = ['@SOS'] + org_tokens + ['@EOS']
                self.data.append(annotated)

        self.vocab2idx = dict()
        with open(vocab_file, 'r') as f:
            for line in f:
                token, idx, _ = line.split('\t')             
                self.vocab2idx[token] = int(idx)

        print(f'# datapoints: {len(self.data)}')
    
    def __getitem__(self, index):
        token_list = self.data[index]
        norm_token_list = [t if t in self.vocab2idx else '@unk'
                           for t in token_list]
        idx_list = [self.vocab2idx[t] for t in norm_token_list]
        idx_tensor = torch.LongTensor(idx_list).unsqueeze(1)
        return idx_tensor
    
    def __len__(self):
        return len(self.data)

class SrcMLLinearBPEData(dss.DatasetFolder):
    def __init__(self, data_dir, ops_file, vocab_file, train_seq_max=512):
        self.root_dir = data_dir
        self.data_files = [os.path.join(data_dir, fname) for fname in os.listdir(data_dir)]
        self.data_size = 1000*(len(self.data_files)-1)
        self.train_seq_max = train_seq_max
        self.subtokenizer = BPETokenizer(ops_file)
        with open(vocab_file, 'rb') as f:
            self.vocab2idx = load(f)
            vocab_size = max(self.vocab2idx.values())
            self.vocab2idx['@SOS'] = vocab_size+1
            self.vocab2idx['@EOS'] = vocab_size+2
        print(f'# datapoints: {self.data_size}')
    
    def __getitem__(self, index):
        fname = f'method{1000*(index//1000+1)}.txt'
        with open(os.path.join(self.root_dir, fname)) as f:
            methods_in_file = f.readlines()
        index_method = methods_in_file[index%1000]
        token_list = self.subtokenizer.tokens2BPETokens(index_method.split())
        idx_list = [self.vocab2idx[t]+1 for t in token_list 
                    if len(t) != 0 and t in self.vocab2idx]
        start_idx = random.randint(0, max(0, len(idx_list)-self.train_seq_max))
        trunc_idx_list = idx_list[start_idx:start_idx+self.train_seq_max]
        idx_tensor = torch.LongTensor(trunc_idx_list).unsqueeze(1)
        return idx_tensor
    
    def __len__(self):
        return self.data_size

def smll_collate(arg):
    lengths = [x.size(0)-1 for x in arg]
    pad_x = torch.nn.utils.rnn.pad_sequence(arg)
    return pad_x, lengths

class SmallLinearBPEData(dss.DatasetFolder):
    def __init__(self, data_file, ops_file, vocab_file, train_seq_max=512):
        with open(data_file) as f:
            self.data = f.readlines()
        self.data_size = len(self.data)
        self.train_seq_max = train_seq_max
        self.subtokenizer = BPETokenizer(ops_file)
        with open(vocab_file, 'rb') as f:
            self.vocab2idx = load(f)
            vocab_size = max(self.vocab2idx.values())
            self.vocab2idx['@SOS'] = vocab_size+1
            self.vocab2idx['@EOS'] = vocab_size+2
        print(f'# datapoints: {self.data_size}')
    
    def __getitem__(self, index):
        index_method = self.data[index]
        token_list = self.subtokenizer.tokens2BPETokens(index_method.split())
        idx_list = [self.vocab2idx[t]+1 for t in token_list 
                    if len(t) != 0 and t in self.vocab2idx]
        start_idx = random.randint(0, max(0, len(idx_list)-self.train_seq_max))
        trunc_idx_list = idx_list[start_idx:start_idx+self.train_seq_max]
        idx_tensor = torch.LongTensor(trunc_idx_list).unsqueeze(1)
        return idx_tensor
    
    def __len__(self):
        return self.data_size

def smll_collate(arg):
    lengths = [x.size(0)-1 for x in arg]
    pad_x = torch.nn.utils.rnn.pad_sequence(arg)
    return pad_x, lengths

############

def lm_collate(arg):
    return arg

def custom_collate(arg):
    roots, targets = zip(*arg)
    targets = torch.Tensor(targets)
    return roots, targets
    
def get_SrcMLLinear_loader(root_path, ops_path, vocab_path, batch_size, num_workers = 2):
    dataset = SrcMLLinearBPEData(root_path, ops_path, vocab_path)
    data_loader = data.DataLoader(
        dataset = dataset,
        batch_size = batch_size,
        shuffle = True,
        num_workers = num_workers,
        collate_fn = smll_collate
    )
    return data_loader

def get_SmallLinear_loader(root_path, ops_path, vocab_path, batch_size, num_workers = 2):
    dataset = SmallLinearBPEData(root_path, ops_path, vocab_path)
    data_loader = data.DataLoader(
        dataset = dataset,
        batch_size = batch_size,
        shuffle = True,
        num_workers = num_workers,
        collate_fn = smll_collate
    )
    return data_loader

   
