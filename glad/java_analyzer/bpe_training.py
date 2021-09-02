import argparse
import tqdm
from pickle import load, dump
from collections import defaultdict, Counter
import os, sys
import javalang
import time
from multiprocessing import Pool

# from bpe_utils import *

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import java_analyzer.var_scope as jvs

# WHITELIST_TOKENS = {'null', '0', '1', 'false', 'true', '2', '3', '""', }

def tok_norm(t):
    if isinstance(t, javalang.tokenizer.String):
        return '@literal_str'
#     elif isinstance(t, javalang.tokenizer.Literal) and t.value not in WHITELIST_TOKENS:
#         return '@literal_other'
    else:
        return t.value.lower()

def file_proc(full_file_path):
    local_tc = Counter()
    tokenized_methods = []
    try:
        with open(full_file_path) as f:
            file_content = f.readlines()
        all_vars = jvs.parseIdentifiers(full_file_path)
    except:
        return tokenized_methods, local_tc
    all_methods = [e for e in all_vars if e.type == "method"]
    for method in all_methods:
        if method.method_start is None: continue # bodyless function
        start_line, end_line = method.method_start.line, method.method_end.line
        method_content = ''.join(file_content[start_line-1:end_line])
        try:
            method_tokens = list(javalang.tokenizer.tokenize(method_content))
        except javalang.tokenizer.LexerError:
            continue
        method_str_tokens = [tok_norm(e) for e in method_tokens]
        tokenized_methods.append(method_str_tokens)
        method_token_count = Counter(method_str_tokens)
        local_tc.update(method_token_count)
    return tokenized_methods, local_tc
    
def count_tokens(root_dir, target_file_prefix, multiprocessing=True):
    token_counter = Counter()
    proc_file_count = 0
    failed_to_parse = 0
    java_files = []
    tokenize_save_file = target_file_prefix + '_tokenized.txt'
    f = open(tokenize_save_file, 'w') # reset tokenized file
    s = time.time()
    for dp, dns, fns in os.walk(root_dir):
        java_files += [os.path.join(dp, e) for e in fns if '.java' in e]
    if multiprocessing:
        with Pool(8) as p:
            method_counts = p.map(file_proc, java_files)
    else:
        method_counts = list(map(file_proc, java_files))
    token_methods = sum([e[0] for e in method_counts], [])
    for tokenized_method in token_methods:
        print(' '.join(tokenized_method), file=f)
    valid_method_counts = [e[1] for e in method_counts if len(e[1]) != 0]
    failed_to_parse += len(method_counts) - len(valid_method_counts)
    for mcount in valid_method_counts:
        token_counter.update(mcount)
    proc_file_count += len(java_files)
    print(f'[{proc_file_count}] Elapsed time {time.time()-s:.2f}s, fail count {failed_to_parse}')
    return token_counter

def get_all_chars(all_tok_chars):
    all_char_freq = Counter()
    for char_list in tqdm.tqdm(all_tok_chars):
        indiv_char_counter = Counter(char_list)
        all_char_freq.update(indiv_char_counter)
    return all_char_freq

def count_pairs(args):
    print('Initializing...')
    token_counter = count_tokens(args)
    tokens = list(token_counter.keys())
    count_vs = [token_counter[k] for k in tokens]
    all_tok_chars = [list(t) + [EOT] for t in tokens]
    with open(args.target_file_prefix + '_token_count.pkl', 'wb') as f:
        dump(token_counter, f)
    
    all_char_freq = get_all_chars(all_tok_chars)
    char2idx = {c:i for i, c in enumerate(all_char_freq.keys())}
    all_bpe_ops = {}

    new_char_idx = len(all_char_freq)-1
    subst_cmds = [wrap_sep(TMP_SEP_TOK.join(char_cmd)) for char_cmd in all_tok_chars]
    print('Initialization complete. Merging...')
    while new_char_idx < args.vocab_size:
        # count all pairs in corpus (count unique tokens at once for speedup)
        all_duet_freq = defaultdict(int)
        for char_list, tok_count in zip(all_tok_chars, count_vs):
            for pair in zip(char_list, char_list[1:]):
                all_duet_freq[pair] += tok_count

        # find max pairs
        max_pairs = sorted(all_duet_freq.keys(), key=all_duet_freq.__getitem__, reverse=True)[:args.speedup]
        used_chars = set()

        # merge
        for p_idx, max_pair in enumerate(max_pairs):
            # make sure the pair we are merging is ok to merge
            if len(set(max_pair) - used_chars) < len(set(max_pair)):
                print(f'{p_idx} | {max_pair} was in {used_chars}, skipped')
                continue # if pair uses already used chars, don't merge
            else:
                used_chars |= set(max_pair)
            new_char_idx += 1

            # actual merge
            rep_str, new_str = wrap_sep(TMP_SEP_TOK.join(max_pair)), wrap_sep(''.join(max_pair))
            subst_cmds = [cmd.replace(rep_str, new_str) for cmd in subst_cmds]
            char2idx[unwrap_sep(new_str)] = new_char_idx
            all_bpe_ops[new_char_idx-len(all_char_freq)] = max_pair
            
            # track progress
            print(f'{p_idx} | [{new_char_idx}/{args.vocab_size}] {max_pair} (count={all_duet_freq[max_pair]})')
        
        all_tok_chars = [unwrap_sep(cmd).split(TMP_SEP_TOK) for cmd in subst_cmds]
    
    print('')
    return char2idx, all_bpe_ops

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Learns BPE tokens from data.')
    parser.add_argument('--raw_file', help='File on which BPE shall be trained.')
    parser.add_argument('--vocab_size', type=int, default=5000,
                        help='How many byte-code pairs to extract.')
    parser.add_argument('--target_file_prefix', type=str, default='./train',
                        help='Prefix of saved vocabulary and merge operator files.')
    parser.add_argument('--speedup', type=int, default=1,
                        help='Increases speed of pair extraction by allowing approximation.')
                        
    args = parser.parse_args()
    vocab, bpe_ops = count_pairs(args.raw_file, args.target_file_prefix)

    with open(args.target_file_prefix + '_vocab_BPE.pkl', 'wb') as f:
        dump(vocab, f)
    with open(args.target_file_prefix + '_varpairs_BPE.pkl', 'wb') as f:
        dump(bpe_ops, f)
