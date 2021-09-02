from collections import defaultdict
from pickle import load
import torch
import torch.nn as nn
from collections import defaultdict
from util import *
import java_analyzer.bpe_tokenizer as jbt

SEP_TOK = '</t>'
ADD_EOT = lambda x: [e + SEP_TOK for e in x]
REM_EOT = lambda x: [e[:-4] for e in x]
UNOPS = ADD_EOT(['!'])
BINOPS = ADD_EOT(['&&', '||', '.', '==', '!=', '<', '>'])
PARENS = ADD_EOT(['(', ')', '()'])
LITERALS = ADD_EOT(['null', '0', '1', 'false', 'true'])
SPECIALS = UNOPS + BINOPS + PARENS + LITERALS
DEVICE = 'cuda'
Z_DIM = 1000

def load_vocab(path):
    with open(path, 'rb') as f:
        vocab2idx = load(f)
        vocab_size = (max(vocab2idx.values())+1+2)+1 # +1 for pad
        vocab2idx['@SOS'] = len(vocab2idx)
        vocab2idx['@EOS'] = len(vocab2idx)
    idx2vocab = {v+1:k for k, v in vocab2idx.items()}
    vocab2idx = {k:v+1 for k, v in vocab2idx.items()}
    return vocab2idx, idx2vocab, vocab_size

def load_model(path, vocab_size):
    # hyperparams
    if 'GRU' in path:
        from models.model import LanguageModel
        bidirectional = False
        lm = LanguageModel(vocab_size, emb_dim=Z_DIM, hidden_size=Z_DIM, bidirectional=bidirectional)
    elif 'Transformer' in path:
        from models.model import TransformerModel
        lm = TransformerModel(vocab_size, emb_dim=256, hidden_size=512, num_layers=3)
        lm.to(torch.float)
    elif 'CNN' in path:
        from models.model import CNNModel
        lm = CNNModel(vocab_size, emb_dim=256, hidden_size=128, num_layers=9)
        lm.to(torch.float)
    else:
        raise ValueError(f'Unrecognized model {path}')
    lm.load_state_dict(torch.load(path))
    lm.to(DEVICE)
    lm.eval()
    return lm

class BeamSearchInfo():
    def __init__(self, seq, logprob, paren_depth, last_tok, true_beam_add = []):
        self._seq = seq[:]
        self._logprob = logprob
        self._paren_depth = paren_depth
        self._last_tok = last_tok
        self._true_beam_add = true_beam_add
    
    def __repr__(self):
        return f'{self._logprob} / {self._paren_depth} / {self._tokens}'
    
    def get_tokens(self):
        return self._seq.split(SEP_TOK)
    
    def next_tokens(self, local_methods, local_vars, var_methods, var_fields):
        assert self._paren_depth != 0 # intention is to call when expr unfinished
        prior_beam_tokens = self.get_tokens()
        last_comp_word = prior_beam_tokens[-2]
        curr_word = prior_beam_tokens[-1]
        
        pred_word_allows = []
        if last_comp_word in REM_EOT(BINOPS) or last_comp_word == '(':
            if last_comp_word == '.': # dot operator
                pred_word_allows = var_methods[prior_beam_tokens[-3]]+var_fields[prior_beam_tokens[-3]]
            else: # other binops
                valid_followups = PARENS[:1] + local_vars + LITERALS + local_methods + UNOPS
                if last_comp_word == '(':
                    valid_followups += PARENS[1:2]
                pred_word_allows = valid_followups
        elif last_comp_word in REM_EOT(UNOPS):
            pred_word_allows = PARENS[:1] + local_vars + local_methods
        elif last_comp_word[-1] == ')':
            pred_word_allows = PARENS[1:2] + BINOPS[:2] + BINOPS[3:]
        else: # identifier case
            eot_last_word = last_comp_word+SEP_TOK
            if eot_last_word in (local_vars + LITERALS): # local var
                pred_word_allows = PARENS[1:2] + BINOPS
            elif eot_last_word in local_methods:
                pred_word_allows = PARENS[:1]
            else: # some variable's field/methods
                assert prior_beam_tokens[-3] == '.', prior_beam_tokens
                root_token = prior_beam_tokens[-4]
                if eot_last_word in var_fields[root_token]:
                    pred_word_allows = PARENS[1:2] + BINOPS
                else:
                    pred_word_allows = PARENS[:1]
        curr_word_allows = [w for w in pred_word_allows if curr_word == w[:len(curr_word)]]
        return curr_word_allows
    
    def updated_copy(self, new_token, new_logprob=0, true_beam_add = False):
        '''returns updated copy of object.'''
        new_seq = self._seq + new_token
        new_paren_depth = self._paren_depth
        new_paren_depth += int('(' in new_token)
        new_paren_depth -= int(')' in new_token)
        if true_beam_add:
            new_tba = self._true_beam_add[:] + [new_token]
        else:
            new_tba = []
        return BeamSearchInfo(new_seq, new_logprob, new_paren_depth, new_token, new_tba)

def proj_to_new_beam(proj_vec, top_k, all_beams, prior_hs, 
                     legality_info, legal_progression, idx2vocab):
    search_size = 2000
    logits, max_idxs = torch.topk(proj_vec, k=search_size, dim=2)
    logits_np = logits.cpu().numpy()
    max_idxs_np = max_idxs.cpu().numpy()
    subseq_probs = []
    for i in range(len(all_beams)):
        # if sequence terminated, don't expand
        if all_beams[i]._paren_depth == 0:
            subseq_probs.append((all_beams[i]._logprob, i, -1))
            continue

        # otherwise, expand as below:
        available_names = all_beams[i].next_tokens(*legality_info)
        if len(available_names) == 0:
            continue # if no progress possible, don't expand
        curr_word = all_beams[i].get_tokens()[-1]
        curr_word_len = len(curr_word)
        legal_nexts = set.union(*[{e[curr_word_len:] for e in legal_progression[n]}
                                   for n in available_names if curr_word in legal_progression[n]])
        next_legals, next_legal_logits = [], []
        for j in range(search_size):
            tok_idx = max_idxs_np[0, i, j]
            try:
                new_word_frag = idx2vocab[tok_idx]
            except KeyError:
                continue
            
            tok_legal = new_word_frag in legal_nexts
            if tok_legal:
                next_legals.append(tok_idx)
                new_tok_logit = logits_np[0, i, j]
                next_legal_logits.append(new_tok_logit)

        next_legal_logprobs = torch.Tensor(next_legal_logits).log_softmax(0).numpy()
        for tok_idx, new_tok_logprob in zip(next_legals, next_legal_logprobs):
            subseq_probs.append((all_beams[i]._logprob + new_tok_logprob, i, tok_idx))
    
    sorted_beam_candidates = sorted(subseq_probs, reverse=True)[:top_k]
    new_beams = [
        all_beams[org_idx].updated_copy(idx2vocab[tok_idx], logp, True) if tok_idx != -1 else all_beams[org_idx]
        for logp, org_idx, tok_idx in sorted_beam_candidates
    ]
    if prior_hs is None:
        new_hs = None # handle stateless techniques
    else:
        new_hs = [prior_hs[:, org_idx:org_idx+1] for _, org_idx, _ in sorted_beam_candidates]
    return new_beams, new_hs

def decode_beam(model, valid_tokens, vocab2idx, idx2vocab,
                max_iter=5, pre_feed=None, beam_width=100):
    # initialization
    iter_count = 0
    pre_count = len(pre_feed) if pre_feed is not None else 0
    log_softmaxer = nn.LogSoftmax(dim=2)
    if model.model_type == 'GRU':
        hs = torch.zeros(1, 1, Z_DIM).to(DEVICE)
    else:
        hs = torch.zeros(0, 1, 1).long().to(DEVICE)
        common_hs = torch.zeros(0, 1, 1).long().to(DEVICE)
        model.to(torch.half)
    all_beams = [BeamSearchInfo('', 0, 0, '@SOS')]
    all_results = []
    local_methods = ADD_EOT([e.name.lower() for e in valid_tokens if e.type == 'method'])
    local_vars = ADD_EOT([e.name.lower() for e in valid_tokens 
                          if not e.type in ('method', 'constructor')])
    var_methods = defaultdict(list)
    var_methods.update(
        {iden.name.lower(): ADD_EOT([e.lower() for e in iden.methods])
         for iden in valid_tokens}
    )
    var_fields = defaultdict(list)
    var_fields.update(
        {iden.name.lower(): ADD_EOT([e.lower() for e in iden.fields])
         for iden in valid_tokens}
    )
    legality_info = (local_methods, local_vars, var_methods, var_fields)

    bpe_tokenizer = jbt.BPETokenizer(BPE_OP_FILE)
    tokenized_cache = {x: bpe_tokenizer.tokens2BPETokens([x]) for x in REM_EOT(SPECIALS)}
    tokenized_cache.update({x: bpe_tokenizer.tokens2BPETokens([x])
                            for x in REM_EOT(local_vars + local_methods)})
    for key in var_methods:
        tokenized_cache.update({e.lower(): bpe_tokenizer.tokens2BPETokens([e]) 
                                for e in REM_EOT(var_methods[key])})
        tokenized_cache.update({e.lower(): bpe_tokenizer.tokens2BPETokens([e]) 
                                for e in REM_EOT(var_fields[key])})
    legal_progression = {
        e+SEP_TOK: {''.join(tokenized_cache[e][:i]) for i in range(len(tokenized_cache[e])+1)}
        for e in tokenized_cache.keys()
    }
    
    log('Expanding...')
    while iter_count < max_iter + pre_count:
        tok_vec = torch.ShortTensor([[[vocab2idx[bsi._last_tok]] for bsi in all_beams]]).to(DEVICE)
        if model.model_type != 'GRU':
            if iter_count == pre_count:
                common_hs = hs[-512:].to(DEVICE)
                hs = torch.zeros(0, 1, 1).long().to(DEVICE)
            hs_list = []
            proj_list = []
            block_size = 250
            for idx in range(0, hs.size(1), block_size):
                hs_input = hs[:, idx:idx+block_size].to(DEVICE)
                hs_input = torch.cat([common_hs.repeat(1, hs_input.size(1), 1), hs_input], dim=0)
                with torch.no_grad():
                    hs_shard, proj_shard = model.decode_cell(tok_vec[:, idx:idx+block_size], hs_input)
                hs_list.append(hs_shard[common_hs.size(0):].cpu())
                proj_list.append(proj_shard.cpu().to(torch.float))
            hs = torch.cat(hs_list, dim=1)
            proj = torch.cat(proj_list, dim=1)
        else:
            with torch.no_grad():
                hs, proj = model.decode_cell(tok_vec.long(), hs)
        
        if iter_count >= pre_count:
            log(f'{iter_count - pre_count + 1} / {max_iter} expansion...')
            all_beams, hs = proj_to_new_beam(
                proj, beam_width, all_beams,
                hs, legality_info, legal_progression, idx2vocab
            )
            hs = torch.cat(hs, dim=1) if hs is not None else None
            for b_rank, beam in enumerate(all_beams):
                beam_toks = beam._seq.split(SEP_TOK)
                beam_if_idx = -beam_toks[::-1].index('if')
                print(b_rank, ''.join(beam_toks[beam_if_idx:]), f'{beam._logprob:.4f}')
        else:
            all_beams = [bsi.updated_copy(pre_feed[iter_count]) for bsi in all_beams]
        iter_count += 1
        all_results.append([(bsi._seq, bsi._logprob) for bsi in all_beams])
    log('Generation Complete.')
    return all_results

def beam_run(feed, valid_tokens, model_path, beam_width=100, max_iter=5):
    '''no-dependency function that only takes prior feed / valid token info;
    returns beams.'''
    v2i, i2v, vocab_size = load_vocab(BPE_VOCAB_FILE)
    lm = load_model(model_path, vocab_size)
    all_results = decode_beam(lm, valid_tokens, v2i, i2v, pre_feed = feed, max_iter=max_iter, beam_width=beam_width)
    return all_results

