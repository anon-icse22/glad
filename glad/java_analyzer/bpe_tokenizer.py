'''
Tokenizes Methods using a pretrained BPE vocabulary.
'''

import java_analyzer.var_scope as var_scope
from collections import defaultdict
import javalang
from pickle import load

EOT = '</t>'
TMP_SEP_TOK = '<SEP>'
SEP_TOK_LEN = len(TMP_SEP_TOK)
wrap_sep = lambda x: TMP_SEP_TOK + x + TMP_SEP_TOK
unwrap_sep = lambda x: x[SEP_TOK_LEN:-SEP_TOK_LEN]

def get_line2method(my_file):
    with open(my_file) as f:
        lines = f.readlines()
        lines_in_file = len(lines)
    all_vars = var_scope.parseIdentifiers(my_file)
    all_methods = [e for e in all_vars if e.type == 'method']
    line2method = dict()
    for method_node in all_methods:
        method_loc = method_node.method_start.line, method_node.method_end.line+1
        for lno in range(*method_loc):
            line2method[lno] = method_loc
    return line2method

class BPETokenizer():
    def __init__(self, bpe_ops_file, lower=True):
        with open(bpe_ops_file, 'rb') as f:
            self._bpe_ops = load(f)
        self._lower = lower
    
    def tokens2BPETokens(self, tok_list):
        func_tokenized_str = ' '.join(tok_list)
        func_tokenized_str = func_tokenized_str.strip()
        if self._lower:
            func_tokenized_str = func_tokenized_str.lower()
        split_func = [EOT if t == ' ' else t for t in func_tokenized_str]
        split_func += [EOT]

        subst_cmd = wrap_sep(TMP_SEP_TOK.join(split_func))
        for bpe_idx in sorted(self._bpe_ops.keys()):
            bpe_pair = self._bpe_ops[bpe_idx]
            rep_str, new_str = wrap_sep(TMP_SEP_TOK.join(bpe_pair)), wrap_sep(''.join(bpe_pair))
            subst_cmd = subst_cmd.replace(rep_str, new_str)
            subst_cmd = subst_cmd.replace(rep_str, new_str) # overlap issues
        split_func = unwrap_sep(subst_cmd).split(TMP_SEP_TOK)
        return split_func

    def cmd2BPETokens(self, func_str):
        func_tokens = list(javalang.tokenizer.tokenize(func_str))
        func_token_names = [
            '@literal_str' if isinstance(e, javalang.tokenizer.String) else e.value
            for e in func_tokens
        ]
        return self.tokens2BPETokens(func_token_names)

    def cmd2Tokens(self, code_str):
        code_tokens = [e.value for e in javalang.tokenizer.tokenize(code_str)]
        return code_tokens

    def get_prelude(self, jfile, lineno):
        '''Return tokenized method prior (not inclusive) to lineno.'''
        line2method = get_line2method(jfile)
        assert lineno in line2method, 'line not in any known method'
        method_start, method_end = line2method[lineno]
        with open(jfile) as f:
            file_lines = f.readlines()
            prelude_lines = file_lines[method_start-1:lineno-1] # -1 due to python indexing
            prelude_str = ''.join(prelude_lines)
            prelude_tokens = self.cmd2BPETokens(prelude_str)
        return prelude_tokens

    