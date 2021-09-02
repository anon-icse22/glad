from pickle import load
import re
from os.path import join

import java_analyzer.var_scope as jvs
import java_analyzer.bpe_tokenizer as jbt
from java_analyzer.test_exception import find_exception
import beam_search
from util import *
from javalang.tokenizer import Position
from java_analyzer.var_scope import StaticJavaAnalyzer
from patch_inject import write_patch_file
import itertools

def get_loc_info(proj, num, abs_java_path, loc, fail_test):
    _, rel_java_path = parse_abs_path(abs_java_path)
    all_identifiers = jvs.parseIdentifiers(abs_java_path)
    non_recursive_identifiers = [e for e in all_identifiers 
                                 if not (e.type=='method' and e.method_start.line <= loc <= e.method_end.line)]
    valid_identifiers = jvs.enhanceIdentifiers(
        rel_java_path, fail_test, proj, num, loc, non_recursive_identifiers
    )
    case_recoverer = dict()
    case_recoverer.update({x.name.lower(): x.name for x in valid_identifiers})
    for iden in valid_identifiers:
        case_recoverer.update({e.lower(): e for e in iden.methods})
        case_recoverer.update({e.lower(): e for e in iden.fields})
    return valid_identifiers, case_recoverer

def tokens2code(bpe_str, prefix, org_tokens, case_recoverer):
    prefix_str = ''.join(prefix)
    prefix_len = len(prefix_str)
    if len(bpe_str) <= len(prefix_str):
        return
    code_str = bpe_str[prefix_len:]
    tokenized = code_str.split('</t>')
    case_recovered_tokens = [case_recoverer[x] if x in case_recoverer else x for x in tokenized]
    case_recovered_tokens = org_tokens + case_recovered_tokens[len(org_tokens):]
    code_str = '</t>'.join(case_recovered_tokens)
    code_str = code_str.replace('</t>.', '.')
    code_str = code_str.replace('.</t>', '.')
    code_str = code_str.replace('</t>(', '(')
    code_str = code_str.replace(')</t>', ')')
    code_str = code_str.replace('</t>', ' ')
    return code_str

def run_beam_for_bug(prefix, search_len, valid_identifiers, model_path, beam_width=500):
    all_results = beam_search.beam_run(
        prefix, valid_identifiers, model_path,
        max_iter=search_len, beam_width=beam_width
    )
    return all_results[-1]

def get_predicate_seed(abs_java_path, lineno, tokenizer):
    '''Returns BPE-tokenized seed, original code, and pattern if available.'''
    with open(abs_java_path) as f:
        target_code = f.read()
    sja = jvs.StaticJavaAnalyzer(target_code)
    if_analysis = jvs.parse_if_condition(sja, lineno)
    raw_curr_line = target_code.split('\n')[lineno-1]
    curr_line_raw_tokens = tokenizer.cmd2Tokens(raw_curr_line)
    if len(curr_line_raw_tokens) < 2:
        # empty current line means omission fault.
        return beam_search.ADD_EOT(['if', '(']), [], 'addition'
    elif if_analysis is not None:
        full_if_cond = jvs.get_code_substr(
            target_code, *if_analysis
        )
        if_from_curr = jvs.get_code_substr(
            target_code, jvs.Position.from_lineno(lineno), if_analysis[1]
        )
        org_seed_tokens = ['if'] + tokenizer.cmd2Tokens(full_if_cond)[:-1] # truncate paren
        seed_bpe_tokens = tokenizer.cmd2BPETokens(if_from_curr)[:-1]
        return seed_bpe_tokens, org_seed_tokens, 'addmod'
    elif 'else' in curr_line_raw_tokens:
        # if *not* in this line...
        curr_line_bpe_tokens = tokenizer.cmd2BPETokens(raw_curr_line)
        if '{' in curr_line_raw_tokens:
            curly_paren_idx = curr_line_bpe_tokens.index('{</t>')
            curr_line_seed_tokens = curr_line_bpe_tokens[:curly_paren_idx]
        else:
            curr_line_seed_tokens = curr_line_bpe_tokens
        true_seed = beam_search.ADD_EOT(['if', '('])
        return curr_line_seed_tokens + true_seed, [], 'modification'
    else:
        print(f'Warning: potentially unrecognized pattern {curr_line_raw_tokens}')
        return beam_search.ADD_EOT(['if', '(']), [], 'addition'

def gen_preds_for_bug_at(proj, bugid, rel_path, lineno, local_info, model_path,
                         search_len=5, beam_width=500):
    bpe_tokenizer = jbt.BPETokenizer(BPE_OP_FILE)
    full_file_path = repo_path(proj, bugid) + rel_path
    prelude_tokens = bpe_tokenizer.get_prelude(full_file_path, lineno)
    repair_seed, seed_org_tokens, repair_pattern = get_predicate_seed(full_file_path, lineno, bpe_tokenizer)

    # get bug local information
    valid_identifiers, case_recoverer = local_info 
    beam_prelude = prelude_tokens + repair_seed
    last_beam_results = run_beam_for_bug(
        beam_prelude, search_len, valid_identifiers, model_path, beam_width
    )
    if repair_pattern == 'addmod':
        add_beam_results = run_beam_for_bug(
            prelude_tokens+beam_search.ADD_EOT(['if', '(']), search_len, valid_identifiers, model_path, beam_width
        )
    else:
        add_beam_results = []

    obtained_patches = []
    last_if_index = -beam_prelude[::-1].index('if</t>')-1
    pre_if_prelude = beam_prelude[:last_if_index]
    for sent, logp in last_beam_results:
        code_candidate = tokens2code(
            sent, pre_if_prelude, seed_org_tokens, case_recoverer
        )
        if code_candidate is not None:
            obtained_patches.append((code_candidate, logp, 'modification' if repair_pattern == 'addmod' else repair_pattern))
    for sent, logp in add_beam_results:
        code_candidate = tokens2code(
            sent, pre_if_prelude, [], case_recoverer
        )
        if code_candidate is not None:
            obtained_patches.append((code_candidate, logp, 'addition'))
    return obtained_patches, repair_pattern

def get_basic_bodies(proj, bugid, failing_test):
    basic_list = [
        '{ return; }',
        '{ return null; }',
        '{ return true; }',
        '{ return false; }',
        '{ return 0; }',
        '{ return 1; }',
        '{ break; }',
        '{ continue; }',
    ]
    test_exceptions = find_exception(proj, bugid, failing_test)
    if len(test_exceptions) != 0:
        for exception in test_exceptions:
            basic_list.append(f'{{ throw new {exception}(); }}')
    return basic_list

def gen_patches_body_reuse(code_lines, predicate, line_location):
    bracketCollector = StaticJavaAnalyzer(''.join(code_lines))
    line_position = Position(line=line_location, column=0)
    _, close_bracket_position = bracketCollector.getParentCurlyBracket(line_position)
    start_if = "{} {{\n".format(predicate)
    end_if = "}\n"
    lines = code_lines[:]
    lines.insert(line_location-1, start_if) # since list index starts from zero
    lines.insert(close_bracket_position.line, end_if)
    return lines

class Patch():
    def __init__(self, name, lines, score):
        self.name = name
        self.lines = lines
        self.score = score
    
    def update(self, index, abs_java_path, patch_line_no, patch_root_dir):
        self.index = index
        self.patch_path = join(patch_root_dir, f'Patch{index}.patch')
        write_patch_file(abs_java_path, self.lines, self.patch_path)

def gen_new_codes(abs_java_path, predicates, pred_scores, fix_type, line_no, bodies=None):
    with open(abs_java_path) as f:
        code = f.read()
        f.seek(0)
        code_lines = f.readlines()
    if fix_type == 'addition':
        assert bodies is not None
        new_if_statements = [(cond[0] + body, pred_scores[cond]) for cond, body in itertools.product(predicates, bodies)]
        new_codes = [Patch(f'ADD {e}', code_lines[:line_no-1] + [e+'\n'] + code_lines[line_no-1:], score)
                     for e, score in new_if_statements]
    elif fix_type == 'wrapping':
        new_codes = [Patch(f'WRAP WITH {pred[0]}', gen_patches_body_reuse(code_lines, pred[0], line_no), pred_scores[pred])
                     for pred in predicates]
    elif fix_type == 'modification':
        sja = jvs.StaticJavaAnalyzer(code)
        if_analysis = jvs.parse_if_condition(sja, line_no)
        if if_analysis is not None:
            new_codes = [Patch(f'CHANGE TO {pred[0]}', 
                               code_lines[:if_analysis[0].line-1] + [pred[0]+'{\n'] + code_lines[if_analysis[1].line:],
                               pred_scores[pred])
                        for pred in predicates]
        else:
            new_codes = [Patch(f'CHANGE ELSE TO ELSE IF {pred[0]}',
                                code_lines[:line_no-1] + [f'}} else {pred[0]} {{\n'] + code_lines[line_no:],
                                pred_scores[pred])
                         for pred in predicates]
    else:
        raise ValueError(f'Unrecognized fix type {fix_type}')
    return new_codes


