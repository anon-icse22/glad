import argparse
from pathlib import Path
import numpy as np

from java_analyzer.var_scope import get_if_condition
from gen_patches import gen_preds_for_bug_at, get_basic_bodies, gen_new_codes, get_loc_info
from filter_patch import *
from util import *
from patch_inject import compile_repo
from fault_localization.get_data import get_failing_test_names
from beam_search import load_model, load_vocab
from models.fine_tune import fine_tune

def remove_duplicates(patch_list):
    # from https://stackoverflow.com/questions/480214
    seen = set()
    seen_add = seen.add
    return [x for x in patch_list if not (x[0] in seen or seen_add(x))]

def pfl_repair(proj, bugid, model_path, search_len, beam_width, stop_on_pass=True):
    from fault_localization.get_data import get_true_buggy_locs

    # initialization
    log('Initializing...')
    buggy_files, true_fix_locations = get_true_buggy_locs(proj, bugid, BUG_INFO_JSON)
    src_prefix = d4j_path_prefix(proj, bugid)
    buggy_files = [src_prefix + e for e in buggy_files]
    assert len(buggy_files) == 1
    # assert len(true_fix_locations[0]) == 1
    compile_repo(repo_path(proj, bugid))
    failing_tests = get_failing_test_names(proj, bugid)
    ex_fail_test = failing_tests[0]
    buggy_file_abs_path = repo_path(proj, bugid) + buggy_files[0]
    patch_save_dir = f'{PATCH_DIR}/{proj}_{bugid}/'
    Path(patch_save_dir).mkdir(parents=True, exist_ok=True)
    local_info = get_loc_info(proj, bugid, buggy_file_abs_path, true_fix_locations[0][0], ex_fail_test)

    # fine-tuning
    log('Fine-tuning...')
    _, _, vocab_size = load_vocab(BPE_VOCAB_FILE)
    org_lm = load_model(model_path, vocab_size)
    ft_lm_path = fine_tune(proj, bugid, BPE_OP_FILE, BPE_VOCAB_FILE, org_lm)
    log('Fine-tuning complete.')

    # generate and evaluate patches
    log('Generating Predicates...')
    generated_predicates, repair_pattern = gen_preds_for_bug_at(
        proj, bugid, buggy_files[0], true_fix_locations[0][0], 
        local_info, ft_lm_path,
        search_len=search_len, beam_width=beam_width
    )
    if repair_pattern in ('addmod', 'modification'):
        org_pred = get_if_condition(buggy_file_abs_path, true_fix_locations[0][0])
        if org_pred is None:
            org_pred = '(true)'
        org_pred = ''.join(org_pred.split())
    else:
        org_pred = None
    log('Generated predicate number:', len(generated_predicates))
    log('Generated predicates:\n'+'\n'.join(f'{e[0]}\t| {e[1]:.4E}, pattern={e[2]}' for e in generated_predicates))
    completed_patches = CompletedPatchFilter.filter_patches(proj, bugid, generated_predicates)
    log('Completed predicate number:', len(completed_patches))
    log('Completed predicates:\n'+'\n'.join(p[0] for p in completed_patches))
    unique_patches = remove_duplicates(completed_patches)
    log('Unique predicate number:', len(unique_patches))
    log('Unique predicates:\n'+'\n'.join(p[0] for p in unique_patches))
    valid_predicates = BooleanPatchFilter.filter_patches(
        proj, bugid, unique_patches, buggy_file_abs_path, 
        true_fix_locations[0][0], ex_fail_test, local_info
    )
    log('Valid predicate number:', len(valid_predicates))
    log('Valid predicates:\n'+'\n'.join(p[0] for p in valid_predicates))
    log('Starting dynamic ranker...')
    add_scores, wrap_scores, mod_scores = DynamicRanker.rerank_patches(
        proj, bugid, valid_predicates, buggy_file_abs_path,
        true_fix_locations[0][0], failing_tests, pred_origin=org_pred
    )
    log('Dynamic ranking complete.')
    add_preds = [p for p in valid_predicates if add_scores[p] != -np.inf and p[2] == 'addition'] # filtering
    add_preds_sorted = list(sorted(add_preds, key=add_scores.__getitem__, reverse=True))
    wrap_preds = [p for p in valid_predicates if wrap_scores[p] != -np.inf and p[2] == 'addition']
    wrap_preds_sorted = list(sorted(wrap_preds, key=wrap_scores.__getitem__, reverse=True))
    mod_preds = [p for p in valid_predicates if mod_scores[p] != -np.inf and p[2] == 'modification']
    mod_preds_sorted = list(sorted(mod_preds, key=mod_scores.__getitem__, reverse=True))
    log('Addable predicate number:', len(add_preds_sorted))
    log('Addable predicates:\n'+'\n'.join([f'{p[0]} | score={add_scores[p]}' for p in add_preds_sorted]))
    log('Wrappable predicate number:', len(wrap_preds_sorted))
    log('Wrappable predicates:\n'+'\n'.join([f'{p[0]} | score={wrap_scores[p]}' for p in wrap_preds_sorted]))
    log('Modifiable predicate number:', len(mod_preds_sorted))
    log('Modifiable predicates:\n'+'\n'.join([f'{p[0]} | score={mod_scores[p]}' for p in mod_preds_sorted]))

    eval_patches = []
    if len(mod_preds) > 0:
        eval_patches += gen_new_codes(buggy_file_abs_path, mod_preds_sorted, 
                                      mod_scores, 'modification', true_fix_locations[0][0])
    if len(add_preds) + len(wrap_preds) > 0:
        basic_bodies = get_basic_bodies(proj, bugid, ex_fail_test)
        valid_bodies = BasicBodyFilter.filter_patches(proj, bugid, basic_bodies, buggy_file_abs_path, true_fix_locations[0][0], [ex_fail_test])
        # combining predicates with basic bodies
        add_patches = gen_new_codes(buggy_file_abs_path, add_preds_sorted, 
                                    add_scores, 'addition', true_fix_locations[0][0], bodies=valid_bodies)
        log('Patches with body:', len(add_patches))
        # combining predicates with wrapping
        wrap_patches = gen_new_codes(buggy_file_abs_path, wrap_preds_sorted, 
                                     wrap_scores, 'wrapping', true_fix_locations[0][0])
        log('Wrapping Patches:', len(wrap_patches))
        eval_patches += add_patches + wrap_patches
    sorted_eval_patches = sorted(eval_patches, key=lambda x: x.score, reverse=True)
    log('Total Patches:', len(sorted_eval_patches))
    log('Evaluation order:\n' + '\n'.join(f'{idx+1} | {p.name}, {p.score:.3f}' for idx, p in enumerate(sorted_eval_patches)))
    for p_idx, patch in enumerate(sorted_eval_patches):
        patch.update(p_idx+1, buggy_file_abs_path, true_fix_locations[0][0], patch_save_dir)
    log(f'Patches saved to {patch_save_dir}')
    plausible_patches = PlausiblePatchFilter.filter_patches(proj, bugid, sorted_eval_patches, stop_on_pass=stop_on_pass)
    log('Plausible patch number:', len(plausible_patches))
    log('Plausible patches:\n' + '\n'.join(p.name for p in plausible_patches))
    if len(plausible_patches) > 0:
        log('Repair successful.')
    else:
        log('Repair failed.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fix defects4j bugs.')
    parser.add_argument('--project', type=str, help='Project Name in defects4j, e.g. Closure.')
    parser.add_argument('--bug_id', type=int, help='Bug ID (number), e.g. 15.')
    parser.add_argument('--model_path', type=str, default='models/weights/jmBPELitOn_GRU_L1.pth', 
                        help='Path of language model to use.')
    parser.add_argument('--search_len', type=int, default=10, help='Beam search length param.')
    parser.add_argument('--beam_width', type=int, default=500, help='Beam search candidate number.')
    parser.add_argument('--stop_on_pass', type=int, default=1, help='If 1, stops on first plausible patch.')
    args = parser.parse_args()
    pfl_repair(args.project, args.bug_id, args.model_path, args.search_len, args.beam_width, bool(args.stop_on_pass))
