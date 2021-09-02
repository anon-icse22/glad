from collections import defaultdict

from util import *
from patch_inject import d4j_evaluate_patch
from gen_patches import gen_new_codes, get_loc_info
from java_analyzer.var_scope import d4jPathToClasspath
from jdb_interface import JDBWrapper

class PatchRanker(object):
    @staticmethod
    def rerank_patches(proj, bugno, patch_list):
        raise NotImplementedError

class PatchFilter(object):
    @staticmethod
    def filter_patches(proj, bugno, patch_list):
        raise NotImplementedError

class CompletedPatchFilter(PatchFilter):
    @staticmethod
    def filter_patches(proj, bugno, patch_list):
        return list(filter(lambda x: x[0].count('(') == x[0].count(')'), patch_list))

class BooleanPatchFilter(PatchFilter):
    @staticmethod
    def filter_patches(proj, bugno, patch_list, abs_jfile, fix_location, target_test, local_info):
        '''assumes all patches are if ( expr ) format'''
        # get side-effect-affectable variables
        valid_identifiers, _ = local_info
        ids2fields = {e.name:[f'{e.name}.{field_name}' for field_name in e.nonprimitive_fields]
                      for e in valid_identifiers if len(e.fields) != 0}
        vulnerable_ids = set(ids2fields.keys())
        org_states = dict()

        curr_port = 31415
        jdbw = JDBWrapper(D4J_HOME, proj, bugno, target_test, port=curr_port)
        _, rel_jfile = parse_abs_path(abs_jfile)
        classpath = d4jPathToClasspath(proj, bugno, rel_jfile, fix_location)
        break_location = JDBWrapper.get_valid_breakpoint(abs_jfile, fix_location)
        bp_succ = jdbw.set_breakpoint(classpath, break_location)
        run_succ = jdbw.run_process()
        assert bp_succ and run_succ, (bp_succ, run_succ)
        raw_pred = [p[0].removeprefix('if') for p in patch_list]
        evaluatable = []
        side_effect_methods = set()
        for vid in vulnerable_ids:
            jdbw._relay_command(f'print {vid}.equals({vid})')
            org_states[vid] = jdbw._relay_command(f'dump {vid}')
            for field_name in ids2fields[vid]:
                org_states[field_name] = jdbw._relay_command(f'dump {field_name}')
        for p in raw_pred:
            try:
                if any(t in side_effect_methods for t in p.split()):
                    print(f'Skipping {p} due to potential side effects.')
                    evaluatable.append(False)
                    continue
                if '@literal_str' not in p:
                    tentative_ok = jdbw.evaluate_expr(f'{p}?0:1') in ('0', '1')
                    for vid in [e for e in vulnerable_ids if e+'.' in p]:
                        new_state = jdbw._relay_command(f'dump {vid}')
                        if new_state != org_states[vid]:
                            bad_method = [t for t in p.split() if vid+'.' in t][0]
                            side_effect_methods.add(bad_method)
                            raise ValueError
                        for field_name in ids2fields[vid]:
                            new_state = jdbw._relay_command(f'dump {field_name}')
                            if new_state != org_states[field_name]:
                                bad_method = [t for t in p.split() if vid+'.' in t][0]
                                side_effect_methods.add(bad_method)
                                raise ValueError
                    evaluatable.append(tentative_ok)
                else:
                    evaluatable.append(False)
            except (TimeoutException, ValueError) as e:
                evaluatable.append(False)
                if type(e) == TimeoutException:
                    print('JDB restarting due to timeout.')
                else:
                    print(f'JDB restarting due to potential side effects in {p}.')
                jdbw.terminate()
                curr_port += 1
                jdbw = JDBWrapper(D4J_HOME, proj, bugno, target_test, port=curr_port)
                jdbw.set_breakpoint(classpath, break_location)
                jdbw.run_process()
                for vid in vulnerable_ids:
                    jdbw._relay_command(f'print {vid}.equals({vid})')
                    org_states[vid] = jdbw._relay_command(f'dump {vid}')
                    for field_name in ids2fields[vid]:
                        org_states[field_name] = jdbw._relay_command(f'dump {field_name}')
        jdbw.exit()
        assert len(patch_list) == len(evaluatable)
        return [p for p, v in zip(patch_list, evaluatable) if v]

class DynamicRanker(PatchRanker):
    @staticmethod
    def rerank_patches(proj, bug_no, pred_list, abs_jfile, fix_location, failing_tests, 
                       pred_origin=None, timeout_min=15):
        import numpy as np
        import time
        from java_analyzer.all_tests import get_relevant_tests
        failing_test_classes = [e.split('::')[0] for e in failing_tests]
        failing_test_methods = [e.split('::')[1] for e in failing_tests]
        relevant_tests = get_relevant_tests(proj, bug_no, set(failing_test_methods))
        all_fail_class_tests = [e for e in relevant_tests if e.split('::')[0] in failing_test_classes]
        passing_test_classes = [e for e in relevant_tests if e.split('::')[0] not in failing_test_classes]
        relevant_tests = all_fail_class_tests + passing_test_classes # make sure failing test classes are evaluated first
        _, rel_jfile = parse_abs_path(abs_jfile)
        classpath = d4jPathToClasspath(proj, bug_no, rel_jfile, fix_location)
        break_location = JDBWrapper.get_valid_breakpoint(abs_jfile, fix_location)
        raw_preds = [p[0].removeprefix('if') for p in pred_list]
        test_eval_results = defaultdict(list)
        org_pred_results = defaultdict(list)
        test_corr_finished = defaultdict(lambda: False)
        start_time = time.time()
        for t_idx, target_test in enumerate(relevant_tests):
            if time.time() - start_time > timeout_min*60:
                break
            print(f'processing {target_test.split("::")[0]}...', flush=True)
            try:
                jdbw = JDBWrapper(D4J_HOME, proj, bug_no, target_test, port=41414+t_idx)
                bp_succ = jdbw.set_breakpoint(classpath, break_location)
                run_succ = jdbw.run_process()
            except TimeoutException:
                jdbw.terminate()
                continue
            assert bp_succ and run_succ, (bp_succ, run_succ)
            try: # ugh
                past_test_name = ''
                while not jdbw._client_terminated:
                    if time.time() - start_time > timeout_min*60:
                        raise TimeoutException # for cleanup
                    try:
                        curr_test_name = jdbw.get_which_test()
                    except IndexError:
                        if not jdbw.move_on():
                            break
                        else:
                            continue
                    
                    if curr_test_name != past_test_name:
                        test_corr_finished[past_test_name] = True
                        past_test_name = curr_test_name
                    
                    time_pred_vals = []
                    for p in raw_preds:
                        try:
                            pred_val = bool(int(jdbw.evaluate_expr(f'{p}?1:0')))
                        except:
                            pred_val = 'err' # placeholder
                        time_pred_vals.append(pred_val)
                    org_pred_evaled = True
                    if pred_origin is not None:
                        try:
                            org_pred_results[curr_test_name].append(
                                bool(int(jdbw.evaluate_expr(f'{pred_origin}?1:0')))
                            )
                        except:
                            org_pred_evaled = False
                    if time_pred_vals.count('err') < (0.1*len(time_pred_vals)) and org_pred_evaled:
                        test_eval_results[curr_test_name].append(time_pred_vals)
                    if not jdbw.move_on(): # checks if testing process has terminated
                        # if terminated, add previous past test to properly terminated
                        test_corr_finished[past_test_name] = True
                        break
            except TimeoutException:
                jdbw.terminate()
        
        test_prob_add = defaultdict(np.float)
        test_prob_wrap = defaultdict(np.float)
        test_prob_mod = defaultdict(np.float)
        total_test_num = 0
        for test_name in test_eval_results.keys():
            if not test_corr_finished[test_name]:
                continue # ignore unfinished tests 
            else:
                total_test_num += 1
            pred_to_time = [*zip(*test_eval_results[test_name])]
            org_pred_to_time = tuple(org_pred_results[test_name])
            if len(pred_to_time) == 0: continue # erroneous tests
            test_fails = test_name in failing_tests
            if test_fails:
                same_update = -np.inf
                diff_update = 0.0
            else:
                same_update = 0.0
                diff_update = -1.0
            err_update = -10.0
            for p_idx, pred in enumerate(pred_list):
                test_prob_add[pred] += diff_update if True in pred_to_time[p_idx] else same_update
                test_prob_add[pred] += err_update if 'err' in pred_to_time[p_idx] else 0.
                test_prob_wrap[pred] += diff_update if False in pred_to_time[p_idx] else same_update
                test_prob_wrap[pred] += err_update if 'err' in pred_to_time[p_idx] else 0.
                if pred_origin is not None:
                    test_prob_mod[pred] += diff_update if org_pred_to_time != pred_to_time[p_idx] else same_update
                    test_prob_mod[pred] += err_update if 'err' in pred_to_time[p_idx] else 0.
        
        log(f'Dynamic reranking assessed results of: [{total_test_num}] tests.')

        for pred in pred_list:
            for scorer in [test_prob_add, test_prob_wrap, test_prob_mod]:
                scorer[pred] += 1/(1-pred[1])-1 # final update to break ties with naturalness
        return test_prob_add, test_prob_wrap, test_prob_mod

class D4JEvaluationFilter(PatchFilter):
    @staticmethod
    def filter_patches(proj, bugno, patch_list, failing_tests=None, stop_on_pass=False):
        scores = []
        for patch in patch_list:
            print(f'Evaluating idx=[{patch.index}] {patch.name}...', flush=True)
            patch_score = d4j_evaluate_patch(repo_path(proj, bugno), patch, tests_to_run=failing_tests)
            print(f'{patch_score} tests failed.')
            scores.append(patch_score)
            if stop_on_pass and patch_score == 0:
                break
        return list(zip(scores, patch_list))

class Fail2PassFilter(D4JEvaluationFilter):
    @staticmethod
    def filter_patches(proj, bugno, patch_list, failing_tests):
        eval_results = super(Fail2PassFilter, Fail2PassFilter).filter_patches(
            proj, bugno, patch_list, failing_tests
        )
        return [p for s, p in eval_results if s == 0]

class PlausiblePatchFilter(PatchFilter):
    @staticmethod
    def filter_patches(proj, bugno, patch_list, stop_on_pass=False):
        eval_results = super(Fail2PassFilter, Fail2PassFilter).filter_patches(
            proj, bugno, patch_list, stop_on_pass=stop_on_pass
        )
        return [p for s, p in eval_results if s == 0]

class BasicBodyFilter(PatchFilter):
    @staticmethod
    def filter_patches(proj, bugno, body_list, buggy_file, fix_location, failing_tests):
        dummy_patch = ('if (true) ', 0, 'addition')
        patch_list = gen_new_codes(buggy_file, [dummy_patch], {dummy_patch: 0.0}, 'addition', fix_location, bodies=body_list)
        patch_save_dir = f'{PATCH_DIR}/{proj}_{bugno}/'
        for p_idx, patch in enumerate(patch_list):
            patch.update(p_idx+1, buggy_file, fix_location, patch_save_dir)
        eval_results = super(Fail2PassFilter, Fail2PassFilter).filter_patches(
            proj, bugno, patch_list, failing_tests
        )
        scores = list(zip(*eval_results))[0]
        return [p for s, p in sorted(zip(scores, body_list)) if s != -1]

