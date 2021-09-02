import os
import subprocess as sp
import difflib
from util import *

def write_patch_file(abs_jfile_path, new_lines, patch_file_loc):
    with open(abs_jfile_path) as f:
        org_lines = f.readlines()
    _, rel_jfile_path = parse_abs_path(abs_jfile_path)
    header = f'diff --git a/{rel_jfile_path} b/{rel_jfile_path}\n'
    main_diff = ''.join(difflib.unified_diff(org_lines, new_lines, fromfile=f'a/{rel_jfile_path}', tofile=f'b/{rel_jfile_path}'))

    with open(patch_file_loc, 'w') as f:
        print(header+main_diff, file=f, end='')

def compile_repo(repo_dir_path):
    curr_dir = os.getcwd()
    os.chdir(repo_dir_path)
    sp.run(['defects4j', 'compile'])
    os.chdir(curr_dir)

def apply_patch(repo_dir_path, patch_file_loc):
    curr_dir = os.getcwd()
    os.chdir(repo_dir_path)
    sp.run(['git', 'apply', '--ignore-whitespace', '--ignore-space-change', 
            os.path.join(curr_dir, patch_file_loc)])
    os.chdir(curr_dir)

def run_select_tests(repo_dir_path, test_names):
    # do not use to evaluate all tests, will be more inefficient;
    # returns True upon test pass, and False upon test fail.
    curr_dir = os.getcwd()
    os.chdir(repo_dir_path)
    test_results = []
    for test_name in test_names:
        test_process = sp.run(['defects4j', 'test', '-t', test_name], capture_output=True)
        captured_stdout = test_process.stdout.decode()
        if len(captured_stdout) == 0:
            test_results.append(-1)
        else:
            stdout_lines = captured_stdout.split('\n')
            failed_test_num = int(stdout_lines[0].removeprefix('Failing tests: '))
            failed_tests = [e.strip(' - ') for e in stdout_lines[1:]] # names of failed tests
            test_results.append(int(failed_test_num == 0))
    os.chdir(curr_dir)
    return test_results

def run_single_test(repo_dir_path, test_name):
    return run_select_tests(repo_dir_path, [test_name])[0]

def run_tests(repo_dir_path):
    '''Returns failing test number.'''
    curr_dir = os.getcwd()
    os.chdir(repo_dir_path)
    test_process = sp.run(['timeout', '3m', 'defects4j', 'test'], capture_output=True)
    captured_stdout = test_process.stdout.decode()
    if len(captured_stdout) == 0:
        return -1 # likely compile error, all tests failed
    else:
        stdout_lines = captured_stdout.split('\n')
        failed_test_num = int(stdout_lines[0].removeprefix('Failing tests: '))
        failed_tests = [e.strip(' - ') for e in stdout_lines[1:]]
        return failed_test_num

def restore_repo(repo_dir_path):
    curr_dir = os.getcwd()
    os.chdir(repo_dir_path)
    sp.run(['git', 'reset', '--hard', 'HEAD'])
    os.chdir(curr_dir)

def d4j_evaluate_patch(repo_dir_path, patch, tests_to_run=None):
    patch_file_loc = patch.patch_path
    apply_patch(repo_dir_path, patch_file_loc)
    if tests_to_run is None:
        test_result = run_tests(repo_dir_path)
    else:
        test_results = run_select_tests(repo_dir_path, tests_to_run)
        test_result = test_results.count(0) if (-1 not in test_results) else -1
    restore_repo(repo_dir_path)
    return test_result