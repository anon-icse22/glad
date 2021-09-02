'''Get information to execute relevant tests.'''

import os, sys
import pandas
import javalang

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from jdb_interface import JDBWrapper
from util import *

def get_all_method_names(abs_java_path):
    method_names = []
    with open(abs_java_path) as f:
        code = f.read()
    tree = javalang.parse.parse(code)
    for path, node in tree:
        if isinstance(node, javalang.tree.MethodDeclaration):
            method_names.append(node.name)
    return method_names

def get_rel_test_classes(proj, bug_id):
    rel_test_info = pandas.read_csv(f'{parentdir}/etc_data/relevant_tests/{proj}.csv', header=None)
    rel_test_info = rel_test_info.set_index(0)
    rel_test_dict = rel_test_info.to_dict()[1]
    return rel_test_dict[bug_id].split(';')

def testclass2filename(test_dir_path, class_name):
    test_rel_path = class_name.replace('.', '/') + '.java'
    return os.path.join(test_dir_path, test_rel_path)

def get_relevant_tests(proj, bug_id, fail_test_method_names=[]):
    repo_path = os.path.join(ROOT_DIR, f'{proj}_{bug_id}')
    test_dir_path = os.path.join(repo_path, d4j_test_path_prefix(proj, bug_id))
    
    relevant_test_classes = [e for e in get_rel_test_classes(proj, bug_id) if '$' not in e]
    class2abspath = {
        c: testclass2filename(test_dir_path, c)
        for c in relevant_test_classes
    }
    relevant_tests = {
        c: get_all_method_names(class2abspath[c])
        for c in relevant_test_classes
    }
    reordered_relevant_tests = {
        c: [e for e in relevant_tests[c] if e in fail_test_method_names] + \
           [e for e in relevant_tests[c] if e not in fail_test_method_names]
        for c in relevant_test_classes
    }
    return [f'{c}::{",".join(reordered_relevant_tests[c])}' for c in relevant_test_classes]