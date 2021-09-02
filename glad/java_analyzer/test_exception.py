import javalang
from javalang.tree import MethodDeclaration, TryStatement
import os,sys
from os.path import join

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from util import *

def find_exception(project, bug_no, failing_test):
    test_package, test_method = failing_test.split('::', 1)
    test_rel_path = test_package.replace(".", "/")
    test_path = join(ROOT_DIR, f"{project}_{bug_no}", d4j_test_path_prefix(project, bug_no), test_rel_path+".java")
    with open(test_path, 'r') as f:
        code = f.read()
    tree = javalang.parse.parse(code)

    exceptions = set([])
    for _, node in tree:
        if (isinstance(node, MethodDeclaration) and node.name == test_method):
            exceptions = set([])
            for annotation in node.annotations:
                if (not annotation.element is None):
                    for element in annotation.element:
                        exceptions.add(element.value.type.name)
            for body in node.body:
                if (isinstance(body, TryStatement)):
                    for catch in body.catches:
                        exceptions.update(catch.parameter.types)
            break
    return exceptions