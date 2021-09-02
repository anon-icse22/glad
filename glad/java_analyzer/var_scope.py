import javalang
from javalang.tree import VariableDeclarator, FormalParameter, MethodDeclaration, ConstructorDeclaration,\
    ClassDeclaration, IfStatement, ForStatement, BlockStatement,\
    BasicType, ReferenceType, ClassCreator
from collections import deque, OrderedDict, defaultdict
import bisect
import time
import os, sys
from bisect import bisect_right, bisect_left
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from jdb_interface import JDBWrapper
from util import *

IdentifierNodeClass = (VariableDeclarator, FormalParameter, MethodDeclaration, ConstructorDeclaration)
TypeNodeClass = (BasicType, ReferenceType)
NormalParentNodeClass = (IfStatement, ForStatement, MethodDeclaration, BlockStatement)
ClassParentNodeClass = (ClassDeclaration, ClassCreator)
ParentNodeClass = NormalParentNodeClass + ClassParentNodeClass

class Position(javalang.tokenizer.Position):
    @staticmethod
    def from_lineno(lineno, maximize=False):
        return Position(lineno, sys.maxsize if maximize else 1)
    
    def __add__(self, y):
        return Position(self.line + y.line, self.column + y.column)
    
    __radd__ = __add__

def get_code_substr(code, start_position: Position, end_position: Position):
    """
    Get code substring.
    
    It collects all characters which position is greater or equal
    than ``start_position`` and less than ``end_position``.
    """
    lines = code.split("\n")
    start_pos, end_pos = map(lambda x: Position(x.line-1, x.column-1), (start_position, end_position)) # zero-indexing
    collect = []
    for lineno in range(start_pos.line, end_pos.line+1):
        line = lines[lineno]
        if (lineno == start_pos.line and lineno == end_pos.line):
            collect.append(line[start_pos.column:end_pos.column])
        elif (lineno == start_pos.line):
            collect.append(line[start_pos.column:])
        elif (lineno == end_pos.line):
            collect.append(line[:end_pos.column])
        else:
            collect.append(line)
        if (lineno != end_pos.line):
            collect.append("\n")
    
    return "".join(collect)

class StaticJavaAnalyzer:

    def __init__(self, code):
        self.bracketDict = {"curly": dict(), "round": dict()}
        self.code = code
        self.parse_result = javalang.parse.parse(code)
        self.nodes = list(filter(lambda node: isinstance(node, javalang.ast.Node), map(lambda x: x[1], self.parse_result) ))
        stacks = {"curly": deque(), "round": deque()}
        dicts = self.bracketDict
        alias = {"{": "curly", "}": "curly", "(": "round", ")": "round"}

        for token in self.tokenize():
            value = token.value
            if (isinstance(token, javalang.tokenizer.Separator)):
                if (value in {"{", "("}):
                    stacks[alias[value]].append(token)
                if (value in {"}", ")"}):
                    start_token = stacks[alias[value]].pop()
                    dicts[alias[value]][start_token.position] = token.position
        
        dicts["curly"] = OrderedDict(sorted(dicts["curly"].items()))
        dicts["round"] = OrderedDict(sorted(dicts["round"].items()))
        self.mapCurlyBrackets = dicts["curly"]
        self.mapRoundBrackets = dicts["round"]
        return
    
    @classmethod
    def fromfilename(cls, filename):
        with open(filename, 'r') as f:
            code = f.read()
        return cls(code)

    def tokenize(self):
        return javalang.tokenizer.tokenize(self.code)

    def getParentCurlyBracket(self, position, bisect_function=bisect_right):
        return self.getParentBracket(position, "curly", bisect_function)
    
    def getParentRoundBracket(self, position, bisect_function=bisect_right):
        return self.getParentBracket(position, "round", bisect_function)
    
    def getParentBracket(self, position, bracket_type, bisect_function=bisect_right):
        mapBrackets = self.bracketDict[bracket_type]
        keys = list(mapBrackets.keys())
        index = bisect_function(keys, position) - 1
        while (True):
            key = keys[index]
            if (mapBrackets[key] > position):
                return key, mapBrackets[key]
            index -= 1
    
    def getNextBracket(self, position, bracket_type):
        ''' Find bracket satisfying `position`<=`position of bracket`'''
        mapBrackets = self.bracketDict[bracket_type]
        keys = list(mapBrackets.keys())
        index = bisect_left(keys, position)
        key = keys[index]
        return key, mapBrackets[key]

    def findNearestPreviousNode(self, position, filter_function=(lambda _: True)):
        prev_node = None
        for node in self.nodes:
            if (hasattr(node, "_position") and filter_function(node)):
                if (node.position > position):
                    return prev_node
                prev_node = node
        return prev_node

def get_if_condition(absJavaPath, lineno):
    with open(absJavaPath) as f:
        target_code = f.read()
    sja = StaticJavaAnalyzer(target_code)
    if_loc = parse_if_condition(sja, lineno)
    if if_loc is None:
        return None
    full_if_cond = get_code_substr(target_code, *if_loc)
    return full_if_cond


def parse_if_condition(staticJavaAnalyzer, lineno):
    '''
    Parse full if condition on ``lineno``.

    If it fails, return None.
    '''
    if_node = staticJavaAnalyzer.findNearestPreviousNode(
        Position.from_lineno(lineno, True), lambda node: isinstance(node, IfStatement))
    if (if_node is None): return None   # No previous if statement
    condition_start, condition_end = staticJavaAnalyzer.getNextBracket(if_node.position, "round")
    if (condition_start.line <= lineno <= condition_end.line):
        return condition_start, condition_end + Position(0, 1)
    else: return None       # previous if statement's condition is not on ``lineno``.`

def parse_if_condition_from_else(staticJavaAnalyzer, lineno):
    else_block_node = staticJavaAnalyzer.findNearestPreviousNode(
        Position.from_lineno(lineno, True), lambda node: isinstance(node, javalang.tree.BlockStatement))
    assert (else_block_node.position.line == lineno)
    track_lineno = lineno
    while True:
        if_block_node = staticJavaAnalyzer.findNearestPreviousNode(
            Position.from_lineno(track_lineno, True), lambda node: isinstance(node, javalang.tree.IfStatement))
        if (if_block_node is None): return None
        if if_block_node.else_statement == else_block_node:
            break
        else: track_lineno = if_block_node.position.line - 1
    condition_start, condition_end = staticJavaAnalyzer.getNextBracket(if_block_node.position, "round")
    return condition_start, condition_end + Position(0, 1)

class Identifier:
    class_suffix = None # set later, as requires global information

    def __init__(self, path, node, bracketCollector):
        assert(isinstance(node, IdentifierNodeClass))
        self.name = node.name
        self.node = node
        self.path = path
        self.declared_position = self.getDeclaredPosition()
        self.scope_start, self.scope_end = self.getScope(bracketCollector)
        # update below later (with self.update()), only when necessary
        self.type = self.getType() # gets type for method/constructor only
        self.methods = []
        self.fields = []
        self.nonprimitive_fields = []
        # None if the type is not a method
        self.method_start, self.method_end = self.getMethodRange(bracketCollector)
        self.class_suffix = None 
    
    # Traverse from target node to root until position value exists
    def getDeclaredPosition(self):
        if hasattr(self.node, "position") and self.node.position != None:
            return self.node.position
        for pathUnit in reversed(self.path):
            if (hasattr(pathUnit, "position") and pathUnit.position != None):
                return pathUnit.position
    
    def getMethodRange(self, bracketCollector):
        if self.type != "method":
            return None, None
        mapCurlyBrackets = bracketCollector.mapCurlyBrackets
        keys = list(mapCurlyBrackets.keys())
        index = bisect.bisect_right(keys, self.declared_position)
        try:
            return self.node.position, mapCurlyBrackets[keys[index]]
        except IndexError:
            return None, None # bodyless function

    def getScope(self, bracketCollector):
        parentBracketStart, parentBracketEnd = bracketCollector.getParentCurlyBracket(self.declared_position, bisect.bisect_left)
        parent = None
        for pathUnit in reversed(self.path):
            if (isinstance(pathUnit, ParentNodeClass)):
                parent = pathUnit
                break

        if (isinstance(parent, ClassParentNodeClass)):
            scopeStart = parentBracketStart
        else:
            scopeStart = self.declared_position

        return scopeStart, parentBracketEnd
    
    # jdb-related methods
    def getMethods(self, jdbw):
        if (self.type in ("method", "constructor")):
            return []
        else:
            method_w_signature = [x[1] for x in jdbw.get_var_methods(self.name)]
            method_names = [x[:x.index('(')] for x in method_w_signature]
            unique_method_names = set(method_names)
            return unique_method_names
    
    def getFields(self, jdbw):
        if (self.type in ("method", "constructor")):
            return []
        else:
            fieldsWithTypes = jdbw.get_var_fields(self.name)
            self.nonprimitive_fields = [x[1] for x in fieldsWithTypes if '.' in x[0]]
            return [x[1] for x in fieldsWithTypes]
    
    def getType(self, jdbw = None):
        if (isinstance(self.node, MethodDeclaration)):
            return "method"
        elif (isinstance(self.node, ConstructorDeclaration)):
            return "constructor"
        elif (isinstance(self.node, (FormalParameter, VariableDeclarator))):
            if jdbw is not None:
                return jdbw.get_var_type(self.name)
            else:
                return None # update later
        else: raise TypeError("Not implemented")

    def update(self, jdbw):
        try:
            self.type = self.getType(jdbw)
            if '.' in self.type: # non-primitive type
                self.methods = self.getMethods(jdbw)
                self.fields = self.getFields(jdbw)
            return True
        except ValueError:
            return False

def parseIdentifiers(absJavaPath):
    '''Performs purely static analysis of java file.'''
    with open(absJavaPath, 'r') as f:
        code = f.read()
    identifierList = []
    tree = javalang.parse.parse(code)
    bracketCollector = StaticJavaAnalyzer(code)

    for path, node in tree:
        if (isinstance(node, IdentifierNodeClass)):
            identifier = Identifier(path, node, bracketCollector)
            identifierList.append(identifier)
    
    # resolve which class method belongs to
    methodIdentifiers = [e for e in identifierList if e.type == "method"]
    methodClasspaths = [[n for n in e.path if type(n) in ClassParentNodeClass]
                        for e in methodIdentifiers]
    creatorCounter = defaultdict(int)
    creatorTracker = defaultdict(lambda: None)
    for mid, mpath in zip(methodIdentifiers, methodClasspaths):
        classpathSuffix = ''
        for cidx, classNode in enumerate(mpath[1:]):
            classpathSuffix += '$'
            if type(classNode) == ClassDeclaration:
                classpathSuffix += classNode.name
            else: # class creator
                ancestry = tuple(mpath[:cidx])
                if creatorTracker[ancestry] != mpath[cidx]:
                    creatorTracker[ancestry] = mpath[cidx]
                    creatorCounter[ancestry] += 1
                classpathSuffix += str(creatorCounter[ancestry])
        mid.class_suffix = classpathSuffix
    return identifierList

def enhanceIdentifiers(relJavaPath, targetTest, projectName, bugNo, lineno, idList):
    # activate JDB
    classPath = d4jPathToClasspath(projectName, bugNo, relJavaPath, lineno)
    jdbw = JDBWrapper(D4J_HOME, projectName, bugNo, targetTest)
    valid_breakpoint = JDBWrapper.get_valid_breakpoint(repo_path(projectName, bugNo) + relJavaPath, lineno)
    breakpoint_succ = jdbw.set_breakpoint(classPath, valid_breakpoint)
    run_succ = jdbw.run_process()
    assert breakpoint_succ and run_succ, (breakpoint_succ, run_succ)

    scopeValidIDList = list(getValidIdentifiers(lineno, idList))
    updateResults = [var.update(jdbw) for var in scopeValidIDList]
    validIdList = [var for var, res in zip(scopeValidIDList, updateResults) if res]
    jdbw.exit()
    return validIdList

def getValidIdentifiers(lineNumber, identifierList):
    return filter(lambda identifier: identifier.scope_end.line > lineNumber > identifier.scope_start.line, identifierList)

def getMethodContainingLine(lineNumber, identifierList):
    targetMethodNodes = [e for e in identifierList 
                         if (e.type == "method" and e.method_start.line <= lineNumber <= e.method_end.line)]
    assert len(targetMethodNodes) == 1, targetMethodNodes
    return targetMethodNodes[0]

def getClasspathSuffix(absJavaPath, line):
    identifierList = parseIdentifiers(absJavaPath)
    targetMethodNode = getMethodContainingLine(line, identifierList)
    return targetMethodNode.class_suffix

def d4jPathToClasspath(proj, bugNum, relJavaPath, line):
    import util
    d4jpp = util.d4j_path_prefix(proj, bugNum)  
    pre_classpath = relJavaPath.removeprefix(d4jpp).removesuffix('.java')
    classpath_basic = pre_classpath.replace('/', '.')
    classpath_full = classpath_basic + getClasspathSuffix(repo_path(proj, bugNum)+relJavaPath, line)
    return classpath_full