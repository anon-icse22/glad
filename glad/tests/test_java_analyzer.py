from context import *
import unittest

class TestVarScope(unittest.TestCase):

    def test_parse_exception(self):
        self.assertEqual(find_exception("Math", 48, "org.apache.commons.math.analysis.solvers.RegulaFalsiSolverTest::testIssue631"), {'ConvergenceException'})
        self.assertEqual(find_exception("Math", 73, "org.apache.commons.math.analysis.solvers.BrentSolverTest::testBadEndpoints"), {'IllegalArgumentException'})
        self.assertEqual(find_exception("Math", 99, "org.apache.commons.math.util.MathUtilsTest::testGcd"), {'ArithmeticException'})
        self.assertEqual(find_exception("Time", 1, "org.joda.time.TestPartial_Constructors::testConstructorEx7_TypeArray_intArray"), {"IllegalArgumentException"})
        self.assertEqual(find_exception("Math", 3, "org.apache.commons.math3.util.MathArraysTest::testLinearCombinationWithSingleElementArray"), set())
    
    def test_parse_if_condition_Math101(self):
        # http://program-repair.org/defects4j-dissection/#!/bug/Math/101 
        expectedResult = """(
            source.substring(startIndex, endIndex).compareTo(
            getImaginaryCharacter()) != 0)"""
        with open(ROOT_DIR + "Math_101/src/java/org/apache/commons/math/complex/ComplexFormat.java", 'r') as f:
            code = f.read()
        sja = StaticJavaAnalyzer(code)

        self.assertEqual(parse_if_condition(sja, 364), None)
        self.assertEqual("(im == null)", get_code_substr(sja.code, *(parse_if_condition(sja, 365))))
        self.assertEqual(parse_if_condition(sja, 366), None)
        self.assertEqual(parse_if_condition(sja, 376), None)
        self.assertEqual(expectedResult, get_code_substr(sja.code, *(parse_if_condition(sja, 377))))
        self.assertEqual(expectedResult, get_code_substr(sja.code, *(parse_if_condition(sja, 378))))
        self.assertEqual(expectedResult, get_code_substr(sja.code, *(parse_if_condition(sja, 379))))
        self.assertEqual(parse_if_condition(sja, 380), None)
        self.assertEqual("(index < n)", get_code_substr(sja.code, *(parse_if_condition(sja, 415))))
        return
    

    def test_breakpoint_forwarding_from_else_branch(self):
        with open("./data/If.java", 'r') as f:
            code = f.read()
        sja = StaticJavaAnalyzer(code)

        self.assertEqual(get_code_substr(sja.code, *parse_if_condition_from_else(sja, 8)), "(a == 4)")
        self.assertEqual(get_code_substr(sja.code, *parse_if_condition_from_else(sja, 15)), "(a == 5)")
        self.assertEqual(get_code_substr(sja.code, *parse_if_condition_from_else(sja, 23)), "(a == 6)")
        return

    def test_parse_identifiers_coverage(self):
        filenames = [
            repo_path("Closure",15) + "src/com/google/javascript/jscomp/FlowSensitiveInlineVariables.java",
            repo_path("Closure", 33) + "src/com/google/javascript/rhino/jstype/PrototypeObjectType.java",
            repo_path("Closure", 38) + "src/com/google/javascript/jscomp/CodeConsumer.java"
            ] 
        
        for filename in filenames:
            sja = StaticJavaAnalyzer.fromfilename(filename)
            ids = parseIdentifiers(filename)
            ids_names = set([identifier.name for identifier in ids])
            token_ids = set([token.value for token in sja.tokenize() if isinstance(token, javalang.tokenizer.Identifier)])

            self.assertEqual(ids_names.difference(token_ids), set())

if __name__ == '__main__':
    unittest.main(verbosity=2)