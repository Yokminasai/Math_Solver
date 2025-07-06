#!/usr/bin/env python3
"""
Comprehensive test suite for the Mathematical Problem Solver
Tests various types of mathematical problems and edge cases
"""

import unittest
import sys
import os
import json
import traceback
from typing import Dict, Any, List
from math_solver import MathSolver
from advanced_math import AdvancedMathSolver
import math
import numpy as np
from sympy import symbols, solve, Eq, simplify

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from math_solver import MathSolver
except ImportError as e:
    print(f"Error importing MathSolver: {e}")
    sys.exit(1)

class TestMathSolver(unittest.TestCase):
    """Comprehensive test suite for MathSolver"""
    
    def setUp(self):
        """Set up test environment"""
        self.solver = MathSolver()
        self.advanced_solver = AdvancedMathSolver()
        self.test_results = []
        
    def tearDown(self):
        """Clean up after tests"""
        pass
    
    def log_test_result(self, test_name: str, problem: str, result: Dict[str, Any], expected_type: str = None):
        """Log test results for analysis"""
        success = "error" not in result
        if expected_type:
            success = success and result.get("type") == expected_type
            
        self.test_results.append({
            "test_name": test_name,
            "problem": problem,
            "success": success,
            "result": result,
            "expected_type": expected_type
        })
        
        if not success:
            print(f"\n❌ {test_name} FAILED")
            print(f"Problem: {problem}")
            print(f"Result: {result}")
        else:
            print(f"✅ {test_name} PASSED")
    
    def test_basic_arithmetic(self):
        """Test basic arithmetic operations"""
        test_cases = [
            ("2 + 3", 5),
            ("10 - 4", 6),
            ("3 * 4", 12),
            ("15 / 3", 5),
            ("2^3", 8),
            ("2 + 3 * 4", 14),
            ("(2 + 3) * 4", 20),
            ("sqrt(16)", 4),
            ("sin(pi/2)", 1),
            ("cos(pi)", -1),
            ("log(e)", 1),
            ("abs(-5)", 5)
        ]
        
        for problem, expected in test_cases:
            result = self.solver.solve_basic_arithmetic(problem)
            self.log_test_result("Basic Arithmetic", problem, result)
            self.assertIsNotNone(result)
            self.assertIn("answer", result)
            self.assertAlmostEqual(float(result["answer"]), expected, places=6)
            self.assertIn("steps", result)
            self.assertTrue(len(result["steps"]) > 0)
    
    def test_equations(self):
        """Test equation solving"""
        test_cases = [
            ("x + 5 = 10", [5]),
            ("x^2 = 16", [-4, 4]),
            ("2x + 3 = 7", [2]),
            ("x^2 + 2x + 1 = 0", [-1]),
            ("x^2 - 4 = 0", [-2, 2])
        ]
        
        for problem, expected in test_cases:
            result = self.solver.solve_equation(problem)
            self.log_test_result("Equations", problem, result)
            self.assertIsNotNone(result)
            self.assertIn("solutions", result)
            solutions = sorted([float(x) for x in result["solutions"]])
            expected = sorted(expected)
            self.assertEqual(len(solutions), len(expected))
            for s, e in zip(solutions, expected):
                self.assertAlmostEqual(s, e, places=6)
            self.assertIn("steps", result)
            self.assertTrue(len(result["steps"]) > 0)
    
    def test_system_of_equations(self):
        """Test system of equations solving"""
        test_cases = [
            (["x + y = 5", "x - y = 1"], {"x": 3, "y": 2}),
            (["2x + y = 7", "x - y = 1"], {"x": 3, "y": 1}),
            (["x + 2y = 8", "3x - y = 7"], {"x": 3, "y": 2.5})
        ]
        
        for equations, expected in test_cases:
            result = self.solver.solve_system_of_equations(equations)
            self.log_test_result("System of Equations", str(equations), result)
            self.assertIsNotNone(result)
            self.assertIn("solutions", result)
            for var, val in expected.items():
                self.assertIn(var, result["solutions"])
                self.assertAlmostEqual(float(result["solutions"][var]), val, places=6)
            self.assertIn("steps", result)
            self.assertTrue(len(result["steps"]) > 0)
    
    def test_calculus(self):
        """Test calculus problems"""
        test_cases = [
            ("derivative of x^2", "2*x"),
            ("integral of x^2", "x^3/3"),
            ("limit of x^2 as x approaches infinity", "oo"),
            ("limit of sin(x)/x as x approaches 0", "1")
        ]
        
        for problem, expected in test_cases:
            result = self.solver.solve_calculus(problem)
            self.log_test_result("Calculus", problem, result)
            self.assertIsNotNone(result)
            self.assertIn("answer", result)
            if expected == "oo":
                self.assertTrue(math.isinf(float(result["answer"])))
            else:
                self.assertEqual(str(result["answer"]).replace(" ", ""), expected.replace(" ", ""))
            self.assertIn("steps", result)
            self.assertTrue(len(result["steps"]) > 0)
    
    def test_trigonometry(self):
        """Test trigonometry problems"""
        test_cases = [
            ("sin(pi/2)", 1),
            ("cos(pi)", -1),
            ("tan(pi/4)", 1),
            ("arcsin(1)", math.pi/2),
            ("arccos(0)", math.pi/2),
            ("arctan(1)", math.pi/4)
        ]
        
        for problem, expected in test_cases:
            result = self.solver.solve_trigonometry(problem)
            self.log_test_result("Trigonometry", problem, result)
            self.assertIsNotNone(result)
            self.assertIn("answer", result)
            self.assertAlmostEqual(float(result["answer"]), expected, places=6)
            self.assertIn("steps", result)
            self.assertTrue(len(result["steps"]) > 0)
    
    def test_statistics(self):
        """Test statistics problems"""
        test_cases = [
            ("mean of [1, 2, 3, 4, 5]", 3),
            ("median of [1, 2, 3, 4, 5]", 3),
            ("mode of [1, 2, 2, 3, 4]", 2),
            ("variance of [1, 2, 3, 4, 5]", 2),
            ("standard deviation of [1, 2, 3, 4, 5]", math.sqrt(2))
        ]
        
        for problem, expected in test_cases:
            result = self.solver.solve_statistics(problem)
            self.log_test_result("Statistics", problem, result)
            self.assertIsNotNone(result)
            self.assertIn("answer", result)
            if isinstance(expected, list):
                self.assertEqual(sorted(result["answer"]), sorted(expected))
            else:
                self.assertAlmostEqual(float(result["answer"]), expected, places=6)
            self.assertIn("steps", result)
            self.assertTrue(len(result["steps"]) > 0)
    
    def test_geometry(self):
        """Test geometry problems"""
        test_cases = [
            ("area of circle radius 2", math.pi * 4),
            ("perimeter of circle radius 2", math.pi * 4),
            ("area of rectangle length 3 width 4", 12),
            ("perimeter of rectangle length 3 width 4", 14),
            ("area of triangle base 3 height 4", 6),
            ("volume of sphere radius 2", (4/3) * math.pi * 8),
            ("volume of cylinder radius 2 height 3", math.pi * 4 * 3)
        ]
        
        for problem, expected in test_cases:
            result = self.solver.solve_geometry(problem)
            self.log_test_result("Geometry", problem, result)
            self.assertIsNotNone(result)
            self.assertIn("answer", result)
            self.assertAlmostEqual(float(result["answer"]), expected, places=6)
            self.assertIn("steps", result)
            self.assertTrue(len(result["steps"]) > 0)
    
    def test_number_theory(self):
        """Test number theory problems"""
        test_cases = [
            ("is 17 prime", True),
            ("is 15 prime", False),
            ("gcd of 12 and 18", 6),
            ("lcm of 12 and 18", 36),
            ("prime factors of 12", [2, 2, 3]),
            ("factors of 12", [1, 2, 3, 4, 6, 12])
        ]
        
        for problem, expected in test_cases:
            result = self.solver.solve_number_theory(problem)
            self.log_test_result("Number Theory", problem, result)
            self.assertIsNotNone(result)
            self.assertIn("answer", result)
            if isinstance(expected, list):
                self.assertEqual(sorted(result["answer"]), sorted(expected))
            else:
                self.assertEqual(result["answer"], expected)
            self.assertIn("steps", result)
            self.assertTrue(len(result["steps"]) > 0)
    
    def test_linear_algebra(self):
        """Test linear algebra problems"""
        test_cases = [
            ("determinant of [[1, 2], [3, 4]]", -2),
            ("rank of [[1, 2], [2, 4]]", 1),
            ("trace of [[1, 2], [3, 4]]", 5)
        ]
        
        for problem, expected in test_cases:
            result = self.solver.solve_linear_algebra(problem)
            self.log_test_result("Linear Algebra", problem, result)
            self.assertIsNotNone(result)
            self.assertIn("answer", result)
            if isinstance(expected, list):
                self.assertEqual(sorted(result["answer"]), sorted(expected))
            else:
                self.assertAlmostEqual(float(result["answer"]), expected, places=6)
            self.assertIn("steps", result)
            self.assertTrue(len(result["steps"]) > 0)
    
    def test_advanced_problems(self):
        """Test advanced mathematical problems"""
        test_cases = [
            ("residue of 1/(z^2+1) at z=i", "1/(2i)"),
            ("fourier series of x^2 from -pi to pi", "pi^2/3 + sum(4*(-1)^n/(n^2) * cos(n*x))"),
            ("laplace transform of t*e^(-t)", "1/(s+1)^2"),
            ("solve pde d^2u/dx^2 = d^2u/dt^2", "u(x,t) = F(x+t) + G(x-t)")
        ]
        for problem, expected in test_cases:
            result = self.advanced_solver.solve_advanced_problem(problem)
            self.log_test_result("Advanced Problems", problem, result)
            self.assertIn('answer', result)
            self.assertTrue(self._expressions_equal(str(result['answer']), expected))
    
    def test_edge_cases(self):
        """Test edge cases and error handling"""
        test_cases = [
            ("", "Invalid input"),
            ("1/0", "Division by zero"),
            ("log(-1)", "Domain error"),
            ("sqrt(-1)", "Complex result"),
            ("solve x^5 + x + 1 = 0", "No algebraic solution"),
            ("∞ + ∞", "Indeterminate form"),
            ("0/0", "Indeterminate form")
        ]
        for problem, expected_error in test_cases:
            result = self.solver.solve_problem(problem)
            self.log_test_result("Edge Cases", problem, result)
            self.assertIn('error', result)
            self.assertIn(expected_error, result['error'])
    
    def _expressions_equal(self, expr1: str, expr2: str) -> bool:
        """Compare two mathematical expressions for equality"""
        try:
            x = symbols('x')
            expr1 = expr1.replace('^', '**')
            expr2 = expr2.replace('^', '**')
            diff = simplify(eval(expr1) - eval(expr2))
            return diff == 0
        except:
            return expr1.strip() == expr2.strip()
    
    def generate_test_report(self):
        """Generate a comprehensive test report"""
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result["success"])
        failed_tests = total_tests - passed_tests
        
        report = {
            "summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "success_rate": (passed_tests / total_tests * 100) if total_tests > 0 else 0
            },
            "test_results": self.test_results
        }
        
        # Save report to file
        with open("test_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"TEST REPORT SUMMARY")
        print(f"{'='*60}")
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {failed_tests}")
        print(f"Success Rate: {report['summary']['success_rate']:.1f}%")
        print(f"{'='*60}")
        
        # Print failed tests
        if failed_tests > 0:
            print(f"\nFAILED TESTS:")
            for result in self.test_results:
                if not result["success"]:
                    print(f"❌ {result['test_name']}: {result['problem']}")
                    if "error" in result["result"]:
                        print(f"   Error: {result['result']['error']}")
        
        return report

class TestAdvancedMathSolver(unittest.TestCase):
    def setUp(self):
        """Set up test cases"""
        self.solver = AdvancedMathSolver()

    def test_complex_analysis(self):
        """Test complex analysis problems"""
        test_cases = [
            ("residue of 1/(z^2+1) at i", "0.5*I"),
            ("laurent series of 1/z around 0", "1/z")
        ]
        
        for problem, expected in test_cases:
            result = self.solver.solve_complex_analysis(problem)
            self.assertIsNotNone(result)
            self.assertIn("answer", result)
            self.assertEqual(str(result["answer"]).replace(" ", ""), expected.replace(" ", ""))
            self.assertIn("steps", result)
            self.assertTrue(len(result["steps"]) > 0)

    def test_abstract_algebra(self):
        """Test abstract algebra problems"""
        test_cases = [
            ("order of group [1, -1, i, -i]", 4),
            ("subgroups of group [1, -1]", [[1], [1, -1]])
        ]
        
        for problem, expected in test_cases:
            result = self.solver.solve_abstract_algebra(problem)
            self.assertIsNotNone(result)
            self.assertIn("answer", result)
            if isinstance(expected, list):
                self.assertEqual(sorted(str(result["answer"])), sorted(str(expected)))
            else:
                self.assertEqual(result["answer"], expected)
            self.assertIn("steps", result)
            self.assertTrue(len(result["steps"]) > 0)

    def test_topology(self):
        """Test topology problems"""
        test_cases = [
            ("is [0,1] connected", True),
            ("is [0,1] compact", True)
        ]
        
        for problem, expected in test_cases:
            result = self.solver.solve_topology(problem)
            self.assertIsNotNone(result)
            self.assertIn("answer", result)
            self.assertEqual(result["answer"], expected)
            self.assertIn("steps", result)
            self.assertTrue(len(result["steps"]) > 0)

    def test_differential_geometry(self):
        """Test differential geometry problems"""
        test_cases = [
            ("curvature of circle radius 1", 1),
            ("geodesics on sphere", "great circles")
        ]
        
        for problem, expected in test_cases:
            result = self.solver.solve_differential_geometry(problem)
            self.assertIsNotNone(result)
            self.assertIn("answer", result)
            if isinstance(expected, (int, float)):
                self.assertAlmostEqual(float(result["answer"]), expected, places=6)
            else:
                self.assertEqual(str(result["answer"]).replace(" ", ""), expected.replace(" ", ""))
            self.assertIn("steps", result)
            self.assertTrue(len(result["steps"]) > 0)

    def test_algebraic_geometry(self):
        """Test algebraic geometry problems"""
        test_cases = [
            ("variety of x^2+y^2=1", "circle"),
            ("intersection of x^2+y^2=1 and y=x", "[1/sqrt(2), 1/sqrt(2)], [-1/sqrt(2), -1/sqrt(2)]")
        ]
        
        for problem, expected in test_cases:
            result = self.solver.solve_algebraic_geometry(problem)
            self.assertIsNotNone(result)
            self.assertIn("answer", result)
            self.assertEqual(str(result["answer"]).replace(" ", ""), expected.replace(" ", ""))
            self.assertIn("steps", result)
            self.assertTrue(len(result["steps"]) > 0)

    def test_number_theory_advanced(self):
        """Test advanced number theory problems"""
        test_cases = [
            ("zeta of 2", str(math.pi**2/6)),
            ("class number of Q(sqrt(-5))", 2)
        ]
        
        for problem, expected in test_cases:
            result = self.solver.solve_number_theory_advanced(problem)
            self.assertIsNotNone(result)
            self.assertIn("answer", result)
            self.assertEqual(str(result["answer"]).replace(" ", ""), expected.replace(" ", ""))
            self.assertIn("steps", result)
            self.assertTrue(len(result["steps"]) > 0)

    def test_functional_analysis(self):
        """Test functional analysis problems"""
        test_cases = [
            ("spectrum of [[1, 0], [0, 2]]", "{1, 2}"),
            ("norm of [3, 4]", 5)
        ]
        
        for problem, expected in test_cases:
            result = self.solver.solve_functional_analysis(problem)
            self.assertIsNotNone(result)
            self.assertIn("answer", result)
            if isinstance(expected, (int, float)):
                self.assertAlmostEqual(float(result["answer"]), expected, places=6)
            else:
                self.assertEqual(str(result["answer"]).replace(" ", ""), expected.replace(" ", ""))
            self.assertIn("steps", result)
            self.assertTrue(len(result["steps"]) > 0)

    def test_optimization(self):
        """Test optimization problems"""
        test_cases = [
            ("minimize x^2+y^2 subject to x+y=1", {"optimal_value": 0.5, "optimal_point": [0.5, 0.5]}),
            ("maximize x+y subject to x^2+y^2<=1", {"optimal_value": math.sqrt(2), "optimal_point": [1/math.sqrt(2), 1/math.sqrt(2)]})
        ]
        
        for problem, expected in test_cases:
            result = self.solver.solve_optimization(problem)
            self.assertIsNotNone(result)
            self.assertIn("answer", result)
            self.assertAlmostEqual(result["answer"]["optimal_value"], expected["optimal_value"], places=6)
            for x, y in zip(result["answer"]["optimal_point"], expected["optimal_point"]):
                self.assertAlmostEqual(x, y, places=6)
            self.assertIn("steps", result)
            self.assertTrue(len(result["steps"]) > 0)

    def test_chaos_theory(self):
        """Test chaos theory problems"""
        test_cases = [
            ("lyapunov exponent of logistic map x->4x(1-x) at x0=0.1", 0.693),
            ("bifurcation points of logistic map", [1, 3, 3.57])
        ]
        
        for problem, expected in test_cases:
            result = self.solver.solve_chaos_theory(problem)
            self.assertIsNotNone(result)
            self.assertIn("answer", result)
            if isinstance(expected, (int, float)):
                self.assertAlmostEqual(float(result["answer"]), expected, places=2)
            else:
                for x, y in zip(sorted(result["answer"]), sorted(expected)):
                    self.assertAlmostEqual(x, y, places=2)
            self.assertIn("steps", result)
            self.assertTrue(len(result["steps"]) > 0)

    def test_quantum_mechanics(self):
        """Test quantum mechanics problems"""
        test_cases = [
            ("eigenstate of harmonic oscillator H=p^2/2+x^2/2", "exp(-x^2/2)/pi^(1/4)"),
            ("expectation value of x^2 in ground state of harmonic oscillator", 0.5)
        ]
        
        for problem, expected in test_cases:
            result = self.solver.solve_quantum_mechanics(problem)
            self.assertIsNotNone(result)
            self.assertIn("answer", result)
            if isinstance(expected, (int, float)):
                self.assertAlmostEqual(float(result["answer"]), expected, places=6)
            else:
                self.assertEqual(str(result["answer"]).replace(" ", ""), expected.replace(" ", ""))
            self.assertIn("steps", result)
            self.assertTrue(len(result["steps"]) > 0)

def run_performance_test():
    """Run performance tests"""
    print("\nRunning Performance Tests...")
    solver = MathSolver()
    
    import time
    
    # Test arithmetic performance
    start_time = time.time()
    for i in range(100):
        solver.solve_basic_arithmetic(f"{i} + {i+1}")
    arithmetic_time = time.time() - start_time
    
    # Test equation solving performance
    start_time = time.time()
    for i in range(50):
        solver.solve_equation(f"x + {i} = {i+1}")
    equation_time = time.time() - start_time
    
    print(f"Arithmetic (100 ops): {arithmetic_time:.3f}s")
    print(f"Equations (50 ops): {equation_time:.3f}s")

def main():
    """Run all tests"""
    print("Starting Comprehensive Math Solver Tests...")
    print("="*60)
    
    # Run unit tests
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run performance tests
    run_performance_test()
    
    # Generate test report
    test_suite = TestMathSolver()
    test_suite.setUp()
    
    # Run all test methods
    test_methods = [method for method in dir(test_suite) if method.startswith('test_')]
    for method_name in test_methods:
        method = getattr(test_suite, method_name)
        if callable(method):
            try:
                method()
            except Exception as e:
                print(f"Error in {method_name}: {e}")
    
    # Generate report
    report = test_suite.generate_test_report()
    
    print(f"\nTest report saved to: test_report.json")
    return report

if __name__ == "__main__":
    main() 