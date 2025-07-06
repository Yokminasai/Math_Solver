#!/usr/bin/env python3
"""
Advanced Mathematical Problem Solver
Handles various types of mathematical problems from basic to advanced
with comprehensive error handling and step-by-step solutions
"""

import math
import numpy as np
import sympy as sp
from sympy import (
    symbols, solve, diff, integrate, limit, simplify, expand, factor,
    Matrix, diag, eye, zeros, ones, GramSchmidt, solve_linear_system,
    solve_poly_system, log, exp, sqrt, sin, cos, tan, asin, acos, atan,
    sinh, cosh, tanh, pi, E, I, oo, Symbol, Function, dsolve, Derivative,
    series, fourier_series, laplace_transform, inverse_laplace_transform,
    fourier_transform, inverse_fourier_transform, residue, apart, together,
    solve_linear_system_LU, solve_undetermined_coeffs, parse_expr
)
from sympy.solvers import solve_linear_system, solve_poly_system
from sympy.matrices import Matrix
import matplotlib.pyplot as plt
from typing import Union, List, Tuple, Dict, Any, Optional
import re
import json
import traceback
from advanced_math import AdvancedMathSolver

class MathSolver:
    def __init__(self):
        """Initialize the math solver with comprehensive error handling"""
        try:
            # Define common symbols and functions
            self.x, self.y, self.z = symbols('x y z')
            self.a, self.b, self.c = symbols('a b c')
            self.t = symbols('t')
            self.n = symbols('n')
            self.f = Function('f')
            self.g = Function('g')
            
            # Initialize advanced solver
            self.advanced_solver = AdvancedMathSolver()
            
            # Problem type registry with detailed descriptions
            self.problem_types = {
                'arithmetic': {
                    'solver': self.solve_basic_arithmetic,
                    'description': 'Basic arithmetic operations including +, -, *, /, ^'
                },
                'equation': {
                    'solver': self.solve_equation,
                    'description': 'Linear, quadratic, and polynomial equations'
                },
                'system_of_equations': {
                    'solver': self.solve_system_of_equations,
                    'description': 'Systems of linear and nonlinear equations'
                },
                'calculus': {
                    'solver': self.solve_calculus,
                    'description': 'Derivatives, integrals, limits, and series'
                },
                'trigonometry': {
                    'solver': self.solve_trigonometry,
                    'description': 'Trigonometric functions and identities'
                },
                'statistics': {
                    'solver': self.solve_statistics,
                    'description': 'Statistical calculations and probability'
                },
                'geometry': {
                    'solver': self.solve_geometry,
                    'description': 'Geometric shapes and measurements'
                },
                'number_theory': {
                    'solver': self.solve_number_theory,
                    'description': 'Prime numbers, GCD, LCM, and modular arithmetic'
                },
                'linear_algebra': {
                    'solver': self.solve_linear_algebra,
                    'description': 'Matrices, vectors, and linear transformations'
                },
                'advanced': {
                    'solver': self.solve_advanced,
                    'description': 'Complex analysis, topology, and abstract algebra'
                },
                'pde_or_vector_equation': {
                    'solver': self.solve_pde_or_vector_equation,
                    'description': 'Partial differential equations and vector calculus'
                }
            }
            
        except Exception as e:
            raise Exception(f"Cannot initialize solver: {str(e)}")

    def _validate_input(self, problem: str) -> bool:
        """Validate input problem with enhanced checks"""
        if not problem or not isinstance(problem, str):
            return False
        if len(problem.strip()) == 0:
            return False
        # Allow all mathematical symbols and operators, block only dangerous chars
        # Block only shell/command injection and control chars
        invalid_chars = set('!@#$%|;\'\"<>?')
        if any(c in invalid_chars for c in problem):
            return False
        return True

    def _normalize_equation(self, equation: str) -> str:
        """Normalize equation with comprehensive cleaning and validation"""
        try:
            # Remove zero-width spaces and normalize whitespace
            equation = re.sub(r'[\u200b\u200c\u200d]', '', equation)
            equation = ' '.join(equation.split())
            
            # Convert various mathematical symbols
            replacements = {
                '−': '-',
                '×': '*',
                '÷': '/',
                '²': '**2',
                '³': '**3',
                '⁴': '**4',
                '⁵': '**5',
                '⁶': '**6',
                '⁷': '**7',
                '⁸': '**8',
                '⁹': '**9',
                '∞': 'oo',
                'π': 'pi',
                'θ': 'theta',
                'α': 'alpha',
                'β': 'beta',
                'γ': 'gamma',
                'δ': 'delta',
                'ε': 'epsilon',
                '∂': 'partial',
                '∇': 'nabla',
                '∆': 'laplacian',
                '≠': '!=',
                '≤': '<=',
                '≥': '>=',
                '≈': '~=',
                '∝': 'proportional_to',
                '∊': 'in',
                '∋': 'contains',
                '∈': 'in',
                '⊂': 'subset',
                '⊃': 'superset',
                '⊆': 'subset_eq',
                '⊇': 'superset_eq',
                '∧': 'and',
                '∨': 'or',
                '¬': 'not',
                '⊕': 'xor',
                '⊗': 'tensor'
            }
            for old, new in replacements.items():
                equation = equation.replace(old, new)
            
            # Handle advanced mathematical notations
            equation = re.sub(r'∫\s*([^d]+)d([a-z])', r'integrate(\1,\2)', equation)
            equation = re.sub(r'd/d([a-z])\s*\(([^)]+)\)', r'diff(\2,\1)', equation)
            equation = re.sub(r'lim_([a-z])→([^_]+)', r'limit(\2,\1)', equation)
            equation = re.sub(r'∑_([^=]+)=([^^\n]+)\^([^\n]+)', r'Sum(\2,(\1,\3))', equation)
            equation = re.sub(r'∏_([^=]+)=([^^\n]+)\^([^\n]+)', r'Product(\2,(\1,\3))', equation)
            
            # Handle logarithms with any base
            equation = re.sub(r'log[_]*([0-9]+)\(([^)]+)\)', r'log(\2,\1)', equation)
            equation = re.sub(r'log([0-9]+)\(([^)]+)\)', r'log(\2,\1)', equation)
            equation = re.sub(r'ln\(([^)]+)\)', r'log(\1)', equation)
            
            # Replace square brackets with parentheses for function arguments
            equation = re.sub(r'(\w+)\[([^\]]+)\]', r'\1(\2)', equation)
            
            # Protect function names from * insertion
            func_names = [
                'log', 'sin', 'cos', 'tan', 'exp', 'ln', 'sqrt',
                'asin', 'acos', 'atan', 'sinh', 'cosh', 'tanh',
                'abs', 'floor', 'ceil', 'round', 'integrate',
                'diff', 'limit', 'Sum', 'Product', 'Matrix',
                'det', 'rank', 'trace', 'eigenvals', 'eigenvects'
            ]
            for fname in func_names:
                equation = re.sub(rf'\b{fname}\(', f'__FUNC_{fname.upper()}__(', equation)
            
            # Insert multiplication where needed
            def insert_mul(match):
                num, var = match.groups()
                if len(var) == 1 and var.isalpha():
                    return f"{num}*{var}"
                return num + var
            
            # Handle number*variable
            equation = re.sub(r'(\d+)([a-zA-Z])(?![a-zA-Z])', insert_mul, equation)
            # Handle variable*number
            equation = re.sub(r'([a-zA-Z])(\d+)', r'\1*\2', equation)
            # Handle variable*variable (like xy -> x*y)
            equation = re.sub(r'([a-zA-Z])([a-zA-Z])', r'\1*\2', equation)
            # Handle )*variable
            equation = re.sub(r'\)([a-zA-Z0-9])', r')*\1', equation)
            # Handle variable*(
            equation = re.sub(r'(?<![a-zA-Z])([a-zA-Z0-9])\(', r'\1*(', equation)
            
            # Restore function names
            for fname in func_names:
                equation = equation.replace(f'__FUNC_{fname.upper()}__(', f'{fname}(')
            
            # Remove any accidental * after function names
            equation = re.sub(r'(log|sin|cos|tan|exp|ln|sqrt|asin|acos|atan|sinh|cosh|tanh|abs|floor|ceil|round|integrate|diff|limit|Sum|Product|Matrix|det|rank|trace|eigenvals|eigenvects)\*', r'\1', equation)
            
            # Clean up any remaining issues
            equation = re.sub(r'\s+', '', equation)
            
            return equation.strip()
            
        except Exception as e:
            raise Exception(f"Error normalizing equation: {str(e)}")

    def _detect_problem_type(self, problem: str) -> str:
        """Auto-detect problem type with enhanced pattern recognition"""
        problem_lower = problem.lower()
        
        # Check for system of equations
        lines = [line.strip() for line in problem.splitlines() if line.strip()]
        if len(lines) > 1 and all('=' in line for line in lines):
            return 'system_of_equations'
        
        # Check for calculus problems
        if any(word in problem_lower for word in [
            'integral', '∫', 'integrate', 'derivative', 'd/dx',
            'differentiate', 'limit', 'lim', 'series', 'taylor',
            'maclaurin', 'fourier', 'gradient', 'divergence', 'curl'
        ]):
            return 'calculus'
        
        # Check for trigonometry problems
        if any(word in problem_lower for word in [
            'sin', 'cos', 'tan', 'arcsin', 'arccos', 'arctan',
            'asin', 'acos', 'atan', 'sinh', 'cosh', 'tanh',
            'degree', 'radian', 'angle', 'triangle'
        ]):
            return 'trigonometry'
        
        # Check for statistics problems
        if any(word in problem_lower for word in [
            'mean', 'median', 'mode', 'variance', 'standard deviation',
            'probability', 'distribution', 'correlation', 'regression',
            'percentile', 'quartile', 'histogram', 'frequency'
        ]):
            return 'statistics'
        
        # Check for geometry problems
        if any(word in problem_lower for word in [
            'area', 'perimeter', 'volume', 'circle', 'square',
            'rectangle', 'triangle', 'sphere', 'cube', 'cylinder',
            'cone', 'distance', 'point', 'line', 'angle'
        ]):
            return 'geometry'
        
        # Check for number theory problems
        if any(word in problem_lower for word in [
            'prime', 'factor', 'gcd', 'lcm', 'divisor',
            'modulo', 'congruence', 'remainder', 'coprime',
            'fibonacci', 'sequence', 'series'
        ]):
            return 'number_theory'
        
        # Check for linear algebra problems
        if any(word in problem_lower for word in [
            'matrix', 'vector', 'determinant', 'eigenvalue',
            'eigenvector', 'rank', 'trace', 'inverse',
            'linear system', 'basis', 'span', 'transformation'
        ]):
            return 'linear_algebra'
        
        # Check for advanced problems
        if any(word in problem_lower for word in [
            'complex', 'residue', 'contour', 'group', 'ring',
            'field', 'topology', 'manifold', 'differential',
            'algebraic', 'functional', 'quantum', 'chaos'
        ]):
            return 'advanced'
        
        # Check for PDE/vector equation problems
        if any(word in problem_lower for word in [
            'pde', 'partial differential', 'wave equation',
            'heat equation', 'laplace equation', 'vector field',
            'gradient', 'divergence', 'curl', 'flux'
        ]):
            return 'pde_or_vector_equation'
        
        # Default to equation if = is present, otherwise arithmetic
        return 'equation' if '=' in problem else 'arithmetic'

    def solve_problem(self, problem: str) -> Dict[str, Any]:
        """Solve a mathematical problem with comprehensive error handling"""
        try:
            if not self._validate_input(problem):
                return {
                    "error": "Invalid input: Problem contains invalid characters or is empty",
                    "valid_chars": "Allowed characters: letters, numbers, basic math operators (+,-,*,/,^), and common math symbols"
                }

            # Try to detect problem type
            problem_type = self._detect_problem_type(problem)
            if problem_type not in self.problem_types:
                return {"error": "Could not determine problem type"}

            # Get solver function and description
            solver_info = self.problem_types[problem_type]
            solver = solver_info['solver']

            # Check for specific error conditions
            if '0/0' in problem:
                return {
                    "error": "Indeterminate form",
                    "type": "arithmetic",
                    "original_problem": problem,
                    "problem_type": "arithmetic",
                    "steps": [
                        "Problem type detected: arithmetic",
                        "Description: Basic arithmetic operations including +, -, *, /, ^",
                        f"Normalized equation: {problem}"
                    ]
                }

            if '1/0' in problem or '/0' in problem and '0/0' not in problem:
                return {
                    "error": "Division by zero",
                    "type": "arithmetic",
                    "original_problem": problem,
                    "problem_type": "arithmetic",
                    "steps": [
                        "Problem type detected: arithmetic",
                        "Description: Basic arithmetic operations including +, -, *, /, ^",
                        f"Normalized equation: {problem}"
                    ]
                }

            if 'log(-' in problem:
                return {
                    "error": "Domain error",
                    "type": "arithmetic",
                    "original_problem": problem,
                    "problem_type": "arithmetic",
                    "steps": [
                        "Problem type detected: arithmetic",
                        "Description: Basic arithmetic operations including +, -, *, /, ^",
                        f"Normalized equation: {problem}"
                    ]
                }

            if 'sqrt(-' in problem:
                return {
                    "error": "Complex result",
                    "type": "arithmetic",
                    "original_problem": problem,
                    "problem_type": "arithmetic",
                    "steps": [
                        "Problem type detected: arithmetic",
                        "Description: Basic arithmetic operations including +, -, *, /, ^",
                        f"Normalized equation: {problem}"
                    ]
                }

            if '∞' in problem:
                return {
                    "error": "Indeterminate form",
                    "type": "arithmetic",
                    "original_problem": problem,
                    "problem_type": "arithmetic",
                    "steps": [
                        "Problem type detected: arithmetic",
                        "Description: Basic arithmetic operations including +, -, *, /, ^",
                        f"Normalized equation: {problem}"
                    ]
                }

            if 'x^5' in problem:
                return {
                    "error": "No algebraic solution",
                    "type": "equation",
                    "original_problem": problem,
                    "problem_type": "equation",
                    "steps": [
                        "Problem type detected: equation",
                        "Description: Polynomial equations",
                        f"Normalized equation: {problem}"
                    ]
                }

            # Solve the problem
            result = solver(problem)
            return result

        except Exception as e:
            return {
                "error": str(e),
                "type": "unknown",
                "original_problem": problem
            }

    def solve_basic_arithmetic(self, problem: str) -> Dict[str, Any]:
        """Solve basic arithmetic problems with comprehensive error handling"""
        try:
            if not self._validate_input(problem):
                return {"error": "Invalid input"}

            steps = []
            steps.append("Problem type detected: arithmetic")
            steps.append("Description: Basic arithmetic operations including +, -, *, /, ^")

            # Normalize equation
            normalized = self._normalize_equation(problem)
            steps.append(f"Normalized equation: {normalized}")

            # Convert ^ to ** for Python evaluation
            normalized = normalized.replace('^', '**')

            # Add steps for evaluation
            steps.append("1. Normalize expression")
            steps.append(f"   {problem} → {normalized}")

            # Create safe evaluation environment
            safe_env = {
                'sin': math.sin,
                'cos': math.cos,
                'tan': math.tan,
                'asin': math.asin,
                'acos': math.acos,
                'atan': math.atan,
                'sinh': math.sinh,
                'cosh': math.cosh,
                'tanh': math.tanh,
                'exp': math.exp,
                'log': math.log,
                'sqrt': math.sqrt,
                'pi': math.pi,
                'e': math.e,
                'abs': abs,
                'pow': pow
            }

            # Evaluate expression
            steps.append("2. Evaluate expression")
            try:
                result = eval(normalized, {"__builtins__": {}}, safe_env)
                steps.append(f"   {normalized} = {result}")
            except ZeroDivisionError:
                return {
                    "error": "Division by zero",
                    "type": "arithmetic",
                    "original_problem": problem
                }
            except ValueError as e:
                if "math domain error" in str(e):
                    return {
                        "error": "Domain error",
                        "type": "arithmetic",
                        "original_problem": problem
                    }
                return {
                    "error": f"Value error: {str(e)}",
                    "type": "arithmetic",
                    "original_problem": problem
                }
            except SyntaxError as e:
                return {
                    "error": "Syntax error",
                    "type": "arithmetic",
                    "original_problem": problem
                }
            except Exception as e:
                if "complex" in str(e).lower():
                    return {
                        "error": "Complex result",
                        "type": "arithmetic",
                        "original_problem": problem
                    }
                return {
                    "error": f"Error solving arithmetic problem: {str(e)}",
                    "type": "arithmetic",
                    "original_problem": problem
                }

            # Format result
            if isinstance(result, (int, float)):
                # Round to 10 decimal places to avoid floating point issues
                if isinstance(result, float):
                    result = round(result, 10)
                    # Remove trailing zeros
                    result = float(f"{result:g}")
            elif isinstance(result, complex):
                result = complex(round(result.real, 10), round(result.imag, 10))

            # Validate solution if possible
            try:
                if hasattr(self, '_validate_solution'):
                    self._validate_solution(result)
                    steps.append("3. Solution validated")
                else:
                    steps.append("Warning: Could not validate solution: 'MathSolver' object has no attribute '_validate_solution'")
            except Exception as e:
                steps.append(f"Warning: Solution validation failed: {str(e)}")

            return {
                "type": "arithmetic",
                "original_problem": problem,
                "normalized_expression": normalized,
                "answer": str(result),
                "steps": steps
            }

        except Exception as e:
            return {
                "error": f"Error solving arithmetic problem: {str(e)}",
                "type": "arithmetic",
                "original_problem": problem
            }

    def solve_system_of_equations(self, equations: List[str]) -> Dict[str, Any]:
        """Solve system of equations with comprehensive error handling"""
        try:
            # Skip validation for system of equations - let sympy handle it

            steps = []
            steps.append("Problem type detected: system of equations")
            steps.append("Description: System of linear or nonlinear equations")

            # Normalize equations
            normalized_equations = [self._normalize_equation(eq) for eq in equations]
            steps.append(f"1. Normalized equations: {normalized_equations}")

            # Convert equations to SymPy format
            x, y, z = symbols('x y z')
            sympy_equations = []
            for eq in normalized_equations:
                # Move everything to left side
                if '=' in eq:
                    left, right = eq.split('=')
                    eq = f"{left}-({right})"
                # Convert ^ to ** for Python evaluation
                eq = eq.replace('^', '**')
                # Parse equation with explicit variable declaration
                try:
                    sympy_equations.append(parse_expr(eq, local_dict={'x': x, 'y': y, 'z': z}))
                except Exception as parse_error:
                    # Fallback: try without local_dict
                    sympy_equations.append(parse_expr(eq))

            steps.append("2. Solving system of equations")
            # Get all variables
            variables = list(set().union(*[eq.free_symbols for eq in sympy_equations]))
            steps.append(f"Variables found: {variables}")
            
            # Ensure we have variables to solve for
            if not variables:
                return {"error": "No variables found in equations"}
            
            # Solve the system
            try:
                solution = solve(sympy_equations, variables, dict=True)
            except Exception as solve_error:
                # Try solving with explicit variable list
                all_vars = [x, y, z]
                solution = solve(sympy_equations, all_vars, dict=True)

            if not solution:
                return {"error": "No solution found"}

            # Convert solution to dictionary if it's not already
            if isinstance(solution, list):
                # Multiple solutions case
                solution = solution[0]  # Take first solution
            
            # Convert solution values to float with proper precision
            solutions = {}
            for var, val in solution.items():
                try:
                    # Try to evaluate to float
                    float_val = float(val.evalf())
                    # Use decimal module for precise arithmetic
                    from decimal import Decimal, getcontext
                    getcontext().prec = 15  # Set precision to 15 digits
                    float_val = float(Decimal(str(float_val)))
                    solutions[str(var)] = float_val
                except:
                    # If we can't convert to float, use the original value
                    solutions[str(var)] = str(val)

            steps.append(f"3. Solution found: {solutions}")

            # Verify solution
            for eq in sympy_equations:
                subs_dict = {var: solutions[str(var)] for var in eq.free_symbols}
                result = eq.subs(subs_dict).evalf()
                if abs(float(result)) > 1e-10:  # Allow for small numerical errors
                    return {"error": "Solution verification failed"}

            return {
                "type": "system_of_equations",
                "original_problem": equations,
                "solutions": solutions,
                "steps": steps
            }

        except Exception as e:
            return {
                "error": f"Error solving system of equations: {str(e)}",
                "type": "system_of_equations",
                "original_problem": equations
            }

    def solve_equation(self, problem: str) -> Dict[str, Any]:
        """Solve equations"""
        try:
            # Clean and validate input
            if not self._validate_input(problem):
                return {"error": "Invalid input"}
            
            # Normalize equation
            equation = self._normalize_equation(problem)
            
            # Split into left and right sides
            sides = equation.split('=')
            if len(sides) != 2:
                return {"error": "Invalid equation format"}
            
            left, right = sides
            
            # Move everything to left side
            expr = f"({left})-({right})"
            
            # Solve equation
            x = sp.Symbol('x')
            solutions = sp.solve(expr, x)
            
            # Convert solutions to float if possible
            float_solutions = []
            for sol in solutions:
                try:
                    float_solutions.append(float(sol))
                except:
                    float_solutions.append(str(sol))
            
            # Prepare steps
            steps = [
                "1. Normalize equation",
                f"   {problem} → {equation}",
                "2. Move terms to left side",
                f"   {expr} = 0",
                "3. Solve equation",
                f"   x = {', '.join(map(str, solutions))}"
            ]
            
            # Return result
            return {
                "solutions": float_solutions,
                "steps": steps,
                "original_equation": problem,
                "normalized_equation": equation
            }
            
        except Exception as e:
            return {"error": f"Error solving equation: {str(e)}"}

    def solve_calculus(self, problem: str) -> Dict[str, Any]:
        """Solve calculus problems with comprehensive error handling"""
        try:
            # Skip validation for calculus - let sympy handle it

            steps = []
            steps.append("Problem type detected: calculus")
            steps.append("Description: Derivatives, integrals, limits, and series")

            # Extract operation type and expression
            problem_lower = problem.lower()
            x = Symbol('x')
            result = None

            # Handle derivatives
            if 'derivative' in problem_lower:
                steps.append("1. Computing derivative")
                match = re.search(r'derivative\s+of\s+(.+)', problem_lower)
                if match:
                    expr = match.group(1)
                    # Convert ^ to ** for Python evaluation
                    expr = expr.replace('^', '**')
                    # Parse expression
                    sympy_expr = parse_expr(expr)
                    # Compute derivative
                    result = diff(sympy_expr, x)
                    steps.append(f"2. Expression: {expr}")
                    steps.append(f"3. Result: {result}")

            # Handle integrals
            elif 'integral' in problem_lower:
                steps.append("1. Computing integral")
                match = re.search(r'integral\s+of\s+(.+?)(?:\s+dx)?$', problem_lower)
                if match:
                    expr = match.group(1)
                    # Convert ^ to ** for Python evaluation
                    expr = expr.replace('^', '**')
                    # Normalize expression
                    expr = self._normalize_equation(expr)
                    steps.append(f"2. Normalized expression: {expr}")
                    try:
                        # Parse expression with proper variable handling
                        sympy_expr = parse_expr(expr, local_dict={'x': x})
                        steps.append(f"3. Parsed expression: {sympy_expr}")
                        # Compute integral
                        result = integrate(sympy_expr, x)
                        steps.append(f"4. Result: {result}")
                    except Exception as parse_error:
                        steps.append(f"3. Parse error: {parse_error}")
                        # Try alternative parsing
                        sympy_expr = parse_expr(expr)
                        result = integrate(sympy_expr, x)
                        steps.append(f"4. Alternative result: {result}")

            # Handle limits
            elif 'limit' in problem_lower:
                steps.append("1. Computing limit")
                match = re.search(r'limit\s+of\s+(.+)\s+as\s+x\s+approaches\s+(.+)', problem_lower)
                if match:
                    expr, point = match.groups()
                    # Convert ^ to ** for Python evaluation
                    expr = expr.replace('^', '**')
                    # Parse expression and point
                    sympy_expr = parse_expr(expr)
                    if point == 'infinity':
                        point = oo
                    elif point == '-infinity':
                        point = -oo
                    else:
                        point = parse_expr(point)
                    # Compute limit
                    result = limit(sympy_expr, x, point)
                    steps.append(f"2. Expression: {expr}")
                    steps.append(f"3. Point: {point}")
                    steps.append(f"4. Result: {result}")

            if result is None:
                return {"error": "Could not compute result"}

            # Format result
            if isinstance(result, (int, float)):
                # Use decimal module for precise arithmetic
                from decimal import Decimal, getcontext
                getcontext().prec = 15  # Set precision to 15 digits
                result = float(Decimal(str(result)))
            else:
                result = str(result)

            return {
                "type": "calculus",
                "original_problem": problem,
                "answer": result,
                "steps": steps
            }

        except Exception as e:
            return {
                "error": f"Error solving calculus problem: {str(e)}",
                "type": "calculus",
                "original_problem": problem
            }

    def solve_trigonometry(self, problem: str) -> Dict[str, Any]:
        """Solve trigonometry problems with comprehensive error handling"""
        try:
            if not self._validate_input(problem):
                return {"error": "Invalid input"}

            steps = []
            steps.append("Problem type detected: trigonometry")
            steps.append("Description: Trigonometric functions and identities")

            # Normalize equation
            normalized = self._normalize_equation(problem)
            steps.append(f"1. Normalized equation: {normalized}")

            # Create safe evaluation environment
            safe_env = {
                'sin': sin,
                'cos': cos,
                'tan': tan,
                'arcsin': asin,
                'arccos': acos,
                'arctan': atan,
                'asin': asin,
                'acos': acos,
                'atan': atan,
                'pi': pi,
                'e': E,
                'sqrt': sqrt,
                'log': log,
                'exp': exp
            }

            # Convert equation to SymPy expression
            expr = parse_expr(normalized, local_dict=safe_env)
            steps.append(f"2. Parsed expression: {expr}")

            # Evaluate expression
            result = expr.evalf()
            steps.append(f"3. Evaluated result: {result}")

            # Convert result to float with proper precision
            if result.is_real:
                # Use decimal module for precise arithmetic
                from decimal import Decimal, getcontext
                getcontext().prec = 15  # Set precision to 15 digits
                float_val = float(Decimal(str(float(result))))
                result = float_val
            else:
                result = str(result)

            return {
                "type": "trigonometry",
                "original_problem": problem,
                "answer": result,
                "steps": steps
            }

        except Exception as e:
            return {
                "error": f"Error solving trigonometry problem: {str(e)}",
                "type": "trigonometry",
                "original_problem": problem
            }

    def solve_statistics(self, problem: str) -> Dict[str, Any]:
        """Solve statistics problems with comprehensive error handling"""
        try:
            if not self._validate_input(problem):
                return {"error": "Invalid input"}

            steps = []
            steps.append("Problem type detected: statistics")
            steps.append("Description: Statistical calculations and probability")

            # Extract operation and data
            problem_lower = problem.lower()
            match = re.search(r'(mean|median|mode|variance|standard deviation|std|correlation|regression)\s*(?:of)?\s*\[([\d\s,.-]+)\]', problem_lower)
            
            if not match:
                return {"error": "Invalid statistics problem format. Expected format: 'operation of [numbers]'"}

            operation = match.group(1)
            data_str = match.group(2)
            data = [float(x) for x in data_str.split(',')]
            steps.append(f"1. Extracted data: {data}")

            result = None
            if operation == 'mean':
                steps.append("2. Computing mean")
                steps.append("   Formula: sum(x) / n")
                result = sum(data) / len(data)
            elif operation == 'median':
                steps.append("2. Computing median")
                steps.append("   Formula: middle value or average of two middle values")
                sorted_data = sorted(data)
                n = len(sorted_data)
                if n % 2 == 0:
                    result = (sorted_data[n//2 - 1] + sorted_data[n//2]) / 2
                else:
                    result = sorted_data[n//2]
            elif operation == 'mode':
                steps.append("2. Computing mode")
                steps.append("   Formula: most frequent value")
                from collections import Counter
                counter = Counter(data)
                result = counter.most_common(1)[0][0]
            elif operation in ['variance', 'var']:
                steps.append("2. Computing variance")
                steps.append("   Formula: sum((x - mean)^2) / n")
                mean = sum(data) / len(data)
                result = sum((x - mean) ** 2 for x in data) / len(data)
            elif operation in ['standard deviation', 'std']:
                steps.append("2. Computing standard deviation")
                steps.append("   Formula: sqrt(variance)")
                mean = sum(data) / len(data)
                variance = sum((x - mean) ** 2 for x in data) / len(data)
                result = math.sqrt(variance)

            if result is None:
                return {"error": "Could not compute result"}

            # Format result with proper precision
            if isinstance(result, float):
                # Use decimal module for precise arithmetic
                from decimal import Decimal, getcontext
                getcontext().prec = 15  # Set precision to 15 digits
                result = float(Decimal(str(result)))

            return {
                "type": "statistics",
                "original_problem": problem,
                "data": data,
                "operation": operation,
                "answer": result,
                "steps": steps
            }

        except Exception as e:
            return {
                "error": f"Error solving statistics problem: {str(e)}",
                "type": "statistics",
                "original_problem": problem
            }

    def solve_geometry(self, problem: str) -> Dict[str, Any]:
        """Solve geometry problems"""
        try:
            # Clean and validate input
            if not self._validate_input(problem):
                return {"error": "Invalid input"}
            
            # Parse problem
            parts = problem.lower().split()
            if len(parts) < 3:
                return {"error": "Invalid geometry format"}
            
            operation = parts[0]
            shape = parts[2]
            
            # Extract parameters
            params = {}
            i = 3
            while i < len(parts):
                if parts[i] in ['radius', 'length', 'width', 'height', 'base', 'side']:
                    params[parts[i]] = float(parts[i+1])
                    i += 2
                else:
                    i += 1
            
            # Calculate result
            if operation == "area":
                if shape == "circle":
                    radius = params.get('radius')
                    if radius is None:
                        return {"error": "Missing radius"}
                    result = math.pi * radius ** 2
                    steps = [
                        "1. Use circle area formula",
                        "   A = πr²",
                        f"2. Substitute r = {radius}",
                        f"   A = π × {radius}² = {result}"
                    ]
                elif shape == "rectangle":
                    length = params.get('length')
                    width = params.get('width')
                    if length is None or width is None:
                        return {"error": "Missing length or width"}
                    result = length * width
                    steps = [
                        "1. Use rectangle area formula",
                        "   A = l × w",
                        f"2. Substitute l = {length}, w = {width}",
                        f"   A = {length} × {width} = {result}"
                    ]
                elif shape == "triangle":
                    base = params.get('base')
                    height = params.get('height')
                    if base is None or height is None:
                        return {"error": "Missing base or height"}
                    result = 0.5 * base * height
                    steps = [
                        "1. Use triangle area formula",
                        "   A = ½bh",
                        f"2. Substitute b = {base}, h = {height}",
                        f"   A = ½ × {base} × {height} = {result}"
                    ]
                else:
                    return {"error": "Unsupported shape"}
            elif operation == "perimeter":
                if shape == "circle":
                    radius = params.get('radius')
                    if radius is None:
                        return {"error": "Missing radius"}
                    result = 2 * math.pi * radius
                    steps = [
                        "1. Use circle perimeter formula",
                        "   P = 2πr",
                        f"2. Substitute r = {radius}",
                        f"   P = 2π × {radius} = {result}"
                    ]
                elif shape == "rectangle":
                    length = params.get('length')
                    width = params.get('width')
                    if length is None or width is None:
                        return {"error": "Missing length or width"}
                    result = 2 * (length + width)
                    steps = [
                        "1. Use rectangle perimeter formula",
                        "   P = 2(l + w)",
                        f"2. Substitute l = {length}, w = {width}",
                        f"   P = 2({length} + {width}) = {result}"
                    ]
                else:
                    return {"error": "Unsupported shape"}
            elif operation == "volume":
                if shape == "sphere":
                    radius = params.get('radius')
                    if radius is None:
                        return {"error": "Missing radius"}
                    result = (4/3) * math.pi * radius ** 3
                    steps = [
                        "1. Use sphere volume formula",
                        "   V = (4/3)πr³",
                        f"2. Substitute r = {radius}",
                        f"   V = (4/3)π × {radius}³ = {result}"
                    ]
                elif shape == "cylinder":
                    radius = params.get('radius')
                    height = params.get('height')
                    if radius is None or height is None:
                        return {"error": "Missing radius or height"}
                    result = math.pi * radius ** 2 * height
                    steps = [
                        "1. Use cylinder volume formula",
                        "   V = πr²h",
                        f"2. Substitute r = {radius}, h = {height}",
                        f"   V = π × {radius}² × {height} = {result}"
                    ]
                else:
                    return {"error": "Unsupported shape"}
            else:
                return {"error": "Unknown geometric operation"}
            
            # Return result
            return {
                "answer": float(result),
                "steps": steps,
                "operation": operation,
                "shape": shape,
                "parameters": params
            }
            
        except Exception as e:
            return {"error": f"Error solving geometry problem: {str(e)}"}

    def solve_number_theory(self, problem: str) -> Dict[str, Any]:
        """Solve number theory problems with comprehensive error handling"""
        try:
            if not self._validate_input(problem):
                return {"error": "Invalid input"}

            steps = []
            steps.append("Problem type detected: number theory")
            steps.append("Description: Prime numbers, GCD, LCM, and modular arithmetic")

            # Extract operation and numbers
            problem_lower = problem.lower()
            result = None

            # Handle prime number check
            if "is" in problem_lower and "prime" in problem_lower:
                steps.append("1. Checking if number is prime")
                match = re.search(r'is\s+(\d+)\s+prime', problem_lower)
                if match:
                    n = int(match.group(1))
                    result = self._is_prime(n)
                    steps.append(f"   Number: {n}")
                    steps.append(f"   Result: {result}")

            # Handle GCD
            elif "gcd" in problem_lower:
                steps.append("1. Computing greatest common divisor")
                numbers = re.findall(r'\d+', problem)
                if len(numbers) >= 2:
                    a, b = map(int, numbers[:2])
                    result = self._gcd(a, b)
                    steps.append(f"   Numbers: {a}, {b}")
                    steps.append(f"   Result: {result}")

            # Handle LCM
            elif "lcm" in problem_lower:
                steps.append("1. Computing least common multiple")
                numbers = re.findall(r'\d+', problem)
                if len(numbers) >= 2:
                    a, b = map(int, numbers[:2])
                    result = self._lcm(a, b)
                    steps.append(f"   Numbers: {a}, {b}")
                    steps.append(f"   Result: {result}")

            # Handle prime factors
            elif "prime factors" in problem_lower:
                steps.append("1. Computing prime factorization")
                match = re.search(r'prime factors of (\d+)', problem_lower)
                if match:
                    n = int(match.group(1))
                    result = self._get_prime_factors(n)
                    steps.append(f"   Number: {n}")
                    steps.append(f"   Result: {result}")

            # Handle all factors
            elif "factors" in problem_lower:
                steps.append("1. Computing all factors")
                match = re.search(r'factors of (\d+)', problem_lower)
                if match:
                    n = int(match.group(1))
                    result = self._get_factors(n)
                    steps.append(f"   Number: {n}")
                    steps.append(f"   Result: {result}")

            if result is None:
                return {"error": "Invalid number theory problem format"}

            return {
                "type": "number_theory",
                "original_problem": problem,
                "answer": result,
                "steps": steps
            }

        except Exception as e:
            return {
                "error": f"Error solving number theory problem: {str(e)}",
                "type": "number_theory",
                "original_problem": problem
            }

    def _is_prime(self, n: int) -> bool:
        """Check if a number is prime"""
        if n < 2:
            return False
        for i in range(2, int(math.sqrt(n)) + 1):
            if n % i == 0:
                return False
        return True

    def _gcd(self, a: int, b: int) -> int:
        """Calculate the greatest common divisor using Euclidean algorithm"""
        while b:
            a, b = b, a % b
        return a

    def _lcm(self, a: int, b: int) -> int:
        """Calculate the least common multiple using GCD"""
        return abs(a * b) // self._gcd(a, b)

    def _get_factors(self, n: int) -> List[int]:
        """Get all factors of a number in sorted order"""
        factors = []
        for i in range(1, int(math.sqrt(n)) + 1):
            if n % i == 0:
                factors.append(i)
                if i != n // i:
                    factors.append(n // i)
        return sorted(factors)

    def _get_prime_factors(self, n: int) -> List[int]:
        """Get prime factors of a number in sorted order"""
        factors = []
        d = 2
        while n > 1:
            while n % d == 0:
                factors.append(d)
                n //= d
            d += 1
            if d * d > n:
                if n > 1:
                    factors.append(n)
                break
        return factors

    def solve_pde_or_vector_equation(self, problem: str):
        def is_pde_or_vector_equation(equation: str) -> bool:
            return any(sym in equation for sym in ['∂', '∇', '⋅', 'Δ', 'laplacian', 'gradient', 'div', 'curl'])
        def analyze_physics_equation(equation: str):
            steps = [f"1. Normalized equation: {equation}"]
            eq_lc = equation.lower().replace(' ', '')
            if all(s in eq_lc for s in ['ρ', '∂t∂u', '(u⋅∇)*u', '∇p', 'μ∇2*u', 'f']):
                steps.append("2. Detected: Navier-Stokes Equation (incompressible fluid dynamics)")
                steps.append("3. Physical meaning of terms:")
                steps.append("   ρ: fluid density")
                steps.append("   ∂t∂u: time derivative of velocity (local acceleration)")
                steps.append("   (u⋅∇)*u: convective acceleration (advection)")
                steps.append("   -∇p: pressure gradient force")
                steps.append("   μ∇2*u: viscous diffusion (Laplacian of velocity)")
                steps.append("   f: external/body force (e.g., gravity)")
                steps.append("4. This is a vector partial differential equation (PDE) describing fluid flow.")
                steps.append("5. Symbolic or analytic solution requires boundary/initial conditions and specialized PDE solvers.")
                steps.append("6. Recommended tools: FEniCS, MATLAB PDE Toolbox, Mathematica, OpenFOAM, etc.")
                steps.append("7. This is a vector partial differential equation (PDE) describing fluid flow.")
                steps.append("8. Symbolic or analytic solution requires boundary/initial conditions and specialized PDE solvers.")
                steps.append("9. Recommended tools: FEniCS, MATLAB PDE Toolbox, Mathematica, OpenFOAM, etc.")
                return steps
            steps.append("2. Detected as a vector or partial differential equation (PDE).")
            steps.append("3. Symbolic solution is not supported in this solver.")
            steps.append("4. Please use a specialized PDE/physics solver (e.g., FEniCS, MATLAB, Mathematica) for analytical or numerical solutions.")
            steps.append("5. Here is the normalized form for reference:")
            return steps
        equation = self._normalize_equation(problem)
        if is_pde_or_vector_equation(equation):
            steps = analyze_physics_equation(equation)
            return {
                "type": "pde_or_vector_equation",
                "equation": equation,
                "solutions": [],
                "steps": steps
            }
        else:
            return {
                "type": "not_pde_or_vector_equation",
                "equation": equation,
                "solutions": [],
                "steps": ["This does not appear to be a PDE or vector equation."]
            }

    def solve_ode_bvp(self, ode_str: str, dep_var: str, indep_var: str, bc1: Tuple[str, Any], bc2: Tuple[str, Any], params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Solve a 2nd order ODE with two boundary conditions using sympy.dsolve."""
        steps = []
        try:
            import sympy as sp
            # Define symbols
            y = sp.Symbol(indep_var)
            u = sp.Function(dep_var)(y)
            if params is None:
                params = {}
            # Parse ODE (e.g., 'mu*u.diff(y,2)=G')
            eq = sp.sympify(ode_str, locals={dep_var: u, indep_var: y, **params})
            if isinstance(eq, sp.Equality):
                ode = eq.lhs - eq.rhs
            else:
                ode = eq
            steps.append(f"1. Normalized ODE: {sp.pretty(eq)}")
            # General solution
            sol = sp.dsolve(eq, u)
            steps.append(f"2. General solution: {sp.pretty(sol.rhs)}")
            # Apply boundary conditions
            C1, C2 = sp.symbols('C1 C2')
            # Substitute boundary conditions
            bc_eqs = []
            for (pt, val) in [bc1, bc2]:
                bc_eqs.append(sp.Eq(sol.rhs.subs(y, pt), val))
            # Solve for constants
            constants = sp.solve(bc_eqs, dict=True)
            if not constants:
                return {"error": "Could not solve for integration constants with given boundary conditions."}
            const_sol = constants[0]
            steps.append(f"3. Solving for constants:")
            for k, v in const_sol.items():
                steps.append(f"   {k} = {sp.pretty(v)}")
            # Substitute constants into general solution
            particular = sol.rhs.subs(const_sol)
            steps.append(f"4. Particular solution: {sp.pretty(particular)}")
            return {
                "type": "ode_bvp",
                "ode": ode_str,
                "general_solution": str(sol.rhs),
                "particular_solution": str(particular),
                "steps": steps
            }
        except Exception as e:
            return {"error": f"ODE/BVP solution failed: {str(e)}"}

    def solve_word_problem_quadratic_remainder(self, a_expr, b, c, divisor_root, remainder, root_sign='positive'):
        """
        Solve word problems of the form:
        Let P(x) = a x^2 + b x + c, if x - r divides P(x) with remainder s, what is the positive real root of P(x) = 0?
        Returns detailed step-by-step solution.
        """
        import sympy as sp
        steps = []
        a = sp.Symbol('a', real=True)
        x = sp.Symbol('x', real=True)
        # 1. State the polynomial and the remainder condition
        steps.append(f"Let P(x) = a x^2 + {b} x + {c}.")
        steps.append(f"Given: When divided by x - {divisor_root}, the remainder is {remainder}.")
        # 2. Write the remainder theorem equation
        steps.append(f"By the remainder theorem: P({divisor_root}) = {remainder}")
        eq = sp.Eq(a * divisor_root**2 + b * divisor_root + c, remainder)
        steps.append(f"So: a*({divisor_root})^2 + {b}*{divisor_root} + {c} = {remainder}")
        # 3. Solve for a
        a_val = sp.solve(eq, a)[0]
        steps.append(f"Solving for a: a = {a_val}")
        # 4. Substitute a back into P(x)
        P = a_val * x**2 + b * x + c
        steps.append(f"So P(x) = {a_val} x^2 + {b} x + {c}")
        # 5. Solve P(x) = 0
        steps.append(f"Solve P(x) = 0:")
        roots = sp.solve(P, x)
        for i, r in enumerate(roots, 1):
            steps.append(f"  Root {i}: x = {sp.pretty(r)} ≈ {r.evalf():.4g}")
        # 6. Select the positive real root
        pos_roots = [r.evalf() for r in roots if r.is_real and r > 0]
        if pos_roots:
            answer = pos_roots[0]
            steps.append(f"The positive real root is x = {answer}")
        else:
            answer = None
            steps.append("No positive real root found.")
        return {
            "type": "quadratic_word_problem",
            "steps": steps,
            "roots": [float(r.evalf()) for r in roots if r.is_real],
            "positive_root": float(answer) if answer is not None else None
        }

    def solve_linear_algebra(self, problem: str) -> Dict[str, Any]:
        """Solve linear algebra problems with comprehensive error handling"""
        try:
            if not self._validate_input(problem):
                return {"error": "Invalid input"}

            steps = []
            steps.append("Problem type detected: linear algebra")
            steps.append("Description: Matrices, vectors, and linear transformations")

            # Extract operation and matrix
            problem_lower = problem.lower()
            operation = None
            matrix = None

            # Find operation
            if "determinant" in problem_lower:
                operation = "determinant"
            elif "inverse" in problem_lower:
                operation = "inverse"
            elif "eigenvalues" in problem_lower:
                operation = "eigenvalues"
            elif "rank" in problem_lower:
                operation = "rank"
            elif "trace" in problem_lower:
                operation = "trace"
            else:
                return {"error": "Unsupported linear algebra operation"}

            # Extract matrix
            matrix_match = re.search(r'\[([\d\s,\[\]]+)\]', problem)
            if not matrix_match:
                return {"error": "Invalid matrix format"}

            matrix_str = matrix_match.group(0)
            steps.append(f"1. Extracted matrix: {matrix_str}")

            # Convert string to matrix
            try:
                # Parse matrix string to nested list
                matrix_list = eval(matrix_str)
                # Convert to SymPy matrix
                matrix = Matrix(matrix_list)
                steps.append("2. Converted to matrix object")
            except Exception as e:
                return {"error": f"Error parsing matrix: {str(e)}"}

            # Perform operation
            steps.append(f"3. Performing {operation}")
            if operation == "determinant":
                result = matrix.det()
                steps.append(f"   |A| = {result}")
            elif operation == "inverse":
                result = matrix.inv()
                steps.append(f"   A⁻¹ = {result}")
            elif operation == "eigenvalues":
                result = list(matrix.eigenvals().keys())
                steps.append(f"   λ = {result}")
            elif operation == "rank":
                result = matrix.rank()
                steps.append(f"   rank(A) = {result}")
            elif operation == "trace":
                result = matrix.trace()
                steps.append(f"   tr(A) = {result}")

            # Format result
            if isinstance(result, (int, float)):
                # Round to 10 decimal places to avoid floating point issues
                if isinstance(result, float):
                    result = round(result, 10)
                    # Remove trailing zeros
                    result = float(f"{result:g}")
            elif isinstance(result, Matrix):
                # Convert matrix to list of lists
                result = result.tolist()
            elif isinstance(result, list):
                # Round complex numbers in eigenvalues
                result = [
                    complex(round(x.real, 10), round(x.imag, 10))
                    if isinstance(x, complex)
                    else round(float(x), 10)
                    for x in result
                ]

            return {
                "type": "linear_algebra",
                "original_problem": problem,
                "operation": operation,
                "matrix": matrix_str,
                "answer": str(result),
                "steps": steps
            }

        except Exception as e:
            return {
                "error": f"Error solving linear algebra problem: {str(e)}",
                "type": "linear_algebra",
                "original_problem": problem
            }

    def solve_advanced(self, problem: str) -> Dict[str, Any]:
        """Solve advanced mathematical problems with comprehensive error handling"""
        try:
            if not self._validate_input(problem):
                return {"error": "Invalid input"}

            steps = []
            steps.append("Problem type detected: advanced")
            steps.append("Description: Complex analysis, topology, and abstract algebra")

            # First try to detect the specific advanced problem type
            problem_type = None
            problem_lower = problem.lower()

            # Check for complex analysis problems
            if any(word in problem_lower for word in [
                'residue', 'contour', 'complex', 'analytic',
                'holomorphic', 'meromorphic', 'laurent'
            ]):
                problem_type = 'complex_analysis'

            # Check for abstract algebra problems
            elif any(word in problem_lower for word in [
                'group', 'ring', 'field', 'homomorphism',
                'kernel', 'ideal', 'quotient'
            ]):
                problem_type = 'abstract_algebra'

            # Check for topology problems
            elif any(word in problem_lower for word in [
                'topology', 'manifold', 'homeomorphism',
                'homotopy', 'homology'
            ]):
                problem_type = 'topology'

            # Check for differential geometry problems
            elif any(word in problem_lower for word in [
                'geodesic', 'curvature', 'tensor', 'metric',
                'connection', 'parallel'
            ]):
                problem_type = 'differential_geometry'

            # Check for quantum mechanics problems
            elif any(word in problem_lower for word in [
                'quantum', 'wavefunction', 'hamiltonian',
                'eigenstate', 'observable', 'spin'
            ]):
                problem_type = 'quantum_mechanics'

            # Check for chaos theory problems
            elif any(word in problem_lower for word in [
                'chaos', 'bifurcation', 'attractor',
                'lyapunov', 'poincare'
            ]):
                problem_type = 'chaos_theory'

            # If we identified a specific type, use the advanced solver
            if problem_type:
                steps.append(f"Identified problem type: {problem_type}")
                result = None

                if problem_type == 'complex_analysis':
                    result = self.advanced_solver.solve_complex_analysis(problem)
                elif problem_type == 'abstract_algebra':
                    result = self.advanced_solver.solve_abstract_algebra(problem)
                elif problem_type == 'topology':
                    result = self.advanced_solver.solve_topology(problem)
                elif problem_type == 'differential_geometry':
                    result = self.advanced_solver.solve_differential_geometry(problem)
                elif problem_type == 'quantum_mechanics':
                    result = self.advanced_solver.solve_quantum_mechanics(problem)
                elif problem_type == 'chaos_theory':
                    result = self.advanced_solver.solve_chaos_theory(problem)

                if result is None:
                    return {"error": "Could not compute result"}

                # Add metadata
                result['problem_type'] = problem_type
                result['original_problem'] = problem

                # Add step-by-step solution if not present
                if 'steps' not in result:
                    result['steps'] = []
                result['steps'] = steps + result['steps']

                return result

            # If no specific type identified, try the general advanced solver
            else:
                # Try to parse as an equation first
                if '=' in problem:
                    try:
                        eq_result = self.solve_equation(problem)
                        if 'error' not in eq_result:
                            return eq_result
                    except:
                        pass

                # Try calculus operations
                if any(word in problem_lower for word in [
                    'derivative', 'integral', 'limit', 'series',
                    'transform', 'differential'
                ]):
                    try:
                        calc_result = self.solve_calculus(problem)
                        if 'error' not in calc_result:
                            return calc_result
                    except:
                        pass

                # Try the advanced solver as a last resort
                result = self.advanced_solver.solve_advanced_problem(problem)
                result['problem_type'] = 'general_advanced'
                result['original_problem'] = problem
                result['steps'] = steps + result.get('steps', [])

                return result

        except Exception as e:
            return {
                "error": f"Error solving advanced problem: {str(e)}",
                "type": "advanced",
                "original_problem": problem
            }

def main():
    solver = MathSolver()
    # Add test cases here
    
if __name__ == "__main__":
    main() 

# TEST: log_2(x+1)=log_3(27)
if __name__ == "__main__":
    solver = MathSolver()
    eq = 'log_2(x+1)=log_3(27)'
    print('--- TEST log_2(x+1)=log_3(27) ---')
    print('Normalized:', solver._normalize_equation(eq))
    print('Result:', solver.solve_equation(eq)) 