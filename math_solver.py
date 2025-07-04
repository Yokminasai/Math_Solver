#!/usr/bin/env python3
"""
Mathematical Problem Solver
Handles various types of mathematical problems from basic to advanced
"""

import math
import numpy as np
import sympy as sp
from sympy import (
    symbols, solve, diff, integrate, limit, simplify, expand, factor,
    Matrix, diag, eye, zeros, ones, GramSchmidt
)
from sympy.solvers import solve_linear_system, solve_poly_system
from sympy.matrices import Matrix
import matplotlib.pyplot as plt
from typing import Union, List, Tuple, Dict, Any, Optional
import re
import json
from advanced_math import AdvancedMathSolver

class MathSolver:
    def __init__(self):
        """Initialize the math solver"""
        try:
            self.x, self.y, self.z = symbols('x y z')
            self.a, self.b, self.c = symbols('a b c')
            self.t = symbols('t')
            self.advanced_solver = AdvancedMathSolver()
        except Exception as e:
            raise Exception(f"Cannot initialize solver: {str(e)}")

    def _auto_insert_mul(self, expr: str) -> str:
        """Insert * between numbers and variables, e.g. 2x → 2*x, -5y → -5*y"""
        try:
            expr = re.sub(r'(?<=\d)(?=[a-zA-Z(])', '*', expr)
            return expr
        except Exception as e:
            raise Exception(f"Error in expression conversion: {str(e)}")

    def _normalize_equation(self, equation: str) -> str:
        """ทำความสะอาดและแปลงสมการให้อยู่ในรูปแบบมาตรฐาน"""
        # แทนที่เครื่องหมายลบพิเศษ
        equation = equation.replace('−', '-')
        
        # แทนที่เครื่องหมายคูณและหาร
        equation = equation.replace('×', '*')
        equation = equation.replace('÷', '/')
        
        # แทนที่ยกกำลัง
        equation = equation.replace('^', '**')
        equation = re.sub(r'(\d+)²', r'\1**2', equation)
        equation = re.sub(r'(\w+)²', r'\1**2', equation)
        
        # เพิ่มเครื่องหมายคูณที่ถูกละไว้
        equation = re.sub(r'(\d+)([a-zA-Z])', r'\1*\2', equation)
        equation = re.sub(r'([a-zA-Z])(\d+)', r'\1*\2', equation)
        equation = re.sub(r'\)([a-zA-Z0-9])', r')*\1', equation)
        equation = re.sub(r'([a-zA-Z0-9])\(', r'\1*(', equation)
        
        # แทนที่ฟังก์ชันตรีโกณมิติ
        equation = re.sub(r'sin\^2', 'sin', equation)
        equation = re.sub(r'cos\^2', 'cos', equation)
        equation = re.sub(r'tan\^2', 'tan', equation)
        
        # ลบช่องว่างที่ไม่จำเป็น
        equation = re.sub(r'\s+', '', equation)
        
        return equation.strip()

    def solve_basic_arithmetic(self, expression: str) -> Dict[str, Any]:
        """
        แก้โจทย์การคำนวณพื้นฐาน (บวก ลบ คูณ หาร)
        """
        try:
            # ทำความสะอาดนิพจน์
            expression = expression.replace('×', '*').replace('÷', '/')
            expression = expression.replace('^', '**')
            
            # จัดการสมการ
            if '=' in expression:
                left, right = expression.split('=', 1)
                left = left.strip()
                right = right.strip()
                
                # ตรวจสอบตัวแปร
                if any(c.isalpha() for c in left + right):
                    return self.solve_equation(expression)
                
                # คำนวณทั้งสองข้าง
                left_result = eval(left)
                right_result = eval(right)
                
                return {
                    "type": "arithmetic_comparison",
                    "left_side": left,
                    "right_side": right,
                    "left_result": left_result,
                    "right_result": right_result,
                    "is_equal": abs(left_result - right_result) < 1e-10,
                    "steps": [
                        f"1. คำนวณด้านซ้าย: {left} = {left_result}",
                        f"2. คำนวณด้านขวา: {right} = {right_result}",
                        f"3. เปรียบเทียบผลลัพธ์: {left_result} {'=' if abs(left_result - right_result) < 1e-10 else '≠'} {right_result}"
                    ]
                }
            else:
                # คำนวณนิพจน์เดี่ยว
                result = eval(expression)
                steps = []
                
                # แยกการคำนวณ
                if '+' in expression or '-' in expression:
                    terms = expression.replace('-', '+-').split('+')
                    terms = [t for t in terms if t]
                    steps.append(f"1. แยกนิพจน์: {' + '.join(terms)}")
                    
                    # คำนวณแต่ละเทอมที่มีการคูณหรือหาร
                    for i, term in enumerate(terms, 1):
                        if '*' in term or '/' in term:
                            term_result = eval(term)
                            steps.append(f"{i+1}. คำนวณ {term} = {term_result}")
                    
                    steps.append(f"{len(steps)+1}. รวมทุกเทอม: {result}")
                else:
                    steps.append(f"1. คำนวณ {expression} = {result}")
                
                return {
                    "type": "arithmetic_calculation",
                    "expression": expression,
                    "result": result,
                    "steps": steps
                }
                
        except Exception as e:
            return {"error": f"ไม่สามารถแก้โจทย์การคำนวณได้: {str(e)}"}

    def solve_system_of_equations(self, equations: List[str]) -> Dict[str, Any]:
        """Solve a system of equations"""
        try:
            # Clean up equations
            equations = [self._normalize_equation(eq) for eq in equations]
            print(f"Cleaned system of equations: {equations}")  # Debug log
            # Extract variables
            variables = set()
            for eq in equations:
                variables.update(re.findall(r'[a-zA-Z]', eq))
            variables = sorted(list(variables))
            if not variables:
                return {"error": "No variables found in the system of equations."}
            print(f"Variables found: {variables}")  # Debug log
            # Convert to sympy symbols
            symbols = sp.symbols(' '.join(variables))
            symbol_dict = dict(zip(variables, symbols))
            # Convert equations to expressions
            expressions = []
            for eq in equations:
                if '=' not in eq:
                    return {"error": f"No '=' sign found in equation: {eq}"}
                left, right = eq.split('=')
                expr = sp.sympify(left.strip()) - sp.sympify(right.strip())
                expressions.append(expr)
            # Solve the system
            solutions = sp.solve(expressions, symbols, dict=True)
            if not solutions:
                return {"error": "No solution found."}
            # Build solution steps in English
            steps = [
                f"1. System of equations:",
                *[f"   {eq}" for eq in equations],
                "",
                f"2. Variables found:",
                f"   {', '.join(variables)}",
                "",
                f"3. Solution(s):"
            ]
            # Add solutions
            for idx, sol in enumerate(solutions, 1):
                step = []
                for var in variables:
                    value = sol[symbol_dict[var]]
                    step.append(f"{var} = {value}")
                steps.append(f"   Solution {idx}: " + ", ".join(step))
            # Verification
            steps.extend(["", "4. Solution verification:"])
            for idx, sol in enumerate(solutions, 1):
                verification = []
                for eq_idx, expr in enumerate(expressions, 1):
                    result = expr.subs(sol)
                    try:
                        is_correct = abs(float(result)) < 1e-10
                    except Exception:
                        is_correct = False
                    checkmark = "✓" if is_correct else "✗"
                    verification.append(f"Eq{eq_idx}: {checkmark}")
                steps.append(f"   Solution {idx}: {' '.join(verification)}")
            return {
                "type": "system_of_equations",
                "equations": equations,
                "solutions": solutions,
                "steps": steps
            }
        except Exception as e:
            return {"error": f"Failed to solve system of equations: {str(e)}"}

    def solve_equation(self, problem: str) -> Dict[str, Any]:
        """แก้สมการ"""
        try:
            # ทำความสะอาดสมการ
            equation = self._normalize_equation(problem)
            print(f"สมการที่ทำความสะอาดแล้ว: {equation}")  # Debug log
            
            # แยกด้านซ้ายและขวาของสมการ
            if '=' not in equation:
                return {"error": "ไม่พบเครื่องหมาย = ในสมการ"}
            
            left, right = equation.split('=')
            left = left.strip()
            right = right.strip()
            
            print(f"ด้านซ้าย: {left}")  # Debug log
            print(f"ด้านขวา: {right}")  # Debug log
            
            # แปลงเป็น sympy expression
            x = sp.Symbol('x')
            
            # แทนที่เครื่องหมายคูณที่ละไว้และจัดการวงเล็บ
            def process_expression(expr: str) -> str:
                # แทนที่เครื่องหมายคูณที่ละไว้
                expr = re.sub(r'(\d+)([a-zA-Z])', r'\1*\2', expr)
                expr = re.sub(r'([a-zA-Z])(\d+)', r'\1*\2', expr)
                expr = re.sub(r'(\d+)\(', r'\1*(', expr)
                expr = re.sub(r'([a-zA-Z])\(', r'\1*(', expr)
                
                # แทนที่เครื่องหมายลบพิเศษ
                expr = expr.replace('−', '-')
                
                # จัดการวงเล็บ
                expr = re.sub(r'\)\(', ')*(', expr)
                
                # แทนที่การคูณที่ละไว้ระหว่างตัวแปร
                expr = re.sub(r'([a-zA-Z])\s*([a-zA-Z])', r'\1*\2', expr)
                
                return expr
            
            left = process_expression(left)
            right = process_expression(right)
            
            print(f"หลังแทนที่เครื่องหมายคูณ:")
            print(f"ด้านซ้าย: {left}")
            print(f"ด้านขวา: {right}")
            
            try:
                left_expr = sp.sympify(left)
                right_expr = sp.sympify(right)
            except Exception as e:
                print(f"ข้อผิดพลาดในการแปลงนิพจน์: {str(e)}")
                # ลองแปลงอีกครั้งโดยใช้การแทนที่เพิ่มเติม
                left = left.replace(' ', '')
                right = right.replace(' ', '')
                try:
                    left_expr = sp.sympify(left)
                    right_expr = sp.sympify(right)
                except Exception as e:
                    return {"error": f"ไม่สามารถแปลงนิพจน์: {str(e)}"}
            
            # ย้ายข้างให้เป็นรูปมาตรฐาน
            equation_expr = left_expr - right_expr
            
            # กระจายพจน์และรวมพจน์
            equation_expr = sp.expand(equation_expr)
            equation_expr = sp.collect(equation_expr, x)
            print(f"สมการในรูปมาตรฐาน (หลังรวมพจน์): {equation_expr} = 0")
            
            # แก้สมการ
            try:
                solutions = sp.solve(equation_expr, x)
                print(f"คำตอบที่ได้: {solutions}")
                
                if not solutions:
                    # ลองใช้วิธีตัวเลข
                    try:
                        # แปลงสมการให้อยู่ในรูปฟังก์ชัน
                        f = sp.lambdify(x, equation_expr)
                        
                        # ลองหาคำตอบด้วยวิธีตัวเลขหลายวิธี
                        from scipy.optimize import fsolve, root
                        
                        solutions = []
                        initial_guesses = [-10, -1, 0, 1, 10]  # เพิ่มจุดเริ่มต้น
                        
                        # ลองใช้ fsolve
                        for x0 in initial_guesses:
                            try:
                                sol = fsolve(f, x0)
                                if abs(f(sol)) < 1e-10:
                                    solutions.extend([complex(s) for s in sol])
                            except:
                                continue
                        
                        # ถ้ายังไม่พบคำตอบ ลองใช้ root
                        if not solutions:
                            for x0 in initial_guesses:
                                try:
                                    result = root(f, x0)
                                    if result.success and abs(f(result.x[0])) < 1e-10:
                                        solutions.extend([complex(s) for s in result.x])
                                except:
                                    continue
                        
                        # ลบคำตอบที่ซ้ำกัน
                        solutions = list(set(solutions))
                        print(f"คำตอบเชิงตัวเลข: {solutions}")
                        
                    except Exception as e:
                        print(f"ไม่สามารถใช้วิธีตัวเลข: {str(e)}")
                        # ลองหาคำตอบด้วยวิธีอื่น
                        try:
                            solutions = list(sp.solveset(equation_expr, x))
                            print(f"คำตอบจาก solveset: {solutions}")
                        except Exception as e:
                            print(f"ไม่สามารถใช้ solveset: {str(e)}")
                            return {"error": "ไม่สามารถหาคำตอบได้"}
                
            except Exception as e:
                print(f"ข้อผิดพลาดในการแก้สมการ: {str(e)}")
                # ลองใช้วิธีอื่น
                try:
                    solutions = list(sp.solveset(equation_expr, x))
                    print(f"คำตอบจาก solveset: {solutions}")
                except Exception as e:
                    print(f"ไม่สามารถใช้ solveset: {str(e)}")
                    return {"error": "ไม่สามารถหาคำตอบได้"}
            
            if not solutions:
                return {"error": "ไม่พบคำตอบที่เป็นไปได้"}
            
            # สร้างขั้นตอนการแก้
            steps = [
                f"1. จัดรูปสมการ:",
                f"   {left} = {right}",
                "",
                f"2. ย้ายข้างและรวมพจน์:",
                f"   {equation_expr} = 0",
                "",
                f"3. แก้สมการ:"
            ]
            
            # จัดรูปคำตอบ
            formatted_solutions = []
            for sol in solutions:
                if isinstance(sol, complex):
                    if abs(sol.imag) < 1e-10:  # คำตอบจริง
                        formatted_solutions.append(f"{float(sol.real):.6g}")
                    else:  # คำตอบเชิงซ้อน
                        formatted_solutions.append(f"{complex(sol):.6g}")
                else:
                    try:
                        float_sol = float(sol)
                        formatted_solutions.append(f"{float_sol:.6g}")
                    except:
                        formatted_solutions.append(str(sol))
            
            # ลบคำตอบที่ซ้ำกัน
            formatted_solutions = list(dict.fromkeys(formatted_solutions))
            
            if len(formatted_solutions) == 1:
                steps.append(f"   x = {formatted_solutions[0]}")
            else:
                for i, sol in enumerate(formatted_solutions, 1):
                    steps.append(f"   x₍{i}₎ = {sol}")
            
            # ตรวจสอบคำตอบ
            steps.extend(["", "4. ตรวจสอบคำตอบ:"])
            for sol in solutions:
                try:
                    result = equation_expr.subs(x, sol)
                    # robust formatting for all sympy types
                    try:
                        sol_val = float(sol)
                        sol_fmt = f"{sol_val:.6g}"
                    except Exception:
                        sol_fmt = str(sol)
                    if abs(complex(result.evalf())) < 1e-10:  # ใกล้เคียง 0
                        steps.append(f"   ✓ x = {sol_fmt} เป็นคำตอบที่ถูกต้อง")
                    else:
                        steps.append(f"   ✗ x = {sol_fmt} ไม่ใช่คำตอบที่ถูกต้อง (ค่าคลาดเคลื่อน = {abs(complex(result.evalf())):.2e})")
                except Exception as e:
                    steps.append(f"   ? ไม่สามารถตรวจสอบ x = {sol} ได้: {str(e)}")
            
            return {
                "type": "equation",
                "equation": equation,
                "solutions": formatted_solutions,
                "steps": steps
            }
            
        except Exception as e:
            print(f"ข้อผิดพลาดในการแก้สมการ: {str(e)}")
            return {"error": f"ไม่สามารถแก้สมการได้: {str(e)}"}
    
    def solve_calculus(self, problem: str) -> Dict[str, Any]:
        """แก้โจทย์แคลคูลัส"""
        try:
            # ตรวจสอบประเภทของโจทย์
            if 'integrate' in problem.lower() or '∫' in problem:
                return self.solve_integral(problem)
            elif 'derivative' in problem.lower() or 'd/dx' in problem:
                return self.solve_derivative(problem)
            elif 'limit' in problem.lower() or 'lim' in problem:
                return self.solve_limit(problem)
            else:
                return {"error": "ไม่สามารถระบุประเภทของโจทย์แคลคูลัสได้"}
            
        except Exception as e:
            return {"error": f"ไม่สามารถแก้โจทย์แคลคูลัสได้: {str(e)}"}
    
    def solve_linear_algebra(self, problem: str) -> Dict[str, Any]:
        """แก้โจทย์พีชคณิตเชิงเส้น"""
        try:
            # ตรวจสอบประเภทของโจทย์
            if 'matrix' in problem.lower() or any(c in problem for c in ['[', ']']):
                return self._solve_matrix_problem(problem)
            elif 'determinant' in problem.lower() or 'det' in problem.lower():
                return self._solve_determinant(problem)
            elif 'eigenvalue' in problem.lower() or 'eigenvector' in problem.lower():
                return self._solve_eigenvalue(problem)
            else:
                return {"error": "ไม่สามารถระบุประเภทของโจทย์พีชคณิตเชิงเส้นได้"}
            
        except Exception as e:
            return {"error": f"ไม่สามารถแก้โจทย์พีชคณิตเชิงเส้นได้: {str(e)}"}
    
    def _solve_matrix_problem(self, problem: str) -> Dict[str, Any]:
        """แก้โจทย์เกี่ยวกับเมทริกซ์"""
        try:
            # แยกเมทริกซ์จากโจทย์
            matrices = []
            for match in re.finditer(r'\[((?:[^][]|\[(?:[^][]|\[[^]]*\])*\])*)\]', problem):
                matrix_str = match.group(1)
                # แปลงข้อความเป็นเมทริกซ์
                rows = [row.strip().split() for row in matrix_str.split(';')]
                matrix = sp.Matrix([[sp.sympify(elem) for elem in row] for row in rows])
                matrices.append(matrix)
            
            if not matrices:
                return {"error": "ไม่พบเมทริกซ์ในโจทย์"}
            
            # ตรวจสอบการดำเนินการ
            if '+' in problem:
                result = sum(matrices[1:], matrices[0])
                operation = 'add'
            elif '-' in problem:
                result = matrices[0]
                for m in matrices[1:]:
                    result -= m
                operation = 'subtract'
            elif '*' in problem or '×' in problem:
                result = matrices[0]
                for m in matrices[1:]:
                    result *= m
                operation = 'multiply'
            else:
                result = matrices[0]
                operation = 'none'
                    
            return {
                "type": "matrix",
                "operation": operation,
                "matrices": matrices,
                "result": result,
                "steps": [
                    "1. เมทริกซ์ที่กำหนด:",
                    *[f"   Matrix {i+1}:\n{m}" for i, m in enumerate(matrices)],
                    "",
                    "2. ผลลัพธ์:",
                    f"   {result}"
                ]
            }
            
        except Exception as e:
            return {"error": f"ไม่สามารถแก้โจทย์เมทริกซ์ได้: {str(e)}"}

    def _solve_determinant(self, problem: str) -> Dict[str, Any]:
        """หาดีเทอร์มิแนนต์ของเมทริกซ์"""
        try:
            # แยกเมทริกซ์จากโจทย์
            match = re.search(r'\[((?:[^][]|\[(?:[^][]|\[[^]]*\])*\])*)\]', problem)
            if not match:
                return {"error": "ไม่พบเมทริกซ์ในโจทย์"}
            
            matrix_str = match.group(1)
            rows = [row.strip().split() for row in matrix_str.split(';')]
            matrix = sp.Matrix([[sp.sympify(elem) for elem in row] for row in rows])
            
            # หาดีเทอร์มิแนนต์
            det = matrix.det()
            
            # หาไมเนอร์และโคแฟกเตอร์
            minors = matrix.minors()
            cofactors = matrix.cofactor_matrix()
                    
            return {
                "type": "determinant",
                "matrix": matrix,
                "determinant": det,
                "steps": [
                    "1. เมทริกซ์ที่กำหนด:",
                    f"   {matrix}",
                    "",
                    "2. ไมเนอร์:",
                    f"   {minors}",
                    "",
                    "3. โคแฟกเตอร์:",
                    f"   {cofactors}",
                    "",
                    "4. ดีเทอร์มิแนนต์:",
                    f"   det = {det}"
                ]
            }
            
        except Exception as e:
            return {"error": f"ไม่สามารถหาดีเทอร์มิแนนต์ได้: {str(e)}"}

    def _solve_eigenvalue(self, problem: str) -> Dict[str, Any]:
        """หาค่าไอเกนและเวกเตอร์ไอเกน"""
        try:
            # แยกเมทริกซ์จากโจทย์
            match = re.search(r'\[((?:[^][]|\[(?:[^][]|\[[^]]*\])*\])*)\]', problem)
            if not match:
                return {"error": "ไม่พบเมทริกซ์ในโจทย์"}
            
            matrix_str = match.group(1)
            rows = [row.strip().split() for row in matrix_str.split(';')]
            matrix = sp.Matrix([[sp.sympify(elem) for elem in row] for row in rows])
            
            # หาค่าไอเกนและเวกเตอร์ไอเกน
            eigenvals = matrix.eigenvals()
            eigenvects = matrix.eigenvects()
            
            # จัดรูปผลลัพธ์
            eigenvalues = []
            eigenvectors = []
            multiplicities = []
            
            for val, mult, vects in eigenvects:
                eigenvalues.append(val)
                multiplicities.append(mult)
                eigenvectors.append(vects)
                    
            return {
                "type": "eigenvalue",
                "matrix": matrix,
                "eigenvalues": eigenvalues,
                "eigenvectors": eigenvectors,
                "multiplicities": multiplicities,
                "steps": [
                    "1. เมทริกซ์ที่กำหนด:",
                    f"   {matrix}",
                    "",
                    "2. ค่าไอเกน:",
                    *[f"   λ₍{i+1}₎ = {val} (ความทวีคูณ = {mult})"
                      for i, (val, mult) in enumerate(zip(eigenvalues, multiplicities))],
                    "",
                    "3. เวกเตอร์ไอเกน:",
                    *[f"   v₍{i+1}₎ = {vects}"
                      for i, vects in enumerate(eigenvectors)]
                ]
            }
            
        except Exception as e:
            return {"error": f"ไม่สามารถหาค่าไอเกนและเวกเตอร์ไอเกนได้: {str(e)}"}

    def solve_integral(self, problem: str) -> Dict[str, Any]:
        """แก้โจทย์อินทิกรัล"""
        try:
            # แยกฟังก์ชันที่ต้องการหาอินทิกรัล
            match = re.search(r'∫\s*([^d]+)(?:dx|dt|du)', problem)
            if not match:
                return {"error": "ไม่พบฟังก์ชันที่ต้องการหาอินทิกรัล"}
            
            func_str = match.group(1).strip()
            print(f"ฟังก์ชันที่ต้องการหาอินทิกรัล: {func_str}")  # Debug log
            
            # แปลงเป็น sympy expression
            x = sp.Symbol('x')
            func = sp.sympify(func_str)
            
            # หาอินทิกรัล
            result = sp.integrate(func, x)
            
            steps = [
                f"1. ฟังก์ชันที่ต้องการหาอินทิกรัล:",
                f"   ∫ {func_str} dx",
                "",
                f"2. ผลลัพธ์:",
                f"   {result} + C"
            ]
                        
            return {
                "type": "calculus_integral",
                "function": func_str,
                "result": f"{result} + C",
                "steps": steps
            }
            
        except Exception as e:
            return {"error": f"ไม่สามารถหาอินทิกรัลได้: {str(e)}"}

    def solve_derivative(self, problem: str) -> Dict[str, Any]:
        """แก้โจทย์อนุพันธ์"""
        try:
            # แยกฟังก์ชันที่ต้องการหาอนุพันธ์
            match = re.search(r'd/dx\s*\(([^)]+)\)', problem)
            if not match:
                return {"error": "ไม่พบฟังก์ชันที่ต้องการหาอนุพันธ์"}
            
            func_str = match.group(1).strip()
            print(f"ฟังก์ชันที่ต้องการหาอนุพันธ์: {func_str}")  # Debug log
            
            # แปลงเป็น sympy expression
            x = sp.Symbol('x')
            func = sp.sympify(func_str)
            
            # หาอนุพันธ์
            result = sp.diff(func, x)
            
            steps = [
                f"1. ฟังก์ชันที่ต้องการหาอนุพันธ์:",
                f"   d/dx({func_str})",
                "",
                f"2. ผลลัพธ์:",
                f"   {result}"
            ]
            
            return {
                "type": "calculus_derivative",
                "function": func_str,
                "result": str(result),
                "steps": steps
            }
            
        except Exception as e:
            return {"error": f"ไม่สามารถหาอนุพันธ์ได้: {str(e)}"}

    def solve_limit(self, problem: str) -> Dict[str, Any]:
        """แก้โจทย์ลิมิต"""
        try:
            # แยกฟังก์ชันและค่าที่ x เข้าใกล้
            match = re.search(r'lim\s*_{x→([^}]+)}\s*([^$]+)', problem)
            if not match:
                return {"error": "ไม่พบฟังก์ชันที่ต้องการหาลิมิต"}
            
            limit_point = match.group(1).strip()
            func_str = match.group(2).strip()
            
            print(f"ค่า x เข้าใกล้: {limit_point}")  # Debug log
            print(f"ฟังก์ชัน: {func_str}")  # Debug log
            
            # แปลงเป็น sympy expression
            x = sp.Symbol('x')
            func = sp.sympify(func_str)
            point = sp.sympify(limit_point)
            
            # หาลิมิต
            result = sp.limit(func, x, point)
            
            steps = [
                f"1. ฟังก์ชันที่ต้องการหาลิมิต:",
                f"   lim_(x→{limit_point}) {func_str}",
                "",
                f"2. ผลลัพธ์:",
                f"   {result}"
            ]
            
            return {
                "type": "calculus_limit",
                "function": func_str,
                "limit_point": limit_point,
                "result": str(result),
                "steps": steps
            }
            
        except Exception as e:
            return {"error": f"ไม่สามารถหาลิมิตได้: {str(e)}"}

    def visualize_solution(self, result: Dict[str, Any]) -> None:
        """สร้างกราฟหรือภาพประกอบการแก้โจทย์"""
        try:
            if result.get("type") in [
                "complex_analysis",
                "differential_geometry",
                "chaos_theory",
                "quantum_mechanics"
            ]:
                self.advanced_solver.visualize_solution(result)
            else:
                # การแสดงผลสำหรับโจทย์พื้นฐาน
                if "function" in result:
                    x = np.linspace(-10, 10, 1000)
                    y = eval(result["function"].replace("x", "x_val"), {"x_val": x, "np": np})
                    plt.figure(figsize=(10, 6))
                    plt.plot(x, y)
                    plt.grid(True)
                    plt.title(f"Graph of {result['function']}")
                    plt.xlabel("x")
                    plt.ylabel("y")
                    plt.show()
        except Exception as e:
            print(f"ไม่สามารถสร้างภาพประกอบได้: {str(e)}")

def main():
    solver = MathSolver()
    # Add test cases here
    
if __name__ == "__main__":
    main() 