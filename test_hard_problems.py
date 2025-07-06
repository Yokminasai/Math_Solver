#!/usr/bin/env python3
"""
Test script for hard mathematical problems
ทดสอบระบบด้วยโจทย์คณิตศาสตร์ยากๆ
"""

from math_solver import MathSolver
from advanced_math import AdvancedMathSolver
import json
import time

def test_hard_problems():
    """ทดสอบโจทย์ยากๆ ต่างๆ"""
    
    print("="*80)
    print("🧮 ทดสอบระบบคณิตศาสตร์ด้วยโจทย์ยากๆ")
    print("="*80)
    
    solver = MathSolver()
    advanced_solver = AdvancedMathSolver()
    
    # 1. Complex Analysis Problems
    print("\n📐 1. COMPLEX ANALYSIS PROBLEMS")
    print("-" * 50)
    
    complex_problems = [
        "residue of 1/(z^2+1) at z=i",
        "laurent series of 1/(z(z-1)) around z=0",
        "contour integral of 1/(z^2+4) around |z|=3"
    ]
    
    for i, problem in enumerate(complex_problems, 1):
        print(f"\n{i}. {problem}")
        try:
            result = advanced_solver.solve_complex_analysis(problem)
            if "error" in result:
                print(f"   ❌ Error: {result['error']}")
            else:
                print(f"   ✅ Result: {result.get('result', result)}")
                if 'steps' in result:
                    print(f"   📝 Steps: {len(result['steps'])} steps")
        except Exception as e:
            print(f"   ❌ Exception: {str(e)}")
    
    # 2. Abstract Algebra Problems
    print("\n🔢 2. ABSTRACT ALGEBRA PROBLEMS")
    print("-" * 50)
    
    algebra_problems = [
        "order and generators of symmetric group S4",
        "is Z/6Z a field? find units and zero divisors",
        "isomorphism between Z4 and U(8)"
    ]
    
    for i, problem in enumerate(algebra_problems, 1):
        print(f"\n{i}. {problem}")
        try:
            result = advanced_solver.solve_abstract_algebra(problem)
            if "error" in result:
                print(f"   ❌ Error: {result['error']}")
            else:
                print(f"   ✅ Result: {result.get('result', result)}")
                if 'steps' in result:
                    print(f"   📝 Steps: {len(result['steps'])} steps")
        except Exception as e:
            print(f"   ❌ Exception: {str(e)}")
    
    # 3. Advanced Number Theory Problems
    print("\n🔢 3. ADVANCED NUMBER THEORY PROBLEMS")
    print("-" * 50)
    
    number_theory_problems = [
        "zeta function at s=3",
        "class number of Q(sqrt(-7))",
        "dirichlet L-function L(1,chi) for character mod 4"
    ]
    
    for i, problem in enumerate(number_theory_problems, 1):
        print(f"\n{i}. {problem}")
        try:
            result = advanced_solver.solve_number_theory_advanced(problem)
            if "error" in result:
                print(f"   ❌ Error: {result['error']}")
            else:
                print(f"   ✅ Result: {result.get('result', result)}")
                if 'steps' in result:
                    print(f"   📝 Steps: {len(result['steps'])} steps")
        except Exception as e:
            print(f"   ❌ Exception: {str(e)}")
    
    # 4. Optimization Problems
    print("\n📈 4. OPTIMIZATION PROBLEMS")
    print("-" * 50)
    
    optimization_problems = [
        "minimize x^2+y^2 subject to x+y=1 using lagrange multipliers",
        "minimize x^2 subject to x>=2 using KKT conditions",
        "euler-lagrange equation for functional J[y]=integral(y'^2+y^2)dx"
    ]
    
    for i, problem in enumerate(optimization_problems, 1):
        print(f"\n{i}. {problem}")
        try:
            result = advanced_solver.solve_optimization(problem)
            if "error" in result:
                print(f"   ❌ Error: {result['error']}")
            else:
                print(f"   ✅ Result: {result.get('result', result)}")
                if 'steps' in result:
                    print(f"   📝 Steps: {len(result['steps'])} steps")
        except Exception as e:
            print(f"   ❌ Exception: {str(e)}")
    
    # 5. Differential Geometry Problems
    print("\n🌐 5. DIFFERENTIAL GEOMETRY PROBLEMS")
    print("-" * 50)
    
    geometry_problems = [
        "gaussian curvature of sphere radius R",
        "geodesic equations on torus",
        "christoffel symbols for metric ds^2=dx^2+dy^2"
    ]
    
    for i, problem in enumerate(geometry_problems, 1):
        print(f"\n{i}. {problem}")
        try:
            result = advanced_solver.solve_differential_geometry(problem)
            if "error" in result:
                print(f"   ❌ Error: {result['error']}")
            else:
                print(f"   ✅ Result: {result.get('result', result)}")
                if 'steps' in result:
                    print(f"   📝 Steps: {len(result['steps'])} steps")
        except Exception as e:
            print(f"   ❌ Exception: {str(e)}")
    
    # 6. Quantum Mechanics Problems
    print("\n⚛️ 6. QUANTUM MECHANICS PROBLEMS")
    print("-" * 50)
    
    quantum_problems = [
        "eigenstates and energy levels of quantum harmonic oscillator",
        "commutator [x,p] and uncertainty principle",
        "wave function of particle in a box"
    ]
    
    for i, problem in enumerate(quantum_problems, 1):
        print(f"\n{i}. {problem}")
        try:
            result = advanced_solver.solve_quantum_mechanics(problem)
            if "error" in result:
                print(f"   ❌ Error: {result['error']}")
            else:
                print(f"   ✅ Result: {result.get('result', result)}")
                if 'steps' in result:
                    print(f"   📝 Steps: {len(result['steps'])} steps")
        except Exception as e:
            print(f"   ❌ Exception: {str(e)}")
    
    # 7. Chaos Theory Problems
    print("\n🦋 7. CHAOS THEORY PROBLEMS")
    print("-" * 50)
    
    chaos_problems = [
        "lyapunov exponent of logistic map x->4x(1-x)",
        "bifurcation points of logistic map",
        "fixed points of system dx/dt=x(1-x)"
    ]
    
    for i, problem in enumerate(chaos_problems, 1):
        print(f"\n{i}. {problem}")
        try:
            result = advanced_solver.solve_chaos_theory(problem)
            if "error" in result:
                print(f"   ❌ Error: {result['error']}")
            else:
                print(f"   ✅ Result: {result.get('result', result)}")
                if 'steps' in result:
                    print(f"   📝 Steps: {len(result['steps'])} steps")
        except Exception as e:
            print(f"   ❌ Exception: {str(e)}")
    
    # 8. Functional Analysis Problems
    print("\n🔍 8. FUNCTIONAL ANALYSIS PROBLEMS")
    print("-" * 50)
    
    functional_problems = [
        "spectrum of operator Tf(x)=xf(x) on L2[0,1]",
        "is operator A=[[1,2],[0,3]] hermitian?",
        "norm of vector [3,4] in Hilbert space"
    ]
    
    for i, problem in enumerate(functional_problems, 1):
        print(f"\n{i}. {problem}")
        try:
            result = advanced_solver.solve_functional_analysis(problem)
            if "error" in result:
                print(f"   ❌ Error: {result['error']}")
            else:
                print(f"   ✅ Result: {result.get('result', result)}")
                if 'steps' in result:
                    print(f"   📝 Steps: {len(result['steps'])} steps")
        except Exception as e:
            print(f"   ❌ Exception: {str(e)}")
    
    print("\n" + "="*80)
    print("🎉 การทดสอบเสร็จสิ้น!")
    print("="*80)

def test_specific_problem():
    """ทดสอบโจทย์เฉพาะที่คุณต้องการ"""
    
    print("\n🔬 ทดสอบโจทย์เฉพาะ")
    print("-" * 30)
    
    solver = MathSolver()
    advanced_solver = AdvancedMathSolver()
    
    while True:
        problem = input("\nใส่โจทย์ที่ต้องการทดสอบ (หรือ 'quit' เพื่อออก): ").strip()
        
        if problem.lower() in ['quit', 'exit', 'q']:
            break
            
        if not problem:
            continue
            
        print(f"\n🔍 กำลังแก้โจทย์: {problem}")
        print("-" * 50)
        
        start_time = time.time()
        
        try:
            # ลองใช้ advanced solver ก่อน
            result = advanced_solver.solve_advanced_problem(problem)
            
            if "error" in result:
                # ถ้า advanced solver ล้มเหลว ลองใช้ basic solver
                result = solver.solve_problem(problem)
            
            end_time = time.time()
            
            print(f"⏱️ เวลาที่ใช้: {end_time - start_time:.2f} วินาที")
            
            if "error" in result:
                print(f"❌ Error: {result['error']}")
            else:
                print(f"✅ Type: {result.get('type', 'Unknown')}")
                
                if 'result' in result:
                    print(f"📊 Result: {result['result']}")
                elif 'answer' in result:
                    print(f"📊 Answer: {result['answer']}")
                elif 'solutions' in result:
                    print(f"📊 Solutions: {result['solutions']}")
                
                if 'steps' in result and result['steps']:
                    print(f"\n📝 Solution Steps ({len(result['steps'])} steps):")
                    for i, step in enumerate(result['steps'], 1):
                        print(f"   {i}. {step}")
                
                if 'latex' in result and result['latex']:
                    print(f"\n🔤 LaTeX: {result['latex']}")
                    
        except Exception as e:
            print(f"❌ Exception: {str(e)}")
            import traceback
            print(traceback.format_exc())

if __name__ == "__main__":
    print("🚀 เริ่มต้นระบบทดสอบโจทย์คณิตศาสตร์ยากๆ")
    
    choice = input("\nเลือกการทดสอบ:\n1. ทดสอบโจทย์ทั้งหมด\n2. ทดสอบโจทย์เฉพาะ\nเลือก (1/2): ").strip()
    
    if choice == "1":
        test_hard_problems()
    elif choice == "2":
        test_specific_problem()
    else:
        print("❌ ตัวเลือกไม่ถูกต้อง") 