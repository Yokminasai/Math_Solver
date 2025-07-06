#!/usr/bin/env python3
"""
Test GUI problems via command line
ทดสอบโจทย์ GUI แบบ command line
"""

from math_solver import MathSolver
from advanced_math import AdvancedMathSolver
import time

def test_gui_problems():
    """ทดสอบโจทย์ที่ GUI ควรรองรับ"""
    
    print("="*80)
    print("🎯 ทดสอบโจทย์ GUI")
    print("="*80)
    
    solver = MathSolver()
    advanced_solver = AdvancedMathSolver()
    
    # โจทย์ที่ GUI ควรรองรับ
    gui_problems = [
        # 1. Basic arithmetic
        {
            "problem": "2 + 3 * 4",
            "expected_type": "arithmetic",
            "description": "การคำนวณพื้นฐาน"
        },
        
        # 2. Simple equation
        {
            "problem": "x + 5 = 10",
            "expected_type": "equation",
            "description": "สมการเชิงเส้น"
        },
        
        # 3. System of equations
        {
            "problem": "x + y = 5\n2x - y = 1",
            "expected_type": "system_of_equations",
            "description": "ระบบสมการ"
        },
        
        # 4. Advanced - Complex analysis
        {
            "problem": "residue of 1/(z^2+1) at z=i",
            "expected_type": "complex_analysis",
            "description": "Complex Analysis - Residue"
        },
        
        # 5. Advanced - Functional analysis
        {
            "problem": "is operator A=[[1,2],[0,3]] hermitian?",
            "expected_type": "functional_analysis",
            "description": "Functional Analysis - Hermitian"
        },
        
        # 6. Advanced - Chaos theory
        {
            "problem": "lyapunov exponent of logistic map x->4x(1-x)",
            "expected_type": "chaos_theory",
            "description": "Chaos Theory - Lyapunov"
        },
        
        # 7. Advanced - Quantum mechanics
        {
            "problem": "eigenstates of quantum harmonic oscillator",
            "expected_type": "quantum_mechanics",
            "description": "Quantum Mechanics - Eigenstates"
        },
        
        # 8. Advanced - Optimization
        {
            "problem": "minimize x^2+y^2 subject to x+y=1",
            "expected_type": "optimization",
            "description": "Optimization - Lagrange"
        }
    ]
    
    results = []
    
    for i, test_case in enumerate(gui_problems, 1):
        print(f"\n{'='*60}")
        print(f"🧮 โจทย์ที่ {i}: {test_case['description']}")
        print(f"📝 โจทย์: {test_case['problem']}")
        print(f"🎯 ประเภทที่คาดหวัง: {test_case['expected_type']}")
        print("-" * 60)
        
        start_time = time.time()
        
        try:
            # ใช้ advanced solver สำหรับทุกโจทย์ (เหมือน GUI)
            result = advanced_solver.solve_advanced_problem(test_case['problem'])
            
            end_time = time.time()
            solve_time = end_time - start_time
            
            # ตรวจสอบผลลัพธ์
            if "error" in result:
                print(f"❌ Error: {result['error']}")
                success = False
            else:
                detected_type = result.get('problem_type', 'unknown')
                print(f"✅ Success! Detected type: {detected_type}")
                print(f"⏱️ เวลาที่ใช้: {solve_time:.2f} วินาที")
                
                # แสดงผลลัพธ์
                if 'result' in result:
                    print(f"📊 Result: {result['result']}")
                elif 'answer' in result:
                    print(f"📊 Answer: {result['answer']}")
                
                # แสดงจำนวนขั้นตอน
                if 'steps' in result:
                    print(f"📝 Steps: {len(result['steps'])} ขั้นตอน")
                    # แสดงขั้นตอนแรก 3 ขั้นตอน
                    for j, step in enumerate(result['steps'][:3], 1):
                        print(f"   {j}. {step}")
                    if len(result['steps']) > 3:
                        print(f"   ... และอีก {len(result['steps'])-3} ขั้นตอน")
                
                success = True
            
            # บันทึกผลลัพธ์
            results.append({
                "problem": test_case['problem'],
                "description": test_case['description'],
                "expected_type": test_case['expected_type'],
                "detected_type": detected_type if 'detected_type' in locals() else 'unknown',
                "success": success,
                "solve_time": solve_time,
                "error": result.get('error', None) if not success else None
            })
            
        except Exception as e:
            print(f"❌ Exception: {str(e)}")
            results.append({
                "problem": test_case['problem'],
                "description": test_case['description'],
                "expected_type": test_case['expected_type'],
                "detected_type": "exception",
                "success": False,
                "solve_time": time.time() - start_time,
                "error": str(e)
            })
    
    # สรุปผลการทดสอบ
    print(f"\n{'='*80}")
    print("📊 สรุปผลการทดสอบ GUI")
    print("="*80)
    
    total_tests = len(results)
    successful_tests = sum(1 for r in results if r['success'])
    failed_tests = total_tests - successful_tests
    
    print(f"🎯 โจทย์ทั้งหมด: {total_tests}")
    print(f"✅ สำเร็จ: {successful_tests}")
    print(f"❌ ล้มเหลว: {failed_tests}")
    print(f"📈 อัตราความสำเร็จ: {(successful_tests/total_tests)*100:.1f}%")
    
    if failed_tests > 0:
        print(f"\n❌ โจทย์ที่ล้มเหลว:")
        for result in results:
            if not result['success']:
                print(f"   - {result['description']}")
                print(f"     Error: {result['error']}")
    
    # แสดงสถิติเวลา
    successful_times = [r['solve_time'] for r in results if r['success']]
    if successful_times:
        avg_time = sum(successful_times) / len(successful_times)
        max_time = max(successful_times)
        min_time = min(successful_times)
        print(f"\n⏱️ สถิติเวลา (โจทย์ที่สำเร็จ):")
        print(f"   เฉลี่ย: {avg_time:.2f} วินาที")
        print(f"   สูงสุด: {max_time:.2f} วินาที")
        print(f"   ต่ำสุด: {min_time:.2f} วินาที")
    
    print(f"\n{'='*80}")
    print("🎉 การทดสอบ GUI เสร็จสิ้น!")
    print("="*80)
    
    return results

def test_specific_gui_problem():
    """ทดสอบโจทย์เฉพาะที่ GUI ควรรองรับ"""
    
    print("\n🔬 ทดสอบโจทย์เฉพาะ GUI")
    print("-" * 40)
    
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
            # ใช้ advanced solver (เหมือน GUI)
            result = advanced_solver.solve_advanced_problem(problem)
            
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
    print("🚀 เริ่มต้นระบบทดสอบ GUI")
    
    choice = input("\nเลือกการทดสอบ:\n1. ทดสอบโจทย์ GUI ที่กำหนดไว้\n2. ทดสอบโจทย์ที่กำหนดเอง\nเลือก (1/2): ").strip()
    
    if choice == "1":
        test_gui_problems()
    elif choice == "2":
        test_specific_gui_problem()
    else:
        print("❌ ตัวเลือกไม่ถูกต้อง") 