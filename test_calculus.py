#!/usr/bin/env python3
"""
Test script for calculus problems
"""

import time
from math_solver import MathSolver

def test_calculus():
    """Test various calculus problems"""
    solver = MathSolver()
    
    test_cases = [
        {
            "name": "Complex Integral - Partial Fractions",
            "problem": "integral of (2*x^2-1)/((4*x-1)*(x^2+1)) dx",
            "expected_type": "calculus"
        },
        {
            "name": "Simple Integral",
            "problem": "integral of x^2 dx",
            "expected_type": "calculus"
        },
        {
            "name": "Trigonometric Integral",
            "problem": "integral of sin(x) dx",
            "expected_type": "calculus"
        },
        {
            "name": "Derivative",
            "problem": "derivative of x^3 + 2*x^2 + 3*x + 1",
            "expected_type": "calculus"
        },
        {
            "name": "Limit",
            "problem": "limit of (x^2-1)/(x-1) as x approaches 1",
            "expected_type": "calculus"
        }
    ]
    
    print("🚀 เริ่มต้นทดสอบ Calculus")
    print("=" * 60)
    
    success_count = 0
    total_time = 0
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🧮 โจทย์ที่ {i}: {test_case['name']}")
        print(f"📝 โจทย์: {test_case['problem']}")
        print("-" * 50)
        
        start_time = time.time()
        try:
            result = solver.solve_calculus(test_case['problem'])
            end_time = time.time()
            elapsed = end_time - start_time
            total_time += elapsed
            
            if "error" in result:
                print(f"❌ Error: {result['error']}")
            else:
                print(f"✅ Success! Time: {elapsed:.3f}s")
                print(f"📊 Answer: {result.get('answer', 'N/A')}")
                print(f"📝 Steps: {len(result.get('steps', []))} steps")
                success_count += 1
                
        except Exception as e:
            print(f"❌ Exception: {str(e)}")
    
    print("\n" + "=" * 60)
    print("📊 สรุปผลการทดสอบ Calculus")
    print("=" * 60)
    print(f"🎯 โจทย์ทั้งหมด: {len(test_cases)}")
    print(f"✅ สำเร็จ: {success_count}")
    print(f"❌ ล้มเหลว: {len(test_cases) - success_count}")
    print(f"📈 อัตราความสำเร็จ: {(success_count/len(test_cases)*100):.1f}%")
    print(f"⏱️ เวลารวม: {total_time:.3f} วินาที")
    print("=" * 60)

if __name__ == "__main__":
    test_calculus() 