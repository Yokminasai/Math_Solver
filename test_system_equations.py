#!/usr/bin/env python3
"""
Test script for system of equations
"""

import time
from math_solver import MathSolver

def test_system_equations():
    """Test various system of equations"""
    solver = MathSolver()
    
    test_cases = [
        {
            "name": "Simple Linear System",
            "equations": ["x + y = 5", "2x - y = 1"],
            "expected_vars": ["x", "y"]
        },
        {
            "name": "Complex Nonlinear System",
            "equations": ["xy + 2x + y = 7", "yz + 3y + 2z = 12", "zx + z + 3x = 15"],
            "expected_vars": ["x", "y", "z"]
        },
        {
            "name": "Quadratic System",
            "equations": ["x^2 + y = 5", "x + y^2 = 3"],
            "expected_vars": ["x", "y"]
        },
        {
            "name": "Mixed System",
            "equations": ["2x + 3y - z = 1", "x - y + 2z = 4", "3x + y + z = 7"],
            "expected_vars": ["x", "y", "z"]
        }
    ]
    
    print("🚀 เริ่มต้นทดสอบระบบสมการ")
    print("=" * 60)
    
    success_count = 0
    total_time = 0
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🧮 โจทย์ที่ {i}: {test_case['name']}")
        print(f"📝 สมการ: {test_case['equations']}")
        print("-" * 50)
        
        start_time = time.time()
        try:
            result = solver.solve_system_of_equations(test_case['equations'])
            end_time = time.time()
            elapsed = end_time - start_time
            total_time += elapsed
            
            if "error" in result:
                print(f"❌ Error: {result['error']}")
            else:
                print(f"✅ Success! Time: {elapsed:.3f}s")
                print(f"📊 Solutions: {result.get('solutions', 'N/A')}")
                print(f"📝 Steps: {len(result.get('steps', []))} steps")
                success_count += 1
                
        except Exception as e:
            print(f"❌ Exception: {str(e)}")
    
    print("\n" + "=" * 60)
    print("📊 สรุปผลการทดสอบระบบสมการ")
    print("=" * 60)
    print(f"🎯 โจทย์ทั้งหมด: {len(test_cases)}")
    print(f"✅ สำเร็จ: {success_count}")
    print(f"❌ ล้มเหลว: {len(test_cases) - success_count}")
    print(f"📈 อัตราความสำเร็จ: {(success_count/len(test_cases)*100):.1f}%")
    print(f"⏱️ เวลารวม: {total_time:.3f} วินาที")
    print("=" * 60)

if __name__ == "__main__":
    test_system_equations() 