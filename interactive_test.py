#!/usr/bin/env python3
"""
Interactive test script for the Mathematical Problem Solver
Allows testing various types of mathematical problems with user input
"""

from math_solver import MathSolver
from advanced_math import AdvancedMathSolver
import json
import sys
import traceback
import time

class InteractiveTest:
    def __init__(self):
        """Initialize the interactive test environment"""
        self.solver = MathSolver()
        self.advanced_solver = AdvancedMathSolver()
        self.test_history = []
        
    def print_welcome(self):
        """Print welcome message and instructions"""
        print("\n" + "="*60)
        print("Welcome to the Mathematical Problem Solver Interactive Test")
        print("="*60)
        print("\nAvailable problem types:")
        for problem_type, info in self.solver.problem_types.items():
            print(f"- {problem_type}: {info['description']}")
        print("\nAdvanced problem types:")
        for problem_type, info in self.advanced_solver.problem_types.items():
            print(f"- {problem_type}: {info['description']}")
        print("\nCommands:")
        print("- 'exit' or 'quit': Exit the program")
        print("- 'help': Show this help message")
        print("- 'history': Show test history")
        print("- 'clear': Clear test history")
        print("- 'save': Save test history to file")
        print("- 'load': Load test history from file")
        print("\nEnter your problem or command:")
        
    def run_test(self):
        """Run the interactive test session"""
        self.print_welcome()
        
        while True:
            try:
                # Get user input
                problem = input("\n> ").strip()
                
                # Handle commands
                if problem.lower() in ['exit', 'quit']:
                    print("\nThank you for using the Mathematical Problem Solver!")
                    break
                elif problem.lower() == 'help':
                    self.print_welcome()
                    continue
                elif problem.lower() == 'history':
                    self.show_history()
                    continue
                elif problem.lower() == 'clear':
                    self.clear_history()
                    continue
                elif problem.lower() == 'save':
                    self.save_history()
                    continue
                elif problem.lower() == 'load':
                    self.load_history()
                    continue
                
                # Solve the problem
                print("\nSolving problem...")
                
                # Try advanced solver first
                result = self.advanced_solver.solve_advanced_problem(problem)
                if "error" in result:
                    # If advanced solver fails, try basic solver
                    result = self.solver.solve_problem(problem)
                
                # Store test result
                test_case = {
                    "problem": problem,
                    "result": result,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                self.test_history.append(test_case)
                
                # Display result
                self.display_result(result)
                
            except KeyboardInterrupt:
                print("\n\nTest session interrupted.")
                break
            except Exception as e:
                print(f"\nError: {str(e)}")
                print(traceback.format_exc())
    
    def display_result(self, result: dict):
        """Display the solution result in a formatted way"""
        print("\nResult:")
        print("-"*60)
        
        if "error" in result:
            print(f"Error: {result['error']}")
            return
        
        # Print problem type and description
        print(f"Problem type: {result.get('problem_type', 'Unknown')}")
        if "description" in result:
            print(f"Description: {result['description']}")
        
        # Print equation if present
        if "equation" in result:
            print(f"\nEquation: {result['equation']}")
        
        # Print answer/solutions
        if "answer" in result:
            print(f"\nAnswer: {result['answer']}")
        elif "solutions" in result:
            if isinstance(result["solutions"], dict):
                print("\nSolutions:")
                for var, val in result["solutions"].items():
                    print(f"{var} = {val}")
            else:
                print("\nSolutions:")
                for i, sol in enumerate(result["solutions"], 1):
                    print(f"x₍{i}₎ = {sol}")
        
        # Print steps if present
        if "steps" in result and result["steps"]:
            print("\nSolution steps:")
            for i, step in enumerate(result["steps"], 1):
                print(f"{i}. {step}")
        
        print("-"*60)
    
    def show_history(self):
        """Display test history"""
        if not self.test_history:
            print("\nNo test history available.")
            return
        
        print("\nTest History:")
        print("="*60)
        for i, test in enumerate(self.test_history, 1):
            print(f"\n{i}. Problem: {test['problem']}")
            print(f"   Time: {test['timestamp']}")
            if "error" in test["result"]:
                print(f"   Error: {test['result']['error']}")
            else:
                if "answer" in test["result"]:
                    print(f"   Answer: {test['result']['answer']}")
                elif "solutions" in test["result"]:
                    print(f"   Solutions: {test['result']['solutions']}")
        print("\n" + "="*60)
    
    def clear_history(self):
        """Clear test history"""
        self.test_history = []
        print("\nTest history cleared.")
    
    def save_history(self):
        """Save test history to file"""
        try:
            filename = input("Enter filename to save (default: test_history.json): ").strip()
            if not filename:
                filename = "test_history.json"
            
            with open(filename, 'w') as f:
                json.dump(self.test_history, f, indent=2)
            print(f"\nTest history saved to {filename}")
            
        except Exception as e:
            print(f"\nError saving history: {str(e)}")
    
    def load_history(self):
        """Load test history from file"""
        try:
            filename = input("Enter filename to load (default: test_history.json): ").strip()
            if not filename:
                filename = "test_history.json"
            
            with open(filename, 'r') as f:
                self.test_history = json.load(f)
            print(f"\nTest history loaded from {filename}")
            
        except Exception as e:
            print(f"\nError loading history: {str(e)}")

def main():
    """Main function"""
    tester = InteractiveTest()
    tester.run_test()

if __name__ == "__main__":
    main() 