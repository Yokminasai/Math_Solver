#!/usr/bin/env python3
"""
Mathematical Problem Solver GUI
A professional interface for solving mathematical problems
"""

import sys
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import json
try:
    from math_solver import MathSolver
except ImportError as e:
    print(f"Cannot load math_solver module: {str(e)}")
    sys.exit(1)
import traceback
import re
from typing import Dict, Any, Optional
try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    import numpy as np
except ImportError as e:
    print(f"Cannot load required modules: {str(e)}")
    print("Please install the required modules with: pip install -r requirements.txt")
    sys.exit(1)

class ScrollableFrame(ttk.Frame):
    """A scrollable frame widget"""
    def __init__(self, container, *args, **kwargs):
        super().__init__(container, *args, **kwargs)
        self.canvas = tk.Canvas(self, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        self.canvas_frame = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.bind('<Configure>', self.on_canvas_configure)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        self.bind_mouse_wheel()
    def on_canvas_configure(self, event):
        self.canvas.itemconfig(self.canvas_frame, width=event.width)
    def bind_mouse_wheel(self):
        def _on_mousewheel(event):
            self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        self.canvas.bind_all("<MouseWheel>", _on_mousewheel)
    def unbind_mouse_wheel(self):
        self.canvas.unbind_all("<MouseWheel>")

class MathSolverGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Mathematical Problem Solver by Yokception")
        style = ttk.Style()
        style.theme_use('clam')
        self.bg_color = "#f0f0f0"
        self.fg_color = "#333333"
        self.accent_color = "#007acc"
        self.error_color = "#dc3545"
        self.success_color = "#28a745"
        style.configure("Title.TLabel", 
                       font=("Helvetica", 24, "bold"),
                       foreground=self.fg_color,
                       background=self.bg_color,
                       padding=10)
        style.configure("Subtitle.TLabel",
                       font=("Helvetica", 12),
                       foreground=self.fg_color,
                       background=self.bg_color,
                       padding=5)
        style.configure("Result.TLabel",
                       font=("Helvetica", 12),
                       foreground=self.fg_color,
                       background=self.bg_color,
                       padding=5,
                       wraplength=600)
        self.solver = MathSolver()
        self.main_container = ttk.Frame(root, padding="10")
        self.main_container.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        self.main_container.columnconfigure(1, weight=1)
        self.create_header()
        self.create_input_section()
        self.create_output_section()
        self.create_footer()
        self.current_plot = None
    def create_header(self):
        title = ttk.Label(self.main_container, 
                         text="Advanced Mathematical Problem Solver",
                         style="Title.TLabel")
        title.grid(row=0, column=0, columnspan=2, sticky=tk.W)
        subtitle = ttk.Label(self.main_container,
                           text="Enter your mathematical problem below",
                           style="Subtitle.TLabel")
        subtitle.grid(row=1, column=0, columnspan=2, sticky=tk.W)
    def create_input_section(self):
        type_frame = ttk.Frame(self.main_container)
        type_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        type_label = ttk.Label(type_frame, text="Problem type:", style="Subtitle.TLabel")
        type_label.pack(side=tk.LEFT, padx=(0, 10))
        self.problem_type = tk.StringVar(value="auto")
        types = [
            ("Auto Detect", "auto"),
            ("Arithmetic", "arithmetic"),
            ("Equation", "equation"),
            ("Calculus", "calculus"),
            ("Trigonometry", "trigonometry"),
            ("Statistics", "statistics"),
            ("Geometry", "geometry"),
            ("Number Theory", "number_theory"),
            ("Linear Algebra", "linear_algebra"),
            ("Advanced", "advanced"),
            ("PDE/Vector Equation", "pde_or_vector_equation")
        ]
        for text, value in types:
            rb = ttk.Radiobutton(type_frame, text=text, value=value, 
                               variable=self.problem_type)
            rb.pack(side=tk.LEFT, padx=5)
        input_frame = ttk.Frame(self.main_container)
        input_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        symbols_frame = ttk.Frame(input_frame)
        symbols_frame.pack(fill=tk.X, pady=(0, 5))
        symbols = [
            ('∫', 'Integral'),
            ('d/dx', 'Derivative'),
            ('lim', 'Limit'),
            ('Σ', 'Summation'),
            ('π', 'Pi'),
            ('√', 'Square root'),
            ('²', 'Square'),
            ('∞', 'Infinity'),
            ('θ', 'Theta'),
            ('±', 'Plus-minus')
        ]
        for symbol, tooltip in symbols:
            btn = ttk.Button(symbols_frame, text=symbol,
                           command=lambda s=symbol: self.insert_symbol(s))
            btn.pack(side=tk.LEFT, padx=2)
            self.create_tooltip(btn, tooltip)
        self.problem_input = scrolledtext.ScrolledText(input_frame, 
                                                     width=60, 
                                                     height=5,
                                                     font=("Helvetica", 12))
        self.problem_input.pack(fill=tk.BOTH, expand=True)
        button_frame = ttk.Frame(self.main_container)
        button_frame.grid(row=4, column=0, columnspan=2, pady=10)
        solve_button = ttk.Button(button_frame,
                                text="Solve",
                                command=self.solve_problem)
        solve_button.pack(side=tk.LEFT, padx=5)
        clear_button = ttk.Button(button_frame,
                                text="Clear",
                                command=self.clear_all)
        clear_button.pack(side=tk.LEFT, padx=5)
    def insert_symbol(self, symbol: str):
        self.problem_input.insert(tk.INSERT, symbol)
    def create_tooltip(self, widget, text):
        def enter(event):
            x, y, _, _ = widget.bbox("insert")
            x += widget.winfo_rootx() + 25
            y += widget.winfo_rooty() + 20
            self.tooltip = tk.Toplevel(widget)
            self.tooltip.wm_overrideredirect(True)
            self.tooltip.wm_geometry(f"+{x}+{y}")
            label = ttk.Label(self.tooltip, text=text, 
                            background="#ffffe0", relief="solid", borderwidth=1)
            label.pack()
        def leave(event):
            if hasattr(self, 'tooltip'):
                self.tooltip.destroy()
        widget.bind('<Enter>', enter)
        widget.bind('<Leave>', leave)
    def create_output_section(self):
        self.output_frame = ScrollableFrame(self.main_container)
        self.output_frame.grid(row=5, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        self.result_frame = ttk.Frame(self.output_frame.scrollable_frame)
        self.result_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self.steps_frame = ttk.Frame(self.output_frame.scrollable_frame)
        self.steps_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self.plot_frame = ttk.Frame(self.output_frame.scrollable_frame)
        self.plot_frame.pack(fill=tk.BOTH, expand=True, pady=5)
    def create_footer(self):
        """Add a credit label at the bottom of the GUI"""
        credit = ttk.Label(self.main_container,
            text="Powered by Yokception",
            style="Subtitle.TLabel",
            font=("Helvetica", 10, "italic"),
            foreground="#888888")
        credit.grid(row=99, column=0, columnspan=2, sticky=(tk.E), pady=(20, 0))
    def solve_problem(self):
        self.clear_output()
        problem = self.problem_input.get("1.0", tk.END).strip()
        if not problem:
            messagebox.showwarning("Error", "Please enter a problem to solve.")
            return
        print(f"Solving: {problem}")
        try:
            problem_type = self.problem_type.get()
            # --- Auto-detect system of equations ---
            lines = [line.strip() for line in problem.splitlines() if line.strip()]
            if (problem_type in ["auto", "equation"]) and len(lines) > 1 and all('=' in line for line in lines):
                result = self.solver.solve_system_of_equations(lines)
                print(f"Result: {result}")
                self.display_result(result)
                return
            # --- Existing logic ---
            if problem_type == "arithmetic":
                result = self.solver.solve_basic_arithmetic(problem)
            elif problem_type == "equation":
                result = self.solver.solve_equation(problem)
            elif problem_type == "calculus":
                result = self.solver.solve_calculus(problem)
            elif problem_type == "trigonometry":
                result = self.solver.solve_trigonometry(problem)
            elif problem_type == "statistics":
                result = self.solver.solve_statistics(problem)
            elif problem_type == "geometry":
                result = self.solver.solve_geometry(problem)
            elif problem_type == "number_theory":
                result = self.solver.solve_number_theory(problem)
            elif problem_type == "linear_algebra":
                result = self.solver.solve_linear_algebra(problem)
            elif problem_type == "advanced":
                if "complex" in problem.lower():
                    result = self.solver.advanced_solver.solve_complex_analysis(problem)
                elif "group theory" in problem.lower():
                    result = self.solver.advanced_solver.solve_abstract_algebra(problem)
                elif "topology" in problem.lower():
                    result = self.solver.advanced_solver.solve_topology(problem)
                elif "differential geometry" in problem.lower():
                    result = self.solver.advanced_solver.solve_differential_geometry(problem)
                elif "algebraic geometry" in problem.lower():
                    result = self.solver.advanced_solver.solve_algebraic_geometry(problem)
                elif "number theory" in problem.lower():
                    result = self.solver.advanced_solver.solve_number_theory_advanced(problem)
                elif "functional analysis" in problem.lower():
                    result = self.solver.advanced_solver.solve_functional_analysis(problem)
                elif "optimization" in problem.lower():
                    result = self.solver.advanced_solver.solve_optimization(problem)
                elif "chaos theory" in problem.lower():
                    result = self.solver.advanced_solver.solve_chaos_theory(problem)
                elif "quantum" in problem.lower():
                    result = self.solver.advanced_solver.solve_quantum_mechanics(problem)
                else:
                    result = {"error": "Please specify the advanced problem type."}
            elif problem_type == "pde_or_vector_equation":
                self.plot_button.config(state="disabled")
                result = self.solver.solve_pde_or_vector_equation(problem)
            else:
                print("Auto-detecting problem type")
                # Try advanced solver first for complex problems
                if any(keyword in problem.lower() for keyword in [
                    'residue', 'laurent', 'contour', 'complex',
                    'group', 'ring', 'field', 'homomorphism',
                    'zeta', 'elliptic', 'class number',
                    'lagrange', 'optimization', 'kkt',
                    'geodesic', 'curvature', 'christoffel',
                    'quantum', 'hamiltonian', 'eigenstate',
                    'lyapunov', 'bifurcation', 'chaos',
                    'operator', 'spectrum', 'hermitian',
                    'functional', 'banach', 'hilbert'
                ]):
                    print("Advanced problem detected, using advanced solver")
                    result = self.solver.advanced_solver.solve_advanced_problem(problem)
                elif '=' in problem:
                    print("Equation detected")
                    # Use advanced solver for equations too to avoid validation issues
                    result = self.solver.advanced_solver.solve_advanced_problem(problem)
                else:
                    print("Basic arithmetic detected")
                    # Use advanced solver for arithmetic too to avoid validation issues
                    result = self.solver.advanced_solver.solve_advanced_problem(problem)
            print(f"Result: {result}")
            self.display_result(result)
        except Exception as e:
            error_msg = f"Error: {str(e)}\n\n{traceback.format_exc()}"
            print(error_msg)
            self.display_error(error_msg)
    def display_result(self, result: Dict[str, Any]):
        print("Displaying result...")
        if "error" in result:
            print(f"Error found: {result['error']}")
            self.display_error(result["error"])
            return
        try:
            result_container = ttk.Frame(self.result_frame)
            result_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
            problem_type = result.get("type", "Unknown type")
            type_label = ttk.Label(
                result_container,
                text=f"Problem type: {problem_type}",
                style="Result.TLabel",
                font=("Segoe UI", 16, "bold")
            )
            type_label.pack(anchor=tk.W, pady=(0, 12))
            if "equation" in result:
                equation_label = ttk.Label(
                    result_container,
                    text=f"Problem: {result['equation']}",
                    style="Result.TLabel",
                    font=("Segoe UI", 15, "bold")
                )
                equation_label.pack(anchor=tk.W, pady=(0, 10))
            if "answer" in result or "solutions" in result or "result" in result:
                answer_frame = ttk.Frame(result_container)
                answer_frame.pack(fill=tk.X, pady=(0, 12))
                answer_label = ttk.Label(
                    answer_frame,
                    text="Answer:",
                    style="Result.TLabel",
                    font=("Segoe UI", 15, "bold")
                )
                answer_label.pack(side=tk.LEFT)
                if "answer" in result:
                    answer_text = result["answer"]
                elif "solutions" in result:
                    if isinstance(result["solutions"], (list, tuple)):
                        if len(result["solutions"]) > 1:
                            answer_text = ", ".join(f"x₍{i+1}₎ = {sol}" 
                                                  for i, sol in enumerate(result["solutions"]))
                        elif len(result["solutions"]) == 1:
                            answer_text = f"x = {result['solutions'][0]}"
                        else:
                            answer_text = "No algebraic solution"
                    else:
                        answer_text = f"x = {result['solutions']}"
                else:
                    answer_text = str(result["result"])
                answer_value = ttk.Label(
                    answer_frame,
                    text=answer_text,
                    style="Result.TLabel",
                    font=("Segoe UI", 15),
                    foreground=self.success_color
                )
                answer_value.pack(side=tk.LEFT, padx=(8, 0))
            if "steps" in result:
                steps_label = ttk.Label(
                    self.steps_frame,
                    text="Solution Steps:",
                    style="Subtitle.TLabel",
                    font=("Segoe UI", 15, "bold")
                )
                steps_label.pack(anchor=tk.W, pady=(14, 7))
                steps_text = tk.Text(
                    self.steps_frame,
                    wrap=tk.WORD,
                    font=("Segoe UI", 13),
                    height=15,
                    width=70,
                    background=self.bg_color,
                    relief=tk.FLAT,
                    padx=12,
                    pady=8,
                    spacing1=6,
                    spacing3=6
                )
                steps_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
                for step in result["steps"]:
                    # Translate Thai to English if needed
                    step_en = step
                    step_en = step_en.replace("แปลงสมการ/normalize", "Normalize equation")
                    step_en = step_en.replace("ย้ายข้าง", "Rearrange")
                    step_en = step_en.replace("ประเมินค่าด้านขวา", "Evaluate right side")
                    step_en = step_en.replace("แปลง log เป็นเลขชี้กำลัง", "Convert log to exponent form")
                    step_en = step_en.replace("แก้ x", "Solve for x")
                    step_en = step_en.replace("คำตอบ", "Answer")
                    step_en = step_en.replace("คําตอบ", "Answer")
                    step_en = step_en.replace("ตรวจสอบ", "Verify")
                    step_en = step_en.replace("ไม่สามารถประเมินค่าด้านขวา", "Cannot evaluate right side")
                    step_en = step_en.replace("สมการเชิงเส้น (Linear equation)", "Linear equation")
                    step_en = step_en.replace("จัดรูปสมการ", "Simplify equation")
                    step_en = step_en.replace("รวมพจน์", "Combine terms")
                    step_en = step_en.replace("ตั้งสมการ", "Set up equation")
                    step_en = step_en.replace("คำนวณด้านขวา", "Calculate right side")
                    step_en = step_en.replace("คำนวณด้านซ้าย", "Calculate left side")
                    step_en = step_en.replace("คำนวณเลขยกกำลัง", "Calculate exponent")
                    step_en = step_en.replace("แยกตัวแปร", "Isolate variable")
                    step_en = step_en.replace("ไม่สามารถตรวจสอบคำตอบ", "Cannot verify answer")
                    step_en = step_en.replace("เกิดข้อผิดพลาด", "Error occurred")
                    step_en = step_en.replace("สมการ", "Equation")
                    steps_text.insert(tk.END, f"{step_en}\n\n")
                steps_text.configure(state=tk.DISABLED)
            print("Result displayed.")
            self.create_visualization(result)
        except Exception as e:
            error_msg = f"Error displaying result: {str(e)}\n\n{traceback.format_exc()}"
            print(error_msg)
            self.display_error(error_msg)
    def display_error(self, error_msg: str):
        error_label = ttk.Label(self.result_frame,
                              text=f"Error: {error_msg}",
                              foreground=self.error_color,
                              style="Result.TLabel")
        error_label.pack(anchor=tk.W)
    def create_visualization(self, result: Dict[str, Any]):
        if self.current_plot:
            self.current_plot.get_tk_widget().destroy()
            self.current_plot = None
        fig = plt.figure(figsize=(8, 6))
        try:
            if result["type"] == "trigonometry":
                x = np.linspace(-2*np.pi, 2*np.pi, 1000)
                if "function" in result:
                    if result["function"] == "sin":
                        y = np.sin(x)
                        plt.plot(x, y)
                        plt.title("Sine function")
                    elif result["function"] == "cos":
                        y = np.cos(x)
                        plt.plot(x, y)
                        plt.title("Cosine function")
                    elif result["function"] == "tan":
                        mask = np.abs(np.tan(x)) <= 5
                        plt.plot(x[mask], np.tan(x)[mask])
                        plt.title("Tangent function")
                plt.grid(True)
                plt.axhline(y=0, color='k', linewidth=0.5)
                plt.axvline(x=0, color='k', linewidth=0.5)
                plt.xlabel('x')
                plt.ylabel('y')
            elif result["type"] == "geometry":
                if result.get("shape") == "circle":
                    circle = plt.Circle((0, 0), result["radius"], fill=False)
                    ax = plt.gca()
                    ax.add_patch(circle)
                    plt.axis('equal')
                    plt.title(f"Circle (radius = {result['radius']})")
                elif result.get("shape") == "rectangle":
                    width = result["width"]
                    height = result["length"]
                    rect = plt.Rectangle((-width/2, -height/2), width, height, fill=False)
                    ax = plt.gca()
                    ax.add_patch(rect)
                    plt.axis('equal')
                    plt.title(f"Rectangle ({width} × {height})")
                elif result.get("shape") == "triangle":
                    points = result.get("points", [(0,0), (1,0), (0.5,1)])
                    polygon = plt.Polygon(points, fill=False)
                    ax = plt.gca()
                    ax.add_patch(polygon)
                    plt.axis('equal')
                    plt.title("Triangle")
                plt.grid(True)
                plt.axhline(y=0, color='k', linewidth=0.5)
                plt.axvline(x=0, color='k', linewidth=0.5)
            elif result["type"] == "statistics":
                if "numbers" in result:
                    plt.hist(result["numbers"], bins='auto', alpha=0.7, color='skyblue')
                    plt.axvline(result["mean"], color='red', linestyle='dashed', linewidth=2, label=f'Mean = {result["mean"]:.2f}')
                    plt.axvline(result["median"], color='green', linestyle='dashed', linewidth=2, label=f'Median = {result["median"]:.2f}')
                    if isinstance(result["mode"], (int, float)):
                        plt.axvline(result["mode"], color='purple', linestyle='dashed', linewidth=2, label=f'Mode = {result["mode"]:.2f}')
                    plt.title("Data Distribution")
                    plt.xlabel("Value")
                    plt.ylabel("Frequency")
                    plt.legend()
            elif result["type"] in ["equation", "system_of_equations"]:
                if "equation" in result and "solutions" in result:
                    eq_str = result["equation"]
                    try:
                        import sympy as sp
                        left, right = eq_str.split('=')
                        x = sp.symbols('x')
                        left_expr = sp.sympify(left)
                        right_expr = sp.sympify(right)
                        f = sp.lambdify(x, left_expr - right_expr, modules=['numpy'])
                        # --- Auto-adjust x-range for log/sqrt/1/x ---
                        # Try to find a valid domain for plotting
                        x_ranges = [(-10, 10), (0.01, 10), (1e-3, 20), (-20, -0.01)]
                        found_valid = False
                        for x_min, x_max in x_ranges:
                            x_vals = np.linspace(x_min, x_max, 1000)
                            try:
                                y_vals = f(x_vals)
                                mask = np.isfinite(y_vals)
                                if np.any(mask):
                                    plt.plot(x_vals[mask], y_vals[mask], label='f(x) = left - right')
                                    plt.axhline(y=0, color='k', linewidth=0.5)
                                    for sol in result["solutions"]:
                                        try:
                                            x_sol = float(sol)
                                            if x_min <= x_sol <= x_max:
                                                plt.plot(x_sol, 0, 'ro', label=f'Solution x = {x_sol:.2f}')
                                        except Exception:
                                            continue
                                    plt.grid(True)
                                    plt.axvline(x=0, color='k', linewidth=0.5)
                                    plt.title("Equation Solution Graph")
                                    plt.xlabel('x')
                                    plt.ylabel('y')
                                    plt.legend()
                                    found_valid = True
                                    break
                            except Exception:
                                continue
                        if not found_valid:
                            plt.clf()
                            plt.text(0.5, 0.5, 'No valid y values to plot (check function domain)', ha='center', va='center', transform=plt.gca().transAxes)
                            plt.title("Equation Solution Graph (No valid data)")
                            plt.axis('off')
                    except Exception as e:
                        plt.clf()
                        plt.text(0.5, 0.5, f'Cannot plot: {e}', ha='center', va='center', transform=plt.gca().transAxes)
                        plt.title("Equation Solution Graph (Error)")
                        plt.axis('off')
            elif result["type"] == "calculus":
                if "function" in result:
                    x = np.linspace(-5, 5, 1000)
                    try:
                        y = eval(result["function"].replace('x', 'x_val') for x_val in x)
                        plt.plot(x, y, label='f(x)')
                        if "derivative" in result:
                            dy = eval(result["derivative"].replace('x', 'x_val') for x_val in x)
                            plt.plot(x, dy, '--', label="f'(x)")
                        plt.grid(True)
                        plt.axhline(y=0, color='k', linewidth=0.5)
                        plt.axvline(x=0, color='k', linewidth=0.5)
                        plt.title("Function and Derivative")
                        plt.xlabel('x')
                        plt.ylabel('y')
                        plt.legend()
                    except:
                        pass
            elif result["type"] == "system_of_equations":
                eqs = result.get("equations", [])
                sols = result.get("solutions", [])
                # Only plot if 2 variables (x, y)
                if len(eqs) == 2 and all(var in eqs[0]+eqs[1] for var in ['x', 'y']):
                    x = np.linspace(-10, 10, 400)
                    y = np.linspace(-10, 10, 400)
                    X, Y = np.meshgrid(x, y)
                    eq1 = eqs[0].replace('^', '**')
                    eq2 = eqs[1].replace('^', '**')
                    # Build lambda functions safely
                    def parse_eq(eq):
                        left, right = eq.split('=')
                        return f"({left})-({right})"
                    try:
                        f1 = lambda x, y: eval(parse_eq(eq1), {"x": x, "y": y, "np": np, "math": np})
                        f2 = lambda x, y: eval(parse_eq(eq2), {"x": x, "y": y, "np": np, "math": np})
                        plt.contour(X, Y, f1(X, Y), levels=[0], colors='b', linewidths=2, linestyles='solid')
                        plt.contour(X, Y, f2(X, Y), levels=[0], colors='g', linewidths=2, linestyles='dashed')
                        # Plot solution points
                        for sol in sols:
                            if isinstance(sol, dict):
                                xval = sol.get('x')
                                yval = sol.get('y')
                                try:
                                    xval = float(xval)
                                    yval = float(yval)
                                    plt.plot(xval, yval, 'ro', label='Solution')
                                except Exception:
                                    continue
                        plt.xlabel('x')
                        plt.ylabel('y')
                        plt.title('System of Equations Solution')
                        from matplotlib.lines import Line2D
                        legend_elements = [
                            Line2D([0], [0], color='b', lw=2, label='Eq1'),
                            Line2D([0], [0], color='g', lw=2, linestyle='dashed', label='Eq2'),
                            Line2D([0], [0], marker='o', color='w', markerfacecolor='r', markersize=8, label='Solution')
                        ]
                        plt.legend(handles=legend_elements)
                    except Exception as e:
                        print(f'Error plotting system of equations: {e}')
            canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            self.current_plot = canvas
        except Exception as e:
            print(f"Error creating visualization: {str(e)}")
        finally:
            plt.close(fig)
    def clear_output(self):
        for widget in self.result_frame.winfo_children():
            widget.destroy()
        for widget in self.steps_frame.winfo_children():
            widget.destroy()
        if self.current_plot:
            self.current_plot.get_tk_widget().destroy()
            self.current_plot = None
    def clear_all(self):
        self.problem_input.delete("1.0", tk.END)
        self.clear_output()
        self.problem_type.set("auto")

def main():
    root = tk.Tk()
    root.geometry("800x600")
    app = MathSolverGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main() 