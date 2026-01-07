"""
R65 Compiler Optimization Passes.

This module contains optimization passes that operate on MIR programs.
"""

from r65.compiler.optimize.dead_function_elim import DeadFunctionEliminator

__all__ = ['DeadFunctionEliminator']
