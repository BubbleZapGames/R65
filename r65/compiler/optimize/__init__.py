"""
R65 Compiler Optimization Passes.

This module contains optimization passes that operate on MIR programs.
"""

from r65.compiler.optimize.dead_function_elim import DeadFunctionEliminator
from r65.compiler.optimize.dead_code_elim import DeadCodeEliminator

__all__ = ['DeadFunctionEliminator', 'DeadCodeEliminator']
