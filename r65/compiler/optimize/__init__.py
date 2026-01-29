"""
R65 Compiler Optimization Passes.

This module contains optimization passes that operate on MIR programs.
"""

from r65.compiler.optimize.dead_function_elim import DeadFunctionEliminator
from r65.compiler.optimize.dead_code_elim import DeadCodeEliminator
from r65.compiler.optimize.peephole import PeepholeOptimizer, optimize_nodes
from r65.compiler.optimize.inline import FunctionInliner
from r65.compiler.optimize.far_to_near import FarToNearOptimizer, RunFarToNearOptimizer

__all__ = [
    'DeadFunctionEliminator',
    'DeadCodeEliminator',
    'PeepholeOptimizer',
    'optimize_nodes',
    'FunctionInliner',
    'FarToNearOptimizer',
    'RunFarToNearOptimizer',
]
