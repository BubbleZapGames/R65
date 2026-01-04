"""
MIR lowerer classes.

Each lowerer handles a specific category of HIR → MIR transformation:
- ExpressionLowerer: Binary ops, unary ops, casts, array/field access
- MatchLowerer: Pattern matching expressions
- CallLowerer: Function and method calls
- AssignmentLowerer: Variable and field assignments
- ConditionLowerer: Conditional branching with short-circuit
- StatementLowerer: Control flow statements
"""

from r65.compiler.mir.lowerers.expression import ExpressionLowerer
from r65.compiler.mir.lowerers.match import MatchLowerer
from r65.compiler.mir.lowerers.call import CallLowerer
from r65.compiler.mir.lowerers.assignment import AssignmentLowerer
from r65.compiler.mir.lowerers.condition import ConditionLowerer

__all__ = [
    'ExpressionLowerer',
    'MatchLowerer',
    'CallLowerer',
    'AssignmentLowerer',
    'ConditionLowerer',
]
