"""
Validates operator usage according to R65 hardware cost model.

Operators are restricted to hardware-efficient operations:
- Multiply (*) only for constants 1, 2, 4, 8
- Divide (/) only for constants 1, 2, 4, 8
- Shift (<<, >>) only for constant amounts
"""

from r65.compiler.hir import *
from r65.compiler.typeck.errors import *


class OperatorValidator:
    """Validates operator restrictions."""

    POWER_OF_TWO_CONSTANTS = {1, 2, 4, 8}

    @staticmethod
    def validate_binary_op(op_node: HIRBinaryOp):
        """Validate binary operator usage."""
        op = op_node.op

        if op == '*':
            OperatorValidator._validate_multiply(op_node)

        elif op == '/':
            OperatorValidator._validate_divide(op_node)

        elif op in ['<<', '>>']:
            OperatorValidator._validate_shift(op_node)

    @staticmethod
    def _validate_multiply(op_node: HIRBinaryOp):
        """Validate multiply operator."""
        left = op_node.left
        right = op_node.right

        # Check if either operand is a constant
        left_const = isinstance(left, HIRIntegerLiteral)
        right_const = isinstance(right, HIRIntegerLiteral)

        if not left_const and not right_const:
            raise TypeCheckError(
                f"multiply operator (*) requires a constant operand (1, 2, 4, or 8)",
                source_loc=op_node.source_loc,
                hint="use mul8() or mul16() for general multiplication"
            )

        # Extract constant value
        const_value = left.value if left_const else right.value

        # Validate power-of-two
        if const_value not in OperatorValidator.POWER_OF_TWO_CONSTANTS:
            raise TypeCheckError(
                f"multiply by {const_value} not supported (only 1, 2, 4, 8 allowed)",
                source_loc=op_node.source_loc,
                hint=f"use mul8(value, {const_value}) or mul16(value, {const_value}) instead"
            )

    @staticmethod
    def _validate_divide(op_node: HIRBinaryOp):
        """Validate divide operator."""
        right = op_node.right

        if not isinstance(right, HIRIntegerLiteral):
            raise TypeCheckError(
                f"divide operator (/) requires a constant divisor (1, 2, 4, or 8)",
                source_loc=op_node.source_loc,
                hint="use div8() or div16() for general division"
            )

        divisor = right.value

        if divisor not in OperatorValidator.POWER_OF_TWO_CONSTANTS:
            raise TypeCheckError(
                f"divide by {divisor} not supported (only 1, 2, 4, 8 allowed)",
                source_loc=op_node.source_loc,
                hint=f"use div8(value, {divisor}) or div16(value, {divisor}) instead"
            )

    @staticmethod
    def _validate_shift(op_node: HIRBinaryOp):
        """Validate shift operators (<< and >>)."""
        right = op_node.right

        if not isinstance(right, HIRIntegerLiteral):
            raise TypeCheckError(
                f"shift operator ({op_node.op}) requires a constant shift amount",
                source_loc=op_node.source_loc,
                hint="use shl() or shr() for variable shift amounts"
            )
