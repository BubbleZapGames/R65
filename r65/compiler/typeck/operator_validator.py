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
                f"Multiply operator (*) requires at least one constant operand\n"
                f"  Use mul(a, b) for general multiplication\n"
                f"  Operator * is restricted to power-of-2 constants (1, 2, 4, 8)",
                source_loc=op_node.source_loc
            )

        # Extract constant value
        const_value = left.value if left_const else right.value

        # Validate power-of-two
        if const_value not in OperatorValidator.POWER_OF_TWO_CONSTANTS:
            raise TypeCheckError(
                f"Multiply operator (*) only allows constants 1, 2, 4, or 8\n"
                f"  Found: {const_value}\n"
                f"  Use mul(a, {const_value}) for general multiplication\n"
                f"  Operator * is optimized to shift instructions (ASL)",
                source_loc=op_node.source_loc
            )

    @staticmethod
    def _validate_divide(op_node: HIRBinaryOp):
        """Validate divide operator."""
        right = op_node.right

        if not isinstance(right, HIRIntegerLiteral):
            raise TypeCheckError(
                f"Divide operator (/) requires constant divisor\n"
                f"  Use div(a, b) for general division\n"
                f"  Operator / is restricted to power-of-2 constants (1, 2, 4, 8)",
                source_loc=op_node.source_loc
            )

        divisor = right.value

        if divisor not in OperatorValidator.POWER_OF_TWO_CONSTANTS:
            raise TypeCheckError(
                f"Divide operator (/) only allows constants 1, 2, 4, or 8\n"
                f"  Found: {divisor}\n"
                f"  Use div(a, {divisor}) for general division\n"
                f"  Operator / is optimized to shift instructions (LSR)",
                source_loc=op_node.source_loc
            )

    @staticmethod
    def _validate_shift(op_node: HIRBinaryOp):
        """Validate shift operators (<< and >>)."""
        right = op_node.right

        if not isinstance(right, HIRIntegerLiteral):
            raise TypeCheckError(
                f"Shift operator ({op_node.op}) requires constant shift amount\n"
                f"  Use shl(a, n) or shr(a, n) for variable shifts\n"
                f"  Constant shifts are optimized to repeated ASL/LSR instructions",
                source_loc=op_node.source_loc
            )
