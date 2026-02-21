"""
Validates operator usage according to R65 hardware cost model.

Operators are restricted to hardware-efficient operations:
- Multiply (*) only for power-of-2 constants (1, 2, 4, 8, 16, 32, 64, 128, 256)
- Divide (/) only for power-of-2 constants (1, 2, 4, 8, 16, 32, 64, 128, 256)
- Shift (<<, >>) only for constant amounts
- Register-specific operations based on 65816 hardware capabilities
"""

from typing import Optional
from r65.compiler.hir import *
from r65.compiler.typeck.errors import *
from r65.compiler.typeck.register_capabilities import (
    can_register_do_binary_op,
    get_register_hint,
)


class OperatorValidator:
    """Validates operator restrictions."""

    POWER_OF_TWO_CONSTANTS = {1, 2, 4, 8, 16, 32, 64, 128, 256}

    # Operators that can be used for increment/decrement
    # X/Y support ++ (X = X + 1) and -- (X = X - 1) but not general add/subtract
    INCREMENT_DECREMENT_OPS = {'+', '-'}

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
                f"multiply operator (*) requires a power-of-2 constant operand (1 to 256)",
                source_loc=op_node.source_loc,
                hint="use mul8() or mul16() for general multiplication"
            )

        # Extract constant value
        const_value = left.value if left_const else right.value

        # Validate power-of-two
        if const_value not in OperatorValidator.POWER_OF_TWO_CONSTANTS:
            raise TypeCheckError(
                f"multiply by {const_value} not supported (only powers of 2 from 1 to 256)",
                source_loc=op_node.source_loc,
                hint=f"use mul8(value, {const_value}) or mul16(value, {const_value}) instead"
            )

    @staticmethod
    def _validate_divide(op_node: HIRBinaryOp):
        """Validate divide operator."""
        right = op_node.right

        if not isinstance(right, HIRIntegerLiteral):
            raise TypeCheckError(
                f"divide operator (/) requires a power-of-2 constant divisor (1 to 256)",
                source_loc=op_node.source_loc,
                hint="use div8() or div16() for general division"
            )

        divisor = right.value

        if divisor not in OperatorValidator.POWER_OF_TWO_CONSTANTS:
            raise TypeCheckError(
                f"divide by {divisor} not supported (only powers of 2 from 1 to 256)",
                source_loc=op_node.source_loc,
                hint=f"use div8(value, {divisor}) or div16(value, {divisor}) instead"
            )

    @staticmethod
    def _validate_shift(op_node: HIRBinaryOp):
        """Validate shift operators (<< and >>)."""
        right = op_node.right

        # Accept integer literals
        if isinstance(right, HIRIntegerLiteral):
            return

        # Accept const identifiers (references to const values)
        if isinstance(right, HIRIdentifier):
            symbol = right.symbol
            if symbol and symbol.kind == SymbolKind.CONST:
                return

        raise TypeCheckError(
            f"shift operator ({op_node.op}) requires a constant shift amount",
            source_loc=op_node.source_loc,
            hint="use shl() or shr() for variable shift amounts"
        )

    @staticmethod
    def validate_register_binary_op(
        op: str,
        target_register: str,
        right_operand: HIRExpression,
        source_loc=None
    ) -> None:
        """
        Validate that a binary operation is allowed on a target register.

        This checks register-specific restrictions based on 65816 hardware:
        - A: Full ALU support (+, -, &, |, ^, <<, >>)
        - X/Y: Only increment (+ 1) and decrement (- 1)
        - B: No binary operations

        Args:
            op: The binary operator
            target_register: Name of the target register (A, X, Y, B)
            right_operand: The right operand (to check for increment/decrement)
            source_loc: Source location for error reporting

        Raises:
            TypeCheckError: If the operation is not allowed on the register
        """
        # Check if register supports this operation
        if can_register_do_binary_op(target_register, op):
            return  # Operation is supported

        # Special case: X/Y can do increment (+ 1) and decrement (- 1)
        if target_register in ('X', 'Y') and op in OperatorValidator.INCREMENT_DECREMENT_OPS:
            if OperatorValidator._is_literal_one(right_operand):
                return  # This is an increment or decrement - allowed

        # Operation not allowed
        hint = get_register_hint(target_register)
        if target_register in ('X', 'Y'):
            hint_extra = f"\n   = hint: move value to A first, perform operation, then transfer to {target_register}"
        else:
            hint_extra = ""

        raise TypeCheckError(
            f"operator '{op}' not allowed on register {target_register}",
            source_loc=source_loc,
            hint=f"{hint}{hint_extra}"
        )

    @staticmethod
    def _is_literal_one(expr: HIRExpression) -> bool:
        """Check if an expression is the literal value 1."""
        if isinstance(expr, HIRIntegerLiteral):
            return expr.value == 1
        return False
