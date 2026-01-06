"""
Condition lowerer: HIR conditions → MIR branching instructions.

Handles short-circuit evaluation for && and ||, bit testing optimizations,
and comparison operations.
"""

from typing import TYPE_CHECKING, Optional, Tuple

from r65.compiler.hir import HIRBinaryOp, HIRIntegerLiteral, HIRIdentifier, HIRExpression, HIRRegister
from r65.compiler.mir.nodes import (
    VirtualRegister, HardwareRegister, Immediate, MemoryLocation,
    Compare, CondBranch, BitTest,
)
from r65.compiler.errors import MIRLoweringError

if TYPE_CHECKING:
    from r65.compiler.mir.builder import MIRBuilder
    from r65.compiler.mir.context import LoweringContext


class ConditionLowerer:
    """
    Lowers HIR condition expressions to MIR branching instructions.

    Handles:
    - Short-circuit evaluation for && and ||
    - BIT instruction optimization for bit 6/7 testing
    - Direct comparison with Compare + CondBranch
    - General boolean conditions

    Calls back to builder.lower_expression() for sub-expression recursion.
    """

    def __init__(self, builder: 'MIRBuilder'):
        """
        Initialize condition lowerer.

        Args:
            builder: Parent MIR builder for dispatch and helpers
        """
        self.builder = builder

    @property
    def ctx(self) -> 'LoweringContext':
        """Access the lowering context."""
        return self.builder.ctx

    def emit(self, instr):
        """Emit an instruction to the current block."""
        self.builder.emit(instr)

    # ========================================================================
    # Comparison Operand Helpers
    # ========================================================================

    def _lower_compare_operand(self, expr: HIRExpression):
        """
        Lower a comparison operand, preferring direct memory/register access.

        For comparisons like `X == TEMP`, we want to use MemoryLocation directly
        for TEMP rather than loading it into a virtual register. This allows
        CPX/CPY to use direct page addressing instead of stack-relative.

        Returns:
            HardwareRegister, MemoryLocation, Immediate, or VirtualRegister
        """
        # Hardware register → return directly
        if isinstance(expr, HIRRegister):
            return HardwareRegister(expr.name)

        # Integer literal → Immediate
        if isinstance(expr, HIRIntegerLiteral):
            return Immediate(expr.value)

        # Static variable → MemoryLocation (avoids loading into vreg)
        if isinstance(expr, HIRIdentifier):
            symbol = expr.symbol

            # Check if aliased to hardware register
            hw_reg = self.ctx.current_function.alias_tracker.get_alias(symbol)
            if hw_reg:
                return hw_reg

            # Check if this is a static variable with explicit location
            if self.builder.has_explicit_location(symbol):
                return self.builder.get_memory_location(symbol)

        # Fall back to general expression lowering
        return self.builder.lower_expression(expr)

    # ========================================================================
    # Main Entry Point
    # ========================================================================

    def lower_condition(self, condition: HIRExpression, true_target: int, false_target: int):
        """
        Lower condition expression with short-circuit evaluation.

        Generates control flow that branches to true_target or false_target based
        on condition result, with short-circuit for && and || operators.

        Args:
            condition: Condition expression to evaluate
            true_target: Block ID to jump to if condition is true
            false_target: Block ID to jump to if condition is false
        """
        comparison_ops = {'==', '!=', '<', '<=', '>', '>='}

        # OPTIMIZATION 0: BIT instruction for bit testing
        bit_test = self._detect_bit_test_pattern(condition)
        if bit_test:
            value_expr, bit_number, inverted = bit_test

            # OPTIMIZATION: For direct variable access (especially hardware registers),
            # use MemoryLocation directly instead of loading the value
            # This allows: BIT $4212 instead of: LDA $4212; STA temp; BIT temp

            if isinstance(value_expr, HIRIdentifier) and self.builder.has_explicit_location(value_expr.symbol):
                # Direct variable access - use MemoryLocation to avoid load/store
                # This optimization only works for static variables (especially hardware registers)
                symbol = value_expr.symbol
                value = self.builder.get_memory_location(symbol)
            else:
                # Complex expression - need to evaluate it first
                value = self.builder.lower_expression(value_expr)

            # Only use BIT if value is not in a hardware register
            # BIT requires a memory operand
            if not isinstance(value, HardwareRegister):
                # Value is in memory - can use BIT optimization
                # Emit BitTest instruction
                self.emit(BitTest(
                    value=value,
                    test_bit=bit_number,
                    type_info=value_expr.expr_type
                ))

                # Branch based on bit value
                # BMI/BPL for bit 7, BVS/BVC for bit 6
                if inverted:
                    # Test is (value & mask) == 0, so bit is clear
                    # Swap targets: if bit clear goto true, else goto false
                    actual_true = true_target
                    actual_false = false_target
                else:
                    # Test is (value & mask) != 0, so bit is set
                    # Normal: if bit set goto true, else goto false
                    actual_true = true_target
                    actual_false = false_target

                # Emit conditional branch
                # We'll use a special comparison string to indicate BIT-based branch
                if bit_number == 7:
                    comparison = 'bit7_set' if not inverted else 'bit7_clear'
                else:  # bit_number == 6
                    comparison = 'bit6_set' if not inverted else 'bit6_clear'

                self.emit(CondBranch(
                    condition=None,  # Uses flags from BitTest
                    true_target=actual_true,
                    false_target=actual_false,
                    comparison=comparison
                ))

                # Add CFG edges
                self.ctx.add_cfg_edge(self.ctx.current_block, self.ctx.current_function.blocks[true_target])
                self.ctx.add_cfg_edge(self.ctx.current_block, self.ctx.current_function.blocks[false_target])
                return
            # If value is in hardware register, fall through to normal comparison handling

        # OPTIMIZATION 1: Short-circuit AND (&&)
        if isinstance(condition, HIRBinaryOp) and condition.op == '&&':
            # For: if (left && right)
            # - Evaluate left
            # - If left is false, jump to false_target (short-circuit)
            # - Otherwise, evaluate right and use its result
            right_eval_block = self.ctx.new_block()

            # Evaluate left condition
            self.lower_condition(condition.left, right_eval_block.block_id, false_target)

            # If left was true, evaluate right
            self.ctx.set_current_block(right_eval_block)
            self.lower_condition(condition.right, true_target, false_target)
            return

        # OPTIMIZATION 2: Short-circuit OR (||)
        elif isinstance(condition, HIRBinaryOp) and condition.op == '||':
            # For: if (left || right)
            # - Evaluate left
            # - If left is true, jump to true_target (short-circuit)
            # - Otherwise, evaluate right and use its result
            right_eval_block = self.ctx.new_block()

            # Evaluate left condition
            self.lower_condition(condition.left, true_target, right_eval_block.block_id)

            # If left was false, evaluate right
            self.ctx.set_current_block(right_eval_block)
            self.lower_condition(condition.right, true_target, false_target)
            return

        # OPTIMIZATION 3: Direct comparison - emit Compare + CondBranch
        elif isinstance(condition, HIRBinaryOp) and condition.op in comparison_ops:
            # Direct comparison - emit Compare instruction
            # Use _lower_compare_operand to get MemoryLocation directly for static
            # variables, enabling CPX/CPY with DP/absolute addressing
            left = self._lower_compare_operand(condition.left)
            right = self._lower_compare_operand(condition.right)

            # Emit Compare instruction
            self.emit(Compare(
                left=left,
                right=right,
                comparison=condition.op,
                type_info=condition.left.expr_type
            ))

            # Emit conditional branch based on comparison flags
            self.emit(CondBranch(
                condition=None,  # No condition vreg - uses flags from Compare
                true_target=true_target,
                false_target=false_target,
                comparison=condition.op
            ))
            # Add CFG edges
            self.ctx.add_cfg_edge(self.ctx.current_block, self.ctx.current_function.blocks[true_target])
            self.ctx.add_cfg_edge(self.ctx.current_block, self.ctx.current_function.blocks[false_target])

        else:
            # General condition - evaluate to boolean and branch on != 0
            cond_value = self.builder.lower_expression(condition)

            # Emit conditional branch: if condition != 0 goto true, else goto false
            self.emit(CondBranch(
                condition=cond_value,
                true_target=true_target,
                false_target=false_target,
                comparison='!='
            ))
            # Add CFG edges
            self.ctx.add_cfg_edge(self.ctx.current_block, self.ctx.current_function.blocks[true_target])
            self.ctx.add_cfg_edge(self.ctx.current_block, self.ctx.current_function.blocks[false_target])

    # ========================================================================
    # Bit Test Pattern Detection
    # ========================================================================

    def _detect_bit_test_pattern(self, condition: HIRExpression) -> Optional[Tuple]:
        """
        Detect if condition is a bit-testing pattern suitable for BIT instruction.

        Detects patterns:
        - (value & 0x80) != 0  =>  bit 7 test
        - (value & 0x40) != 0  =>  bit 6 test
        - (value & 0x80) == 0  =>  bit 7 test (inverted)
        - (value & 0x40) == 0  =>  bit 6 test (inverted)

        Returns:
            tuple: (value_expr, bit_number, inverted) or None
        """
        # Check if it's a comparison with 0
        if not isinstance(condition, HIRBinaryOp):
            return None

        if condition.op not in ('==', '!='):
            return None

        # Check pattern: (value & mask) op 0
        left = condition.left
        right = condition.right

        # Swap if needed: 0 op (value & mask)
        if isinstance(left, HIRIntegerLiteral) and left.value == 0:
            left, right = right, left

        # Now check: (value & mask) op 0
        if not isinstance(right, HIRIntegerLiteral) or right.value != 0:
            return None

        # Check if left is (value & mask)
        if not isinstance(left, HIRBinaryOp) or left.op != '&':
            return None

        # Check the mask value
        mask_expr = left.right
        if not isinstance(mask_expr, HIRIntegerLiteral):
            return None

        mask = mask_expr.value
        bit_number = None

        if mask == 0x80:
            bit_number = 7
        elif mask == 0x40:
            bit_number = 6
        else:
            return None  # Only support bit 6 and 7

        # Determine if test is inverted (== 0 means inverted)
        inverted = (condition.op == '==')

        return (left.left, bit_number, inverted)
