"""
Condition lowerer: HIR conditions → MIR branching instructions.

Handles short-circuit evaluation for && and ||, bit testing optimizations,
and comparison operations.
"""

from typing import TYPE_CHECKING, Optional, Tuple

from r65.compiler.hir import HIRBinaryOp, HIRUnaryOp, HIRIntegerLiteral, HIRBooleanLiteral, HIRIdentifier, HIRExpression, HIRRegister, HIRStatusFlagAccess
from r65.compiler.mir.nodes import (
    VirtualRegister, HardwareRegister, Immediate, MemoryLocation,
    Compare, CondBranch, BitTest, StatusFlagTest, BinaryOp,
    Load, Move,
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
    - Bitwise expression optimization (ORA/AND/EOR + branch on Z flag)
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

        # OPTIMIZATION: Direct STATUS flag condition (e.g., if STATUS.Carry)
        if isinstance(condition, HIRStatusFlagAccess):
            self._lower_status_flag_condition(condition, true_target, false_target, inverted=False)
            return

        # OPTIMIZATION: Negated STATUS flag condition (e.g., if !STATUS.Carry)
        if isinstance(condition, HIRUnaryOp) and condition.op == '!':
            if isinstance(condition.operand, HIRStatusFlagAccess):
                self._lower_status_flag_condition(condition.operand, true_target, false_target, inverted=True)
                return

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

        # OPTIMIZATION: Bitwise expression conditions (ORA/AND/EOR + branch)
        # For patterns like: if (A | B), if (X & mask), if !(A ^ B)
        if self._try_bitwise_condition(condition, true_target, false_target):
            return

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
    # STATUS Flag Condition Lowering
    # ========================================================================

    def _lower_status_flag_condition(self, flag_expr: HIRStatusFlagAccess,
                                      true_target: int, false_target: int, inverted: bool):
        """
        Lower STATUS flag condition to optimized branch instructions.

        For branchable flags (Carry, Zero, Overflow, Negative):
            - Emits StatusFlagTest (no-op) + CondBranch with status-specific comparison
            - Generates: BCS/BCC, BEQ/BNE, BVS/BVC, BMI/BPL

        For non-branchable flags (Irq, Decimal, Index, Accumulator):
            - Emits StatusFlagTest which generates PHP; PLA; AND #mask
            - Then uses BNE/BEQ based on result

        Args:
            flag_expr: STATUS flag access expression
            true_target: Block ID if condition is true
            false_target: Block ID if condition is false
            inverted: True if condition is negated (e.g., !STATUS.Carry)
        """
        from r65.compiler.hir.status_flags import is_branchable_flag

        flag_name = flag_expr.flag_name

        # Emit StatusFlagTest instruction
        # For branchable flags, this is a no-op (codegen handles it)
        # For non-branchable flags, this emits PHP; PLA; AND #mask
        self.emit(StatusFlagTest(
            flag_name=flag_name,
            bit_position=flag_expr.bit_position,
            bit_mask=flag_expr.bit_mask
        ))

        # Determine comparison string for code generation
        if is_branchable_flag(flag_name):
            # Branchable flag: use direct branch instruction
            if inverted:
                comparison = f'status_{flag_name.lower()}_clear'
            else:
                comparison = f'status_{flag_name.lower()}_set'
        else:
            # Non-branchable flag: test was PHP; PLA; AND #mask
            # Result is in A: 0 if flag clear, non-zero if flag set
            # Use != 0 for set, == 0 for clear
            if inverted:
                comparison = 'status_nonbranch_clear'  # BEQ (result == 0)
            else:
                comparison = 'status_nonbranch_set'    # BNE (result != 0)

        self.emit(CondBranch(
            condition=None,  # Uses flags from StatusFlagTest
            true_target=true_target,
            false_target=false_target,
            comparison=comparison
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

    # ========================================================================
    # Bitwise Expression Condition Optimization
    # ========================================================================

    def _try_bitwise_condition(self, condition: HIRExpression,
                                true_target: int, false_target: int) -> bool:
        """
        Try to optimize bitwise expressions used as conditions.

        Handles patterns like:
        - if (A | B) {}         → ORA + BNE
        - if (A & mask) {}      → AND + BNE
        - if (X ^ Y) {}         → EOR + BNE
        - if (A | B | C) {}     → chained ORA + BNE
        - if !(A | B) {}        → ORA + BEQ (inverted)
        - if ((A | B) == 0) {}  → ORA + BEQ
        - if ((A | B) != 0) {}  → ORA + BNE
        - if ((A & mask) > 5) {} → AND + CMP + branch

        Returns True if optimization was applied, False otherwise.
        """
        expr = condition
        inverted = False
        comparison_op = None
        compare_value = None

        # Handle negation: if !(expr)
        if isinstance(expr, HIRUnaryOp) and expr.op == '!':
            expr = expr.operand
            inverted = True

        # Handle explicit comparison with zero: if (expr != 0) or if (expr == 0)
        if isinstance(expr, HIRBinaryOp) and expr.op in ('==', '!='):
            if self._is_zero_literal(expr.right):
                if expr.op == '==':
                    inverted = not inverted
                expr = expr.left
            elif self._is_zero_literal(expr.left):
                if expr.op == '==':
                    inverted = not inverted
                expr = expr.right

        # Handle other comparisons: if (expr > N), if (expr <= N), etc.
        # These need to compute the bitwise result then compare
        if isinstance(expr, HIRBinaryOp) and expr.op in ('<', '<=', '>', '>='):
            inner_expr = expr.left
            if self._is_bitwise_expr(inner_expr) and self._all_operands_safe(inner_expr):
                comparison_op = expr.op
                compare_value = expr.right
                expr = inner_expr

        # Check if expr is a bitwise operation
        if not self._is_bitwise_expr(expr):
            return False

        # Check all operands are safe (non-volatile, no side effects)
        if not self._all_operands_safe(expr):
            return False

        # Emit optimized bitwise chain
        result = self._emit_bitwise_chain(expr)

        # Emit comparison if needed (for > < >= <=)
        if comparison_op is not None:
            compare_operand = self._lower_compare_operand(compare_value)
            self.emit(Compare(
                left=result,
                right=compare_operand,
                comparison=comparison_op,
                type_info=expr.expr_type
            ))
            self.emit(CondBranch(
                condition=None,  # Uses flags from Compare
                true_target=true_target,
                false_target=false_target,
                comparison=comparison_op
            ))
        else:
            # Direct branch on Z flag from bitwise operation
            # != 0 means branch if non-zero (BNE), == 0 means branch if zero (BEQ)
            branch_op = '==' if inverted else '!='
            self.emit(CondBranch(
                condition=result,
                true_target=true_target,
                false_target=false_target,
                comparison=branch_op
            ))

        # Add CFG edges
        self.ctx.add_cfg_edge(self.ctx.current_block, self.ctx.current_function.blocks[true_target])
        self.ctx.add_cfg_edge(self.ctx.current_block, self.ctx.current_function.blocks[false_target])

        return True

    def _is_zero_literal(self, expr: HIRExpression) -> bool:
        """Check if expression is the literal value 0."""
        return isinstance(expr, HIRIntegerLiteral) and expr.value == 0

    def _is_bitwise_expr(self, expr: HIRExpression) -> bool:
        """Check if expression is a bitwise operation (&, |, ^)."""
        return isinstance(expr, HIRBinaryOp) and expr.op in ('&', '|', '^')

    def _all_operands_safe(self, expr: HIRExpression) -> bool:
        """
        Check all operands in expression tree are safe for optimization.

        Safe operands:
        - Integer/boolean literals
        - Hardware registers (A, X, Y)
        - Non-volatile static variables
        - Nested bitwise ops with safe operands

        Unsafe (require short-circuit or order preservation):
        - Volatile #[hw] variables
        - Function calls
        - Complex expressions with side effects
        """
        # Literals are always safe
        if isinstance(expr, (HIRIntegerLiteral, HIRBooleanLiteral)):
            return True

        # Hardware registers are safe
        if isinstance(expr, HIRRegister):
            return True

        # Static variables - check for volatile
        if isinstance(expr, HIRIdentifier):
            symbol = expr.symbol
            # Reject volatile #[hw] variables - must preserve evaluation order
            if hasattr(symbol, 'hw_address') and symbol.hw_address is not None:
                return False
            return True

        # Bitwise ops on safe operands (recursive check)
        if isinstance(expr, HIRBinaryOp) and expr.op in ('&', '|', '^'):
            return (self._all_operands_safe(expr.left) and
                    self._all_operands_safe(expr.right))

        # Everything else is not safe for this optimization
        return False

    def _get_direct_operand(self, expr: HIRExpression):
        """
        Get operand directly without allocating a virtual register.

        Returns MemoryLocation, Immediate, or HardwareRegister for simple operands.
        Falls back to lower_expression() for complex expressions.
        """
        # Integer literal → Immediate
        if isinstance(expr, HIRIntegerLiteral):
            return Immediate(expr.value)

        # Hardware register → HardwareRegister
        if isinstance(expr, HIRRegister):
            return HardwareRegister(expr.name)

        # Identifier → check for direct access
        if isinstance(expr, HIRIdentifier):
            symbol = expr.symbol

            # Check if aliased to hardware register
            hw_reg = self.ctx.current_function.alias_tracker.get_alias(symbol)
            if hw_reg:
                return hw_reg

            # Check for explicit memory location (zeropage, ram, etc.)
            if self.builder.has_explicit_location(symbol):
                return self.builder.get_memory_location(symbol)

        # Complex expression - fall back to vreg
        return self.builder.lower_expression(expr)

    def _emit_bitwise_chain(self, expr: HIRExpression):
        """
        Emit optimized bitwise chain that operates in the accumulator.

        For (A | B | C): LDA A / ORA B / ORA C
        Result stays in accumulator - no stack temporaries needed.

        Returns HardwareRegister('A') since result is in accumulator.
        """
        if not isinstance(expr, HIRBinaryOp):
            return self._get_direct_operand(expr)

        # Flatten chain of same operator for efficiency
        operands = self._flatten_bitwise_chain(expr, expr.op)

        # Get first operand directly (avoid vreg allocation)
        first = self._get_direct_operand(operands[0])
        acc = HardwareRegister('A')

        # Load first operand into accumulator
        if isinstance(first, MemoryLocation):
            self.emit(Load(
                dest=acc,
                source=first,
                type_info=expr.expr_type
            ))
        elif isinstance(first, Immediate):
            # Use Move for immediate → A
            self.emit(Move(
                dest=acc,
                source=first,
                type_info=expr.expr_type
            ))
        elif isinstance(first, HardwareRegister):
            if first.name != 'A':
                # Transfer X/Y to A
                self.emit(Move(
                    dest=acc,
                    source=first,
                    type_info=expr.expr_type
                ))
            # else: already in A, no load needed
        elif isinstance(first, VirtualRegister):
            # Complex expression result - transfer from vreg to A
            self.emit(Move(
                dest=acc,
                source=first,
                type_info=expr.expr_type
            ))

        # Apply bitwise operations with remaining operands
        for operand in operands[1:]:
            right = self._get_direct_operand(operand)

            self.emit(BinaryOp(
                dest=acc,
                left=acc,
                op=expr.op,
                right=right,
                type_info=expr.expr_type
            ))

        return acc

    def _flatten_bitwise_chain(self, expr: HIRExpression, op: str) -> list:
        """
        Flatten chained bitwise ops of the same operator.

        (a | b | c) with op='|' -> [a, b, c]
        (a | (b & c)) with op='|' -> [a, (b & c)]  (different op, don't flatten)
        """
        if isinstance(expr, HIRBinaryOp) and expr.op == op:
            return (self._flatten_bitwise_chain(expr.left, op) +
                    self._flatten_bitwise_chain(expr.right, op))
        return [expr]
