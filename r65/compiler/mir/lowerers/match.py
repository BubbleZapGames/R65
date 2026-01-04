"""
Match expression lowerer: HIR match expressions → MIR instructions.

Handles pattern matching with two strategies:
- Jump table for dense sequential integer patterns (O(1))
- Conditional branch chain for sparse or complex patterns (O(n))
"""

from typing import TYPE_CHECKING, Union, Dict, Optional, Tuple

from r65.compiler.hir import (
    HIRMatchExpression,
    HIRLiteralPattern, HIREnumPattern, HIRWildcardPattern,
    HIRIdentifierPattern, HIROrPattern,
)
from r65.compiler.mir.nodes import (
    VirtualRegister, HardwareRegister, Immediate,
    Move, Compare, Jump, CondBranch, JumpTable,
)
from r65.compiler.errors import MIRLoweringError

if TYPE_CHECKING:
    from r65.compiler.mir.builder import MIRBuilder
    from r65.compiler.mir.context import LoweringContext


class MatchLowerer:
    """
    Lowers HIR match expressions to MIR instructions.

    Analyzes patterns and chooses between:
    - Jump table for dense sequential integer patterns (O(1))
    - Conditional branch chain for sparse or complex patterns (O(n))

    Calls back to builder.lower_expression() for sub-expression recursion.
    """

    def __init__(self, builder: 'MIRBuilder'):
        """
        Initialize match lowerer.

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
    # Main Entry Point
    # ========================================================================

    def lower_match_expression(self, expr: HIRMatchExpression) -> VirtualRegister:
        """
        Lower match expression to conditional branches or jump table.

        Analyzes patterns and chooses between:
        - Jump table for dense sequential integer patterns (O(1))
        - Conditional branch chain for sparse or complex patterns (O(n))

        Args:
            expr: HIR match expression

        Returns:
            VirtualRegister holding the match result
        """
        # Check if jump table optimization is applicable
        use_jump_table, min_val, max_val, value_to_arm = self._analyze_for_jump_table(expr)

        if use_jump_table:
            return self._lower_with_jump_table(expr, min_val, max_val, value_to_arm)
        else:
            return self._lower_with_branches(expr)

    # ========================================================================
    # Jump Table Analysis
    # ========================================================================

    def _analyze_for_jump_table(self, expr: HIRMatchExpression) -> Tuple[bool, Optional[int], Optional[int], Optional[Dict[int, int]]]:
        """
        Analyze match expression to determine if jump table optimization is applicable.

        Returns:
            tuple: (use_jump_table, min_value, max_value, value_to_arm_index)
                   or (False, None, None, None) if not suitable
        """
        # Extract literal/enum values from patterns
        pattern_values = []
        has_catchall = False

        for i, arm in enumerate(expr.arms):
            if isinstance(arm.pattern, HIRLiteralPattern):
                if isinstance(arm.pattern.value, int):
                    pattern_values.append((arm.pattern.value, i))
                else:
                    # Non-integer literal (e.g., bool) - can't use jump table
                    return (False, None, None, None)
            elif isinstance(arm.pattern, HIREnumPattern):
                # Enum patterns map to integer values
                pattern_values.append((arm.pattern.variant_value, i))
            elif isinstance(arm.pattern, HIRWildcardPattern) or isinstance(arm.pattern, HIRIdentifierPattern):
                has_catchall = True
                # Keep track but don't add to pattern_values
            else:
                # Or-pattern or other complex pattern - skip jump table optimization
                return (False, None, None, None)

        if not pattern_values:
            return (False, None, None, None)

        # Check if patterns form a dense range
        values = [v for v, _ in pattern_values]
        min_val = min(values)
        max_val = max(values)
        range_size = max_val - min_val + 1
        num_patterns = len(pattern_values)

        # Heuristic: Use jump table if:
        # 1. Range is not too large (< 256 entries for 8-bit index)
        # 2. Density is good (>= 50% coverage)
        # 3. We have at least 3 patterns (otherwise linear is fine)
        MAX_JUMP_TABLE_SIZE = 256
        MIN_DENSITY = 0.5
        MIN_PATTERNS = 3

        if range_size > MAX_JUMP_TABLE_SIZE:
            return (False, None, None, None)

        density = num_patterns / range_size
        if density < MIN_DENSITY or num_patterns < MIN_PATTERNS:
            return (False, None, None, None)

        # Build value-to-arm-index mapping
        value_to_arm = {}
        for value, arm_index in pattern_values:
            value_to_arm[value] = arm_index

        return (True, min_val, max_val, value_to_arm)

    # ========================================================================
    # Branch-Based Lowering
    # ========================================================================

    def _lower_with_branches(self, expr: HIRMatchExpression) -> VirtualRegister:
        """
        Lower match expression to conditional branches (fallback/default strategy).

        Lowers to a chain of if-then-else blocks:
        - Compare scrutinee against each pattern
        - Branch to arm body if match
        - Fall through to next pattern if no match
        - Collect results into a result vreg

        Args:
            expr: HIR match expression

        Returns:
            VirtualRegister holding the match result
        """
        # Lower scrutinee once
        scrutinee_vreg = self.builder.lower_expression(expr.scrutinee)
        scrutinee_type = expr.scrutinee.expr_type

        # Allocate result register
        result_vreg = self.ctx.alloc_vreg(expr.expr_type, "match_result")

        # Create merge block (where all arms converge)
        merge_block = self.ctx.new_block()

        # Lower each arm to a chain of conditional branches
        for i, arm in enumerate(expr.arms):
            is_last_arm = (i == len(expr.arms) - 1)

            # Create block for this arm's body
            arm_block = self.ctx.new_block()

            # Create block for next pattern check (or merge if last arm)
            next_block = merge_block if is_last_arm else self.ctx.new_block()

            # Emit pattern matching logic
            self._lower_pattern_match(arm.pattern, scrutinee_vreg, scrutinee_type, arm_block, next_block)

            # Emit arm body in arm_block
            self.ctx.set_current_block(arm_block)

            # Handle identifier pattern binding
            if isinstance(arm.pattern, HIRIdentifierPattern):
                # Bind scrutinee to the pattern variable
                binding_vreg = self.ctx.alloc_vreg(
                    arm.pattern.symbol.var_type,
                    arm.pattern.name
                )
                self.ctx.symbol_to_vreg[id(arm.pattern.symbol)] = binding_vreg
                self.emit(Move(dest=binding_vreg, source=scrutinee_vreg, type_info=arm.pattern.symbol.var_type))

            # Lower arm body
            arm_result = self.builder.lower_expression(arm.body)

            # Move result to result_vreg
            self.emit(Move(dest=result_vreg, source=arm_result, type_info=expr.expr_type))

            # Jump to merge block
            self.emit(Jump(target=merge_block.block_id))
            self.ctx.add_cfg_edge(arm_block, merge_block)

            # Continue with next pattern (if not last)
            if not is_last_arm:
                self.ctx.set_current_block(next_block)

        # Set current block to merge
        self.ctx.set_current_block(merge_block)

        return result_vreg

    # ========================================================================
    # Jump Table Lowering
    # ========================================================================

    def _lower_with_jump_table(self, expr: HIRMatchExpression, min_val: int, max_val: int, value_to_arm: dict) -> VirtualRegister:
        """
        Lower match expression using jump table optimization.

        Generates a jump table for O(1) pattern matching on dense integer ranges.

        Args:
            expr: HIR match expression
            min_val: Minimum pattern value
            max_val: Maximum pattern value
            value_to_arm: Mapping from pattern value to arm index

        Returns:
            VirtualRegister holding the match result
        """
        # Lower scrutinee once
        scrutinee_vreg = self.builder.lower_expression(expr.scrutinee)
        scrutinee_type = expr.scrutinee.expr_type

        # Allocate result register
        result_vreg = self.ctx.alloc_vreg(expr.expr_type, "match_result")

        # Create merge block
        merge_block = self.ctx.new_block()

        # Create blocks for each arm
        arm_blocks = []
        for _ in expr.arms:
            arm_blocks.append(self.ctx.new_block())

        # Find default arm (wildcard or identifier pattern)
        default_arm_index = None
        for i, arm in enumerate(expr.arms):
            if isinstance(arm.pattern, (HIRWildcardPattern, HIRIdentifierPattern)):
                default_arm_index = i
                break

        # Build jump table: array of block IDs indexed by (value - min_val)
        range_size = max_val - min_val + 1
        jump_table = []
        for offset in range(range_size):
            value = min_val + offset
            if value in value_to_arm:
                arm_index = value_to_arm[value]
                jump_table.append(arm_blocks[arm_index].block_id)
            elif default_arm_index is not None:
                # Use default arm for missing values
                jump_table.append(arm_blocks[default_arm_index].block_id)
            else:
                # No default - this shouldn't happen if exhaustiveness checking works
                # For now, jump to merge (unreachable in correct code)
                jump_table.append(merge_block.block_id)

        # Determine default target (for out-of-bounds)
        default_target = arm_blocks[default_arm_index].block_id if default_arm_index is not None else merge_block.block_id

        # Emit jump table instruction
        self.emit(JumpTable(
            scrutinee=scrutinee_vreg,
            base_value=min_val,
            targets=jump_table,
            default_target=default_target,
            type_info=scrutinee_type
        ))

        # Add CFG edges from current block to all possible targets
        current_block = self.ctx.current_block
        current_function = self.ctx.current_function
        for block_id in set(jump_table + [default_target]):
            if block_id in current_function.blocks:
                self.ctx.add_cfg_edge(current_block, current_function.blocks[block_id])

        # Lower each arm body
        for i, arm in enumerate(expr.arms):
            arm_block = arm_blocks[i]
            self.ctx.set_current_block(arm_block)

            # Handle identifier pattern binding
            if isinstance(arm.pattern, HIRIdentifierPattern):
                binding_vreg = self.ctx.alloc_vreg(
                    arm.pattern.symbol.var_type,
                    arm.pattern.name
                )
                self.ctx.symbol_to_vreg[id(arm.pattern.symbol)] = binding_vreg
                self.emit(Move(dest=binding_vreg, source=scrutinee_vreg, type_info=arm.pattern.symbol.var_type))

            # Lower arm body
            arm_result = self.builder.lower_expression(arm.body)

            # Move result to result_vreg
            self.emit(Move(dest=result_vreg, source=arm_result, type_info=expr.expr_type))

            # Jump to merge
            self.emit(Jump(target=merge_block.block_id))
            self.ctx.add_cfg_edge(arm_block, merge_block)

        # Set current block to merge
        self.ctx.set_current_block(merge_block)

        return result_vreg

    # ========================================================================
    # Pattern Matching
    # ========================================================================

    def _lower_pattern_match(self, pattern, scrutinee_vreg, scrutinee_type, match_block, no_match_block):
        """
        Emit code to test if scrutinee matches pattern.
        Branch to match_block if matches, no_match_block otherwise.
        """
        if isinstance(pattern, HIRLiteralPattern):
            # Compare scrutinee with literal value
            literal_vreg = self.ctx.alloc_vreg(
                scrutinee_type,
                f"literal_{pattern.value}"
            )
            self.emit(Move(dest=literal_vreg, source=Immediate(pattern.value), type_info=scrutinee_type))

            # Emit comparison and conditional branch
            self.emit(Compare(left=scrutinee_vreg, right=literal_vreg, comparison="==", type_info=scrutinee_type))
            self.emit(CondBranch(
                condition=None,
                true_target=match_block.block_id,
                false_target=no_match_block.block_id,
                comparison="=="
            ))
            # Add CFG edges
            self.ctx.add_cfg_edge(self.ctx.current_block, match_block)
            self.ctx.add_cfg_edge(self.ctx.current_block, no_match_block)

        elif isinstance(pattern, HIREnumPattern):
            # Compare scrutinee with enum variant value
            variant_vreg = self.ctx.alloc_vreg(
                scrutinee_type,
                f"{pattern.enum_name}_{pattern.variant_name}"
            )
            self.emit(Move(dest=variant_vreg, source=Immediate(pattern.variant_value), type_info=scrutinee_type))

            # Emit comparison and conditional branch
            self.emit(Compare(left=scrutinee_vreg, right=variant_vreg, comparison="==", type_info=scrutinee_type))
            self.emit(CondBranch(
                condition=None,
                true_target=match_block.block_id,
                false_target=no_match_block.block_id,
                comparison="=="
            ))
            # Add CFG edges
            self.ctx.add_cfg_edge(self.ctx.current_block, match_block)
            self.ctx.add_cfg_edge(self.ctx.current_block, no_match_block)

        elif isinstance(pattern, HIRWildcardPattern):
            # Wildcard always matches - unconditional jump
            self.emit(Jump(target=match_block.block_id))
            self.ctx.add_cfg_edge(self.ctx.current_block, match_block)

        elif isinstance(pattern, HIRIdentifierPattern):
            # Identifier always matches - unconditional jump
            # Binding happens in the arm block
            self.emit(Jump(target=match_block.block_id))
            self.ctx.add_cfg_edge(self.ctx.current_block, match_block)

        elif isinstance(pattern, HIROrPattern):
            # Or pattern: try each sub-pattern, jump to match_block if any matches
            for i, subpat in enumerate(pattern.patterns):
                is_last = (i == len(pattern.patterns) - 1)
                next_subpat_block = no_match_block if is_last else self.ctx.new_block()

                self._lower_pattern_match(subpat, scrutinee_vreg, scrutinee_type, match_block, next_subpat_block)

                if not is_last:
                    self.ctx.set_current_block(next_subpat_block)

        else:
            raise MIRLoweringError(f"Unknown pattern type in MIR lowering: {type(pattern).__name__}")
