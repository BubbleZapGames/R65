# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Expression lowerer: HIR expressions → MIR instructions.

Handles binary ops, unary ops, type casts, array indexing,
field access, dereference, and address-of operations.
"""

from typing import TYPE_CHECKING, Optional

from r65.compiler.hir import (
    HIRBinaryOp, HIRUnaryOp, HIRTypeCast,
    HIRArrayIndex, HIRFieldAccess, HIRDereference, HIRAddressOf,
    HIRIdentifier,
)
from r65.compiler.hir.types import PointerTypeInfo
from r65.compiler.mir.nodes import (
    VirtualRegister, HardwareRegister, Immediate, MemoryLocation, Move,
    Load, LoadIndirect, BinaryOp, UnaryOp, TypeConvert, ToBool, Compare,
    CondBranch,
)
from r65.compiler.errors import MIRLoweringError

if TYPE_CHECKING:
    from r65.compiler.mir.builder import MIRBuilder
    from r65.compiler.mir.context import LoweringContext


class ExpressionLowerer:
    """
    Lowers HIR expressions to MIR instructions.

    Calls back to builder.lower_expression() for sub-expression recursion.
    """

    def __init__(self, builder: 'MIRBuilder'):
        """
        Initialize expression lowerer.

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
    # Binary Operations
    # ========================================================================

    def lower_binary_op(self, expr: HIRBinaryOp):
        """
        Lower binary operation.

        Args:
            expr: HIR binary operation

        Returns:
            VirtualRegister holding result, or Immediate for constant expressions
        """
        # Try constant folding first
        const_result = self._try_eval_const_binary(expr)
        if const_result is not None:
            return Immediate(const_result)

        comparison_ops = {'==', '!=', '<', '<=', '>', '>='}

        if expr.op in comparison_ops:
            return self._lower_comparison_op(expr)
        else:
            return self._lower_arithmetic_op(expr)

    def _try_eval_const_binary(self, expr: HIRBinaryOp) -> Optional[int]:
        """Try to evaluate a binary operation at compile time with type masking."""
        from r65.compiler.hir.hir_const_eval import try_eval_const_binary_masked
        return try_eval_const_binary_masked(expr)

    def _lower_comparison_op(self, expr: HIRBinaryOp) -> VirtualRegister:
        """Lower comparison operation to boolean result."""
        left = self.builder.lower_expression(expr.left)
        right = self.builder.lower_expression(expr.right)

        if expr.op == '==':
            # For equality: compute (left ^ right) and check if zero
            temp = self.ctx.alloc_vreg(expr.left.expr_type, "eq_temp")
            self.emit(BinaryOp(
                dest=temp,
                left=left,
                right=right,
                op='^',
                type_info=expr.left.expr_type
            ))
            return self.builder._emit_conditional_set(
                temp, true_when_nonzero=False,
                result_type=expr.expr_type, hint="eq_result"
            )

        elif expr.op == '!=':
            # For inequality: compute (left ^ right) and check if non-zero
            temp = self.ctx.alloc_vreg(expr.left.expr_type, "ne_temp")
            self.emit(BinaryOp(
                dest=temp,
                left=left,
                right=right,
                op='^',
                type_info=expr.left.expr_type
            ))
            return self.builder._emit_conditional_set(
                temp, true_when_nonzero=True,
                result_type=expr.expr_type, hint="ne_result"
            )

        else:
            # For <, <=, >, >=: relational ops branch on Compare's flags
            # (BCC/BCS/BMI/BPL).
            return self._emit_compare_to_bool(
                left, right, expr.op,
                left_type=expr.left.expr_type,
                result_type=expr.expr_type,
                hint=f"{expr.op}_result",
            )

    def _emit_compare_to_bool(self, left, right, op: str,
                               left_type, result_type, hint: str) -> VirtualRegister:
        """Emit Compare + CondBranch + set 0/1 for a relational comparison.

        Mirrors the if-condition path (condition.py) which uses Compare with
        the comparison op so codegen picks BCC/BCS/BMI/BPL appropriately.
        """
        result = self.ctx.alloc_vreg(result_type, hint)

        true_block = self.builder.cfg_builder.new_block()
        false_block = self.builder.cfg_builder.new_block()
        merge_block = self.builder.cfg_builder.new_block()

        self.emit(Compare(
            left=left,
            right=right,
            comparison=op,
            type_info=left_type,
        ))
        self.emit(CondBranch(
            condition=None,  # uses flags from Compare
            true_target=true_block.block_id,
            false_target=false_block.block_id,
            comparison=op,
        ))
        self.builder.cfg_builder.add_edge(self.builder.current_block, true_block)
        self.builder.cfg_builder.add_edge(self.builder.current_block, false_block)

        self.builder.current_block = true_block
        self.emit(Move(dest=result, source=Immediate(1), type_info=result_type))
        self.builder.jump_to(merge_block)

        self.builder.current_block = false_block
        self.emit(Move(dest=result, source=Immediate(0), type_info=result_type))
        self.builder.jump_to(merge_block)

        self.builder.current_block = merge_block
        return result

    def _lower_arithmetic_op(self, expr: HIRBinaryOp) -> VirtualRegister:
        """Lower arithmetic/bitwise binary operation."""

        folded = self._lower_constant_chain(expr)
        if folded is not None:
            return folded

        left = self.builder.lower_expression(expr.left)

        # OPTIMIZATION: If left operand uses A register (hardware reg or will load into A),
        # and right operand is a simple static variable, pass right as MemoryLocation
        # directly instead of loading it into a vreg (which would clobber A).
        left_uses_a = (
            isinstance(left, HardwareRegister) and left.name == 'A'
        )

        right_memloc = self._try_get_direct_memory_operand(expr.right)

        if left_uses_a and right_memloc is not None:
            # Use memory location directly - avoids Load instruction that would clobber A
            right = right_memloc
        else:
            # Normal path: lower expression (may generate Load instruction)
            if left_uses_a:
                # Lowering the right operand may emit instructions that clobber A
                # (e.g., TypeConvert for zero-extension). Save A to a vreg first
                # so we don't lose the left operand value.
                save_vreg = self.ctx.alloc_vreg(expr.left.expr_type, "save_left")
                self.emit(Move(dest=save_vreg, source=left, type_info=expr.left.expr_type))
                left = save_vreg
            right = self.builder.lower_expression(expr.right)

        result = self.ctx.alloc_vreg(expr.expr_type, f"{expr.op}_result")

        self.emit(BinaryOp(
            dest=result,
            left=left,
            right=right,
            op=expr.op,
            type_info=expr.expr_type
        ))

        return result

    def _lower_constant_chain(self, expr: HIRBinaryOp) -> Optional[VirtualRegister]:
        """One operation for a chain with a constant at each step.

        `player.x + 8 - PLR_BBOX_HWDT` is `x + 6`: integer `+`/`-` wrap at
        the type's width, so a chain of them is `±x + k` whatever the order,
        and `10 - G - 3` is `7 - G` (a reverse subtract when G is in A). A
        chain of one bitwise operator combines its constants the same way.
        Every step must have the root's integer type; a width change is a
        cast node and ends the chain. A chain that cancels out is left
        alone rather than handing back x's own vreg as the result.
        """
        from r65.compiler.hir.hir_const_eval import try_eval_const_int, _get_type_mask
        from r65.compiler.hir.types import BasicTypeInfo, strip_newtype

        def int_name(type_info):
            t = strip_newtype(type_info) if type_info is not None else None
            if isinstance(t, BasicTypeInfo) and t.name in ('u8', 'u16', 'i8', 'i16'):
                return t.name
            return None

        name = int_name(expr.expr_type)
        if name is None or expr.op not in ('+', '-', '&', '|', '^'):
            return None
        m = _get_type_mask(expr.expr_type)
        additive = expr.op in ('+', '-')

        node, sign, k, steps = expr, 1, (0 if additive else None), 0
        while (isinstance(node, HIRBinaryOp) and int_name(node.expr_type) == name
               and ((node.op in ('+', '-')) if additive else node.op == expr.op)):
            lc, rc = try_eval_const_int(node.left), try_eval_const_int(node.right)
            if (lc is None) == (rc is None):
                break
            if not additive:
                c = lc if rc is None else rc
                k = c if k is None else {'&': k & c, '|': k | c, '^': k ^ c}[node.op]
                node = node.right if rc is None else node.left
            elif rc is not None:
                k += sign * (rc if node.op == '+' else -rc)
                node = node.left
            else:
                k += sign * lc
                if node.op == '-':
                    sign = -sign
                node = node.right
            steps += 1

        if steps < 2:
            return None
        k &= m
        identity = {'+': 0, '-': 0, '&': m, '|': 0, '^': 0}[expr.op]
        if sign == 1 and k == identity:
            return None

        x = self.builder.lower_expression(node)
        if isinstance(x, Immediate):
            v = x.value
            if additive:
                return Immediate((sign * v + k) & m)
            return Immediate({'&': v & k, '|': v | k, '^': v ^ k}[expr.op] & m)
        result =self.ctx.alloc_vreg(expr.expr_type, f"{expr.op}_result")
        if not additive:
            left, op, right = x, expr.op, Immediate(k)
        elif sign == -1:
            left, op, right = Immediate(k), '-', x
        elif k > m // 2:
            left, op, right = x, '-', Immediate((-k) & m)
        else:
            left, op, right = x, '+', Immediate(k)
        self.emit(BinaryOp(dest=result, left=left, right=right, op=op,
                           type_info=expr.expr_type))
        return result

    def _try_get_direct_memory_operand(self, expr) -> 'MemoryLocation':
        """
        Try to get a MemoryLocation for an expression without generating Load.

        Returns MemoryLocation if the expression is a simple static variable
        that can be used directly as a memory operand for instructions like ADC.
        Returns None if the expression requires computation (and thus a vreg).

        This optimization avoids clobbering A when loading operands for binary ops.
        """
        from r65.compiler.hir import HIRIdentifier
        from r65.compiler.hir.symbol_table import SymbolKind

        if not isinstance(expr, HIRIdentifier):
            return None

        symbol = expr.symbol

        # Only static variables can be used as direct memory operands
        if symbol.kind != SymbolKind.STATIC_VAR:
            return None

        # Don't use direct memory for variables aliased to hardware registers
        if self.ctx.current_function:
            hw_reg = self.ctx.current_function.alias_tracker.get_alias(symbol)
            if hw_reg:
                return None

        # Get the memory location for this static variable
        return self.builder.get_memory_location(symbol)

    # ========================================================================
    # Unary Operations
    # ========================================================================

    def lower_unary_op(self, expr: HIRUnaryOp) -> VirtualRegister:
        """
        Lower unary operation.

        Args:
            expr: HIR unary operation

        Returns:
            VirtualRegister holding result, or Immediate for constant expressions
        """
        # Try constant folding first
        const_result = self._try_eval_const_unary(expr)
        if const_result is not None:
            return Immediate(const_result)

        operand = self.builder.lower_expression(expr.operand)

        # Ensure operand is a register (not immediate)
        if isinstance(operand, Immediate):
            temp = self.ctx.alloc_vreg(expr.operand.expr_type, "unary_temp")
            self.emit(Move(dest=temp, source=operand, type_info=expr.operand.expr_type))
            operand = temp

        result = self.ctx.alloc_vreg(expr.expr_type, f"{expr.op}_result")

        if expr.op == '!':
            # Typeck admits only a bool, which is 0 or 1: flip bit 0.
            self.emit(BinaryOp(dest=result, left=operand, right=Immediate(1), op='^',
                               type_info=expr.expr_type))
            return result

        self.emit(UnaryOp(
            dest=result,
            operand=operand,
            op=expr.op,
            type_info=expr.expr_type
        ))

        return result

    def _try_eval_const_unary(self, expr: HIRUnaryOp) -> Optional[int]:
        """Try to evaluate a unary operation at compile time with type masking."""
        from r65.compiler.hir.hir_const_eval import try_eval_const_int, _get_type_mask
        value = try_eval_const_int(expr.operand)
        if value is None:
            return None
        if expr.op == '-':
            result = -value
        elif expr.op == '~':
            result = ~value
        elif expr.op == '!':
            return 0 if value else 1
        else:
            return None
        return result & _get_type_mask(expr.expr_type)

    def _try_eval_const_cast(self, expr) -> Optional[int]:
        """Try to evaluate a type cast at compile time with type masking."""
        from r65.compiler.hir.hir_const_eval import try_eval_const_int, _get_type_mask
        value = try_eval_const_int(expr.expr)
        if value is None:
            return None
        return value & _get_type_mask(expr.target_type)

    # ========================================================================
    # Type Casts
    # ========================================================================

    def lower_type_cast(self, expr: HIRTypeCast) -> VirtualRegister:
        """
        Lower type cast (explicit conversion).

        Handles:
        - Widening: u8→u16 (zero-extend), i8→i16 (sign-extend)
        - Narrowing: u16→u8 (truncate to low byte)
        - Same-size reinterpretation: u8↔i8, u16↔i16 (zero-cost)
        - Boolean conversions

        Args:
            expr: HIR type cast expression

        Returns:
            VirtualRegister holding converted value
        """
        # Try constant folding: if the source expression is a compile-time constant,
        # evaluate the cast at compile time and return an immediate.
        const_result = self._try_eval_const_cast(expr)
        if const_result is not None:
            return Immediate(const_result)

        source_operand = self.builder.lower_expression(expr.expr)
        source_type = expr.expr.expr_type
        target_type = expr.target_type

        source_size = self.builder._get_type_size(source_type)
        target_size = self.builder._get_type_size(target_type)

        result = self.ctx.alloc_vreg(target_type, "cast_result")

        # Boolean conversions - check BEFORE size comparisons since bool is 1 byte.
        # Compare the payloads: a newtype stringifies to its own name, so
        # `n as Flag` would skip normalization and store a non-canonical bool.
        from r65.compiler.hir.types import strip_newtype
        source_bool = str(strip_newtype(source_type)) == 'bool'
        target_bool = str(strip_newtype(target_type)) == 'bool'

        if target_bool and not source_bool:
            return self._lower_to_bool(source_operand, source_type, target_type)

        if source_bool and not target_bool:
            # bool -> integer: just move (0 or 1 value already correct)
            self.emit(Move(dest=result, source=source_operand, type_info=target_type))
            return result

        # Same size reinterpretation (zero-cost)
        if source_size == target_size:
            self.emit(Move(dest=result, source=source_operand, type_info=target_type))
            return result

        # Widening (8-bit → 16-bit)
        if source_size == 1 and target_size == 2:
            # OPTIMIZATION: If source is a hardware register with a u16 alias binding,
            # the value is already 16-bit at runtime. Return the register directly
            # to avoid unnecessary spilling to scratch registers.
            if isinstance(source_operand, HardwareRegister):
                alias_tracker = getattr(self.ctx.current_function, 'alias_tracker', None)
                if alias_tracker:
                    binding_type = alias_tracker.get_register_binding_type(source_operand.name)
                    if binding_type and hasattr(binding_type, 'name'):
                        if binding_type.name in ('u16', 'i16'):
                            # Register already holds 16-bit value, use it directly
                            return source_operand

            self.emit(TypeConvert(
                dest=result,
                source=source_operand,
                source_type=source_type,
                target_type=target_type
            ))
            return result

        # Narrowing (16-bit → 8-bit)
        if source_size == 2 and target_size == 1:
            self.emit(TypeConvert(
                dest=result,
                source=source_operand,
                source_type=source_type,
                target_type=target_type
            ))
            return result

        # Pointer to integer cast (for DMA address setup and low-level programming)
        # Far pointer (3 bytes) → u16: extract low 16 bits (address within bank)
        # Near pointer (2 bytes) → u16: same size, direct move
        # Pointer → u8: extract lowest byte
        if isinstance(source_type, PointerTypeInfo):
            if target_size == 2:
                # Cast pointer to u16 - extract low 16 bits of address
                self.emit(TypeConvert(
                    dest=result,
                    source=source_operand,
                    source_type=source_type,
                    target_type=target_type
                ))
                return result
            elif target_size == 1:
                # Cast pointer to u8 - extract lowest byte of address
                self.emit(TypeConvert(
                    dest=result,
                    source=source_operand,
                    source_type=source_type,
                    target_type=target_type
                ))
                return result

        # Integer to pointer cast
        if isinstance(target_type, PointerTypeInfo):
            self.emit(TypeConvert(
                dest=result,
                source=source_operand,
                source_type=source_type,
                target_type=target_type
            ))
            return result

        raise MIRLoweringError(f"Unsupported type cast: {source_type} to {target_type}", source_loc=expr.source_loc)

    def _lower_to_bool(self, source_operand, source_type, target_type) -> VirtualRegister:
        """
        Convert value to boolean (0 = false, non-zero = true).

        Uses branchless ToBool instruction which generates:
            CMP #1    ; C=1 if value >= 1 (non-zero)
            LDA #0
            ADC #0    ; A = carry (0 or 1)
        """
        result = self.ctx.alloc_vreg(target_type, "bool_result")
        self.emit(ToBool(dest=result, source=source_operand, source_type=source_type))
        return result

    # ========================================================================
    # Array Indexing
    # ========================================================================

    def lower_array_index(self, expr: HIRArrayIndex) -> VirtualRegister:
        """
        Lower array indexing.

        Computes: array[index] → Load from (base_address + index * element_size)

        Handles two cases:
        1. Direct array indexing: array[index] where array is a static array
        2. Pointer indexing: ptr[index] where ptr is *T or far *T

        Args:
            expr: HIR array index expression

        Returns:
            VirtualRegister holding array element value
        """
        element_type = expr.expr_type
        element_size = self.builder._get_type_size(element_type)

        index_operand = self.builder.lower_expression(expr.index)
        index_type = expr.index.expr_type  # Type of the index (u8 or u16)

        # Check if this is pointer indexing (ptr[index]) vs array indexing (array[index])
        array_type = expr.array.expr_type
        if isinstance(array_type, PointerTypeInfo):
            # Pointer indexing: load through pointer with indirect addressing
            return self._lower_pointer_index(expr, index_operand, element_size, element_type, array_type)

        # An array reached through a pointer (`self.bytes[i]` via auto-deref,
        # `(*p)[i]`, `p[i][j]`). Folds the field offsets and every index into
        # Y and loads indirect through the pointer (see
        # emit_pointer_deref_array_access).
        path = self.builder.pointer_index_path(expr)
        if path is not None:
            ptr_expr, base_field_offset, terms = path
            result = self.ctx.alloc_vreg(element_type, "deref_arr_elem")
            return self.builder.emit_pointer_deref_array_access(
                ptr_expr=ptr_expr,
                index_expr=expr.index,
                element_size=element_size,
                const_offset=base_field_offset,
                result_type=element_type,
                is_load=True, dest=result,
                inner_terms=terms[:-1],
            )

        # Regular array indexing — resolve the array base (a bare static array
        # or an array that is a field of a statically-located struct).
        base_memloc, reuse_base_key, terms = self.builder.resolve_array_base_memloc(expr.array)
        result = self.ctx.alloc_vreg(element_type, "array_elem")
        if terms:
            # `T[i][j]`: more than one index, summed into X.
            elem_memloc = self.builder.multi_index_memloc(
                base_memloc, terms, expr.index, index_operand, element_size)
            self.emit(Load(dest=result, source=elem_memloc, type_info=element_type))
            return result
        element_size = self.builder.plane_stride(getattr(base_memloc, 'symbol', None),
                                                 element_size)

        if isinstance(index_operand, Immediate):
            return self._lower_constant_index(
                result, base_memloc, index_operand.value, element_size, element_type
            )
        else:
            reuse_key = self.builder.x_index_reuse_key(reuse_base_key, expr.index)
            return self._lower_variable_index(
                result, base_memloc, index_operand, element_size, element_type,
                index_type, reuse_key
            )

    def _lower_pointer_index(self, expr: HIRArrayIndex, index_operand, element_size, element_type, ptr_type: PointerTypeInfo) -> VirtualRegister:
        """
        Lower pointer indexing: ptr[index]

        Uses indirect indexed addressing:
        - *T (near): LDA (ptr),Y
        - far *T: LDA [ptr],Y

        Args:
            expr: HIR array index expression
            index_operand: The index value (Immediate or VirtualRegister)
            element_size: Size of each element in bytes
            element_type: Type of the element
            ptr_type: The pointer type info

        Returns:
            VirtualRegister holding the loaded value
        """
        result = self.ctx.alloc_vreg(element_type, "ptr_elem")

        # Get the pointer value (should be in a vreg for parameters)
        ptr_operand = self.builder.lower_expression(expr.array)

        # Use the index type (not element type) for the Move to Y register.
        # The index type determines the bit width: u16 indices must load as 16-bit
        # to avoid truncation when index >= 256.
        index_type = expr.index.expr_type

        # For indexed addressing, we need the index in Y register
        # If element_size > 1, multiply first
        if element_size > 1:
            index_operand = self.builder.scale_index(index_operand, element_size, index_type)

        # Move index to Y register for indirect indexed addressing
        y_reg = HardwareRegister('Y')
        self.emit(Move(dest=y_reg, source=index_operand, type_info=index_type))

        # If pointer is in a hardware register or immediate, move to a vreg first
        if isinstance(ptr_operand, HardwareRegister):
            ptr_vreg = self.ctx.alloc_vreg(ptr_type, "ptr_temp")
            self.emit(Move(dest=ptr_vreg, source=ptr_operand, type_info=ptr_type))
            ptr_operand = ptr_vreg
        elif isinstance(ptr_operand, Immediate):
            # Immediate pointer (rare, but handle it)
            ptr_vreg = self.ctx.alloc_vreg(ptr_type, "ptr_temp")
            self.emit(Move(dest=ptr_vreg, source=ptr_operand, type_info=ptr_type))
            ptr_operand = ptr_vreg

        # Emit LoadIndirect with Y indexing
        self.emit(LoadIndirect(
            dest=result,
            pointer=ptr_operand,
            is_far=ptr_type.is_far,
            index_register='Y',
            type_info=element_type
        ))

        return result

    def _lower_constant_index(self, result, base_memloc, index_value, element_size, element_type):
        """Lower constant array index with compile-time offset."""
        offset = index_value * element_size
        elem_memloc = self.builder._create_offset_memloc(base_memloc, offset, base_memloc.symbol)

        self.emit(Load(dest=result, source=elem_memloc, type_info=element_type))
        return result

    def _lower_variable_index(self, result, base_memloc, index_operand, element_size, element_type, index_type=None, reuse_key=None):
        """Lower variable array index with indexed addressing."""
        source = self._x_indexed_memloc(base_memloc, index_operand, element_size,
                                        index_type or element_type, reuse_key)
        self.emit(Load(dest=result, source=source, type_info=element_type))
        return result

    def _x_indexed_memloc(self, base_memloc, index_operand, element_size, offset_type, reuse_key):
        """`base,X` with X = index * element_size, shared by loads and stores.

        *offset_type* is the index's type: a u16 index must load X as 16 bits
        or an index past 255 truncates. A u8 index widens before it is scaled.
        Consecutive `arr[i]` with the same (array, index) skip the scale and
        Move when the reuse cache says X still holds it. `base_memloc.offset`
        carries over, so an array that is a struct field (address None,
        offset the field's) still resolves at codegen time.
        """
        if not self.builder.x_index_cache_hit(reuse_key):
            offset_operand = index_operand
            if element_size > 1:
                offset_operand = self.builder.scale_index(index_operand, element_size, offset_type)
            self.emit(Move(dest=HardwareRegister('X'), source=offset_operand, type_info=offset_type))
            self.builder.x_index_cache_set(reuse_key)
        return MemoryLocation(
            storage_type=base_memloc.storage_type,
            address=base_memloc.address,
            symbol=base_memloc.symbol,
            index_register='X',
            offset=base_memloc.offset
        )

    # ========================================================================
    # Field Access
    # ========================================================================

    def lower_field_access(self, expr: HIRFieldAccess) -> VirtualRegister:
        """
        Lower struct field access.

        Computes: struct.field → Load from (base_address + field_offset)
        For array[index].field → Load from (array_base + index * struct_size + field_offset)
        For ptr.field (auto_deref) → Load indirect through pointer + offset

        Args:
            expr: HIR field access expression

        Returns:
            VirtualRegister holding field value
        """
        field_offset = expr.field_offset
        if field_offset is None:
            raise MIRLoweringError(f"Field offset not computed for field: {expr.field_name}", source_loc=expr.source_loc)

        result = self.ctx.alloc_vreg(expr.expr_type, f"field_{expr.field_name}")

        # Handle auto-dereference case (self.field where self is a pointer)
        if getattr(expr, 'auto_deref', False):
            self._lower_pointer_field_access(expr, result, field_offset)
            return result

        # Nested inline aggregate: outer.inner.leaf. Fold the chain's constant
        # offsets and lower against the innermost base.
        if isinstance(expr.base, HIRFieldAccess):
            base, total_offset = self.builder.peel_field_chain(expr)
            self._lower_nested_field_access(expr, result, base, total_offset)
            return result

        if isinstance(expr.base, HIRIdentifier):
            struct_symbol = expr.base.symbol
            # Check if this struct is decomposed into per-field vregs
            field_vregs = self.builder._decomposed_structs.get(id(struct_symbol))
            if field_vregs is not None:
                field_vreg = field_vregs.get(expr.field_name)
                if field_vreg is None:
                    raise MIRLoweringError(f"Unknown field '{expr.field_name}' on decomposed struct", source_loc=expr.source_loc)
                self.emit(Move(dest=result, source=field_vreg, type_info=expr.expr_type))
                return result

            # Simple case: static_struct.field
            base_memloc = self.builder.get_memory_location(struct_symbol)
            field_memloc = self.builder._create_offset_memloc(base_memloc, field_offset, struct_symbol)
            self.emit(Load(dest=result, source=field_memloc, type_info=expr.expr_type))

        elif isinstance(expr.base, HIRDereference):
            # Explicit dereference case: (*ptr).field
            self._lower_deref_field_access(expr, result, field_offset)

        elif isinstance(expr.base, HIRArrayIndex):
            # Array case: array[index].field
            self._lower_array_field_access(expr, result, field_offset)

        else:
            raise MIRLoweringError(
                f"Field access only supports static structs, pointer dereference, "
                f"and array indexing, got: {type(expr.base)}",
                source_loc=expr.source_loc
            )

        return result

    def _lower_nested_field_access(self, expr: HIRFieldAccess, result: VirtualRegister,
                                   base, total_offset: int):
        """Lower `outer.inner.leaf` against the peeled base and folded offset.

        Mirrors the single-level dispatch below, but every case takes the summed
        offset instead of just this node's own.
        """
        if isinstance(base, HIRFieldAccess) and base.auto_deref:
            # Chain bottoms out at a pointer: self.inner.leaf
            self.builder.emit_indirect_field_access(
                base.base, field_offset=total_offset + (base.field_offset or 0),
                result_type=expr.expr_type, is_load=True, dest=result)

        elif isinstance(base, HIRDereference):
            self.builder.emit_indirect_field_access(
                base.pointer, field_offset=total_offset,
                result_type=expr.expr_type, is_load=True, dest=result)

        elif isinstance(base, HIRArrayIndex):
            self.builder.emit_static_array_field_access(
                base, field_offset=total_offset,
                result_type=expr.expr_type, is_load=True, dest=result)

        elif isinstance(base, HIRIdentifier):
            symbol = base.symbol
            self.builder.require_addressable_aggregate(symbol, expr)
            base_memloc = self.builder.get_memory_location(symbol)
            field_memloc = self.builder._create_offset_memloc(base_memloc, total_offset, symbol)
            self.emit(Load(dest=result, source=field_memloc, type_info=expr.expr_type))

        else:
            raise MIRLoweringError(
                f"Nested field access base unsupported: {type(base)}",
                source_loc=expr.source_loc
            )

    def _lower_pointer_field_access(self, expr: HIRFieldAccess, result: VirtualRegister, field_offset: int):
        """Lower pointer-based field access (auto-dereference): self.field where self is *Struct."""
        self.builder.emit_indirect_field_access(
            expr.base, field_offset=field_offset, result_type=expr.expr_type,
            is_load=True, dest=result)

    def _lower_deref_field_access(self, expr: HIRFieldAccess, result: VirtualRegister, field_offset: int):
        """Lower (*ptr).field — the pointer lives inside an HIRDereference base."""
        self.builder.emit_indirect_field_access(
            expr.base.pointer, field_offset=field_offset, result_type=expr.expr_type,
            is_load=True, dest=result)

    def _lower_array_field_access(self, expr: HIRFieldAccess, result: VirtualRegister, field_offset: int):
        """Lower array[index].field access (array_base + index*struct_size + field_offset)."""
        self.builder.emit_static_array_field_access(
            expr.base, field_offset=field_offset, result_type=expr.expr_type,
            is_load=True, dest=result)


    # ========================================================================
    # Dereference
    # ========================================================================

    def lower_dereference(self, expr: HIRDereference) -> VirtualRegister:
        """
        Lower pointer dereference (*ptr).

        Generates LoadIndirect instruction to read through pointer.

        Args:
            expr: HIR dereference expression

        Returns:
            VirtualRegister holding dereferenced value
        """
        from r65.compiler.hir.types import PointerTypeInfo

        ptr_operand = self.builder.lower_expression(expr.pointer)

        pointer_type = expr.pointer.expr_type
        if not isinstance(pointer_type, PointerTypeInfo):
            raise MIRLoweringError(f"Dereference of non-pointer type: {pointer_type}", source_loc=expr.source_loc)

        result = self.ctx.alloc_vreg(expr.expr_type, "deref_result")

        self.emit(LoadIndirect(
            dest=result,
            pointer=ptr_operand,
            is_far=pointer_type.is_far,
            type_info=expr.expr_type
        ))

        return result

    # ========================================================================
    # Address-Of
    # ========================================================================

    def lower_addressof(self, expr: HIRAddressOf) -> VirtualRegister:
        """
        Lower address-of operator (&variable or &array[index]).

        For static variables, loads the address as an immediate value.
        For array indexing, computes base_address + index * element_size.

        Args:
            expr: HIR address-of expression

        Returns:
            VirtualRegister holding the address
        """
        # Handle address-of array index: &array[index]
        if isinstance(expr.operand, HIRArrayIndex):
            return self._lower_addressof_array_index(expr)

        # Handle address-of field access: &var.field or &ptr.field
        if isinstance(expr.operand, HIRFieldAccess):
            return self._lower_addressof_field_access(expr)

        if not isinstance(expr.operand, HIRIdentifier):
            raise MIRLoweringError(
                f"Address-of only supports static variables, array indexing, or field access, got: {type(expr.operand)}",
                source_loc=expr.source_loc
            )

        symbol = expr.operand.symbol

        # If this is a promoted aggregate local, resolve to the synthetic static
        # so codegen can find the allocation
        if id(symbol) in self.builder._promoted_locals:
            symbol = self.builder._promoted_locals[id(symbol)]

        self.builder.get_memory_location(symbol)  # Validate symbol has location

        result = self.ctx.alloc_vreg(expr.expr_type, f"addr_of_{symbol.name}")

        # Create symbolic address immediate (resolved in codegen)
        addr_immediate = Immediate(0, symbol=symbol)

        self.emit(Move(dest=result, source=addr_immediate, type_info=expr.expr_type))

        # Propagate symbol to vreg for type conversion to access (e.g., near-to-far ptr)
        result.symbol = symbol

        return result

    def _lower_addressof_array_index(self, expr: HIRAddressOf) -> VirtualRegister:
        """
        Lower address-of array index: &array[index]

        Computes: base_address + index * element_size

        Array-base shapes:
          1. Bare static array: &ARR[i]
          2. Static struct field array: &AGG.field[i] (incl. nested), folded
             through resolve_array_base_memloc so the field offset rides on
             the symbolic base.
          3. Pointer-relative field array: &ptr.field[i] / &self.field[i] —
             auto-deref field access. The base is a runtime pointer; the
             field offset and scaled index are added to it at runtime.
          4. Indexed pointer: &p[i] — the pointer's value is the base.
          5. Multi-index: &T[i][j], &ROWS[i].name[j].

        Args:
            expr: HIR address-of expression with HIRArrayIndex operand

        Returns:
            VirtualRegister holding pointer to the array element
        """
        from r65.compiler.hir.types import ArrayTypeInfo

        array_index = expr.operand
        array_expr = array_index.array

        # Element size from the array's type (same source as the read/write paths).
        array_type = array_expr.expr_type
        if isinstance(array_type, ArrayTypeInfo):
            element_size = self.builder._get_type_size(array_type.element_type)
        else:
            element_size = 1

        # Case 3: &ptr.field[index] — auto-deref field access over a pointer.
        # The base address only exists at runtime, so emit a runtime add
        # rather than a symbolic Immediate.
        if (isinstance(array_expr, HIRFieldAccess) and
                getattr(array_expr, 'auto_deref', False)):
            return self._lower_addressof_array_index_via_pointer(
                expr, array_expr.base, array_expr.field_offset or 0,
                array_index.index, element_size, array_expr.field_name
            )

        # Case 4: &p[index] — indexing a pointer itself (a parameter, a local,
        # any pointer-valued expression). Its value is the base, known only
        # at runtime; the element is the pointee. The static path below would
        # take the pointer variable's own address.
        if isinstance(array_type, PointerTypeInfo):
            return self._lower_addressof_array_index_via_pointer(
                expr, array_expr, 0, array_index.index,
                self.builder._get_type_size(array_type.pointee_type), "ptr"
            )

        # &p[i][j], &(*p)[i], &self.outer.arr[i]: the remaining shapes reached
        # through a pointer.
        path = self.builder.pointer_index_path(array_index)
        if path is not None:
            return self._lower_addressof_pointer_path(expr, *path)

        # &T[i][j], &ROWS[i].name[j]: an element below the outermost index.
        cursor = array_expr
        while isinstance(cursor, HIRFieldAccess) and not getattr(cursor, 'auto_deref', False):
            cursor = cursor.base
        if isinstance(cursor, HIRArrayIndex):
            return self._lower_addressof_multi_index(expr, element_size)

        # Cases 1 and 2 share the symbolic-base path. For a struct-field array,
        # walk the HIRFieldAccess chain summing field offsets so we end at the
        # bare identifier (the static symbol) plus a single folded offset. We
        # do this directly rather than via resolve_array_base_memloc because
        # that helper bakes the field offset into the memloc's *address* (not
        # its *offset* field) when the address is already known, leaving us no
        # way to recover the per-field contribution for a symbolic Immediate.
        base_offset = 0
        cursor = array_expr
        while (isinstance(cursor, HIRFieldAccess) and
                not getattr(cursor, 'auto_deref', False)):
            if cursor.field_offset is None:
                raise MIRLoweringError(
                    f"Field offset not computed for &{cursor.field_name}[..]",
                    source_loc=expr.source_loc
                )
            base_offset += cursor.field_offset
            cursor = cursor.base

        if isinstance(cursor, HIRIdentifier):
            array_symbol = cursor.symbol
            self.builder.get_memory_location(array_symbol)
            element_size = self.builder.plane_stride(array_symbol, element_size)
        else:
            raise MIRLoweringError(
                f"Address-of array index requires a static array or struct field, "
                f"got: {type(array_expr)}",
                source_loc=expr.source_loc
            )

        result = self.ctx.alloc_vreg(expr.expr_type, f"addr_of_{array_symbol.name}_elem")

        # Lower the index
        index_operand = self.builder.lower_expression(array_index.index)

        if isinstance(index_operand, Immediate):
            # Constant index: compute offset at compile time
            offset = base_offset + index_operand.value * element_size

            # Create symbolic address with offset (resolved in codegen)
            addr_immediate = Immediate(offset, symbol=array_symbol)

            self.emit(Move(dest=result, source=addr_immediate, type_info=expr.expr_type))
        else:
            # Variable index: compute base + index * element_size at runtime
            # First, load base address (folded field offset rides on the symbol)
            base_addr = self.ctx.alloc_vreg(expr.expr_type, f"base_addr_{array_symbol.name}")
            addr_immediate = Immediate(base_offset, symbol=array_symbol)

            self.emit(Move(dest=base_addr, source=addr_immediate, type_info=expr.expr_type))

            # index * element_size as a u16 byte offset (a u8 index is
            # widened, so the 16-bit add never reads junk in its high byte)
            offset_vreg = self.builder.scale_index(
                index_operand, element_size, array_index.index.expr_type)

            # Add base + offset
            self.emit(BinaryOp(
                dest=result,
                op='+',
                left=base_addr,
                right=offset_vreg,
                type_info=expr.expr_type
            ))

        # Propagate the array's symbol onto the result so a downstream
        # `as far *T` cast can recover the bank byte. Without this,
        # _emit_near_to_far_pointer falls back to bank=0 and produces a
        # bogus pointer (e.g. for ROM arrays in banks other than 0).
        result.symbol = array_symbol

        return result

    def _lower_addressof_multi_index(self, expr: HIRAddressOf, element_size: int) -> VirtualRegister:
        """Lower `&T[i][j]` / `&ROWS[i].name[j]`: the symbol, plus the folded
        constant offset, plus the sum of every variable index scaled by its
        stride (an `#[soa]` field resolves to its plane)."""
        array_index = expr.operand
        base_memloc, _key, terms = self.builder.resolve_array_base_memloc(array_index.array)
        symbol = base_memloc.symbol
        index_operand = self.builder.lower_expression(array_index.index)
        const, total = self.builder.sum_index_terms(
            terms, array_index.index, index_operand, element_size)

        # resolve_array_base_memloc folds a known address into `address` and an
        # unknown one into `offset`; either way, express it against the symbol.
        if base_memloc.address is not None:
            offset = base_memloc.address - self.builder.get_memory_location(symbol).address
        else:
            offset = base_memloc.offset
        offset += const
        addr_immediate = Immediate(offset, symbol=symbol)

        result = self.ctx.alloc_vreg(expr.expr_type, f"addr_of_{symbol.name}_elem")
        if total is None:
            self.emit(Move(dest=result, source=addr_immediate, type_info=expr.expr_type))
            return result
        base_addr = self.ctx.alloc_vreg(expr.expr_type, f"base_addr_{symbol.name}")
        self.emit(Move(dest=base_addr, source=addr_immediate, type_info=expr.expr_type))
        self.emit(BinaryOp(dest=result, op='+', left=base_addr, right=total,
                           type_info=expr.expr_type))
        return result

    def _lower_addressof_pointer_path(self, expr: HIRAddressOf, ptr_expr,
                                      field_offset: int, terms) -> VirtualRegister:
        """Lower `&p[i][j]` and the like: the pointer's value, plus the folded
        field offsets and constant indices, plus the sum of the variable
        indices scaled by their strides (see pointer_index_path)."""
        ptr_type = expr.expr_type
        array_index = expr.operand
        base_ptr = self.builder.lower_expression(ptr_expr)
        index_operand = self.builder.lower_expression(array_index.index)
        const, total = self.builder.sum_index_terms(
            terms[:-1], array_index.index, index_operand, terms[-1][1])

        result = base_ptr
        for addend in (Immediate(field_offset + const) if field_offset + const else None,
                       total):
            if addend is None:
                continue
            summed = self.ctx.alloc_vreg(ptr_type, "addr_of_ptr_elem")
            self.emit(BinaryOp(dest=summed, op='+', left=result, right=addend,
                               type_info=ptr_type))
            result = summed
        if result is base_ptr:
            result = self.ctx.alloc_vreg(ptr_type, "addr_of_ptr_elem")
            self.emit(Move(dest=result, source=base_ptr, type_info=ptr_type))
        return result

    def _lower_addressof_array_index_via_pointer(
        self, expr, base_expr, field_offset, index_expr, element_size, name
    ):
        """Lower &ptr.field[index] or &ptr[index] — a pointer-relative address.

        Produces: base_ptr + field_offset + index * element_size, all at
        runtime, since the pointer's value is only known dynamically.

        base_expr lowers to the pointer vreg: an auto-deref field access's
        .base (field_offset its field's), or the indexed pointer itself
        (field_offset 0). element_size is the size of what is indexed.
        """

        ptr_type = expr.expr_type

        result = self.ctx.alloc_vreg(ptr_type, f"addr_of_{name}_elem")

        # Lower base pointer once. Same lowering path that
        # _lower_addressof_field_access uses for &ptr.field.
        base_ptr = self.builder.lower_expression(base_expr)

        index_operand = self.builder.lower_expression(index_expr)

        # Fast paths for constant index: fold everything into one immediate
        # add (or a plain Move when total offset is zero, e.g. &self.field[0]
        # with field at offset 0).
        if isinstance(index_operand, Immediate):
            total_offset = field_offset + index_operand.value * element_size
            if total_offset == 0:
                self.emit(Move(dest=result, source=base_ptr, type_info=ptr_type))
            else:
                self.emit(BinaryOp(
                    dest=result,
                    op='+',
                    left=base_ptr,
                    right=Immediate(total_offset),
                    type_info=ptr_type
                ))
            return result

        # Variable index: scale to a u16 byte offset, then add to
        # (base + field_offset).
        offset_vreg = self.builder.scale_index(
            index_operand, element_size, index_expr.expr_type)

        # Fold the field offset into the base pointer first so the final add
        # is a single (vreg + vreg) op — no triple-add MIR node exists.
        if field_offset != 0:
            shifted = self.ctx.alloc_vreg(ptr_type, "ptr_field_base")
            self.emit(BinaryOp(
                dest=shifted,
                op='+',
                left=base_ptr,
                right=Immediate(field_offset),
                type_info=ptr_type
            ))
            base_ptr = shifted

        self.emit(BinaryOp(
            dest=result,
            op='+',
            left=base_ptr,
            right=offset_vreg,
            type_info=ptr_type
        ))
        return result

    def _lower_addressof_field_access(self, expr: HIRAddressOf) -> VirtualRegister:
        """
        Lower address-of field access: &var.field or &ptr.field

        Computes: base_address + field_offset

        For static variables: static_address + field_offset
        For pointer auto-deref: pointer_value + field_offset
        """

        field_access = expr.operand
        field_offset = field_access.field_offset or 0

        if field_access.auto_deref:
            # &ptr.field — ptr is already a pointer, compute ptr + offset
            base_vreg = self.builder.lower_expression(field_access.base)
            result = self.ctx.alloc_vreg(expr.expr_type, f"addr_of_field_{field_access.field_name}")

            if field_offset == 0:
                self.emit(Move(dest=result, source=base_vreg, type_info=expr.expr_type))
            else:
                offset_imm = Immediate(field_offset)
                self.emit(BinaryOp(
                    dest=result,
                    op='+',
                    left=base_vreg,
                    right=offset_imm,
                    type_info=expr.expr_type
                ))
            return result

        # &ARR[i].field — the element's address plus the field's offset.
        # `&ARR[i]` already has a lowering that knows how to scale the index and
        # fold a symbolic base, so reuse it rather than repeating that here.
        # This is what lets a tagged enum's payload binding be a method receiver: `t`
        # expands to `POOL__data[i].Variant`, so `t.method()` needs the address
        # of a field of an array element.
        if isinstance(field_access.base, HIRArrayIndex):
            element_addr = HIRAddressOf(operand=field_access.base,
                                        expr_type=expr.expr_type,
                                        source_loc=expr.source_loc)
            base_vreg = self._lower_addressof_array_index(element_addr)
            result = self.ctx.alloc_vreg(
                expr.expr_type, f"addr_of_elem_{field_access.field_name}")
            if field_offset == 0:
                self.emit(Move(dest=result, source=base_vreg,
                               type_info=expr.expr_type))
            else:
                self.emit(BinaryOp(dest=result, op='+', left=base_vreg,
                                   right=Immediate(field_offset),
                                   type_info=expr.expr_type))
            return result

        # &static_var.field — compute static address + offset
        if not isinstance(field_access.base, HIRIdentifier):
            raise MIRLoweringError(
                f"Address-of field access requires static variable or pointer base, got: {type(field_access.base)}",
                source_loc=expr.source_loc
            )

        symbol = field_access.base.symbol
        self.builder.get_memory_location(symbol)

        result = self.ctx.alloc_vreg(expr.expr_type, f"addr_of_{symbol.name}_{field_access.field_name}")

        addr_immediate = Immediate(field_offset, symbol=symbol)

        self.emit(Move(dest=result, source=addr_immediate, type_info=expr.expr_type))
        result.symbol = symbol

        return result
