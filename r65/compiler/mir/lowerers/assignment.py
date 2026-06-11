# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Assignment lowerer: HIR assignments → MIR instructions.

Handles variable assignments, register assignments, field access assignments,
array index assignments, and pointer dereference assignments.
"""

from typing import TYPE_CHECKING, Union

from r65.compiler.hir import (
    HIRAssignment, HIRMultiAssignment, HIRBinaryOp, HIRRegister, HIRIdentifier,
    HIRFieldAccess, HIRArrayIndex, HIRDereference, HIRStatusFlagAccess, HIRBooleanLiteral,
    HIRTypeCast,
)
from r65.compiler.hir.types import MultiReturnTypeInfo
from r65.compiler.mir.nodes import (
    VirtualRegister, HardwareRegister, Immediate, MemoryLocation,
    Move, Store, StoreIndirect, BinaryOp, StatusFlagSet, Push, Pull,
)
from r65.compiler.errors import MIRLoweringError

if TYPE_CHECKING:
    from r65.compiler.mir.builder import MIRBuilder
    from r65.compiler.mir.context import LoweringContext


class AssignmentLowerer:
    """
    Lowers HIR assignment expressions to MIR instructions.

    Handles:
    - Variable assignments (identifier targets)
    - Register assignments (direct hardware register targets)
    - Field access assignments (struct.field = value)
    - Array index assignments (array[index] = value)
    - Pointer dereference assignments (*ptr = value)

    Calls back to builder.lower_expression() for sub-expression recursion.
    """

    def __init__(self, builder: 'MIRBuilder'):
        """
        Initialize assignment lowerer.

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

    def lower_assignment(self, expr: HIRAssignment) -> Union[VirtualRegister, HardwareRegister]:
        """
        Lower assignment.

        Args:
            expr: HIR assignment

        Returns:
            VirtualRegister or HardwareRegister with assigned value
        """
        # Operator overloading (Tier A): `a OP= b` on an aggregate was redirected by
        # the type checker to a `a.<op>_assign(&b)` method call.
        if getattr(expr, 'opassign_call', None):
            return self.builder.lower_expression(expr.opassign_call)

        # Clone assignment: `dst = src.clone()` copies an aggregate in place.
        if getattr(expr.value, 'clone_info', None):
            return self.builder._lower_clone_assignment(expr)

        # Reassigning the index variable (i = ..., i++, i += 1, for-increment —
        # all desugar to HIRAssignment with an HIRIdentifier target) invalidates
        # any X-index reuse cached for it. Field/element stores have a
        # HIRFieldAccess/HIRArrayIndex target, so they do not trip this.
        if isinstance(expr.target, HIRIdentifier):
            self.builder.x_index_cache_invalidate_symbol(
                getattr(expr.target, 'symbol', None))

        # OPTIMIZATION: Detect pattern `target = target op value` for hardware registers
        # Generate BinaryOp(dest=target, left=target, op, right=value) directly
        # instead of temp = target op value; target = temp
        if isinstance(expr.value, HIRBinaryOp) and isinstance(expr.target, HIRRegister):
            binary_op = expr.value
            # Check if it's target = target op value
            # The left operand may be wrapped in HIRTypeCast for mode promotion
            left_op = binary_op.left
            if isinstance(left_op, HIRTypeCast):
                left_op = left_op.expr  # Unwrap the cast
            if (isinstance(left_op, HIRRegister) and
                left_op.name == expr.target.name):
                # Direct hardware register op: A = A + TEMP becomes BinaryOp(dest=A, left=A, right=memloc)
                hw_reg = HardwareRegister(expr.target.name)

                # CRITICAL: For reg = reg op EXPR, lowering EXPR must NOT emit
                # instructions that clobber the target register.
                #
                # Safe right operands (emit no A-clobbering instructions):
                # - HIRIdentifier with explicit memory location → MemoryLocation
                # - HIRIntegerLiteral → Immediate
                # - HIRIdentifier/HIRRegister mapped to vreg → existing vreg reference
                #
                # Unsafe right operands (emit TypeConvert/Load that clobber A):
                # - HIRTypeCast, complex expressions
                # For unsafe cases, fall through to the normal path which uses vregs.
                right = None
                if isinstance(binary_op.right, HIRIdentifier):
                    symbol = binary_op.right.symbol
                    # Check if it has explicit memory location (not an alias)
                    alias = self.ctx.current_function.alias_tracker.get_alias(symbol)
                    if alias is None and self.builder.has_explicit_location(symbol):
                        # Use memory location directly - no Load instruction needed
                        right = self.builder.get_memory_location(symbol)

                if right is None:
                    # Try lowering - safe for immediates, vregs, hw regs
                    # but NOT for TypeCast which emits A-clobbering TypeConvert
                    if not isinstance(binary_op.right, HIRTypeCast):
                        right = self.builder.lower_expression(binary_op.right)

                if right is not None:
                    self.emit(BinaryOp(
                        dest=hw_reg,
                        left=hw_reg,
                        op=binary_op.op,
                        right=right,
                        type_info=expr.expr_type
                    ))
                    return hw_reg

                # Right operand is a TypeCast or other complex expression that
                # may clobber A during lowering. Fall through to the normal
                # assignment path which uses vregs for both operands.

        # OPTIMIZATION: Direct memory-to-register load for X/Y registers
        # Avoid going through a virtual register (which gets allocated to stack)
        # This allows LDX/LDY to be used directly instead of LDA; TAX/TAY
        if isinstance(expr.target, HIRRegister) and isinstance(expr.value, HIRIdentifier):
            target_reg = expr.target.name
            if target_reg in ('X', 'Y', 'A'):
                symbol = expr.value.symbol
                # Check if it has explicit memory location (not an alias)
                hw_alias = self.ctx.current_function.alias_tracker.get_alias(symbol)
                if hw_alias is None and self.builder.has_explicit_location(symbol):
                    # Direct load from memory to hardware register
                    hw_reg = HardwareRegister(target_reg)
                    mem_loc = self.builder.get_memory_location(symbol)
                    self.emit(Move(dest=hw_reg, source=mem_loc, type_info=expr.expr_type))
                    return hw_reg

        # Lower value
        value = self.builder.lower_expression(expr.value)

        # When a tuple-returning function is used in a single-target assignment,
        # lower_function_call returns None (it skips vreg allocation for tuple returns).
        # The first element of the tuple is in the A register.
        if value is None:
            from r65.compiler.hir.types import MultiReturnTypeInfo
            value_type = getattr(expr.value, 'expr_type', None)
            if isinstance(value_type, MultiReturnTypeInfo):
                value = HardwareRegister('A')

        # Lower target
        if isinstance(expr.target, HIRIdentifier):
            return self._lower_identifier_assignment(expr, value)

        elif isinstance(expr.target, HIRRegister):
            return self._lower_register_assignment(expr, value)

        elif isinstance(expr.target, HIRFieldAccess):
            return self._lower_field_assignment(expr, value)

        elif isinstance(expr.target, HIRArrayIndex):
            return self._lower_array_assignment(expr, value)

        elif isinstance(expr.target, HIRDereference):
            return self._lower_dereference_assignment(expr, value)

        elif isinstance(expr.target, HIRStatusFlagAccess):
            return self._lower_status_flag_assignment(expr)

        else:
            # Unsupported target
            raise MIRLoweringError(f"Unsupported assignment target: {type(expr.target)}", source_loc=expr.source_loc)

    # ========================================================================
    # Identifier Assignment
    # ========================================================================

    def _lower_identifier_assignment(self, expr: HIRAssignment, value) -> Union[VirtualRegister, HardwareRegister]:
        """Lower assignment to an identifier (variable)."""
        symbol = expr.target.symbol

        # Check if aliased to hardware register
        hw_reg = self.ctx.current_function.alias_tracker.get_alias(symbol)
        if hw_reg:
            # Move to hardware register
            if not (isinstance(value, HardwareRegister) and value.name == hw_reg.name):
                self.emit(Move(dest=hw_reg, source=value, type_info=expr.expr_type))
            return hw_reg

        # Check if has explicit memory location
        if self.builder.has_explicit_location(symbol):
            mem_loc = self.builder.get_memory_location(symbol)
            self.emit(Store(source=value, dest=mem_loc, type_info=expr.expr_type))
            # Invalidate cached vreg — the memory location now holds a new
            # value, so any previously cached vreg is stale.  The next read
            # of this symbol will emit a fresh Load from memory.
            symbol_id = id(symbol)
            if symbol_id in self.ctx.symbol_to_vreg:
                del self.ctx.symbol_to_vreg[symbol_id]
            return value

        # Otherwise, update virtual register
        symbol_id = id(symbol)
        if symbol_id in self.ctx.symbol_to_vreg:
            vreg = self.ctx.symbol_to_vreg[symbol_id]
            if vreg != value:
                self.emit(Move(dest=vreg, source=value, type_info=expr.expr_type))
            return vreg
        else:
            # Allocate new virtual register
            vreg = self.ctx.alloc_vreg(expr.expr_type, symbol.name)
            self.ctx.symbol_to_vreg[symbol_id] = vreg
            if vreg != value:
                self.emit(Move(dest=vreg, source=value, type_info=expr.expr_type))
            return vreg

    # ========================================================================
    # Register Assignment
    # ========================================================================

    def _lower_register_assignment(self, expr: HIRAssignment, value) -> HardwareRegister:
        """Lower assignment to a hardware register."""
        hw_reg = HardwareRegister(expr.target.name)
        if not (isinstance(value, HardwareRegister) and value.name == hw_reg.name):
            self.emit(Move(dest=hw_reg, source=value, type_info=expr.expr_type))
        return hw_reg

    # ========================================================================
    # Field Assignment
    # ========================================================================

    def _lower_field_assignment(self, expr: HIRAssignment, value):
        """Lower assignment to a struct field."""
        field_access = expr.target
        field_offset = field_access.field_offset
        if field_offset is None:
            raise MIRLoweringError(f"Field offset not computed for field: {field_access.field_name}", source_loc=expr.source_loc)

        # Handle auto-dereference case (self.field = value where self is a pointer)
        if getattr(field_access, 'auto_deref', False):
            self._lower_pointer_field_assignment(field_access, value, field_offset, expr.expr_type)
            return value

        if isinstance(field_access.base, HIRIdentifier):
            struct_symbol = field_access.base.symbol
            # Check if this struct is decomposed into per-field vregs
            field_vregs = self.builder._decomposed_structs.get(id(struct_symbol))
            if field_vregs is not None:
                field_vreg = field_vregs.get(field_access.field_name)
                if field_vreg is None:
                    raise MIRLoweringError(
                        f"Unknown field '{field_access.field_name}' on decomposed struct",
                        source_loc=expr.source_loc
                    )
                self.emit(Move(dest=field_vreg, source=value, type_info=expr.expr_type))
                return value

            # Simple case: static_struct.field = value
            base_memloc = self.builder.get_memory_location(struct_symbol)
            field_memloc = self.builder._create_offset_memloc(base_memloc, field_offset, struct_symbol)
            self.emit(Store(source=value, dest=field_memloc, type_info=expr.expr_type))

        elif isinstance(field_access.base, HIRDereference):
            # Explicit dereference case: (*ptr).field = value
            self._lower_deref_field_assignment(field_access, value, field_offset, expr.expr_type)

        elif isinstance(field_access.base, HIRArrayIndex):
            # Array case: array[index].field = value
            self._lower_array_field_assignment(field_access, value, field_offset, expr.expr_type)

        else:
            raise MIRLoweringError(
                f"Field access only supports static structs, pointer dereference, "
                f"and array indexing, got: {type(field_access.base)}",
                source_loc=expr.source_loc
            )

        return value

    def _lower_pointer_field_assignment(self, field_access: HIRFieldAccess, value, field_offset: int, type_info):
        """Lower pointer-based field assignment (auto-deref): self.field = value where self is *Struct."""
        self.builder.emit_indirect_field_access(
            field_access.base, field_offset=field_offset, result_type=type_info,
            is_load=False, source=value)

    def _lower_deref_field_assignment(self, field_access: HIRFieldAccess, value, field_offset: int, type_info):
        """Lower (*ptr).field = value — the pointer lives inside an HIRDereference base."""
        self.builder.emit_indirect_field_access(
            field_access.base.pointer, field_offset=field_offset, result_type=type_info,
            is_load=False, source=value)

    def _lower_array_field_assignment(self, field_access: HIRFieldAccess, value, field_offset: int, type_info):
        """Lower array[index].field = value (store to array_base + index*struct_size + field_offset)."""
        self.builder.emit_static_array_field_access(
            field_access.base, field_offset=field_offset, result_type=type_info,
            is_load=False, source=value)


    # ========================================================================
    # Array Assignment
    # ========================================================================

    def _lower_array_assignment(self, expr: HIRAssignment, value):
        """Lower assignment to an array element or pointer index."""
        from r65.compiler.hir.types import PointerTypeInfo

        array_index = expr.target
        element_type = expr.expr_type
        element_size = self.builder._get_type_size(element_type)

        base_type = array_index.array.expr_type

        # Check if base is a pointer type (ptr[i] = x) vs array type (arr[i] = x).
        # The pointer path lowers array_index.array as a value expression, so it
        # already supports a pointer that is itself a struct field.
        if isinstance(base_type, PointerTypeInfo):
            return self._lower_pointer_index_assignment(expr, value, base_type)

        # Pointer-deref'd array base (`self.bytes[i] = v` via auto-deref,
        # or `(*p)[i] = v`). Folds the outer field offset into Y and stores
        # indirect through the pointer (see emit_pointer_deref_array_access).
        deref = self.builder.try_pointer_deref_array_base(array_index.array)
        if deref is not None:
            ptr_expr, base_field_offset = deref
            self.builder.emit_pointer_deref_array_access(
                ptr_expr=ptr_expr,
                index_expr=array_index.index,
                element_size=element_size,
                element_type=element_type,
                const_offset=base_field_offset,
                result_type=element_type,
                is_load=False, source=value,
            )
            return value

        # Resolve the array base — a bare static array or an array that is a
        # field of a statically-located struct (STRUCT.array_field[i] = x).
        base_memloc, reuse_base_key = self.builder.resolve_array_base_memloc(array_index.array)

        # Lower index expression for array indexing
        index_operand = self.builder.lower_expression(array_index.index)
        index_type = array_index.index.expr_type  # Type of the index (u8 or u16)

        # Calculate offset and create memory location
        if isinstance(index_operand, Immediate):
            return self._lower_constant_index_assignment(value, base_memloc, index_operand.value, element_size, element_type)
        else:
            reuse_key = self.builder.x_index_reuse_key(reuse_base_key, array_index.index)
            return self._lower_variable_index_assignment(value, base_memloc, index_operand, element_size, element_type, index_type, reuse_key)

    def _lower_pointer_index_assignment(self, expr: HIRAssignment, value, pointer_type):
        """Lower assignment through indexed pointer (ptr[i] = x)."""

        array_index = expr.target
        element_type = expr.expr_type
        # Use the index type (not element type) for the Move to Y register.
        # The index type determines the bit width: u16 indices must load as 16-bit
        # to avoid truncation when index >= 256.
        index_type = array_index.index.expr_type

        # Lower index expression
        index_operand = self.builder.lower_expression(array_index.index)

        # Scale index by element size for multi-byte elements
        # (ptr[i] byte offset = i * sizeof(element), matching _lower_pointer_index in expression.py)
        element_size = self.builder._get_type_size(element_type)
        if element_size > 1:
            if isinstance(index_operand, Immediate):
                index_operand = Immediate(index_operand.value * element_size)
            else:
                # Delegate to the read path's _compute_index_offset so writes
                # stay symmetric with reads: it widens a u8/i8 index to u16
                # before the shift. A raw `index << log2(size)` sized as the
                # u8 index overflows whenever index * element_size > 255
                # (e.g. ptr[150] of u16), corrupting the store address.
                index_operand = self.builder.expr_lowerer._compute_index_offset(
                    index_operand, element_size, index_type
                )

        # Lower the pointer expression to get the pointer value
        ptr_operand = self.builder.lower_expression(array_index.array)

        # Determine index register - must be Y for [dp],Y addressing
        index_register = None
        if isinstance(index_operand, HardwareRegister):
            if index_operand.name == 'Y':
                index_register = 'Y'
            elif index_operand.name == 'X':
                # Move X to Y for indirect addressing
                y_reg = HardwareRegister('Y')
                self.emit(Move(dest=y_reg, source=index_operand, type_info=index_type))
                index_register = 'Y'
            else:
                raise MIRLoweringError(f"Pointer indexing requires X or Y register, got: {index_operand.name}", source_loc=expr.source_loc)
        else:
            # Move index value to Y register
            y_reg = HardwareRegister('Y')
            self.emit(Move(dest=y_reg, source=index_operand, type_info=index_type))
            index_register = 'Y'

        # Emit StoreIndirect with index register
        self.emit(StoreIndirect(
            source=value,
            pointer=ptr_operand,
            is_far=pointer_type.is_far,
            type_info=element_type,
            index_register=index_register
        ))
        return value

    def _lower_constant_index_assignment(self, value, base_memloc, index_value, element_size, element_type):
        """Lower constant array index assignment with compile-time offset."""
        offset = index_value * element_size

        # Create offset memory location
        elem_memloc = self.builder._create_offset_memloc(base_memloc, offset, base_memloc.symbol)

        # Emit store to the element location
        self.emit(Store(source=value, dest=elem_memloc, type_info=element_type))
        return value

    def _lower_variable_index_assignment(self, value, base_memloc, index_operand, element_size, element_type, index_type=None, reuse_key=None):
        """Lower variable array index assignment with indexed addressing."""
        # Use the index type (not element type) for offset computation and X register load.
        # The index type determines the bit width: u16 indices must load as 16-bit
        # to avoid truncation when index >= 256.
        offset_type = index_type if index_type is not None else element_type

        # Create indexed memory location with X register. Preserve
        # base_memloc.offset so an array that is a struct field
        # (address=None, offset=field_offset) still resolves at codegen time.
        indexed_memloc = MemoryLocation(
            storage_type=base_memloc.storage_type,
            address=base_memloc.address,
            symbol=base_memloc.symbol,
            is_volatile=base_memloc.is_volatile,
            index_register='X',  # Mark as indexed with X
            offset=base_memloc.offset
        )

        # X holds index * element_size. Consecutive arr[i] accesses with the
        # same (array, index) can reuse it (skip the scale + Move) when the
        # reuse cache says X is still valid.
        if not self.builder.x_index_cache_hit(reuse_key):
            offset_operand = index_operand
            # If element size > 1, scale the index. Delegate to the read path's
            # _compute_index_offset so writes stay symmetric with reads: it
            # widens a u8/i8 index to u16 before the shift. A raw
            # `index << log2(size)` sized as the u8 index overflows whenever
            # index * element_size > 255 (e.g. arr[150] in a [u16; N]),
            # corrupting the store address.
            if element_size > 1:
                offset_operand = self.builder.expr_lowerer._compute_index_offset(
                    index_operand, element_size, offset_type
                )

            self.emit(Move(dest=HardwareRegister('X'), source=offset_operand, type_info=offset_type))
            self.builder.x_index_cache_set(reuse_key)

        # Emit store using indexed addressing (e.g., STA $20,X)
        self.emit(Store(source=value, dest=indexed_memloc, type_info=element_type))
        return value

    # ========================================================================
    # Dereference Assignment
    # ========================================================================

    def _lower_dereference_assignment(self, expr: HIRAssignment, value):
        """Lower assignment through pointer dereference."""
        from r65.compiler.hir.types import PointerTypeInfo

        deref = expr.target
        pointer_type = deref.pointer.expr_type

        if not isinstance(pointer_type, PointerTypeInfo):
            raise MIRLoweringError(f"Dereference of non-pointer type: {pointer_type}", source_loc=expr.source_loc)

        # Lower the pointer expression to get the pointer value
        ptr_operand = self.builder.lower_expression(deref.pointer)

        # Emit StoreIndirect
        self.emit(StoreIndirect(
            source=value,
            pointer=ptr_operand,
            is_far=pointer_type.is_far,
            type_info=expr.expr_type
        ))
        return value

    # ========================================================================
    # STATUS Flag Assignment
    # ========================================================================

    def _lower_status_flag_assignment(self, expr: HIRAssignment) -> Immediate:
        """
        Lower assignment to STATUS flag (e.g., STATUS.Carry = true).

        Emits StatusFlagSet instruction which generates:
        - SEC/CLC for Carry
        - SEI/CLI for Irq
        - SED/CLD for Decimal
        - SEP/REP for Index/Accumulator
        """
        target = expr.target
        value_expr = expr.value

        # Determine if setting or clearing the flag
        if isinstance(value_expr, HIRBooleanLiteral):
            set_flag = value_expr.value
        else:
            # Type checker should have ensured it's a boolean literal
            # but handle the case where it's not for robustness
            raise MIRLoweringError(
                f"STATUS flag assignment requires constant boolean value, "
                f"got {type(value_expr).__name__}",
                source_loc=expr.source_loc
            )

        self.emit(StatusFlagSet(
            flag_name=target.flag_name,
            value=set_flag
        ))

        # Return immediate value representing the assignment
        return Immediate(1 if set_flag else 0)

    # ========================================================================
    # Multi-Assignment (Tuple Destructuring)
    # ========================================================================

    def lower_multi_assignment(self, expr: HIRMultiAssignment) -> HardwareRegister:
        """
        Lower multi-assignment (tuple destructuring).

        Handles: (A, X) = func() where func returns a tuple.

        For tuple returns, values are placed in registers in order.
        For (u8, u8) tuples in m8 mode, uses A, B, X, Y order.
        Otherwise uses A, X, Y order.

        Args:
            expr: HIR multi-assignment expression

        Returns:
            HardwareRegister of the first element (A)
        """
        # Lower the value expression (function call returning tuple)
        # This will execute the call and leave results in registers
        self.builder.lower_expression(expr.value)

        # Determine return register order from callee's return type
        return_registers = self.builder._get_callee_return_registers(expr.value)

        # Get the tuple type from the expression
        value_type = expr.value.expr_type
        if not isinstance(value_type, MultiReturnTypeInfo):
            raise MIRLoweringError(f"Multi-assignment requires tuple type, got: {value_type}", source_loc=expr.source_loc)

        num_elements = len(value_type.element_types)
        if num_elements > len(return_registers):
            raise MIRLoweringError(f"Tuple has too many elements ({num_elements}), max supported is {len(return_registers)}", source_loc=expr.source_loc)

        # Assign each target from the corresponding return register
        # Note: We need to be careful about order if targets overlap with source registers
        # For (A, X) = func(), if func returns in A and X, and targets are A and X,
        # no moves are needed. But if targets are (X, A), we need to swap.

        # Collect assignments that need to be made
        assignments = []
        for i, target in enumerate(expr.targets):
            source_reg = HardwareRegister(return_registers[i])
            elem_type = value_type.element_types[i]

            if isinstance(target, HIRRegister):
                target_reg = HardwareRegister(target.name)
                if target_reg.name != source_reg.name:
                    assignments.append((target_reg, source_reg, elem_type))
            elif isinstance(target, HIRIdentifier):
                # Assign to variable
                symbol = target.symbol
                hw_reg = self.ctx.current_function.alias_tracker.get_alias(symbol)
                if hw_reg:
                    if hw_reg.name != source_reg.name:
                        assignments.append((hw_reg, source_reg, elem_type))
                else:
                    # Store to memory location
                    if self.builder.has_explicit_location(symbol):
                        mem_loc = self.builder.get_memory_location(symbol)
                        self.emit(Store(source=source_reg, dest=mem_loc, type_info=elem_type))
                        # Invalidate cached vreg (memory now has a new value)
                        s_id = id(symbol)
                        if s_id in self.ctx.symbol_to_vreg:
                            del self.ctx.symbol_to_vreg[s_id]
                    else:
                        # Update or create virtual register
                        symbol_id = id(symbol)
                        if symbol_id in self.ctx.symbol_to_vreg:
                            vreg = self.ctx.symbol_to_vreg[symbol_id]
                        else:
                            vreg = self.ctx.alloc_vreg(elem_type, symbol.name)
                            self.ctx.symbol_to_vreg[symbol_id] = vreg
                        self.emit(Move(dest=vreg, source=source_reg, type_info=elem_type))
            else:
                raise MIRLoweringError(f"Unsupported multi-assignment target: {type(target)}", source_loc=expr.source_loc)

        # Emit register-to-register moves with cycle detection
        # Handle cycles (e.g., (X, A) = func() where func returns in A, X) using temp register
        self._emit_moves_with_cycle_handling(assignments)

        # Return the first element's register
        return HardwareRegister(return_registers[0])

    def _emit_moves_with_cycle_handling(self, assignments: list):
        """
        Sequentialize a set of parallel register copies (dest <- source).

        Tuple-return destructuring produces copies that are semantically
        simultaneous, so emitting them in list order can clobber a value
        another copy still needs. This affects both cycles (X<->A) AND
        chains (X<-A then Y<-X reads the already-overwritten X).

        Standard parallel-copy sequentialization: repeatedly emit any copy
        whose destination is not still needed as a source — this drains
        chains and the tails of cycles. When only cycles remain, break one
        by parking a source in a temporary (a free register, or the stack
        when A/X/Y are all occupied) and repointing its reader at the temp.

        Args:
            assignments: List of (dest_reg, source_reg, elem_type) tuples.
                Sources are distinct (separate return registers) and
                destinations are distinct; identity copies are pre-filtered.
        """
        if not assignments:
            return

        # Mutable [dest, source, type]; source may be rewritten to a temp
        # register or to STACK (parked on the hardware stack) when a cycle
        # is broken.
        moves = [[d, s, t] for (d, s, t) in assignments]

        # A register is safe to use as a scratch temp only if no copy ever
        # writes it (so we cannot clobber an already-produced result) AND it
        # is not currently needed as a source. The first half is static; the
        # second is recomputed at each break (a register frees up once the
        # copies reading it have drained).
        result_regs = {m[0].name for m in moves}

        STACK = object()  # sentinel: value parked on the hardware stack

        def needed_as_source(name, exclude):
            return any(
                o[1] is not STACK and o[1].name == name
                for o in moves if o is not exclude
            )

        while moves:
            # Drain every copy that is safe now: its destination is not
            # needed as a source by any other remaining copy.
            progressed = True
            while progressed:
                progressed = False
                for mv in list(moves):
                    dest, src, ty = mv
                    if not needed_as_source(dest.name, mv):
                        if src is STACK:
                            self.emit(Pull(register=dest))
                        else:
                            self.emit(Move(dest=dest, source=src, type_info=ty))
                        moves.remove(mv)
                        progressed = True
            if not moves:
                return

            # Only cycles remain (every dest is still some copy's source).
            # Break one cycle with a temporary.
            rem_srcs = {m[1].name for m in moves if m[1] is not STACK}
            free_regs = [r for r in ('A', 'X', 'Y')
                         if r not in result_regs and r not in rem_srcs]
            if free_regs:
                # Park the picked copy's source in a free register and
                # repoint that copy (the unique reader of the source) at it.
                d0, s0, ty0 = moves[0]
                temp = HardwareRegister(free_regs[0])
                self.emit(Move(dest=temp, source=s0, type_info=ty0))
                moves[0][1] = temp
            else:
                # No free register => A, X and Y are all involved, so the
                # cycle necessarily contains an X<->Y edge. Park that
                # 16-bit source on the stack so push/pull widths match
                # (PHA/PLA would mismatch PHX/PLX on a u8/u16 cycle).
                break_mv = next(
                    (m for m in moves
                     if m[0].name in ('X', 'Y') and m[1].name in ('X', 'Y')),
                    None
                )
                if break_mv is None:
                    raise MIRLoweringError(
                        "Unsupported register permutation in tuple "
                        "destructuring: cyclic move with no free temporary "
                        "register",
                        source_loc=self.builder._current_source_loc
                    )
                self.emit(Push(register=break_mv[1]))
                break_mv[1] = STACK
