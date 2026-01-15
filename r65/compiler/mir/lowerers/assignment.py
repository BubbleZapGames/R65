"""
Assignment lowerer: HIR assignments → MIR instructions.

Handles variable assignments, register assignments, field access assignments,
array index assignments, and pointer dereference assignments.
"""

from typing import TYPE_CHECKING, Union

from r65.compiler.hir import (
    HIRAssignment, HIRMultiAssignment, HIRBinaryOp, HIRRegister, HIRIdentifier,
    HIRFieldAccess, HIRArrayIndex, HIRDereference, HIRStatusFlagAccess, HIRBooleanLiteral,
)
from r65.compiler.hir.types import TupleTypeInfo
from r65.compiler.mir.nodes import (
    VirtualRegister, HardwareRegister, Immediate, MemoryLocation,
    Move, Store, StoreIndirect, BinaryOp, StatusFlagSet,
)
from r65.compiler.mir.lowerers.multiply import compute_array_field_offset
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
        # OPTIMIZATION: Detect pattern `target = target op value` for hardware registers
        # Generate BinaryOp(dest=target, left=target, op, right=value) directly
        # instead of temp = target op value; target = temp
        if isinstance(expr.value, HIRBinaryOp) and isinstance(expr.target, HIRRegister):
            binary_op = expr.value
            # Check if it's target = target op value
            if (isinstance(binary_op.left, HIRRegister) and
                binary_op.left.name == expr.target.name):
                # Direct hardware register op: A = A + TEMP becomes BinaryOp(dest=A, left=A, right=memloc)
                hw_reg = HardwareRegister(expr.target.name)

                # CRITICAL: For A = A op MEMORY, we must NOT emit a Load instruction
                # that would clobber A. Check if right operand is a memory location
                # and use it directly instead of going through lower_expression.
                right = None
                if isinstance(binary_op.right, HIRIdentifier):
                    symbol = binary_op.right.symbol
                    # Check if it has explicit memory location (not an alias)
                    alias = self.ctx.current_function.alias_tracker.get_alias(symbol)
                    if alias is None and self.builder.has_explicit_location(symbol):
                        # Use memory location directly - no Load instruction needed
                        right = self.builder.get_memory_location(symbol)

                if right is None:
                    # Fall back to lower_expression for other cases (immediates, etc.)
                    right = self.builder.lower_expression(binary_op.right)

                self.emit(BinaryOp(
                    dest=hw_reg,
                    left=hw_reg,
                    op=binary_op.op,
                    right=right,
                    type_info=expr.expr_type
                ))
                return hw_reg

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
            raise MIRLoweringError(f"Unsupported assignment target: {type(expr.target)}")

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
            raise MIRLoweringError(f"Field offset not computed for field: {field_access.field_name}")

        if isinstance(field_access.base, HIRIdentifier):
            # Simple case: static_struct.field = value
            struct_symbol = field_access.base.symbol
            base_memloc = self.builder.get_memory_location(struct_symbol)
            field_memloc = self.builder._create_offset_memloc(base_memloc, field_offset, struct_symbol)
            self.emit(Store(source=value, dest=field_memloc, type_info=expr.expr_type))

        elif isinstance(field_access.base, HIRArrayIndex):
            # Array case: array[index].field = value
            self._lower_array_field_assignment(field_access, value, field_offset, expr.expr_type)

        else:
            raise MIRLoweringError(
                f"Field access only supports static structs and array indexing, got: {type(field_access.base)}"
            )

        return value

    def _lower_array_field_assignment(self, field_access: HIRFieldAccess, value, field_offset: int, type_info):
        """
        Lower array[index].field = value assignment.

        Computes: store to (array_base + index * struct_size + field_offset)
        """
        array_index_expr = field_access.base  # HIRArrayIndex

        if not isinstance(array_index_expr.array, HIRIdentifier):
            raise MIRLoweringError(
                f"Array field assignment requires static array, got: {type(array_index_expr.array)}"
            )

        array_symbol = array_index_expr.array.symbol
        struct_type = array_index_expr.expr_type  # The struct type (element type of array)
        struct_size = self.builder._get_type_size(struct_type)

        # Lower the index expression
        index_operand = self.builder.lower_expression(array_index_expr.index)

        if isinstance(index_operand, Immediate):
            # Constant index: compute offset at compile time
            total_offset = (index_operand.value * struct_size) + field_offset
            base_memloc = self.builder.get_memory_location(array_symbol)
            field_memloc = self.builder._create_offset_memloc(base_memloc, total_offset, array_symbol)
            self.emit(Store(source=value, dest=field_memloc, type_info=type_info))
        else:
            # Variable index: compute offset at runtime
            offset_operand = self._compute_array_field_offset(
                index_operand, struct_size, field_offset, struct_type
            )

            # Move offset to X register for indexed addressing
            x_reg = HardwareRegister('X')
            self.emit(Move(dest=x_reg, source=offset_operand, type_info=struct_type))

            # Create indexed memory location
            base_memloc = self.builder.get_memory_location(array_symbol)
            indexed_memloc = MemoryLocation(
                storage_type=base_memloc.storage_type,
                address=base_memloc.address,
                symbol=array_symbol,
                is_volatile=base_memloc.is_volatile,
                index_register='X'
            )

            self.emit(Store(source=value, dest=indexed_memloc, type_info=type_info))

    def _compute_array_field_offset(self, index_operand, struct_size: int, field_offset: int, type_info):
        """
        Compute byte offset for array[index].field access.

        Delegates to shared multiply module. See docs/struct-array-indexing.md.
        """
        return compute_array_field_offset(
            index_operand, struct_size, field_offset, type_info,
            self.ctx, self.emit
        )

    # ========================================================================
    # Array Assignment
    # ========================================================================

    def _lower_array_assignment(self, expr: HIRAssignment, value):
        """Lower assignment to an array element."""
        array_index = expr.target
        element_type = expr.expr_type
        element_size = self.builder._get_type_size(element_type)

        # Lower index expression
        index_operand = self.builder.lower_expression(array_index.index)

        # Get the array symbol
        if not isinstance(array_index.array, HIRIdentifier):
            raise MIRLoweringError(f"Array indexing only supports static arrays currently, got: {type(array_index.array)}")

        array_symbol = array_index.array.symbol

        # Calculate offset and create memory location
        if isinstance(index_operand, Immediate):
            return self._lower_constant_index_assignment(value, array_symbol, index_operand.value, element_size, element_type)
        else:
            return self._lower_variable_index_assignment(value, array_symbol, index_operand, element_size, element_type)

    def _lower_constant_index_assignment(self, value, array_symbol, index_value, element_size, element_type):
        """Lower constant array index assignment with compile-time offset."""
        offset = index_value * element_size
        base_memloc = self.builder.get_memory_location(array_symbol)

        # Create offset memory location
        elem_memloc = self.builder._create_offset_memloc(base_memloc, offset, array_symbol)

        # Emit store to the element location
        self.emit(Store(source=value, dest=elem_memloc, type_info=element_type))
        return value

    def _lower_variable_index_assignment(self, value, array_symbol, index_operand, element_size, element_type):
        """Lower variable array index assignment with indexed addressing."""
        offset_operand = index_operand

        # If element size > 1, multiply index by element_size
        if element_size > 1:
            offset_vreg = self.ctx.alloc_vreg(element_type, "array_offset")
            # Check if element_size is power of 2 - use shift instead of multiply
            if element_size & (element_size - 1) == 0:  # Is power of 2
                # Calculate shift amount: log2(element_size)
                shift_amount = 0
                temp = element_size
                while temp > 1:
                    shift_amount += 1
                    temp >>= 1
                shift_immediate = Immediate(shift_amount)
                # offset = index << shift_amount
                self.emit(BinaryOp(
                    dest=offset_vreg,
                    left=index_operand,
                    right=shift_immediate,
                    op='<<',
                    type_info=element_type
                ))
            else:
                # Non-power-of-2: use multiplication
                size_immediate = Immediate(element_size)
                self.emit(BinaryOp(
                    dest=offset_vreg,
                    left=index_operand,
                    right=size_immediate,
                    op='*',
                    type_info=element_type
                ))
            offset_operand = offset_vreg

        # Move offset to X register for indexed addressing
        x_reg = HardwareRegister('X')
        self.emit(Move(dest=x_reg, source=offset_operand, type_info=element_type))

        # Create indexed memory location with X register
        base_memloc = self.builder.get_memory_location(array_symbol)
        indexed_memloc = MemoryLocation(
            storage_type=base_memloc.storage_type,
            address=base_memloc.address,
            symbol=array_symbol,
            is_volatile=base_memloc.is_volatile,
            index_register='X'  # Mark as indexed with X
        )

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
            raise MIRLoweringError(f"Dereference of non-pointer type: {pointer_type}")

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
                f"got {type(value_expr).__name__}"
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

        For tuple returns, values are placed in registers in order:
        - First element -> A register
        - Second element -> X register
        - Third element -> Y register (if applicable)

        Args:
            expr: HIR multi-assignment expression

        Returns:
            HardwareRegister of the first element (A)
        """
        # Lower the value expression (function call returning tuple)
        # This will execute the call and leave results in A, X, (Y)
        self.builder.lower_expression(expr.value)

        # Map tuple index to return register
        return_registers = ['A', 'X', 'Y']

        # Get the tuple type from the expression
        value_type = expr.value.expr_type
        if not isinstance(value_type, TupleTypeInfo):
            raise MIRLoweringError(f"Multi-assignment requires tuple type, got: {value_type}")

        num_elements = len(value_type.element_types)
        if num_elements > len(return_registers):
            raise MIRLoweringError(f"Tuple has too many elements ({num_elements}), max supported is {len(return_registers)}")

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
                raise MIRLoweringError(f"Unsupported multi-assignment target: {type(target)}")

        # Emit register-to-register moves
        # TODO: Handle cycles in assignments (e.g., (X, A) = func() where func returns in A, X)
        # For now, emit moves in order - may need temp register for swaps
        for target_reg, source_reg, elem_type in assignments:
            self.emit(Move(dest=target_reg, source=source_reg, type_info=elem_type))

        # Return the first element's register
        return HardwareRegister(return_registers[0])
