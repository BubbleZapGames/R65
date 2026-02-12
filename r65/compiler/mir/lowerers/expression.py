"""
Expression lowerer: HIR expressions → MIR instructions.

Handles binary ops, unary ops, type casts, array indexing,
field access, dereference, and address-of operations.
"""

from typing import TYPE_CHECKING, Union, Optional

from r65.compiler.hir import (
    HIRBinaryOp, HIRUnaryOp, HIRTypeCast,
    HIRArrayIndex, HIRFieldAccess, HIRDereference, HIRAddressOf,
    HIRIdentifier,
)
from r65.compiler.hir.types import PointerTypeInfo
from r65.compiler.mir.nodes import (
    VirtualRegister, HardwareRegister, Immediate, MemoryLocation,
    Move, Load, LoadIndirect, BinaryOp, UnaryOp, TypeConvert, ToBool,
)
from r65.compiler.mir.lowerers.multiply import compute_array_field_offset
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
            # For <, <=, >, >=: use subtraction to set flags
            temp = self.ctx.alloc_vreg(expr.left.expr_type, "cmp_temp")
            self.emit(BinaryOp(
                dest=temp,
                left=left,
                right=right,
                op='-',
                type_info=expr.left.expr_type
            ))
            return self.builder._emit_conditional_set(
                temp, true_when_nonzero=True,
                result_type=expr.expr_type, hint=f"{expr.op}_result"
            )

    def _lower_arithmetic_op(self, expr: HIRBinaryOp) -> VirtualRegister:
        """Lower arithmetic/bitwise binary operation."""
        from r65.compiler.hir import HIRRegister
        from r65.compiler.mir.nodes import MemoryLocation

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
            VirtualRegister holding result
        """
        operand = self.builder.lower_expression(expr.operand)

        # Ensure operand is a register (not immediate)
        if isinstance(operand, Immediate):
            temp = self.ctx.alloc_vreg(expr.operand.expr_type, "unary_temp")
            self.emit(Move(dest=temp, source=operand, type_info=expr.operand.expr_type))
            operand = temp

        result = self.ctx.alloc_vreg(expr.expr_type, f"{expr.op}_result")

        self.emit(UnaryOp(
            dest=result,
            operand=operand,
            op=expr.op,
            type_info=expr.expr_type
        ))

        return result

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
        source_operand = self.builder.lower_expression(expr.expr)
        source_type = expr.expr.expr_type
        target_type = expr.target_type

        source_size = self.builder._get_type_size(source_type)
        target_size = self.builder._get_type_size(target_type)

        result = self.ctx.alloc_vreg(target_type, "cast_result")

        # Boolean conversions - check BEFORE size comparisons since bool is 1 byte
        if str(target_type) == 'bool' and str(source_type) != 'bool':
            return self._lower_to_bool(source_operand, source_type, target_type)

        if str(source_type) == 'bool' and str(target_type) != 'bool':
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

        raise MIRLoweringError(f"Unsupported type cast: {source_type} to {target_type}")

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

        if not isinstance(expr.array, HIRIdentifier):
            raise MIRLoweringError(
                f"Array indexing only supports identifiers, got: {type(expr.array)}"
            )

        # Check if this is pointer indexing (ptr[index]) vs array indexing (array[index])
        array_type = expr.array.expr_type
        if isinstance(array_type, PointerTypeInfo):
            # Pointer indexing: load through pointer with indirect addressing
            return self._lower_pointer_index(expr, index_operand, element_size, element_type, array_type)

        # Regular array indexing
        array_symbol = expr.array.symbol
        result = self.ctx.alloc_vreg(element_type, "array_elem")

        if isinstance(index_operand, Immediate):
            return self._lower_constant_index(
                result, array_symbol, index_operand.value, element_size, element_type
            )
        else:
            return self._lower_variable_index(
                result, array_symbol, index_operand, element_size, element_type, index_type
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

        # For indexed addressing, we need the index in Y register
        # If element_size > 1, multiply first
        if element_size > 1:
            index_operand = self._compute_index_offset(index_operand, element_size, element_type)

        # Move index to Y register for indirect indexed addressing
        y_reg = HardwareRegister('Y')
        self.emit(Move(dest=y_reg, source=index_operand, type_info=element_type))

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

    def _lower_constant_index(self, result, array_symbol, index_value, element_size, element_type):
        """Lower constant array index with compile-time offset."""
        offset = index_value * element_size
        base_memloc = self.builder.get_memory_location(array_symbol)
        elem_memloc = self.builder._create_offset_memloc(base_memloc, offset, array_symbol)

        self.emit(Load(dest=result, source=elem_memloc, type_info=element_type))
        return result

    def _lower_variable_index(self, result, array_symbol, index_operand, element_size, element_type, index_type=None):
        """Lower variable array index with indexed addressing."""
        # Use the index type (not element type) for offset computation and X register load.
        # The index type determines the bit width: u16 indices must load as 16-bit
        # to avoid truncation when index >= 256.
        offset_type = index_type if index_type is not None else element_type
        offset_operand = index_operand

        # Multiply index by element_size if > 1
        if element_size > 1:
            offset_operand = self._compute_index_offset(index_operand, element_size, offset_type)

        # Move offset to X register for indexed addressing
        x_reg = HardwareRegister('X')
        self.emit(Move(dest=x_reg, source=offset_operand, type_info=offset_type))

        # Create indexed memory location
        base_memloc = self.builder.get_memory_location(array_symbol)
        indexed_memloc = MemoryLocation(
            storage_type=base_memloc.storage_type,
            address=base_memloc.address,
            symbol=array_symbol,
            is_volatile=base_memloc.is_volatile,
            index_register='X'
        )

        self.emit(Load(dest=result, source=indexed_memloc, type_info=element_type))
        return result

    def _compute_index_offset(self, index_operand, element_size, element_type):
        """Compute byte offset from array index."""
        offset_vreg = self.ctx.alloc_vreg(element_type, "array_offset")

        # Use shift for power-of-2 sizes
        if element_size & (element_size - 1) == 0:
            shift_amount = 0
            temp = element_size
            while temp > 1:
                shift_amount += 1
                temp >>= 1

            self.emit(BinaryOp(
                dest=offset_vreg,
                left=index_operand,
                right=Immediate(shift_amount),
                op='<<',
                type_info=element_type
            ))
        else:
            # Non-power-of-2: use multiplication
            self.emit(BinaryOp(
                dest=offset_vreg,
                left=index_operand,
                right=Immediate(element_size),
                op='*',
                type_info=element_type
            ))

        return offset_vreg

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
            raise MIRLoweringError(f"Field offset not computed for field: {expr.field_name}")

        result = self.ctx.alloc_vreg(expr.expr_type, f"field_{expr.field_name}")

        # Handle auto-dereference case (self.field where self is a pointer)
        if getattr(expr, 'auto_deref', False):
            self._lower_pointer_field_access(expr, result, field_offset)
            return result

        if isinstance(expr.base, HIRIdentifier):
            # Simple case: static_struct.field
            struct_symbol = expr.base.symbol
            base_memloc = self.builder.get_memory_location(struct_symbol)
            field_memloc = self.builder._create_offset_memloc(base_memloc, field_offset, struct_symbol)
            self.emit(Load(dest=result, source=field_memloc, type_info=expr.expr_type))

        elif isinstance(expr.base, HIRArrayIndex):
            # Array case: array[index].field
            self._lower_array_field_access(expr, result, field_offset)

        else:
            raise MIRLoweringError(
                f"Field access only supports static structs and array indexing, got: {type(expr.base)}"
            )

        return result

    def _lower_pointer_field_access(self, expr: HIRFieldAccess, result: VirtualRegister, field_offset: int):
        """
        Lower pointer-based field access (auto-dereference).

        For self.field where self is *StructName:
        - Load the pointer value
        - Use indirect addressing with offset to access the field

        Uses LoadIndirect with field offset for efficient code generation.
        """
        from r65.compiler.mir.nodes import LoadIndirect
        from r65.compiler.hir.types import PointerTypeInfo

        # Lower the base expression (the pointer)
        ptr_vreg = self.builder.lower_expression(expr.base)

        # Get pointer type info to determine if it's a far pointer
        base_type = expr.base.expr_type
        is_far = isinstance(base_type, PointerTypeInfo) and base_type.is_far

        # Emit LoadIndirect with field offset
        self.emit(LoadIndirect(
            dest=result,
            pointer=ptr_vreg,
            is_far=is_far,
            type_info=expr.expr_type,
            offset=field_offset
        ))

    def _lower_array_field_access(self, expr: HIRFieldAccess, result: VirtualRegister, field_offset: int):
        """
        Lower array[index].field access.

        Computes: array_base + (index * struct_size) + field_offset
        """
        array_index_expr = expr.base  # HIRArrayIndex

        if not isinstance(array_index_expr.array, HIRIdentifier):
            raise MIRLoweringError(
                f"Array field access requires static array, got: {type(array_index_expr.array)}"
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
            self.emit(Load(dest=result, source=field_memloc, type_info=expr.expr_type))
        else:
            # Variable index: compute offset at runtime
            # offset = (index * struct_size) + field_offset
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

            self.emit(Load(dest=result, source=indexed_memloc, type_info=expr.expr_type))

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
            raise MIRLoweringError(f"Dereference of non-pointer type: {pointer_type}")

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

        if not isinstance(expr.operand, HIRIdentifier):
            raise MIRLoweringError(
                f"Address-of only supports static variables or array indexing, got: {type(expr.operand)}"
            )

        symbol = expr.operand.symbol
        self.builder.get_memory_location(symbol)  # Validate symbol has location

        result = self.ctx.alloc_vreg(expr.expr_type, f"addr_of_{symbol.name}")

        # Create symbolic address immediate (resolved in codegen)
        addr_immediate = Immediate(0)
        addr_immediate.symbol = symbol

        self.emit(Move(dest=result, source=addr_immediate, type_info=expr.expr_type))

        # Propagate symbol to vreg for type conversion to access (e.g., near-to-far ptr)
        result.symbol = symbol

        return result

    def _lower_addressof_array_index(self, expr: HIRAddressOf) -> VirtualRegister:
        """
        Lower address-of array index: &array[index]

        Computes: base_address + index * element_size

        Args:
            expr: HIR address-of expression with HIRArrayIndex operand

        Returns:
            VirtualRegister holding pointer to the array element
        """
        from r65.compiler.hir.types import ArrayTypeInfo, BasicTypeInfo

        array_index = expr.operand
        if not isinstance(array_index.array, HIRIdentifier):
            raise MIRLoweringError(
                f"Address-of array index requires static array, got: {type(array_index.array)}"
            )

        array_symbol = array_index.array.symbol
        self.builder.get_memory_location(array_symbol)  # Validate symbol has location

        # Get element size
        array_type = array_index.array.expr_type
        if isinstance(array_type, ArrayTypeInfo):
            element_size = self.builder._get_type_size(array_type.element_type)
        else:
            element_size = 1

        result = self.ctx.alloc_vreg(expr.expr_type, f"addr_of_{array_symbol.name}_elem")

        # Lower the index
        index_operand = self.builder.lower_expression(array_index.index)

        if isinstance(index_operand, Immediate):
            # Constant index: compute offset at compile time
            offset = index_operand.value * element_size

            # Create symbolic address with offset (resolved in codegen)
            addr_immediate = Immediate(offset)
            addr_immediate.symbol = array_symbol
            addr_immediate.symbol_offset = offset

            self.emit(Move(dest=result, source=addr_immediate, type_info=expr.expr_type))
        else:
            # Variable index: compute base + index * element_size at runtime
            # First, load base address
            base_addr = self.ctx.alloc_vreg(expr.expr_type, f"base_addr_{array_symbol.name}")
            addr_immediate = Immediate(0)
            addr_immediate.symbol = array_symbol

            self.emit(Move(dest=base_addr, source=addr_immediate, type_info=expr.expr_type))

            # If element_size > 1, multiply index by element_size
            if element_size > 1:
                offset_type = BasicTypeInfo('u16')
                offset_vreg = self._compute_index_offset(index_operand, element_size, offset_type)
            else:
                offset_vreg = index_operand

            # Add base + offset
            self.emit(BinaryOp(
                dest=result,
                op='+',
                left=base_addr,
                right=offset_vreg,
                type_info=expr.expr_type
            ))

        return result
