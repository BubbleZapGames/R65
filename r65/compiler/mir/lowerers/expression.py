"""
Expression lowerer: HIR expressions → MIR instructions.

Handles binary ops, unary ops, type casts, array indexing,
field access, dereference, and address-of operations.
"""

from typing import TYPE_CHECKING, Union

from r65.compiler.hir import (
    HIRBinaryOp, HIRUnaryOp, HIRTypeCast,
    HIRArrayIndex, HIRFieldAccess, HIRDereference, HIRAddressOf,
    HIRIdentifier,
)
from r65.compiler.mir.nodes import (
    VirtualRegister, HardwareRegister, Immediate, MemoryLocation,
    Move, Load, LoadIndirect, BinaryOp, UnaryOp, TypeConvert,
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

    def lower_binary_op(self, expr: HIRBinaryOp) -> VirtualRegister:
        """
        Lower binary operation.

        Args:
            expr: HIR binary operation

        Returns:
            VirtualRegister holding result
        """
        comparison_ops = {'==', '!=', '<', '<=', '>', '>='}

        if expr.op in comparison_ops:
            return self._lower_comparison_op(expr)
        else:
            return self._lower_arithmetic_op(expr)

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
        left = self.builder.lower_expression(expr.left)
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

        # Same size reinterpretation (zero-cost)
        if source_size == target_size:
            self.emit(Move(dest=result, source=source_operand, type_info=target_type))
            return result

        # Widening (8-bit → 16-bit)
        if source_size == 1 and target_size == 2:
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

        # Boolean conversions
        if str(target_type) == 'bool':
            return self._lower_to_bool(source_operand, source_type, target_type)

        if str(source_type) == 'bool':
            self.emit(Move(dest=result, source=source_operand, type_info=target_type))
            return result

        raise MIRLoweringError(f"Unsupported type cast: {source_type} to {target_type}")

    def _lower_to_bool(self, source_operand, source_type, target_type) -> VirtualRegister:
        """Convert value to boolean (0 = false, non-zero = true)."""
        if isinstance(source_operand, Immediate):
            temp = self.ctx.alloc_vreg(source_type, "bool_temp")
            self.emit(Move(dest=temp, source=source_operand, type_info=source_type))
            source_operand = temp

        return self.builder._emit_conditional_set(
            source_operand, true_when_nonzero=True,
            result_type=target_type, hint="bool_result"
        )

    # ========================================================================
    # Array Indexing
    # ========================================================================

    def lower_array_index(self, expr: HIRArrayIndex) -> VirtualRegister:
        """
        Lower array indexing.

        Computes: array[index] → Load from (base_address + index * element_size)

        Args:
            expr: HIR array index expression

        Returns:
            VirtualRegister holding array element value
        """
        element_type = expr.expr_type
        element_size = self.builder._get_type_size(element_type)

        index_operand = self.builder.lower_expression(expr.index)

        if not isinstance(expr.array, HIRIdentifier):
            raise MIRLoweringError(
                f"Array indexing only supports static arrays currently, got: {type(expr.array)}"
            )

        array_symbol = expr.array.symbol
        result = self.ctx.alloc_vreg(element_type, "array_elem")

        if isinstance(index_operand, Immediate):
            return self._lower_constant_index(
                result, array_symbol, index_operand.value, element_size, element_type
            )
        else:
            return self._lower_variable_index(
                result, array_symbol, index_operand, element_size, element_type
            )

    def _lower_constant_index(self, result, array_symbol, index_value, element_size, element_type):
        """Lower constant array index with compile-time offset."""
        offset = index_value * element_size
        base_memloc = self.builder.get_memory_location(array_symbol)
        elem_memloc = self.builder._create_offset_memloc(base_memloc, offset, array_symbol)

        self.emit(Load(dest=result, source=elem_memloc, type_info=element_type))
        return result

    def _lower_variable_index(self, result, array_symbol, index_operand, element_size, element_type):
        """Lower variable array index with indexed addressing."""
        offset_operand = index_operand

        # Multiply index by element_size if > 1
        if element_size > 1:
            offset_operand = self._compute_index_offset(index_operand, element_size, element_type)

        # Move offset to X register for indexed addressing
        x_reg = HardwareRegister('X')
        self.emit(Move(dest=x_reg, source=offset_operand, type_info=element_type))

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

        Args:
            expr: HIR field access expression

        Returns:
            VirtualRegister holding field value
        """
        field_offset = expr.field_offset
        if field_offset is None:
            raise MIRLoweringError(f"Field offset not computed for field: {expr.field_name}")

        if not isinstance(expr.base, HIRIdentifier):
            raise MIRLoweringError(
                f"Field access only supports static structs currently, got: {type(expr.base)}"
            )

        struct_symbol = expr.base.symbol
        result = self.ctx.alloc_vreg(expr.expr_type, f"field_{expr.field_name}")

        base_memloc = self.builder.get_memory_location(struct_symbol)
        field_memloc = self.builder._create_offset_memloc(base_memloc, field_offset, struct_symbol)

        self.emit(Load(dest=result, source=field_memloc, type_info=expr.expr_type))
        return result

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
        Lower address-of operator (&variable).

        For static variables, loads the address as an immediate value.

        Args:
            expr: HIR address-of expression

        Returns:
            VirtualRegister holding the address
        """
        if not isinstance(expr.operand, HIRIdentifier):
            raise MIRLoweringError(
                f"Address-of only supports static variables, got: {type(expr.operand)}"
            )

        symbol = expr.operand.symbol
        self.builder.get_memory_location(symbol)  # Validate symbol has location

        result = self.ctx.alloc_vreg(expr.expr_type, f"addr_of_{symbol.name}")

        # Create symbolic address immediate (resolved in codegen)
        addr_immediate = Immediate(0)
        addr_immediate.symbol = symbol

        self.emit(Move(dest=result, source=addr_immediate, type_info=expr.expr_type))

        return result
