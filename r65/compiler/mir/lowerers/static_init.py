# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Static init lowerer: generates __init_start() for static variable initialization.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, List

from r65.compiler.hir import (
    HIRStaticDecl, HIRFunctionDecl, HIRStructDecl,
    HIRExpression, HIRIntegerLiteral, HIRBooleanLiteral, HIREnumVariantExpr, HIRTypeCast,
    HIRArrayFillExpr, HIRArrayLiteralExpr, HIRStringLiteral, HIRStructLiteralExpr,
    HIRIdentifier, HIRFunctionAddress, HIRAddressOf,
    SymbolKind,
)
from r65.compiler.hir.attributes import StorageKind

from r65.compiler.mir.nodes import (
    MIRFunction, BasicBlock,
    VirtualRegister, FunctionPointer, MemoryLocation,
    Store, Return,
    MemoryFill, BlockCopy, ROMDataRef, SymbolByte,
)

from r65.compiler.mir.virtual_registers import VirtualRegisterAllocator
from r65.compiler.mir.register_tracker import RegisterAliasTracker
from r65.compiler.mir.cfg import CFGBuilder
from r65.compiler.typeck.processor_mode import ProcessorMode
from r65.compiler.errors import MIRLoweringError
from r65.compiler.typeck.type_utils import TypeUtils

if TYPE_CHECKING:
    from r65.compiler.mir.builder import MIRBuilder
    from r65.compiler.mir.context import LoweringContext


@dataclass(eq=False)
class RuntimeSlot:
    """A run-time element of a local aggregate literal (`[x, 1, 2]`).

    Holds every byte position of its slot in the initializer's byte list, so
    the slot's offset is where it first appears; the constant bytes around it
    are laid out as usual and the builder stores *expr* there afterwards.
    One HIR expression shared by several slots (`[x; 4]` expands to the same
    node four times) is evaluated once.
    """
    expr: HIRExpression
    size: int
    what: str  # the element, for errors ("element of local array `a`")


class StaticInitLowerer:
    """
    Generates __init_start() function for static variable initialization.

    Handles array fills, array literals, string literals, struct literals,
    function pointers, and scalar initializers.
    """

    def __init__(self, builder: 'MIRBuilder'):
        self.builder = builder

    @property
    def ctx(self) -> 'LoweringContext':
        return self.builder.ctx

    def emit(self, instr):
        """Emit an instruction to the current block."""
        self.builder.emit(instr)

    def generate_init_function(self, statics: List[HIRStaticDecl]) -> MIRFunction:
        """
        Generate __init_start() function for static initialization.

        This function initializes all static variables with initializers.
        It should be called at the beginning of the program's entry point.

        Initialization strategies:
        - Array fill [value; count]: Loop fill (efficient for zero fills)
        - Array literal [a, b, c]: Block copy from ROM (MVN instruction)
        - Scalar values: Simple store

        Args:
            statics: List of HIRStaticDecl with initializers

        Returns:
            MIRFunction for __init_start()
        """
        # Create MIR function structure for __init_start()
        mir_func = MIRFunction(
            name="__init_start",
            parameters=[],
            return_type=None,  # void return
            blocks={},
            entry_block_id=0,
            exit_block_ids=[],
            mode_attr=None,  # No specific mode requirement
            preserves_attr=None,
            bank_attr=None,
            interrupt_attr=None,
            is_entry=False,
            is_far=False,
            vreg_allocator=VirtualRegisterAllocator(),
            alias_tracker=RegisterAliasTracker()
        )

        # Set current function context
        self.builder.current_function = mir_func
        self.builder.cfg_builder = CFGBuilder(mir_func)
        self.builder.symbol_to_vreg.clear()
        self.builder.loop_stack.clear()
        self.builder.current_mode = ProcessorMode.default()

        # Create entry block
        entry_block = self.builder.cfg_builder.new_block()
        mir_func.entry_block_id = entry_block.block_id
        self.builder.current_block = entry_block

        # Generate initialization code for each static variable
        for static_decl in statics:
            # Check if this is a ROM static - ROM data should be accessed directly,
            # not copied to RAM. We still create the ROM data section but skip BlockCopy.
            # ROM storage is indicated by storage_attr being None (immutable static)
            is_rom_storage = static_decl.storage_attr is None

            # Get memory location for the static
            mem_loc = self.builder.get_memory_location(static_decl.symbol)
            initializer = static_decl.initializer

            # Handle different initializer types
            if isinstance(initializer, HIRArrayFillExpr):
                # Array fill: a MemoryFill loop for a constant scalar into
                # RAM; a ROM table otherwise (see _emit_array_fill_init)
                self._emit_array_fill_init(static_decl, mem_loc, initializer,
                                           is_rom=is_rom_storage)

            elif isinstance(initializer, HIRArrayLiteralExpr):
                # Array literal: create ROM data, copy to RAM only if not ROM
                self._emit_array_literal_init(static_decl, mem_loc, initializer, skip_copy=is_rom_storage)

            elif isinstance(initializer, HIRStringLiteral):
                # String literal: create ROM data, copy to RAM only if not ROM
                self._emit_string_literal_init(static_decl, mem_loc, initializer, skip_copy=is_rom_storage)

            elif isinstance(initializer, HIRStructLiteralExpr):
                # Struct literal: create ROM data, copy to RAM only if not ROM
                self._emit_struct_literal_init(static_decl, mem_loc, initializer, skip_copy=is_rom_storage)

            elif is_rom_storage and not TypeUtils.is_aggregate_type(static_decl.var_type):
                # A scalar in ROM: its bytes become a data label, read in
                # place.
                self._emit_rom_scalar_init(static_decl, initializer)

            elif is_rom_storage:
                # Any other ROM initializer (`include_bytes!`) is laid out
                # by its own path, not stored at runtime.
                pass

            elif self._is_function_pointer_init(initializer):
                # Function pointer: emit Store with FunctionPointer directly
                func_name = self._get_function_name(initializer)
                func_ptr = FunctionPointer(function_name=func_name)
                self.emit(Store(
                    source=func_ptr,
                    dest=mem_loc,
                    type_info=static_decl.var_type
                ))

            else:
                # Scalar value: simple store
                init_value = self.builder.lower_expression(initializer)
                self.emit(Store(
                    source=init_value,
                    dest=mem_loc,
                    type_info=static_decl.var_type
                ))

        # Emit return instruction
        self.emit(Return(values=[]))

        # Find exit blocks
        mir_func.exit_block_ids = self.builder.cfg_builder.find_exit_blocks()

        return mir_func

    def _emit_rom_scalar_init(self, static_decl: HIRStaticDecl,
                              initializer: HIRExpression):
        """Lay out an immutable scalar static as ROM data under its label."""
        name = static_decl.name
        size = self.builder._get_type_size(static_decl.var_type)
        if self._is_function_pointer_init(initializer):
            func_name = self._get_function_name(initializer)
            data = [SymbolByte(func_name, part)
                    for part in ('low', 'high', 'bank')[:size]]
        else:
            data = self._extract_initializer_bytes(
                initializer, size, f"initializer of static `{name}`")
        self._emit_rom_init(static_decl, None, data, size, skip_copy=True)

    def _is_function_pointer_init(self, initializer: HIRExpression) -> bool:
        """Check if initializer is a function pointer (identifier or address-of function)."""
        # Direct function reference: handler
        if isinstance(initializer, HIRIdentifier):
            if initializer.symbol and initializer.symbol.kind == SymbolKind.FUNCTION:
                return True
        # Explicit function address: &handler (HIRFunctionAddress)
        if isinstance(initializer, HIRFunctionAddress):
            return True
        return False

    def _get_function_name(self, initializer: HIRExpression) -> str:
        """Extract function name from function pointer initializer."""
        if isinstance(initializer, HIRIdentifier):
            return initializer.name
        if isinstance(initializer, HIRFunctionAddress):
            return initializer.function_name
        raise MIRLoweringError(f"Cannot extract function name from {type(initializer).__name__}", source_loc=self.builder._current_source_loc)

    def _emit_array_fill_init(
        self,
        static_decl: HIRStaticDecl,
        mem_loc: MemoryLocation,
        fill_expr: HIRArrayFillExpr,
        is_rom: bool = False
    ):
        """
        Emit MemoryFill instruction for array fill expression.

        Example: [0; 256] fills 256 elements with 0 using a loop.

        A fill is a loop only for a constant scalar into RAM. A ROM static has
        no loop to run - it needs its bytes in a table, or a read of it lands
        wherever the RAM allocator would have put it - and an aggregate value
        (`[[7; 2]; 3]`) is no single byte to store, so both become the same
        table an array literal would, expanded element by element.
        """
        from r65.compiler.hir.types import ArrayTypeInfo

        # Get element type and size
        array_type = static_decl.var_type
        if isinstance(array_type, ArrayTypeInfo):
            element_size = self.builder._get_type_size(array_type.element_type)
        else:
            element_size = 1  # Default to 1 byte

        # Get fill value - must be constant for efficient code gen
        # Extract constant value without emitting instructions
        fill_value = self._extract_constant_value(fill_expr.fill_value)
        if fill_value is None or is_rom:
            expanded = HIRArrayLiteralExpr(
                elements=[fill_expr.fill_value] * fill_expr.count,
                source_loc=fill_expr.source_loc,
            )
            expanded.expr_type = getattr(fill_expr, 'expr_type', None)
            self._emit_array_literal_init(static_decl, mem_loc, expanded, skip_copy=is_rom)
            return

        self.emit(MemoryFill(
            dest=mem_loc,
            fill_value=fill_value,
            count=fill_expr.count,
            element_size=element_size
        ))

    def _extract_symbol_address_bytes(
        self, expr: HIRExpression, size: int, allow_runtime: bool = False
    ) -> Optional[List[SymbolByte]]:
        """
        Lower `&SOME_STATIC` (with any surrounding casts) into link-time
        address bytes, for use inside a static initializer.

        Returns None if `expr` is not an address-of a ROM static, so callers
        can fall through to their own error reporting. Only immutable statics
        are supported: their data label is known to the assembler, whereas a
        RAM static's address is assigned after MIR lowering — so with
        *allow_runtime* a mutable static's address is left to run-time code.
        """
        # Peel casts: `&FOO as far *u8` is the idiomatic spelling.
        inner = expr
        while isinstance(inner, HIRTypeCast):
            inner = inner.expr

        if not isinstance(inner, HIRAddressOf) or inner.operand is None:
            return None
        target = inner.operand
        if not isinstance(target, HIRIdentifier):
            return None

        symbol = target.symbol
        if symbol is None:
            symbol = self.builder._hir_program.symbol_table.lookup(target.name)
        if symbol is None or symbol.kind != SymbolKind.STATIC_VAR:
            return None

        # Mutable statics live in RAM; their address is not a link-time label.
        if symbol.is_mutable:
            if allow_runtime:
                return None
            raise MIRLoweringError(
                f"cannot take the address of mutable static `{target.name}` in a "
                f"static initializer (only immutable statics have a link-time address)",
                source_loc=self.builder._current_source_loc
            )

        label = getattr(symbol, 'rom_label', None) or f"__{target.name}_data"

        parts = ['low', 'high', 'bank']
        if size < 2 or size > 3:
            raise MIRLoweringError(
                f"cannot store the address of `{target.name}` in a {size}-byte field; "
                f"use a near pointer (2 bytes) or far pointer (3 bytes)",
                source_loc=self.builder._current_source_loc
            )
        return [SymbolByte(label, part) for part in parts[:size]]

    def _extract_initializer_bytes(
        self, expr: HIRExpression, size: int, what: str,
        allow_runtime: bool = False
    ) -> List:
        """
        Lower one static-initializer element to `size` little-endian bytes.

        Accepts compile-time integers, `&IMMUTABLE_STATIC` address-of
        expressions, and the aggregates built from them at any depth: a struct
        literal, an array literal or fill, a string literal for a byte array.
        Anything else is a hard error — silently substituting zero here
        produces a ROM that links cleanly and misbehaves at runtime — unless
        *allow_runtime* (a local's initializer, which runs as code): then the
        element's bytes are one RuntimeSlot for the builder to store.
        """
        aggregate = self._extract_aggregate_bytes(expr, size, what, allow_runtime)
        if aggregate is not None:
            return aggregate

        value = self._extract_constant_value(expr)
        if value is not None:
            return [(value >> (i * 8)) & 0xFF for i in range(size)]

        symbol_bytes = self._extract_symbol_address_bytes(expr, size, allow_runtime)
        if symbol_bytes is not None:
            return symbol_bytes

        if allow_runtime:
            return [RuntimeSlot(expr, size, what)] * size

        raise MIRLoweringError(
            f"{what} must be a compile-time constant or the address of an "
            f"immutable static",
            source_loc=self.builder._current_source_loc
        )

    def _extract_aggregate_bytes(
        self, expr: HIRExpression, size: int, what: str,
        allow_runtime: bool = False
    ) -> Optional[List]:
        """The `size` bytes of an aggregate initializer, or None if *expr* is
        not one. A struct literal or string shorter than its slot (a union, a
        short string) is zero-padded; anything that would overrun it, or an
        array whose elements do not divide it, is an error rather than a
        shifted table."""
        if isinstance(expr, HIRStructLiteralExpr):
            data = self._extract_struct_literal_bytes(expr, allow_runtime)
        elif isinstance(expr, HIRStringLiteral):
            data = list(expr.processed_bytes)
        elif isinstance(expr, (HIRArrayLiteralExpr, HIRArrayFillExpr)):
            if isinstance(expr, HIRArrayFillExpr):
                elements = [expr.fill_value] * expr.count
            else:
                elements = expr.elements
            if not elements:
                return [0] * size
            from r65.compiler.hir.types import ArrayTypeInfo
            expr_type = getattr(expr, 'expr_type', None)
            if isinstance(expr_type, ArrayTypeInfo):
                element_size = self.builder._get_type_size(expr_type.element_type)
            else:
                element_size = size // len(elements)
            if element_size * len(elements) != size:
                raise MIRLoweringError(
                    f"{what} has {len(elements)} element(s), which do not fill "
                    f"its {size} bytes",
                    source_loc=getattr(expr, 'source_loc', None) or self.builder._current_source_loc
                )
            data = []
            for elem in elements:
                data.extend(self._extract_initializer_bytes(
                    elem, element_size, what, allow_runtime))
        else:
            return None
        if len(data) > size:
            raise MIRLoweringError(
                f"{what} is {len(data)} bytes, more than its {size}",
                source_loc=getattr(expr, 'source_loc', None) or self.builder._current_source_loc
            )
        return data + [0] * (size - len(data))

    def _extract_constant_value(self, expr: HIRExpression) -> Optional[int]:
        """
        Extract constant value from expression without emitting instructions.

        Delegates to the HIR const evaluator which handles literals, casts,
        binary/unary ops, const identifiers, and other compile-time expressions.
        """
        from r65.compiler.hir.hir_const_eval import try_eval_const_int
        return try_eval_const_int(expr, self.builder._hir_program.symbol_table)

    def _emit_array_literal_init(
        self,
        static_decl: HIRStaticDecl,
        mem_loc: MemoryLocation,
        literal_expr: HIRArrayLiteralExpr,
        skip_copy: bool = False
    ):
        """
        Emit BlockCopy instruction for array literal expression.

        Example: [1, 2, 3, 4] stores data in ROM and copies to RAM.
        Also handles arrays of struct literals like [Card { ... }, Card { ... }].
        """
        # An #[soa] field plane pads a 3- or 7-byte element to its stride
        # (`soa_stride`), so the table must too or the index would walk off
        # the data.
        data_bytes, element_size = self.array_literal_bytes(
            literal_expr, static_decl.var_type,
            f"element of static array `{static_decl.name}`",
            stride=getattr(static_decl, 'soa_stride', None))
        self._emit_rom_init(static_decl, mem_loc, data_bytes, element_size, skip_copy)

    def array_literal_bytes(self, literal_expr: HIRArrayLiteralExpr, array_type,
                            what: str, stride: Optional[int] = None,
                            allow_runtime: bool = False):
        """The ROM bytes of an array literal of type *array_type*, and its
        element size. Each element is padded to *stride* when that is given.
        With *allow_runtime*, run-time elements are RuntimeSlots."""
        from r65.compiler.hir.types import ArrayTypeInfo
        if isinstance(array_type, ArrayTypeInfo):
            element_size = self.builder._get_type_size(array_type.element_type)
        else:
            element_size = 1
        padding = [0] * ((stride or element_size) - element_size)
        data_bytes = []
        for elem in literal_expr.elements:
            data_bytes.extend(self._extract_initializer_bytes(
                elem, element_size, what, allow_runtime))
            data_bytes.extend(padding)
        return data_bytes, element_size

    @staticmethod
    def string_literal_bytes(string_literal: HIRStringLiteral, array_type) -> List[int]:
        """The bytes of a string literal, zero-padded to its array's size."""
        from r65.compiler.hir.types import ArrayTypeInfo
        data_bytes = list(string_literal.processed_bytes)
        if isinstance(array_type, ArrayTypeInfo):
            data_bytes += [0] * (array_type.size - len(data_bytes))
        return data_bytes

    def _emit_rom_init(self, static_decl: HIRStaticDecl, mem_loc: Optional[MemoryLocation],
                       data_bytes: List, element_size: int, skip_copy: bool):
        """Lay *data_bytes* out under the static's `__NAME_data` label. A ROM
        static (*skip_copy*) is read in place, so its symbol takes the label;
        a RAM static is copied from it, and `&` of it stays the RAM address."""
        label = f"__{static_decl.name}_data"
        rom_data = ROMDataRef(label=label, data=data_bytes, element_size=element_size)
        self.builder._rom_data_sections.append(rom_data)
        if skip_copy:
            static_decl.symbol.rom_label = label
        else:
            self.emit(BlockCopy(dest=mem_loc, rom_data=rom_data, count=len(data_bytes)))

    def _aggregate_field_info(self, struct_decl):
        """Map an aggregate's fields to (offset, size) and return its total size.

        Handles both HIR declarations (offsets already computed) and AST ones
        (reached when a literal is lowered before the symbol is rewritten in pass 2).
        Union layout comes from `layout_fields`, so a union's fields all land at
        offset 0 and its size is that of its largest field.
        """
        from r65.compiler.frontend import ast

        field_info = {}  # name -> (offset, size)

        if isinstance(struct_decl, HIRStructDecl):
            total_size = 0
            for field in struct_decl.fields:
                field_size = self.builder._get_type_size(field.field_type)
                field_info[field.name] = (field.offset, field_size)
                total_size = max(total_size, field.offset + field_size)
            return field_info, total_size

        if isinstance(struct_decl, ast.StructDecl):
            from r65.compiler.hir.types import TypeResolver
            from r65.compiler.hir.ast_const_eval import ConstEvaluator
            from r65.compiler.hir.unified_type_utils import layout_fields
            symbol_table = self.builder._hir_program.symbol_table
            type_resolver = TypeResolver(symbol_table, ConstEvaluator(symbol_table))
            sizes = [
                self.builder._get_type_size(type_resolver.resolve_type(f.field_type))
                for f in struct_decl.fields
            ]
            offsets, total_size = layout_fields(sizes, struct_decl.is_union)
            for field, offset, size in zip(struct_decl.fields, offsets, sizes):
                field_info[field.name] = (offset, size)
            return field_info, total_size

        raise MIRLoweringError(
            f"Unexpected struct definition type: {type(struct_decl).__name__}",
            source_loc=self.builder._current_source_loc
        )

    def _extract_struct_literal_bytes(self, struct_expr: HIRStructLiteralExpr,
                                      allow_runtime: bool = False) -> List:
        """
        Extract constant bytes from a struct literal expression (with
        *allow_runtime*, run-time fields are RuntimeSlots).

        For structs implementing traits, the __type_id field at offset 0 is
        auto-initialized with the struct's TypeId value from the symbol table.
        """
        # Find struct definition
        struct_decl = struct_expr.struct_decl
        if struct_decl is None:
            symbol = self.builder._hir_program.symbol_table.lookup(struct_expr.struct_name)
            if symbol:
                struct_decl = symbol.definition

        if struct_decl is None:
            raise MIRLoweringError(f"Cannot find struct definition for {struct_expr.struct_name}", source_loc=self.builder._current_source_loc)

        field_info, total_size = self._aggregate_field_info(struct_decl)

        # Create byte array for struct data
        data_bytes = [0] * total_size

        # Fill in field values at their offsets
        for field_init in struct_expr.fields:
            if field_init.name not in field_info:
                continue

            offset, field_size = field_info[field_init.name]
            field_bytes = self._extract_initializer_bytes(
                field_init.value, field_size,
                f"initializer for field `{struct_expr.struct_name}.{field_init.name}`",
                allow_runtime
            )

            # Store as little-endian bytes at the field's offset
            for i in range(field_size):
                data_bytes[offset + i] = field_bytes[i]

        # Auto-initialize __type_id for structs implementing traits
        if '__type_id' in field_info:
            type_id_sym = self.builder._hir_program.symbol_table.lookup(
                f"{struct_expr.struct_name}::TYPE_ID"
            )
            if type_id_sym:
                data_bytes[0] = type_id_sym.const_value & 0xFF

        return data_bytes

    def _emit_string_literal_init(
        self,
        static_decl: HIRStaticDecl,
        mem_loc: MemoryLocation,
        string_literal: HIRStringLiteral,
        skip_copy: bool = False
    ):
        """
        Emit BlockCopy instruction for string literal initialization.

        The processed_bytes field holds the escape-processed bytes from type
        checking; they are zero-padded to the declared array size.
        """
        data_bytes = self.string_literal_bytes(string_literal, static_decl.var_type)
        self._emit_rom_init(static_decl, mem_loc, data_bytes, 1, skip_copy)

    def _emit_struct_literal_init(
        self,
        static_decl: HIRStaticDecl,
        mem_loc: MemoryLocation,
        struct_expr: HIRStructLiteralExpr,
        skip_copy: bool = False
    ):
        """
        Emit BlockCopy instruction for struct literal expression.

        Example: Player { x: 10, y: 20, health: 100 } stores data in ROM and copies to RAM.
        """
        data_bytes = self._extract_struct_literal_bytes(struct_expr)
        self._emit_rom_init(static_decl, mem_loc, data_bytes, 1, skip_copy)
