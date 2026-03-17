# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Static init lowerer: generates __init_start() for static variable initialization.

Extracted from MIRBuilder to improve modularity and reduce file size.
"""

from typing import TYPE_CHECKING, Optional, List

from r65.compiler.hir import (
    HIRStaticDecl, HIRFunctionDecl, HIRStructDecl,
    HIRExpression, HIRIntegerLiteral, HIRBooleanLiteral, HIREnumVariantExpr, HIRTypeCast,
    HIRArrayFillExpr, HIRArrayLiteralExpr, HIRStringLiteral, HIRStructLiteralExpr,
    HIRIdentifier, HIRFunctionAddress,
    SymbolKind,
)
from r65.compiler.hir.attributes import StorageKind

from r65.compiler.mir.nodes import (
    MIRFunction, BasicBlock,
    VirtualRegister, FunctionPointer, MemoryLocation,
    Store, Return,
    MemoryFill, BlockCopy, ROMDataRef,
)

from r65.compiler.mir.virtual_registers import VirtualRegisterAllocator
from r65.compiler.mir.register_tracker import RegisterAliasTracker
from r65.compiler.mir.cfg import CFGBuilder
from r65.compiler.typeck.processor_mode import ProcessorMode
from r65.compiler.errors import MIRLoweringError

if TYPE_CHECKING:
    from r65.compiler.mir.builder import MIRBuilder
    from r65.compiler.mir.context import LoweringContext


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
                # Array fill: use MemoryFill instruction (loop-based)
                # Note: ROM with fill expression doesn't make sense, but handle it
                if not is_rom_storage:
                    self._emit_array_fill_init(static_decl, mem_loc, initializer)

            elif isinstance(initializer, HIRArrayLiteralExpr):
                # Array literal: create ROM data, copy to RAM only if not ROM
                self._emit_array_literal_init(static_decl, mem_loc, initializer, skip_copy=is_rom_storage)

            elif isinstance(initializer, HIRStringLiteral):
                # String literal: create ROM data, copy to RAM only if not ROM
                self._emit_string_literal_init(static_decl, mem_loc, initializer, skip_copy=is_rom_storage)

            elif isinstance(initializer, HIRStructLiteralExpr):
                # Struct literal: create ROM data, copy to RAM only if not ROM
                self._emit_struct_literal_init(static_decl, mem_loc, initializer, skip_copy=is_rom_storage)

            elif self._is_function_pointer_init(initializer):
                # Function pointer: emit Store with FunctionPointer directly
                # Skip for ROM statics (function pointer tables in ROM)
                if not is_rom_storage:
                    func_name = self._get_function_name(initializer)
                    func_ptr = FunctionPointer(function_name=func_name)
                    self.emit(Store(
                        source=func_ptr,
                        dest=mem_loc,
                        type_info=static_decl.var_type
                    ))

            else:
                # Scalar value: simple store
                # Skip for ROM statics (constants in ROM don't need runtime init)
                if not is_rom_storage:
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
        fill_expr: HIRArrayFillExpr
    ):
        """
        Emit MemoryFill instruction for array fill expression.

        Example: [0; 256] fills 256 elements with 0 using a loop.
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
        if fill_value is None:
            fill_value = 0  # Fallback

        self.emit(MemoryFill(
            dest=mem_loc,
            fill_value=fill_value,
            count=fill_expr.count,
            element_size=element_size
        ))

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
        from r65.compiler.hir.types import ArrayTypeInfo, StructTypeInfo

        # Get element type and size
        array_type = static_decl.var_type
        if isinstance(array_type, ArrayTypeInfo):
            element_size = self.builder._get_type_size(array_type.element_type)
            element_type = array_type.element_type
        else:
            element_size = 1  # Default to 1 byte
            element_type = None

        # Extract constant values from all elements without emitting instructions
        data_bytes = []
        for elem in literal_expr.elements:
            # Check if element is a struct literal
            if isinstance(elem, HIRStructLiteralExpr):
                struct_bytes = self._extract_struct_literal_bytes(elem)
                data_bytes.extend(struct_bytes)
            else:
                value = self._extract_constant_value(elem)
                if value is None:
                    value = 0  # Fallback for non-constant

                # Store as little-endian bytes
                if element_size == 1:
                    data_bytes.append(value & 0xFF)
                elif element_size == 2:
                    data_bytes.append(value & 0xFF)
                    data_bytes.append((value >> 8) & 0xFF)
                else:
                    # Handle larger types if needed
                    for i in range(element_size):
                        data_bytes.append((value >> (i * 8)) & 0xFF)

        # Create ROM data reference using variable name
        label = f"__{static_decl.name}_data"

        rom_data = ROMDataRef(
            label=label,
            data=data_bytes,
            element_size=element_size
        )
        self.builder._rom_data_sections.append(rom_data)

        # Store ROM label in symbol for code generation to use (ROM statics only).
        # For RAM/zeropage statics, the rom_label is only for initialization -
        # address-of should use the runtime address, not the ROM data label.
        if skip_copy:
            if hasattr(static_decl.symbol, 'rom_label'):
                static_decl.symbol.rom_label = label
            else:
                setattr(static_decl.symbol, 'rom_label', label)

        # Emit block copy instruction (unless this is ROM storage)
        if not skip_copy:
            self.emit(BlockCopy(
                dest=mem_loc,
                rom_data=rom_data,
                count=len(data_bytes)
            ))

    def _extract_struct_literal_bytes(self, struct_expr: HIRStructLiteralExpr) -> List[int]:
        """
        Extract constant bytes from a struct literal expression.

        For structs implementing traits, the __type_id field at offset 0 is
        auto-initialized with the struct's TypeId value from the symbol table.
        """
        from r65.compiler.frontend import ast

        # Find struct definition
        struct_decl = struct_expr.struct_decl
        if struct_decl is None:
            symbol = self.builder._hir_program.symbol_table.lookup(struct_expr.struct_name)
            if symbol:
                struct_decl = symbol.definition

        if struct_decl is None:
            raise MIRLoweringError(f"Cannot find struct definition for {struct_expr.struct_name}", source_loc=self.builder._current_source_loc)

        # Calculate field offsets and sizes
        total_size = 0
        field_info = {}  # name -> (offset, size)

        if isinstance(struct_decl, HIRStructDecl):
            for field in struct_decl.fields:
                field_size = self.builder._get_type_size(field.field_type)
                field_info[field.name] = (field.offset, field_size)
                total_size = max(total_size, field.offset + field_size)
        elif isinstance(struct_decl, ast.StructDecl):
            from r65.compiler.hir.types import TypeResolver
            from r65.compiler.hir.const_eval import ConstEvaluator
            type_resolver = TypeResolver(self.builder._hir_program.symbol_table, ConstEvaluator(self.builder._hir_program.symbol_table))
            current_offset = 0
            for field in struct_decl.fields:
                field_type = type_resolver.resolve_type(field.field_type)
                field_size = self.builder._get_type_size(field_type)
                field_info[field.name] = (current_offset, field_size)
                current_offset += field_size
            total_size = current_offset
        else:
            raise MIRLoweringError(f"Unexpected struct definition type: {type(struct_decl).__name__}", source_loc=self.builder._current_source_loc)

        # Create byte array for struct data
        data_bytes = [0] * total_size

        # Fill in field values at their offsets
        for field_init in struct_expr.fields:
            if field_init.name not in field_info:
                continue

            offset, field_size = field_info[field_init.name]
            value = self._extract_constant_value(field_init.value)
            if value is None:
                value = 0  # Fallback for non-constant

            # Store as little-endian bytes at the field's offset
            for i in range(field_size):
                data_bytes[offset + i] = (value >> (i * 8)) & 0xFF

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

        String literals are converted to byte arrays. The processed_bytes field
        contains the escape-sequence-processed byte values from type checking.
        Zero-padding is applied to match the declared array size.
        """
        from r65.compiler.hir.types import ArrayTypeInfo

        # Get the array size from the declared type
        array_type = static_decl.var_type
        if isinstance(array_type, ArrayTypeInfo):
            array_size = array_type.size
        else:
            # Shouldn't happen if type checking passed
            array_size = len(string_literal.processed_bytes)

        # Get processed bytes (escape sequences already handled by type checker)
        data_bytes = list(string_literal.processed_bytes)

        # Zero-pad to match array size
        while len(data_bytes) < array_size:
            data_bytes.append(0)

        # Create ROM data reference using variable name
        label = f"__{static_decl.name}_data"

        rom_data = ROMDataRef(
            label=label,
            data=data_bytes,
            element_size=1  # Strings are always u8 arrays
        )
        self.builder._rom_data_sections.append(rom_data)

        # Store ROM label in symbol for code generation to use (ROM statics only).
        # For RAM/zeropage statics, the rom_label is only for initialization -
        # address-of should use the runtime address, not the ROM data label.
        if skip_copy:
            if hasattr(static_decl.symbol, 'rom_label'):
                static_decl.symbol.rom_label = label
            else:
                setattr(static_decl.symbol, 'rom_label', label)

        # Emit block copy instruction (unless this is ROM storage)
        if not skip_copy:
            self.emit(BlockCopy(
                dest=mem_loc,
                rom_data=rom_data,
                count=len(data_bytes)
            ))

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
        from r65.compiler.hir.types import StructTypeInfo
        from r65.compiler.hir import HIRStructDecl
        from r65.compiler.frontend import ast

        # Get struct definition to know field sizes and offsets
        struct_decl = struct_expr.struct_decl
        if struct_decl is None:
            # Look up from symbol table
            symbol = self.builder._hir_program.symbol_table.lookup(struct_expr.struct_name)
            if symbol:
                struct_decl = symbol.definition

        if struct_decl is None:
            raise MIRLoweringError(f"Cannot find struct definition for {struct_expr.struct_name}", source_loc=self.builder._current_source_loc)

        # Calculate total size of struct and field offsets
        # Handle both HIR and AST struct declarations
        total_size = 0
        field_info = {}  # name -> (offset, size)

        if isinstance(struct_decl, HIRStructDecl):
            # HIR struct has pre-computed offsets
            for field in struct_decl.fields:
                field_size = self.builder._get_type_size(field.field_type)
                field_info[field.name] = (field.offset, field_size)
                total_size = max(total_size, field.offset + field_size)
        elif isinstance(struct_decl, ast.StructDecl):
            # AST struct - need to compute offsets
            from r65.compiler.hir.types import TypeResolver
            from r65.compiler.hir.const_eval import ConstEvaluator
            type_resolver = TypeResolver(self.builder._hir_program.symbol_table, ConstEvaluator(self.builder._hir_program.symbol_table))
            current_offset = 0
            for field in struct_decl.fields:
                field_type = type_resolver.resolve_type(field.field_type)
                field_size = self.builder._get_type_size(field_type)
                field_info[field.name] = (current_offset, field_size)
                current_offset += field_size
            total_size = current_offset
        else:
            raise MIRLoweringError(f"Unexpected struct definition type: {type(struct_decl).__name__}", source_loc=self.builder._current_source_loc)

        # Create byte array for struct data
        data_bytes = [0] * total_size

        # Fill in field values at their offsets
        for field_init in struct_expr.fields:
            if field_init.name not in field_info:
                continue

            offset, field_size = field_info[field_init.name]
            value = self._extract_constant_value(field_init.value)
            if value is None:
                value = 0  # Fallback for non-constant

            # Store as little-endian bytes at the field's offset
            for i in range(field_size):
                data_bytes[offset + i] = (value >> (i * 8)) & 0xFF

        # Auto-initialize __type_id for structs implementing traits
        if '__type_id' in field_info:
            type_id_sym = self.builder._hir_program.symbol_table.lookup(
                f"{struct_expr.struct_name}::TYPE_ID"
            )
            if type_id_sym:
                data_bytes[0] = type_id_sym.const_value & 0xFF

        # Create ROM data reference using variable name
        label = f"__{static_decl.name}_data"

        rom_data = ROMDataRef(
            label=label,
            data=data_bytes,
            element_size=1  # Struct is treated as a block of bytes
        )
        self.builder._rom_data_sections.append(rom_data)

        # Store ROM label in symbol for code generation to use (ROM statics only).
        # For RAM/zeropage statics, the rom_label is only for initialization -
        # address-of should use the runtime address, not the ROM data label.
        if skip_copy:
            if hasattr(static_decl.symbol, 'rom_label'):
                static_decl.symbol.rom_label = label
            else:
                setattr(static_decl.symbol, 'rom_label', label)

        # Emit block copy instruction (unless this is ROM storage)
        if not skip_copy:
            self.emit(BlockCopy(
                dest=mem_loc,
                rom_data=rom_data,
                count=len(data_bytes)
            ))
