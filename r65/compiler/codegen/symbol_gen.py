"""
Symbol definition generator: emits .DEFINE and .EQU directives.

Generates symbol definitions from memory allocations and constants.
"""

from typing import Optional
from r65.compiler.hir import HIRConstDecl
from r65.compiler.codegen.memory_alloc import *
from r65.compiler.codegen.emitter import *


class SymbolDefinitionGenerator:
    """
    Generates symbol definition sections in assembly.

    Emits .DEFINE directives for variables and .EQU directives for constants,
    organized by storage type.
    """

    def __init__(self, emitter: AssemblyEmitter, allocator: MemoryAllocator):
        """
        Initialize symbol definition generator.

        Args:
            emitter: Assembly emitter
            allocator: Memory allocator with allocations
        """
        self.emitter = emitter
        self.allocator = allocator

    # ========================================================================
    # Direct Page (Zero-Page) Definitions
    # ========================================================================

    def emit_zeropage_definitions(self):
        """
        Emit zero-page variable definitions.

        Generated:
            ; ============================================================================
            ; Direct Page Allocations
            ; ============================================================================
            .DEFINE TEMP $20            ; main.r65:5
            .DEFINE COUNTER $22         ; Auto-allocated
        """
        allocations = self.allocator.get_allocations_by_type('zeropage')

        if not allocations:
            return

        self.emitter.emit_section_header("Direct Page Allocations")

        # Sort by address for readability
        allocations.sort(key=lambda a: a.address)

        for alloc in allocations:
            comment = self._make_allocation_comment(alloc)
            self.emitter.emit_define(alloc.symbol.name, alloc.address, comment)

        self.emitter.emit_blank_line()

    # ========================================================================
    # RAM Definitions
    # ========================================================================

    def emit_ram_definitions(self):
        """
        Emit RAM variable definitions.

        Generated:
            ; ============================================================================
            ; RAM Allocations
            ; ============================================================================
            .DEFINE BUFFER $7E0000      ; 256 bytes
            .DEFINE PLAYER_DATA $7E0100 ; 100 bytes
        """
        allocations = self.allocator.get_allocations_by_type('ram')

        if not allocations:
            return

        self.emitter.emit_section_header("RAM Allocations")

        # Sort by address
        allocations.sort(key=lambda a: a.address)

        for alloc in allocations:
            comment = self._make_allocation_comment(alloc)
            self.emitter.emit_define(alloc.symbol.name, alloc.address, comment)

        self.emitter.emit_blank_line()

    # ========================================================================
    # Hardware Register Definitions
    # ========================================================================

    def emit_hw_definitions(self):
        """
        Emit hardware register definitions.

        Generated:
            ; ============================================================================
            ; Hardware Register Definitions
            ; ============================================================================
            .DEFINE INIDISP $2100       ; Screen brightness
            .DEFINE HVBJOY $4212        ; VBlank/HBlank status
        """
        allocations = self.allocator.get_allocations_by_type('hw')

        if not allocations:
            return

        self.emitter.emit_section_header("Hardware Register Definitions")

        # Sort by address
        allocations.sort(key=lambda a: a.address)

        for alloc in allocations:
            comment = self._make_hw_comment(alloc)
            self.emitter.emit_define(alloc.symbol.name, alloc.address, comment)

        self.emitter.emit_blank_line()

    # ========================================================================
    # Constant Definitions
    # ========================================================================

    def emit_constant_definitions(self, constants: list[HIRConstDecl]):
        """
        Emit constant definitions.

        Args:
            constants: List of constant declarations from HIR

        Generated:
            ; ============================================================================
            ; Constants
            ; ============================================================================
            .EQU SCREEN_WIDTH 256
            .EQU SCREEN_HEIGHT 224
        """
        if not constants:
            return

        self.emitter.emit_section_header("Constants")

        for const in constants:
            # Get evaluated value
            if hasattr(const, 'evaluated_value') and const.evaluated_value is not None:
                value = const.evaluated_value
            else:
                # Fallback to 0 if not evaluated
                value = 0

            self.emitter.emit_equ(const.name, value)

        self.emitter.emit_blank_line()

    # ========================================================================
    # ROM Data Definitions
    # ========================================================================

    def emit_rom_data(self):
        """
        Emit ROM data with initializers.

        For static variables in ROM with include_bytes! initializers,
        emits .INCBIN directives.

        Generated:
            ; ============================================================================
            ; ROM Data
            ; ============================================================================
            GRAPHICS_DATA:
            .INCBIN "gfx.bin"
        """
        from r65.compiler.hir import HIRIncludeBytesExpr

        allocations = self.allocator.get_allocations_by_type('rom')

        if not allocations:
            return

        # Check if any ROM allocations have include_bytes initializers
        has_rom_data = False
        for alloc in allocations:
            if hasattr(alloc.symbol, 'definition') and alloc.symbol.definition:
                static_decl = alloc.symbol.definition
                if hasattr(static_decl, 'initializer') and static_decl.initializer:
                    if isinstance(static_decl.initializer, HIRIncludeBytesExpr):
                        has_rom_data = True
                        break

        if not has_rom_data:
            return

        self.emitter.emit_section_header("ROM Data")

        # Emit ROM data for each allocation with include_bytes initializer
        for alloc in allocations:
            if hasattr(alloc.symbol, 'definition') and alloc.symbol.definition:
                static_decl = alloc.symbol.definition
                if hasattr(static_decl, 'initializer') and static_decl.initializer:
                    if isinstance(static_decl.initializer, HIRIncludeBytesExpr):
                        # Emit label and .INCBIN directive
                        label = f"{alloc.symbol.name}_data"
                        filepath = static_decl.initializer.path
                        self.emitter.emit_incbin(filepath, label=label)

        self.emitter.emit_blank_line()

    # ========================================================================
    # All Definitions
    # ========================================================================

    def emit_all_definitions(self, constants: Optional[list[HIRConstDecl]] = None):
        """
        Emit all symbol definitions in order.

        Args:
            constants: Optional list of constant declarations

        Order:
        1. Constants (.EQU)
        2. Direct Page (.DEFINE)
        3. Hardware Registers (.DEFINE)
        4. RAM (.DEFINE)
        5. ROM Data (.INCBIN)
        """
        # Constants first
        if constants:
            self.emit_constant_definitions(constants)

        # Then variables by storage type
        self.emit_zeropage_definitions()
        self.emit_hw_definitions()
        self.emit_ram_definitions()

        # ROM data with initializers
        self.emit_rom_data()

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def _make_allocation_comment(self, alloc: AllocationInfo) -> str:
        """
        Generate comment for allocation.

        Args:
            alloc: Allocation info

        Returns:
            Comment string (e.g., "main.r65:5" or "Auto-allocated - 16 bytes")
        """
        parts = []

        # Source info if available
        if alloc.source_info:
            parts.append(alloc.source_info)
        elif alloc.is_explicit:
            parts.append("Explicit")
        else:
            parts.append("Auto-allocated")

        # Size info if > 1 byte
        if alloc.size > 1:
            parts.append(f"{alloc.size} bytes")

        return " - ".join(parts) if parts else ""

    def _make_hw_comment(self, alloc: AllocationInfo) -> str:
        """
        Generate comment for hardware register.

        Args:
            alloc: Allocation info

        Returns:
            Comment string (e.g., "Screen brightness")
        """
        # For hardware registers, try to add descriptive comment
        # This could be enhanced with a hardware register database
        if alloc.source_info:
            return alloc.source_info

        # Default
        return "Hardware register"
