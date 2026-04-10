# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Symbol definition generator: emits .DEFINE and .EQU directives.

Generates symbol definitions from memory allocations and constants.
"""

from typing import Optional
from r65.compiler.hir import HIRConstDecl
from r65.compiler.codegen.memory_alloc import MemoryAllocator, AllocationInfo
from r65.compiler.codegen.emitter import AssemblyEmitter


class SymbolDefinitionGenerator:
    """
    Generates symbol definition sections in assembly.

    Emits .DEFINE directives for variables and .EQU directives for constants,
    organized by storage type. RAM variables use .RAMSECTION for correct
    bank byte metadata with WLA-DX's #: operator.
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
    # Allocation Section Emission (Common Pattern)
    # ========================================================================

    def _emit_allocation_section(self, storage_type: str, section_title: str,
                                  use_hw_comment: bool = False):
        """
        Emit a section of variable definitions.

        Common pattern for emitting allocation sections:
        1. Get allocations by type
        2. Skip if empty
        3. Emit section header
        4. Sort by address
        5. Emit definitions
        6. Emit blank line

        Args:
            storage_type: Storage type to query ('zeropage', 'ram', 'hw', etc.)
            section_title: Title for section header
            use_hw_comment: Use hardware register comment format
        """
        allocations = self.allocator.get_allocations_by_type(storage_type)

        if not allocations:
            return

        self.emitter.emit_section_header(section_title)

        # Sort by address for readability
        allocations.sort(key=lambda a: a.address)

        for alloc in allocations:
            comment = self._make_hw_comment(alloc) if use_hw_comment else self._make_allocation_comment(alloc)
            self.emitter.emit_define(alloc.symbol.name, alloc.address, comment)

        self.emitter.emit_blank_line()

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
        self._emit_allocation_section('zeropage', "Direct Page Allocations")

    # ========================================================================
    # RAM Definitions
    # ========================================================================

    def emit_ram_definitions(self):
        """
        Emit RAM variable definitions using .RAMSECTION blocks.

        Uses WLA-DX .RAMSECTION instead of .DEFINE so that the #: (bank byte)
        operator returns the correct bank ($7E or $7F) for WRAM addresses.

        Generated:
            ; ============================================================================
            ; RAM Allocations
            ; ============================================================================
            .RAMSECTION "ram.7E" BANK $7E SLOT 1 FORCE ORGA $2000
                BUFFER dsb 256
                PLAYER_DATA dsb 100
            .ENDS
        """
        allocations = self.allocator.get_allocations_by_type('ram')

        if not allocations:
            return

        self.emitter.emit_section_header("RAM Allocations")

        # Sort by address
        allocations.sort(key=lambda a: a.address)

        # Split into bank $7E and bank $7F groups
        bank_7e = [a for a in allocations if a.address < 0x7F0000]
        bank_7f = [a for a in allocations if a.address >= 0x7F0000]

        if bank_7e:
            self._emit_ram_ramsection(bank_7e, bank=0x7E, slot=1,
                                      section_name="ram.7E",
                                      bank_base=0x7E0000)

        if bank_7f:
            self._emit_ram_ramsection(bank_7f, bank=0x7F, slot=2,
                                      section_name="ram.7F",
                                      bank_base=0x7F0000)

        self.emitter.emit_blank_line()

    def _emit_ram_ramsection(self, allocations: list, bank: int, slot: int,
                              section_name: str, bank_base: int):
        """
        Emit a single .RAMSECTION block for a set of RAM allocations.

        Inserts padding `dsb` entries for gaps between allocations.

        Args:
            allocations: Sorted list of AllocationInfo in this bank
            bank: Bank number (0x7E or 0x7F)
            slot: WLA-DX slot number
            section_name: Section name for the .RAMSECTION
            bank_base: Base address of the bank (e.g., 0x7E0000)
        """
        # Compute the origin address relative to the slot
        first_addr = allocations[0].address
        orga = first_addr - bank_base

        entries = []
        current_addr = first_addr

        for alloc in allocations:
            # Insert padding for gaps
            gap = alloc.address - current_addr
            if gap > 0:
                entries.append((f"__pad_{current_addr - bank_base:04X}", gap, "padding"))

            comment = self._make_allocation_comment(alloc)
            entries.append((alloc.symbol.name, alloc.size, comment))
            current_addr = alloc.address + alloc.size

        self.emitter.emit_ramsection(section_name, bank, slot, orga, entries)

    # ========================================================================
    # Low RAM Definitions
    # ========================================================================

    def emit_lowram_definitions(self):
        """
        Emit low RAM variable definitions ($0100-$1FFF).

        Generated:
            ; ============================================================================
            ; Low RAM Allocations
            ; ============================================================================
            .DEFINE BUFFER $0100        ; 256 bytes
            .DEFINE TEMP $0200          ; Auto-allocated
        """
        self._emit_allocation_section('lowram', "Low RAM Allocations")

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
        self._emit_allocation_section('hw', "Hardware Register Definitions", use_hw_comment=True)

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
        emits .INCBIN directives. Respects #[bank(n)] attributes.

        Generated:
            ; ============================================================================
            ; ROM Data
            ; ============================================================================
            .BANK 1 SLOT 0
            GRAPHICS_DATA:
            .INCBIN "gfx.bin"
        """
        from r65.compiler.hir import HIRIncludeBytesExpr
        from collections import defaultdict
        import os

        allocations = self.allocator.get_allocations_by_type('rom')

        if not allocations:
            return

        # Collect ROM data allocations with include_bytes initializers
        # Group by bank number
        bank_data = defaultdict(list)  # bank_number -> [(alloc, static_decl, file_size)]
        auto_bank_data = []  # allocations without explicit bank

        for alloc in allocations:
            if hasattr(alloc.symbol, 'definition') and alloc.symbol.definition:
                static_decl = alloc.symbol.definition
                if hasattr(static_decl, 'initializer') and static_decl.initializer:
                    if isinstance(static_decl.initializer, HIRIncludeBytesExpr):
                        filepath = static_decl.initializer.path
                        try:
                            file_size = os.path.getsize(filepath)
                        except OSError:
                            file_size = 0

                        # Check for bank attribute
                        if hasattr(static_decl, 'bank_attr') and static_decl.bank_attr:
                            bank_num = static_decl.bank_attr.bank_number
                            if bank_num is not None:
                                bank_data[bank_num].append((alloc, static_decl, file_size))
                            else:
                                # Auto-placement mode
                                auto_bank_data.append((alloc, static_decl, file_size))
                        else:
                            # No bank attribute - use auto-placement
                            auto_bank_data.append((alloc, static_decl, file_size))

        if not bank_data and not auto_bank_data:
            return

        self.emitter.emit_section_header("ROM Data")

        bank_size = 0x8000  # 32KB per bank

        # Emit data for each explicit bank in order
        for bank_num in sorted(bank_data.keys()):
            self.emitter.emit_bank_directive(bank_num, slot=0)
            for alloc, static_decl, file_size in bank_data[bank_num]:
                label = f"{alloc.symbol.name}_data"
                filepath = static_decl.initializer.path
                alloc.symbol.rom_label = label
                self.emitter.emit_incbin(filepath, label=label)

        # Emit auto-bank data starting at bank 4 (or higher if explicit banks used)
        if auto_bank_data:
            # Find next available bank after explicit banks
            if bank_data:
                auto_start_bank = max(bank_data.keys()) + 1
                if auto_start_bank < 4:
                    auto_start_bank = 4
            else:
                auto_start_bank = 4

            current_bank = auto_start_bank
            current_size = 0
            self.emitter.emit_bank_directive(current_bank, slot=0)

            for alloc, static_decl, file_size in auto_bank_data:
                # Check if we need to switch banks
                if current_size + file_size > bank_size:
                    current_bank += 1
                    current_size = 0
                    self.emitter.emit_bank_directive(current_bank, slot=0)

                label = f"{alloc.symbol.name}_data"
                filepath = static_decl.initializer.path
                alloc.symbol.rom_label = label
                self.emitter.emit_incbin(filepath, label=label)
                current_size += file_size

        self.emitter.emit_blank_line()

    # ========================================================================
    # Stack Bounds
    # ========================================================================

    def emit_stack_bounds(self):
        """
        Emit WLA-DX defines for the stack region bounds.

        These expose `#[stack(lo, hi)]` (or the default region) to user
        `asm!()` blocks — e.g. for stack-guard macros that check the stack
        pointer against the declared floor.
        """
        lower = self.allocator.stack_lower
        upper = self.allocator.stack_upper
        if lower is None or upper is None:
            return

        self.emitter.emit_section_header("Stack Region")
        self.emitter.emit_define("__R65_STACK_LO", lower, "Stack lower bound")
        self.emitter.emit_define("__R65_STACK_HI", upper, "Stack upper bound (S init)")
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
        1. Stack bounds (.DEFINE __R65_STACK_LO / __R65_STACK_HI)
        2. Constants (.EQU)
        3. Direct Page (.DEFINE)
        4. Low RAM (.DEFINE)
        5. Hardware Registers (.DEFINE)
        6. RAM (.DEFINE)
        7. ROM Data (.INCBIN)
        """
        # Stack bounds first — stable well-known symbols usable from asm!()
        self.emit_stack_bounds()

        # Constants
        if constants:
            self.emit_constant_definitions(constants)

        # Then variables by storage type
        self.emit_zeropage_definitions()
        self.emit_lowram_definitions()
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
