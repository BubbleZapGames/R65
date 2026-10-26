# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Symbol definition generator: emits .DEFINE and .EQU directives.

Generates symbol definitions from memory allocations and constants.
"""

from typing import Dict, List, Optional, Tuple
from r65.compiler.hir import HIRConstDecl
from r65.compiler.codegen.memory_alloc import MemoryAllocator, AllocationInfo
from r65.compiler.codegen.emitter import AssemblyEmitter


def rom_static_bank(static_decl) -> Optional[int]:
    """The WLA bank a static's ROM data goes in: its #[bank(n)], or None
    under #[bank(auto)] (the caller picks the auto bank). A static with no
    bank attribute (RAM, extern) has its data in bank 0."""
    bank_attr = getattr(static_decl, 'bank_attr', None)
    if bank_attr is None:
        return 0
    return bank_attr.bank_number


# #[bank(auto)] never goes below this bank, leaving banks 0-3 to
# hand-placed code and data.
AUTO_BANK_FLOOR = 4


def include_bytes_by_bank(mir_program, bank_size: int) -> Dict[int, List[Tuple[str, str, int]]]:
    """Where each include_bytes! static's data goes: (label, path, size) by
    WLA bank, and each static's rom_label set to its label.

    A #[bank(n)] static goes in bank n; #[bank(auto)] ones pack upward from
    `auto_bank_number`, *bank_size* bytes per bank. The data is emitted in
    its bank's window together with the bank's code: a bank opened on its
    own beforehand would be re-entered by `.BANK` alone, which WLA-DX
    continues at the offset the previous bank reached, laying the code
    over the data.
    """
    from r65.compiler.hir import HIRIncludeBytesExpr
    placed: Dict[int, List[Tuple[str, str, int]]] = {}
    auto: List[Tuple[str, str, int]] = []
    for decl in mir_program.statics:
        init = getattr(decl, 'initializer', None)
        if getattr(decl, 'is_extern', False) or not isinstance(init, HIRIncludeBytesExpr):
            continue
        label = f"{decl.symbol.name}_data"
        decl.symbol.rom_label = label
        # Sized by the frontend, which resolved the path against the including file
        entry = (label, init.path, init.size)
        bank = rom_static_bank(decl)
        if bank is None:
            auto.append(entry)
        else:
            placed.setdefault(bank, []).append(entry)

    bank, used = auto_bank_number(mir_program), 0
    for entry in auto:
        if used and used + entry[2] > bank_size:
            bank, used = bank + 1, 0
        placed.setdefault(bank, []).append(entry)
        used += entry[2]
    return placed


def auto_bank_number(mir_program) -> int:
    """The bank #[bank(auto)] code and ROM data start in: the one above every
    explicit #[bank(n)] function or static, at least AUTO_BANK_FLOOR. Code
    placement, ROM data placement and far_to_near all read it from here."""
    explicit = [f.bank_attr.bank_number for f in mir_program.functions
                if f.bank_attr is not None and f.bank_attr.bank_number is not None]
    explicit += [b for b in map(rom_static_bank, mir_program.statics) if b is not None]
    return max(max(explicit, default=-1) + 1, AUTO_BANK_FLOOR)


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
    # Overlay Definitions
    # ========================================================================

    def emit_overlay_definitions(self):
        """
        Emit planes that overlay another allocation's bytes.

        An `#[soa]` payload's field planes sit inside its enum's data region
        (docs/enums.md, "#[soa] Payloads"). They are always `.DEFINE`d — even in
        main RAM, where the region itself is a `.RAMSECTION` entry — because a
        `.RAMSECTION` lists bytes consecutively and cannot express two names for
        the same ones.

        Generated:
            ; ============================================================================
            ; Overlay Planes
            ; ============================================================================
            .DEFINE POOL__kind__Fireball__heat $0210  ; overlays POOL__kind__data+0
        """
        allocations = self.allocator.get_overlay_allocations()
        if not allocations:
            return
        self.emitter.emit_section_header("Overlay Planes")
        allocations.sort(key=lambda a: (a.address, a.symbol.name))
        for alloc in allocations:
            self.emitter.emit_define(alloc.symbol.name, alloc.address,
                                     f"{alloc.size} bytes, overlay")
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
            .DEFINE TEMP $0020          ; main.r65:5
            .DEFINE COUNTER $0022       ; Auto-allocated
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
            .BASE $7E
            .RAMSECTION "ram.7E" BANK $00 SLOT 1 FORCE ORGA $2000
                BUFFER dsb 256
                PLAYER_DATA dsb 100
            .ENDS
            .BASE $<rom_base>
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
        Emit low RAM variable definitions ($0000-$1FFF, shared with direct page).

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

            # Skip struct/dict constants — WLA-DX .EQU only accepts integers
            # (a bool is one, written as 1/0 by emit_equ)
            if not isinstance(value, int):
                continue

            self.emitter.emit_equ(const.name, value)

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

    def emit_all_definitions(self, constants: Optional[list[HIRConstDecl]]):
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
        6. RAM (.RAMSECTION)
        7. Overlay planes (.DEFINE)

        include_bytes! data is placed by include_bytes_by_bank and emitted in
        its bank's window with the code.
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
        self.emit_overlay_definitions()

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
            Comment string: the allocation's source info, else
            "Hardware register"
        """
        if alloc.source_info:
            return alloc.source_info

        # Default
        return "Hardware register"
