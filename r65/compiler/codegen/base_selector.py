# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Base class for instruction selector components.

Provides common functionality shared across all selector classes:
- Parent reference and emitter property
- Common emit helper methods
- Location resolution through LocationResolver
- Abstract interface for selector composition
"""

from abc import ABC
from typing import TYPE_CHECKING

from r65.compiler.codegen.opcodes import Opcode, STORE_MNEMONICS
from r65.compiler.codegen.asm_nodes import Immediate, Address
from r65.compiler.codegen.location_resolver import (
    LocationResolver, default_resolver
)
from r65.compiler.codegen.register_alloc import PhysicalLocation, LocationKind

if TYPE_CHECKING:
    from r65.compiler.codegen.instruction_select import InstructionSelector
    from r65.compiler.codegen.emitter import AssemblyEmitter


class BaseSelector(ABC):
    """
    Base class for instruction selector components.

    Provides common emit helpers, location resolution, and parent reference
    pattern used by all selector classes (CallSelector, CompareSelector, etc.).

    Subclasses should:
    1. Implement instruction-specific selection methods
    2. Use the provided emit helpers for code generation
    3. Use location resolution methods for addressing mode selection
    """

    def __init__(self, parent: 'InstructionSelector'):
        """
        Initialize selector.

        Args:
            parent: Parent instruction selector (for emitter and helper access)
        """
        self.parent = parent
        self._resolver = default_resolver

    @property
    def emitter(self) -> 'AssemblyEmitter':
        """Get the assembly emitter from parent."""
        return self.parent.emitter

    @property
    def resolver(self) -> LocationResolver:
        """Get the location resolver."""
        return self._resolver

    # ========================================================================
    # Common Emit Helpers
    # ========================================================================

    def _emit_instr(self, opcode: Opcode, operand=None, comment: str | None = None):
        """Emit an instruction with optional operand and comment."""
        self.emitter.emit_instr(opcode, operand, comment, self.parent._current_source_loc)

    def _emit_implied(self, opcode: Opcode, comment: str | None = None):
        """Emit an implied addressing mode instruction (no operand)."""
        self.emitter.emit_instr(opcode, None, comment, self.parent._current_source_loc)

    def _emit_immediate(self, opcode: Opcode, value: int, comment: str | None = None):
        """Emit an immediate addressing mode instruction."""
        self.emitter.emit_instr(opcode, Immediate(value), comment, self.parent._current_source_loc)

    def _emit_address(self, opcode: Opcode, addr: int | str, comment: str | None = None):
        """Emit an instruction with an address operand (absolute, DP, or label)."""
        self.emitter.emit_instr(opcode, Address(addr), comment, self.parent._current_source_loc)

    def _emit_branch(self, opcode: Opcode, label: str, comment: str | None = None):
        """Emit a branch instruction to a label."""
        self.emitter.emit_instr(opcode, Address(label), comment, self.parent._current_source_loc)

    def _emit_jump(self, opcode: Opcode, label: str, comment: str | None = None):
        """Emit a jump instruction to a label."""
        self.emitter.emit_instr(opcode, Address(label), comment, self.parent._current_source_loc)

    # ========================================================================
    # Location Resolution Helpers
    # ========================================================================

    def _get_opcode_for_location(self, mnemonic: str, location: PhysicalLocation) -> tuple[Opcode, Address]:
        """
        Get opcode and operand for a mnemonic and location.

        Convenience wrapper around resolver methods.

        Args:
            mnemonic: Base instruction mnemonic
            location: Physical location

        Returns:
            Tuple of (Opcode, operand)
        """
        return self._resolver.resolve_and_get_opcode(mnemonic, location)

    def _offset_location(self, location: PhysicalLocation, offset: int) -> PhysicalLocation:
        """
        Create a new location offset by the given number of bytes.

        Args:
            location: Original location
            offset: Byte offset to add

        Returns:
            New PhysicalLocation with offset applied
        """
        return self._resolver.offset_location(location, offset)

    # ========================================================================
    # Load/Store Helpers with Workaround Support
    # ========================================================================

    def _emit_load(self, mnemonic: str, location: PhysicalLocation, comment: str = None):
        """
        Emit a load instruction with the appropriate addressing mode.

        Args:
            mnemonic: Load mnemonic (LDA, LDX, LDY)
            location: Source location
            comment: Optional comment
        """
        opcode, operand = self._resolver.resolve_and_get_opcode(mnemonic, location)
        self._emit_instr(opcode, operand, comment)

    def _emit_store_from_reg(self, reg: str, dest_loc: PhysicalLocation,
                             is_u16: bool, value_bytes: int,
                             comment: str = None):
        """Store hardware register `reg` to `dest_loc`, writing `value_bytes` bytes.

        Two width parameters, because they answer different questions and are
        not the same set:

        `value_bytes` is the destination's true width, from the MIR node's
        type_info. X and Y are unconditionally 16-bit in R65, so STX/STY always
        write TWO bytes -- a 1-byte destination cannot use them at all, whatever
        the addressing mode. This is the only thing value_bytes decides.

        `is_u16` is the existing _is_16bit predicate, threaded through unchanged,
        and drives only the accumulator width for a direct store from A. It is
        False for a 2-byte struct and for 4-byte u32/f32, so collapsing the two
        would quietly change the A path for those.
        """
        from r65.compiler.codegen.errors import InstructionSelectionError

        if reg == 'A':
            if is_u16:
                self.parent._ensure_m16_mode()
            else:
                # 8-bit store from A must ensure m8 to avoid a 16-bit STA
                # (critical for write-twice PPU registers like BGnHOFS/BGnVOFS)
                self.parent._ensure_m8_mode()
            self._emit_load_store('STA', dest_loc, comment)
            return

        if reg == 'B':
            # Goes through the XBA manager so the tracked state stays in sync,
            # and so the access is pinned to m8.
            self.parent._access_b_value_in_a()
            self._emit_load_store('STA', dest_loc, comment)
            self.parent._ensure_xba_state_normal("Restore A register")
            return

        if reg not in ('X', 'Y'):
            raise InstructionSelectionError(
                f"Cannot store from hardware register: {reg}",
                source_loc=self.parent._current_source_loc)

        # STX/STY store index-width bytes, and R65 is unconditionally x16. A
        # 1-byte destination would take its neighbour with it, so it has to go
        # through A regardless of addressing mode.
        direct = (value_bytes == 2
                  and self.parent._supports_addressing(STORE_MNEMONICS[reg], dest_loc))
        if direct:
            self._emit_load_store(STORE_MNEMONICS[reg], dest_loc, comment)
            return

        # Pin the accumulator BEFORE the transfer. TXA/TYA are M-sized: run in
        # m16 the transfer copies 16 bits, which both widens the store and
        # overwrites B with the index register's high byte.
        if value_bytes == 2:
            self.parent._ensure_m16_mode()
        else:
            self.parent._ensure_m8_mode()

        # A may still hold something live (an '@ A' parameter used after this
        # store). The mode is pinned above and unchanged until the pull, so the
        # push and the pull are the same width and the frame balances.
        a_is_live = not self.parent._hw_reg_is_free('A')
        if a_is_live:
            self._emit_instr(Opcode.PHA, comment="Save A (live value)")
            if dest_loc.kind == LocationKind.STACK:
                dest_loc = self.parent._offset_location(dest_loc, value_bytes)

        self._emit_instr(Opcode.TXA if reg == 'X' else Opcode.TYA,
                         comment=f"Transfer to A (no {STORE_MNEMONICS[reg]} with this addressing)")
        self._emit_load_store('STA', dest_loc, comment)

        if a_is_live:
            self._emit_instr(Opcode.PLA, comment="Restore A (live value)")
        else:
            self.parent._mark_a_modified()

    def _emit_load_store(self, mnemonic: str, location: PhysicalLocation, comment: str = None):
        """Emit a load/store instruction using the parent's opcode selection.

        Routes through parent._get_opcode_for_location, which applies the
        D=S / SET_DBR / spill-offset addressing adjustments. (The resolver-only
        path in _emit_load/_emit_store bypasses those, so callers must use this.)
        """
        opcode, operand = self.parent._get_opcode_for_location(mnemonic, location)
        self._emit_instr(opcode, operand, comment)

    # ========================================================================
    # Pointer Store Helpers
    # ========================================================================

    def _emit_label_pointer_store(self, label: str, dest_loc, n_bytes: int, noun: str = "function"):
        """Store a label's address as low/high[/bank] immediate bytes.

        n_bytes=2 for a near pointer, 3 for far. Emits LDA #<label>/STA,
        LDA #>label/STA, and (when far) LDA #:label/STA. `noun` selects the
        comment wording ("function" or "label").
        """
        byte_info = [("<", f"Load {noun} address low byte"),
                     (">", f"Load {noun} address high byte"),
                     (":", f"Load {noun} bank byte")]
        for i in range(n_bytes):
            prefix, comment = byte_info[i]
            loc = dest_loc if i == 0 else self.parent._offset_location(dest_loc, i)
            self._emit_instr(Opcode.LDA_IMMEDIATE, Immediate(f"{prefix}{label}"), comment)
            self._emit_load_store('STA', loc)

    def _emit_pointer_mem_copy(self, src_loc, dest_loc, n_bytes: int):
        """Copy an n_bytes pointer from src to dest as byte-by-byte LDA/STA pairs.

        Caller is responsible for ensuring 8-bit accumulator mode beforehand.
        """
        for i in range(n_bytes):
            s = src_loc if i == 0 else self.parent._offset_location(src_loc, i)
            d = dest_loc if i == 0 else self.parent._offset_location(dest_loc, i)
            self._emit_load_store('LDA', s)
            self._emit_load_store('STA', d)

    # ========================================================================
    # Operand Location Helpers
    # ========================================================================

    def _get_operand_location(self, operand) -> PhysicalLocation:
        """
        Get the physical location for a MIR operand.

        Delegates to parent's operand location tracking.

        Args:
            operand: MIR operand (VirtualRegister, Immediate, etc.)

        Returns:
            PhysicalLocation for the operand
        """
        return self.parent._get_operand_location(operand)


