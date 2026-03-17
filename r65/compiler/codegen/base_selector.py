# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Base class for instruction selector components.

Provides common functionality shared across all selector classes:
- Parent reference and emitter property
- Common emit helper methods
- Location resolution through LocationResolver
- Abstract interface for selector composition
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

from r65.compiler.codegen.opcodes import Opcode
from r65.compiler.codegen.asm_nodes import Immediate, Address
from r65.compiler.codegen.location_resolver import (
    LocationResolver, StoreResolver, default_resolver
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

    def _emit_store(self, mnemonic: str, location: PhysicalLocation, comment: str = None):
        """
        Emit a store instruction, handling STX/STY limitations.

        STX and STY have limited addressing mode support. This method
        automatically applies workarounds when needed.

        Args:
            mnemonic: Store mnemonic (STA, STX, STY)
            location: Destination location
            comment: Optional comment
        """
        if StoreResolver.needs_workaround(mnemonic, location):
            # Transfer to A first, then use STA
            transfer_op = StoreResolver.get_transfer_opcode(mnemonic)
            self._emit_instr(transfer_op, comment=f"Transfer to A (no {mnemonic} with this addressing)")
            opcode, operand = self._resolver.resolve_and_get_opcode('STA', location)
            self._emit_instr(opcode, operand, comment)
            self.parent._mark_a_modified()
        else:
            opcode, operand = self._resolver.resolve_and_get_opcode(mnemonic, location)
            self._emit_instr(opcode, operand, comment)

    def _emit_load_store(self, mnemonic: str, location: PhysicalLocation, comment: str = None):
        """
        Emit a load or store instruction based on the mnemonic.

        Convenience method that routes to _emit_load or _emit_store.

        Args:
            mnemonic: Instruction mnemonic
            location: Memory location
            comment: Optional comment
        """
        if mnemonic.startswith('ST'):
            self._emit_store(mnemonic, location, comment)
        else:
            self._emit_load(mnemonic, location, comment)

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



class SelectorComponent:
    """
    Mixin for selector components that can be composed.

    Provides a standard interface for selectors that can be combined
    or used independently within the instruction selection pipeline.
    """

    def set_parent(self, parent: 'InstructionSelector'):
        """Set the parent instruction selector."""
        self.parent = parent

    def set_resolver(self, resolver: LocationResolver):
        """Set the location resolver."""
        self._resolver = resolver
