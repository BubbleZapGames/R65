# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Location resolution and opcode selection for 65816 code generation.

Provides unified handling of memory locations, addressing modes,
and opcode selection through a strategy-based approach.

This module consolidates:
- Address calculation utilities
- Opcode selection based on location kind
- Location offset and manipulation
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Tuple

from r65.compiler.codegen.register_alloc import PhysicalLocation, LocationKind
from r65.compiler.codegen.opcodes import Opcode, OPCODE_VARIANTS
from r65.compiler.codegen.asm_nodes import Address, StackOffset, Immediate
from r65.compiler.codegen.constants import DP_BOUNDARY
from r65.compiler.codegen.errors import unsupported_addressing_mode
from r65.compiler.errors import InstructionSelectionError


class AddressingMode(Enum):
    """65816 addressing modes."""
    IMPLIED = auto()
    IMMEDIATE = auto()
    DP = auto()              # Direct Page
    DP_X = auto()            # Direct Page, X
    DP_Y = auto()            # Direct Page, Y
    ABSOLUTE = auto()        # Absolute 16-bit
    ABSOLUTE_X = auto()      # Absolute, X
    ABSOLUTE_Y = auto()      # Absolute, Y
    STACK = auto()           # Stack Relative
    LONG = auto()            # Long 24-bit
    LONG_X = auto()          # Long, X
    DP_INDIRECT = auto()     # (dp)
    DP_INDIRECT_X = auto()   # (dp,X)
    DP_INDIRECT_Y = auto()   # (dp),Y
    DP_INDIRECT_LONG = auto()     # [dp]
    DP_INDIRECT_LONG_Y = auto()   # [dp],Y
    STACK_INDIRECT_Y = auto()     # (sr,S),Y


# Mapping from AddressingMode to opcode variant key
_MODE_TO_VARIANT_KEY = {
    AddressingMode.IMMEDIATE: 'IMMEDIATE',
    AddressingMode.DP: 'DP',
    AddressingMode.DP_X: 'DP_X',
    AddressingMode.DP_Y: 'DP_Y',
    AddressingMode.ABSOLUTE: 'ABSOLUTE',
    AddressingMode.ABSOLUTE_X: 'ABSOLUTE_X',
    AddressingMode.ABSOLUTE_Y: 'ABSOLUTE_Y',
    AddressingMode.STACK: 'STACK',
    AddressingMode.LONG: 'LONG',
    AddressingMode.LONG_X: 'LONG_X',
    AddressingMode.DP_INDIRECT: 'DP_INDIRECT',
    AddressingMode.DP_INDIRECT_X: 'DP_INDIRECT_X',
    AddressingMode.DP_INDIRECT_Y: 'DP_INDIRECT_Y',
    AddressingMode.DP_INDIRECT_LONG: 'DP_INDIRECT_LONG',
    AddressingMode.DP_INDIRECT_LONG_Y: 'DP_INDIRECT_LONG_Y',
    AddressingMode.STACK_INDIRECT_Y: 'STACK_INDIRECT_Y',
}


@dataclass
class ResolvedLocation:
    """Result of resolving a location to an addressing mode and operand."""
    mode: AddressingMode
    operand: Address | StackOffset | Immediate
    address: Optional[int] = None  # Numeric address if applicable
    label: Optional[str] = None    # Label if applicable
    is_dp: bool = False            # True if direct page address


class LocationResolver:
    """
    Resolves physical locations to addressing modes and operands.

    Encapsulates all the logic for determining the correct addressing mode
    based on location kind, address range, and index register usage.
    """

    def __init__(self, dp_boundary: int = DP_BOUNDARY):
        """
        Initialize resolver.

        Args:
            dp_boundary: Address boundary between DP and absolute (default 0x100)
        """
        self.dp_boundary = dp_boundary

    def resolve(self, location: PhysicalLocation) -> ResolvedLocation:
        """
        Resolve a physical location to addressing mode and operand.

        Args:
            location: Physical location to resolve

        Returns:
            ResolvedLocation with mode, operand, and metadata
        """
        if location.kind == LocationKind.STACK:
            return self._resolve_stack(location)
        elif location.kind == LocationKind.SCRATCH:
            return self._resolve_scratch(location)
        elif location.kind == LocationKind.MEMORY:
            return self._resolve_memory(location)
        elif location.kind == LocationKind.IMMEDIATE:
            return self._resolve_immediate(location)
        elif location.kind == LocationKind.HARDWARE:
            raise InstructionSelectionError(
                f"Cannot resolve hardware register {location.hw_register} as memory operand", source_loc=None)
        else:
            raise InstructionSelectionError(f"Unknown location kind: {location.kind}", source_loc=None)

    def _resolve_stack(self, location: PhysicalLocation) -> ResolvedLocation:
        """Resolve stack-relative location."""
        return ResolvedLocation(
            mode=AddressingMode.STACK,
            operand=StackOffset(location.stack_offset),
            address=location.stack_offset
        )

    def _resolve_scratch(self, location: PhysicalLocation) -> ResolvedLocation:
        """Resolve scratch register (always DP) location."""
        addr = location.scratch_addr
        mode = self._get_indexed_mode(
            location.index_register,
            is_dp=True
        )
        return ResolvedLocation(
            mode=mode,
            operand=Address(addr),
            address=addr,
            is_dp=True
        )

    def _resolve_memory(self, location: PhysicalLocation) -> ResolvedLocation:
        """Resolve memory location (RAM, ROM, etc.)."""
        # Check for ROM label first
        if location.memory_label:
            # ROM data labels can be placed in any bank by the linker,
            # so always use long (24-bit) addressing for correctness
            mode = self._get_indexed_mode(
                location.index_register,
                is_dp=False,
                is_long=True  # Cross-bank ROM data needs long addressing
            )
            return ResolvedLocation(
                mode=mode,
                operand=Address(location.memory_label),
                label=location.memory_label,
                is_dp=False
            )

        # Numeric address
        addr = location.memory_addr
        is_dp = addr < self.dp_boundary
        is_long = addr > 0xFFFF  # 24-bit address needs long addressing
        mode = self._get_indexed_mode(location.index_register, is_dp, is_long)

        return ResolvedLocation(
            mode=mode,
            operand=Address(addr),
            address=addr,
            is_dp=is_dp
        )

    def _resolve_immediate(self, location: PhysicalLocation) -> ResolvedLocation:
        """Resolve immediate value location."""
        return ResolvedLocation(
            mode=AddressingMode.IMMEDIATE,
            operand=Immediate(location.immediate_value),
            address=location.immediate_value
        )

    def _get_indexed_mode(self, index_register: Optional[str], is_dp: bool, is_long: bool = False) -> AddressingMode:
        """Determine addressing mode based on index register, DP status, and address size."""
        if is_long:
            # 24-bit long addressing (bank + address)
            if index_register == 'X':
                return AddressingMode.LONG_X
            elif index_register == 'Y':
                # Note: LONG_Y doesn't exist on 65816, would need workaround
                # For now, fall back to ABSOLUTE_Y (caller should handle this case)
                return AddressingMode.ABSOLUTE_Y
            else:
                return AddressingMode.LONG
        elif is_dp:
            # Direct page (zero page) addressing
            if index_register == 'X':
                return AddressingMode.DP_X
            elif index_register == 'Y':
                return AddressingMode.DP_Y
            else:
                return AddressingMode.DP
        else:
            # 16-bit absolute addressing
            if index_register == 'X':
                return AddressingMode.ABSOLUTE_X
            elif index_register == 'Y':
                return AddressingMode.ABSOLUTE_Y
            else:
                return AddressingMode.ABSOLUTE

    def get_opcode(self, mnemonic: str, resolved: ResolvedLocation) -> Opcode:
        """
        Get the appropriate opcode for a mnemonic and resolved location.

        Args:
            mnemonic: Base instruction mnemonic (e.g., 'LDA', 'STA')
            resolved: Resolved location with addressing mode

        Returns:
            Opcode variant for the addressing mode

        Raises:
            InstructionSelectionError: If mnemonic doesn't support the mode
        """
        variants = OPCODE_VARIANTS.get(mnemonic)
        if not variants:
            raise InstructionSelectionError(f"No opcode variants for mnemonic: {mnemonic}", source_loc=None)

        variant_key = _MODE_TO_VARIANT_KEY.get(resolved.mode)
        if not variant_key:
            raise InstructionSelectionError(f"No variant key for mode: {resolved.mode}", source_loc=None)

        opcode = variants.get(variant_key)
        if not opcode:
            raise unsupported_addressing_mode(mnemonic, variant_key.lower().replace('_', ' '), source_loc=None)

        return opcode

    def resolve_and_get_opcode(self, mnemonic: str, location: PhysicalLocation) -> Tuple[Opcode, Address | StackOffset | Immediate]:
        """
        Convenience method to resolve location and get opcode in one call.

        Args:
            mnemonic: Base instruction mnemonic
            location: Physical location

        Returns:
            Tuple of (Opcode, operand)
        """
        resolved = self.resolve(location)
        opcode = self.get_opcode(mnemonic, resolved)
        return opcode, resolved.operand

    def offset_location(self, location: PhysicalLocation, offset: int) -> PhysicalLocation:
        """
        Create a new location offset by the given number of bytes.

        Used for accessing high bytes of multi-byte values.

        Args:
            location: Original location
            offset: Byte offset to add

        Returns:
            New PhysicalLocation with offset applied
        """
        if location.kind == LocationKind.STACK:
            return PhysicalLocation(
                kind=LocationKind.STACK,
                stack_offset=location.stack_offset + offset,
                size=location.size
            )
        elif location.kind == LocationKind.SCRATCH:
            return PhysicalLocation(
                kind=LocationKind.SCRATCH,
                scratch_addr=location.scratch_addr + offset,
                size=location.size,
                index_register=location.index_register
            )
        elif location.kind == LocationKind.MEMORY:
            if location.memory_label:
                # For labeled locations, append offset to label
                new_label = f"{location.memory_label}+{offset}"
                return PhysicalLocation(
                    kind=LocationKind.MEMORY,
                    memory_label=new_label,
                    size=location.size,
                    index_register=location.index_register
                )
            else:
                return PhysicalLocation(
                    kind=LocationKind.MEMORY,
                    memory_addr=location.memory_addr + offset,
                    size=location.size,
                    index_register=location.index_register
                )
        elif location.kind == LocationKind.IMMEDIATE:
            # For immediate values, shift to get the high byte
            new_value = (location.immediate_value >> (offset * 8)) & 0xFF
            return PhysicalLocation(
                kind=LocationKind.IMMEDIATE,
                immediate_value=new_value,
                size=1
            )
        else:
            raise InstructionSelectionError(f"Cannot offset location kind: {location.kind}", source_loc=None)

    def is_direct_page(self, location: PhysicalLocation) -> bool:
        """Check if a location uses direct page addressing."""
        if location.kind == LocationKind.SCRATCH:
            return True
        elif location.kind == LocationKind.MEMORY:
            if location.memory_label:
                return False  # Labels are always absolute
            return location.memory_addr < self.dp_boundary
        return False


# ============================================================================
# Store Instruction Handling
# ============================================================================

class StoreResolver:
    """
    Handles special cases for store instructions (STX, STY limitations).

    65816 STX and STY have limited addressing modes:
    - STX: No X-indexed, no stack-relative
    - STY: No Y-indexed, no stack-relative

    This class determines when a workaround is needed.
    """

    # Instructions with limited addressing mode support
    LIMITED_STORES = {
        'STX': {'blocked_index': 'X', 'transfer_to_a': Opcode.TXA},
        'STY': {'blocked_index': 'Y', 'transfer_to_a': Opcode.TYA},
    }

    @classmethod
    def needs_workaround(cls, mnemonic: str, location: PhysicalLocation) -> bool:
        """
        Check if a store instruction needs a workaround.

        Args:
            mnemonic: Store mnemonic (STX, STY, STA)
            location: Destination location

        Returns:
            True if the store needs to go through A register
        """
        if mnemonic not in cls.LIMITED_STORES:
            return False

        limits = cls.LIMITED_STORES[mnemonic]

        # Check stack-relative (not supported by STX/STY)
        if location.kind == LocationKind.STACK:
            return True

        # Check self-indexed (e.g., STX addr,X)
        if location.index_register == limits['blocked_index']:
            return True

        return False

    @classmethod
    def get_transfer_opcode(cls, mnemonic: str) -> Optional[Opcode]:
        """Get the transfer-to-A opcode for a store instruction."""
        if mnemonic in cls.LIMITED_STORES:
            return cls.LIMITED_STORES[mnemonic]['transfer_to_a']
        return None


# ============================================================================
# Singleton Instance
# ============================================================================

# Default resolver instance for common use
default_resolver = LocationResolver()
