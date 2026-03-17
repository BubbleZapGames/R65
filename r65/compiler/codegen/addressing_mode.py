# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Addressing mode selection: choose optimal 65816 addressing modes.

Selects the most efficient addressing mode based on operand type,
address range, and instruction requirements.
"""

from typing import Optional, Tuple
from enum import Enum
from r65.compiler.codegen.register_alloc import PhysicalLocation, LocationKind
from r65.compiler.errors import AddressingModeError
from r65.compiler.codegen.constants import DP_MAX, ABSOLUTE_MIN, ABSOLUTE_MAX


class AddressingMode(Enum):
    """65816 addressing modes."""
    IMMEDIATE = "immediate"              # #$42
    ACCUMULATOR = "accumulator"          # A (implied)
    DIRECT_PAGE = "direct_page"          # $20 (zero-page)
    ABSOLUTE = "absolute"                # $2000
    DIRECT_PAGE_X = "direct_page_x"      # $20,X
    DIRECT_PAGE_Y = "direct_page_y"      # $20,Y
    ABSOLUTE_X = "absolute_x"            # $2000,X
    ABSOLUTE_Y = "absolute_y"            # $2000,Y
    INDIRECT = "indirect"                # ($42)
    INDIRECT_Y = "indirect_y"            # ($42),Y
    LONG = "long"                        # $7E0000 (24-bit)
    LONG_X = "long_x"                    # $7E0000,X


class AddressingModeSelector:
    """
    Selects optimal addressing modes for 65816 instructions.

    Chooses the smallest/fastest addressing mode that can reach
    the operand address.
    """

    def __init__(self):
        """Initialize addressing mode selector."""
        pass

    # ========================================================================
    # Main Selection Methods
    # ========================================================================

    def select_for_location(self,
                           location: PhysicalLocation,
                           index_register: Optional[str] = None,
                           is_indirect: bool = False) -> Tuple[AddressingMode, str]:
        """
        Select addressing mode for physical location.

        Args:
            location: Physical location of operand
            index_register: If using indexed mode ("X" or "Y")
            is_indirect: If using indirect addressing

        Returns:
            Tuple of (AddressingMode, operand_string)
        """
        # Hardware registers use accumulator mode (implied)
        if location.is_hw():
            if location.hw_register == 'A':
                return (AddressingMode.ACCUMULATOR, "A")
            else:
                # X, Y cannot be used as addressing mode operands
                raise AddressingModeError(f"Cannot use {location.hw_register} as operand", source_loc=None)

        # Get effective address
        if location.kind == LocationKind.SCRATCH:
            addr = location.scratch_addr
        elif location.kind == LocationKind.MEMORY:
            addr = location.memory_addr
        elif location.kind == LocationKind.STACK:
            # Stack addressing requires special handling
            # For now, not supported
            raise AddressingModeError("Stack addressing not yet implemented", source_loc=None)
        else:
            raise AddressingModeError(f"Unknown location kind: {location.kind}", source_loc=None)

        # Select addressing mode based on address range and modifiers
        return self._select_mode(addr, index_register, is_indirect)

    def select_immediate(self, value: int, is_16bit: bool = False) -> Tuple[AddressingMode, str]:
        """
        Select immediate addressing mode.

        Args:
            value: Immediate value
            is_16bit: True for 16-bit immediate

        Returns:
            Tuple of (AddressingMode, operand_string)
        """
        if is_16bit:
            # 16-bit immediate
            operand = f"#${value:04X}"
        else:
            # 8-bit immediate
            operand = f"#${value & 0xFF:02X}"

        return (AddressingMode.IMMEDIATE, operand)

    # ========================================================================
    # Mode Selection Logic
    # ========================================================================

    def _select_mode(self,
                    addr: int,
                    index_register: Optional[str],
                    is_indirect: bool) -> Tuple[AddressingMode, str]:
        """
        Select mode based on address, indexing, and indirection.

        Args:
            addr: Memory address
            index_register: Index register name ("X" or "Y") or None
            is_indirect: True for indirect addressing

        Returns:
            Tuple of (AddressingMode, operand_string)
        """
        # Determine address range
        is_direct_page = (0 <= addr <= DP_MAX)
        is_absolute = (ABSOLUTE_MIN <= addr <= ABSOLUTE_MAX)
        is_long = (addr > ABSOLUTE_MAX)

        # Indirect addressing
        if is_indirect:
            if not is_direct_page:
                raise AddressingModeError("Indirect addressing requires zero-page pointer", source_loc=None)

            if index_register == 'Y':
                # ($42),Y - indirect indexed
                operand = f"(${addr:02X}),Y"
                return (AddressingMode.INDIRECT_Y, operand)
            else:
                # ($42) - indirect
                operand = f"(${addr:02X})"
                return (AddressingMode.INDIRECT, operand)

        # Indexed addressing
        if index_register:
            if is_long:
                # Long indexed
                if index_register == 'X':
                    operand = f"${addr:06X},X"
                    return (AddressingMode.LONG_X, operand)
                else:
                    raise AddressingModeError("Long addressing only supports X indexing", source_loc=None)

            elif is_absolute:
                # Absolute indexed
                if index_register == 'X':
                    operand = f"${addr:04X},X"
                    return (AddressingMode.ABSOLUTE_X, operand)
                elif index_register == 'Y':
                    operand = f"${addr:04X},Y"
                    return (AddressingMode.ABSOLUTE_Y, operand)

            else:  # is_direct_page
                # Direct page indexed
                if index_register == 'X':
                    operand = f"${addr:02X},X"
                    return (AddressingMode.DIRECT_PAGE_X, operand)
                elif index_register == 'Y':
                    operand = f"${addr:02X},Y"
                    return (AddressingMode.DIRECT_PAGE_Y, operand)

        # Non-indexed addressing
        if is_long:
            # 24-bit absolute long
            operand = f"${addr:06X}"
            return (AddressingMode.LONG, operand)

        elif is_absolute:
            # 16-bit absolute
            operand = f"${addr:04X}"
            return (AddressingMode.ABSOLUTE, operand)

        else:  # is_direct_page
            # 8-bit direct page (fastest!)
            operand = f"${addr:02X}"
            return (AddressingMode.DIRECT_PAGE, operand)

    # ========================================================================
    # Optimization Helpers
    # ========================================================================

    def can_use_direct_page(self, addr: int) -> bool:
        """
        Check if address can use direct page mode.

        Args:
            addr: Memory address

        Returns:
            True if address is in zero-page range
        """
        return 0 <= addr <= 0xFF

    def should_use_stz(self, value: int) -> bool:
        """
        Check if STZ (store zero) can be used instead of LDA #0; STA.

        STZ is 65816-specific and more efficient for storing zero.

        Args:
            value: Value being stored

        Returns:
            True if value is zero (can use STZ)
        """
        return value == 0

    def get_cycle_count(self, mode: AddressingMode, instruction: str) -> int:
        """
        Get approximate cycle count for addressing mode.

        This is a simplified estimate. Actual cycles depend on:
        - Processor mode (8-bit vs 16-bit)
        - Page boundary crossings
        - Memory access speed

        Args:
            mode: Addressing mode
            instruction: Instruction mnemonic (e.g., "LDA")

        Returns:
            Estimated cycle count
        """
        # Simplified cycle counts for common instructions
        if mode == AddressingMode.IMMEDIATE:
            return 2  # Fastest
        elif mode == AddressingMode.ACCUMULATOR:
            return 2
        elif mode == AddressingMode.DIRECT_PAGE:
            return 3  # Fast - zero-page
        elif mode == AddressingMode.DIRECT_PAGE_X:
            return 4
        elif mode == AddressingMode.DIRECT_PAGE_Y:
            return 4
        elif mode == AddressingMode.ABSOLUTE:
            return 4  # Slower - 16-bit address
        elif mode == AddressingMode.ABSOLUTE_X:
            return 4  # May be 5 with page crossing
        elif mode == AddressingMode.ABSOLUTE_Y:
            return 4  # May be 5 with page crossing
        elif mode == AddressingMode.INDIRECT:
            return 5  # Slower - pointer dereference
        elif mode == AddressingMode.INDIRECT_Y:
            return 5  # May be 6 with page crossing
        elif mode == AddressingMode.LONG:
            return 5  # 24-bit address
        elif mode == AddressingMode.LONG_X:
            return 5
        else:
            return 4  # Default estimate

    def get_byte_size(self, mode: AddressingMode, instruction: str) -> int:
        """
        Get instruction size in bytes for addressing mode.

        Args:
            mode: Addressing mode
            instruction: Instruction mnemonic

        Returns:
            Instruction size in bytes
        """
        # Base instruction byte
        size = 1

        # Add operand bytes
        if mode == AddressingMode.IMMEDIATE:
            size += 1  # 8-bit immediate (may be 2 for 16-bit mode)
        elif mode == AddressingMode.ACCUMULATOR:
            size += 0  # No operand
        elif mode in (AddressingMode.DIRECT_PAGE,
                     AddressingMode.DIRECT_PAGE_X,
                     AddressingMode.DIRECT_PAGE_Y):
            size += 1  # 8-bit address
        elif mode in (AddressingMode.ABSOLUTE,
                     AddressingMode.ABSOLUTE_X,
                     AddressingMode.ABSOLUTE_Y):
            size += 2  # 16-bit address
        elif mode in (AddressingMode.INDIRECT,
                     AddressingMode.INDIRECT_Y):
            size += 1  # 8-bit zero-page pointer
        elif mode in (AddressingMode.LONG,
                     AddressingMode.LONG_X):
            size += 3  # 24-bit address

        return size

    # ========================================================================
    # Format Helpers
    # ========================================================================

    def format_operand(self,
                      location: PhysicalLocation,
                      index_register: Optional[str] = None,
                      is_indirect: bool = False) -> str:
        """
        Format location as assembly operand string.

        This is a convenience wrapper around select_for_location
        that returns just the operand string.

        Args:
            location: Physical location
            index_register: Optional index register
            is_indirect: Use indirect addressing

        Returns:
            Formatted operand string
        """
        mode, operand = self.select_for_location(location, index_register, is_indirect)
        return operand

    def format_immediate(self, value: int, is_16bit: bool = False) -> str:
        """
        Format immediate value as assembly operand.

        Args:
            value: Immediate value
            is_16bit: True for 16-bit immediate

        Returns:
            Formatted operand string (e.g., "#$42")
        """
        mode, operand = self.select_immediate(value, is_16bit)
        return operand
