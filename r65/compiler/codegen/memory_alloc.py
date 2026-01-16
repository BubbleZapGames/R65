"""
Memory allocation: tracks and allocates memory locations.

Manages zero-page, RAM, and ROM allocations with support for both
explicit (user-specified) and automatic (compiler-assigned) addresses.
"""

from typing import Dict, Optional, Set, Tuple
from dataclasses import dataclass
from r65.compiler.hir import HIRStaticDecl, Symbol
from r65.compiler.hir.attributes import StorageKind
from r65.compiler.errors import MemoryAllocationError
from r65.compiler.codegen.type_utils import get_type_size as _get_type_size
from r65.compiler.codegen.constants import (
    LOWRAM_START, LOWRAM_END, ZEROPAGE_END, RAM_START, RAM_END
)


@dataclass
class AllocationInfo:
    """Information about an allocated memory location."""
    symbol: Symbol
    address: int
    size: int  # Size in bytes
    storage_type: str  # 'zeropage', 'ram', 'rom', 'hw'
    is_explicit: bool  # True if user-specified address
    source_info: Optional[str] = None  # Source file:line for comments


class MemoryAllocator:
    """
    Allocates and tracks memory locations for static variables.

    Handles both explicit (user-specified) and automatic (compiler-assigned)
    memory allocations with conflict detection.
    """

    def __init__(self):
        """Initialize memory allocator."""
        # Address ranges (from constants.py)
        # Low RAM is $0000-$1FFF (8KB)
        # - Zeropage ($0000-$00FF) uses direct page addressing
        # - Rest of low RAM ($0100-$1FFF) uses absolute addressing
        # - Stack can occupy any slice within low RAM
        self.lowram_start = LOWRAM_START
        self.lowram_end = LOWRAM_END
        self.zeropage_end = ZEROPAGE_END

        self.ram_start = RAM_START
        self.ram_end = RAM_END

        # Stack reservation (set via #[stack(lower, upper)])
        self.stack_lower: Optional[int] = None
        self.stack_upper: Optional[int] = None

        # Allocations: symbol name → AllocationInfo
        self.allocations: Dict[str, AllocationInfo] = {}

        # Address usage tracking (for conflict detection)
        # Single tracking for all of low RAM ($0000-$1FFF)
        self.lowram_used: Set[int] = set()  # Individual bytes
        self.ram_used: Set[Tuple[int, int]] = set()  # (start, end) ranges

    # ========================================================================
    # Size Calculation
    # ========================================================================

    def get_type_size(self, type_info) -> int:
        """
        Get size in bytes for a type.

        Args:
            type_info: Type information from HIR

        Returns:
            Size in bytes
        """
        return _get_type_size(type_info)

    # ========================================================================
    # Zero-Page Allocation (subset of low RAM with direct page addressing)
    # ========================================================================

    def _find_zeropage_fit(self, size: int) -> int:
        """
        Find first available zeropage address that fits 'size' bytes.

        Zeropage is $0000-$00FF within low RAM, using direct page addressing.

        Args:
            size: Number of contiguous bytes needed

        Returns:
            First available address

        Raises:
            Exception: If no contiguous block available
        """
        addr = self.lowram_start  # Start at $0000
        while addr + size - 1 <= self.zeropage_end:
            # Check if all bytes in range are free (shared with low RAM)
            if all(i not in self.lowram_used for i in range(addr, addr + size)):
                return addr
            addr += 1
        raise MemoryAllocationError(f"Out of zero-page space (need {size} contiguous bytes)")

    def allocate_zeropage(self, symbol: Symbol, static_decl: HIRStaticDecl,
                         explicit_addr: Optional[int] = None) -> AllocationInfo:
        """
        Allocate zero-page location ($0000-$00FF).

        Zeropage is part of low RAM but uses direct page addressing.

        Args:
            symbol: Symbol to allocate
            static_decl: Static declaration from HIR
            explicit_addr: User-specified address (or None for auto)

        Returns:
            AllocationInfo for the allocation

        Raises:
            Exception: If address out of range or no space for auto-allocation
        """
        size = self.get_type_size(static_decl.var_type)

        if explicit_addr is not None:
            # Explicit address - use as-is, no collision check
            address = explicit_addr
            is_explicit = True

            # Validate range only
            if address < 0 or address + size - 1 > self.zeropage_end:
                raise MemoryAllocationError(
                    f"Zero-page address ${address:02X} for '{symbol.name}' "
                    f"out of range ($00-$FF)"
                )
        else:
            # Auto-allocate - find next available address that fits
            address = self._find_zeropage_fit(size)
            is_explicit = False

        # Mark as used in shared low RAM tracking
        for i in range(address, address + size):
            self.lowram_used.add(i)

        # Create allocation info
        alloc = AllocationInfo(
            symbol=symbol,
            address=address,
            size=size,
            storage_type='zeropage',
            is_explicit=is_explicit,
            source_info=None
        )

        self.allocations[symbol.name] = alloc
        return alloc

    # ========================================================================
    # Stack Reservation
    # ========================================================================

    def set_stack_region(self, lower: int, upper: int):
        """
        Reserve a region for the stack.

        Args:
            lower: Lower bound of stack region (e.g., 0x1000)
            upper: Upper bound of stack region (e.g., 0x1FFF)

        The stack grows downward, so 'upper' is where S is initialized,
        and 'lower' is the minimum address the stack can reach.
        """
        if lower < self.lowram_start or upper > self.lowram_end:
            raise MemoryAllocationError(
                f"Stack region ${lower:04X}-${upper:04X} must be within "
                f"low RAM (${self.lowram_start:04X}-${self.lowram_end:04X})"
            )
        if lower > upper:
            raise MemoryAllocationError(
                f"Stack lower bound ${lower:04X} must be <= upper bound ${upper:04X}"
            )

        self.stack_lower = lower
        self.stack_upper = upper

        # Mark stack region as used in lowram
        for i in range(lower, upper + 1):
            self.lowram_used.add(i)

    # ========================================================================
    # Low RAM Allocation (above zeropage, $0100-$1FFF for auto-allocation)
    # ========================================================================

    def _find_lowram_fit(self, size: int) -> int:
        """
        Find first available low RAM address that fits 'size' bytes.

        Auto-allocation starts at $0100 (after zeropage section).
        Uses shared lowram_used tracking with zeropage.

        Args:
            size: Number of contiguous bytes needed

        Returns:
            First available address

        Raises:
            Exception: If no contiguous block available
        """
        # Start auto-allocation after zeropage ($0100+)
        addr = self.zeropage_end + 1
        while addr + size - 1 <= self.lowram_end:
            # Check if all bytes in range are free
            if all(i not in self.lowram_used for i in range(addr, addr + size)):
                return addr
            addr += 1
        raise MemoryAllocationError(f"Out of low RAM space (need {size} contiguous bytes)")

    def allocate_lowram(self, symbol: Symbol, static_decl: HIRStaticDecl,
                        explicit_addr: Optional[int] = None) -> AllocationInfo:
        """
        Allocate low RAM location.

        Auto-allocation: $0100-$1FFF (after zeropage)
        Explicit: $0000-$1FFF (full low RAM range)

        Args:
            symbol: Symbol to allocate
            static_decl: Static declaration from HIR
            explicit_addr: User-specified address (or None for auto)

        Returns:
            AllocationInfo for the allocation

        Raises:
            Exception: If address out of range or no space for auto-allocation
        """
        size = self.get_type_size(static_decl.var_type)

        if explicit_addr is not None:
            # Explicit address - use as-is, no collision check
            # Allow full low RAM range ($0000-$1FFF)
            address = explicit_addr
            is_explicit = True

            if address < self.lowram_start or address + size - 1 > self.lowram_end:
                raise MemoryAllocationError(
                    f"Low RAM address ${address:04X} for '{symbol.name}' "
                    f"out of range (${self.lowram_start:04X}-${self.lowram_end:04X})"
                )
        else:
            # Auto-allocate - find next available address (starts at $0100)
            address = self._find_lowram_fit(size)
            is_explicit = False

        # Mark as used
        for i in range(address, address + size):
            self.lowram_used.add(i)

        # Create allocation info
        alloc = AllocationInfo(
            symbol=symbol,
            address=address,
            size=size,
            storage_type='lowram',
            is_explicit=is_explicit,
            source_info=None
        )

        self.allocations[symbol.name] = alloc
        return alloc

    # ========================================================================
    # RAM Allocation
    # ========================================================================

    def _find_ram_fit(self, size: int) -> int:
        """
        Find first available RAM address that fits 'size' bytes.

        Args:
            size: Number of contiguous bytes needed

        Returns:
            First available address

        Raises:
            Exception: If no space available
        """
        address = self.ram_start

        # Find first free address that doesn't conflict
        while True:
            if address + size - 1 > self.ram_end:
                raise MemoryAllocationError(f"Out of RAM space (need {size} bytes)")

            # Check for conflicts with existing allocations
            conflict = False
            for start, end in self.ram_used:
                if not (address + size - 1 < start or address > end):
                    # Conflict found - try after this allocation
                    address = end + 1
                    conflict = True
                    break

            if not conflict:
                return address

    def allocate_ram(self, symbol: Symbol, static_decl: HIRStaticDecl,
                    explicit_addr: Optional[int] = None) -> AllocationInfo:
        """
        Allocate RAM location.

        Args:
            symbol: Symbol to allocate
            static_decl: Static declaration from HIR
            explicit_addr: User-specified address (or None for auto)

        Returns:
            AllocationInfo for the allocation

        Raises:
            Exception: If address out of range or no space for auto-allocation
        """
        size = self.get_type_size(static_decl.var_type)

        if explicit_addr is not None:
            # Explicit address - use as-is, no collision check
            address = explicit_addr
            is_explicit = True

            # Validate range only
            if address < self.ram_start or address + size - 1 > self.ram_end:
                raise MemoryAllocationError(
                    f"RAM address ${address:06X} for '{symbol.name}' "
                    f"out of range (${self.ram_start:06X}-${self.ram_end:06X})"
                )
        else:
            # Auto-allocate - find next available address that fits
            address = self._find_ram_fit(size)
            is_explicit = False

        # Mark as used
        self.ram_used.add((address, address + size - 1))

        # Create allocation info
        alloc = AllocationInfo(
            symbol=symbol,
            address=address,
            size=size,
            storage_type='ram',
            is_explicit=is_explicit,
            source_info=None
        )

        self.allocations[symbol.name] = alloc
        return alloc

    # ========================================================================
    # Hardware Register Allocation
    # ========================================================================

    def allocate_hw(self, symbol: Symbol, static_decl: HIRStaticDecl,
                   hw_addr: int) -> AllocationInfo:
        """
        Allocate hardware register (memory-mapped I/O).

        Args:
            symbol: Symbol to allocate
            static_decl: Static declaration from HIR
            hw_addr: Hardware register address

        Returns:
            AllocationInfo for the allocation
        """
        size = self.get_type_size(static_decl.var_type)

        alloc = AllocationInfo(
            symbol=symbol,
            address=hw_addr,
            size=size,
            storage_type='hw',
            is_explicit=True,
            source_info=None
        )

        self.allocations[symbol.name] = alloc
        return alloc

    # ========================================================================
    # ROM Allocation
    # ========================================================================

    def allocate_rom(self, symbol: Symbol, static_decl: HIRStaticDecl,
                    explicit_addr: Optional[int] = None) -> AllocationInfo:
        """
        Allocate ROM location (read-only data).

        Args:
            symbol: Symbol to allocate
            static_decl: Static declaration from HIR
            explicit_addr: User-specified address (or None for auto)

        Returns:
            AllocationInfo for the allocation

        Note:
            ROM allocations are handled differently - they're placed in
            the code section by the emitter. This just tracks the allocation.
        """
        size = self.get_type_size(static_decl.var_type)

        alloc = AllocationInfo(
            symbol=symbol,
            address=explicit_addr or 0,  # Will be resolved during emission
            size=size,
            storage_type='rom',
            is_explicit=explicit_addr is not None,
            source_info=None
        )

        self.allocations[symbol.name] = alloc
        return alloc

    # ========================================================================
    # Bulk Allocation
    # ========================================================================

    def allocate_all(self, static_decls: list[HIRStaticDecl]):
        """
        Allocate all static declarations in source order.

        Args:
            static_decls: List of static declarations from HIR

        Allocates addresses in the order declarations appear in source code.
        Auto-allocated variables get the next available address that fits.
        Explicit addresses are used as-is without collision checking.
        """
        for decl in static_decls:
            if decl.storage_attr and decl.storage_attr.address is not None:
                self._allocate_static(decl, explicit_addr=decl.storage_attr.address)
            else:
                self._allocate_static(decl, explicit_addr=None)

    def _allocate_static(self, static_decl: HIRStaticDecl, explicit_addr: Optional[int]):
        """
        Allocate a single static variable.

        Args:
            static_decl: Static declaration from HIR
            explicit_addr: Explicit address or None for auto
        """
        symbol = static_decl.symbol
        storage_attr = static_decl.storage_attr

        if storage_attr is None:
            # No storage attr = ROM (immutable static)
            self.allocate_rom(symbol, static_decl, explicit_addr)
            return

        storage_kind = storage_attr.storage_kind

        if storage_kind == StorageKind.ZEROPAGE:
            self.allocate_zeropage(symbol, static_decl, explicit_addr)
        elif storage_kind == StorageKind.LOWRAM:
            self.allocate_lowram(symbol, static_decl, explicit_addr)
        elif storage_kind == StorageKind.RAM:
            self.allocate_ram(symbol, static_decl, explicit_addr)
        elif storage_kind == StorageKind.HW:
            if explicit_addr is None:
                raise MemoryAllocationError(f"Hardware register '{symbol.name}' must have explicit address")
            self.allocate_hw(symbol, static_decl, explicit_addr)
        else:
            raise MemoryAllocationError(f"Unknown storage kind: {storage_kind}")

    # ========================================================================
    # Query
    # ========================================================================

    def get_allocation(self, symbol: Symbol) -> Optional[AllocationInfo]:
        """
        Get allocation info for a symbol.

        Args:
            symbol: Symbol to query

        Returns:
            AllocationInfo or None if not allocated
        """
        return self.allocations.get(symbol.name)

    def get_address(self, symbol: Symbol) -> Optional[int]:
        """
        Get address for a symbol.

        Args:
            symbol: Symbol to query

        Returns:
            Address or None if not allocated
        """
        alloc = self.allocations.get(symbol.name)
        return alloc.address if alloc else None

    def get_all_allocations(self) -> list[AllocationInfo]:
        """
        Get all allocations.

        Returns:
            List of all AllocationInfo objects
        """
        return list(self.allocations.values())

    def get_allocations_by_type(self, storage_type: str) -> list[AllocationInfo]:
        """
        Get allocations of a specific type.

        Args:
            storage_type: 'zeropage', 'ram', 'hw', or 'rom'

        Returns:
            List of AllocationInfo objects
        """
        return [a for a in self.allocations.values() if a.storage_type == storage_type]
