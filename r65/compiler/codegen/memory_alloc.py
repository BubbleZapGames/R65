"""
Memory allocation: tracks and allocates memory locations.

Manages zero-page, RAM, and ROM allocations with support for both
explicit (user-specified) and automatic (compiler-assigned) addresses.
"""

from typing import Dict, Optional, Set, Tuple
from dataclasses import dataclass
from r65.compiler.hir import HIRStaticDecl, Symbol
from r65.compiler.hir.attributes import StorageKind


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
        # Address ranges
        self.zeropage_start = 0x10  # Reserve $00-$0F for system
        self.zeropage_end = 0xFF
        self.zeropage_next = self.zeropage_start

        self.ram_start = 0x7E0000  # SNES Work RAM starts at $7E0000
        self.ram_end = 0x7FFFFF
        self.ram_next = self.ram_start

        # Allocations: symbol name → AllocationInfo
        self.allocations: Dict[str, AllocationInfo] = {}

        # Address usage tracking (for conflict detection)
        self.zeropage_used: Set[int] = set()  # Individual bytes
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
        # Get type name
        if hasattr(type_info, 'name'):
            type_name = type_info.name
        else:
            type_name = str(type_info)

        # Basic types
        if type_name in ('u8', 'i8', 'bool'):
            return 1
        elif type_name in ('u16', 'i16'):
            return 2

        # Array types
        if hasattr(type_info, 'element_type') and hasattr(type_info, 'size'):
            element_size = self.get_type_size(type_info.element_type)
            return element_size * type_info.size

        # Struct types
        if hasattr(type_info, 'fields'):
            total = 0
            for field in type_info.fields:
                total += self.get_type_size(field.field_type)
            return total

        # Pointer types
        if type_name.startswith('near<'):
            return 2  # 16-bit pointer
        elif type_name.startswith('far<'):
            return 3  # 24-bit pointer

        # Default: assume 1 byte
        return 1

    # ========================================================================
    # Zero-Page Allocation
    # ========================================================================

    def allocate_zeropage(self, symbol: Symbol, static_decl: HIRStaticDecl,
                         explicit_addr: Optional[int] = None) -> AllocationInfo:
        """
        Allocate zero-page location.

        Args:
            symbol: Symbol to allocate
            static_decl: Static declaration from HIR
            explicit_addr: User-specified address (or None for auto)

        Returns:
            AllocationInfo for the allocation

        Raises:
            Exception: If address conflicts or out of range
        """
        # Get size
        size = self.get_type_size(static_decl.var_type)

        # Determine address
        if explicit_addr is not None:
            # Explicit address
            address = explicit_addr
            is_explicit = True

            # Validate range
            if address < 0 or address + size - 1 > self.zeropage_end:
                raise Exception(
                    f"Zero-page address ${address:02X} for '{symbol.name}' "
                    f"out of range ($00-$FF)"
                )

            # Check for conflicts
            for i in range(address, address + size):
                if i in self.zeropage_used:
                    raise Exception(
                        f"Zero-page address ${i:02X} already allocated "
                        f"(conflict with '{symbol.name}')"
                    )
        else:
            # Auto-allocate
            address = self.zeropage_next
            is_explicit = False

            # Check if we have space
            if address + size - 1 > self.zeropage_end:
                raise Exception(
                    f"Out of zero-page space for '{symbol.name}' "
                    f"(need {size} bytes, only {self.zeropage_end - address + 1} available)"
                )

            # Advance next pointer
            self.zeropage_next = address + size

        # Mark as used
        for i in range(address, address + size):
            self.zeropage_used.add(i)

        # Create allocation info
        alloc = AllocationInfo(
            symbol=symbol,
            address=address,
            size=size,
            storage_type='zeropage',
            is_explicit=is_explicit,
            source_info=None  # Will be filled in later
        )

        self.allocations[symbol.name] = alloc
        return alloc

    # ========================================================================
    # RAM Allocation
    # ========================================================================

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
            Exception: If address conflicts or out of range
        """
        # Get size
        size = self.get_type_size(static_decl.var_type)

        # Determine address
        if explicit_addr is not None:
            # Explicit address
            address = explicit_addr
            is_explicit = True

            # Validate range
            if address < self.ram_start or address + size - 1 > self.ram_end:
                raise Exception(
                    f"RAM address ${address:06X} for '{symbol.name}' "
                    f"out of range (${self.ram_start:06X}-${self.ram_end:06X})"
                )

            # Check for conflicts
            for start, end in self.ram_used:
                if not (address + size - 1 < start or address > end):
                    raise Exception(
                        f"RAM address ${address:06X} conflicts with existing allocation "
                        f"(${start:06X}-${end:06X})"
                    )
        else:
            # Auto-allocate - find first free spot
            address = self.ram_next
            is_explicit = False

            # Find first free address that doesn't conflict
            while True:
                # Check if we're out of space
                if address + size - 1 > self.ram_end:
                    raise Exception(
                        f"Out of RAM space for '{symbol.name}' "
                        f"(need {size} bytes)"
                    )

                # Check for conflicts with existing allocations
                conflict = False
                for start, end in self.ram_used:
                    if not (address + size - 1 < start or address > end):
                        # Conflict found - try after this allocation
                        address = end + 1
                        conflict = True
                        break

                if not conflict:
                    # Found free spot
                    break

            # Update next pointer for subsequent allocations
            self.ram_next = address + size

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
        Allocate all static declarations.

        Args:
            static_decls: List of static declarations from HIR

        Processes in order:
        1. Explicit allocations first (to reserve addresses)
        2. Auto allocations second (to fill gaps)
        """
        # Separate explicit and auto allocations
        explicit = []
        auto = []

        for decl in static_decls:
            if decl.storage_attr and decl.storage_attr.address is not None:
                explicit.append(decl)
            else:
                auto.append(decl)

        # Allocate explicit first
        for decl in explicit:
            self._allocate_static(decl, explicit_addr=decl.storage_attr.address)

        # Then auto-allocate
        for decl in auto:
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

        if not storage_attr:
            # No storage attribute - default to RAM
            self.allocate_ram(symbol, static_decl, explicit_addr)
            return

        storage_kind = storage_attr.storage_kind

        if storage_kind == StorageKind.ZEROPAGE:
            self.allocate_zeropage(symbol, static_decl, explicit_addr)
        elif storage_kind == StorageKind.RAM:
            self.allocate_ram(symbol, static_decl, explicit_addr)
        elif storage_kind == StorageKind.HW:
            if explicit_addr is None:
                raise Exception(f"Hardware register '{symbol.name}' must have explicit address")
            self.allocate_hw(symbol, static_decl, explicit_addr)
        elif storage_kind == StorageKind.ROM:
            self.allocate_rom(symbol, static_decl, explicit_addr)
        else:
            raise Exception(f"Unknown storage kind: {storage_kind}")

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
