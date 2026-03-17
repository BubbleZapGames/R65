# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Selector context for instruction selection composition.

Provides shared state and services for all selector components,
enabling loose coupling and easier composition of selectors.
"""

from typing import TYPE_CHECKING, Callable, Optional

from r65.compiler.codegen.location_resolver import LocationResolver, default_resolver
from r65.compiler.codegen.register_alloc import RegisterAllocator, PhysicalLocation

if TYPE_CHECKING:
    from r65.compiler.codegen.emitter import AssemblyEmitter
    from r65.compiler.codegen.memory_alloc import MemoryAllocator
    from r65.compiler.mir.nodes import MIRFunction


class SelectorContext:
    """
    Shared context for instruction selectors.

    Provides access to:
    - Assembly emitter for code generation
    - Register allocator for virtual register mapping
    - Memory allocator for static variables
    - Location resolver for addressing mode selection
    - Current function context

    This allows selectors to be composed without tight coupling
    to the main InstructionSelector class.
    """

    def __init__(
        self,
        emitter: 'AssemblyEmitter',
        register_allocator: RegisterAllocator,
        memory_allocator: 'MemoryAllocator',
        resolver: LocationResolver = None,
        current_function: 'MIRFunction' = None
    ):
        """
        Initialize selector context.

        Args:
            emitter: Assembly emitter for code generation
            register_allocator: Register allocator for virtual registers
            memory_allocator: Memory allocator for static variables
            resolver: Location resolver (uses default if None)
            current_function: Current MIR function being generated
        """
        self.emitter = emitter
        self.reg_alloc = register_allocator
        self.mem_alloc = memory_allocator
        self.resolver = resolver or default_resolver
        self.current_function = current_function

        # Callbacks for A register state tracking
        self._mark_a_modified_callback: Optional[Callable[[], None]] = None
        self._get_operand_location_callback: Optional[Callable] = None

    def set_a_modified_callback(self, callback: Callable[[], None]):
        """Set callback for marking A register as modified."""
        self._mark_a_modified_callback = callback

    def set_operand_location_callback(self, callback: Callable):
        """Set callback for getting operand locations."""
        self._get_operand_location_callback = callback

    def mark_a_modified(self):
        """Notify that the A register has been modified."""
        if self._mark_a_modified_callback:
            self._mark_a_modified_callback()

    def get_operand_location(self, operand) -> PhysicalLocation:
        """Get the physical location for an operand."""
        if self._get_operand_location_callback:
            return self._get_operand_location_callback(operand)
        raise NotImplementedError("Operand location callback not set")

    @property
    def is_far_function(self) -> bool:
        """Check if current function is a far function."""
        return self.current_function and self.current_function.is_far

    @property
    def function_mode(self):
        """Get the mode attribute of current function."""
        if self.current_function:
            return self.current_function.mode_attr
        return None


class ComposableSelector:
    """
    Base class for selectors that use composition over inheritance.

    Selectors derived from this class receive their dependencies
    through the SelectorContext rather than a parent reference.
    """

    def __init__(self, context: SelectorContext):
        """
        Initialize with context.

        Args:
            context: Shared selector context
        """
        self._context = context

    @property
    def emitter(self):
        """Get the assembly emitter."""
        return self._context.emitter

    @property
    def resolver(self) -> LocationResolver:
        """Get the location resolver."""
        return self._context.resolver

    @property
    def reg_alloc(self) -> RegisterAllocator:
        """Get the register allocator."""
        return self._context.reg_alloc

    @property
    def mem_alloc(self):
        """Get the memory allocator."""
        return self._context.mem_alloc

    def _mark_a_modified(self):
        """Notify that A register was modified."""
        self._context.mark_a_modified()

    def _get_operand_location(self, operand) -> PhysicalLocation:
        """Get physical location for an operand."""
        return self._context.get_operand_location(operand)
