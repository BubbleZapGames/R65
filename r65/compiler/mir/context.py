"""
Lowering context: shared mutable state for MIR lowering.

Separates state management from transformation logic to enable
modular lowerer classes.
"""

from dataclasses import dataclass, field
from typing import Any, Optional, Dict, List, Tuple, TYPE_CHECKING

from r65.compiler.mir.nodes import (
    MIRFunction, BasicBlock, MIRInstruction,
    VirtualRegister, HardwareRegister, MemoryLocation,
)
from r65.compiler.mir.cfg import CFGBuilder
from r65.compiler.typeck.processor_mode import ProcessorMode

if TYPE_CHECKING:
    from r65.compiler.hir import HIRFunctionDecl


@dataclass
class LoweringContext:
    """
    Shared mutable state for HIR → MIR lowering.

    This context is passed to all lowerer classes, providing:
    - Current lowering position (function, block)
    - Symbol tables and mappings
    - Control flow state (loop stack)
    - Processor mode tracking
    """

    # ========================================================================
    # Current Lowering Position
    # ========================================================================

    current_function: Optional[MIRFunction] = None
    current_block: Optional[BasicBlock] = None
    cfg_builder: Optional[CFGBuilder] = None

    # ========================================================================
    # Processor Mode
    # ========================================================================

    current_mode: ProcessorMode = field(default_factory=ProcessorMode.default)

    # ========================================================================
    # Symbol Tables
    # ========================================================================

    # id(Symbol) → VirtualRegister mapping for current function
    symbol_to_vreg: Dict[int, VirtualRegister] = field(default_factory=dict)

    # Set of symbol ids in symbol_to_vreg that came from explicit memory
    # locations (statics).  Used by _invalidate_memloc_cache() to quickly
    # find which cache entries to evict at branch merge points.
    memloc_cached_symbols: set = field(default_factory=set)

    # Function name → HIRFunctionDecl mapping (for looking up during calls)
    function_decls: Dict[str, 'HIRFunctionDecl'] = field(default_factory=dict)

    # ========================================================================
    # Control Flow State
    # ========================================================================

    # Loop stack for break/continue: (continue_target, break_target, label, result_vreg_or_None)
    loop_stack: List[Tuple[int, int, Optional[str], Any]] = field(default_factory=list)

    # ========================================================================
    # Program State
    # ========================================================================

    # Track if we generated __init_start() function
    has_init_start: bool = False

    # Current source location for error reporting
    current_source_loc: Any = None

    # ========================================================================
    # Convenience Methods
    # ========================================================================

    def alloc_vreg(self, type_info, name: str = None) -> VirtualRegister:
        """Allocate a new virtual register in the current function."""
        return self.current_function.vreg_allocator.alloc(type_info, name)

    def get_alias(self, symbol) -> Optional[HardwareRegister]:
        """Get hardware register alias for a symbol, if any."""
        return self.current_function.alias_tracker.get_alias(symbol)

    def add_alias(self, symbol, hw_reg: HardwareRegister):
        """Add a hardware register alias for a symbol."""
        self.current_function.alias_tracker.add_alias(symbol, hw_reg)

    def push_loop(self, continue_target: int, break_target: int, label: Optional[str] = None, result_vreg=None):
        """Push loop targets onto the stack."""
        self.loop_stack.append((continue_target, break_target, label, result_vreg))

    def pop_loop(self):
        """Pop loop targets from the stack."""
        return self.loop_stack.pop()

    def get_loop_targets(self):
        """Get current loop's (continue_target, break_target, label, result_vreg)."""
        if not self.loop_stack:
            raise RuntimeError("No active loop for break/continue")
        return self.loop_stack[-1]

    # ========================================================================
    # Function Setup/Teardown
    # ========================================================================

    def begin_function(self, mir_func: MIRFunction):
        """Initialize context for lowering a new function."""
        self.current_function = mir_func
        self.cfg_builder = CFGBuilder(mir_func)
        self.symbol_to_vreg.clear()
        self.memloc_cached_symbols.clear()
        self.loop_stack.clear()

    def end_function(self):
        """Finalize function lowering."""
        self.current_function.exit_block_ids = self.cfg_builder.find_exit_blocks()

    def set_current_block(self, block: BasicBlock):
        """Set the current block for instruction emission."""
        self.current_block = block

    # ========================================================================
    # Block Creation
    # ========================================================================

    def new_block(self) -> BasicBlock:
        """Create a new basic block and add to current function."""
        block = self.cfg_builder.new_block()
        self.current_function.blocks[block.block_id] = block
        return block

    def add_cfg_edge(self, from_block: BasicBlock, to_block: BasicBlock):
        """Add an edge in the control flow graph."""
        self.cfg_builder.add_edge(from_block, to_block)
