"""
Hardware register state tracker for optimized instruction selection.

Tracks the contents and liveness of A, X, Y registers during code generation
to enable optimizations like:
- Avoiding redundant loads when a value is already in a register
- Using X/Y as scratch storage when their aliased values are dead
- Register-to-register transfers instead of memory loads
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Set, Any
from enum import Enum


class RegisterContents(Enum):
    """What kind of value a register contains."""
    UNKNOWN = "unknown"       # Contents unknown or clobbered
    VREG = "vreg"            # Contains a virtual register value
    IMMEDIATE = "immediate"   # Contains an immediate value
    PARAMETER = "parameter"   # Contains a function parameter (aliased)


@dataclass
class RegisterState:
    """
    State of a single hardware register.

    Tracks what value the register contains and whether that value
    is still needed (live).
    """
    # What kind of value is in this register
    contents: RegisterContents = RegisterContents.UNKNOWN

    # The actual value (VirtualRegister, int, or HIR Symbol)
    value: Optional[Any] = None

    # Is this value still live (needed for future instructions)?
    is_live: bool = False

    # Instruction index of last use (-1 if not tracked)
    last_use_index: int = -1

    def clear(self):
        """Mark register as clobbered/unknown."""
        self.contents = RegisterContents.UNKNOWN
        self.value = None
        self.is_live = False
        self.last_use_index = -1

    def set_vreg(self, vreg, last_use: int = -1):
        """Set register to contain a virtual register value."""
        self.contents = RegisterContents.VREG
        self.value = vreg
        self.is_live = last_use >= 0
        self.last_use_index = last_use

    def set_immediate(self, value: int):
        """Set register to contain an immediate value."""
        self.contents = RegisterContents.IMMEDIATE
        self.value = value
        self.is_live = False  # Immediates don't have liveness
        self.last_use_index = -1

    def set_parameter(self, symbol, last_use: int = -1):
        """Set register to contain a parameter value."""
        self.contents = RegisterContents.PARAMETER
        self.value = symbol
        self.is_live = last_use >= 0
        self.last_use_index = last_use

    def is_free(self) -> bool:
        """Check if register is available for scratch use."""
        return not self.is_live or self.contents == RegisterContents.UNKNOWN


@dataclass
class HardwareRegisterTracker:
    """
    Tracks hardware register state during instruction selection.

    Enables optimizations by knowing:
    - What value each register currently holds
    - Whether that value is still needed
    - When registers become free for scratch use

    Usage:
        tracker = HardwareRegisterTracker()
        tracker.initialize_from_parameters(mir_func)

        # During instruction selection:
        if tracker.contains_vreg('X', some_vreg):
            emit TXA instead of loading from memory

        if tracker.is_free('X'):
            use X as scratch register
    """

    # State for each hardware register
    a: RegisterState = field(default_factory=RegisterState)
    x: RegisterState = field(default_factory=RegisterState)
    y: RegisterState = field(default_factory=RegisterState)

    # Maps register name to state for convenient access
    _states: Dict[str, RegisterState] = field(default_factory=dict, init=False)

    # Current instruction index (for liveness tracking)
    current_index: int = 0

    def __post_init__(self):
        """Initialize register state mapping."""
        self._states = {
            'A': self.a,
            'X': self.x,
            'Y': self.y,
        }

    def get_state(self, reg_name: str) -> Optional[RegisterState]:
        """Get state for a register by name."""
        return self._states.get(reg_name.upper())

    def initialize_from_parameters(self, mir_func, vreg_last_uses: Dict[Any, int] = None):
        """
        Initialize register state from function parameters.

        For parameters with register aliases (e.g., `idx @ X`), marks
        the register as containing that parameter value.

        Args:
            mir_func: MIR function being generated
            vreg_last_uses: Map from VirtualRegister/Symbol to last use instruction index
        """
        vreg_last_uses = vreg_last_uses or {}

        # Reset all registers
        self.a.clear()
        self.x.clear()
        self.y.clear()
        self.current_index = 0

        # Check alias tracker for parameter bindings
        if hasattr(mir_func, 'alias_tracker') and mir_func.alias_tracker:
            for symbol_id, alias in mir_func.alias_tracker.aliases.items():
                reg_name = alias.hardware_reg.name.upper()
                state = self.get_state(reg_name)
                if state:
                    # Find last use for this parameter
                    # Use id() since Symbol is not hashable
                    last_use = vreg_last_uses.get(id(alias.symbol), -1)
                    state.set_parameter(alias.symbol, last_use)

    def advance_to(self, instruction_index: int):
        """
        Advance to a new instruction index.

        Updates liveness - registers become free after their last use.

        Args:
            instruction_index: Current instruction being generated
        """
        self.current_index = instruction_index

        # Update liveness for each register
        for state in self._states.values():
            if state.is_live and state.last_use_index >= 0:
                if instruction_index > state.last_use_index:
                    state.is_live = False

    def mark_clobbered(self, reg_name: str):
        """
        Mark a register as clobbered (contents unknown).

        Called when an instruction modifies a register.

        Args:
            reg_name: Register name ('A', 'X', or 'Y')
        """
        state = self.get_state(reg_name)
        if state:
            state.clear()

    def mark_contains_vreg(self, reg_name: str, vreg, last_use: int = -1):
        """
        Mark that a register now contains a virtual register value.

        Args:
            reg_name: Register name
            vreg: VirtualRegister that was loaded
            last_use: Last instruction index where vreg is used
        """
        state = self.get_state(reg_name)
        if state:
            state.set_vreg(vreg, last_use)

    def mark_contains_immediate(self, reg_name: str, value: int):
        """
        Mark that a register now contains an immediate value.

        Args:
            reg_name: Register name
            value: Immediate value
        """
        state = self.get_state(reg_name)
        if state:
            state.set_immediate(value)

    def contains_vreg(self, reg_name: str, vreg) -> bool:
        """
        Check if a register currently contains a specific virtual register.

        Args:
            reg_name: Register name to check
            vreg: VirtualRegister to look for

        Returns:
            True if register contains that vreg value
        """
        state = self.get_state(reg_name)
        if not state:
            return False
        if state.contents != RegisterContents.VREG:
            return False
        # Compare by id for VirtualRegister
        if hasattr(vreg, 'id') and hasattr(state.value, 'id'):
            return vreg.id == state.value.id
        return state.value == vreg

    def contains_parameter(self, reg_name: str, symbol) -> bool:
        """
        Check if a register currently contains a specific parameter.

        Args:
            reg_name: Register name to check
            symbol: Parameter symbol to look for

        Returns:
            True if register contains that parameter value
        """
        state = self.get_state(reg_name)
        if not state:
            return False
        if state.contents != RegisterContents.PARAMETER:
            return False
        return id(state.value) == id(symbol)

    def contains_immediate(self, reg_name: str, value: int) -> bool:
        """
        Check if a register currently contains a specific immediate.

        Args:
            reg_name: Register name to check
            value: Immediate value to look for

        Returns:
            True if register contains that immediate value
        """
        state = self.get_state(reg_name)
        if not state:
            return False
        return state.contents == RegisterContents.IMMEDIATE and state.value == value

    def is_free(self, reg_name: str) -> bool:
        """
        Check if a register is free for scratch use.

        A register is free if:
        - Its contents are unknown, OR
        - Its value is no longer live (past last use)

        Args:
            reg_name: Register name to check

        Returns:
            True if register can be used as scratch
        """
        state = self.get_state(reg_name)
        return state.is_free() if state else True

    def find_register_containing(self, vreg) -> Optional[str]:
        """
        Find which register (if any) contains a virtual register.

        Args:
            vreg: VirtualRegister to look for

        Returns:
            Register name ('A', 'X', 'Y') or None
        """
        for reg_name, state in self._states.items():
            if state.contents == RegisterContents.VREG:
                if hasattr(vreg, 'id') and hasattr(state.value, 'id'):
                    if vreg.id == state.value.id:
                        return reg_name
                elif state.value == vreg:
                    return reg_name
        return None

    def get_free_index_register(self) -> Optional[str]:
        """
        Get a free index register (X or Y) for scratch use.

        Returns:
            'X' or 'Y' if one is free, None otherwise
        """
        if self.is_free('X'):
            return 'X'
        if self.is_free('Y'):
            return 'Y'
        return None


def compute_vreg_last_uses(mir_func) -> Dict[int, int]:
    """
    Compute the last use instruction index for each virtual register.

    Scans all blocks and instructions to find where each vreg is last used.

    Args:
        mir_func: MIR function to analyze

    Returns:
        Dictionary mapping id(VirtualRegister) or id(Symbol) to last use instruction index
    """
    from r65.compiler.mir.nodes import VirtualRegister, HardwareRegister
    from r65.compiler.mir.liveness import LivenessAnalyzer

    last_uses: Dict[int, int] = {}  # id -> last instruction index

    # Create a single analyzer for the function
    analyzer = LivenessAnalyzer(mir_func)

    # Linear scan to find last use of each vreg
    instr_index = 0

    for block_id in sorted(mir_func.blocks.keys()):
        block = mir_func.blocks[block_id]
        for instr in block.instructions:
            # Get uses from this instruction
            uses = analyzer._get_uses(instr)

            for use in uses:
                if isinstance(use, VirtualRegister):
                    # Use vreg.id as key since VirtualRegister has __hash__
                    last_uses[use.id] = instr_index
                elif isinstance(use, HardwareRegister):
                    # Hardware registers might be aliased to symbols
                    # Track by register name
                    pass

            instr_index += 1

    # Also track hardware register aliases (parameters like `idx @ X`)
    # These need to be tracked by the symbol they're aliased to
    if hasattr(mir_func, 'alias_tracker') and mir_func.alias_tracker:
        # For each aliased parameter, find its last use
        # by scanning instructions that use the hardware register
        instr_index = 0
        for block_id in sorted(mir_func.blocks.keys()):
            block = mir_func.blocks[block_id]
            for instr in block.instructions:
                uses = analyzer._get_uses(instr)
                for use in uses:
                    if isinstance(use, HardwareRegister):
                        # Check if this hw reg is aliased
                        for symbol_id, alias in mir_func.alias_tracker.aliases.items():
                            if alias.hardware_reg.name == use.name:
                                # This instruction uses the aliased parameter
                                last_uses[id(alias.symbol)] = instr_index

                instr_index += 1

    return last_uses
