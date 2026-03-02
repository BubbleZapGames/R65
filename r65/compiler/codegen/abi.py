"""
ABI and stack frame abstractions for the R65 compiler.

Provides centralized definitions for:
- ABIInfo: Static calling-convention facts about a function
- StackFrameLayout: Stack frame structure and offset computations
- StackStateTracker: Runtime SP displacement tracking during codegen
"""

from dataclasses import dataclass
from typing import Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from r65.compiler.mir.nodes import MIRFunction


class ABIInfo:
    """Static ABI knowledge for a function.

    Encapsulates all calling-convention facts known before codegen starts.
    Replaces inline computations like ``3 if is_far else 2``.
    """

    __slots__ = (
        'is_far', 'is_entry', 'is_interrupt', 'is_trait_method',
        'entry_m_mode', 'preserves', 'databank_mode',
        'has_far_ptr_stack_params', 'self_far_uses_d_equals_s',
    )

    def __init__(
        self,
        is_far: bool = False,
        is_entry: bool = False,
        is_interrupt: bool = False,
        is_trait_method: bool = False,
        entry_m_mode=None,
        preserves: Tuple[str, ...] = (),
        databank_mode=None,
        has_far_ptr_stack_params: bool = False,
        self_far_uses_d_equals_s: bool = False,
    ):
        self.is_far = is_far
        self.is_entry = is_entry
        self.is_interrupt = is_interrupt
        self.is_trait_method = is_trait_method
        self.entry_m_mode = entry_m_mode
        self.preserves = preserves
        self.databank_mode = databank_mode
        self.has_far_ptr_stack_params = has_far_ptr_stack_params
        self.self_far_uses_d_equals_s = self_far_uses_d_equals_s

    @property
    def return_addr_size(self) -> int:
        """2 for near (JSR/RTS), 3 for far (JSL/RTL)."""
        return 3 if self.is_far else 2

    @property
    def accu_size(self) -> int:
        """1 in m8 mode, 2 in m16 mode."""
        from r65.compiler.typeck.processor_mode import ModeState
        return 2 if self.entry_m_mode == ModeState.M16 else 1

    def push_size(self, reg: str) -> int:
        """Bytes pushed by PHx for a given register name."""
        if reg == 'A':
            return self.accu_size
        if reg in ('X', 'Y', 'D'):
            return 2
        if reg in ('STATUS', 'DBR', 'B'):
            return 1
        raise ValueError(f"Unknown register: {reg}")

    @property
    def prologue_stack_bytes(self) -> int:
        """Total bytes pushed by prologue (DBR + preserves + far-ptr PHD).

        Does NOT include interrupt scratch saves (pushed before frame alloc)
        or the return address itself.
        """
        from r65.compiler.hir.attributes import DataBankMode

        total = 0

        # DBR management: PHB pushes 1 byte
        if self.is_far and self.databank_mode == DataBankMode.INLINE:
            total += 1

        # Register preservation pushes
        for reg in self.preserves:
            total += self.push_size(reg)

        # Far self D=S path: PHB (1 byte) + PHY (2 bytes) for self pointer on stack
        if self.self_far_uses_d_equals_s:
            total += 3

        # Far pointer stack params: PHD pushes 2 bytes
        if self.has_far_ptr_stack_params:
            total += 2

        return total

    @classmethod
    def from_mir_function(cls, mir_func: 'MIRFunction') -> 'ABIInfo':
        """Construct from MIR function metadata."""
        preserves: Tuple[str, ...] = ()
        if mir_func.preserves_attr:
            preserves = tuple(mir_func.preserves_attr.registers)

        databank_mode = None
        if mir_func.mode_attr:
            databank_mode = mir_func.mode_attr.databank

        is_interrupt = mir_func.interrupt_attr is not None

        return cls(
            is_far=mir_func.is_far,
            is_entry=mir_func.is_entry,
            is_interrupt=is_interrupt,
            is_trait_method=mir_func.is_trait_method,
            entry_m_mode=mir_func.entry_m_mode,
            preserves=preserves,
            databank_mode=databank_mode,
            has_far_ptr_stack_params=mir_func.has_far_ptr_stack_params,
            self_far_uses_d_equals_s=mir_func.self_far_uses_d_equals_s,
        )


@dataclass
class StackFrameLayout:
    """Describes a function's stack frame structure.

    Single source of truth for the frame layout.  Encodes the relationship
    between prologue bytes, local frame, outgoing arg area, and parameter
    offsets.
    """
    abi: ABIInfo
    local_frame_size: int = 0        # From slot allocator
    outgoing_arg_bytes: int = 0      # Max outgoing stack args across all calls

    @property
    def total_frame_size(self) -> int:
        """Bytes allocated by prologue TSC/SBC/TCS."""
        return self.local_frame_size + self.outgoing_arg_bytes

    @property
    def stack_base_offset(self) -> int:
        """Base offset for stack-allocated locals from SP.

        Entry functions: 1; regular: prologue_bytes + 1.
        """
        if self.abi.is_entry:
            return 1
        return self.abi.prologue_stack_bytes + 1

    def local_offset(self, slot_num: int) -> int:
        """Stack offset for a local variable slot (from SP after frame alloc)."""
        return self.stack_base_offset + self.outgoing_arg_bytes + slot_num

    def param_offset(self, base_offset: int) -> int:
        """Final stack offset for a parameter with given initial base_offset.

        base_offset comes from builder.py (return_addr_size + 1 + position).
        """
        return base_offset + self.abi.prologue_stack_bytes + self.total_frame_size

    def outgoing_arg_offset(self, arg_position: int) -> int:
        """Stack offset for writing to the outgoing arg area (1-indexed from SP)."""
        return 1 + arg_position

    @property
    def has_frame(self) -> bool:
        return self.total_frame_size > 0


class StackStateTracker:
    """Tracks runtime SP displacement during instruction emission.

    At any point: actual_SP = frame_SP - displacement.
    All stack-relative addresses need +displacement adjustment.
    """

    __slots__ = ('_displacement',)

    def __init__(self):
        self._displacement: int = 0

    @property
    def displacement(self) -> int:
        """Current SP displacement (bytes pushed since frame base)."""
        return self._displacement

    def push(self, n_bytes: int) -> None:
        """Record n_bytes pushed onto stack (PHx, PHA for args)."""
        self._displacement += n_bytes

    def pop(self, n_bytes: int) -> None:
        """Record n_bytes popped from stack (PLx, cleanup).

        Displacement can go negative temporarily when the PLD/PHD dance
        pops the saved D register before pushing call arguments. The
        subsequent PHAs will bring it back to positive territory.
        """
        self._displacement -= n_bytes

    def adjust_offset(self, static_offset: int) -> int:
        """Adjust a frame-relative offset for current SP displacement."""
        return static_offset + self._displacement

    def reset(self) -> None:
        """Reset displacement to 0 (new block boundary)."""
        self._displacement = 0
