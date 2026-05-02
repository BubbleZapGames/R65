# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
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
        'far_ptr_strategy', 'skip_dbr_inline',
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
        far_ptr_strategy=None,
        skip_dbr_inline: bool = False,
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
        self.far_ptr_strategy = far_ptr_strategy
        self.skip_dbr_inline = skip_dbr_inline

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
    def pre_frame_prologue_bytes(self) -> int:
        """Bytes pushed by prologue BEFORE frame allocation.

        These pushes sit deeper in the stack than locals (at higher ,S offsets),
        so they must NOT shift the local stack base offset. They DO contribute to
        param offsets (params are still deeper than these pushes).

        Currently only PHB+PHY for self_far_uses_d_equals_s.
        """
        return 3 if self.self_far_uses_d_equals_s else 0

    @property
    def post_frame_prologue_bytes(self) -> int:
        """Bytes pushed by prologue AFTER frame allocation.

        These pushes sit on top of locals (at lower ,S offsets) and shift the
        local stack base offset. Includes DBR inline, preserves, and the
        post-frame far-ptr PHD/PHB.
        """
        from r65.compiler.hir.attributes import DataBankMode

        total = 0

        # DBR management: PHB pushes 1 byte (skipped when no bank-dependent access)
        if self.is_far and self.databank_mode == DataBankMode.INLINE and not self.skip_dbr_inline:
            total += 1

        # Register preservation pushes
        for reg in self.preserves:
            total += self.push_size(reg)

        # Far pointer stack params: PHD (2 bytes) for D=S, PHB (1 byte) for SET_DBR
        if self.has_far_ptr_stack_params:
            from r65.compiler.mir.nodes import FarPtrStrategy
            if self.far_ptr_strategy == FarPtrStrategy.SET_DBR:
                total += 1  # PHB
            else:
                total += 2  # PHD (D_EQUALS_S or no strategy set yet)

        return total

    @property
    def prologue_stack_bytes(self) -> int:
        """Total bytes pushed by prologue (DBR + preserves + far-ptr PHD).

        Does NOT include interrupt scratch saves (pushed before frame alloc)
        or the return address itself.
        """
        return self.pre_frame_prologue_bytes + self.post_frame_prologue_bytes

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
            far_ptr_strategy=mir_func.far_ptr_strategy,
            skip_dbr_inline=getattr(mir_func, '_skip_dbr_inline', False),
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

        Locals are shifted ONLY by prologue bytes pushed AFTER frame allocation
        (post_frame_prologue_bytes). Pre-frame pushes (e.g., PHB+PHY for
        far-self D=S) sit deeper than the frame and don't shift locals.

        Entry functions: 1; regular: post_frame_prologue_bytes + 1.
        """
        if self.abi.is_entry:
            return 1
        return self.abi.post_frame_prologue_bytes + 1

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
