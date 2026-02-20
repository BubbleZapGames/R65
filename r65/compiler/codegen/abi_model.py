"""
ABI model abstraction for the R65 compiler.

Defines selectable ABI policies that control calling convention decisions:
- Default: Traditional stack-based parameters with TSC/SBC/TCS frame allocation
- FixedStack: Zero-frame model with hw registers + scratch only, PHB-per-byte frames
"""

import sys
from abc import ABC, abstractmethod
from enum import Enum


class ABIKind(Enum):
    DEFAULT = "Default"
    FIXED_STACK = "FixedStack"


class ABIModel(ABC):
    """Global ABI policy object controlling calling convention decisions.

    This is NOT per-function state — it's a compile-wide policy.
    Per-function ABI facts remain in ABIInfo from abi.py.
    """

    def __init__(self, kind: ABIKind):
        self.kind = kind

    @abstractmethod
    def run_param_analysis(self, mir_program, scratch_pool, disable_scratch_params: bool):
        """Run the parameter promotion pass appropriate for this ABI."""

    @abstractmethod
    def compute_outgoing_args(self, mir_program, compute_fn):
        """Compute outgoing arg bytes, or no-op if ABI doesn't use them."""

    @abstractmethod
    def emit_frame_alloc(self, emit_instr, frame_size: int, force_direct_stack: bool):
        """Emit stack frame allocation instructions."""

    @property
    @abstractmethod
    def max_pla_dealloc_size(self) -> int:
        """Max frame bytes for PLA-per-byte dealloc."""

    def emit_frame_dealloc(self, selector, frame_size: int, return_count: int):
        """Emit frame deallocation (epilogue counterpart to emit_frame_alloc).

        For small frames (<= max_pla_dealloc_size): PLA-per-byte.
        For large frames: TSC/CLC/ADC/TCS preserving A.
        """
        if frame_size <= 0:
            return
        if frame_size <= self.max_pla_dealloc_size:
            selector.emit_pla_frame_dealloc(frame_size, return_count)
        else:
            current_mode = selector.parent.emitter.get_accu_mode()
            selector.emit_sp_adjust_preserving_a(frame_size, return_count, current_mode)

    # -- shared helpers --

    def _emit_phb_alloc(self, emit_instr, frame_size: int):
        from r65.compiler.codegen.opcodes import Opcode
        for _ in range(frame_size):
            emit_instr(Opcode.PHB, comment=f"Allocate frame ({frame_size} bytes)")

    def _emit_tsc_alloc(self, emit_instr, frame_size: int):
        from r65.compiler.codegen.opcodes import Opcode
        from r65.compiler.codegen.asm_nodes import Immediate
        from r65.compiler.codegen.constants import M_FLAG
        emit_instr(Opcode.REP_IMMEDIATE, Immediate(M_FLAG), "16-bit A for frame setup")
        emit_instr(Opcode.TSC, comment="Get stack pointer")
        emit_instr(Opcode.SEC, comment="Set carry for subtraction")
        emit_instr(Opcode.SBC_IMMEDIATE, Immediate(frame_size), f"Allocate {frame_size} bytes for locals")
        emit_instr(Opcode.TCS, comment="Update stack pointer")
        emit_instr(Opcode.SEP_IMMEDIATE, Immediate(M_FLAG), "Restore 8-bit A")

    # -- call argument preamble --

    def emit_call_args(self, selector, instr, pre_arg_stack_adj=0) -> int:
        """Emit call arguments. Returns stack_bytes_pushed (>0 only in PHA mode).

        Concrete on the base class — both ABIs share identical logic
        (FixedStack simply never has STACK-mechanism args).

        Args:
            selector: CallInstructionSelector instance
            instr: Call MIR instruction
            pre_arg_stack_adj: Stack offset adjustment from pre-arg operations
        """
        from r65.compiler.mir.nodes import ArgumentMechanism
        from r65.compiler.codegen.register_alloc import LocationKind
        from r65.compiler.codegen.type_utils import get_type_size

        stack_args = [arg for arg in instr.args if arg.mechanism == ArgumentMechanism.STACK]
        other_args = [arg for arg in instr.args if arg.mechanism != ArgumentMechanism.STACK]

        spill_offset = selector.get_current_spill_offset()
        stack_bytes_pushed = 0

        if spill_offset > 0 and stack_args:
            for arg in reversed(stack_args):
                arg_loc = selector.parent._get_operand_location(arg.value)

                if arg_loc.kind == LocationKind.STACK and pre_arg_stack_adj != 0:
                    arg_loc = selector.parent._offset_location(arg_loc, pre_arg_stack_adj)

                arg_size = 1
                if arg.param_type is not None:
                    arg_size = get_type_size(arg.param_type)
                elif hasattr(arg.value, 'type_info') and arg.value.type_info:
                    arg_size = get_type_size(arg.value.type_info)

                selector.emit_pha_stack_argument(arg, arg_loc, arg_size)
                stack_bytes_pushed += arg_size
                selector.region_state.stack_tracker.push(arg_size)
        elif stack_args:
            outgoing_offset = 1
            for arg in stack_args:
                arg_loc = selector.parent._get_operand_location(arg.value)

                if arg_loc.kind == LocationKind.STACK and pre_arg_stack_adj != 0:
                    arg_loc = selector.parent._offset_location(arg_loc, pre_arg_stack_adj)

                selector.emit_outgoing_stack_argument(arg, arg_loc, outgoing_offset)

                arg_size = 1
                if arg.param_type is not None:
                    arg_size = get_type_size(arg.param_type)
                elif hasattr(arg.value, 'type_info') and arg.value.type_info:
                    arg_size = get_type_size(arg.value.type_info)
                outgoing_offset += arg_size

        sorted_other_args = sorted(other_args, key=selector.arg_sort_key)
        for arg in sorted_other_args:
            arg_loc = selector.parent._get_operand_location(arg.value)

            if arg_loc.kind == LocationKind.STACK and pre_arg_stack_adj != 0:
                arg_loc = selector.parent._offset_location(arg_loc, pre_arg_stack_adj)

            if arg.mechanism == ArgumentMechanism.REGISTER:
                selector.emit_register_argument(arg, arg_loc)

            elif arg.mechanism == ArgumentMechanism.VARIABLE:
                selector.emit_variable_argument(arg, arg_loc)

            elif arg.mechanism == ArgumentMechanism.SCRATCH_PARAM:
                selector.emit_scratch_param_argument(arg, arg_loc)

        return stack_bytes_pushed

    def emit_trait_dispatch_args(self, selector, instr) -> int:
        """Emit trait dispatch arguments. Returns stack_bytes_pushed.

        Concrete on the base class — both ABIs share identical logic.

        Args:
            selector: CallInstructionSelector instance
            instr: TraitDispatch MIR instruction
        """
        from r65.compiler.mir.nodes import ArgumentMechanism
        from r65.compiler.codegen.register_alloc import LocationKind
        from r65.compiler.codegen.type_utils import get_type_size

        stack_args = [arg for arg in instr.args if arg.mechanism == ArgumentMechanism.STACK]
        self_y_arg = None
        for arg in instr.args:
            if arg.mechanism == ArgumentMechanism.SELF_Y:
                self_y_arg = arg
                break

        spill_offset = selector.get_current_spill_offset()
        stack_bytes_pushed = 0

        if spill_offset > 0 and stack_args:
            for arg in reversed(stack_args):
                arg_loc = selector.parent._get_operand_location(arg.value)
                arg_size = 1
                if arg.param_type is not None:
                    arg_size = get_type_size(arg.param_type)
                elif hasattr(arg.value, 'type_info') and arg.value.type_info:
                    arg_size = get_type_size(arg.value.type_info)
                selector.emit_pha_stack_argument(arg, arg_loc, arg_size)
                stack_bytes_pushed += arg_size
                selector.region_state.stack_tracker.push(arg_size)
        elif stack_args:
            outgoing_offset = 1
            for arg in stack_args:
                arg_loc = selector.parent._get_operand_location(arg.value)
                selector.emit_outgoing_stack_argument(arg, arg_loc, outgoing_offset)
                arg_size = 1
                if arg.param_type is not None:
                    arg_size = get_type_size(arg.param_type)
                elif hasattr(arg.value, 'type_info') and arg.value.type_info:
                    arg_size = get_type_size(arg.value.type_info)
                outgoing_offset += arg_size

        if self_y_arg is not None:
            selector.load_y_with_self(self_y_arg, 0)

        return stack_bytes_pushed

    def __repr__(self):
        return f"ABIModel({self.kind.value})"


class ABIDefault(ABIModel):
    def __init__(self):
        super().__init__(ABIKind.DEFAULT)

    def run_param_analysis(self, mir_program, scratch_pool, disable_scratch_params: bool):
        if not disable_scratch_params:
            from r65.compiler.analysis.scratch_params import analyze_scratch_params
            analyze_scratch_params(mir_program, scratch_pool)

    def compute_outgoing_args(self, mir_program, compute_fn):
        compute_fn(mir_program)

    def emit_frame_alloc(self, emit_instr, frame_size: int, force_direct_stack: bool):
        if frame_size <= 0:
            return
        if frame_size <= 4 and not force_direct_stack:
            self._emit_phb_alloc(emit_instr, frame_size)
        else:
            self._emit_tsc_alloc(emit_instr, frame_size)

    @property
    def max_pla_dealloc_size(self) -> int:
        return 4


class ABIFixedStack(ABIModel):
    def __init__(self):
        super().__init__(ABIKind.FIXED_STACK)

    def run_param_analysis(self, mir_program, scratch_pool, disable_scratch_params: bool):
        from r65.compiler.analysis.fixedstack_params import promote_all_stack_params
        promote_all_stack_params(mir_program, scratch_pool)

    def compute_outgoing_args(self, mir_program, compute_fn):
        pass  # no outgoing arg area in FixedStack

    def emit_frame_alloc(self, emit_instr, frame_size: int, force_direct_stack: bool):
        if frame_size <= 0:
            return
        if force_direct_stack:
            self._emit_tsc_alloc(emit_instr, frame_size)
        else:
            self._emit_phb_alloc(emit_instr, frame_size)

    @property
    def max_pla_dealloc_size(self) -> int:
        return sys.maxsize


# Singleton instances
ABI_DEFAULT = ABIDefault()
ABI_FIXED_STACK = ABIFixedStack()


def abi_model_from_string(name: str) -> ABIModel:
    """Create ABIModel from CLI string argument.

    Args:
        name: "Default" or "FixedStack"

    Returns:
        Corresponding ABIModel instance

    Raises:
        ValueError: If name is not recognized
    """
    if name == "Default":
        return ABI_DEFAULT
    elif name == "FixedStack":
        return ABI_FIXED_STACK
    else:
        raise ValueError(f"Unknown ABI model: {name!r}. Expected 'Default' or 'FixedStack'.")
