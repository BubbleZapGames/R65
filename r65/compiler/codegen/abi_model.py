"""
ABI model abstraction for the R65 compiler.

Defines selectable ABI policies that control calling convention decisions:
- Default: PHA-based argument passing with caller PLX cleanup (default)
- FixedStack: Zero-frame model with hw registers + scratch only, PHB-per-byte frames
- Pascal: Apple IIGS/Pascal convention — all params on stack, callee cleanup, stack result space
"""

import sys
from abc import ABC, abstractmethod
from enum import Enum


class ABIKind(Enum):
    FIXED_STACK = "FixedStack"
    PASCAL = "Pascal"
    DEFAULT = "Default"


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

    @property
    def frame_alloc_clobbers_a_threshold(self) -> int:
        """Max frame size that can be allocated without clobbering A.

        Frames up to this size use push-based allocation (PHB/PHX/PHY/PHA).
        Frames above this size use TSC/SEC/SBC/TCS which clobbers A.
        """
        return 4

    # -- shared helpers --

    def _emit_phb_alloc(self, emit_instr, frame_size: int):
        from r65.compiler.codegen.opcodes import Opcode
        for _ in range(frame_size):
            emit_instr(Opcode.PHB, comment=f"Allocate frame ({frame_size} bytes)")

    def _emit_register_push_alloc(self, emit_instr, frame_size: int):
        """Allocate frame using PHX/PHY (2 bytes each) and PHB (1 byte remainder).

        More efficient than PHB-per-byte: PHX/PHY push 2 bytes at 4 cycles
        vs 2×PHB at 6 cycles. Alternates PHX/PHY to avoid needing both
        registers to hold specific values.

        Uses PHB (not PHA) for the 1-byte remainder because PHA's push size
        depends on the M flag: 1 byte in m8, 2 bytes in m16. Functions with
        @ A: u16 parameters enter in m16 mode (caller does REP #$20 before
        JSR), so PHA would push 2 bytes instead of 1. PHB always pushes
        exactly 1 byte regardless of processor mode.
        """
        from r65.compiler.codegen.opcodes import Opcode
        remaining = frame_size
        use_x = True
        while remaining >= 2:
            if use_x:
                emit_instr(Opcode.PHX, comment=f"Allocate frame ({frame_size} bytes)")
            else:
                emit_instr(Opcode.PHY, comment=f"Allocate frame ({frame_size} bytes)")
            use_x = not use_x
            remaining -= 2
        if remaining == 1:
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
        from r65.compiler.codegen.opcodes import Opcode
        from r65.compiler.codegen.constants import M_FLAG

        stack_args = [arg for arg in instr.args if arg.mechanism == ArgumentMechanism.STACK]
        other_args = [arg for arg in instr.args if arg.mechanism != ArgumentMechanism.STACK]

        spill_offset = selector.get_current_spill_offset()
        stack_bytes_pushed = 0

        # Detect A→A collision: stack arg setup uses LDA which clobbers A.
        # If a register arg targets A and its source is already in A
        # (hw-coalescenced), save A to a temp register before stack arg
        # setup and restore after.
        a_save_reg = None
        a_param_is_16bit = False
        if stack_args:
            for arg in other_args:
                if arg.mechanism != ArgumentMechanism.REGISTER:
                    continue
                target_reg = arg.location.name if hasattr(arg.location, 'name') else str(arg.location)
                if target_reg != 'A':
                    continue
                arg_loc = selector.parent._get_operand_location(arg.value)
                if arg_loc.is_hw('A'):
                    if arg.param_type is not None:
                        a_param_is_16bit = get_type_size(arg.param_type) >= 2
                    # Find a temp register not used as a register arg target
                    reg_targets = set()
                    for a in other_args:
                        if a.mechanism == ArgumentMechanism.REGISTER:
                            t = a.location.name if hasattr(a.location, 'name') else str(a.location)
                            reg_targets.add(t)
                    if 'Y' not in reg_targets:
                        a_save_reg = 'Y'
                    elif 'X' not in reg_targets:
                        a_save_reg = 'X'
                    break

        if a_save_reg:
            # TAY/TAX: with X flag=0 (always 16-bit index in R65),
            # transfers full 16-bit accumulator regardless of M flag.
            op = Opcode.TAY if a_save_reg == 'Y' else Opcode.TAX
            selector._emit_implied(op, f"Save A to {a_save_reg} before stack arg setup")

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

        if a_save_reg:
            # TYA/TXA: transfer size depends on M flag. For 16-bit A values,
            # ensure m16 so all 16 bits are restored.
            if a_param_is_16bit:
                current_mode = selector.parent.emitter.get_accu_mode()
                if current_mode != 16:
                    selector._emit_immediate(Opcode.REP_IMMEDIATE, M_FLAG,
                                             "m16 for A restore from index reg")
                    selector.parent.emitter.emit_accu_mode(16)
            op = Opcode.TYA if a_save_reg == 'Y' else Opcode.TXA
            selector._emit_implied(op, f"Restore A from {a_save_reg} after stack arg setup")

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
            if instr.self_is_far:
                selector.load_y_with_far_self(self_y_arg, 0)
            else:
                selector.load_y_with_self(self_y_arg, 0)

        return stack_bytes_pushed

    def emit_return_values(self, selector, instr):
        """Emit return value loading for a Return instruction.

        Concrete on the base class — Default and FixedStack share identical
        register-based logic. Pascal overrides to write to stack result space.

        Args:
            selector: ControlFlowInstructionSelector instance
            instr: Return MIR instruction
        """
        from r65.compiler.codegen.register_alloc import LocationKind
        from r65.compiler.codegen.opcodes import Opcode
        from r65.compiler.errors import InstructionSelectionError

        if not instr.values:
            return

        return_registers = selector._get_return_register_order()
        if len(instr.values) > len(return_registers):
            raise InstructionSelectionError(
                f"Too many return values (max {len(return_registers)})",
                source_loc=selector.parent._current_source_loc)

        # Process in reverse order to avoid clobbering A
        # Reverse order: Y first, then X, then B (XBA to store), then A last
        for i in range(len(instr.values) - 1, -1, -1):
            value = instr.values[i]
            target_reg = return_registers[i]
            value_loc = selector.parent._get_operand_location(value)

            if value_loc.kind == LocationKind.RETURN_SINKABLE:
                # Deferred load: emit the load directly into the target register
                src_loc = selector.parent._get_operand_location(value_loc.source_location)
                if target_reg == 'B':
                    selector.parent._emit_load('LDA', src_loc)
                    selector.parent._store_to_b_from_a()
                elif target_reg in ('X', 'Y'):
                    selector.parent._emit_load('LDA', src_loc)
                    if target_reg == 'X':
                        selector._emit_implied(Opcode.TAX)
                    else:
                        selector._emit_implied(Opcode.TAY)
                else:  # 'A'
                    selector.parent._emit_load('LDA', src_loc)
                continue
            elif value_loc.is_hw(target_reg):
                pass  # Already in correct register
            elif target_reg == 'B':
                # B return: load value into A, then XBA to store in B
                if value_loc.is_hw('A'):
                    selector.parent._store_to_b_from_a()
                elif value_loc.is_hw():
                    selector.parent._emit_register_transfer(value_loc.hw_register, 'A')
                    selector.parent._store_to_b_from_a()
                else:
                    selector.parent._emit_load('LDA', value_loc)
                    selector.parent._store_to_b_from_a()
            elif value_loc.is_hw():
                selector.parent._emit_register_transfer(value_loc.hw_register, target_reg)
            elif target_reg in ('X', 'Y') and value_loc.kind == LocationKind.STACK:
                # Handle stack-relative addressing: LDX/LDY don't support sr,S mode
                selector.parent._emit_load('LDA', value_loc)
                if target_reg == 'X':
                    selector._emit_implied(Opcode.TAX, "Transfer to X (no LDX sr,S)")
                else:
                    selector._emit_implied(Opcode.TAY, "Transfer to Y (no LDY sr,S)")
            else:
                load_mnem = {'A': 'LDA', 'X': 'LDX', 'Y': 'LDY'}.get(target_reg, 'LDA')
                selector.parent._emit_load(load_mnem, value_loc)

    def __repr__(self):
        return f"ABIModel({self.kind.value})"


class ABIFixedStack(ABIModel):
    """Fixed-stack ABI with no dynamic frame allocation (--abi FixedStack).

    All parameters are promoted to hardware registers or direct-page scratch
    locations — no stack-passed parameters are permitted. This eliminates
    the need for an outgoing-arg area and simplifies frame layout to just
    preserves and a minimal local frame.

    Frame allocation always uses PHB per byte (never TSC/SBC/TCS), and
    deallocation always uses PLA per byte (max_pla_dealloc_size is
    unlimited). This keeps the stack pointer movement predictable and
    bounded, which is useful for environments where stack depth must be
    statically analyzable. Recursive functions are rejected at compile time
    under this ABI.

    Return values are passed in hardware registers (A, B, X, Y), identical
    to the Default ABI.
    """

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

    @property
    def frame_alloc_clobbers_a_threshold(self) -> int:
        return sys.maxsize  # PHB never clobbers A


class ABIPascal(ABIModel):
    """Pascal/Apple IIGS calling convention (--abi Pascal).

    All parameters go on the stack regardless of register bindings (@ A,
    @ X, @ Y annotations are ignored). Parameters are pushed left-to-right
    via PHA — the first parameter is pushed first and ends up deepest, the
    last parameter sits closest to the return address. No scratch promotion
    or outgoing-arg area is used.

    The caller pushes result space (sized to the return type) onto the stack
    before any parameters. After the call returns, the callee has cleaned
    up the parameter bytes, leaving just the result space at TOS for the
    caller to PLA.

    Frame allocation uses PHB per byte for small frames (<=4) and
    TSC/SEC/SBC/TCS for larger ones, same as Default. The callee writes
    its return value into the result space via STA offset,S before the
    epilogue, then removes parameter bytes (but not result space) using
    the existing PLX/TSC/ADC/TCS/PHX/RTS cleanup machinery.
    """

    def __init__(self):
        super().__init__(ABIKind.PASCAL)

    def run_param_analysis(self, mir_program, scratch_pool, disable_scratch_params: bool):
        # No scratch promotion — all params stay on stack
        pass

    def compute_outgoing_args(self, mir_program, compute_fn):
        # Pascal callers push via PHA (not STA to outgoing area)
        # No outgoing arg area needed
        for func in mir_program.functions:
            func.max_outgoing_arg_bytes = 0

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

    def emit_call_args(self, selector, instr, pre_arg_stack_adj=0) -> int:
        """Pascal caller: push result space, then all params L->R via PHA.

        All arguments use PHA regardless of their original mechanism.
        Returns total stack_bytes_pushed (result space + param bytes).
        """
        from r65.compiler.mir.nodes import ArgumentMechanism
        from r65.compiler.codegen.register_alloc import LocationKind
        from r65.compiler.codegen.type_utils import get_type_size

        stack_bytes_pushed = 0

        # Step 1: Push result space (if non-void return)
        result_bytes = getattr(instr, 'pascal_result_bytes', 0)
        if result_bytes > 0:
            # Push zero bytes for result space
            for _ in range(result_bytes):
                selector._emit_push('A', "Result space (Pascal)")
                # A contents don't matter — caller will overwrite later
            stack_bytes_pushed += result_bytes
            selector.region_state.stack_tracker.push(result_bytes)

        # Step 2: Push all params left-to-right (param0 first, paramN last)
        # In Pascal, ALL args are stack-pushed regardless of mechanism
        all_args = list(instr.args)
        for arg in all_args:
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

        return stack_bytes_pushed

    def emit_return_values(self, selector, instr):
        """Pascal callee: write return value to stack result space.

        The result space was pushed by the caller before params. At this point
        (before epilogue/frame dealloc), the offset from SP to the result space is:
          frame_size + prologue_bytes + return_addr_size + total_param_bytes + 1

        Args:
            selector: ControlFlowInstructionSelector instance
            instr: Return MIR instruction
        """
        from r65.compiler.codegen.register_alloc import LocationKind
        from r65.compiler.codegen.opcodes import Opcode
        from r65.compiler.codegen.asm_nodes import StackOffset
        from r65.compiler.codegen.constants import M_FLAG

        func = selector.current_function
        if not func or func.pascal_result_space_bytes == 0:
            # Void function — no result to write
            return

        if not instr.values:
            return

        result_bytes = func.pascal_result_space_bytes
        total_param_bytes = func.pascal_total_param_bytes

        # Compute offset from SP to the start of result space
        frame_size = 0
        if selector.parent.reg_alloc and selector.parent.reg_alloc.has_frame_allocation:
            frame_size = selector.parent.reg_alloc.frame_size
        prologue_bytes = func.abi_info.prologue_stack_bytes if func.abi_info else 0
        return_addr_size = 3 if func.is_far else 2

        result_offset = frame_size + prologue_bytes + return_addr_size + total_param_bytes + 1

        value = instr.values[0]
        value_loc = selector.parent._get_operand_location(value)

        if result_bytes == 1:
            # 8-bit result: load into A, STA offset,S
            if not value_loc.is_hw('A'):
                selector.parent._emit_load('LDA', value_loc)
            selector.emitter.emit_instr(Opcode.STA_STACK, StackOffset(result_offset),
                                        "Store return value to Pascal result space")
        elif result_bytes == 2:
            # 16-bit result: ensure m16, load into A, STA offset,S
            current_mode = selector.parent.emitter.get_accu_mode()
            if current_mode != 16:
                selector._emit_immediate(Opcode.REP_IMMEDIATE, M_FLAG,
                                         "16-bit A for Pascal result store")
                selector.parent.emitter.emit_accu_mode(16)
            if not value_loc.is_hw('A'):
                selector.parent._emit_load('LDA', value_loc)
            selector.emitter.emit_instr(Opcode.STA_STACK, StackOffset(result_offset),
                                        "Store return value to Pascal result space")


class ABIDefault(ABIModel):
    """Default calling convention with PHA-based argument passing (--abi Default).

    Eliminates the permanent outgoing-arg area from caller stack frames.
    Instead, arguments are pushed via PHA before each call and cleaned up
    by the caller via PLX after the call returns. This produces smaller
    frames (no outgoing area inflation) and smaller code (PHA is 1 byte
    vs STA d,S at 2 bytes).

    Parameter passing uses the same hybrid mechanism as Default (register,
    variable-bound, scratch-promoted, and stack), but stack arguments are
    always emitted via PHA instead of STA to a fixed outgoing area. Callee
    behavior is completely unchanged — params appear at the same stack
    offsets above the return address.

    Frame allocation uses register pushes (PHX/PHY for 2-byte chunks,
    PHA for 1-byte remainder) for frames up to 8 bytes, and
    TSC/SEC/SBC/TCS for larger frames. This is more efficient than
    PHB-per-byte (PHX/PHY push 2 bytes at 4 cycles vs 2×PHB at 6 cycles)
    and avoids clobbering A for frames up to 8 bytes.

    Return values are passed in hardware registers (A, B, X, Y), identical
    to the Default ABI. For calls that return in X (multi-register returns),
    the cleanup uses a scratch save to avoid clobbering X.
    """

    def __init__(self):
        super().__init__(ABIKind.DEFAULT)

    def run_param_analysis(self, mir_program, scratch_pool, disable_scratch_params: bool):
        if not disable_scratch_params:
            from r65.compiler.analysis.scratch_params import analyze_scratch_params
            analyze_scratch_params(mir_program, scratch_pool)

    def compute_outgoing_args(self, mir_program, compute_fn):
        # Default ABI uses PHA per call — no permanent outgoing area
        for func in mir_program.functions:
            func.max_outgoing_arg_bytes = 0

    def emit_frame_alloc(self, emit_instr, frame_size: int, force_direct_stack: bool):
        if frame_size <= 0:
            return
        if frame_size <= 8 and not force_direct_stack:
            self._emit_register_push_alloc(emit_instr, frame_size)
        else:
            self._emit_tsc_alloc(emit_instr, frame_size)

    @property
    def max_pla_dealloc_size(self) -> int:
        return 4

    @property
    def frame_alloc_clobbers_a_threshold(self) -> int:
        return 8  # Register pushes (PHX/PHY/PHA) don't clobber A

    def emit_call_args(self, selector, instr, pre_arg_stack_adj=0) -> int:
        """Default caller: always push stack args via PHA, then set up register args.

        Unlike Default which uses STA to a fixed outgoing area, Default always
        pushes via PHA regardless of spill state. Register/variable/scratch args
        are handled identically to Default.

        Returns total stack_bytes_pushed for caller cleanup.
        """
        from r65.compiler.mir.nodes import ArgumentMechanism
        from r65.compiler.codegen.register_alloc import LocationKind
        from r65.compiler.codegen.type_utils import get_type_size
        from r65.compiler.codegen.opcodes import Opcode
        from r65.compiler.codegen.constants import M_FLAG

        stack_args = [arg for arg in instr.args if arg.mechanism == ArgumentMechanism.STACK]
        other_args = [arg for arg in instr.args if arg.mechanism != ArgumentMechanism.STACK]

        stack_bytes_pushed = 0

        # Detect A→A collision: PHA stack arg setup uses LDA which clobbers A.
        # If a register arg targets A and its source is already in A
        # (hw-coalescenced), save A to a temp register before stack arg
        # setup and restore after.
        a_save_reg = None
        a_param_is_16bit = False
        if stack_args:
            for arg in other_args:
                if arg.mechanism != ArgumentMechanism.REGISTER:
                    continue
                target_reg = arg.location.name if hasattr(arg.location, 'name') else str(arg.location)
                if target_reg != 'A':
                    continue
                arg_loc = selector.parent._get_operand_location(arg.value)
                if arg_loc.is_hw('A'):
                    if arg.param_type is not None:
                        a_param_is_16bit = get_type_size(arg.param_type) >= 2
                    # Find a temp register not used as a register arg target
                    reg_targets = set()
                    for a in other_args:
                        if a.mechanism == ArgumentMechanism.REGISTER:
                            t = a.location.name if hasattr(a.location, 'name') else str(a.location)
                            reg_targets.add(t)
                    if 'Y' not in reg_targets:
                        a_save_reg = 'Y'
                    elif 'X' not in reg_targets:
                        a_save_reg = 'X'
                    break

        if a_save_reg:
            op = Opcode.TAY if a_save_reg == 'Y' else Opcode.TAX
            selector._emit_implied(op, f"Save A to {a_save_reg} before stack arg setup")

        # Always use PHA path for stack args (right-to-left for correct callee layout)
        if stack_args:
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

        if a_save_reg:
            if a_param_is_16bit:
                current_mode = selector.parent.emitter.get_accu_mode()
                if current_mode != 16:
                    selector._emit_immediate(Opcode.REP_IMMEDIATE, M_FLAG,
                                             "m16 for A restore from index reg")
                    selector.parent.emitter.emit_accu_mode(16)
            op = Opcode.TYA if a_save_reg == 'Y' else Opcode.TXA
            selector._emit_implied(op, f"Restore A from {a_save_reg} after stack arg setup")

        # Non-stack args: same as Default
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
        """Default trait dispatch: always push stack args via PHA.

        Same as Default's trait dispatch but forces PHA path for stack args.
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

        stack_bytes_pushed = 0

        # Always use PHA path for stack args
        if stack_args:
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

        if self_y_arg is not None:
            if instr.self_is_far:
                selector.load_y_with_far_self(self_y_arg, 0)
            else:
                selector.load_y_with_self(self_y_arg, 0)

        return stack_bytes_pushed


# Singleton instances
ABI_FIXED_STACK = ABIFixedStack()
ABI_PASCAL = ABIPascal()
ABI_DEFAULT = ABIDefault()


def abi_model_from_string(name: str) -> ABIModel:
    """Create ABIModel from CLI string argument.

    Args:
        name: "Default", "FixedStack", or "Pascal"

    Returns:
        Corresponding ABIModel instance

    Raises:
        ValueError: If name is not recognized
    """
    if name == "FixedStack":
        return ABI_FIXED_STACK
    elif name == "Pascal":
        return ABI_PASCAL
    elif name == "Default":
        return ABI_DEFAULT
    else:
        raise ValueError(f"Unknown ABI model: {name!r}. Expected 'Default', 'FixedStack', or 'Pascal'.")
