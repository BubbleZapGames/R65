# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
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

    # -- shared frame-allocation helpers --

    def _emit_phb_alloc(self, emit_instr, frame_size: int):
        from r65.compiler.codegen.opcodes import Opcode
        for i in range(frame_size):
            comment = f"Allocate frame ({frame_size} bytes)" if i == 0 else ""
            emit_instr(Opcode.PHB, comment=comment)

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
        first = True
        while remaining >= 2:
            op = Opcode.PHX if use_x else Opcode.PHY
            comment = f"Allocate frame ({frame_size} bytes)" if first else ""
            emit_instr(op, comment=comment)
            first = False
            use_x = not use_x
            remaining -= 2
        if remaining == 1:
            comment = f"Allocate frame ({frame_size} bytes)" if first else ""
            emit_instr(Opcode.PHB, comment=comment)

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

    # -- call argument emission --

    def _use_pha_for_stack_args(self, spill_offset: int) -> bool:
        """When True, stack args go through PHA; otherwise through a fixed outgoing area.

        Overridden by ABIs that always push via PHA (e.g. Default).
        """
        return spill_offset > 0

    def emit_call_args(self, selector, instr, pre_arg_stack_adj=0) -> int:
        """Emit call arguments. Returns stack_bytes_pushed (>0 only in PHA path).

        Shared across Default and FixedStack. Pascal overrides to push all args
        L→R after result space. Default overrides only _use_pha_for_stack_args
        to force the PHA branch.

        Args:
            selector: CallInstructionSelector instance
            instr: Call MIR instruction
            pre_arg_stack_adj: Stack offset adjustment from pre-arg operations
        """
        from r65.compiler.mir.nodes import ArgumentMechanism
        from r65.compiler.codegen.opcodes import Opcode

        stack_args = [a for a in instr.args if a.mechanism == ArgumentMechanism.STACK]
        other_args = [a for a in instr.args if a.mechanism != ArgumentMechanism.STACK]

        spill_offset = selector.get_current_spill_offset()
        use_pha = self._use_pha_for_stack_args(spill_offset)

        a_save_reg, a_param_is_16bit, a_resident_stack_idx = (
            self._compute_a_save_for_call(selector, stack_args, other_args, use_pha))

        if a_save_reg:
            op = Opcode.TAY if a_save_reg == 'Y' else Opcode.TAX
            selector._emit_implied(op, f"Save A to {a_save_reg} before stack arg setup")

        stack_bytes_pushed = 0
        if stack_args:
            if use_pha:
                stack_bytes_pushed = self._emit_stack_args_pha(
                    selector, stack_args, a_save_reg, pre_arg_stack_adj)
            else:
                self._emit_stack_args_outgoing(
                    selector, stack_args, a_resident_stack_idx, pre_arg_stack_adj)

        if a_save_reg:
            self._restore_a_from(selector, a_save_reg, a_param_is_16bit)

        self._emit_other_args(selector, other_args, pre_arg_stack_adj)

        self_y_arg = next(
            (a for a in instr.args if a.mechanism == ArgumentMechanism.SELF_Y), None)
        if self_y_arg is not None:
            self._emit_self_y(selector, self_y_arg)

        return stack_bytes_pushed

    @staticmethod
    def _arg_size(arg) -> int:
        from r65.compiler.codegen.type_utils import get_type_size
        if arg.param_type is not None:
            return get_type_size(arg.param_type)
        if hasattr(arg.value, 'type_info') and arg.value.type_info:
            return get_type_size(arg.value.type_info)
        return 1

    @staticmethod
    def _register_target_name(arg) -> str:
        return arg.location.name if hasattr(arg.location, 'name') else str(arg.location)

    def _compute_a_save_for_call(self, selector, stack_args, other_args, use_pha):
        """Decide whether A must be saved across stack-arg setup.

        Returns (a_save_reg, a_param_is_16bit, a_resident_stack_idx) where
        a_save_reg is 'Y', 'X', or None.

        A save is needed when:
          - any non-stack arg's value is in A (LDA in stack setup would
            clobber it), or
          - a stack arg's value is in A but it isn't the last one (PHA path
            processes args in reverse, so non-last A-resident args get
            clobbered by prior LDAs).
        """
        from r65.compiler.mir.nodes import ArgumentMechanism

        if not stack_args:
            return None, False, None

        a_resident_stack_idx = None
        for i, arg in enumerate(stack_args):
            arg_loc = selector.parent._get_operand_location(arg.value)
            if arg_loc.is_hw('A'):
                a_resident_stack_idx = i
                break

        a_in_other = False
        a_param_is_16bit = False
        for arg in other_args:
            arg_loc = selector.parent._get_operand_location(arg.value)
            if not arg_loc.is_hw('A'):
                continue
            a_in_other = True
            if (arg.mechanism == ArgumentMechanism.REGISTER
                    and arg.param_type is not None):
                a_param_is_16bit = self._arg_size(arg) >= 2
            break

        a_stack_needs_save = (
            a_resident_stack_idx is not None
            and a_resident_stack_idx != len(stack_args) - 1
            and use_pha
        )

        if not (a_in_other or a_stack_needs_save):
            return None, a_param_is_16bit, a_resident_stack_idx

        reg_targets = {
            self._register_target_name(a)
            for a in other_args if a.mechanism == ArgumentMechanism.REGISTER
        }
        if 'Y' not in reg_targets:
            return 'Y', a_param_is_16bit, a_resident_stack_idx
        if 'X' not in reg_targets:
            return 'X', a_param_is_16bit, a_resident_stack_idx
        return None, a_param_is_16bit, a_resident_stack_idx

    def _emit_stack_args_pha(self, selector, stack_args, a_save_reg, pre_arg_stack_adj) -> int:
        """Push stack args right-to-left via PHA. Returns bytes pushed."""
        from r65.compiler.codegen.register_alloc import LocationKind
        from r65.compiler.codegen.opcodes import Opcode

        stack_bytes_pushed = 0
        a_clobbered = False
        for arg in reversed(stack_args):
            arg_loc = selector.parent._get_operand_location(arg.value)

            if arg_loc.is_hw('A') and a_clobbered and a_save_reg:
                op = Opcode.TYA if a_save_reg == 'Y' else Opcode.TXA
                selector._emit_implied(op, f"Restore A from {a_save_reg}")

            if arg_loc.kind == LocationKind.STACK and pre_arg_stack_adj != 0:
                arg_loc = selector.parent._offset_location(arg_loc, pre_arg_stack_adj)

            arg_size = self._arg_size(arg)
            selector.emit_pha_stack_argument(arg, arg_loc, arg_size)
            stack_bytes_pushed += arg_size
            selector.region_state.stack_tracker.push(arg_size)

            if not arg_loc.is_hw('A'):
                a_clobbered = True
        return stack_bytes_pushed

    def _emit_stack_args_outgoing(self, selector, stack_args,
                                  a_resident_stack_idx, pre_arg_stack_adj):
        """Write stack args into the pre-allocated outgoing area via STA."""
        from r65.compiler.codegen.register_alloc import LocationKind

        offsets = []
        outgoing_offset = 1
        for arg in stack_args:
            offsets.append(outgoing_offset)
            outgoing_offset += self._arg_size(arg)

        # A-resident arg goes first so subsequent LDAs don't clobber its source.
        indices = list(range(len(stack_args)))
        if a_resident_stack_idx is not None and len(stack_args) > 1:
            indices.remove(a_resident_stack_idx)
            indices.insert(0, a_resident_stack_idx)

        for i in indices:
            arg = stack_args[i]
            arg_loc = selector.parent._get_operand_location(arg.value)
            if arg_loc.kind == LocationKind.STACK and pre_arg_stack_adj != 0:
                arg_loc = selector.parent._offset_location(arg_loc, pre_arg_stack_adj)
            selector.emit_outgoing_stack_argument(arg, arg_loc, offsets[i])

    def _restore_a_from(self, selector, a_save_reg, a_param_is_16bit):
        """Emit TYA/TXA to restore A, forcing m16 if the param is 16-bit."""
        from r65.compiler.codegen.opcodes import Opcode
        from r65.compiler.codegen.constants import M_FLAG
        if a_param_is_16bit:
            current_mode = selector.parent.emitter.get_accu_mode()
            if current_mode != 16:
                selector._emit_immediate(Opcode.REP_IMMEDIATE, M_FLAG,
                                         "m16 for A restore from index reg")
                selector.parent.emitter.emit_accu_mode(16)
        op = Opcode.TYA if a_save_reg == 'Y' else Opcode.TXA
        selector._emit_implied(op, f"Restore A from {a_save_reg} after stack arg setup")

    def _emit_other_args(self, selector, other_args, pre_arg_stack_adj):
        """Emit non-stack args with WAR-hazard handling for scratch params.

        Sort order:
          1. A-resident args that go to non-A targets — emit while A still holds
             the source value (transfer/STA before other args' LDA clobbers it).
          2. Everything else, per selector.arg_sort_key.

        After ordering, scratch params are reordered for WAR hazards: if a
        scratch param's source lives at another scratch param's target, the
        reader must run before the writer. When that forces a non-A-resident
        scratch param to precede an A-resident one, the non-A param is wrapped
        in TAY/TYA so A survives.
        """
        from r65.compiler.mir.nodes import ArgumentMechanism
        from r65.compiler.codegen.register_alloc import LocationKind
        from r65.compiler.codegen.opcodes import Opcode
        from r65.compiler.codegen.constants import M_FLAG

        def is_a_resident_priority(arg):
            arg_loc = selector.parent._get_operand_location(arg.value)
            if not arg_loc.is_hw('A'):
                return False
            if arg.mechanism != ArgumentMechanism.REGISTER:
                return True
            return self._register_target_name(arg) != 'A'

        sorted_other_args = sorted(other_args, key=selector.arg_sort_key)
        sorted_other_args.sort(key=lambda a: 0 if is_a_resident_priority(a) else 1)

        sorted_other_args, needs_a_save = self._reorder_scratch_params(
            selector, sorted_other_args)

        # Additional WAR case: a REGISTER A target whose source is already in A
        # (the no-op case) coexists with SCRATCH_PARAM/VARIABLE args whose
        # setup clobbers A. Mark scratch params so they get bracketed below.
        def is_a_target_with_a_source(arg):
            if arg.mechanism != ArgumentMechanism.REGISTER:
                return False
            if self._register_target_name(arg) != 'A':
                return False
            return selector.parent._get_operand_location(arg.value).is_hw('A')

        if not needs_a_save:
            a_in_place_arg = next(
                (a for a in sorted_other_args if is_a_target_with_a_source(a)), None)
            if a_in_place_arg is not None:
                clobbers_a = any(
                    a is not a_in_place_arg
                    and a.mechanism in (ArgumentMechanism.SCRATCH_PARAM,
                                        ArgumentMechanism.VARIABLE)
                    for a in sorted_other_args
                )
                if clobbers_a:
                    for a in sorted_other_args:
                        if (a is not a_in_place_arg
                                and a.mechanism == ArgumentMechanism.SCRATCH_PARAM):
                            needs_a_save.add(a)

        def adjust_loc(loc):
            if loc.kind == LocationKind.STACK and pre_arg_stack_adj != 0:
                return selector.parent._offset_location(loc, pre_arg_stack_adj)
            return loc

        if not needs_a_save:
            for arg in sorted_other_args:
                arg_loc = adjust_loc(selector.parent._get_operand_location(arg.value))
                if arg.mechanism == ArgumentMechanism.REGISTER:
                    selector.emit_register_argument(arg, arg_loc)
                elif arg.mechanism == ArgumentMechanism.VARIABLE:
                    selector.emit_variable_argument(arg, arg_loc)
                elif arg.mechanism == ArgumentMechanism.SCRATCH_PARAM:
                    selector.emit_scratch_param_argument(arg, arg_loc)
            return

        # WAR path: split, save A around the conflicting setups.
        non_scratch = []
        non_a_scratch = []
        a_scratch = []
        for arg in sorted_other_args:
            if arg.mechanism != ArgumentMechanism.SCRATCH_PARAM:
                non_scratch.append(arg)
            elif selector.parent._get_operand_location(arg.value).is_hw('A'):
                a_scratch.append(arg)
            else:
                non_a_scratch.append(arg)

        for arg in non_scratch:
            arg_loc = adjust_loc(selector.parent._get_operand_location(arg.value))
            if arg.mechanism == ArgumentMechanism.REGISTER:
                selector.emit_register_argument(arg, arg_loc)
            elif arg.mechanism == ArgumentMechanism.VARIABLE:
                selector.emit_variable_argument(arg, arg_loc)

        # TAY transfers 16 bits in x16 mode regardless of M; TYA is M-sized,
        # so if A holds a 16-bit value we must force m16 around the TYA.
        war_a_is_16bit = False
        for war_arg in sorted_other_args:
            if war_arg.mechanism != ArgumentMechanism.REGISTER:
                continue
            if self._register_target_name(war_arg) != 'A':
                continue
            aloc = selector.parent._get_operand_location(war_arg.value)
            if not aloc.is_hw('A'):
                continue
            if (war_arg.param_type is not None
                    and self._arg_size(war_arg) == 2):
                war_a_is_16bit = True
                break

        if war_a_is_16bit:
            cur = selector.parent.emitter.get_accu_mode()
            if cur != 16:
                selector._emit_immediate(Opcode.REP_IMMEDIATE, M_FLAG,
                                         "m16 to preserve A across WAR save")
                selector.parent.emitter.emit_accu_mode(16)

        selector._emit_implied(Opcode.TAY,
                              "Save A (WAR hazard: non-A scratch before A-resident)")

        for arg in non_a_scratch:
            arg_loc = adjust_loc(selector.parent._get_operand_location(arg.value))
            selector.emit_scratch_param_argument(arg, arg_loc)

        if war_a_is_16bit:
            cur = selector.parent.emitter.get_accu_mode()
            if cur != 16:
                selector._emit_immediate(Opcode.REP_IMMEDIATE, M_FLAG,
                                         "m16 for TYA WAR restore")
                selector.parent.emitter.emit_accu_mode(16)
        selector._emit_implied(Opcode.TYA, "Restore A (WAR hazard resolved)")

        for arg in a_scratch:
            arg_loc = adjust_loc(selector.parent._get_operand_location(arg.value))
            selector.emit_scratch_param_argument(arg, arg_loc)

    def _emit_self_y(self, selector, self_y_arg):
        """Load Y (and DBR for far self) for a SELF_Y argument."""
        from r65.compiler.hir.types import PointerTypeInfo
        if (isinstance(self_y_arg.param_type, PointerTypeInfo)
                and self_y_arg.param_type.is_far):
            selector.load_y_with_far_self(self_y_arg, 0)
        else:
            selector.load_y_with_self(self_y_arg, 0)

    def _reorder_scratch_params(self, selector, args):
        """Topologically sort scratch params for WAR safety.

        If scratch param A's source lives at scratch param B's target address,
        A must be emitted before B (read before write). Cycles fall back to
        original order.

        Returns (reordered_args, needs_a_save) where needs_a_save is the set
        of scratch-param argument objects that must be bracketed by TAY/TYA
        because reordering moved a non-A-resident scratch param ahead of an
        A-resident one. The caller emits the save/restore.
        """
        from r65.compiler.mir.nodes import ArgumentMechanism
        from r65.compiler.codegen.register_alloc import LocationKind

        all_scratch = []         # (arg, is_a_resident)
        scratch_positions = []   # positions in `args`
        for pos, arg in enumerate(args):
            if arg.mechanism != ArgumentMechanism.SCRATCH_PARAM:
                continue
            arg_loc = selector.parent._get_operand_location(arg.value)
            all_scratch.append((arg, arg_loc.is_hw('A')))
            scratch_positions.append(pos)

        if len(all_scratch) <= 1:
            return args, set()

        # Map every byte of every scratch param's target back to its index.
        target_addrs = {}
        for i, (arg, _) in enumerate(all_scratch):
            addr = arg.scratch_addr
            for byte_off in range(self._arg_size(arg)):
                target_addrs[addr + byte_off] = i

        # i must precede j when i's source bytes overlap j's target bytes.
        must_precede = {}
        for i, (arg, _) in enumerate(all_scratch):
            arg_loc = selector.parent._get_operand_location(arg.value)
            if arg_loc.kind != LocationKind.SCRATCH:
                continue
            src_addr = arg_loc.scratch_addr
            src_size = arg_loc.size or 1
            for byte_off in range(src_size):
                j = target_addrs.get(src_addr + byte_off)
                if j is not None and j != i:
                    must_precede.setdefault(i, set()).add(j)

        if not must_precede:
            return args, set()

        a_indices = {i for i, (_, is_a) in enumerate(all_scratch) if is_a}
        non_a_before_a = set()
        for i, deps in must_precede.items():
            if i in a_indices:
                continue
            if any(j in a_indices for j in deps):
                non_a_before_a.add(i)

        # Topological sort (post-order DFS, then reverse).
        ordered = []
        visited = set()
        in_progress = set()

        def visit(idx):
            if idx in visited or idx in in_progress:
                return
            in_progress.add(idx)
            for dep in must_precede.get(idx, ()):
                visit(dep)
            in_progress.discard(idx)
            visited.add(idx)
            ordered.append(idx)

        for i in range(len(all_scratch)):
            visit(i)
        ordered.reverse()

        reordered_scratch = [all_scratch[i][0] for i in ordered]

        needs_a_save = set()
        if non_a_before_a:
            arg_to_idx = {id(a): i for i, (a, _) in enumerate(all_scratch)}
            for arg in reordered_scratch:
                if arg_to_idx[id(arg)] in non_a_before_a:
                    needs_a_save.add(arg)

        # Slot the reordered scratch args back into their original positions
        # in the broader args list (non-scratch args stay where they were).
        result = []
        scratch_iter = iter(reordered_scratch)
        scratch_pos_set = set(scratch_positions)
        for pos, arg in enumerate(args):
            if pos in scratch_pos_set:
                result.append(next(scratch_iter))
            else:
                result.append(arg)
        return result, needs_a_save

    def emit_trait_dispatch_args(self, selector, instr) -> int:
        """Emit trait dispatch arguments. Returns stack_bytes_pushed.

        Scratch params first (they write to DP, not stack), then stack args.
        Stack args use PHA or outgoing area per _use_pha_for_stack_args.
        SELF_Y is deferred to the caller via pending fields so chain
        coalescing can elide the DBR-set on MIDDLE/END members.
        """
        from r65.compiler.mir.nodes import ArgumentMechanism
        from r65.compiler.codegen.register_alloc import LocationKind

        stack_args = [a for a in instr.args if a.mechanism == ArgumentMechanism.STACK]
        scratch_args = [a for a in instr.args if a.mechanism == ArgumentMechanism.SCRATCH_PARAM]
        self_y_arg = next(
            (a for a in instr.args if a.mechanism == ArgumentMechanism.SELF_Y), None)

        for arg in scratch_args:
            arg_loc = selector.parent._get_operand_location(arg.value)
            selector.emit_scratch_param_argument(arg, arg_loc)

        spill_offset = selector.get_current_spill_offset()
        use_pha = self._use_pha_for_stack_args(spill_offset)
        stack_bytes_pushed = 0

        if stack_args:
            if use_pha:
                for arg in reversed(stack_args):
                    arg_loc = selector.parent._get_operand_location(arg.value)
                    arg_size = self._arg_size(arg)
                    selector.emit_pha_stack_argument(arg, arg_loc, arg_size)
                    stack_bytes_pushed += arg_size
                    selector.region_state.stack_tracker.push(arg_size)
            else:
                outgoing_offset = 1
                for arg in stack_args:
                    arg_loc = selector.parent._get_operand_location(arg.value)
                    selector.emit_outgoing_stack_argument(arg, arg_loc, outgoing_offset)
                    outgoing_offset += self._arg_size(arg)

        # stack_bytes_pushed is tracked via region_state.stack_tracker; the
        # resolver applies it centrally, so pending_self_y_stack_bytes stays 0.
        selector._pending_self_y_arg = self_y_arg
        selector._pending_self_y_stack_bytes = 0

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
                # Handle stack-relative addressing: LDX/LDY don't support sr,S mode.
                # Force m16 — TAX/TAY in m8/x16 transfers B:A (stale B) into a
                # 16-bit index register, corrupting the high byte.
                selector.parent._ensure_m16_mode()
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
        from r65.compiler.codegen.register_alloc import LocationKind

        stack_bytes_pushed = 0

        # Step 1: Push result space (if non-void return)
        result_bytes = getattr(instr, 'pascal_result_bytes', 0)
        if result_bytes > 0:
            for _ in range(result_bytes):
                selector._emit_push('A', "Result space (Pascal)")
                # A contents don't matter — caller will overwrite later
            stack_bytes_pushed += result_bytes
            selector.region_state.stack_tracker.push(result_bytes)

        # Step 2: Push all params left-to-right (param0 first, paramN last).
        # In Pascal, ALL args are stack-pushed regardless of mechanism.
        for arg in instr.args:
            arg_loc = selector.parent._get_operand_location(arg.value)

            if arg_loc.kind == LocationKind.STACK and pre_arg_stack_adj != 0:
                arg_loc = selector.parent._offset_location(arg_loc, pre_arg_stack_adj)

            arg_size = self._arg_size(arg)
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
        from r65.compiler.codegen.opcodes import Opcode
        from r65.compiler.codegen.asm_nodes import StackOffset
        from r65.compiler.codegen.constants import M_FLAG
        from r65.compiler.errors import InstructionSelectionError

        func = selector.current_function
        if not func or func.pascal_result_space_bytes == 0:
            # Void function — no result to write
            return

        if not instr.values:
            return

        result_bytes = func.pascal_result_space_bytes
        if result_bytes not in (1, 2):
            raise InstructionSelectionError(
                f"Pascal ABI: unsupported return size {result_bytes} bytes "
                f"(only 1 or 2 supported)",
                source_loc=selector.parent._current_source_loc)

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

        if result_bytes == 2:
            current_mode = selector.parent.emitter.get_accu_mode()
            if current_mode != 16:
                selector._emit_immediate(Opcode.REP_IMMEDIATE, M_FLAG,
                                         "16-bit A for Pascal result store")
                selector.parent.emitter.emit_accu_mode(16)

        if not value_loc.is_hw('A'):
            selector.parent._emit_load('LDA', value_loc)
        selector.emitter.emit_instr(
            Opcode.STA_STACK, StackOffset(result_offset),
            "Store return value to Pascal result space")


class ABIDefault(ABIModel):
    """Default calling convention with PHA-based argument passing (--abi Default).

    Eliminates the permanent outgoing-arg area from caller stack frames.
    Instead, arguments are pushed via PHA before each call and cleaned up
    by the caller via PLX after the call returns. This produces smaller
    frames (no outgoing area inflation) and smaller code (PHA is 1 byte
    vs STA d,S at 2 bytes).

    Parameter passing uses a hybrid mechanism (register, variable-bound,
    scratch-promoted, and stack). Stack arguments are always emitted via
    PHA — the only ABI-level difference from the base implementation is
    that the outgoing-area STA path is never taken (see _use_pha_for_stack_args).
    Callee behavior is unchanged — params appear at the same stack offsets
    above the return address regardless.

    Frame allocation uses register pushes (PHX/PHY for 2-byte chunks,
    PHB for 1-byte remainder) for frames up to 8 bytes, and
    TSC/SEC/SBC/TCS for larger frames. PHX/PHY push 2 bytes at 4 cycles
    vs 2×PHB at 6 cycles, and neither clobbers A.

    Return values are passed in hardware registers (A, B, X, Y). For
    calls that return in X (multi-register returns), caller-side cleanup
    uses a scratch save to avoid clobbering X.
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
        return 8  # Register pushes (PHX/PHY/PHB) don't clobber A

    def _use_pha_for_stack_args(self, spill_offset: int) -> bool:
        return True  # Default ABI always uses PHA — never the outgoing-area STA path


# Singleton instances
ABI_FIXED_STACK = ABIFixedStack()
ABI_PASCAL = ABIPascal()
ABI_DEFAULT = ABIDefault()


_ABI_BY_NAME = {
    "Default": ABI_DEFAULT,
    "FixedStack": ABI_FIXED_STACK,
    "Pascal": ABI_PASCAL,
}


def abi_model_from_string(name: str) -> ABIModel:
    """Create ABIModel from CLI string argument.

    Args:
        name: "Default", "FixedStack", or "Pascal"

    Returns:
        Corresponding ABIModel instance

    Raises:
        ValueError: If name is not recognized
    """
    try:
        return _ABI_BY_NAME[name]
    except KeyError:
        raise ValueError(
            f"Unknown ABI model: {name!r}. "
            f"Expected one of: {', '.join(sorted(_ABI_BY_NAME))}.")
