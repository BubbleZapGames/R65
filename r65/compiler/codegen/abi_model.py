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

        # Detect A-resident collision: if any arg's value is already in A
        # (hw-coalesced), stack arg setup for OTHER args uses LDA which
        # clobbers A. We need to either reorder processing (outgoing path)
        # or save/restore A (PHA path).
        a_save_reg = None
        a_param_is_16bit = False
        a_resident_stack_idx = None  # Index of A-resident stack arg

        if stack_args:
            # Check if any stack arg has its value in A
            for i, arg in enumerate(stack_args):
                arg_loc = selector.parent._get_operand_location(arg.value)
                if arg_loc.is_hw('A'):
                    a_resident_stack_idx = i
                    break

            # Check other_args for A-resident values (register-A or scratch/variable)
            a_in_other = False
            for arg in other_args:
                arg_loc = selector.parent._get_operand_location(arg.value)
                if not arg_loc.is_hw('A'):
                    continue
                a_in_other = True
                if (arg.mechanism == ArgumentMechanism.REGISTER and
                        arg.param_type is not None):
                    a_param_is_16bit = get_type_size(arg.param_type) >= 2
                break

            # Need save if other_arg is in A (stack processing would clobber it),
            # or PHA path with A-resident stack arg not at last position.
            a_stack_needs_save = (
                a_resident_stack_idx is not None and
                len(stack_args) > 1 and
                a_resident_stack_idx != len(stack_args) - 1 and
                spill_offset > 0
            )
            if (a_in_other or a_stack_needs_save) and not a_save_reg:
                reg_targets = set()
                for a in other_args:
                    if a.mechanism == ArgumentMechanism.REGISTER:
                        t = a.location.name if hasattr(a.location, 'name') else str(a.location)
                        reg_targets.add(t)
                if 'Y' not in reg_targets:
                    a_save_reg = 'Y'
                elif 'X' not in reg_targets:
                    a_save_reg = 'X'

        if a_save_reg:
            # TAY/TAX: with X flag=0 (always 16-bit index in R65),
            # transfers full 16-bit accumulator regardless of M flag.
            op = Opcode.TAY if a_save_reg == 'Y' else Opcode.TAX
            selector._emit_implied(op, f"Save A to {a_save_reg} before stack arg setup")

        if spill_offset > 0 and stack_args:
            # PHA path: push args in reverse order. If an A-resident arg
            # isn't first in reversed order, prior args' LDA clobbers A.
            # Use saved temp register to restore A when needed.
            a_clobbered = False
            for arg in reversed(stack_args):
                arg_loc = selector.parent._get_operand_location(arg.value)

                # Restore A from temp if it was clobbered by prior args' LDA
                if arg_loc.is_hw('A') and a_clobbered and a_save_reg:
                    op = Opcode.TYA if a_save_reg == 'Y' else Opcode.TXA
                    selector._emit_implied(op, f"Restore A from {a_save_reg}")

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

                if not arg_loc.is_hw('A'):
                    a_clobbered = True
        elif stack_args:
            # Outgoing STA path: each arg goes to a pre-computed offset.
            # If an A-resident arg exists, process it first to avoid
            # clobbering A with other args' LDA.
            offsets = []
            outgoing_offset = 1
            for arg in stack_args:
                offsets.append(outgoing_offset)
                arg_size = 1
                if arg.param_type is not None:
                    arg_size = get_type_size(arg.param_type)
                elif hasattr(arg.value, 'type_info') and arg.value.type_info:
                    arg_size = get_type_size(arg.value.type_info)
                outgoing_offset += arg_size

            # Build processing order: A-resident arg first if present
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

        # Process non-stack args. Stable-sort to prioritize A-resident
        # non-register args (STA from A before other args' LDA clobbers it).
        sorted_other_args = sorted(other_args, key=selector.arg_sort_key)
        sorted_other_args.sort(key=lambda arg: (
            0 if (arg.mechanism != ArgumentMechanism.REGISTER and
                  selector.parent._get_operand_location(arg.value).is_hw('A'))
            else 1))

        # Reorder scratch params to avoid WAR hazards: if a scratch param's
        # source value resides at another scratch param's target address,
        # the reader must be processed before the writer.
        sorted_other_args = self._reorder_scratch_params(selector, sorted_other_args)

        # Check if any non-A scratch param must precede an A-resident
        # scratch param (WAR hazard). If so, we need to:
        # 1. Save A to Y (before any non-A scratch params clobber A)
        # 2. Emit all non-A scratch params first
        # 3. Restore A from Y
        # 4. Emit A-resident scratch params last
        has_war_with_a = any(
            getattr(arg, '_needs_a_save', False)
            for arg in sorted_other_args
        )

        if has_war_with_a:
            # Split into: non-scratch args, non-A scratch, A-resident scratch
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

            # Emit non-scratch args first (register, variable)
            for arg in non_scratch:
                arg_loc = selector.parent._get_operand_location(arg.value)
                if arg_loc.kind == LocationKind.STACK and pre_arg_stack_adj != 0:
                    arg_loc = selector.parent._offset_location(arg_loc, pre_arg_stack_adj)
                if arg.mechanism == ArgumentMechanism.REGISTER:
                    selector.emit_register_argument(arg, arg_loc)
                elif arg.mechanism == ArgumentMechanism.VARIABLE:
                    selector.emit_variable_argument(arg, arg_loc)

            # Save A to Y (TAY always transfers 16 bits in x16 mode)
            selector._emit_implied(Opcode.TAY,
                                  "Save A (WAR hazard: non-A scratch before A-resident)")

            # Emit non-A scratch params (these may LDA from sources that
            # overlap A-resident param targets)
            for arg in non_a_scratch:
                arg_loc = selector.parent._get_operand_location(arg.value)
                if arg_loc.kind == LocationKind.STACK and pre_arg_stack_adj != 0:
                    arg_loc = selector.parent._offset_location(arg_loc, pre_arg_stack_adj)
                selector.emit_scratch_param_argument(arg, arg_loc)

            # Restore A from Y
            selector._emit_implied(Opcode.TYA,
                                  "Restore A (WAR hazard resolved)")

            # Emit A-resident scratch params last (STA from A)
            for arg in a_scratch:
                arg_loc = selector.parent._get_operand_location(arg.value)
                if arg_loc.kind == LocationKind.STACK and pre_arg_stack_adj != 0:
                    arg_loc = selector.parent._offset_location(arg_loc, pre_arg_stack_adj)
                selector.emit_scratch_param_argument(arg, arg_loc)
        else:
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

        # Handle SELF_Y argument last (after stack args are pushed).
        # Loads Y = self addr and sets DBR = self bank via load_y_with_far_self,
        # or just Y = self addr via load_y_with_self.
        self_y_arg = next((a for a in instr.args if a.mechanism == ArgumentMechanism.SELF_Y), None)
        if self_y_arg is not None:
            from r65.compiler.hir.types import PointerTypeInfo
            if isinstance(self_y_arg.param_type, PointerTypeInfo) and self_y_arg.param_type.is_far:
                selector.load_y_with_far_self(self_y_arg, 0)
            else:
                selector.load_y_with_self(self_y_arg, 0)

        return stack_bytes_pushed

    def _reorder_scratch_params(self, selector, args):
        """Reorder scratch params to avoid WAR (Write-After-Read) hazards.

        If scratch param A's source value lives at scratch param B's target
        address, A must be emitted before B (read before write). Uses
        topological sort; cycles are broken by emitting in original order.

        A-resident scratch params participate in the hazard analysis (their
        target addresses may overlap other params' source locations), but
        are handled specially: when a non-A-resident param must precede an
        A-resident param, the non-A param is moved before the A-resident
        param with A saved/restored around it.
        """
        from r65.compiler.mir.nodes import ArgumentMechanism
        from r65.compiler.codegen.register_alloc import LocationKind
        from r65.compiler.codegen.type_utils import get_type_size

        # Collect all scratch params, noting which are A-resident
        all_scratch = []  # list of (arg, is_a_resident)
        scratch_positions = []  # original positions in args list
        for pos, arg in enumerate(args):
            if arg.mechanism != ArgumentMechanism.SCRATCH_PARAM:
                continue
            arg_loc = selector.parent._get_operand_location(arg.value)
            all_scratch.append((arg, arg_loc.is_hw('A')))
            scratch_positions.append(pos)

        if len(all_scratch) <= 1:
            return args

        # Build target address set for ALL scratch params (including A-resident)
        target_addrs = {}  # byte_addr -> scratch index
        for i, (arg, _is_a) in enumerate(all_scratch):
            addr = arg.scratch_addr
            param_size = 1
            if arg.param_type is not None:
                param_size = get_type_size(arg.param_type)
            for byte_off in range(param_size):
                target_addrs[addr + byte_off] = i

        # Build dependency graph: arg i must come before arg j if
        # arg i's source is at an address that arg j will write to
        must_precede = {}  # i -> set of j (i must come before j)
        for i, (arg, _is_a) in enumerate(all_scratch):
            arg_loc = selector.parent._get_operand_location(arg.value)
            if arg_loc.kind != LocationKind.SCRATCH:
                continue
            src_addr = arg_loc.scratch_addr
            src_size = arg_loc.size or 1
            for byte_off in range(src_size):
                check_addr = src_addr + byte_off
                if check_addr in target_addrs:
                    j = target_addrs[check_addr]
                    if j != i:  # Don't add self-dependency
                        must_precede.setdefault(i, set()).add(j)

        if not must_precede:
            return args  # No hazards

        # Check if any non-A param must precede an A-resident param.
        # If so, we need A save/restore: move the conflicting non-A
        # readers before the A-resident writers.
        a_indices = {i for i, (_arg, is_a) in enumerate(all_scratch) if is_a}
        non_a_before_a = set()  # non-A indices that must precede an A-resident
        for i, deps in must_precede.items():
            if i not in a_indices:
                # non-A param i must precede some set of params
                for j in deps:
                    if j in a_indices:
                        non_a_before_a.add(i)

        # Topological sort of ALL scratch args
        ordered = []
        visited = set()
        in_progress = set()

        def visit(idx):
            if idx in visited:
                return
            if idx in in_progress:
                return  # Cycle — break by emitting in original order
            in_progress.add(idx)
            for dep in must_precede.get(idx, ()):
                visit(dep)
            in_progress.discard(idx)
            visited.add(idx)
            ordered.append(idx)

        for i in range(len(all_scratch)):
            visit(i)

        # visit() appends in post-order — reverse for correct order.
        ordered.reverse()

        # Rebuild args list preserving non-scratch args in place
        reordered_scratch = [all_scratch[i][0] for i in ordered]

        # If non-A params must precede A-resident params, we need to
        # wrap the moved params with A save/restore. We mark them
        # so emit_call_args can handle the save/restore.
        if non_a_before_a:
            for arg in reordered_scratch:
                idx_in_all = next(
                    i for i, (a, _) in enumerate(all_scratch) if a is arg
                )
                if idx_in_all in non_a_before_a:
                    # Mark this arg as needing A save/restore around it
                    arg._needs_a_save = True

        result = []
        scratch_iter = iter(reordered_scratch)
        scratch_pos_set = set(scratch_positions)
        for pos, arg in enumerate(args):
            if pos in scratch_pos_set:
                result.append(next(scratch_iter))
            else:
                result.append(arg)

        return result

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
        scratch_args = [arg for arg in instr.args if arg.mechanism == ArgumentMechanism.SCRATCH_PARAM]
        self_y_arg = None
        for arg in instr.args:
            if arg.mechanism == ArgumentMechanism.SELF_Y:
                self_y_arg = arg
                break

        # Emit scratch param arguments first (they write to DP, don't touch stack)
        for arg in scratch_args:
            arg_loc = selector.parent._get_operand_location(arg.value)
            selector.emit_scratch_param_argument(arg, arg_loc)

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

        # Self loading (DBR set + Y load for far self, Y load for near) is
        # delegated to the caller via emit_trait_dispatch so that chain
        # coalescing can elide the DBR-set step on MIDDLE/END members.
        # The stack-bytes-pushed adjustment is already applied centrally by
        # the resolver via spill_offset (region_state.stack_tracker), so we
        # pass 0 here — matches the pre-refactor behavior.
        if self_y_arg is not None:
            selector._pending_self_y_arg = self_y_arg
            selector._pending_self_y_stack_bytes = 0
        else:
            selector._pending_self_y_arg = None
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

        # Detect A-resident collision: if any arg's value is already in A
        # (hw-coalesced), PHA stack arg setup for OTHER args uses LDA which
        # clobbers A. Save A to a temp register and restore before the
        # A-resident arg is pushed.
        a_save_reg = None
        a_param_is_16bit = False

        if stack_args:
            # Check if any stack arg has its value in A and would be
            # clobbered during PHA loop (not last position = not first
            # in reversed order).
            a_in_stack_needs_save = False
            for i, arg in enumerate(stack_args):
                arg_loc = selector.parent._get_operand_location(arg.value)
                if arg_loc.is_hw('A'):
                    # Last position is processed first in reversed order,
                    # so A is still valid. Other positions need save.
                    if i != len(stack_args) - 1:
                        a_in_stack_needs_save = True
                    break

            # Check other_args for A-resident values (register-A or scratch/variable)
            a_in_other = False
            for arg in other_args:
                if arg.mechanism != ArgumentMechanism.REGISTER:
                    arg_loc = selector.parent._get_operand_location(arg.value)
                    if arg_loc.is_hw('A'):
                        a_in_other = True
                        break
                else:
                    target_reg = arg.location.name if hasattr(arg.location, 'name') else str(arg.location)
                    if target_reg != 'A':
                        continue
                    arg_loc = selector.parent._get_operand_location(arg.value)
                    if arg_loc.is_hw('A'):
                        a_in_other = True
                        if arg.param_type is not None:
                            a_param_is_16bit = get_type_size(arg.param_type) >= 2
                        break

            # Need save if A-resident arg would be clobbered
            if (a_in_other or a_in_stack_needs_save):
                reg_targets = set()
                for a in other_args:
                    if a.mechanism == ArgumentMechanism.REGISTER:
                        t = a.location.name if hasattr(a.location, 'name') else str(a.location)
                        reg_targets.add(t)
                if 'Y' not in reg_targets:
                    a_save_reg = 'Y'
                elif 'X' not in reg_targets:
                    a_save_reg = 'X'

        if a_save_reg:
            op = Opcode.TAY if a_save_reg == 'Y' else Opcode.TAX
            selector._emit_implied(op, f"Save A to {a_save_reg} before stack arg setup")

        # Always use PHA path for stack args (right-to-left for correct callee layout).
        # If an A-resident stack arg isn't first in reversed order, prior args'
        # LDA clobbers A. Restore from temp register when needed.
        if stack_args:
            a_clobbered = False
            for arg in reversed(stack_args):
                arg_loc = selector.parent._get_operand_location(arg.value)

                # Restore A from temp if it was clobbered by prior args' LDA
                if arg_loc.is_hw('A') and a_clobbered and a_save_reg:
                    op = Opcode.TYA if a_save_reg == 'Y' else Opcode.TXA
                    selector._emit_implied(op, f"Restore A from {a_save_reg}")

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

                if not arg_loc.is_hw('A'):
                    a_clobbered = True

        if a_save_reg:
            if a_param_is_16bit:
                current_mode = selector.parent.emitter.get_accu_mode()
                if current_mode != 16:
                    selector._emit_immediate(Opcode.REP_IMMEDIATE, M_FLAG,
                                             "m16 for A restore from index reg")
                    selector.parent.emitter.emit_accu_mode(16)
            op = Opcode.TYA if a_save_reg == 'Y' else Opcode.TXA
            selector._emit_implied(op, f"Restore A from {a_save_reg} after stack arg setup")

        # Non-stack args. Stable-sort to prioritize A-resident non-register
        # args (STA from A before other args' LDA clobbers it).
        sorted_other_args = sorted(other_args, key=selector.arg_sort_key)
        sorted_other_args.sort(key=lambda arg: (
            0 if (arg.mechanism != ArgumentMechanism.REGISTER and
                  selector.parent._get_operand_location(arg.value).is_hw('A'))
            else 1))

        # Reorder scratch params to avoid WAR hazards
        sorted_other_args = self._reorder_scratch_params(selector, sorted_other_args)

        # Check for WAR hazard with A-resident scratch params
        has_war_with_a = any(
            getattr(arg, '_needs_a_save', False)
            for arg in sorted_other_args
        )

        if has_war_with_a:
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
                arg_loc = selector.parent._get_operand_location(arg.value)
                if arg_loc.kind == LocationKind.STACK and pre_arg_stack_adj != 0:
                    arg_loc = selector.parent._offset_location(arg_loc, pre_arg_stack_adj)
                if arg.mechanism == ArgumentMechanism.REGISTER:
                    selector.emit_register_argument(arg, arg_loc)
                elif arg.mechanism == ArgumentMechanism.VARIABLE:
                    selector.emit_variable_argument(arg, arg_loc)

            selector._emit_implied(Opcode.TAY,
                                  "Save A (WAR hazard: non-A scratch before A-resident)")
            for arg in non_a_scratch:
                arg_loc = selector.parent._get_operand_location(arg.value)
                if arg_loc.kind == LocationKind.STACK and pre_arg_stack_adj != 0:
                    arg_loc = selector.parent._offset_location(arg_loc, pre_arg_stack_adj)
                selector.emit_scratch_param_argument(arg, arg_loc)
            selector._emit_implied(Opcode.TYA,
                                  "Restore A (WAR hazard resolved)")
            for arg in a_scratch:
                arg_loc = selector.parent._get_operand_location(arg.value)
                if arg_loc.kind == LocationKind.STACK and pre_arg_stack_adj != 0:
                    arg_loc = selector.parent._offset_location(arg_loc, pre_arg_stack_adj)
                selector.emit_scratch_param_argument(arg, arg_loc)
        else:
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

        # Handle SELF_Y argument last (after stack args are pushed).
        self_y_arg = next((a for a in instr.args if a.mechanism == ArgumentMechanism.SELF_Y), None)
        if self_y_arg is not None:
            from r65.compiler.hir.types import PointerTypeInfo
            if isinstance(self_y_arg.param_type, PointerTypeInfo) and self_y_arg.param_type.is_far:
                selector.load_y_with_far_self(self_y_arg, 0)
            else:
                selector.load_y_with_self(self_y_arg, 0)

        return stack_bytes_pushed

    def emit_trait_dispatch_args(self, selector, instr) -> int:
        """Default trait dispatch: always push stack args via PHA.

        Same as Default's trait dispatch but forces PHA path for stack args.
        Scratch param args are emitted first (they write to DP, not stack).
        """
        from r65.compiler.mir.nodes import ArgumentMechanism
        from r65.compiler.codegen.register_alloc import LocationKind
        from r65.compiler.codegen.type_utils import get_type_size

        stack_args = [arg for arg in instr.args if arg.mechanism == ArgumentMechanism.STACK]
        scratch_args = [arg for arg in instr.args if arg.mechanism == ArgumentMechanism.SCRATCH_PARAM]
        self_y_arg = None
        for arg in instr.args:
            if arg.mechanism == ArgumentMechanism.SELF_Y:
                self_y_arg = arg
                break

        # Emit scratch param arguments first (they write to DP, don't touch stack)
        for arg in scratch_args:
            arg_loc = selector.parent._get_operand_location(arg.value)
            selector.emit_scratch_param_argument(arg, arg_loc)

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

        # Self loading deferred to emit_trait_dispatch (chain-aware).
        # stack_bytes_pushed is tracked via region_state.stack_tracker —
        # the resolver applies it centrally; pass 0 to avoid double-count.
        if self_y_arg is not None:
            selector._pending_self_y_arg = self_y_arg
            selector._pending_self_y_stack_bytes = 0
        else:
            selector._pending_self_y_arg = None
            selector._pending_self_y_stack_bytes = 0

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
