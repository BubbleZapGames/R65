# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
ABI model abstraction for the R65 compiler.

Defines selectable ABI policies that control calling convention decisions:
- Default: PHA-based argument passing with caller cleanup (default)
- FixedStack: no stack params (hw registers + scratch only), minimal PHB-per-byte frames
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

        Frames up to this size use push-based allocation (PHB, or PHX/PHY/PHB).
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

    def emit_call_args(self, selector, instr, pre_arg_stack_adj=0) -> int:
        """Emit call arguments. Returns the bytes the stack args pushed.

        Shared across Default and FixedStack (which has no stack args). Pascal
        overrides to push all args L→R after result space.

        Args:
            selector: CallInstructionSelector instance
            instr: Call MIR instruction
            pre_arg_stack_adj: Stack offset adjustment from pre-arg operations
        """
        from r65.compiler.mir.nodes import ArgumentMechanism
        from r65.compiler.codegen.opcodes import Opcode

        stack_args = [a for a in instr.args if a.mechanism == ArgumentMechanism.STACK]
        other_args = [a for a in instr.args if a.mechanism != ArgumentMechanism.STACK]

        # The call's preserved set is exactly what _compute_hw_spills skipped
        # spilling, so it is exactly what may still hold an unsaved live value.
        preserved = selector.callee_preserves(instr)

        a_save_reg, a_param_is_16bit = (
            self._compute_a_save_for_call(selector, stack_args, other_args, preserved))

        parked_bytes = 0
        if a_save_reg == 'S':
            parked_bytes = 2 if a_param_is_16bit else 1
            _force_accu_mode(selector, 16 if a_param_is_16bit else 8)
            selector._emit_implied(Opcode.PHA, "Park A above the stack args")
            selector.region_state.stack_tracker.push(parked_bytes)
        elif a_save_reg:
            op = Opcode.TAY if a_save_reg == 'Y' else Opcode.TAX
            selector._emit_implied(op, f"Save A to {a_save_reg} before stack arg setup")

        stack_bytes_pushed = 0
        if stack_args:
            stack_bytes_pushed = self._emit_stack_args_pha(
                selector, stack_args, a_save_reg, pre_arg_stack_adj, a_param_is_16bit)

        if a_save_reg:
            self._restore_a_from(selector, a_save_reg, a_param_is_16bit, stack_bytes_pushed)

        self._emit_other_args(selector, other_args, pre_arg_stack_adj, preserved)

        self_y_arg = next(
            (a for a in instr.args if a.mechanism == ArgumentMechanism.SELF_Y), None)
        if self_y_arg is not None:
            self._emit_self_y(selector, self_y_arg)

        # A parked byte is popped with the arguments, which lie below it
        return stack_bytes_pushed + parked_bytes

    @staticmethod
    def _saved_value_size(arg) -> int:
        """Size of the value as it sits in A, not of the parameter it feeds."""
        from r65.compiler.codegen.type_utils import get_type_size
        type_info = getattr(arg.value, 'type_info', None)
        if type_info is not None:
            return get_type_size(type_info)
        return ABIModel._arg_size(arg)

    @staticmethod
    def _arg_size(arg) -> int:
        from r65.compiler.codegen.type_utils import arg_param_size
        return arg_param_size(arg)

    @staticmethod
    def _register_target_name(arg) -> str:
        return arg.location.name if hasattr(arg.location, 'name') else str(arg.location)

    def _compute_a_save_for_call(self, selector, stack_args, other_args,
                                 preserved=frozenset()):
        """Decide whether A must be saved across stack-arg setup.

        Returns (a_save_reg, a_param_is_16bit) where a_save_reg is 'Y', 'X',
        'S' (pushed above the stack args and read back stack-relative), or
        None when no save is needed.

        An index register is used only if no argument goes to it and the call
        does not preserve it (`preserved`, from callee_preserves): a register
        the callee preserves is not spilled, so a live value in it is sitting
        unsaved and TAY/TAX would destroy it before the JSR.

        A save is needed when:
          - any non-stack arg's value is in A (LDA in stack setup would
            clobber it), or
          - a stack arg's value is in A but it isn't the last one (args are
            pushed in reverse, so the LDAs of the args after it clobber A
            first).
        """
        from r65.compiler.mir.nodes import ArgumentMechanism

        if not stack_args:
            return None, False

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
            # The restore is pinned to this width whatever the mechanism: a
            # u16 scratch-promoted or variable-bound argument parked in Y
            # and restored with an 8-bit TYA comes back with B's stale high
            # byte, not its own.
            a_param_is_16bit = self._arg_size(arg) >= 2
            break

        a_stack_needs_save = (
            a_resident_stack_idx is not None
            and a_resident_stack_idx != len(stack_args) - 1
        )
        if not a_in_other and a_stack_needs_save:
            a_param_is_16bit = self._arg_size(stack_args[a_resident_stack_idx]) >= 2

        if not (a_in_other or a_stack_needs_save):
            return None, a_param_is_16bit

        reg_targets = {
            self._register_target_name(a)
            for a in other_args if a.mechanism == ArgumentMechanism.REGISTER
        }
        for reg in ('Y', 'X'):
            if reg not in reg_targets and reg not in preserved:
                return reg, a_param_is_16bit
        return 'S', a_param_is_16bit

    def _emit_stack_args_pha(self, selector, stack_args, a_save_reg, pre_arg_stack_adj,
                             a_param_is_16bit=False) -> int:
        """Push stack args right-to-left via PHA. Returns bytes pushed."""
        from r65.compiler.codegen.register_alloc import LocationKind
        from r65.compiler.codegen.opcodes import Opcode

        stack_bytes_pushed = 0
        a_clobbered = False
        for arg in reversed(stack_args):
            arg_loc = selector.parent._get_operand_location(arg.value)

            if arg_loc.is_hw('A') and a_clobbered and a_save_reg == 'S':
                self._restore_a_from(selector, 'S', a_param_is_16bit, stack_bytes_pushed)
            elif arg_loc.is_hw('A') and a_clobbered and a_save_reg:
                _force_accu_mode(
                    selector, 16 if self._saved_value_size(arg) >= 2 else 8)
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

    def _restore_a_from(self, selector, a_save_reg, a_param_is_16bit, pushed_since=0):
        """Emit TYA/TXA to restore A, pinned to the width of the saved value.

        TYA/TXA are M-sized, so the mode has to be right in both directions: too
        wide and the restore drags B in behind A, too narrow and the high byte
        is dropped. A value parked on the stack ('S') sits just above the
        `pushed_since` argument bytes pushed after it, and is read in place.
        """
        from r65.compiler.codegen.opcodes import Opcode
        _force_accu_mode(selector, 16 if a_param_is_16bit else 8)
        if a_save_reg == 'S':
            from r65.compiler.codegen.asm_nodes import StackOffset
            selector.parent.emitter.emit_instr(
                Opcode.LDA_STACK, StackOffset(pushed_since + 1),
                "Restore A parked above the stack args")
            return
        op = Opcode.TYA if a_save_reg == 'Y' else Opcode.TXA
        selector._emit_implied(op, f"Restore A from {a_save_reg} after stack arg setup")

    def _pick_a_save(self, selector, ordered, locs, width, preserved=frozenset()):
        """Where to park the '@ A' argument. See _RegASave / _StackASave.

        TAY/TAX move index-width bits regardless of M, so a register save needs
        no mode pinning when it opens — only when it restores.

        `preserved` is what the call keeps (`callee_preserves`: the callee's
        `#[preserves(...)]` and the index registers its code never writes).
        Those registers must not be used: `_compute_hw_spills` skips spilling
        them precisely because the callee leaves them, so a live value in one
        is sitting unsaved. Parking A there destroys it *before* the JSR, where
        no promise the callee can keep applies. Every other register is either spilled,
        dead, or inside an active clobber region whose start already saved it.
        """
        from r65.compiler.mir.nodes import ArgumentMechanism

        targets = {self._register_target_name(a) for a in ordered
                   if a.mechanism == ArgumentMechanism.REGISTER}
        sources = {l.hw_register for l in locs if l.is_hw()}

        for reg in ('Y', 'X'):
            if reg in targets or reg in sources or reg in preserved:
                continue
            if width == 16 and selector.parent.emitter.get_index_mode() != 16:
                continue
            return _RegASave(reg, width)

        # Always available here: nothing in this pass moves SP, and every
        # stack-relative source re-resolves against the tracker displacement.
        return _StackASave(width)

    def _emit_other_args(self, selector, other_args, pre_arg_stack_adj,
                         preserved=frozenset()):
        """Emit non-stack args, keeping an A-resident value alive across them.

        Two requirements drive the shape:

          R1  an argument whose *source* is A needs A intact when it is emitted.
              There can be several (`f(v, v)`).
          R2  the argument whose *target* is A needs A intact all the way to the
              JSR — it emits nothing at all, so the value simply has to survive
              everything else.

        The SELF_Y load that `emit_call_args` runs after this also clobbers A,
        but it cannot coexist with an `@ A` argument: SELF_Y is only produced for
        trait methods (`mir/lowerers/call.py`), and trait methods reject register
        bindings on their parameters outright.

        R2 is why ordering cannot fix this on its own, and why this walks the
        argument list once inserting a save/restore rather than partitioning by
        mechanism: a partition emits every REGISTER argument before the bracket
        opens, which is exactly where the clobber lives.

        The ordering is load-bearing; this decides only where the bracket goes.
        """
        from r65.compiler.mir.nodes import ArgumentMechanism
        from r65.compiler.codegen.register_alloc import LocationKind
        from r65.compiler.codegen.opcodes import Opcode

        other_args = self._read_a_source_first(selector, other_args)

        def is_a_resident_priority(arg):
            arg_loc = selector.parent._get_operand_location(arg.value)
            if not arg_loc.is_hw('A'):
                return False
            if arg.mechanism != ArgumentMechanism.REGISTER:
                return True
            return self._register_target_name(arg) != 'A'

        ordered = sorted(other_args, key=selector.arg_sort_key)
        ordered.sort(key=lambda a: 0 if is_a_resident_priority(a) else 1)
        ordered = self._reorder_scratch_params(selector, ordered)
        ordered, swap = self._order_index_reads(selector, ordered)

        def adjust_loc(loc):
            if loc.kind == LocationKind.STACK and pre_arg_stack_adj != 0:
                return selector.parent._offset_location(loc, pre_arg_stack_adj)
            return loc

        def emit(arg):
            if swap and arg is swap[0]:
                self._emit_index_swap(selector)
                return
            if swap and arg is swap[1]:
                return                        # the swap delivered it
            # Resolved late: a stack save shifts every stack-relative source.
            arg_loc = adjust_loc(selector.parent._get_operand_location(arg.value))
            selector.emit_argument_setup(arg, arg_loc)

        def clobbers_a(arg, loc):
            if swap and (arg is swap[0] or arg is swap[1]):
                return False                  # PHX; TYX; PLY
            return selector.arg_setup_clobbers_a(arg, loc)

        locs = [selector.parent._get_operand_location(a.value) for a in ordered]
        a_idxs = [i for i, l in enumerate(locs) if l.is_hw('A')]
        if not a_idxs:
            for arg in ordered:
                emit(arg)
            return

        last = a_idxs[-1]
        sink = (ordered[last].mechanism == ArgumentMechanism.REGISTER
                and self._register_target_name(ordered[last]) == 'A')
        span = len(ordered) if sink else last
        clobbering = [j for j in range(span)
                      if j != last
                      and clobbers_a(ordered[j], locs[j])]

        if not clobbering:
            for arg in ordered:
                emit(arg)
            return

        # The width of the value sitting in A, which is not the same question
        # as how wide the parameter it feeds is -- a u8 passed to a u16
        # parameter still occupies one byte of A. Restoring the wrong width
        # either drops the high byte or drags B back in with it.
        width = 16 if self._saved_value_size(ordered[last]) >= 2 else 8

        # Park A in B: `XBA` before the clobber, and the `@ B` argument's own
        # closing `XBA` puts it back while delivering B — so the restore is
        # free. Needs the B argument to be the only clobberer (nothing else may
        # disturb B), to load from memory rather than from A (a bare `XBA` would
        # not reload it), and an 8-bit value (in m16 B *is* its high half).
        if width == 8 and sink and len(clobbering) == 1:
            j = clobbering[0]
            if (ordered[j].mechanism == ArgumentMechanism.REGISTER
                    and self._register_target_name(ordered[j]) == 'B'
                    and not locs[j].is_hw('A')):
                _force_accu_mode(selector, 8)
                selector._emit_implied(Opcode.XBA,
                                       "Park '@ A' argument in B across setup")
                for arg in ordered:
                    emit(arg)
                return

        a_positions = set(a_idxs)
        save = self._pick_a_save(selector, ordered, locs, width, preserved)
        save.open(selector)
        dirty = False
        for i, arg in enumerate(ordered):
            if sink and i == last:
                continue                      # deferred past everything else
            if i in a_positions and dirty:
                save.close(selector, final=(i == last))
                dirty = False
            emit(arg)
            if clobbers_a(arg, locs[i]):
                dirty = True
        if sink:
            if dirty:
                save.close(selector, final=True)
            emit(ordered[last])               # normally a no-op; that is the point

    def _index_write(self, arg, loc):
        """The index register setting up *arg* overwrites, or None."""
        from r65.compiler.mir.nodes import ArgumentMechanism
        if arg.mechanism == ArgumentMechanism.REGISTER:
            target = self._register_target_name(arg)
            if target in ('X', 'Y') and not loc.is_hw(target):
                return target
        return None

    def _read_a_source_first(self, selector, args):
        """Move an `@ A` argument's X/Y source into A before anything writes it.

        A is loaded last, after X and Y, so an `@ A` argument read from an
        index register another argument is loaded into would read the new
        value. Transferred first, it is a value sitting in A like any other,
        which the bracket in _emit_other_args keeps alive to the call.

        When that other argument's own source is A, the two swap: the index
        register is pushed, takes A, and is pulled back into A (in m16, as
        it was pushed, then narrowed to the parameter).
        """
        from dataclasses import replace
        from r65.compiler.mir.nodes import ArgumentMechanism, HardwareRegister
        from r65.compiler.codegen.opcodes import Opcode
        from r65.compiler.errors import CodegenError
        locs = [selector.parent._get_operand_location(a.value) for a in args]
        writer = {self._index_write(a, l): i for i, (a, l) in enumerate(zip(args, locs))}
        for i, (arg, loc) in enumerate(zip(args, locs)):
            if not (arg.mechanism == ArgumentMechanism.REGISTER
                    and self._register_target_name(arg) == 'A'
                    and loc.is_hw() and loc.hw_register in writer):
                continue
            src, w = loc.hw_register, writer[loc.hw_register]
            in_a = [j for j, l in enumerate(locs) if l.is_hw('A')]
            width = 16 if self._arg_size(arg) >= 2 else 8
            out = list(args)
            out[i] = replace(arg, value=HardwareRegister('A'))
            if not in_a:
                _force_accu_mode(selector, width)
                op = Opcode.TXA if src == 'X' else Opcode.TYA
                selector._emit_implied(op, "`@ A` argument read before its source is reloaded")
                return out
            if in_a != [w] or any(self._register_target_name(a) == 'B' for a in args
                                  if a.mechanism == ArgumentMechanism.REGISTER):
                raise CodegenError(
                    "internal: an `@ A` argument read from an index register "
                    "another argument overwrites, while A also feeds a third")
            selector._emit_push(src, f"Swap A and {src} arguments")
            selector.region_state.stack_tracker.push(2)
            selector.emit_argument_setup(args[w], locs[w])
            _force_accu_mode(selector, 16)
            selector._emit_pull('A')
            selector.region_state.stack_tracker.pop(2)
            _force_accu_mode(selector, width)
            del out[w]
            return out
        return args

    def _order_index_reads(self, selector, ordered):
        """*ordered*, with every read of X or Y before the argument that loads it.

        Register arguments load in rank order (B, X/Y, A) and A-resident
        sources go first, neither of which sees that one argument's source
        may be another's target: `f(Y, M)` into `a @ X, b @ Y` loaded Y and
        passed it twice. An argument that writes X or Y now waits for every
        other reader of it, and otherwise keeps its place. X and Y each read
        by the other's argument is a swap, done in one step; the returned
        (X argument, Y argument) pair marks it.
        """
        from r65.compiler.errors import CodegenError
        locs = [selector.parent._get_operand_location(a.value) for a in ordered]
        reads = [{r for r in ('X', 'Y') if l.is_hw(r)} for l in locs]
        writes = [self._index_write(a, l) for a, l in zip(ordered, locs)]
        xi = next((i for i, w in enumerate(writes) if w == 'X'), None)
        yi = next((i for i, w in enumerate(writes) if w == 'Y'), None)
        swap = None
        if xi is not None and yi is not None and 'Y' in reads[xi] and 'X' in reads[yi]:
            swap = (ordered[xi], ordered[yi])
        unit = {xi, yi} if swap else set()

        pending, result = list(range(len(ordered))), []
        while pending:
            for i in pending:
                if i in unit:
                    blocked = any(reads[j] for j in pending if j not in unit)
                else:
                    blocked = writes[i] is not None and any(
                        writes[i] in reads[j] for j in pending if j != i)
                if not blocked:
                    break
            else:
                raise CodegenError("internal: register arguments read one "
                                   "another's targets in a cycle")
            picked = sorted(unit) if i in unit else [i]
            for k in picked:
                pending.remove(k)
                result.append(ordered[k])
        return result, swap

    @staticmethod
    def _emit_index_swap(selector):
        """X and Y each take the other's value: A, B and the mode untouched."""
        selector._emit_push('X', "Swap X and Y arguments")
        selector.region_state.stack_tracker.push(2)
        selector._emit_transfer('Y', 'X')
        selector._emit_pull('Y')
        selector.region_state.stack_tracker.pop(2)

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

        Returns the reordered argument list.
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
            return args

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
            return args

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
        return result

    def emit_trait_dispatch_args(self, selector, instr) -> int:
        """Emit trait dispatch arguments. Returns stack_bytes_pushed.

        Scratch params first (they write to DP, not stack), then stack args.
        Stack args are pushed right-to-left via PHA.
        SELF_Y is deferred to the caller via pending fields so chain
        coalescing can elide the DBR-set on MIDDLE/END members.
        """
        from r65.compiler.mir.nodes import ArgumentMechanism

        stack_args = [a for a in instr.args if a.mechanism == ArgumentMechanism.STACK]
        scratch_args = [a for a in instr.args if a.mechanism == ArgumentMechanism.SCRATCH_PARAM]
        self_y_arg = next(
            (a for a in instr.args if a.mechanism == ArgumentMechanism.SELF_Y), None)

        for arg in scratch_args:
            arg_loc = selector.parent._get_operand_location(arg.value)
            selector.emit_scratch_param_argument(arg, arg_loc)

        stack_bytes_pushed = 0
        for arg in reversed(stack_args):
            arg_loc = selector.parent._get_operand_location(arg.value)
            arg_size = self._arg_size(arg)
            selector.emit_pha_stack_argument(arg, arg_loc, arg_size)
            stack_bytes_pushed += arg_size
            selector.region_state.stack_tracker.push(arg_size)

        # stack_bytes_pushed is tracked via region_state.stack_tracker; the
        # resolver applies it centrally, so pending_self_y_stack_bytes stays 0.
        selector._pending_self_y_arg = self_y_arg
        selector._pending_self_y_stack_bytes = 0

        return stack_bytes_pushed

    def emit_return_values(self, selector, instr):
        """Move a Return's values into their return registers, then switch
        to the exit mode.

        Concrete on the base class — Default and FixedStack share identical
        register-based logic. Pascal overrides to write to stack result space.

        The values are one parallel move, each read as it stood at the Return;
        _ReturnMarshaller orders it.

        Args:
            selector: ControlFlowInstructionSelector instance
            instr: Return MIR instruction
        """
        from r65.compiler.errors import InstructionSelectionError

        if not instr.values:
            return

        return_registers = selector._get_return_register_order()
        if len(instr.values) > len(return_registers):
            raise InstructionSelectionError(
                f"Too many return values (max {len(return_registers)})",
                source_loc=selector.parent._current_source_loc)

        _ReturnMarshaller(selector, instr.values, return_registers).run()
        selector._switch_to_exit_mode()

    def __repr__(self):
        return f"ABIModel({self.kind.value})"



class _ReturnMove:
    """One return value bound for its register; *pushed* is the byte count
    once its source has been parked on the stack."""

    __slots__ = ('index', 'value', 'target', 'width', 'pushed')

    def __init__(self, index, value, target, width):
        self.index = index
        self.value = value
        self.target = target
        self.width = width
        self.pushed = None


class _ReturnMarshaller:
    """Schedule a Return's moves into A, B, X and Y as one parallel move.

    Invariant: a move is emitted only when nothing it writes is read by
    another pending move or already holds a final value. A 16-bit value in A
    occupies B too. Loads into X/Y from the stack go through A, and B is
    written through A (XBA), so X and Y are filled first, then B, then A.
    When every remaining move is blocked, a blocking register source is
    pushed and later pulled straight into its target (LIFO); the stack
    tracker shifts stack-relative sources over the pushed bytes meanwhile.
    """

    def __init__(self, selector, values, return_registers):
        from r65.compiler.hir.types import MultiReturnTypeInfo
        from r65.compiler.codegen.type_utils import scalar_width
        from r65.compiler.typeck.processor_mode import ModeState

        self.sel = selector
        self.isel = selector.parent
        self.tracker = selector.parent.call_selector.region_state.stack_tracker
        func = selector.current_function
        ret_type = func.return_type if func else None
        exit_width = 2 if (func and func.exit_m_mode == ModeState.M16) else 1
        self.moves = []
        for i, value in enumerate(values):
            if isinstance(ret_type, MultiReturnTypeInfo):
                width = scalar_width(ret_type.element_types[i])
            else:
                width = scalar_width(ret_type)
            if width not in (1, 2):
                width = exit_width
            self.moves.append(_ReturnMove(i, value, return_registers[i], width))
        self.parked = []   # moves whose source is on the stack, push order
        self.done = set()  # registers holding a final value

    # -- facts about a move ------------------------------------------------

    def _loc(self, m):
        return self.isel._get_operand_location(m.value)

    def _in_place(self, m):
        return m.pushed is None and self._loc(m).is_hw(m.target)

    def _reads(self, m):
        if m.pushed is not None:
            return set()
        loc = self._loc(m)
        if not loc.is_hw():
            return set()
        if loc.hw_register == 'A' and m.width == 2:
            return {'A', 'B'}
        return {loc.hw_register}

    def _writes(self, m):
        """Registers emitting *m* writes, in the current accumulator mode."""
        from r65.compiler.codegen.register_alloc import LocationKind
        t = m.target
        if m.pushed is not None:
            if t in ('X', 'Y'):
                return {t}
            return {'A', 'B'} if (t == 'B' or m.pushed == 2) else {'A'}
        loc = self._loc(m)
        if t == 'B' or (t == 'A' and (m.width == 2 or loc.is_hw('B'))):
            return {'A', 'B'}
        if t == 'A':
            return {'A'}
        if loc.is_hw():
            src = loc.hw_register
            if src in ('X', 'Y'):
                return {t}
            if src == 'A':
                # An 8-bit A moving from m8 is zero-extended through B.
                if m.width == 1 and self.isel.emitter.get_accu_mode() != 16:
                    return {t, 'B'}
                return {t}
            return {'A', 'B', t}
        if loc.kind == LocationKind.STACK:
            return {'A', 'B', t}
        return {t}

    def _final(self, m):
        return {'A', 'B'} if (m.target == 'A' and m.width == 2) else {m.target}

    # -- emission ----------------------------------------------------------

    def _set_width(self, width):
        if width == 2:
            self.isel._ensure_m16_mode()
        else:
            self.isel._ensure_m8_mode()

    def _emit(self, m):
        from r65.compiler.codegen.register_alloc import LocationKind
        from r65.compiler.codegen.opcodes import Opcode
        isel, t = self.isel, m.target
        if m.pushed is not None:
            self._pull(m)
            return
        loc = self._loc(m)
        if loc.is_hw(t):
            return
        via_a = loc.kind == LocationKind.STACK
        if t == 'A':
            self._set_width(m.width)
            if loc.is_hw():
                isel._emit_register_transfer(loc.hw_register, 'A')
            else:
                isel._emit_load('LDA', loc)
        elif t == 'B':
            if loc.is_hw() and not loc.is_hw('A'):
                isel._emit_register_transfer(loc.hw_register, 'A')
            elif not loc.is_hw():
                isel._emit_load('LDA', loc)
            isel._store_to_b_from_a()
        elif loc.is_hw():
            if loc.hw_register == 'A' and m.width == 2:
                isel._ensure_m16_mode()
            isel._emit_register_transfer(loc.hw_register, t)
        elif via_a:
            # No LDX/LDY sr,S. Stack loads use m16: TAX/TAY in m8/x16 move
            # B:A, and B may hold stale data.
            if loc.kind == LocationKind.STACK or m.width == 2:
                isel._ensure_m16_mode()
            isel._emit_load('LDA', loc)
            self.sel._emit_implied(Opcode.TAX if t == 'X' else Opcode.TAY,
                                   f"Transfer to {t}")
        else:
            isel._emit_load('LDX' if t == 'X' else 'LDY', loc)

    def _park(self, m):
        from r65.compiler.codegen.opcodes import Opcode
        src = self._loc(m).hw_register
        if src == 'A':
            n = 2 if (m.width == 2 or m.target in ('X', 'Y')) else 1
            self._set_width(n)
            self.sel._emit_implied(Opcode.PHA, "Park return value")
        else:
            n = 2
            self.sel._emit_implied(Opcode.PHX if src == 'X' else Opcode.PHY,
                                   "Park return value")
        self.tracker.push(n)
        m.pushed = n
        self.parked.append(m)

    def _pull(self, m):
        from r65.compiler.codegen.opcodes import Opcode
        if m.target in ('X', 'Y'):
            self.sel._emit_implied(Opcode.PLX if m.target == 'X' else Opcode.PLY,
                                   "Pull parked return value")
        else:
            self._set_width(m.pushed)
            self.sel._emit_implied(Opcode.PLA, "Pull parked return value")
            self.isel._invalidate_xba_state()
            if m.target == 'B':
                self.isel._store_to_b_from_a()
        self.tracker.pop(m.pushed)
        self.parked.pop()

    # -- scheduling --------------------------------------------------------

    def run(self):
        from r65.compiler.errors import InstructionSelectionError
        rank = {'X': 0, 'Y': 0, 'B': 1, 'A': 2}
        pending = sorted(self.moves, key=lambda m: (rank[m.target], -m.index))

        def blocked_by(m):
            guarded = set(self.done)
            for p in pending:
                if p is not m:
                    guarded |= self._reads(p)
            return self._writes(m) & guarded

        while not all(self._in_place(m) for m in pending):
            xy_left = any(m.target in ('X', 'Y') and m.pushed is None
                          and not self._in_place(m) for m in pending)
            candidates = [
                m for m in pending
                if not self._in_place(m)
                and (m.pushed is None or m is self.parked[-1])
                and not (m.target in ('A', 'B') and xy_left)]
            ready = next((m for m in candidates if not blocked_by(m)), None)
            if ready is not None:
                self._emit(ready)
                pending.remove(ready)
                self.done |= self._final(ready)
                continue
            # Every candidate is blocked: park a register source that blocks one.
            victim = None
            for m in candidates:
                blocking = blocked_by(m)
                victim = next(
                    (p for p in pending
                     if p is not m and p.pushed is None
                     and self._loc(p).is_hw()
                     and self._loc(p).hw_register in ('A', 'X', 'Y')
                     and self._reads(p) & blocking), None)
                if victim is not None:
                    break
            if victim is None:
                raise InstructionSelectionError(
                    "Cannot order the moves of the return values into their "
                    "registers",
                    source_loc=self.isel._current_source_loc)
            self._park(victim)


class ABIFixedStack(ABIModel):
    """Fixed-stack ABI with no dynamic frame allocation (--abi FixedStack).

    All parameters are promoted to hardware registers or direct-page scratch
    locations — no stack-passed parameters are permitted, which leaves the
    frame layout to just preserves and a minimal local frame.

    Frame allocation uses PHB per byte, except TSC/SBC/TCS when
    force_direct_stack (interrupt handlers); deallocation always uses PLA per byte (max_pla_dealloc_size is
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
    is used.

    The caller pushes result space (sized to the return type) onto the stack
    before any parameters. After the call returns, the callee has cleaned
    up the parameter bytes, leaving just the result space at TOS for the
    caller to PLA.

    Frame allocation uses PHB per byte for small frames (<=4) and
    TSC/SEC/SBC/TCS for larger ones. The callee writes
    its return value into the result space via STA offset,S before the
    epilogue, then removes parameter bytes (but not result space) using
    the existing PLX/TSC/ADC/TCS/PHX/RTS cleanup machinery.
    """

    def __init__(self):
        super().__init__(ABIKind.PASCAL)

    def run_param_analysis(self, mir_program, scratch_pool, disable_scratch_params: bool):
        # No scratch promotion — all params stay on stack
        pass

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
        abi_info = func.abi_info
        result_offset = (frame_size + abi_info.prologue_stack_bytes
                         + abi_info.return_addr_size + total_param_bytes + 1)

        value = instr.values[0]
        value_loc = selector.parent._get_operand_location(value)

        if result_bytes == 2:
            current_mode = selector.parent.emitter.get_accu_mode()
            if current_mode != 16:
                selector._emit_immediate(Opcode.REP_IMMEDIATE, M_FLAG,
                                         "16-bit A for Pascal result store")
                selector.parent.emitter.emit_accu_mode(16)

        # A loop-promoted value can sit in X or Y; those are transfers, not loads.
        if value_loc.is_hw('X'):
            selector._emit_implied(Opcode.TXA, "Return value from X")
        elif value_loc.is_hw('Y'):
            selector._emit_implied(Opcode.TYA, "Return value from Y")
        elif not value_loc.is_hw('A'):
            selector.parent._emit_load('LDA', value_loc)
        selector.emitter.emit_instr(
            Opcode.STA_STACK, StackOffset(result_offset),
            "Store return value to Pascal result space")


class ABIDefault(ABIModel):
    """Default calling convention with PHA-based argument passing (--abi Default).

    Stack arguments are pushed via PHA before each call and cleaned up by
    the caller after the call returns. Parameter passing uses a
    hybrid mechanism (register, variable-bound, scratch-promoted, and
    stack).

    Frame allocation uses register pushes (PHX/PHY for 2-byte chunks,
    PHB for 1-byte remainder) for frames up to 8 bytes, and
    TSC/SEC/SBC/TCS for larger frames. PHX/PHY push 2 bytes at 4 cycles
    vs 2×PHB at 6 cycles, and neither clobbers A.

    Return values are passed in hardware registers (A, B, X, Y). The
    caller pops its pushed arguments into whichever of X/Y holds no result
    (CallSelector._emit_caller_arg_cleanup).
    """

    def __init__(self):
        super().__init__(ABIKind.DEFAULT)

    def run_param_analysis(self, mir_program, scratch_pool, disable_scratch_params: bool):
        if not disable_scratch_params:
            from r65.compiler.analysis.scratch_params import analyze_scratch_params
            analyze_scratch_params(mir_program, scratch_pool)

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


# Singleton instances
ABI_FIXED_STACK = ABIFixedStack()
ABI_PASCAL = ABIPascal()
ABI_DEFAULT = ABIDefault()


_ABI_BY_NAME = {abi.kind.value: abi for abi in (ABI_DEFAULT, ABI_FIXED_STACK, ABI_PASCAL)}


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


def emit_d_equals_s_keeping_c(emitter):
    """Point D at S, as S stands after a PHD, without losing C (A and B).

    TSC writes all of C, so C is parked with a 16-bit PHA - two bytes in any
    entry mode - and the two INCs step D back over it. The emitter's tracked
    width is restored; an unknown width ends in m8.
    """
    from r65.compiler.codegen.opcodes import Opcode
    from r65.compiler.codegen.asm_nodes import Immediate
    from r65.compiler.codegen.constants import M_FLAG
    entry_bits = emitter.get_accu_mode()
    if entry_bits != 16:
        emitter.emit_instr(Opcode.REP_IMMEDIATE, Immediate(M_FLAG), "16-bit A to park C")
        emitter.emit_accu_mode(16)
    emitter.emit_instr(Opcode.PHA, None, "Park C across D=S setup")
    emitter.emit_instr(Opcode.TSC, None, "Transfer Stack to A")
    emitter.emit_instr(Opcode.INC)
    emitter.emit_instr(Opcode.INC, None, "Skip the parked C")
    emitter.emit_instr(Opcode.TCD, None, "D = S")
    emitter.emit_instr(Opcode.PLA, None, "Restore C")
    if entry_bits != 16:
        emitter.emit_instr(Opcode.SEP_IMMEDIATE, Immediate(M_FLAG), "Back to 8-bit A")
        emitter.emit_accu_mode(8)


def _force_accu_mode(selector, bits):
    """Pin the accumulator to `bits`, in whichever direction is needed.

    Narrowing matters as much as widening: a `TYA` in m16 writes 16 bits into
    A, and the high half *is* B — which may hold a `@ B` argument.
    """
    from r65.compiler.codegen.opcodes import Opcode
    from r65.compiler.codegen.constants import M_FLAG
    if selector.parent.emitter.get_accu_mode() == bits:
        return
    op = Opcode.REP_IMMEDIATE if bits == 16 else Opcode.SEP_IMMEDIATE
    selector._emit_immediate(op, M_FLAG,
                             f"m{bits} pinned for '@ A' argument save")
    selector.parent.emitter.emit_accu_mode(bits)


class _RegASave:
    """Park A in an index register: TAY/TAX out, TYA/TXA back.

    The transfer out moves index-width bits whatever M says, so only the
    restore needs its mode pinned.
    """

    def __init__(self, reg, width):
        self.reg = reg
        self.width = width

    def open(self, selector):
        from r65.compiler.codegen.opcodes import Opcode
        op = Opcode.TAY if self.reg == 'Y' else Opcode.TAX
        selector._emit_implied(op, f"Save '@ A' argument in {self.reg}")

    def close(self, selector, final=True):
        from r65.compiler.codegen.opcodes import Opcode
        _force_accu_mode(selector, self.width)
        op = Opcode.TYA if self.reg == 'Y' else Opcode.TXA
        selector._emit_implied(op, f"Restore '@ A' argument from {self.reg}")


class _StackASave:
    """Park A on the stack. PHA/PLA are M-sized, so both ends pin the mode.

    The push is tracked, so stack-relative argument sources resolve against the
    shifted displacement automatically — they must therefore be resolved after
    open(), not before.
    """

    def __init__(self, width):
        self.width = width
        self.bytes = 2 if width == 16 else 1

    def open(self, selector):
        from r65.compiler.codegen.opcodes import Opcode
        _force_accu_mode(selector, self.width)
        selector._emit_implied(Opcode.PHA, "Save '@ A' argument (no free index register)")
        selector.region_state.stack_tracker.push(self.bytes)

    def close(self, selector, final=True):
        from r65.compiler.codegen.opcodes import Opcode
        _force_accu_mode(selector, self.width)
        if not final:
            # Needs three or more simultaneously A-resident arguments *and*
            # both index registers taken. Not constructible today: `@ X`/`@ Y`
            # parameters must be u16, so feeding them an 8-bit A-resident value
            # takes an `as u16` cast, and the cast spills it to a frame slot --
            # at which point it is no longer A-resident. Raise rather than
            # invent a second PLA and unbalance the frame.
            raise NotImplementedError(
                "multiple restores from a stack-parked '@ A' argument")
        selector._emit_implied(Opcode.PLA, "Restore '@ A' argument")
        selector.region_state.stack_tracker.pop(self.bytes)
