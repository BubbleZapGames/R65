# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Accumulator width demand: the m-flag width codegen will want for each MIR
instruction, and the width it leaves the CPU in.

The MIR mode tracker (mir/mode_tracker.py) follows explicit SetMode only,
so in an m8 function every block is "m8" and codegen restores m8 at each
block entry - often straight into a REP for the block's u16 work, and in
a loop on every pass. This module predicts what codegen does instead, so
block entry widths can be chosen where the code wants them.

`instruction_width(instr, func)` gives `(need, leaves)`:

    need    8 or 16: the width the instruction's first accumulator-width
            dependent operation runs in (codegen switches to it first if
            the CPU is in the other); None: it reads no m flag (index
            register work, branches).
    leaves  8 or 16: the width codegen leaves after it; None: unchanged.

Codegen switches lazily (`_ensure_m8_mode`/`_ensure_m16_mode`) and never
back, so most instructions both need and leave the width of their type.
The prediction is a cost model only: codegen still switches before every
instruction that needs a width, so a wrong guess costs a SEP/REP, never
wrong code.
"""

from typing import Dict, Optional, Tuple

from r65.compiler.codegen.type_utils import scalar_width
from r65.compiler.mir.nodes import (
    AggregateCopy, BankByte, BinaryOp, BitTest, BlockCopy, CallBase, Compare,
    CondBranch, HardwareRegister, Immediate, IncDecMemory, InlineAsm, Jump,
    Load, LoadIndirect, LookupTable, MemoryLocation, MIRFunction, Move, Pull,
    Push, Return, ReturnFromInterrupt, Rotate, SetMode, StatusFlagSet,
    StatusFlagTest, Store, StoreIndirect, ToBool, TypeConvert, UnaryOp,
    VirtualRegister,
)
from r65.compiler.typeck.processor_mode import ModeState

M_FLAG = 0x20

Width = Optional[int]


def _bits(type_info) -> int:
    """The width a scalar of *type_info* is worked on in A: 16 for two
    bytes, else 8 (a far pointer's bank byte goes last, in m8)."""
    return 16 if scalar_width(type_info) == 2 else 8


def _mode_bits(state, default: int = 8) -> int:
    if state is None:
        return default
    return 16 if state == ModeState.M16 else 8


def _index_regs(func: MIRFunction) -> set:
    """Ids of the vregs codegen keeps in X or Y."""
    promoted = getattr(func, 'loop_promoted_hw_vregs', None) or {}
    return {v.id for v in promoted.values()}


def _in_index(value, index_ids: set) -> bool:
    """Whether *value* lives in X or Y (or is a constant loaded there)."""
    if isinstance(value, HardwareRegister):
        return value.name in ('X', 'Y')
    if isinstance(value, VirtualRegister):
        return value.id in index_ids
    return False


def _on_stack(on_stack, value) -> bool:
    return (on_stack is not None and isinstance(value, VirtualRegister)
            and on_stack(value))


def instruction_width(instr, func: MIRFunction, index_ids: Optional[set] = None,
                      prev=None, on_stack=None) -> Tuple[Width, Width]:
    """`(need, leaves)` for *instr*, which follows *prev* in its block (None
    at the block's start); see the module docstring. *on_stack(vreg)* says
    whether register allocation put a vreg in a stack slot (codegen passes
    it; without it, none is assumed to be)."""
    if index_ids is None:
        index_ids = _index_regs(func)
    cls = type(instr)

    if cls in (Jump, StatusFlagSet, ReturnFromInterrupt):
        return None, None
    if cls is CondBranch:
        if instr.condition is None:
            return None, None
        bits = _bits(getattr(instr.condition, 'type_info', None))
        return bits, bits
    if cls is SetMode:
        if not instr.mask & M_FLAG:
            return None, None
        return None, 8 if instr.is_set else 16
    if cls is StatusFlagTest:
        from r65.compiler.hir.status_flags import is_branchable_flag
        if is_branchable_flag(instr.flag_name):
            return None, None
        return 8, 8

    if cls is Compare:
        if _in_index(instr.left, index_ids):
            return None, None        # CPX/CPY follow the x flag
        bits = _bits(instr.type_info)
        if (bits == 16 and instr.comparison in ('==', '!=')
                and isinstance(instr.right, Immediate) and instr.right.value == 0
                and isinstance(instr.left, VirtualRegister)
                and getattr(prev, 'dest', None) == instr.left):
            return None, None        # the Z computing the value left
        return bits, bits

    if cls is Move:
        source_free = (isinstance(instr.source, Immediate)
                       or _in_index(instr.source, index_ids))
        word = scalar_width(instr.type_info) == 2
        if _in_index(instr.dest, index_ids):
            # LDX #, TXY, TAX of a value just computed in A, LDX dp; only a
            # stack slot (no LDX d,S) goes through LDA.
            if source_free or (word and (
                    getattr(prev, 'dest', None) == instr.source
                    or not _on_stack(on_stack, instr.source))):
                return None, None
        elif (word and _in_index(instr.source, index_ids)
                and isinstance(instr.dest, VirtualRegister)
                and not _on_stack(on_stack, instr.dest)):
            return None, None        # STX dp
        bits = _bits(instr.type_info)
        return bits, bits

    if cls is BinaryOp:
        width = scalar_width(instr.type_info)
        if (_in_index(instr.dest, index_ids) and _in_index(instr.left, index_ids)
                and instr.op in ('+', '-') and isinstance(instr.right, Immediate)
                and instr.right.value in (1, 2)):
            return None, None        # INX/DEY
        if width == 3:
            return 16, 8
        bits = _bits(instr.type_info)
        return bits, bits

    if cls is Store:
        if (_in_index(instr.source, index_ids)
                and scalar_width(instr.type_info) == 2):
            return None, None        # STX/STY
        bits = _bits(instr.type_info)
        return bits, bits

    if cls is TypeConvert:
        if scalar_width(instr.source_type) == 3:
            return 8, 8              # from a far pointer: byte by byte
        bits = _bits(instr.target_type)
        return bits, bits

    if cls in (Load, LoadIndirect, StoreIndirect, IncDecMemory, UnaryOp,
               BankByte, Rotate, ToBool, BitTest):
        bits = _bits(getattr(instr, 'type_info', None))
        return bits, bits

    if isinstance(instr, CallBase):
        need = _mode_bits(getattr(instr, 'callee_entry_m_mode', None))
        leaves = _mode_bits(getattr(instr, 'callee_exit_m_mode', None))
        args = getattr(instr, 'args', None)
        if args:
            # Argument setup comes first, in the first argument's width.
            need = _bits(getattr(args[0], 'type_info', None))
        return need, leaves

    if cls is Return:
        exit_bits = _mode_bits(func.exit_m_mode)
        if instr.values:
            value = instr.values[0]
            type_info = getattr(value, 'type_info', None) or func.return_type
            return _bits(type_info), exit_bits
        return None, exit_bits

    if cls is BlockCopy:
        return 16, 8
    if cls is AggregateCopy:
        return 16, 16
    if cls in (Push, Pull):
        if instr.register.name in ('A', 'B'):
            bits = _mode_bits(func.entry_m_mode)
            return bits, bits
        return None, None
    if cls is InlineAsm:
        # Inline asm is opaque: it runs in the function's width, as it
        # always has, and leaves whatever its own SEP/REP set.
        bits = _mode_bits(func.entry_m_mode)
        return bits, _asm_leaves(instr) or bits

    # LookupTable, JumpTable, MemoryFill, StatusFlagRead and the rest: m8.
    return 8, 8


def _asm_leaves(instr: InlineAsm) -> Width:
    """The width the last M-setting SEP/REP in *instr*'s text leaves."""
    import re
    leaves = None
    for line in instr.instructions:
        text = line.strip().upper()
        if not text.startswith(('SEP', 'REP')):
            continue
        match = re.search(r'#\$([0-9A-F]+)', text) or re.search(r'#(\d+)', text)
        if match is None:
            continue
        value = int(match.group(1), 16 if '$' in text else 10)
        if value & M_FLAG:
            leaves = 8 if text.startswith('SEP') else 16
    return leaves


def block_width(block, func: MIRFunction, index_ids: Optional[set] = None,
                on_stack=None) -> Tuple[Width, Width]:
    """`(first_need, exit)` for *block*: the width its first width-reading
    instruction wants, and the width it ends in (None: its entry width)."""
    if index_ids is None:
        index_ids = _index_regs(func)
    first = None
    exit_width = None
    prev = None
    for instr in block.instructions:
        need, leaves = instruction_width(instr, func, index_ids, prev, on_stack)
        prev = instr
        if first is None and exit_width is None and need is not None:
            first = need
        if leaves is not None:
            exit_width = leaves
        elif need is not None:
            exit_width = need
    return first, exit_width


# ----------------------------------------------------------------------------
# Entry-width assignment
# ----------------------------------------------------------------------------

# A switch inside a loop runs about this many times per run of the code
# around it.
_LOOP_WEIGHT = 10
_MAX_DEPTH = 4
_MAX_ROUNDS = 20


def _loop_depths(func: MIRFunction) -> Dict[int, int]:
    """Loop nesting depth per block: one per loop, however many back edges
    (`continue`) it has."""
    from r65.compiler.analysis.loop_analysis import find_loops
    bodies: Dict[int, set] = {}
    for header, body in find_loops(func):
        bodies.setdefault(header, set()).update(body)
    depth: Dict[int, int] = {}
    for body in bodies.values():
        for bid in body:
            depth[bid] = depth.get(bid, 0) + 1
    return depth


class WidthAssignment:
    """Entry widths for a function's blocks, chosen to run the fewest
    SEP/REP: `entry[block_id]` is 8 or 16.

    A switch is needed at the top of a block whose first width-reading
    instruction wants the other width, and on an edge whose source ends
    in a width its target does not enter in. Each is weighted by how
    often it runs - `_LOOP_WEIGHT` per loop level, an edge at the shallower
    of its ends - and the widths are improved one block at a time until
    no single change lowers the total. The entry block keeps the
    function's entry width.
    """

    def __init__(self, func: MIRFunction, on_stack=None):
        self.func = func
        index_ids = _index_regs(func)
        self.summary = {bid: block_width(block, func, index_ids, on_stack)
                        for bid, block in func.blocks.items()}
        depths = _loop_depths(func)
        self.weight = {bid: _LOOP_WEIGHT ** min(depths.get(bid, 0), _MAX_DEPTH)
                       for bid in func.blocks}
        entry_block = func.blocks[func.entry_block_id]
        self.entry_bits = _mode_bits(
            getattr(entry_block.entry_mode, 'm_mode', None) or func.entry_m_mode)
        # From the function's width everywhere, and from each block's first
        # need; the second only when strictly cheaper, so a tie keeps the
        # function's width (and the switch inside the block, not a REQUIRED
        # one at its top).
        self.entry = {bid: self.entry_bits for bid in func.blocks}
        self._improve()
        plain, plain_cost = self.entry, self.cost()
        self.entry = {bid: first if first is not None else self.entry_bits
                      for bid, (first, _exit) in self.summary.items()}
        self.entry[func.entry_block_id] = self.entry_bits
        self._improve()
        if self.cost() >= plain_cost:
            self.entry = plain

    def exit_width(self, bid: int) -> int:
        exit_width = self.summary[bid][1]
        return exit_width if exit_width is not None else self.entry[bid]

    def _edges(self, bid: int):
        block = self.func.blocks[bid]
        for pred in block.predecessors:
            if pred in self.func.blocks:
                yield pred, bid
        for succ in block.successors:
            if succ in self.func.blocks:
                yield bid, succ

    def _edge_cost(self, src: int, dst: int) -> int:
        if self.exit_width(src) == self.entry[dst]:
            return 0
        return min(self.weight[src], self.weight[dst])

    def _local_cost(self, bid: int) -> int:
        """The cost of every switch *bid*'s entry width decides."""
        first = self.summary[bid][0]
        cost = self.weight[bid] if first is not None and first != self.entry[bid] else 0
        return cost + sum(self._edge_cost(s, d) for s, d in set(self._edges(bid)))

    def cost(self) -> int:
        """The weighted number of switches the assignment runs."""
        total = 0
        for bid, block in self.func.blocks.items():
            first = self.summary[bid][0]
            if first is not None and first != self.entry[bid]:
                total += self.weight[bid]
            for succ in block.successors:
                if succ in self.func.blocks:
                    total += self._edge_cost(bid, succ)
        return total

    def _improve(self) -> None:
        for _ in range(_MAX_ROUNDS):
            changed = False
            for bid in self.func.blocks:
                if bid == self.func.entry_block_id:
                    continue
                before = self._local_cost(bid)
                self.entry[bid] ^= 8 ^ 16
                if self._local_cost(bid) < before:
                    changed = True
                else:
                    self.entry[bid] ^= 8 ^ 16
            if not changed:
                return
