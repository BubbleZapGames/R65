# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Index-Stream Splitting Pass.

The MIR builder indexes every static access through X, setting X right
before it. A statement that touches two arrays at two different indexes -
`oam_buffer.set!(spr, particles[i].x, ...)` copying a pool field into the
OAM buffer - therefore reloads X back and forth::

    X = Move %i              X = Move %i
    %a = Load [PX,X]         %a = Load [PX,X]
    %t = %spr << 2           %t = %spr << 2
    X = Move %t        ->    Y = Move %t
    Store %a -> [OAM,X]      Store %a -> [OAM,Y]
    X = Move %i              %b = Load [PY,X]
    %b = Load [PY,X]         Store %b -> [OAM+1,Y]
    %u = %spr << 2
    X = Move %u
    Store %b -> [OAM+1,X]

This pass moves one of the streams to Y: within a block, the index value
(keyed by a block-local value number, so the two `%spr << 2` are one) set
into X more than once, interleaved with another, gets Y instead. Its later
X loads go, X loads that then find X already holding their value go, and
so do the index computations nothing reads any more.

What Y can take is narrow: the 65816 indexes by Y only at absolute (and,
for LDX/STX, direct page) addresses - there is no long,Y - and only LDA,
STA, ADC, SBC, AND, ORA, EOR and CMP have abs,Y. So every access of the
stream must be one of those on a zeropage/lowram static. Y must also be
free across the stream: nothing in between may touch it (a call, inline
asm, a pointer access's `LDY #offset`, a Y-bound value), and nothing may
read the Y that was there before.

It runs after loop register promotion, so it sees who holds Y. A loop's
induction variable - what its header compares - keeps Y: it is stepped
every iteration, and moving it to a stack slot would cost more than the
stream saves on the iterations that run it. Any other value promoted to Y
(a sprite slot stepped only when one is drawn) gives Y up to the stream.
"""

from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional

from r65.compiler.analysis.loop_analysis import find_loops
from r65.compiler.mir.liveness import (
    InstructionLivenessAnalyzer, liveness_defs, liveness_uses,
)
from r65.compiler.mir.nodes import (
    CLOBBERS_ALL, BinaryOp, Compare, HardwareRegister, Immediate, Load,
    LoadIndirect, MemoryLocation, MIRProgram, Move, OperandRole, Store,
    StoreIndirect, TypeConvert, UnaryOp, VirtualRegister, count_vreg_reads,
    index_clobbers, iter_memory_operands, iter_operands, map_memory_operands,
)

# Statics addressed absolute (or direct page): the only ones Y can index.
_Y_STORAGE = frozenset({'zeropage', 'lowram'})
# BinaryOp operators whose memory operand form has an abs,Y opcode.
_Y_ALU_OPS = frozenset({'+', '-', '&', '|', '^'})


def _size(type_info) -> Optional[int]:
    size = getattr(type_info, 'size_bytes', None)
    return size if isinstance(size, int) else None


def _is_reg(operand, name: str) -> bool:
    return type(operand) is HardwareRegister and operand.name == name


def _x_indexed(instr) -> bool:
    return any(loc.index_register == 'X' for loc in iter_memory_operands(instr))


@dataclass
class _Segment:
    """X set (by `move`, or by something opaque when None) and the
    instructions that index by it, up to the next X write."""
    start: int
    key: object
    move: Optional[int] = None
    accesses: List[int] = field(default_factory=list)
    reads_x: bool = False   # something reads X's value, not just indexes by it
    last: int = 0


class _ValueKeys:
    """Block-local value numbers: equal keys are equal values."""

    def __init__(self):
        self.version: Dict[object, int] = {}
        self.pure: Dict[int, object] = {}   # vreg id -> key of its in-block def

    def of(self, operand):
        if type(operand) is Immediate:
            return ('imm', operand.value)
        if type(operand) is VirtualRegister:
            known = self.pure.get(operand.id)
            if known is not None:
                return known
            return ('v', operand.id, self.version.get(operand.id, 0))
        if type(operand) is HardwareRegister:
            return ('hw', operand.name, self.version.get(operand.name, 0))
        return ('opaque', id(operand))

    def after(self, instr):
        """Account for *instr*'s definitions."""
        new_key = None
        dest = getattr(instr, 'dest', None)
        if type(dest) is VirtualRegister and not any(True for _ in iter_memory_operands(instr)):
            if type(instr) is BinaryOp:
                new_key = ('bin', instr.op, self.of(instr.left), self.of(instr.right),
                           _size(instr.type_info))
            elif type(instr) is TypeConvert:
                new_key = ('conv', self.of(instr.source), _size(instr.source_type),
                           _size(instr.target_type))
            elif type(instr) is Move:
                new_key = self.of(instr.source)
        for d in liveness_defs(instr):
            name = d.id if type(d) is VirtualRegister else getattr(d, 'name', d)
            self.version[name] = self.version.get(name, 0) + 1
            if type(d) is VirtualRegister:
                self.pure.pop(d.id, None)
        if new_key is not None:
            self.pure[dest.id] = new_key


def writes_x(instr) -> bool:
    """Whether *instr* changes X."""
    if type(instr) in CLOBBERS_ALL or 'X' in index_clobbers(instr):
        return True
    for d in liveness_defs(instr):
        if _is_reg(d, 'X'):
            return True
        if type(d) is VirtualRegister and d.register_hint == 'X':
            return True
    return False


def x_segments(instrs: List) -> List[_Segment]:
    """Split *instrs* at every X write into segments: what X was set to,
    and the instructions that index by it."""
    keys = _ValueKeys()
    segments: List[_Segment] = []
    cur: Optional[_Segment] = None
    for i, instr in enumerate(instrs):
        if type(instr) is Move and _is_reg(instr.dest, 'X'):
            cur = _Segment(start=i, move=i, last=i,
                           key=(keys.of(instr.source), _size(instr.type_info)))
            segments.append(cur)
        elif writes_x(instr):
            cur = None
            segments.append(_Segment(start=i, key=('opaque', i), last=i))
        else:
            if _x_indexed(instr):
                if cur is None:
                    cur = _Segment(start=i, key=('opaque', i), last=i)
                    segments.append(cur)
                cur.accesses.append(i)
                cur.last = i
            if cur is not None and any(
                    _is_reg(v, 'X') for _, v in iter_operands(instr, role=OperandRole.READ)):
                cur.reads_x = True
                cur.last = i
        keys.after(instr)
    return segments


def redundant_x_moves(instrs: List, segments: List[_Segment], skip=frozenset()):
    """The `X = Move` instructions (indices, with their sources) that find X
    already holding their value; *skip* are moves no longer writing X."""
    found, sources = set(), []
    x_key = None
    for seg in segments:
        if seg.move in skip:
            continue
        if seg.move is None:
            x_key = None
        elif seg.key == x_key and seg.key[0][0] != 'opaque':
            found.add(seg.move)
            sources.append(instrs[seg.move].source)
        else:
            x_key = seg.key
    return found, sources


def drop_dead(func, block, sources):
    """Delete the in-block pure computations the dropped loads read."""
    work = [s.id for s in sources if type(s) is VirtualRegister]
    while work:
        reads = count_vreg_reads(func)
        vid = work.pop()
        if reads.get(vid, 0):
            continue
        keep = []
        for ins in block.instructions:
            dest = getattr(ins, 'dest', None)
            if (type(dest) is VirtualRegister and dest.id == vid
                    and type(ins) in (BinaryOp, UnaryOp, TypeConvert, Move)
                    and not any(True for _ in iter_memory_operands(ins))):
                work.extend(v.id for _, v in iter_operands(ins, role=OperandRole.READ)
                            if type(v) is VirtualRegister)
                continue
            keep.append(ins)
        block.instructions = keep


class IndexStreamSplitter:
    """Give one of a block's alternating X index streams Y instead."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.split = 0

    def optimize(self, mir_program: MIRProgram) -> int:
        """Split every stream it can. Returns how many moved to Y."""
        self.split = 0
        for func in mir_program.functions:
            if self._excluded(func):
                continue
            self._induction = self._induction_vregs(func)
            for block in func.blocks.values():
                while self._split_block(func, block):
                    self.split += 1
        return self.split

    @staticmethod
    def _induction_vregs(func) -> set:
        """Vregs a loop header's compare reads: the loops' induction variables."""
        ids = set()
        for header, _ in find_loops(func):
            for ins in func.blocks[header].instructions:
                if type(ins) is Compare:
                    ids |= {v.id for _, v in iter_operands(ins, role=OperandRole.READ)
                            if type(v) is VirtualRegister}
        return ids

    @staticmethod
    def _evict(func, vreg_id: int):
        """Take a promoted value out of Y: it lives in a stack slot instead."""
        for reg, vreg in list(func.loop_promoted_hw_vregs.items()):
            if vreg.id == vreg_id:
                del func.loop_promoted_hw_vregs[reg]
        for block in func.blocks.values():
            for ins in block.instructions:
                for _, v in iter_operands(ins):
                    if type(v) is VirtualRegister and v.id == vreg_id:
                        v.register_hint = None

    @staticmethod
    def _excluded(func) -> bool:
        """A trait method's `self`, or a parameter, lives in Y."""
        if getattr(func, 'is_trait_method', False):
            return True
        for param in func.parameters or ():
            binding = getattr(param, 'binding', None)
            if getattr(binding, 'register_name', None) == 'Y':
                return True
        return False

    # -- analysis ------------------------------------------------------------

    @staticmethod
    def _y_capable(instr) -> bool:
        """An access abs,Y can do, on a static abs,Y reaches."""
        locs = [loc for loc in iter_memory_operands(instr) if loc.index_register == 'X']
        if len(locs) != 1 or locs[0].storage_type not in _Y_STORAGE:
            return False
        loc = locs[0]
        kind = type(instr)
        if kind is Load:
            return instr.source is loc
        if kind is Store:
            return instr.dest is loc
        if kind is BinaryOp:
            return (instr.op in _Y_ALU_OPS and instr.right is loc
                    and type(instr.left) is not MemoryLocation)
        if kind is Compare:
            return instr.right is loc and type(instr.left) is not MemoryLocation
        return False

    @staticmethod
    def _touches_y(instr) -> bool:
        if type(instr) in CLOBBERS_ALL or type(instr) in (LoadIndirect, StoreIndirect):
            return True
        if 'Y' in index_clobbers(instr):
            return True
        return any(_is_reg(op, 'Y')
                   for op in list(liveness_uses(instr)) + list(liveness_defs(instr)))

    def _y_occupants(self, func, block, liveness, start: int, end: int):
        """The Y-promoted vregs the window [start, end] would displace, or
        None when one of them is a loop's induction variable."""
        hinted = {}
        for b in func.blocks.values():
            for ins in b.instructions:
                for _, v in iter_operands(ins):
                    if type(v) is VirtualRegister and v.register_hint == 'Y':
                        hinted[v.id] = v
        occupants = set()
        for vid, v in hinted.items():
            used = any(any(type(o) is VirtualRegister and o.id == vid
                           for _, o in iter_operands(block.instructions[i]))
                       for i in range(start, end + 1))
            if used or liveness.is_live_after(v, block.block_id, start - 1):
                if vid in self._induction:
                    return None
                occupants.add(vid)
        return occupants

    # -- the rewrite -----------------------------------------------------------

    def _split_block(self, func, block) -> bool:
        instrs = block.instructions
        segments = x_segments(instrs)
        groups: Dict[object, List[_Segment]] = {}
        for seg in segments:
            if seg.move is not None:
                groups.setdefault(seg.key, []).append(seg)

        liveness = None
        for key, segs in sorted(groups.items(), key=lambda kv: -len(kv[1])):
            if len(segs) < 2:
                continue
            first, last_seg = segs[0], segs[-1]
            # Interleaved with another stream, or Y saves nothing.
            if not any(s.key != key and first.start < s.start < last_seg.start
                       for s in segments):
                continue
            if any(s.reads_x for s in segs):
                continue
            if not all(self._y_capable(instrs[a]) for s in segs for a in s.accesses):
                continue
            end = max(s.last for s in segs)
            if any(self._touches_y(instrs[i]) for i in range(first.start, end + 1)):
                continue
            if liveness is None:
                liveness = InstructionLivenessAnalyzer(func)
            if liveness.is_hw_reg_live_after('Y', block.block_id, first.start - 1):
                continue
            occupants = self._y_occupants(func, block, liveness, first.start, end)
            if occupants is None:
                continue
            # X at the block's end must not change if a successor reads it.
            last_x_write = max(s.start for s in segments if s.move is not None or not s.accesses)
            if (any(s.start == last_x_write for s in segs)
                    and liveness.is_hw_reg_live_after('X', block.block_id, len(instrs) - 1)):
                continue
            for vid in occupants:
                self._evict(func, vid)
            self._rewrite(func, block, segments, segs)
            if self.verbose:
                print(f"  index-streams: {func.name} block {block.block_id}: "
                      f"{len(segs)} X loads -> 1 Y load")
            return True
        return False

    def _rewrite(self, func, block, segments: List[_Segment], segs: List[_Segment]):
        instrs = list(block.instructions)
        drop = set()
        moved_sources = []

        def to_y(loc):
            return replace(loc, index_register='Y') if loc.index_register == 'X' else loc

        for n, seg in enumerate(segs):
            move = instrs[seg.move]
            if n == 0:
                y_move = replace(move, dest=HardwareRegister('Y'))
                instrs[seg.move] = y_move
            else:
                drop.add(seg.move)
                moved_sources.append(move.source)
            for a in seg.accesses:
                map_memory_operands(instrs[a], to_y)

        # An X load that now finds X still holding its value goes too.
        found, sources = redundant_x_moves(instrs, segments, {seg.move for seg in segs})
        drop |= found
        moved_sources += sources

        block.instructions = [ins for i, ins in enumerate(instrs) if i not in drop]
        drop_dead(func, block, moved_sources)
        self._hoist(func, block, y_move)

    def _hoist(self, func, block, y_move):
        """Move the Y load, with the pure computation feeding only it, up
        the block as far as its inputs and Y allow. Set after the first
        access's value was loaded, computing the index would need A while
        that value waits in it: hoisted, the value goes straight across."""
        instrs = block.instructions
        reads = count_vreg_reads(func)
        pos = next(i for i, ins in enumerate(instrs) if ins is y_move)
        chain = [pos]
        wanted = {v.id for _, v in iter_operands(y_move, role=OperandRole.READ)
                  if type(v) is VirtualRegister}
        for k in range(pos - 1, -1, -1):
            ins = instrs[k]
            dest = getattr(ins, 'dest', None)
            if not (type(dest) is VirtualRegister and dest.id in wanted):
                continue
            if (type(ins) not in (BinaryOp, UnaryOp, TypeConvert, Move)
                    or any(True for _ in iter_memory_operands(ins))
                    or reads.get(dest.id, 0) != 1):
                break
            chain.append(k)
            wanted.discard(dest.id)
            wanted |= {v.id for _, v in iter_operands(ins, role=OperandRole.READ)
                       if type(v) is VirtualRegister}
        chain.sort()
        members = [instrs[k] for k in chain]
        inputs = {v.id for ins in members for _, v in iter_operands(ins, role=OperandRole.READ)
                  if type(v) is VirtualRegister}
        outputs = {ins.dest.id for ins in members if type(ins.dest) is VirtualRegister}
        target = chain[0]
        while target > 0:
            prev = instrs[target - 1]
            if self._touches_y(prev) or any(
                    type(v) is VirtualRegister and v.register_hint == 'Y'
                    for _, v in iter_operands(prev)):
                break
            if any(type(d) is VirtualRegister and (d.id in inputs or d.id in outputs)
                   for d in liveness_defs(prev)):
                break
            target -= 1
        if target == chain[0]:
            return
        liveness = InstructionLivenessAnalyzer(func)
        y_hinted = {v.id: v for b in func.blocks.values() for ins in b.instructions
                    for _, v in iter_operands(ins)
                    if type(v) is VirtualRegister and v.register_hint == 'Y'}

        def y_busy(at):
            return (liveness.is_hw_reg_live_after('Y', block.block_id, at)
                    or any(liveness.is_live_after(v, block.block_id, at)
                           for v in y_hinted.values()))

        while target < chain[0] and y_busy(target - 1):
            target += 1
        if target == chain[0]:
            return
        rest = [ins for i, ins in enumerate(instrs) if i not in set(chain)]
        block.instructions = rest[:target] + members + rest[target:]
