# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Jump threading of a branch on a value its predecessor set to a constant.

An inlined `fn is3() -> bool { return G == 3; }` tested by `if !is3()`
builds the 0/1 result in two blocks, joins them, copies the result into the
call's result vreg, and only then branches on it::

    B3: %2 = %1 ^ #3 ; CondBranch %2 != 0 ? B5 : B4
    B4: %3 = #1 ; Jump B6                 ->   B4: Jump B2
    B5: %3 = #0 ; Jump B6                      B5: Jump B1
    B6: %0 = %3 ; Jump B7
    B7: Compare %0 != #0 ; CondBranch ? B2 : B1

Each predecessor that ends `%v = #k; Jump B` knows which way the branch at
the end of the path from B goes, so it jumps there itself. The constant
Moves no longer read, and the blocks no longer reached, are deleted.

Soundness. The path from B to its branch may hold only vreg copies and
jumps, and the vregs it writes may be read only on the path: a threaded
predecessor skips those writes, so nothing else may see them. The branch
reads a copy of the constant through a `CondBranch %v` (non-zero) or an
equality `Compare %v, #c`. A path block reachable from the new target is a
loop header (or in a loop), and is left alone: jumping past it would enter
the loop a second way.

It runs right after inlining, where the shape appears. No dead-code pass runs
after inlining, so this one prunes what it orphans itself, and only that: a
result never read by name is read through the register it lands in (see
const_prop), so a general dead-store sweep here would not be safe.
"""

from typing import Dict, List, Optional, Set, Tuple

from r65.compiler.mir.nodes import (
    Compare, CondBranch, Jump, MIRFunction, MIRProgram, Move,
    OperandRole, VirtualRegister, iter_operands,
)
from r65.compiler.optimize.const_prop import _plain
from r65.compiler.optimize.dead_code_elim import DeadCodeEliminator
from r65.compiler.optimize.loop_rotate import LoopRotator

# How many blocks a path may run through before its branch.
_MAX_PATH = 4


def _is_copy(instr) -> bool:
    return (type(instr) is Move and isinstance(instr.dest, VirtualRegister)
            and (isinstance(instr.source, VirtualRegister) or _plain(instr.source)))


def _vreg_reads(func: MIRFunction) -> Dict[int, Set[int]]:
    """vreg id -> the blocks that read it."""
    reads: Dict[int, Set[int]] = {}
    for bid, block in func.blocks.items():
        for instr in block.instructions:
            for _, v in iter_operands(instr, role=OperandRole.READ):
                if isinstance(v, VirtualRegister):
                    reads.setdefault(v.id, set()).add(bid)
    return reads


class JumpThreader:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.threaded = 0
        self.changed_funcs: List[MIRFunction] = []

    def optimize(self, mir_program: MIRProgram) -> int:
        self.threaded = 0
        self.changed_funcs = []
        for func in mir_program.functions:
            if self._optimize_function(func):
                self.changed_funcs.append(func)
        return self.threaded

    def _optimize_function(self, func: MIRFunction) -> bool:
        reads_before = _vreg_reads(func)
        changed = False
        progress = True
        while progress:
            progress = False
            reads = _vreg_reads(func)
            for pid in list(func.blocks):
                pred = func.blocks[pid]
                if not pred.instructions or type(pred.instructions[-1]) is not Jump:
                    continue
                start = pred.instructions[-1].target
                env = self._constants_at_end(pred)
                if not env:
                    continue
                found = self._follow(func, start, env, reads)
                if found is None:
                    continue
                path, target = found
                if pid in path or self._reaches_any(func, target, set(path)):
                    continue
                LoopRotator._retarget(func, pid, start, target)
                self.threaded += 1
                changed = progress = True
                if self.verbose:
                    print(f"  jump-thread: {func.name} B{pid} -> B{target} past {path}")
        if changed:
            DeadCodeEliminator()._eliminate_unreachable_blocks(func)
            self._drop_orphaned_constants(func, reads_before)
        return changed

    @staticmethod
    def _constants_at_end(block) -> Dict[int, int]:
        """vreg id -> the constant it holds at the end of *block*."""
        env: Dict[int, int] = {}
        for instr in block.instructions:
            writes = [v.id for _, v in iter_operands(instr, role=OperandRole.WRITE)
                      if isinstance(v, VirtualRegister)]
            for vid in writes:
                env.pop(vid, None)
            if type(instr) is Move and isinstance(instr.dest, VirtualRegister) \
                    and _plain(instr.source):
                env[instr.dest.id] = instr.source.value
        return env

    def _follow(self, func: MIRFunction, start: int, env: Dict[int, int],
                reads: Dict[int, Set[int]]) -> Optional[Tuple[List[int], int]]:
        """`(path, target)`: the blocks from *start* to a branch whose
        direction *env* decides, and where it goes; None if there is none."""
        env = dict(env)
        path: List[int] = []
        written: Set[int] = set()
        bid = start
        while len(path) < _MAX_PATH and bid in func.blocks and bid not in path:
            path.append(bid)
            instrs = func.blocks[bid].instructions
            if not instrs:
                return None
            tail = instrs[-1]
            body = instrs[:-1]
            compare = None
            if type(tail) is CondBranch and tail.condition is None:
                if not body or type(body[-1]) is not Compare:
                    return None
                compare, body = body[-1], body[:-1]
            if not all(_is_copy(i) for i in body):
                return None
            for move in body:
                written.add(move.dest.id)
                src = move.source
                if _plain(src):
                    env[move.dest.id] = src.value
                elif src.id in env:
                    env[move.dest.id] = env[src.id]
                else:
                    env.pop(move.dest.id, None)
            if type(tail) is Jump:
                bid = tail.target
                continue
            if type(tail) is not CondBranch:
                return None
            taken = self._decide(tail, compare, env)
            if taken is None:
                return None
            # Skipped writes must not be seen past the path.
            on_path = set(path)
            if any(not reads.get(vid, set()) <= on_path for vid in written):
                return None
            return path, tail.true_target if taken else tail.false_target
        return None

    @staticmethod
    def _decide(branch: CondBranch, compare: Optional[Compare],
                env: Dict[int, int]) -> Optional[bool]:
        if compare is None:
            cond = branch.condition
            if not isinstance(cond, VirtualRegister) or cond.id not in env:
                return None
            if branch.comparison == '!=':
                return env[cond.id] != 0
            if branch.comparison == '==':
                return env[cond.id] == 0
            return None
        if branch.comparison != compare.comparison or compare.comparison not in ('==', '!='):
            return None
        left, right = compare.left, compare.right
        if not (isinstance(left, VirtualRegister) and left.id in env and _plain(right)):
            return None
        equal = env[left.id] == right.value
        return equal if compare.comparison == '==' else not equal

    @staticmethod
    def _reaches_any(func: MIRFunction, start: int, blocks: Set[int]) -> bool:
        seen: Set[int] = set()
        work = [start]
        while work:
            bid = work.pop()
            if bid in blocks:
                return True
            if bid in seen or bid not in func.blocks:
                continue
            seen.add(bid)
            work.extend(func.blocks[bid].successors)
        return False

    @staticmethod
    def _drop_orphaned_constants(func: MIRFunction, reads_before: Dict[int, Set[int]]) -> None:
        """Delete `Move %v = #k` whose vreg was read before threading and is
        not now: the reads were on paths that are gone."""
        reads = _vreg_reads(func)
        dead = {vid for vid in reads_before if vid not in reads}
        for block in func.blocks.values():
            block.instructions = [
                i for i in block.instructions
                if not (type(i) is Move and isinstance(i.dest, VirtualRegister)
                        and i.dest.id in dead and _plain(i.source))]
