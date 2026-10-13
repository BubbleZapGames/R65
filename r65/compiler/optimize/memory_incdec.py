# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Memory INC/DEC Folding Pass.

Rewrites a static or plane element stepped by one in place::

    %a = Load [m]
    %b = %a - #1                     ->     IncDecMemory [m] -= 1
    Store %b -> [m]

which instruction selection emits as a single read-modify-write::

    DEC heat,X                      ; was: LDA heat,X / SEC / SBC #1 / STA heat,X

The 65816 has INC/DEC on memory for direct page and absolute, each also
`,X`, at either accumulator width. They leave A alone, so the value never
occupies it, and they set N/Z from the result — so when the next thing the
block does is test the new value against zero::

    %c = Load [m]
    Compare %c == #0                ->     (folded into the IncDecMemory)
    CondBranch …                           CondBranch …

the reload and compare go too, and the branch reads the Z flag DEC left:
`DEC heat,X / BNE`. Only `==` and `!=` qualify; N alone is not a correct
`<`/`>` against zero for every type.

The three instructions must be adjacent, which is how the MIR builder emits
`m = m ± 1`, `m += 1` and `m++`; nothing between them can then change the
index register or the memory. Locations INC/DEC cannot address are left
alone: `#[ram]` statics live in bank $7E behind long addressing (there is no
INC long), ROM is read-only, a `#[hw]` register must see exactly the accesses
written, and `abs,Y` has no INC form.

Done in MIR so indexed and 16-bit operands are seen.
"""

from typing import Dict, List

from r65.compiler.mir.nodes import (
    BinaryOp, Compare, CondBranch, Immediate, IncDecMemory, Load,
    MIRProgram, Store, VirtualRegister, count_vreg_reads,
)

# Storage INC/DEC can address with a 16-bit operand: direct page / absolute.
_INC_DEC_STORAGE = frozenset({'zeropage', 'lowram'})


def _size(type_info):
    size = getattr(type_info, 'size_bytes', None)
    return size if isinstance(size, int) else None


class MemoryIncDecFolder:
    """Fold `Load m; m ± 1; Store m` into one INC/DEC on memory."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.folded = 0
        self.branch_flags_reused = 0

    def optimize(self, mir_program: MIRProgram) -> int:
        """Rewrite every foldable step in the program. Returns the count."""
        self.folded = 0
        self.branch_flags_reused = 0
        for func in mir_program.functions:
            reads = count_vreg_reads(func)
            for block in func.blocks.values():
                block.instructions = self._rewrite_block(block.instructions, reads)
        return self.folded

    def _rewrite_block(self, instrs: List, reads: Dict[int, int]) -> List:
        out: List = []
        i = 0
        while i < len(instrs):
            step = self._match_step(instrs, i, reads)
            if step is None:
                out.append(instrs[i])
                i += 1
                continue
            out.append(step)
            self.folded += 1
            i += 3
            if self._match_zero_test(instrs, i, step, reads):
                step.sets_branch_flags = True
                self.branch_flags_reused += 1
                i += 2  # the reload and the Compare; the CondBranch stays
        return out

    @staticmethod
    def _foldable_location(loc) -> bool:
        return (loc is not None
                and not loc.is_volatile
                and loc.storage_type in _INC_DEC_STORAGE
                and loc.index_register in (None, 'X'))

    def _match_step(self, instrs: List, i: int, reads: Dict[int, int]):
        """An IncDecMemory for `instrs[i:i+3]` if it is `Load; ±1; Store`."""
        if i + 2 >= len(instrs):
            return None
        load, op, store = instrs[i], instrs[i + 1], instrs[i + 2]
        if not (isinstance(load, Load) and isinstance(op, BinaryOp)
                and isinstance(store, Store)):
            return None
        a, b = load.dest, op.dest
        if not (isinstance(a, VirtualRegister) and isinstance(b, VirtualRegister)):
            return None
        if a.register_hint or b.register_hint:
            return None
        if reads.get(a.id, 0) != 1 or reads.get(b.id, 0) != 1:
            return None
        if not self._foldable_location(load.source) or store.dest != load.source:
            return None
        if not (isinstance(op.left, VirtualRegister) and op.left.id == a.id
                and isinstance(store.source, VirtualRegister)
                and store.source.id == b.id):
            return None
        if op.op not in ('+', '-') or not isinstance(op.right, Immediate) \
                or op.right.value != 1:
            return None
        size = _size(load.type_info)
        if size not in (1, 2) or _size(op.type_info) != size \
                or _size(store.type_info) != size:
            return None
        return IncDecMemory(location=load.source,
                            delta=1 if op.op == '+' else -1,
                            type_info=load.type_info,
                            source_loc=getattr(load, 'source_loc', None))

    def _match_zero_test(self, instrs: List, i: int, step: IncDecMemory,
                         reads: Dict[int, int]) -> bool:
        """`Load m; Compare == / != #0; CondBranch` ending the block, right
        after *step* — the test INC/DEC's Z flag already answers."""
        if i + 2 != len(instrs) - 1:
            return False
        reload, cmp, branch = instrs[i], instrs[i + 1], instrs[i + 2]
        if not (isinstance(reload, Load) and isinstance(cmp, Compare)
                and isinstance(branch, CondBranch)):
            return False
        c = reload.dest
        if not isinstance(c, VirtualRegister) or c.register_hint \
                or reads.get(c.id, 0) != 1:
            return False
        if reload.source != step.location or _size(reload.type_info) != _size(step.type_info):
            return False
        if not (isinstance(cmp.left, VirtualRegister) and cmp.left.id == c.id
                and isinstance(cmp.right, Immediate) and cmp.right.value == 0
                and cmp.comparison in ('==', '!=')):
            return False
        return branch.condition is None and branch.comparison == cmp.comparison
