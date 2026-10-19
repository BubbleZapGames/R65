# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Web-Splitting Pass.

MIR is not SSA: a local assigned again keeps its vreg, so one vreg can carry
values that never meet. `sieve` clears its flags with `i` and then reuses
it for the prime loop::

    %1 = Move #0          loop 1: X = Move %1 ... %1 = %1 + 1
    %1 = Move #2          loop 2: %5 = %1 + %1 ...

Every decision made per vreg - loop promotion, A residency, a stack slot -
then sees both lives at once, and loop 2's `j + i` keeps loop 1's counter
out of X. A web is a set of defs joined by the uses they reach: a use
reached by several defs (a join, a loop back edge) ties them together.
Each web after the first gets its own vreg, so the values are allocated
apart.

Parameters and vregs with a register hint keep their one vreg: their
register is chosen for them before this pass. A web with a use no def
reaches (a read before the first write) keeps the original vreg.
"""

from dataclasses import replace
from typing import Dict, List, Tuple

from r65.compiler.mir.nodes import (
    MIRFunction, MIRProgram, OperandRole, VirtualRegister, iter_operands,
    map_operands,
)

# Def site of a value no def reaches: (vid, block, index) with no block.
_ENTRY = (-1, -1)


class WebSplitter:
    """Give each def-use web of a reassigned local its own vreg."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def optimize(self, mir_program: MIRProgram) -> int:
        return sum(self._split(func) for func in mir_program.functions)

    def _split(self, func: MIRFunction) -> int:
        if func.vreg_allocator is None:
            return 0
        fixed = {v.id for v in func.param_to_vreg.values() if v is not None}
        if func.self_y_vreg is not None:
            fixed.add(func.self_y_vreg.id)
        def_count: Dict[int, int] = {}
        for block in func.blocks.values():
            for instr in block.instructions:
                for _, v in iter_operands(instr):
                    if type(v) is VirtualRegister and v.register_hint is not None:
                        fixed.add(v.id)
                for _, v in iter_operands(instr, role=OperandRole.WRITE):
                    if type(v) is VirtualRegister:
                        def_count[v.id] = def_count.get(v.id, 0) + 1
        candidates = {vid for vid, n in def_count.items() if n > 1 and vid not in fixed}
        if not candidates:
            return 0

        reach_in = self._reaching_defs(func, candidates)

        # Union-find over def sites; each use joins the defs reaching it.
        parent: Dict[tuple, tuple] = {}

        def find(s):
            parent.setdefault(s, s)
            while parent[s] != s:
                parent[s] = parent[parent[s]]
                s = parent[s]
            return s

        use_site: Dict[Tuple[int, int, int], tuple] = {}
        def_sites: Dict[int, List[tuple]] = {}
        entry_used = set()
        for bid, block in func.blocks.items():
            state = dict(reach_in.get(bid, {}))
            for idx, instr in enumerate(block.instructions):
                for _, v in iter_operands(instr, role=OperandRole.READ):
                    if type(v) is not VirtualRegister or v.id not in candidates:
                        continue
                    sites = state.get(v.id) or {(v.id,) + _ENTRY}
                    sites = sorted(sites)
                    first = sites[0]
                    for s in sites[1:]:
                        parent[find(s)] = find(first)
                    find(first)
                    use_site[(bid, idx, v.id)] = first
                    if (v.id,) + _ENTRY in sites:
                        entry_used.add(v.id)
                for _, v in iter_operands(instr, role=OperandRole.WRITE):
                    if type(v) is VirtualRegister and v.id in candidates:
                        site = (v.id, bid, idx)
                        find(site)
                        def_sites.setdefault(v.id, []).append(site)
                        state[v.id] = frozenset([site])

        # A web per root; the one holding the entry (or the first def) keeps
        # the vreg, the others get fresh ids. A def nothing reads stays on
        # the vreg: it ends whatever value was live before it, so it cannot
        # overwrite one still in use.
        read = {find(site) for site in use_site.values()}
        new_id: Dict[tuple, int] = {}
        split = 0
        for vid in sorted(candidates):
            roots = []
            if vid in entry_used:
                roots.append(find((vid,) + _ENTRY))
            for site in def_sites.get(vid, []):
                r = find(site)
                if r in read and r not in roots:
                    roots.append(r)
            if len(roots) < 2:
                continue
            split += 1
            for r in roots[1:]:
                new_id[r] = func.vreg_allocator.alloc(None).id
        if not split:
            return 0

        for bid, block in func.blocks.items():
            for idx, instr in enumerate(block.instructions):
                def rename_use(v, bid=bid, idx=idx):
                    if type(v) is VirtualRegister:
                        site = use_site.get((bid, idx, v.id))
                        if site is not None and find(site) in new_id:
                            return replace(v, id=new_id[find(site)])
                    return v

                def rename_def(v, bid=bid, idx=idx):
                    if type(v) is VirtualRegister and v.id in candidates:
                        root = find((v.id, bid, idx))
                        if root in new_id:
                            return replace(v, id=new_id[root])
                    return v

                map_operands(instr, rename_use, role=OperandRole.READ)
                map_operands(instr, rename_def, role=OperandRole.WRITE)
        return split

    @staticmethod
    def _reaching_defs(func: MIRFunction, candidates) -> Dict[int, Dict[int, frozenset]]:
        """Per block, the def sites of each candidate that reach its entry.
        The function entry is reached by the `_ENTRY` pseudo-def."""
        gen: Dict[int, Dict[int, frozenset]] = {}
        for bid, block in func.blocks.items():
            last: Dict[int, frozenset] = {}
            for idx, instr in enumerate(block.instructions):
                for _, v in iter_operands(instr, role=OperandRole.WRITE):
                    if type(v) is VirtualRegister and v.id in candidates:
                        last[v.id] = frozenset([(v.id, bid, idx)])
            gen[bid] = last

        preds: Dict[int, List[int]] = {bid: [] for bid in func.blocks}
        for bid, block in func.blocks.items():
            for succ in block.successors:
                if succ in preds:
                    preds[succ].append(bid)

        entry = func.entry_block_id
        reach_in: Dict[int, Dict[int, frozenset]] = {bid: {} for bid in func.blocks}
        reach_in[entry] = {vid: frozenset([(vid,) + _ENTRY]) for vid in candidates}
        reach_out: Dict[int, Dict[int, frozenset]] = {}
        work = list(func.blocks)
        while work:
            bid = work.pop()
            if bid == entry:
                merged = dict(reach_in[entry])
            else:
                merged = {}
            for p in preds[bid]:
                for vid, sites in reach_out.get(p, {}).items():
                    merged[vid] = merged.get(vid, frozenset()) | sites
            if bid in reach_out and merged == reach_in[bid]:
                continue
            reach_in[bid] = merged
            out = dict(merged)
            out.update(gen[bid])
            if reach_out.get(bid) != out:
                reach_out[bid] = out
                work.extend(s for s in func.blocks[bid].successors if s in func.blocks)
        return reach_in
