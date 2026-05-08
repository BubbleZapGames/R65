# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Forward dataflow over the asm node list that computes the accumulator
m-flag mode (m8/m16/⊥) entering every node.

Why this exists:
    The peephole optimizer used to track mode by sniffing `.ACCU`
    directives interleaved in the asm stream. Those directives are a
    *hint* the codegen emits for WLA-DX (which tracks mode linearly,
    not by control flow); they are not a primitive of CPU semantics.
    Reading them as truth led to the optimizer believing one thing
    while the runtime CPU was in a different mode — most painfully at
    loop back-edges, where a `SEP #$20` at the tail leaves the CPU in
    m8 even though the body's `.ACCU 16` directive made the optimizer
    (and WLA-DX) assemble m16 immediates that then mis-decoded at
    runtime.

    The single source of truth for runtime mode is:
        SEP #$20  ⟶ m8
        REP #$20  ⟶ m16
        PLP, RTI  ⟶ unknown (depends on stacked P)
    plus the CFG induced by labels, branches, and jumps. This module
    computes the exact set of modes that may hold at each node by
    running a standard forward dataflow on that CFG.

Public API:
    compute_modes(nodes, entry_modes=None) -> ModeInfo

ModeInfo answers:
    - incoming_at(idx)     → set of modes (8 / 16 / None) that may
                             hold when nodes[idx] is about to execute
    - unique_mode_at(idx)  → the single mode if all paths agree, else
                             None
    - has_mixed_known(idx) → True iff at least one path arrives in m8
                             AND at least one in m16
    - required_at(idx)     → the mode the instruction REQUIRES given
                             its operand width, or None for
                             mode-independent ops

Notes on `.ACCU`:
    This pass deliberately ignores `.ACCU` directives. Their content
    is a function of the dataflow result (we re-emit them from this
    analysis at the end of optimization), not an input to it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from r65.compiler.codegen.opcodes import (
    Opcode,
    BRANCH_OPCODES,
    JUMP_OPCODES,
    RETURN_OPCODES,
)


# m-flag bit in the SEP/REP immediate operand.
M_FLAG = 0x20

# Accumulator-immediate opcodes whose encoded byte width depends on the
# current m flag. If one of these has an immediate value > $FF, the
# instruction can ONLY have been intended for m16 — that's evidence we
# can use to fix mode-fix decisions at mixed-mode merge points.
A_IMM_OPCODES = frozenset({
    Opcode.LDA_IMMEDIATE,
    Opcode.AND_IMMEDIATE,
    Opcode.ORA_IMMEDIATE,
    Opcode.EOR_IMMEDIATE,
    Opcode.ADC_IMMEDIATE,
    Opcode.SBC_IMMEDIATE,
    Opcode.CMP_IMMEDIATE,
    Opcode.BIT_IMMEDIATE,
})


# A mode is one of {8, 16, None}. None means "unknown" (entry point
# without a declared mode, or after PLP/RTI which restore P from the
# stack).
Mode = Optional[int]


@dataclass
class ModeInfo:
    """Per-node mode information from the dataflow analysis.

    Internal representation: `incoming[i]` is the set of modes that may
    hold immediately *before* nodes[i] executes. The set may contain
    integers 8 or 16, or None for "unknown". `pred_count[i]` is the
    number of distinct physical predecessor edges into nodes[i]
    (fall-through plus branch/jump sources).
    """

    incoming: Dict[int, Set[Mode]] = field(default_factory=dict)
    required: Dict[int, Mode] = field(default_factory=dict)
    pred_count: Dict[int, int] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Query helpers — preferred read API for consumers.
    # ------------------------------------------------------------------

    def incoming_at(self, idx: int) -> Set[Mode]:
        return self.incoming.get(idx, set())

    def required_at(self, idx: int) -> Mode:
        return self.required.get(idx)

    def unique_mode_at(self, idx: int) -> Mode:
        """Return the mode if all paths agree on a definite value, else None."""
        modes = self.incoming.get(idx, set())
        defined = {m for m in modes if m is not None}
        if len(defined) == 1 and None not in modes:
            return next(iter(defined))
        return None

    def has_mixed_known(self, idx: int) -> bool:
        """True iff some path arrives in m8 and some in m16."""
        modes = self.incoming.get(idx, set())
        return 8 in modes and 16 in modes

    def is_reachable(self, idx: int) -> bool:
        return bool(self.incoming.get(idx))

    def is_join(self, idx: int) -> bool:
        """True iff this node has more than one physical predecessor edge.

        A label that is only reachable by fall-through from a single
        predecessor is NOT a join; mode-fix coercion at such a label is
        always redundant with whatever coercion was applied upstream.
        """
        return self.pred_count.get(idx, 0) > 1


# ----------------------------------------------------------------------
# Internals
# ----------------------------------------------------------------------

# Unconditional control transfers — successor is the branch target only,
# not the next-in-line node. JUMP_OPCODES already excludes conditional
# branches (BCC/BEQ/etc.), so JMP/JML/etc. all qualify. BRA is a branch,
# but it's also unconditional, so we treat it specially.
_UNCONDITIONAL_TRANSFER = JUMP_OPCODES | RETURN_OPCODES | frozenset({Opcode.BRA})


def _required_mode_of(node) -> Mode:
    """Return the mode that node REQUIRES given its operand, or None."""
    # Local import to avoid module-load-time cycles.
    from r65.compiler.codegen.asm_nodes import Instruction, Immediate

    if not isinstance(node, Instruction):
        return None
    if node.opcode not in A_IMM_OPCODES:
        return None
    if not isinstance(node.operand, Immediate):
        return None
    val = node.operand.value
    if isinstance(val, int) and val > 0xFF:
        return 16
    # Sub-$100 immediates fit either mode; they don't constrain.
    return None


def _transfer(in_modes: Set[Mode], node) -> Set[Mode]:
    """Compute the OUT mode set for a node given its IN mode set."""
    from r65.compiler.codegen.asm_nodes import Instruction, Immediate

    if not isinstance(node, Instruction):
        # Labels, directives, comments — pass mode through unchanged.
        return set(in_modes)

    op = node.opcode
    if op == Opcode.SEP_IMMEDIATE:
        if isinstance(node.operand, Immediate) and isinstance(node.operand.value, int):
            if node.operand.value & M_FLAG:
                return {8}
    elif op == Opcode.REP_IMMEDIATE:
        if isinstance(node.operand, Immediate) and isinstance(node.operand.value, int):
            if node.operand.value & M_FLAG:
                return {16}
    elif op in (Opcode.PLP, Opcode.RTI):
        return {None}

    return set(in_modes)


def _branch_target(node) -> Optional[str]:
    """If node is a branch/jump with a label target, return the label name."""
    from r65.compiler.codegen.asm_nodes import Instruction, Address

    if not isinstance(node, Instruction):
        return None
    if node.opcode not in BRANCH_OPCODES and node.opcode not in JUMP_OPCODES:
        return None
    operand = node.operand
    if operand is None:
        return None
    val = getattr(operand, 'value', None)
    if isinstance(val, str):
        return val
    return None


def _is_unconditional_terminator(node) -> bool:
    """True if control cannot fall through past this node to nodes[i+1]."""
    from r65.compiler.codegen.asm_nodes import Instruction

    if not isinstance(node, Instruction):
        return False
    return node.opcode in _UNCONDITIONAL_TRANSFER


def compute_modes(
    nodes: List,
    entry_modes: Optional[Dict[str, Mode]] = None,
) -> ModeInfo:
    """Run forward dataflow over `nodes`; return per-node mode info.

    Args:
        nodes: linearized list of asm nodes (Instruction / Label /
               Directive / Comment / BlankLine, in emission order).
        entry_modes: optional `{label_name: initial_mode}`. Function
            entry labels may be seeded here so the analysis starts with
            a known mode rather than ⊥. Common values:
                {entry_name: 8}   — m8 entry (the default)
                {entry_name: 16}  — m16 entry (e.g. `@ A: u16`)
                {entry_name: None}— "unknown" (legal but conservative)

    Returns:
        ModeInfo. Every reachable node has a non-empty `incoming` set;
        unreachable nodes have an empty set (use `is_reachable`).
    """
    from r65.compiler.codegen.asm_nodes import Label, Instruction

    entry_modes = entry_modes or {}
    n = len(nodes)
    info = ModeInfo()

    # Pre-pass: index labels and stash required-mode per instruction.
    label_idx: Dict[str, int] = {}
    for i, node in enumerate(nodes):
        if isinstance(node, Label):
            # First occurrence wins if (somehow) duplicated.
            label_idx.setdefault(node.name, i)
        info.required[i] = _required_mode_of(node)

    for i in range(n):
        info.incoming[i] = set()
        info.pred_count[i] = 0

    # Count physical predecessors (fall-through + branch/jump sources).
    # The fall-through count is fixed by source order: every node has at
    # most one fall-through predecessor (the previous index), unless that
    # previous node unconditionally transfers control elsewhere.
    for i in range(1, n):
        prev = nodes[i - 1]
        if not _is_unconditional_terminator(prev):
            info.pred_count[i] += 1
    for i, node in enumerate(nodes):
        target = _branch_target(node)
        if target is not None and target in label_idx:
            info.pred_count[label_idx[target]] += 1

    # Seed: node 0 starts the program. Function entry labels (and their
    # explicit declared modes) seed their respective indices too.
    if n > 0:
        info.incoming[0].add(entry_modes.get('__entry__', None))
    for label, mode in entry_modes.items():
        if label in label_idx:
            info.incoming[label_idx[label]].add(mode)

    # Worklist iteration. Lattice height is small (≤3 distinct values
    # in any incoming set), so this converges in O(n) iterations of the
    # full pass for any realistic program.
    changed = True
    while changed:
        changed = False
        for i in range(n):
            in_set = info.incoming[i]
            if not in_set:
                continue  # unreachable so far — don't propagate ⊥-only sets
            out_set = _transfer(in_set, nodes[i])
            node = nodes[i]

            # Fall-through successor (next index), unless this node
            # unconditionally transfers control elsewhere.
            if not _is_unconditional_terminator(node) and i + 1 < n:
                if not out_set.issubset(info.incoming[i + 1]):
                    info.incoming[i + 1] |= out_set
                    changed = True

            # Branch / jump successor.
            target = _branch_target(node)
            if target is not None and target in label_idx:
                tgt_idx = label_idx[target]
                if not out_set.issubset(info.incoming[tgt_idx]):
                    info.incoming[tgt_idx] |= out_set
                    changed = True

    return info


def first_constraining_mode_after(
    nodes: List,
    info: ModeInfo,
    start_idx: int,
    *,
    max_lookahead: int = 30,
) -> Mode:
    """Look forward from start_idx for the first signal indicating what
    mode the body that follows is meant to execute in. Returns the
    declared mode (8 or 16) if found within max_lookahead nodes, else
    None.

    Signals checked, in scan order at each node:

      1. `.ACCU 8` / `.ACCU 16` directive — the codegen's *declared*
         mode for the next instructions. This is the most direct
         expression of compiler intent and is preferred over indirect
         evidence: it captures the m-mode the codegen was assuming when
         it picked operand widths (e.g. for stack-relative loads whose
         encoding doesn't itself disambiguate the mode).

      2. An accumulator-immediate operation whose value can only fit
         in m16 (`AND #$8000`, etc.) — a *hard* constraint: if the
         instruction is to behave correctly the mode must be m16.

      3. An explicit `SEP #$20` / `REP #$20` — itself declares the
         mode for what follows.

      4. Function exit (`RTS` / `RTL` / `RTI`) — give up; downstream
         code is in another function.

    Mode-independent instructions and unconditional intra-function
    transfers (BRA / JMP) do not stop the scan — codegen often emits
    BRA-to-next-label or JMP between adjacent blocks of one loop, and
    the mode signal we want lives just past that no-op transfer.
    """
    from r65.compiler.codegen.asm_nodes import Instruction, Directive

    end = min(start_idx + max_lookahead, len(nodes))
    for k in range(start_idx, end):
        node = nodes[k]
        if isinstance(node, Directive) and node.name == '.ACCU':
            if node.args:
                if node.args[0] == '8':
                    return 8
                if node.args[0] == '16':
                    return 16
            continue
        if not isinstance(node, Instruction):
            continue
        if node.opcode == Opcode.SEP_IMMEDIATE:
            return 8
        if node.opcode == Opcode.REP_IMMEDIATE:
            return 16
        req = info.required_at(k)
        if req is not None:
            return req
        if node.opcode in RETURN_OPCODES:
            return None
    return None
