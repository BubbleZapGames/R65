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


# Status-flag bits in the SEP/REP immediate operand.
M_FLAG = 0x20  # accumulator width (1 = m8)
X_FLAG = 0x10  # index-register width (1 = x8)

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

    Internal representation: `incoming[i]` is the set of m-flag modes
    (accumulator width) that may hold immediately *before* nodes[i]
    executes; `x_incoming[i]` is the parallel set for the x-flag
    (index-register width). Each set may contain integers 8 or 16, or
    None for "unknown". `pred_count[i]` is the number of distinct
    physical predecessor edges into nodes[i] (fall-through plus
    branch/jump sources).

    The two flag dimensions are tracked in parallel rather than as a
    Cartesian product because SEP/REP can update them independently and
    callers ask about them independently.
    """

    incoming: Dict[int, Set[Mode]] = field(default_factory=dict)
    x_incoming: Dict[int, Set[Mode]] = field(default_factory=dict)
    required: Dict[int, Mode] = field(default_factory=dict)
    pred_count: Dict[int, int] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Query helpers — preferred read API for consumers.
    # ------------------------------------------------------------------

    def incoming_at(self, idx: int) -> Set[Mode]:
        return self.incoming.get(idx, set())

    def x_incoming_at(self, idx: int) -> Set[Mode]:
        return self.x_incoming.get(idx, set())

    def required_at(self, idx: int) -> Mode:
        return self.required.get(idx)

    def unique_mode_at(self, idx: int) -> Mode:
        """Return the m-flag mode if all paths agree on a definite value, else None."""
        return self._unique(self.incoming.get(idx, set()))

    def unique_x_mode_at(self, idx: int) -> Mode:
        """Return the x-flag mode if all paths agree on a definite value, else None."""
        return self._unique(self.x_incoming.get(idx, set()))

    @staticmethod
    def _unique(modes: Set[Mode]) -> Mode:
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


def _transfer(in_modes: Set[Mode], node, *, flag: int = M_FLAG) -> Set[Mode]:
    """Compute the OUT mode set for a node given its IN mode set.

    `flag` selects which P-flag to track (M_FLAG for the accumulator,
    X_FLAG for index registers). The transfer function is identical
    for both — only the bit being inspected on SEP/REP differs.
    """
    from r65.compiler.codegen.asm_nodes import Instruction, Immediate

    if not isinstance(node, Instruction):
        # Labels, directives, comments — pass mode through unchanged.
        return set(in_modes)

    op = node.opcode
    if op == Opcode.SEP_IMMEDIATE:
        if isinstance(node.operand, Immediate) and isinstance(node.operand.value, int):
            if node.operand.value & flag:
                return {8}
    elif op == Opcode.REP_IMMEDIATE:
        if isinstance(node.operand, Immediate) and isinstance(node.operand.value, int):
            if node.operand.value & flag:
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
    entry_x_modes: Optional[Dict[str, Mode]] = None,
) -> ModeInfo:
    """Run forward dataflow over `nodes`; return per-node mode info.

    Tracks two independent dimensions in one pass:
      - m-flag (accumulator width) → ``incoming`` / ``unique_mode_at``
      - x-flag (index-register width) → ``x_incoming`` / ``unique_x_mode_at``

    Args:
        nodes: linearized list of asm nodes (Instruction / Label /
               Directive / Comment / BlankLine, in emission order).
        entry_modes: optional ``{label_name: initial_m_mode}`` seed.
            Function entry labels may be seeded here so the analysis
            starts with a known mode rather than ⊥. Common values:
                {entry_name: 8}   — m8 entry (the default)
                {entry_name: 16}  — m16 entry (e.g. `@ A: u16`)
                {entry_name: None}— "unknown" (legal but conservative)
        entry_x_modes: optional ``{label_name: initial_x_mode}`` seed.
            R65 keeps x16 across all functions; pass ``None`` to use
            the default seeding (x16 at the program entry).

    Returns:
        ModeInfo. Every reachable node has a non-empty ``incoming`` set;
        unreachable nodes have an empty set (use ``is_reachable``).
    """
    from r65.compiler.codegen.asm_nodes import Label, Instruction

    entry_modes = entry_modes or {}
    entry_x_modes = entry_x_modes or {}
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
        info.x_incoming[i] = set()
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
    # explicit declared modes) seed their respective indices too. We
    # deliberately do NOT seed an x-mode default here — callers that
    # know the program's actual entry mode pass it via ``entry_x_modes``;
    # otherwise the analysis stays unseeded so consumers (e.g. the
    # mode-directive rewrite pass) won't speculate an `.INDEX` directive
    # for unrelated test fragments.
    #
    # Seeded labels are *frozen* — fall-through from a textually
    # preceding function (which was a no-op in the runtime CFG, since
    # functions end in RTS/RTL/RTI) must not contaminate the seeded
    # mode. Without this, a function with a `.ACCU 8` anchor whose
    # textual predecessor is a RawAsm header (mode unknown) would have
    # incoming = {None, 8}, and unique_mode_at would yield None.
    frozen_m: Set[int] = set()
    frozen_x: Set[int] = set()
    if n > 0:
        info.incoming[0].add(entry_modes.get('__entry__', None))
        if '__entry__' in entry_x_modes:
            info.x_incoming[0].add(entry_x_modes['__entry__'])
            frozen_x.add(0)
    for label, mode in entry_modes.items():
        if label in label_idx:
            idx = label_idx[label]
            info.incoming[idx] = {mode}
            frozen_m.add(idx)
    for label, mode in entry_x_modes.items():
        if label in label_idx:
            idx = label_idx[label]
            info.x_incoming[idx] = {mode}
            frozen_x.add(idx)

    # Worklist iteration. Lattice height is small (≤3 distinct values
    # in any incoming set), so this converges in O(n) iterations of the
    # full pass for any realistic program.
    changed = True
    while changed:
        changed = False
        for i in range(n):
            in_set = info.incoming[i]
            in_x_set = info.x_incoming[i]
            if not in_set and not in_x_set:
                continue  # unreachable so far — don't propagate ⊥-only sets
            out_set = _transfer(in_set, nodes[i], flag=M_FLAG) if in_set else set()
            out_x_set = _transfer(in_x_set, nodes[i], flag=X_FLAG) if in_x_set else set()
            node = nodes[i]

            # Fall-through successor (next index), unless this node
            # unconditionally transfers control elsewhere. Frozen
            # successors (seeded function/block entries) reject
            # incoming propagation — their mode is authoritative.
            if not _is_unconditional_terminator(node) and i + 1 < n:
                nxt = i + 1
                if out_set and nxt not in frozen_m and not out_set.issubset(info.incoming[nxt]):
                    info.incoming[nxt] |= out_set
                    changed = True
                if out_x_set and nxt not in frozen_x and not out_x_set.issubset(info.x_incoming[nxt]):
                    info.x_incoming[nxt] |= out_x_set
                    changed = True

            # Branch / jump successor.
            target = _branch_target(node)
            if target is not None and target in label_idx:
                tgt_idx = label_idx[target]
                if out_set and tgt_idx not in frozen_m and not out_set.issubset(info.incoming[tgt_idx]):
                    info.incoming[tgt_idx] |= out_set
                    changed = True
                if out_x_set and tgt_idx not in frozen_x and not out_x_set.issubset(info.x_incoming[tgt_idx]):
                    info.x_incoming[tgt_idx] |= out_x_set
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
    from r65.compiler.codegen.asm_nodes import Instruction, ModeChange

    end = min(start_idx + max_lookahead, len(nodes))
    for k in range(start_idx, end):
        node = nodes[k]
        if isinstance(node, ModeChange) and node.flag == 'ACCU':
            return node.bits
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
