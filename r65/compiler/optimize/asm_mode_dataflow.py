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

# The set of modes that may hold at a node is a tiny lattice — any subset
# of {8, 16, None} — so it is stored as a 3-bit mask rather than a Python
# set. This removes ~2M set allocations per classickong compile (the mode
# dataflow ran as ~38% of compile time, dominated by per-node set() calls).
# A mask of 0 is the empty set (⊥ — unreachable / not yet reached).
_BIT_M8 = 1
_BIT_M16 = 2
_BIT_UNK = 4            # the "unknown" mode (None)
_BIT_KNOWN = _BIT_M8 | _BIT_M16


def _mode_bit(mode: Mode) -> int:
    """Encode a single mode (8 / 16 / None) as its lattice bit."""
    if mode == 8:
        return _BIT_M8
    if mode == 16:
        return _BIT_M16
    return _BIT_UNK


def _bits_to_set(bits: int) -> Set[Mode]:
    """Decode a lattice bitmask back to a set of modes (query-boundary helper)."""
    s: Set[Mode] = set()
    if bits & _BIT_M8:
        s.add(8)
    if bits & _BIT_M16:
        s.add(16)
    if bits & _BIT_UNK:
        s.add(None)
    return s


def _unique_of_bits(bits: int) -> Mode:
    """The single definite mode iff the mask is exactly {8} or {16}, else None.

    Matches the old _unique: a value is unique only when exactly one known
    mode is present and "unknown" is not — i.e. the mask is _BIT_M8 or _BIT_M16.
    """
    if bits == _BIT_M8:
        return 8
    if bits == _BIT_M16:
        return 16
    return None


@dataclass
class ModeInfo:
    """Per-node mode information from the dataflow analysis.

    Internal representation: `incoming[i]` is the bitmask of m-flag modes
    (accumulator width) that may hold immediately *before* nodes[i]
    executes; `x_incoming[i]` is the parallel mask for the x-flag
    (index-register width). Bits are _BIT_M8 / _BIT_M16 / _BIT_UNK; 0 is
    the empty set. `pred_count[i]` is the number of distinct physical
    predecessor edges into nodes[i] (fall-through plus branch/jump sources).

    All four are dense lists indexed by node position (filled by
    compute_modes). The two flag dimensions are tracked in parallel rather
    than as a Cartesian product because SEP/REP can update them
    independently and callers ask about them independently.
    """

    incoming: List[int] = field(default_factory=list)
    x_incoming: List[int] = field(default_factory=list)
    required: List[Mode] = field(default_factory=list)
    pred_count: List[int] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Query helpers — preferred read API for consumers.
    # ------------------------------------------------------------------

    def incoming_at(self, idx: int) -> Set[Mode]:
        return _bits_to_set(self.incoming[idx]) if 0 <= idx < len(self.incoming) else set()

    def x_incoming_at(self, idx: int) -> Set[Mode]:
        return _bits_to_set(self.x_incoming[idx]) if 0 <= idx < len(self.x_incoming) else set()

    def required_at(self, idx: int) -> Mode:
        return self.required[idx] if 0 <= idx < len(self.required) else None

    def unique_mode_at(self, idx: int) -> Mode:
        """Return the m-flag mode if all paths agree on a definite value, else None."""
        return _unique_of_bits(self.incoming[idx]) if 0 <= idx < len(self.incoming) else None

    def unique_x_mode_at(self, idx: int) -> Mode:
        """Return the x-flag mode if all paths agree on a definite value, else None."""
        return _unique_of_bits(self.x_incoming[idx]) if 0 <= idx < len(self.x_incoming) else None

    def has_mixed_known(self, idx: int) -> bool:
        """True iff some path arrives in m8 and some in m16."""
        bits = self.incoming[idx] if 0 <= idx < len(self.incoming) else 0
        return (bits & _BIT_KNOWN) == _BIT_KNOWN

    def is_reachable(self, idx: int) -> bool:
        return 0 <= idx < len(self.incoming) and self.incoming[idx] != 0

    def is_join(self, idx: int) -> bool:
        """True iff this node has more than one physical predecessor edge.

        A label that is only reachable by fall-through from a single
        predecessor is NOT a join; mode-fix coercion at such a label is
        always redundant with whatever coercion was applied upstream.
        """
        return (self.pred_count[idx] if 0 <= idx < len(self.pred_count) else 0) > 1


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


def _transfer(in_bits: int, node, *, flag: int = M_FLAG) -> int:
    """Compute the OUT mode bitmask for a node given its IN bitmask.

    `flag` selects which P-flag to track (M_FLAG for the accumulator,
    X_FLAG for index registers). The transfer function is identical
    for both — only the bit being inspected on SEP/REP differs.
    """
    from r65.compiler.codegen.asm_nodes import Instruction, Immediate

    if not isinstance(node, Instruction):
        # Labels, directives, comments — pass mode through unchanged.
        return in_bits

    op = node.opcode
    if op == Opcode.SEP_IMMEDIATE:
        if isinstance(node.operand, Immediate) and isinstance(node.operand.value, int):
            if node.operand.value & flag:
                return _BIT_M8
    elif op == Opcode.REP_IMMEDIATE:
        if isinstance(node.operand, Immediate) and isinstance(node.operand.value, int):
            if node.operand.value & flag:
                return _BIT_M16
    elif op in (Opcode.PLP, Opcode.RTI):
        return _BIT_UNK

    return in_bits


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
    info.incoming = [0] * n
    info.x_incoming = [0] * n
    info.required = [None] * n
    info.pred_count = [0] * n

    # Pre-pass: index labels and stash required-mode per instruction.
    label_idx: Dict[str, int] = {}
    for i, node in enumerate(nodes):
        if isinstance(node, Label):
            # First occurrence wins if (somehow) duplicated.
            label_idx.setdefault(node.name, i)
        info.required[i] = _required_mode_of(node)

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
        info.incoming[0] |= _mode_bit(entry_modes.get('__entry__', None))
        if '__entry__' in entry_x_modes:
            info.x_incoming[0] |= _mode_bit(entry_x_modes['__entry__'])
            frozen_x.add(0)
    for label, mode in entry_modes.items():
        if label in label_idx:
            idx = label_idx[label]
            info.incoming[idx] = _mode_bit(mode)
            frozen_m.add(idx)
    for label, mode in entry_x_modes.items():
        if label in label_idx:
            idx = label_idx[label]
            info.x_incoming[idx] = _mode_bit(mode)
            frozen_x.add(idx)

    # Worklist iteration. Lattice height is small (≤3 distinct modes in any
    # incoming mask), so this converges in O(n) iterations of the full pass
    # for any realistic program. Modes are 3-bit masks: union is `|`, the
    # "already a subset" test is `cur | out == cur` (i.e. union doesn't grow).
    inc = info.incoming
    xinc = info.x_incoming
    changed = True
    while changed:
        changed = False
        for i in range(n):
            in_bits = inc[i]
            in_x_bits = xinc[i]
            if not in_bits and not in_x_bits:
                continue  # unreachable so far — don't propagate ⊥
            node = nodes[i]
            out_bits = _transfer(in_bits, node, flag=M_FLAG) if in_bits else 0
            out_x_bits = _transfer(in_x_bits, node, flag=X_FLAG) if in_x_bits else 0

            # Fall-through successor (next index), unless this node
            # unconditionally transfers control elsewhere. Frozen
            # successors (seeded function/block entries) reject
            # incoming propagation — their mode is authoritative.
            if not _is_unconditional_terminator(node) and i + 1 < n:
                nxt = i + 1
                if out_bits and nxt not in frozen_m:
                    merged = inc[nxt] | out_bits
                    if merged != inc[nxt]:
                        inc[nxt] = merged
                        changed = True
                if out_x_bits and nxt not in frozen_x:
                    merged = xinc[nxt] | out_x_bits
                    if merged != xinc[nxt]:
                        xinc[nxt] = merged
                        changed = True

            # Branch / jump successor.
            target = _branch_target(node)
            if target is not None and target in label_idx:
                tgt_idx = label_idx[target]
                if out_bits and tgt_idx not in frozen_m:
                    merged = inc[tgt_idx] | out_bits
                    if merged != inc[tgt_idx]:
                        inc[tgt_idx] = merged
                        changed = True
                if out_x_bits and tgt_idx not in frozen_x:
                    merged = xinc[tgt_idx] | out_x_bits
                    if merged != xinc[tgt_idx]:
                        xinc[tgt_idx] = merged
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
