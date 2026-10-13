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

import functools
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from r65.compiler.codegen.constants import M_FLAG, X_FLAG
from r65.compiler.codegen.opcodes import (
    Opcode,
    BRANCH_OPCODES,
    CALL_OPCODES,
    JUMP_OPCODES,
    RETURN_OPCODES,
    M16_IMMEDIATE_OPCODES,
)




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
    # An accumulator immediate (sized by the m flag) above $FF can only have
    # been meant for m16 - evidence for mode-fix decisions at merge points.
    if node.opcode not in M16_IMMEDIATE_OPCODES:
        return None
    if not isinstance(node.operand, Immediate):
        return None
    val = node.operand.value
    if isinstance(val, int) and val > 0xFF:
        return 16
    # Sub-$100 immediates fit either mode; they don't constrain.
    return None


# Every 65816 mnemonic the compiler knows. An inline-asm line starting
# with one of these (other than the mode-setting ones handled first)
# leaves the m/x flags alone.
_MNEMONICS = frozenset(op.name.split('_')[0] for op in Opcode)
_IMMEDIATE_RE = re.compile(r'#\s*(\$[0-9A-F]+|%[01]+|[0-9]+)\b')


@functools.lru_cache(maxsize=4096)
def _rawasm_effect(text: str):
    """The (m, x) lattice masks a raw asm line leaves, None for each flag
    it does not touch, or None if it touches neither.

    Inline asm reaches the stream as text, so SEP/REP/PLP inside an
    `asm!` would otherwise be invisible. A line that cannot be read with
    certainty - an unknown mnemonic (a macro), an `.INCLUDE`, a SEP/REP
    whose immediate does not parse - makes both flags unknown.
    """
    line = text.split(';', 1)[0].strip().upper()
    if not line:
        return None
    if line.startswith('.'):
        if line.startswith('.INCLUDE'):
            return (_BIT_UNK, _BIT_UNK)
        return None
    if line.endswith(':'):
        return None
    mnemonic = line.split()[0].split('.')[0]
    if mnemonic in ('SEP', 'REP'):
        match = _IMMEDIATE_RE.search(line)
        if match is None:
            return (_BIT_UNK, _BIT_UNK)
        digits = match.group(1)
        if digits.startswith('$'):
            value = int(digits[1:], 16)
        elif digits.startswith('%'):
            value = int(digits[1:], 2)
        else:
            value = int(digits)
        bits = _BIT_M8 if mnemonic == 'SEP' else _BIT_M16
        return (bits if value & M_FLAG else None,
                bits if value & X_FLAG else None)
    if mnemonic in ('PLP', 'RTI', 'XCE'):
        return (_BIT_UNK, _BIT_UNK)
    if mnemonic in _MNEMONICS:
        return None
    return (_BIT_UNK, _BIT_UNK)


def _modeled_effect(node, call_exit_modes: Dict[str, Mode]):
    """The (m, x) masks `node` leaves under the modeled-calls analysis
    (None per flag for the ordinary transfer), or None for neither."""
    from r65.compiler.codegen.asm_nodes import Instruction, RawAsm

    if isinstance(node, RawAsm):
        return _rawasm_effect(node.text)
    if not isinstance(node, Instruction):
        return None
    if node.opcode in CALL_OPCODES:
        # A call returns in the callee's exit mode; an unknown callee
        # (indirect, or a label codegen did not describe) in any mode.
        target = getattr(node.operand, 'value', None)
        if isinstance(target, str) and target in call_exit_modes:
            return (_mode_bit(call_exit_modes[target]), None)
        return (_BIT_UNK, None)
    if node.opcode == Opcode.XCE:
        return (_BIT_UNK, _BIT_UNK)
    return None


def compute_modes(
    nodes: List,
    entry_modes: Optional[Dict[str, Mode]] = None,
    entry_x_modes: Optional[Dict[str, Mode]] = None,
    start_bits: Optional[Tuple[int, int]] = None,
    call_exit_modes: Optional[Dict[str, Mode]] = None,
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
        start_bits: optional ``(m_mask, x_mask)`` lattice masks for
            nodes[0], replacing the program-start seed. For a slice of a
            larger stream, pass the masks the whole stream computed at
            the slice's first node so the slice sees the same modes.
        call_exit_modes: when given, model what the plain analysis
            passes straight through: a JSR/JSL leaves the callee's exit
            mode from this ``{label: mode}`` map (unknown for any other
            callee), XCE leaves both flags unknown, and inline asm
            (RawAsm) applies its SEP/REP/PLP - unknown where it cannot
            be read. Without it, calls and RawAsm are mode-transparent.

    Returns:
        ModeInfo. Every reachable node has a non-empty ``incoming`` set;
        unreachable nodes have an empty set (use ``is_reachable``).
    """
    from r65.compiler.codegen.asm_nodes import Immediate, Instruction, Label, RawAsm

    entry_modes = entry_modes or {}
    entry_x_modes = entry_x_modes or {}
    n = len(nodes)
    info = ModeInfo()
    info.incoming = [0] * n
    info.x_incoming = [0] * n
    info.required = [None] * n
    info.pred_count = [0] * n

    # Pre-pass: resolve each node once. `fixed_m[i]` / `fixed_x[i]` is the
    # mask the node leaves when reached (-1: passes its input through),
    # `falls[i]` whether control falls through to i+1, and `target[i]` the
    # label a branch/jump names. Every transfer is the identity or a
    # constant, so the loop below is pure mask arithmetic.
    required = info.required
    fixed_m = [-1] * n
    fixed_x = [-1] * n
    falls = [True] * n
    target_name: List[Optional[str]] = [None] * n
    label_idx: Dict[str, int] = {}
    modeled = call_exit_modes is not None
    for i, node in enumerate(nodes):
        cls = type(node)
        if cls is Instruction:
            op = node.opcode
            operand = node.operand
            if op in _UNCONDITIONAL_TRANSFER:
                falls[i] = False
            if op in M16_IMMEDIATE_OPCODES:
                required[i] = _required_mode_of(node)
            elif op == Opcode.SEP_IMMEDIATE or op == Opcode.REP_IMMEDIATE:
                if isinstance(operand, Immediate) and isinstance(operand.value, int):
                    bits = _BIT_M8 if op == Opcode.SEP_IMMEDIATE else _BIT_M16
                    if operand.value & M_FLAG:
                        fixed_m[i] = bits
                    if operand.value & X_FLAG:
                        fixed_x[i] = bits
            elif op == Opcode.PLP or op == Opcode.RTI:
                fixed_m[i] = fixed_x[i] = _BIT_UNK
            if op in BRANCH_OPCODES or op in JUMP_OPCODES:
                value = getattr(operand, 'value', None) if operand is not None else None
                if isinstance(value, str):
                    target_name[i] = value
        elif cls is Label:
            # First occurrence wins if (somehow) duplicated.
            label_idx.setdefault(node.name, i)
        elif cls is not RawAsm:
            continue
        if modeled:
            # Calls, XCE and inline asm, which the plain analysis passes
            # straight through.
            effect = _modeled_effect(node, call_exit_modes)
            if effect is not None:
                if effect[0] is not None:
                    fixed_m[i] = effect[0]
                if effect[1] is not None:
                    fixed_x[i] = effect[1]
    target = [label_idx.get(name, -1) if name is not None else -1 for name in target_name]

    # Count physical predecessors (fall-through + branch/jump sources).
    # The fall-through count is fixed by source order: every node has at
    # most one fall-through predecessor (the previous index), unless that
    # previous node unconditionally transfers control elsewhere.
    pred_count = info.pred_count
    for i in range(n):
        if falls[i] and i + 1 < n:
            pred_count[i + 1] += 1
        if target[i] >= 0:
            pred_count[target[i]] += 1

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
    frozen_m = bytearray(n)
    frozen_x = bytearray(n)
    if n > 0 and start_bits is not None:
        info.incoming[0], info.x_incoming[0] = start_bits
    elif n > 0:
        info.incoming[0] |= _mode_bit(entry_modes.get('__entry__', None))
        if '__entry__' in entry_x_modes:
            info.x_incoming[0] |= _mode_bit(entry_x_modes['__entry__'])
            frozen_x[0] = 1
    for label, mode in entry_modes.items():
        if label in label_idx:
            idx = label_idx[label]
            info.incoming[idx] = _mode_bit(mode)
            frozen_m[idx] = 1
    for label, mode in entry_x_modes.items():
        if label in label_idx:
            idx = label_idx[label]
            info.x_incoming[idx] = _mode_bit(mode)
            frozen_x[idx] = 1

    # Propagate to the least fixpoint. Each transfer is the identity or a
    # constant, both distributive over union, so the fixpoint does not
    # depend on visiting order: sweep forward over the nodes whose input
    # grew, and re-sweep only from the earliest one a back-edge grew.
    # Modes are 3-bit masks; union is `|`.
    inc = info.incoming
    xinc = info.x_incoming
    dirty = bytearray(1 if (inc[i] or xinc[i]) else 0 for i in range(n))
    start = 0
    while start < n:
        rewind = n
        for i in range(start, n):
            if not dirty[i]:
                continue
            dirty[i] = 0
            in_bits = inc[i]
            in_x_bits = xinc[i]
            fm = fixed_m[i]
            fx = fixed_x[i]
            out_bits = (in_bits if fm < 0 else fm) if in_bits else 0
            out_x_bits = (in_x_bits if fx < 0 else fx) if in_x_bits else 0

            # Fall-through successor, then branch/jump successor. Frozen
            # successors (seeded function/block entries) reject incoming
            # propagation — their mode is authoritative.
            if falls[i] and i + 1 < n:
                nxt = i + 1
                if out_bits and not frozen_m[nxt] and inc[nxt] | out_bits != inc[nxt]:
                    inc[nxt] |= out_bits
                    dirty[nxt] = 1
                if out_x_bits and not frozen_x[nxt] and xinc[nxt] | out_x_bits != xinc[nxt]:
                    xinc[nxt] |= out_x_bits
                    dirty[nxt] = 1
            tgt = target[i]
            if tgt >= 0:
                grew = False
                if out_bits and not frozen_m[tgt] and inc[tgt] | out_bits != inc[tgt]:
                    inc[tgt] |= out_bits
                    grew = True
                if out_x_bits and not frozen_x[tgt] and xinc[tgt] | out_x_bits != xinc[tgt]:
                    xinc[tgt] |= out_x_bits
                    grew = True
                if grew:
                    dirty[tgt] = 1
                    if tgt <= i and tgt < rewind:
                        rewind = tgt
        start = rewind

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
