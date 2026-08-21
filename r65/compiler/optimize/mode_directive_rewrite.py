# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Reconcile ``ModeChange`` (`.ACCU` / `.INDEX`) directives in a
post-peephole asm node stream with the actual mode dataflow, so
peephole's individual passes don't have to keep the directives in
sync with the SEP/REP they pair with.

Two flavors of `ModeChange` in the input:

  - **Label-anchored** — directives that appear immediately after a
    Label (with only blank lines / comments between). Codegen emits
    these to assert WLA-DX's view of the mode at function and block
    entries based on MIR-derived knowledge. The asm-level dataflow
    can't reconstruct this — it doesn't see across JSR/JSL — so we
    *preserve* these directives and use them to seed the dataflow.

  - **Mid-block** — directives codegen emitted next to the SEP/REP
    that triggered them. Peephole may have moved, removed, or
    duplicated those SEP/REP, leaving the directives stale. We *strip*
    these and re-emit fresh ones immediately before each instruction
    whose runtime mode (per dataflow) doesn't match what WLA-DX
    currently believes.

The combined output: directives are accurate at every point in the
stream, without the peephole passes needing to bookkeep them.
"""

from __future__ import annotations

import re

from typing import Dict, List, Optional, Set, Tuple

from r65.compiler.codegen.asm_nodes import (
    AsmNode, BlankLine, Comment, Instruction, Label, ModeChange,
)
from r65.compiler.optimize.asm_mode_dataflow import compute_modes


# Bare identifiers inside an operand, directive argument, or raw asm
# blob. Over-approximating here is safe: a stray match only makes
# `_reconcile_anchors_with_cfg` leave one more anchor untouched.
_IDENTIFIER_RE = re.compile(r'[A-Za-z_][A-Za-z0-9_]*')


# ---------------------------------------------------------------------------
# Directive recognition
# ---------------------------------------------------------------------------


def _mode_directive_kind(node: AsmNode) -> Optional[Tuple[str, int]]:
    """If ``node`` is a `.ACCU N` or `.INDEX N` mode directive, return
    a tuple of (flag, bits) — e.g. (``"ACCU"``, 8). Otherwise return None.

    Mode directives flow through the canonical typed `ModeChange` node;
    `Directive(".ACCU", ...)` and `RawAsm(".ACCU N")` are no longer
    used for mode tracking.
    """
    if isinstance(node, ModeChange):
        return (node.flag, node.bits)
    return None


def _is_skippable(node: AsmNode) -> bool:
    """Nodes that don't carry runtime semantics — used to walk past
    blank lines / comments when looking for the directive that *follows*
    a label."""
    return isinstance(node, (Comment, BlankLine))


# ---------------------------------------------------------------------------
# Anchored vs mid-block classification
# ---------------------------------------------------------------------------


def _classify_directives(
    nodes: List[AsmNode],
) -> Tuple[Set[int], Dict[str, int], Dict[str, int]]:
    """Walk ``nodes`` once and classify every mode directive.

    Returns:
        anchored_indices: indices of directives to *preserve* — those
            that are the first mode directive of their kind (.ACCU or
            .INDEX) after a Label, with only blanks/comments between.
            These encode MIR-derived knowledge the dataflow can't
            reconstruct (e.g. m16 entry of a `@ A: u16` function).
        entry_m: ``{label_name: bits}`` for label-anchored .ACCU
            directives, used to seed `compute_modes(entry_modes=...)`.
        entry_x: ``{label_name: bits}`` for label-anchored .INDEX
            directives, used to seed `compute_modes(entry_x_modes=...)`.

    Directives at indices not in ``anchored_indices`` are mid-block
    (paired with SEP/REP/PLP/RTI by the codegen) and will be stripped
    by ``normalize_mode_directives`` before re-emission.
    """
    anchored: Set[int] = set()
    entry_m: Dict[str, int] = {}
    entry_x: Dict[str, int] = {}

    n = len(nodes)
    for i, node in enumerate(nodes):
        if not isinstance(node, Label):
            continue
        seen_m = False
        seen_x = False
        j = i + 1
        while j < n:
            nj = nodes[j]
            if _is_skippable(nj):
                j += 1
                continue
            kind = _mode_directive_kind(nj)
            if kind is None:
                break
            tag, bits = kind
            if tag == 'ACCU' and not seen_m:
                anchored.add(j)
                entry_m[node.name] = bits
                seen_m = True
            elif tag == 'INDEX' and not seen_x:
                anchored.add(j)
                entry_x[node.name] = bits
                seen_x = True
            j += 1
    return anchored, entry_m, entry_x


# ---------------------------------------------------------------------------
# Mode-after-node helper
# ---------------------------------------------------------------------------


# Note on WLA-DX semantics:
# WLA-DX assembles linearly and sizes accumulator/index immediates from
# the most-recently-seen `.ACCU` / `.INDEX` directive. It does NOT
# auto-track SEP/REP/PLP/RTI — those change runtime mode but leave
# WLA-DX's belief stale until another directive is emitted. Our
# `wla_m` / `wla_x` trackers therefore only update when a directive
# passes through (anchored) or we emit one (mid-block); they never
# advance through SEP/REP/PLP/RTI.


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _reconcile_anchors_with_cfg(
    stripped: List[AsmNode],
    stripped_is_anchor: List[bool],
    entry_m: Dict[str, int],
    entry_x: Dict[str, int],
) -> None:
    """Correct label anchors that the asm CFG contradicts, in place.

    An anchored `.ACCU` is codegen's belief about the mode a block is
    entered in, and `compute_modes` *freezes* the label to it — real
    predecessor edges are ignored. That is exactly right for a function
    entry, which the asm-level dataflow can only reach through a JSR/JSL
    it does not model. It is wrong for an ordinary block label, whose
    only predecessors are branches the dataflow can see: peephole may
    have deleted the block-entry SEP/REP that made the anchor true (say,
    because the next instruction overwrote it), leaving a stale hint
    that would make WLA-DX size the block's immediates for a width the
    CPU is not in.

    So: run the dataflow with no seeds at all. Where it *proves* a
    unique mode at an anchored label from in-file edges alone, that
    proof wins and the anchor is rewritten to match. Where it proves
    nothing — an unreachable label, or one only entered by a call — the
    anchor stands.
    """
    from r65.compiler.codegen.asm_nodes import Directive, Instruction, RawAsm
    from r65.compiler.codegen.opcodes import (
        Opcode, BRANCH_OPCODES, JUMP_OPCODES,
    )

    # Only a branch or a jump is an edge `compute_modes` models. Every
    # other way of naming a label — a JSR/JSL, a `#<label>` immediate
    # loaded into a function pointer, a `.DW` jump-table entry — is an
    # entry the dataflow cannot see, so the mode it arrives in is not
    # constrained by the branches that also reach it. Leave those
    # anchors alone.
    #
    # (Vector-table entries like `nmi_handler` are named only by the
    # ROM header, so nothing in the stream mentions them at all; they
    # are protected instead by the `is_reachable` test below, which is
    # false for a label with no in-file edges.)
    escaped: Set[str] = set()

    def note_names(text: str) -> None:
        escaped.update(_IDENTIFIER_RE.findall(text))

    for node in stripped:
        if isinstance(node, Instruction):
            if node.opcode in BRANCH_OPCODES or node.opcode in JUMP_OPCODES:
                continue
            value = getattr(node.operand, 'value', None)
            if isinstance(value, str):
                note_names(value)
        elif isinstance(node, Directive):
            for arg in getattr(node, 'args', ()) or ():
                if isinstance(arg, str):
                    note_names(arg)
        elif isinstance(node, RawAsm):
            note_names(node.text)

    cfg = compute_modes(stripped)

    n = len(stripped)
    for i, node in enumerate(stripped):
        if not isinstance(node, Label) or node.name in escaped:
            continue
        if not cfg.is_reachable(i):
            continue

        proven_m = cfg.unique_mode_at(i)
        proven_x = cfg.unique_x_mode_at(i)

        j = i + 1
        while j < n:
            nj = stripped[j]
            if _is_skippable(nj):
                j += 1
                continue
            kind = _mode_directive_kind(nj)
            if kind is None or not stripped_is_anchor[j]:
                break
            tag, bits = kind
            if tag == 'ACCU' and proven_m is not None and proven_m != bits:
                stripped[j] = ModeChange('ACCU', proven_m, nj.source_loc)
                entry_m[node.name] = proven_m
            elif tag == 'INDEX' and proven_x is not None and proven_x != bits:
                stripped[j] = ModeChange('INDEX', proven_x, nj.source_loc)
                entry_x[node.name] = proven_x
            j += 1


def normalize_mode_directives(nodes: List[AsmNode]) -> List[AsmNode]:
    """Reconcile ``.ACCU`` / ``.INDEX`` directives in ``nodes`` against
    the asm-mode dataflow.

    Algorithm:
      1. Classify every existing mode directive as label-anchored
         (preserve, MIR-derived) or mid-block (strip, paired with SEP/
         REP/PLP/RTI which peephole may have rewritten).
      2. Build a stripped node list that keeps the anchored directives
         in place but drops everything else.
      3. Run ``compute_modes`` on the stripped list, seeded with the
         anchored directives so the dataflow knows the runtime mode at
         function and block entries.
      4. Walk the stripped list. Track WLA-DX's currently-believed mode
         (the value of the most-recently-emitted directive of each
         kind). Wherever the dataflow reports a unique mode that
         differs from WLA-DX's belief, insert a fresh directive of the
         appropriate kind before that node.

    The result has correct directives at every mode change without
    requiring peephole's individual passes to bookkeep the pairing.
    """
    if not nodes:
        return nodes

    anchored, entry_m, entry_x = _classify_directives(nodes)

    # Step 2: keep anchored ModeChange directives; drop mid-block ones
    # (we'll re-emit fresh ModeChange nodes from the dataflow result
    # below).
    stripped: List[AsmNode] = []
    # Map from index-in-stripped → True if this slot is an anchored
    # directive carried over from the input (so the rebuild loop knows
    # not to re-insert another `.ACCU` immediately before it).
    stripped_is_anchor: List[bool] = []
    for i, node in enumerate(nodes):
        kind = _mode_directive_kind(node)
        if kind is None:
            stripped.append(node)
            stripped_is_anchor.append(False)
        elif i in anchored:
            stripped.append(node)
            stripped_is_anchor.append(True)
        # else: mid-block directive → drop.

    _reconcile_anchors_with_cfg(stripped, stripped_is_anchor, entry_m, entry_x)

    info = compute_modes(
        stripped,
        entry_modes=entry_m,
        entry_x_modes=entry_x,
    )

    # Step 4: walk and re-emit. Track WLA-DX's belief from anchored
    # directives we kept (they update the belief without requiring an
    # extra emission). We only insert fresh directives immediately
    # *before* an Instruction — labels, comments, raw asm, etc. carry
    # no runtime semantics and don't move the assembler's cursor in a
    # way that would mis-size the next immediate.
    result: List[AsmNode] = []
    # WLA-DX's default state at the start of assembly is m8 / x8. Seed
    # our trackers to match so we don't emit a redundant `.ACCU 8` at
    # the very first m8 instruction (which would otherwise differ from
    # the `None` sentinel).
    wla_m: Optional[int] = 8
    wla_x: Optional[int] = 8
    for i, node in enumerate(stripped):
        if stripped_is_anchor[i]:
            # Anchored directive — passes through and updates belief.
            kind = _mode_directive_kind(node)
            assert kind is not None
            tag, bits = kind
            if tag == 'ACCU':
                wla_m = bits
            else:
                wla_x = bits
            result.append(node)
            continue

        if isinstance(node, Instruction):
            # Decide whether the dataflow needs us to insert a fresh
            # directive before this instruction so WLA-DX's view
            # matches runtime mode for sizing the immediate (and for
            # subsequent instructions until the next directive).
            #
            # Emitted as `ModeChange` — the canonical typed form. The
            # serializer in `emitter.emit_node` renders it indented so
            # it doesn't collide with column-0 section directives.
            m_here = info.unique_mode_at(i)
            if m_here is not None and m_here != wla_m:
                result.append(ModeChange('ACCU', m_here))
                wla_m = m_here
            x_here = info.unique_x_mode_at(i)
            if x_here is not None and x_here != wla_x:
                result.append(ModeChange('INDEX', x_here))
                wla_x = x_here

        result.append(node)
        # wla_m / wla_x are intentionally NOT advanced through SEP/REP/
        # PLP/RTI — see the WLA-DX semantics note above. They only
        # change when an actual directive is emitted (handled above).

    return result
