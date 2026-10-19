# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Peephole optimization for R65 assembly code.

Applies local optimizations to AsmNode instruction sequences to eliminate
redundant operations and improve code quality.

Pattern matching uses the typed Opcode enum; a few helpers
(`_is_indexed_addressing`, `_reads_register`) match on opcode names.
"""

import itertools
from typing import List, NamedTuple, Optional, Tuple, TYPE_CHECKING, Set
from dataclasses import dataclass

from r65.compiler.codegen.opcodes import (
    Opcode, mnemonic, nz_register, ACCUMULATOR_OPCODES,
    BRANCH_OPCODES, JUMP_OPCODES, CALL_OPCODES,
    LOAD_A_OPCODES, LOAD_X_OPCODES, LOAD_Y_OPCODES,
    STORE_A_OPCODES, STORE_X_OPCODES, STORE_Y_OPCODES,
    MEMORY_WRITE_OPCODES, NZ_OF_A_OPCODES, RETURN_OPCODES, MODIFIES_A_OPCODES,
    CARRY_READ_OPCODES, CARRY_WRITE_OPCODES,
    PUSH_OPCODES, PULL_OPCODES,
)
from r65.compiler.codegen.constants import M_FLAG, X_FLAG

if TYPE_CHECKING:
    from r65.compiler.codegen.asm_nodes import AsmNode, Instruction


# ============================================================================
# Opcode Categories for Pattern Matching
# ============================================================================

# Indirect addressing opcodes - these can read/write to any address
# Dead store analysis must be conservative when these are present
INDIRECT_ADDRESSING_OPCODES: Set[Opcode] = {
    # DP indirect
    Opcode.LDA_DP_INDIRECT, Opcode.LDA_DP_INDIRECT_X, Opcode.LDA_DP_INDIRECT_Y,
    Opcode.LDA_DP_INDIRECT_LONG, Opcode.LDA_DP_INDIRECT_LONG_Y,
    Opcode.STA_DP_INDIRECT, Opcode.STA_DP_INDIRECT_X, Opcode.STA_DP_INDIRECT_Y,
    Opcode.STA_DP_INDIRECT_LONG, Opcode.STA_DP_INDIRECT_LONG_Y,
    # Stack indirect
    Opcode.LDA_STACK_INDIRECT_Y, Opcode.STA_STACK_INDIRECT_Y,
    # ADC/AND/CMP/EOR/ORA/SBC indirect variants
    Opcode.ADC_DP_INDIRECT, Opcode.ADC_DP_INDIRECT_X, Opcode.ADC_DP_INDIRECT_Y,
    Opcode.ADC_DP_INDIRECT_LONG, Opcode.ADC_DP_INDIRECT_LONG_Y, Opcode.ADC_STACK_INDIRECT_Y,
    Opcode.AND_DP_INDIRECT, Opcode.AND_DP_INDIRECT_X, Opcode.AND_DP_INDIRECT_Y,
    Opcode.AND_DP_INDIRECT_LONG, Opcode.AND_DP_INDIRECT_LONG_Y, Opcode.AND_STACK_INDIRECT_Y,
    Opcode.CMP_DP_INDIRECT, Opcode.CMP_DP_INDIRECT_X, Opcode.CMP_DP_INDIRECT_Y,
    Opcode.CMP_DP_INDIRECT_LONG, Opcode.CMP_DP_INDIRECT_LONG_Y, Opcode.CMP_STACK_INDIRECT_Y,
    Opcode.EOR_DP_INDIRECT, Opcode.EOR_DP_INDIRECT_X, Opcode.EOR_DP_INDIRECT_Y,
    Opcode.EOR_DP_INDIRECT_LONG, Opcode.EOR_DP_INDIRECT_LONG_Y, Opcode.EOR_STACK_INDIRECT_Y,
    Opcode.ORA_DP_INDIRECT, Opcode.ORA_DP_INDIRECT_X, Opcode.ORA_DP_INDIRECT_Y,
    Opcode.ORA_DP_INDIRECT_LONG, Opcode.ORA_DP_INDIRECT_LONG_Y, Opcode.ORA_STACK_INDIRECT_Y,
    Opcode.SBC_DP_INDIRECT, Opcode.SBC_DP_INDIRECT_X, Opcode.SBC_DP_INDIRECT_Y,
    Opcode.SBC_DP_INDIRECT_LONG, Opcode.SBC_DP_INDIRECT_LONG_Y, Opcode.SBC_STACK_INDIRECT_Y,
}

# What _hoist_loop_invariant_loads also has to treat as writing A: a call
# returns its value in A (and its body may use A). And the width changes it
# may not hoist across - the immediate is sized for the width where it
# stood, not where it would land.
_HOIST_CLOBBERS_A: Set[Opcode] = {
    Opcode.JSR, Opcode.JSR_INDIRECT_X, Opcode.JSL,
}
_HOIST_CHANGES_WIDTH: Set[Opcode] = {
    Opcode.REP_IMMEDIATE, Opcode.SEP_IMMEDIATE, Opcode.PLP, Opcode.XCE,
}

# Push/pull pairs for redundant stack operation elimination
PUSH_PULL_PAIRS = {PUSH_OPCODES[reg]: PULL_OPCODES[reg]
                   for reg in ('A', 'X', 'Y', 'STATUS', 'D', 'DBR')}

# Map store opcode sets to corresponding load opcode sets (for redundant load elimination)
STORE_TO_LOAD_MAP = {
    # Each store opcode maps to the set of corresponding load opcodes
    **{op: LOAD_A_OPCODES for op in STORE_A_OPCODES},
    **{op: LOAD_X_OPCODES for op in STORE_X_OPCODES},
    **{op: LOAD_Y_OPCODES for op in STORE_Y_OPCODES},
}

# Instructions that read from a memory operand (for dead store analysis)
READS_FROM_MEMORY_OPCODES: Set[Opcode] = (
    LOAD_A_OPCODES | LOAD_X_OPCODES | LOAD_Y_OPCODES |
    # ADC variants
    {Opcode.ADC_DP, Opcode.ADC_DP_X, Opcode.ADC_ABSOLUTE,
     Opcode.ADC_ABSOLUTE_X, Opcode.ADC_ABSOLUTE_Y,
     Opcode.ADC_DP_INDIRECT, Opcode.ADC_DP_INDIRECT_X, Opcode.ADC_DP_INDIRECT_Y,
     Opcode.ADC_DP_INDIRECT_LONG, Opcode.ADC_DP_INDIRECT_LONG_Y,
     Opcode.ADC_LONG, Opcode.ADC_LONG_X, Opcode.ADC_STACK, Opcode.ADC_STACK_INDIRECT_Y} |
    # SBC variants
    {Opcode.SBC_DP, Opcode.SBC_DP_X, Opcode.SBC_ABSOLUTE,
     Opcode.SBC_ABSOLUTE_X, Opcode.SBC_ABSOLUTE_Y,
     Opcode.SBC_DP_INDIRECT, Opcode.SBC_DP_INDIRECT_X, Opcode.SBC_DP_INDIRECT_Y,
     Opcode.SBC_DP_INDIRECT_LONG, Opcode.SBC_DP_INDIRECT_LONG_Y,
     Opcode.SBC_LONG, Opcode.SBC_LONG_X, Opcode.SBC_STACK, Opcode.SBC_STACK_INDIRECT_Y} |
    # AND variants
    {Opcode.AND_DP, Opcode.AND_DP_X, Opcode.AND_ABSOLUTE,
     Opcode.AND_ABSOLUTE_X, Opcode.AND_ABSOLUTE_Y,
     Opcode.AND_DP_INDIRECT, Opcode.AND_DP_INDIRECT_X, Opcode.AND_DP_INDIRECT_Y,
     Opcode.AND_DP_INDIRECT_LONG, Opcode.AND_DP_INDIRECT_LONG_Y,
     Opcode.AND_LONG, Opcode.AND_LONG_X, Opcode.AND_STACK, Opcode.AND_STACK_INDIRECT_Y} |
    # ORA variants
    {Opcode.ORA_DP, Opcode.ORA_DP_X, Opcode.ORA_ABSOLUTE,
     Opcode.ORA_ABSOLUTE_X, Opcode.ORA_ABSOLUTE_Y,
     Opcode.ORA_DP_INDIRECT, Opcode.ORA_DP_INDIRECT_X, Opcode.ORA_DP_INDIRECT_Y,
     Opcode.ORA_DP_INDIRECT_LONG, Opcode.ORA_DP_INDIRECT_LONG_Y,
     Opcode.ORA_LONG, Opcode.ORA_LONG_X, Opcode.ORA_STACK, Opcode.ORA_STACK_INDIRECT_Y} |
    # EOR variants
    {Opcode.EOR_DP, Opcode.EOR_DP_X, Opcode.EOR_ABSOLUTE,
     Opcode.EOR_ABSOLUTE_X, Opcode.EOR_ABSOLUTE_Y,
     Opcode.EOR_DP_INDIRECT, Opcode.EOR_DP_INDIRECT_X, Opcode.EOR_DP_INDIRECT_Y,
     Opcode.EOR_DP_INDIRECT_LONG, Opcode.EOR_DP_INDIRECT_LONG_Y,
     Opcode.EOR_LONG, Opcode.EOR_LONG_X, Opcode.EOR_STACK, Opcode.EOR_STACK_INDIRECT_Y} |
    # CMP variants
    {Opcode.CMP_DP, Opcode.CMP_DP_X, Opcode.CMP_ABSOLUTE,
     Opcode.CMP_ABSOLUTE_X, Opcode.CMP_ABSOLUTE_Y,
     Opcode.CMP_DP_INDIRECT, Opcode.CMP_DP_INDIRECT_X, Opcode.CMP_DP_INDIRECT_Y,
     Opcode.CMP_DP_INDIRECT_LONG, Opcode.CMP_DP_INDIRECT_LONG_Y,
     Opcode.CMP_LONG, Opcode.CMP_LONG_X, Opcode.CMP_STACK, Opcode.CMP_STACK_INDIRECT_Y} |
    # CPX variants
    {Opcode.CPX_DP, Opcode.CPX_ABSOLUTE} |
    # CPY variants
    {Opcode.CPY_DP, Opcode.CPY_ABSOLUTE} |
    # BIT variants
    {Opcode.BIT_DP, Opcode.BIT_DP_X, Opcode.BIT_ABSOLUTE, Opcode.BIT_ABSOLUTE_X}
)

# Control flow instructions that end a basic block
# CALL_OPCODES (JSR/JSL) are included because the callee may modify any memory location,
# so dead store analysis must stop at calls to avoid removing stores that are actually used
CONTROL_FLOW_OPCODES: Set[Opcode] = (
    BRANCH_OPCODES | JUMP_OPCODES | CALL_OPCODES | {Opcode.RTS, Opcode.RTL, Opcode.RTI}
)

# Instructions that modify the stack pointer — invalidate stack-relative operand tracking
STACK_MODIFYING_OPCODES: Set[Opcode] = {
    Opcode.PHA, Opcode.PHX, Opcode.PHY, Opcode.PHP, Opcode.PHD, Opcode.PHB,
    Opcode.PLA, Opcode.PLX, Opcode.PLY, Opcode.PLP, Opcode.PLD, Opcode.PLB,
    Opcode.TCS,
}

# LDA opcodes with deterministic addressing (value depends only on operand/SP)
TRACKABLE_LDA_OPCODES: frozenset = frozenset({
    Opcode.LDA_IMMEDIATE,
    Opcode.LDA_DP,
    Opcode.LDA_ABSOLUTE,
    Opcode.LDA_STACK,
})

# Indirect store opcodes that could alias any address
INDIRECT_STORE_OPCODES: frozenset = frozenset({
    Opcode.STA_DP_INDIRECT, Opcode.STA_DP_INDIRECT_X, Opcode.STA_DP_INDIRECT_Y,
    Opcode.STA_DP_INDIRECT_LONG, Opcode.STA_DP_INDIRECT_LONG_Y,
    Opcode.STA_STACK_INDIRECT_Y,
})


# ============================================================================
# Statistics Tracking
# ============================================================================

@dataclass
class OptimizationStats:
    """Track optimization statistics."""
    redundant_loads_eliminated: int = 0
    dead_stores_eliminated: int = 0
    redundant_transfers_eliminated: int = 0
    redundant_stack_ops_eliminated: int = 0
    redundant_mode_changes_eliminated: int = 0
    dead_mode_changes_eliminated: int = 0
    carry_ops_folded_into_rep: int = 0
    redundant_and_before_sep_eliminated: int = 0
    branch_over_branch_eliminated: int = 0
    branch_to_next_eliminated: int = 0
    branch_threading_applied: int = 0
    tracked_loads_eliminated: int = 0
    identity_copies_eliminated: int = 0
    loop_invariant_loads_hoisted: int = 0
    unreachable_nodes_eliminated: int = 0
    stz_conversions: int = 0
    inc_dec_folded: int = 0
    redundant_cmp_zero_eliminated: int = 0

    @property
    def total(self) -> int:
        from dataclasses import fields
        return sum(getattr(self, f.name) for f in fields(self))


# ============================================================================
# Mode Switch Helpers
# ============================================================================

def _mode_switch_residue(node):
    """What must stay of a SEP/REP once its M bit is known to be a no-op.

    A switch can carry other flags with the M bit: `REP #$21` is the mode
    switch plus the carry clear the frame setup folded into it, and
    `REP #$30` also sets 16-bit X/Y. Dropping such a switch because A is
    already the right width drops those too (for example the carry clear
    an interrupt prologue's frame allocation depends on). Returns
    the replacement nodes: none for a bare `#$20`, CLC/SEC for carry
    alone, else the switch without its M bit.
    """
    from r65.compiler.codegen.asm_nodes import Instruction, Immediate
    rest = node.operand.value & ~0x20
    if rest == 0:
        return []
    if rest == 0x01:
        op = Opcode.CLC if node.opcode == Opcode.REP_IMMEDIATE else Opcode.SEC
        return [Instruction(op, None, node.comment, node.source_loc)]
    return [Instruction(node.opcode, Immediate(rest), node.comment, node.source_loc)]


# ============================================================================
# Accumulator / N-Z Liveness
# ============================================================================
#
# A pass that deletes an instruction writing A or N/Z (an LDA, an
# `AND #$00FF`) is sound only if nothing downstream reads what it wrote:
# `STA $10; LDA $10; BEQ` needs the LDA for its Z flag even though A is
# unchanged. `_live_after` answers that by walking forward from a
# position along every control-flow path, tracking three resources:
#
#   'A'  - the accumulator's low byte
#   'B'  - its high byte (read by XBA, TAX/TAY in x16, TCD/TCS, the
#          MVN/MVP count, and every A read in m16)
#   'NZ' - the N and Z flags
#
# No flag crosses a call or a return in the R65 ABI, so JSR/JSL and
# RTS/RTL/RTI kill N/Z. Both read A (an argument or return register).
# They read B in m16 (part of a u16 A) and, in m8, only where the ABI
# passes a value in B (a `@ B` parameter or a B return slot) or where
# the function is unknown; an m8 callee treats B as garbage otherwise.
# A call to a known function that does not preserve A also kills A and
# B: the caller cannot rely on either surviving it.
# Straight-line inline asm is read line by line for the A and B it may
# read; it is assumed to write neither and passes N/Z through. Inline asm
# that branches, jumps, returns, pushes P, changes mode or holds a label
# or data, an opcode missing from the tables below, an indirect jump, a
# branch out of the node list, and a walk past `_LIVENESS_BUDGET`
# instructions all count as reading everything.

_NZ_WRITERS = frozenset({
    'LDA', 'LDX', 'LDY', 'AND', 'ORA', 'EOR', 'ADC', 'SBC',
    'CMP', 'CPX', 'CPY', 'INC', 'DEC', 'INX', 'DEX', 'INY', 'DEY',
    'ASL', 'LSR', 'ROL', 'ROR', 'TAX', 'TAY', 'TXA', 'TYA', 'TXY', 'TYX',
    'TSX', 'TDC', 'TCD', 'TSC', 'PLA', 'PLX', 'PLY', 'PLB', 'PLD', 'PLP',
    'XBA', 'BIT',  # BIT #imm writes Z only, so it is excluded below
})
_NZ_READERS = frozenset({'BEQ', 'BNE', 'BMI', 'BPL', 'PHP'})

# Instructions that read A's low byte whatever their addressing mode.
_A_READERS = frozenset({
    'STA', 'ADC', 'SBC', 'AND', 'ORA', 'EOR', 'CMP', 'BIT', 'TRB', 'TSB',
    'TAX', 'TAY', 'TCD', 'TCS', 'PHA', 'XBA', 'MVN', 'MVP',
})
# Read-modify-write ops read A only in their accumulator form
# (ACCUMULATOR_OPCODES); `ASL $10` leaves A alone.
_RMW_MNEMONICS = frozenset({'ASL', 'LSR', 'ROL', 'ROR', 'INC', 'DEC'})
# Writers of A's low byte that do not read it first.
_A_WRITERS = frozenset({'LDA', 'TXA', 'TYA', 'PLA', 'TDC', 'TSC'})
# Readers of B whatever the m flag (16-bit transfers, x16 TAX/TAY,
# the block-move count).
_B_READERS_ANY_MODE = frozenset({'XBA', 'TAX', 'TAY', 'TCD', 'TCS', 'MVN', 'MVP'})
# Writers of B whatever the m flag.
_B_WRITERS_ANY_MODE = frozenset({'TDC', 'TSC'})

# Instructions that touch none of A, B and N/Z.
_ACC_FLAG_NEUTRAL = frozenset({
    'STX', 'STY', 'STZ', 'CLC', 'SEC', 'CLD', 'SED', 'CLI', 'SEI', 'CLV',
    'NOP', 'WDM', 'SEP', 'REP', 'XCE', 'TXS', 'PHX', 'PHY', 'PHB', 'PHD',
    'PHK', 'PEA', 'PEI', 'PER', 'BCC', 'BCS', 'BVC', 'BVS', 'BRA', 'BRL',
    'JMP',
})

_LIVENESS_BUDGET = 256

# Inline-asm mnemonics that make a RawAsm block opaque to `_live_after`.
_RAWASM_OPAQUE_MNEMONICS = frozenset({
    'BCC', 'BCS', 'BEQ', 'BNE', 'BMI', 'BPL', 'BVC', 'BVS', 'BRA', 'BRL',
    'JMP', 'JML', 'JSR', 'JSL', 'RTS', 'RTL', 'RTI', 'PHP', 'BRK', 'COP',
    'SEP', 'REP', 'PLP', 'XCE',
})
_KNOWN_MNEMONICS = frozenset(mnemonic(op) for op in Opcode)

_ALL_RESOURCES = frozenset({'A', 'B', 'NZ'})


class AccAbi(NamedTuple):
    """How a function's ABI uses the accumulator, for `_live_after`."""
    passes_b: bool     # a `@ B` parameter or a B return slot
    preserves_a: bool  # `#[preserves(A)]`: A and B survive a call to it


def _call_return_effect(instr, m16: Optional[bool],
                        abi: Optional[AccAbi]) -> Tuple[frozenset, frozenset]:
    """(reads, writes) of a call or return; `abi` is the callee's (for a
    call) or the enclosing function's (for a return), None if unknown."""
    reads = {'A'}
    if m16 is not False or abi is None or abi.passes_b:
        reads.add('B')
    writes = {'NZ'}
    if instr.opcode in CALL_OPCODES and abi is not None and not abi.preserves_a:
        writes.update(('A', 'B'))
    return frozenset(reads), frozenset(writes)


def _acc_flag_effect(instr, m16: Optional[bool]) -> Tuple[frozenset, frozenset]:
    """(reads, writes) of a non-call, non-return instr over
    {'A', 'B', 'NZ'}.

    `m16` is the accumulator width at instr: True, False, or None when
    the mode dataflow cannot say. A reads also read B unless m8 is
    known; LDA-class writes cover B only when m16 is known.
    """
    op = instr.opcode
    mn = mnemonic(op)

    reads: Set[str] = set()
    writes: Set[str] = set()
    if mn in _NZ_READERS:
        reads.add('NZ')
    if mn in _NZ_WRITERS and op != Opcode.BIT_IMMEDIATE:
        writes.add('NZ')
    a_read = op in ACCUMULATOR_OPCODES if mn in _RMW_MNEMONICS else mn in _A_READERS
    if a_read:
        reads.add('A')
        if m16 is not False:
            reads.add('B')
    if mn in _B_READERS_ANY_MODE:
        reads.add('B')
    if mn in _A_WRITERS:
        writes.add('A')
        if m16 is True or mn in _B_WRITERS_ANY_MODE:
            writes.add('B')

    if not (reads or writes) and mn not in _ACC_FLAG_NEUTRAL:
        return _ALL_RESOURCES, frozenset()
    return frozenset(reads), frozenset(writes)


def _rawasm_reads(node, m16: Optional[bool]) -> Optional[frozenset]:
    """The subset of {'A', 'B'} an inline-asm RawAsm may read, or None
    if it is not straight-line code `_live_after` can step over (see the
    section comment)."""
    reads: Set[str] = set()
    for line in node.text.splitlines():
        code = line.split(';', 1)[0].strip()
        if not code:
            continue
        if code.endswith(':') or code.startswith('.'):
            return None
        words = code.split()
        mn = words[0].upper().split('.')[0]
        if mn in _RAWASM_OPAQUE_MNEMONICS:
            return None
        if mn not in _KNOWN_MNEMONICS:
            reads.update(('A', 'B'))
            continue
        operand = words[1].upper() if len(words) > 1 else ''
        a_read = (operand in ('', 'A')) if mn in _RMW_MNEMONICS else mn in _A_READERS
        if a_read:
            reads.add('A')
            if m16 is not False:
                reads.add('B')
        if mn in _B_READERS_ANY_MODE:
            reads.add('B')
    return frozenset(reads)


def _live_after(nodes: List['AsmNode'], start: int, resources,
                labels, mode_at, abi_at=lambda k: None) -> bool:
    """True if any of `resources` may be read on a path from nodes[start].

    `labels` maps each label name in `nodes` to its index; `mode_at(k)`
    gives the accumulator width (8, 16 or None) at index k, and
    `abi_at(k)` the `AccAbi` of the call or return at k (None if
    unknown). A resource is dead on a path once something writes it
    without reading it first.
    """
    from r65.compiler.codegen.asm_nodes import Instruction, RawAsm

    worklist = [(start, frozenset(resources))]
    seen = set()
    budget = _LIVENESS_BUDGET
    while worklist:
        k, live = worklist.pop()
        while live and (k, live) not in seen:
            seen.add((k, live))
            if k >= len(nodes):
                break  # nothing executes past the last node
            node = nodes[k]
            if isinstance(node, RawAsm):
                mode = mode_at(k)
                reads = _rawasm_reads(node, None if mode is None else mode == 16)
                if reads is None or reads & live:
                    return True
                k += 1
                continue
            if not isinstance(node, Instruction):
                k += 1
                continue
            budget -= 1
            if budget < 0:
                return True
            mode = mode_at(k)
            m16 = None if mode is None else mode == 16
            if node.opcode in CALL_OPCODES or node.opcode in RETURN_OPCODES:
                reads, writes = _call_return_effect(node, m16, abi_at(k))
            else:
                reads, writes = _acc_flag_effect(node, m16)
            if reads & live:
                return True
            live = live - writes
            op = node.opcode
            if op in RETURN_OPCODES:
                break
            if op in BRANCH_OPCODES or op in JUMP_OPCODES:
                target = getattr(node.operand, 'value', None)
                if not isinstance(target, str) or target not in labels:
                    return True
                if op in (Opcode.BRA, Opcode.BRL) or op in JUMP_OPCODES:
                    k = labels[target]
                    continue
                worklist.append((labels[target], live))
            k += 1
    return False


# ============================================================================
# Peephole Optimizer
# ============================================================================

class PeepholeOptimizer:
    """
    Peephole optimizer that works directly on AsmNode objects.

    Uses typed Opcode enum for efficient pattern matching without string parsing.
    Applies multiple optimization passes to eliminate redundant operations.
    """

    def __init__(self, volatile_names: Set[str] = None, volatile_addresses: Set[int] = None):
        """
        Initialize peephole optimizer.

        Args:
            volatile_names: Set of variable names that are volatile (from #[hw] attributes).
                           Stores to these locations will not be eliminated.
            volatile_addresses: Set of hardware register addresses (from #[hw] attributes).
                               Stores to these addresses will not be eliminated.
        """
        self.stats = OptimizationStats()
        self.volatile_names = volatile_names or set()
        self.volatile_addresses = volatile_addresses or set()
        # Set while one region of a split stream is being optimized: the
        # mode masks the whole stream had at the region's first node, and
        # the other regions (for queries that must see the whole program).
        self._start_bits: Optional[Tuple[int, int]] = None
        self._outside: List[List['AsmNode']] = []
        # Seeds for the passes' mode dataflow; see optimize().
        self._entry_modes: dict = {}
        self._call_exit_modes: Optional[dict] = None
        self._acc_abi: Optional[dict] = None

    @property
    def optimizations_applied(self) -> int:
        """Total number of optimizations applied."""
        return self.stats.total

    def optimize(self, nodes: List['AsmNode'],
                 region_starts: Optional[List['AsmNode']] = None,
                 entry_modes: Optional[dict] = None,
                 call_exit_modes: Optional[dict] = None,
                 acc_abi: Optional[dict] = None) -> List['AsmNode']:
        """
        Apply peephole optimizations to AsmNode list.

        Args:
            nodes: List of AsmNode objects
            region_starts: Optional first nodes of independent regions
                (one per function). Each region is iterated to its own
                fixpoint, so a function that has converged is not
                re-scanned while another still changes.
            entry_modes: ``{label: mode}`` each function is entered in
                (None where unknown, e.g. an interrupt handler).
            call_exit_modes: ``{label: mode}`` each function returns in.
                Given both, the passes' mode dataflow starts every
                function at its entry mode and follows calls and inline
                asm; without them function bodies are unreachable to it
                and the mode-driven passes leave them alone.
            acc_abi: ``{label: AccAbi}`` for each function, used to
                tell which calls and returns read B and which calls
                clobber A. Calls and returns of unlisted functions are
                taken to read A and B and to clobber neither.

        Returns:
            Optimized node list
        """
        from r65.compiler.optimize.mode_directive_rewrite import (
            function_entry_seeds, normalize_mode_directives,
        )

        # Strip and re-emit mode directives once up-front. This drops
        # the mid-block `.ACCU` / `.INDEX` directives codegen pairs with
        # SEP/REP/PLP/RTI and reduces them to a clean stream where the
        # asm-mode dataflow is the sole source of truth. Passes need
        # not keep `.ACCU` in sync with SEP/REP; the final normalize
        # regenerates them after all optimizations settle.
        nodes = normalize_mode_directives(nodes, call_exit_modes, entry_modes)

        if call_exit_modes is not None:
            self._entry_modes = function_entry_seeds(nodes, entry_modes or {})
            self._call_exit_modes = call_exit_modes

        self._acc_abi = acc_abi
        regions = _split_regions(nodes, region_starts)
        if len(regions) == 1:
            nodes = self._optimize_region(nodes)
        else:
            # Seed each region with the modes the whole stream reaches its
            # first node in, so the passes' dataflow matches a whole-stream run.
            info = self._modes(nodes)
            seeds = []
            pos = 0
            for region in regions:
                seeds.append((info.incoming[pos], info.x_incoming[pos]))
                pos += len(region)
            for k, region in enumerate(regions):
                self._start_bits = seeds[k]
                self._outside = regions[:k] + regions[k + 1:]
                regions[k] = self._optimize_region(region)
            self._start_bits = None
            self._outside = []
            nodes = [node for region in regions for node in region]

        # Final pass: regenerate `.ACCU` / `.INDEX` directives from the
        # mode dataflow. The optimizations above may have left stale or
        # missing directives; this guarantees the final stream is
        # consistent without each pass having to maintain the pairing.
        nodes = normalize_mode_directives(nodes, call_exit_modes, entry_modes)

        return nodes

    def _optimize_region(self, nodes: List['AsmNode']) -> List['AsmNode']:
        """Apply the optimization passes to `nodes` until none changes it."""
        # Apply optimization passes until no more changes
        changed = True
        while changed:
            prev_total = self.stats.total
            # Invalidate the mode-dataflow cache used by
            # `_get_accu_mode_at`. Each pass below may produce a fresh
            # `nodes` list, so caches keyed on `id(nodes)` from the
            # previous iteration are stale and must not be reused.
            self._accu_mode_cache = None
            nodes = self._eliminate_redundant_load_after_store(nodes)
            nodes = self._eliminate_identity_copies(nodes)
            nodes = self._eliminate_redundant_loads_tracked(nodes)
            nodes = self._eliminate_dead_stores(nodes)
            nodes = self._eliminate_redundant_transfers(nodes)
            nodes = self._eliminate_redundant_stack_ops(nodes)
            nodes = self._eliminate_redundant_mode_changes(nodes)
            nodes = self._eliminate_dead_mode_changes(nodes)
            nodes = self._fold_carry_setup_into_rep(nodes)
            nodes = self._eliminate_cross_block_mode_changes(nodes)
            nodes = self._eliminate_redundant_and_before_sep(nodes)
            nodes = self._eliminate_branch_over_branch(nodes)
            nodes = self._thread_branches(nodes)
            nodes = self._eliminate_branch_to_next_label(nodes)
            nodes = self._inline_branch_to_return(nodes)
            nodes = self._hoist_loop_invariant_loads(nodes)
            nodes = self._hoist_loop_mode_switches(nodes)
            nodes = self._eliminate_unreachable_code(nodes)
            nodes = self._convert_zero_stores_to_stz(nodes)
            nodes = self._fold_inc_dec_accumulator(nodes)
            nodes = self._eliminate_redundant_cmp_zero(nodes)
            changed = self.stats.total > prev_total

        return nodes

    def _modes(self, nodes: List['AsmNode']):
        """The asm-mode dataflow over `nodes`, seeded as optimize() set up."""
        from r65.compiler.optimize.asm_mode_dataflow import compute_modes
        return compute_modes(nodes, entry_modes=self._entry_modes,
                             start_bits=self._start_bits,
                             call_exit_modes=self._call_exit_modes)

    def _rewrite_adjacent_pairs(self, nodes: List['AsmNode'], match) -> List['AsmNode']:
        """Scan for adjacent Instruction pairs and rewrite matches.

        match(n0, n1, i) returns the replacement node list for the pair at
        nodes[i], nodes[i + 1] — [n0] to keep only the first, [] to drop
        both — or None to leave n0 untouched.
        Non-Instruction nodes and a trailing instruction are passed through.
        The match callback is responsible for any stat bookkeeping.
        """
        from r65.compiler.codegen.asm_nodes import Instruction

        optimized = []
        i = 0
        while i < len(nodes):
            node = nodes[i]
            if (isinstance(node, Instruction) and i + 1 < len(nodes)
                    and isinstance(nodes[i + 1], Instruction)):
                replacement = match(node, nodes[i + 1], i)
                if replacement is not None:
                    optimized.extend(replacement)
                    i += 2
                    continue
            optimized.append(node)
            i += 1
        return optimized

    def _eliminate_redundant_load_after_store(self, nodes: List['AsmNode']) -> List['AsmNode']:
        """
        Eliminate redundant load immediately after store to same location.

        Patterns:
            STA $XX; LDA $XX -> STA $XX  (A still contains the value)
            STX $XX; LDX $XX -> STX $XX
            STY $XX; LDY $XX -> STY $XX

        The register is unchanged but N/Z need not be: the load sets them
        from the value and the store leaves whatever came before. The load
        goes when the instruction before the store already set N/Z from
        that register's value, or when N/Z are dead after the load. A
        `#[hw]` operand is never dropped — reading it back need not return
        what was written, and the read itself can be a side effect.
        """
        def match(node, next_node, i):
            if (node.opcode in STORE_TO_LOAD_MAP and
                    next_node.opcode in STORE_TO_LOAD_MAP[node.opcode] and
                    node.operand == next_node.operand and
                    self._same_addressing_mode(node.opcode, next_node.opcode) and
                    not self._is_hardware_register(node.operand) and
                    (self._nz_describes_register(nodes, i, mnemonic(node.opcode)[2])
                     or not self._acc_live_after(nodes, i + 2, ('NZ',)))):
                self.stats.redundant_loads_eliminated += 1
                return [node]  # keep store, drop redundant load
            return None

        return self._rewrite_adjacent_pairs(nodes, match)

    @staticmethod
    def _nz_describes_register(nodes: List['AsmNode'], i: int, reg: str) -> bool:
        """True if the instruction just before nodes[i] set N/Z from the
        value it left in `reg` ('A', 'X' or 'Y')."""
        from r65.compiler.codegen.asm_nodes import Comment, Directive, Instruction, ModeChange

        for k in range(i - 1, -1, -1):
            prev = nodes[k]
            if isinstance(prev, (Comment, Directive, ModeChange)):
                continue
            if not isinstance(prev, Instruction):
                return False
            return nz_register(prev.opcode) == reg
        return False

    def _same_addressing_mode(self, op1: Opcode, op2: Opcode) -> bool:
        """Check if two opcodes use the same addressing mode."""
        from r65.compiler.codegen.opcodes import addressing_mode
        return addressing_mode(op1) == addressing_mode(op2)

    def _eliminate_identity_copies(self, nodes: List['AsmNode']) -> List['AsmNode']:
        """
        Eliminate adjacent LDA/STA pairs where source and destination are identical.

        Patterns:
            LDA $NN; STA $NN -> (removed)  — DP identity copy
            LDA $NN,S; STA $NN,S -> (removed)  — Stack-relative identity copy

        The STA writes back what was just read, so it always goes (unless
        the operand is `#[hw]`, where both accesses are I/O). The LDA
        still sets A — B too in m16 — and N/Z for whatever follows, so
        it goes only when all of those are dead after the pair.
        """
        def match(node, next_node, i):
            if (node.opcode in LOAD_A_OPCODES and
                    next_node.opcode in STORE_A_OPCODES and
                    node.operand == next_node.operand and
                    self._same_addressing_mode(node.opcode, next_node.opcode) and
                    not self._is_hardware_register(node.operand)):
                self.stats.identity_copies_eliminated += 1
                live = ('A', 'NZ') if self._mode_info(nodes).unique_mode_at(i) == 8 \
                    else ('A', 'B', 'NZ')
                if self._acc_live_after(nodes, i + 2, live):
                    return [node]  # the load's result is still needed
                return []  # identity copy — drop both
            return None

        return self._rewrite_adjacent_pairs(nodes, match)

    def _eliminate_dead_stores(self, nodes: List['AsmNode']) -> List['AsmNode']:
        """
        Eliminate dead stores that are overwritten before being read.

        Pattern: STA $XX; ... (no read of $XX); STA $XX -> ... ; STA $XX
        """
        from r65.compiler.codegen.asm_nodes import Instruction

        optimized = []
        i = 0

        while i < len(nodes):
            node = nodes[i]

            if not isinstance(node, Instruction):
                optimized.append(node)
                i += 1
                continue

            # Only check STA (most common case)
            if node.opcode not in STORE_A_OPCODES:
                optimized.append(node)
                i += 1
                continue

            store_operand = node.operand

            # Skip if no operand or if using indexed addressing (can't track reliably)
            if store_operand is None or self._is_indexed_addressing(node.opcode):
                optimized.append(node)
                i += 1
                continue

            # Skip `#[hw]` locations (stores are I/O side effects)
            if self._is_hardware_register(store_operand):
                optimized.append(node)
                i += 1
                continue

            # Look ahead to see if there's another store to same address
            # before the value is read
            is_dead = self._is_dead_store(nodes, i, store_operand)

            if is_dead:
                i += 1
                self.stats.dead_stores_eliminated += 1
                continue

            optimized.append(node)
            i += 1

        return optimized

    def _is_indexed_addressing(self, opcode: Opcode) -> bool:
        """Check if opcode uses indexed addressing."""
        name = opcode.name
        return '_X' in name or '_Y' in name

    def _is_hardware_register(self, operand) -> bool:
        """
        Check if operand is a hardware register (volatile).

        Hardware register writes have side effects (I/O operations) and
        must never be eliminated, even if the value is overwritten later.

        Checks volatile register names and addresses from #[hw] attributes.
        """
        from r65.compiler.codegen.asm_nodes import Address

        if operand is None:
            return False

        # Extract the actual value from Address objects
        value = operand
        if isinstance(operand, Address):
            value = operand.value

        # Check if this is a known volatile register name (from #[hw] attribute)
        if isinstance(value, str):
            val = value.strip()
            if val in self.volatile_names:
                return True
            # Also check for hex address format against known volatile addresses
            if val.startswith('$'):
                try:
                    addr = int(val[1:], 16)
                    if addr in self.volatile_addresses:
                        return True
                except ValueError:
                    pass
        elif isinstance(value, int):
            # Check if address is in known volatile addresses
            if value in self.volatile_addresses:
                return True

        return False

    def _get_accu_mode_at(self, nodes: List['AsmNode'], idx: int) -> Optional[int]:
        """Determine accumulator width (8 or 16) at the given index.

        Backed by the asm-mode dataflow: returns the unique mode that
        all paths into nodes[idx] agree on, or None when the dataflow is
        inconclusive — the index is unreachable to it, a path arrives in
        unknown mode (post-PLP/RTI/call), or two paths disagree.

        Caches the dataflow result keyed on `id(nodes)`; callers in a
        rewrite-heavy pass (e.g. `_eliminate_dead_stores`) should
        invalidate `self._accu_mode_cache` whenever they hand in a
        freshly-allocated node list.
        """
        return self._mode_info(nodes).unique_mode_at(idx)

    def _mode_info(self, nodes: List['AsmNode']):
        """The seeded asm-mode dataflow for `nodes`, cached on its identity."""
        cache = getattr(self, '_accu_mode_cache', None)
        if cache is None or cache[0] is not nodes:
            self._accu_mode_cache = (nodes, self._modes(nodes))
        return self._accu_mode_cache[1]

    def _acc_live_after(self, nodes: List['AsmNode'], start: int, resources) -> bool:
        """True if any of `resources` ('A', 'B', 'NZ') may be read on a
        path from nodes[start]. See `_live_after`."""
        from r65.compiler.codegen.asm_nodes import Label

        cache = getattr(self, '_label_index_cache', None)
        if cache is None or cache[0] is not nodes:
            known = self._acc_abi or {}
            labels = {}
            owner = []  # the function each node belongs to, if known
            current = None
            for k, n in enumerate(nodes):
                if isinstance(n, Label):
                    labels[n.name] = k
                    if n.name in known:
                        current = n.name
                owner.append(current)
            self._label_index_cache = cache = (nodes, labels, owner)
        _, labels, owner = cache
        info = self._mode_info(nodes)
        return _live_after(nodes, start, resources, labels, info.unique_mode_at,
                           lambda k: self._abi_at(nodes[k], owner[k]))

    def _abi_at(self, instr: 'Instruction', owner: Optional[str]) -> Optional[AccAbi]:
        """The `AccAbi` of the call `instr` targets, or of `owner` (the
        function `instr` sits in) for a return; None if unknown."""
        if instr.opcode in RETURN_OPCODES:
            func = owner
        else:
            target = getattr(instr.operand, 'value', None)
            func = target if isinstance(target, str) else None
        return (self._acc_abi or {}).get(func)

    def _is_dead_store(self, nodes: List['AsmNode'], store_idx: int, store_operand) -> bool:
        """
        Check if a store is dead (overwritten before read).

        For stack-relative stores ($XX,S), extends analysis past unconditional
        branches (BRA/BRL): if no instruction in the entire node list reads
        from the stored address, the store is dead. This catches temporaries
        whose only reader was eliminated by a prior pass.

        In m16 mode, a 16-bit store to offset N also writes to N+1, so reads
        from N+1 must also prevent elimination.
        """
        from r65.compiler.codegen.asm_nodes import Instruction, Label, StackOffset, Address

        is_stack_relative = isinstance(store_operand, StackOffset)

        # In m16 mode, STA writes 2 bytes: offset N and N+1.
        # We need to check reads from the adjacent byte too - unless the
        # store is proven m8.
        adjacent_operand = None
        if is_stack_relative:
            accu_mode = self._get_accu_mode_at(nodes, store_idx)
            if accu_mode != 8:
                adjacent_operand = StackOffset(store_operand.offset + 1)
        elif isinstance(store_operand, Address) and isinstance(store_operand.value, int):
            accu_mode = self._get_accu_mode_at(nodes, store_idx)
            if accu_mode != 8:
                adjacent_operand = Address(store_operand.value + 1)

        j = store_idx + 1

        while j < len(nodes):
            next_node = nodes[j]

            # Label = potential branch target, stop analysis
            if isinstance(next_node, Label):
                return False

            if not isinstance(next_node, Instruction):
                j += 1
                continue

            # Control flow
            if next_node.opcode in CONTROL_FLOW_OPCODES:
                # For stack-relative stores, check if branch target reads value
                if is_stack_relative:
                    if next_node.opcode in (Opcode.BRA, Opcode.BRL):
                        if not self._any_instruction_reads(
                            nodes, store_operand, adjacent_operand):
                            return True
                return False

            # Stack-modifying instructions invalidate stack-relative analysis.
            # After PHA/PLX/etc., the same $XX,S offset refers to a different
            # physical address, so we can't compare operands across SP changes.
            if is_stack_relative and next_node.opcode in STACK_MODIFYING_OPCODES:
                return False

            # Mode change = stop analysis (16-bit mode can read adjacent bytes)
            # REP/SEP change how many bytes are accessed by subsequent instructions
            if next_node.opcode in (Opcode.REP_IMMEDIATE, Opcode.SEP_IMMEDIATE):
                return False

            # Indirect addressing = conservative, might read from any address
            # We can't track what address is being accessed through the pointer
            if next_node.opcode in INDIRECT_ADDRESSING_OPCODES:
                return False

            # Another store to same location = first store is dead
            if (next_node.opcode in STORE_A_OPCODES and
                next_node.operand == store_operand and
                not self._is_indexed_addressing(next_node.opcode)):
                # Mode switches end the scan, so the overwriting store has
                # this store's width and covers N+1 too. Reads of N+1 were
                # checked on earlier iterations.
                if adjacent_operand is None:
                    return True
                return True

            # Read from same location = store is not dead
            if self._reads_from_location(next_node, store_operand):
                return False

            # Read from adjacent byte (m16 overlap) = store is not dead
            if adjacent_operand is not None and self._reads_from_location(next_node, adjacent_operand):
                return False

            j += 1

        return False

    def _any_instruction_reads(self, nodes: List['AsmNode'],
                              store_operand,
                              adjacent_operand=None) -> bool:
        """
        Check if any instruction in the node list reads from store_operand
        (or adjacent_operand for m16 overlap).

        Conservatively scans all nodes - including the other regions when
        one region of a split stream is being optimized. This may miss
        optimization opportunities when multiple functions use the same
        stack offset, but is always safe (no false negatives).
        """
        from r65.compiler.codegen.asm_nodes import Instruction, StackOffset

        is_stack = isinstance(store_operand, StackOffset)

        for node in itertools.chain(nodes, *self._outside):
            if not isinstance(node, Instruction):
                continue
            if self._reads_from_location(node, store_operand):
                return True
            if adjacent_operand is not None and self._reads_from_location(node, adjacent_operand):
                return True
            # Indirect STA instructions (e.g. STA ($nn,S),Y) read the
            # pointer from their operand even though they are stores.
            # If the operand matches, the stored value IS being read.
            if (node.opcode in INDIRECT_ADDRESSING_OPCODES
                    and (node.operand == store_operand or
                         (adjacent_operand is not None and node.operand == adjacent_operand))):
                return True
            # Stack-modifying opcodes (PHB/PHA/PHX/PHY/PLB/PLA/PLX/PLY)
            # shift the stack pointer, so a store to $N,S may be read as
            # $(N+1),S after a push. We can't reliably match offsets across
            # SP changes, so conservatively assume the store is read.
            if is_stack and node.opcode in STACK_MODIFYING_OPCODES:
                return True

        return False

    def _reads_from_location(self, instr: 'Instruction', operand) -> bool:
        """Check if instruction reads from the given memory location.

        Also matches StackOffset(N) against Address(N) since they alias
        when D=S mode is active (far pointer functions use PHD/TSC/TCD
        to make stack offsets equivalent to direct page offsets).
        """
        if instr.opcode not in READS_FROM_MEMORY_OPCODES:
            return False
        if instr.operand == operand:
            return True
        # StackOffset(N) and Address(N) alias under D=S
        from r65.compiler.codegen.asm_nodes import StackOffset, Address
        if isinstance(operand, StackOffset) and isinstance(instr.operand, Address):
            return instr.operand.value == operand.offset
        if isinstance(operand, Address) and isinstance(instr.operand, StackOffset):
            return operand.value == instr.operand.offset
        return False

    def _eliminate_redundant_transfers(self, nodes: List['AsmNode']) -> List['AsmNode']:
        """
        Eliminate redundant register transfers.

        Patterns:
            TAX; TXA -> TAX  (A still has value, TXA is redundant)
            TAY; TYA -> TAY
            TXA; TAX -> TXA
            TYA; TAY -> TYA
            TXY; TYX -> TXY
            TYX; TXY -> TYX
        """
        # Map transfer to its reverse
        reverse_transfers = {
            Opcode.TAX: Opcode.TXA,
            Opcode.TAY: Opcode.TYA,
            Opcode.TXA: Opcode.TAX,
            Opcode.TYA: Opcode.TAY,
            Opcode.TXY: Opcode.TYX,
            Opcode.TYX: Opcode.TXY,
        }

        def match(node, next_node, _i):
            if (node.opcode in reverse_transfers and
                    next_node.opcode == reverse_transfers[node.opcode]):
                self.stats.redundant_transfers_eliminated += 1
                return [node]  # keep first, drop redundant reverse transfer
            return None

        return self._rewrite_adjacent_pairs(nodes, match)

    def _eliminate_redundant_stack_ops(self, nodes: List['AsmNode']) -> List['AsmNode']:
        """
        Eliminate redundant push/pull pairs with nothing between them.

        Pattern: PHA; PLA -> (nothing, if A isn't needed on stack)

        Note: This is conservative - only removes immediately adjacent pairs.
        """
        def match(node, next_node, _i):
            if (node.opcode in PUSH_PULL_PAIRS and
                    next_node.opcode == PUSH_PULL_PAIRS[node.opcode]):
                self.stats.redundant_stack_ops_eliminated += 1
                return []  # adjacent push/pull pair — remove both
            return None

        return self._rewrite_adjacent_pairs(nodes, match)

    def _eliminate_redundant_mode_changes(self, nodes: List['AsmNode']) -> List['AsmNode']:
        """Eliminate REP/SEP that don't change the actual CPU mode.

        Backed by the asm-mode dataflow: a SEP #$20 at index i is
        redundant iff every path into i already arrives in m8 (and
        symmetrically for REP). The dataflow's CFG-aware mode tracking
        catches non-adjacent redundancies (e.g. across label
        boundaries).

        Any trailing `.ACCU` directive codegen paired with the removed
        SEP/REP is left in the stream — the final
        ``normalize_mode_directives`` call rebuilds mid-block
        directives from the dataflow result, so this pass doesn't have
        to bookkeep the pairing.
        """
        from r65.compiler.codegen.asm_nodes import Instruction, Immediate, Directive

        info = self._modes(nodes)

        optimized: List = []
        i = 0
        n = len(nodes)
        while i < n:
            node = nodes[i]
            if not isinstance(node, Instruction):
                optimized.append(node)
                i += 1
                continue

            target_mode = None
            if (node.opcode == Opcode.SEP_IMMEDIATE
                    and isinstance(node.operand, Immediate)
                    and isinstance(node.operand.value, int)
                    and node.operand.value & M_FLAG):
                target_mode = 8
            elif (node.opcode == Opcode.REP_IMMEDIATE
                    and isinstance(node.operand, Immediate)
                    and isinstance(node.operand.value, int)
                    and node.operand.value & M_FLAG):
                target_mode = 16

            if target_mode is not None and info.unique_mode_at(i) == target_mode:
                # Already in the target mode on every path → SEP/REP is
                # a no-op. Drop it.
                self.stats.redundant_mode_changes_eliminated += 1
                optimized.extend(_mode_switch_residue(node))
                i += 1
                continue

            optimized.append(node)
            i += 1

        return optimized

    def _eliminate_dead_mode_changes(self, nodes: List['AsmNode']) -> List['AsmNode']:
        """Drop a SEP/REP whose every bit is rewritten by the next one.

        `_eliminate_redundant_mode_changes` asks "is the CPU already in
        the mode this switch asks for?" — a question about what comes
        *before*. This pass asks the dual question: does anything ever
        observe what this switch wrote? Codegen routinely emits a
        block-entry mode restore straight into the block's own mode
        requirement::

            SEP #$20  ; REQUIRED: restore m8 mode for block
            REP #$20  ; 16-bit A

        The SEP is a 3-cycle, 2-byte no-op: nothing between the two
        reads P, so every bit it set is immediately cleared again.

        A switch at `i` is dead iff the next node that isn't a comment,
        a `.ACCU`/`.INDEX` hint or a label nothing refers to overwrites
        every bit `i` wrote — another SEP/REP whose mask covers `i`'s, or
        a PLP/RTI, which reload the whole of P from the stack. Anything
        else in between — a real instruction, a label something branches
        to or names, inline asm — makes the write observable and stops
        the rewrite. A label nothing names is only ever fallen into::

            JSR div16
            SEP #$20  ; Restore m8 after u16 return
        game_add_score__L11:          ; no branch to it
            REP #$20
        """
        from r65.compiler.codegen.asm_nodes import (
            Instruction, Immediate, Comment, ModeChange, Label,
        )
        unreferenced = self._unreferenced_labels(nodes)

        ALL_FLAGS = 0xFF

        def written_mask(node) -> Optional[int]:
            """The P bits this node writes, or None if it isn't a SEP/REP.

            Only SEP/REP are ever *candidates* for removal. PLP and RTI
            also write P, but they pop it off the stack — dropping one
            would leave the stack a byte deep.
            """
            if (isinstance(node, Instruction)
                    and node.opcode in (Opcode.SEP_IMMEDIATE,
                                        Opcode.REP_IMMEDIATE)
                    and isinstance(node.operand, Immediate)
                    and isinstance(node.operand.value, int)):
                return node.operand.value
            return None

        def killed_mask(node) -> Optional[int]:
            """The P bits this node overwrites regardless of their old value."""
            if (isinstance(node, Instruction)
                    and node.opcode in (Opcode.PLP, Opcode.RTI)):
                # Both reload the whole of P from the stack, so they bury
                # any SEP/REP standing in front of them.
                return ALL_FLAGS
            return written_mask(node)

        n = len(nodes)
        dead: Set[int] = set()
        for i, node in enumerate(nodes):
            mask = written_mask(node)
            if mask is None:
                continue

            # Walk to the next node that can observe the flags.
            j = i + 1
            while j < n:
                nj = nodes[j]
                if isinstance(nj, (Comment, ModeChange)) or (
                        isinstance(nj, Label) and nj.name in unreferenced):
                    j += 1
                    continue
                break
            if j >= n:
                continue

            next_mask = killed_mask(nodes[j])
            # A SEP and a REP write opposite values, so covering the
            # mask is enough — whatever `i` wrote, `j` overwrites.
            if next_mask is not None and mask & ~next_mask == 0:
                dead.add(i)

        if not dead:
            return nodes

        self.stats.dead_mode_changes_eliminated += len(dead)
        return [node for i, node in enumerate(nodes) if i not in dead]

    def _unreferenced_labels(self, nodes: List['AsmNode']) -> Set[str]:
        """Labels in *nodes* that no instruction, directive or inline asm
        anywhere in the program names, and that are no function entry: they
        are only entered by falling into them."""
        from r65.compiler.codegen.asm_nodes import Directive, Instruction, Label, RawAsm
        from r65.compiler.optimize.mode_directive_rewrite import _IDENTIFIER_RE
        named: Set[str] = set(self._entry_modes or ())
        for region in [nodes, *self._outside]:
            for node in region:
                if isinstance(node, Instruction):
                    value = getattr(node.operand, 'value', None)
                    if isinstance(value, str):
                        named.update(_IDENTIFIER_RE.findall(value))
                elif isinstance(node, Directive):
                    for arg in getattr(node, 'args', ()) or ():
                        if isinstance(arg, str):
                            named.update(_IDENTIFIER_RE.findall(arg))
                elif isinstance(node, RawAsm):
                    named.update(_IDENTIFIER_RE.findall(node.text))
        return {n.name for n in nodes if isinstance(n, Label) and n.name not in named}

    def _fold_carry_setup_into_rep(self, nodes: List['AsmNode']) -> List['AsmNode']:
        """Fold the carry setup of a stack adjustment into its REP.

        Every frame allocation and teardown switches to m16 first, so
        there is always a REP sitting two instructions above the CLC or
        SEC::

            REP #$20    SEP -> m16          REP #$20
            TSC                             TSC
            CLC                             SEC
            ADC #$0A                        SBC #$0A
            TCS                             TCS

        REP takes a bitmask, and bit 0 is the carry flag — so `REP #$21`
        does the mode switch *and* clears carry in the same three cycles
        the mode switch already cost. The add form then needs no CLC at
        all. The subtract form wants carry set, which REP cannot do, but
        borrowing is just an off-by-one: with carry clear `SBC #n`
        subtracts n+1, so `SBC #(n-1)` lands on the same address.

        Two cycles and one byte per stack adjustment, at both ends of
        every function with a frame.

        Only a TSC is tolerated between the REP and the carry op —
        nothing else in these sequences, and it leaves carry alone. That
        keeps the rewrite obviously safe: no instruction in the window
        reads the carry we are now clearing early.
        """
        from r65.compiler.codegen.asm_nodes import (
            Instruction, Immediate, Comment, ModeChange,
        )

        C_FLAG = 0x01

        def skip_inert(i: int) -> int:
            while i < len(nodes) and isinstance(nodes[i], (Comment, ModeChange)):
                i += 1
            return i

        optimized: List = []
        drop: Set[int] = set()
        rewrite: dict = {}

        for i, node in enumerate(nodes):
            if (not isinstance(node, Instruction)
                    or node.opcode != Opcode.REP_IMMEDIATE
                    or not isinstance(node.operand, Immediate)
                    or not isinstance(node.operand.value, int)
                    or node.operand.value & C_FLAG):
                continue

            j = skip_inert(i + 1)
            if (j < len(nodes) and isinstance(nodes[j], Instruction)
                    and nodes[j].opcode == Opcode.TSC):
                j = skip_inert(j + 1)
            if j >= len(nodes) or not isinstance(nodes[j], Instruction):
                continue

            carry_op = nodes[j]
            if carry_op.opcode == Opcode.CLC:
                drop.add(j)
            elif carry_op.opcode == Opcode.SEC:
                k = skip_inert(j + 1)
                if (k >= len(nodes) or not isinstance(nodes[k], Instruction)
                        or nodes[k].opcode != Opcode.SBC_IMMEDIATE
                        or not isinstance(nodes[k].operand, Immediate)
                        or not isinstance(nodes[k].operand.value, int)
                        or nodes[k].operand.value < 1):
                    continue
                drop.add(j)
                borrowed = nodes[k]
                rewrite[k] = Instruction(
                    borrowed.opcode,
                    Immediate(borrowed.operand.value - 1),
                    borrowed.comment,
                    borrowed.source_loc,
                )
            else:
                continue

            rewrite[i] = Instruction(
                node.opcode,
                Immediate(node.operand.value | C_FLAG),
                (node.comment or '') + ' (carry cleared here)' if node.comment
                else 'Mode switch, carry cleared',
                node.source_loc,
                required=node.required,
            )

        if not drop:
            return nodes

        self.stats.carry_ops_folded_into_rep += len(drop)
        for i, node in enumerate(nodes):
            if i in drop:
                continue
            optimized.append(rewrite.get(i, node))
        return optimized

    def _eliminate_cross_block_mode_changes(self, nodes: List['AsmNode']) -> List['AsmNode']:
        """Use the asm-mode dataflow to:

        1. Eliminate SEP/REP at a label when every predecessor already
           arrives in the target mode (single-mode merge — the SEP/REP
           is a no-op).
        2. Insert a 'Mode fix' SEP/REP at labels that are real CFG
           join points (multiple physical predecessors) where the
           predecessors disagree on mode and there is no existing
           mode switch.

        The mode information is computed by `compute_modes()` over the
        full asm node list. SEP/REP/PLP/RTI are the only inputs that
        drive the dataflow; `.ACCU` directives are advisory metadata
        for WLA-DX and are not consulted for the dataflow, but a
        following `.ACCU` is taken as codegen's declared intent when
        choosing a mode fix's width.

        Insertions and eliminations are interleaved with re-computation
        so that a fix at one join properly clears the linearly-
        propagated 'mixed' state on downstream non-join labels.
        """
        from r65.compiler.codegen.asm_nodes import (
            Instruction, Immediate, Directive, Label, ModeChange,
        )
        from r65.compiler.optimize.asm_mode_dataflow import (
            first_constraining_mode_after,
        )

        # Loop until no more rewrites apply. Each rewrite changes the
        # node list, so we recompute the dataflow each pass. To keep
        # progress monotonic we insert at most one mode-fix per outer
        # iteration (eliminations are unbounded — they don't depend on
        # later decisions).
        any_change = True
        while any_change:
            any_change = False
            info = self._modes(nodes)
            new_nodes: List = []
            i = 0
            n = len(nodes)
            pending_fix_inserted = False
            while i < n:
                node = nodes[i]
                if not isinstance(node, Label):
                    new_nodes.append(node)
                    i += 1
                    continue

                # Find the next SEP/REP, walking past mode-hint
                # directives and co-located labels.
                j = i + 1
                sep_rep_idx = None
                while j < n:
                    nj = nodes[j]
                    if isinstance(nj, ModeChange) and nj.flag == 'ACCU':
                        j += 1
                        continue
                    if isinstance(nj, Label):
                        # Co-located labels share arriving modes; their
                        # SEP/REP (if any) belongs to the merged group.
                        j += 1
                        continue
                    if isinstance(nj, Instruction):
                        op = nj.opcode
                        if (op in (Opcode.SEP_IMMEDIATE, Opcode.REP_IMMEDIATE)
                                and isinstance(nj.operand, Immediate)
                                and isinstance(nj.operand.value, int)
                                and nj.operand.value & M_FLAG):
                            sep_rep_idx = j
                    break

                # ------------------------------------------------------
                # 1. Eliminate redundant SEP/REP at this label.
                # ------------------------------------------------------
                if sep_rep_idx is not None:
                    sep_rep = nodes[sep_rep_idx]
                    target_mode = 8 if sep_rep.opcode == Opcode.SEP_IMMEDIATE else 16
                    is_required = sep_rep.required
                    if not is_required:
                        arriving = info.incoming_at(sep_rep_idx)
                        defined = {m for m in arriving if m is not None}
                        all_match = (
                            defined == {target_mode} and None not in arriving
                        )
                        if all_match:
                            # The label, co-located labels (still branch
                            # targets) and their hints.
                            new_nodes.extend(nodes[i:sep_rep_idx])
                            # Skip the SEP/REP itself, keeping any flags
                            # it set besides M. Any trailing
                            # `.ACCU` directive codegen paired with it
                            # is left untouched — the final
                            # `normalize_mode_directives` call rebuilds
                            # mid-block directives from the dataflow.
                            new_nodes.extend(_mode_switch_residue(sep_rep))
                            i = sep_rep_idx + 1
                            self.stats.redundant_mode_changes_eliminated += 1
                            any_change = True
                            continue

                # ------------------------------------------------------
                # 2. Insert a Mode fix SEP/REP at the FIRST true join
                #    where predecessors disagree on mode and there is
                #    no existing switch.
                #
                #    We insert at most one fix per outer iteration so
                #    the next iteration's dataflow re-computation can
                #    update the modes of every label downstream — most
                #    "mixed" labels are mixed only because m8 from a
                #    back-edge propagates through them, and a single
                #    fix at the dominating join cleans up the entire
                #    chain. Labels with only fall-through predecessors
                #    are never fixed directly: they inherit the mode
                #    of whatever upstream join already covered them.
                # ------------------------------------------------------
                if (sep_rep_idx is None
                        and not pending_fix_inserted
                        and info.is_join(i)
                        and info.has_mixed_known(i)):
                    expected_mode = first_constraining_mode_after(
                        nodes, info, i + 1
                    )
                    if expected_mode is None:
                        # No m-determining op nearby; default to m8.
                        expected_mode = 8

                    opcode = (Opcode.SEP_IMMEDIATE
                              if expected_mode == 8
                              else Opcode.REP_IMMEDIATE)
                    fix = Instruction(
                        opcode,
                        Immediate(M_FLAG),
                        "Mode fix: predecessors disagree on A size",
                    )

                    new_nodes.append(node)  # the label
                    i += 1
                    while i < n and isinstance(nodes[i], Directive):
                        new_nodes.append(nodes[i])
                        i += 1
                    new_nodes.append(fix)
                    any_change = True
                    pending_fix_inserted = True
                    continue

                new_nodes.append(node)
                i += 1

            nodes = new_nodes

        return nodes

    def _eliminate_redundant_and_before_sep(self, nodes: List['AsmNode']) -> List['AsmNode']:
        """
        Remove AND #$00FF when followed by SEP #$20.

        The mask leaves A's low byte alone, but it also zeroes B and sets
        N/Z, and m8 code still reaches B: TAX/TAY copy all 16 bits into
        the x16 index registers, and XBA, TCD and any A use after a REP
        read it. So the AND goes only when B (if the AND ran in m16) and
        N/Z are both dead after the SEP.

        Pattern: AND #$00FF (or #$FF), SEP #$xx where xx & 0x20 != 0
        """
        from r65.compiler.codegen.asm_nodes import Instruction, Directive, Immediate

        optimized = []
        i = 0

        while i < len(nodes):
            node = nodes[i]

            # Check for AND #$00FF or AND #$FF (immediate)
            if isinstance(node, Instruction) and node.opcode == Opcode.AND_IMMEDIATE:
                if isinstance(node.operand, Immediate) and node.operand.value in (0x00FF, 0xFF):
                    # Look ahead for SEP with M flag, skipping directives
                    next_instr_idx = i + 1
                    directives_between = []

                    while next_instr_idx < len(nodes):
                        next_node = nodes[next_instr_idx]
                        if isinstance(next_node, Directive):
                            directives_between.append(next_node)
                            next_instr_idx += 1
                        elif isinstance(next_node, Instruction):
                            break
                        else:
                            break

                    if next_instr_idx < len(nodes):
                        next_instr = nodes[next_instr_idx]

                        if (isinstance(next_instr, Instruction) and
                            next_instr.opcode == Opcode.SEP_IMMEDIATE and
                            isinstance(next_instr.operand, Immediate) and
                            next_instr.operand.value & M_FLAG and
                            not self._acc_live_after(
                                nodes, next_instr_idx + 1,
                                ('NZ',) if self._mode_info(nodes).unique_mode_at(i) == 8
                                else ('B', 'NZ'))):
                            # Pattern matched - skip the AND, keep directives and SEP
                            optimized.extend(directives_between)
                            i = next_instr_idx  # Will emit SEP on next iteration
                            self.stats.redundant_and_before_sep_eliminated += 1
                            continue

            optimized.append(node)
            i += 1

        return optimized

    def _eliminate_branch_over_branch(self, nodes: List['AsmNode']) -> List['AsmNode']:
        """
        Eliminate conditional branch over unconditional branch when the
        conditional target is the immediately following label.

        Pattern:
            Bcc label_A       ; conditional branch
            [labels/directives/comments]
            BRA label_B       ; unconditional branch
            [labels/directives/comments]
            label_A:          ; conditional target is right here

        Becomes:
            B!cc label_B      ; inverted condition, target BRA's destination
            [labels/directives/comments preserved]
            label_A:          ; kept (may be targeted by other branches)

        Labels between Bcc and BRA are safe to skip because they contain
        no instructions — any branch targeting them reaches the same BRA.
        """
        from r65.compiler.codegen.asm_nodes import Instruction, Label, Address
        from r65.compiler.codegen.asm_nodes import invert_branch

        CONDITIONAL_BRANCHES = BRANCH_OPCODES - {Opcode.BRA, Opcode.BRL}

        optimized = []
        i = 0

        while i < len(nodes):
            node = nodes[i]

            if (isinstance(node, Instruction) and
                node.opcode in CONDITIONAL_BRANCHES and
                isinstance(node.operand, Address) and
                isinstance(node.operand.value, str)):

                cond_target = node.operand.value

                # Look ahead for BRA, skipping directives/comments AND labels.
                # Labels with no instructions between them and the BRA are
                # transparent — branches to them reach the same BRA destination.
                j = i + 1
                between = []
                while j < len(nodes) and not isinstance(nodes[j], Instruction):
                    between.append(nodes[j])
                    j += 1

                if (j < len(nodes) and
                    isinstance(nodes[j], Instruction) and
                    nodes[j].opcode == Opcode.BRA and
                    isinstance(nodes[j].operand, Address)):

                    bra_node = nodes[j]
                    bra_target = bra_node.operand

                    # Self-loop guard: if the BRA targets a label that appears
                    # in `between` (i.e. `label: BRA label`), the BRA is its
                    # own body — e.g. `#[entry]` / `-> !` halt loops. Dropping
                    # the BRA would leave the label naked and let control fall
                    # through into whatever follows. Leave this BRA alone.
                    bra_target_name = (bra_target.value
                                       if isinstance(bra_target.value, str)
                                       else None)
                    if bra_target_name is not None and any(
                        isinstance(b, Label) and b.name == bra_target_name
                        for b in between
                    ):
                        optimized.append(node)
                        i += 1
                        continue

                    # Look ahead past the BRA for the conditional target label,
                    # skipping directives/comments and non-target labels
                    k = j + 1
                    between2 = []
                    found_target = False
                    while k < len(nodes):
                        if isinstance(nodes[k], Instruction):
                            break  # Hit an instruction before finding target
                        if isinstance(nodes[k], Label) and nodes[k].name == cond_target:
                            found_target = True
                            break
                        between2.append(nodes[k])
                        k += 1

                    if found_target:

                        inverted = invert_branch(node.opcode)
                        if inverted is not None:
                            # Emit inverted branch to BRA's target
                            optimized.append(Instruction(
                                inverted, bra_target, bra_node.comment,
                                node.source_loc))
                            # Keep labels/directives/comments that were between
                            optimized.extend(between)
                            optimized.extend(between2)
                            # Label will be appended on next iteration
                            i = k
                            self.stats.branch_over_branch_eliminated += 1
                            continue

            optimized.append(node)
            i += 1

        return optimized

    def _thread_branches(self, nodes: List['AsmNode']) -> List['AsmNode']:
        """
        Thread branches through intermediate unconditional branches.

        When a branch (conditional or BRA) targets a label whose first
        instruction is BRA, redirect it to the BRA's ultimate target.

        Example:
            BPL label_A       ->  BPL label_B
            ...                   ...
            label_A:              label_A:
            BRA label_B           BRA label_B  (now dead, removed by later passes)
        """
        from r65.compiler.codegen.asm_nodes import Instruction, Label, Address, RawAsm

        ALL_BRANCHES = BRANCH_OPCODES

        # Build label -> BRA target map: for each label immediately followed
        # by a BRA (skipping other labels/directives), record the BRA target.
        # RawAsm (inline asm, data) emits bytes and stops the scan; ModeChange
        # nodes and comments are not instructions and are skipped.
        label_to_bra_target: dict[str, str] = {}
        for i, node in enumerate(nodes):
            if isinstance(node, Label):
                # Find the first real instruction after this label
                j = i + 1
                while j < len(nodes):
                    nj = nodes[j]
                    if isinstance(nj, Instruction):
                        break
                    if isinstance(nj, RawAsm):
                        break
                    j += 1
                if (j < len(nodes) and
                    isinstance(nodes[j], Instruction) and
                    nodes[j].opcode == Opcode.BRA and
                    isinstance(nodes[j].operand, Address) and
                    isinstance(nodes[j].operand.value, str)):
                    label_to_bra_target[node.name] = nodes[j].operand.value

        # Follow chains (label_A -> BRA label_B -> BRA label_C)
        def resolve(label: str, depth: int = 0) -> str:
            if depth > 10:
                return label
            if label in label_to_bra_target:
                return resolve(label_to_bra_target[label], depth + 1)
            return label

        # Rewrite branch targets
        optimized = []
        for node in nodes:
            if (isinstance(node, Instruction) and
                node.opcode in ALL_BRANCHES and
                isinstance(node.operand, Address) and
                isinstance(node.operand.value, str)):

                target = node.operand.value
                resolved = resolve(target)
                if resolved != target:
                    optimized.append(Instruction(
                        node.opcode,
                        Address(resolved),
                        node.comment,
                        node.source_loc,
                    ))
                    self.stats.branch_threading_applied += 1
                    continue

            optimized.append(node)

        return optimized

    def _hoist_loop_invariant_loads(self, nodes: List['AsmNode']) -> List['AsmNode']:
        """
        Hoist loop-invariant LDA #imm out of bottom-tested loops.

        Pattern: a bottom-tested loop (body ending with BCC/BCS/BNE/BEQ to
        body label) where LDA #imm appears in the body and A is not modified
        by any other instruction in the loop. The LDA is moved before the
        loop header label.

        Only applies when:
        - Exactly one LDA in the loop body (the invariant load)
        - No other instruction in the loop modifies A - a call counts (the
          callee's return value comes back in A, and it may use A anyway),
          as do TSC/TDC and any inline asm
        - The LDA is LDA_IMMEDIATE
        - The loop never changes the accumulator's width (REP/SEP/PLP/XCE):
          the immediate's size is the width where it was, and the hoisted
          copy would run in whatever width the loop is entered in
        """
        from r65.compiler.codegen.asm_nodes import Instruction, Label, Address, RawAsm

        # Find bottom-tested loops: label -> body -> BCC/BCS/BNE/BEQ label
        # Build label position map
        label_positions: dict[str, int] = {}
        for i, node in enumerate(nodes):
            if isinstance(node, Label):
                label_positions[node.name] = i

        # Find back-edges (conditional branches that jump backward to a label)
        loops = []  # (label_idx, branch_idx, label_name)
        for i, node in enumerate(nodes):
            if (isinstance(node, Instruction) and
                    node.opcode in BRANCH_OPCODES and
                    node.opcode != Opcode.BRA and
                    isinstance(getattr(node, 'operand', None), Address) and
                    isinstance(node.operand.value, str)):
                target = node.operand.value
                if target in label_positions and label_positions[target] < i:
                    loops.append((label_positions[target], i, target))

        if not loops:
            return nodes

        # Process each loop
        hoists = {}  # label_idx -> (lda_instruction, lda_body_offset)
        for label_idx, branch_idx, label_name in loops:
            # Collect body instructions (between label and branch, inclusive of branch)
            body_instrs = []
            lda_positions = []
            other_a_mods = False

            for j in range(label_idx + 1, branch_idx + 1):
                n = nodes[j]
                if isinstance(n, RawAsm):
                    other_a_mods = True        # inline asm: assume it touches A
                    continue
                if not isinstance(n, Instruction):
                    continue
                body_instrs.append((j, n))
                if n.opcode == Opcode.LDA_IMMEDIATE:
                    lda_positions.append(j)
                elif n.opcode in MODIFIES_A_OPCODES or n.opcode in _HOIST_CLOBBERS_A:
                    other_a_mods = True
                elif n.opcode in _HOIST_CHANGES_WIDTH:
                    other_a_mods = True        # the immediate's width would not hold

            # Only hoist if exactly one LDA #imm and nothing else modifies A
            if len(lda_positions) == 1 and not other_a_mods:
                # Don't hoist if the instruction before the label is an
                # unconditional branch — inserted code would be unreachable.
                # This happens in bottom-tested loops where the initial entry
                # is BRA label, skipping any code placed before the label.
                prev_idx = label_idx - 1
                while prev_idx >= 0 and not isinstance(nodes[prev_idx], Instruction):
                    prev_idx -= 1
                if prev_idx >= 0 and isinstance(nodes[prev_idx], Instruction):
                    prev_op = nodes[prev_idx].opcode
                    if prev_op in (Opcode.BRA, Opcode.BRL, Opcode.JMP_ABSOLUTE, Opcode.JMP_LONG):
                        continue
                hoists[lda_positions[0]] = label_idx

        if not hoists:
            return nodes

        # Move each hoisted LDA to just before its loop's label.
        optimized = []
        insert_before: dict[int, list] = {}  # label_idx -> [instructions to insert]
        for lda_idx, label_idx in hoists.items():
            if label_idx not in insert_before:
                insert_before[label_idx] = []
            insert_before[label_idx].append(nodes[lda_idx])

        for i, node in enumerate(nodes):
            if i in insert_before:
                optimized.extend(insert_before[i])
            if i in hoists:
                self.stats.loop_invariant_loads_hoisted += 1
                continue
            optimized.append(node)

        return optimized

    def _hoist_loop_mode_switches(self, nodes: List['AsmNode']) -> List['AsmNode']:
        """
        Hoist SEP/REP from inside bottom-tested loops to before the loop header.

        Pattern: Label / SEP|REP / body / BCC|BNE Label
        When the back-edge arrives in the target mode already (body doesn't change
        it back), hoisting the SEP/REP before the label makes it execute once
        instead of every iteration. The in-loop copy is removed here, and the
        label's anchored `.ACCU` is retagged.

        ``.ACCU`` / ``.INDEX`` directives that codegen paired with the
        SEP/REP are not tracked here — they're regenerated from the
        asm-mode dataflow by ``normalize_mode_directives`` at the end
        of ``optimize()``, so any hoisted SEP/REP automatically gets a
        fresh, correctly-positioned directive in the final output.
        """
        from r65.compiler.codegen.asm_nodes import (
            Instruction, Immediate, Directive, Label, ModeChange, Address,
            BlankLine, Comment,
        )

        BACK_EDGE_OPCODES = {Opcode.BCC, Opcode.BNE, Opcode.BRA, Opcode.BCS, Opcode.BEQ,
                             Opcode.BPL, Opcode.BMI}

        # Build label reference counts
        label_refs: dict[str, int] = {}
        for node in nodes:
            if isinstance(node, Instruction) and node.opcode in (BRANCH_OPCODES | JUMP_OPCODES):
                if hasattr(node.operand, 'value') and isinstance(node.operand.value, str):
                    label_refs[node.operand.value] = label_refs.get(node.operand.value, 0) + 1

        # Find candidates: Label followed by SEP/REP (possibly with `.ACCU`
        # / `.INDEX` directives between — those are inert hints we walk
        # past to find the real instruction).
        hoists = []  # (label_idx, sep_idx, target_mode)
        info = None

        for i, node in enumerate(nodes):
            if not isinstance(node, Label):
                continue

            label_name = node.name
            # Only handle labels with exactly 1 reference (the back-edge)
            if label_refs.get(label_name, 0) != 1:
                continue

            # Look for SEP/REP after label, skipping the inert nodes that
            # can sit between the two — mode-hint directives, comments,
            # blank lines. Same skip set as the anchor retag below, so a
            # comment can't make one walk find the switch and the other
            # miss it.
            j = i + 1
            sep_idx = None
            while j < len(nodes):
                if isinstance(nodes[j], (ModeChange, Comment, BlankLine)):
                    j += 1
                    continue
                # Only a bare M switch moves: one that also sets carry
                # or X would set it once instead of every iteration.
                if isinstance(nodes[j], Instruction):
                    if (nodes[j].opcode in (Opcode.SEP_IMMEDIATE,
                                            Opcode.REP_IMMEDIATE) and
                            isinstance(nodes[j].operand, Immediate) and
                            nodes[j].operand.value == M_FLAG):
                        sep_idx = j
                break

            if sep_idx is None:
                continue

            target_mode = 8 if nodes[sep_idx].opcode == Opcode.SEP_IMMEDIATE else 16

            # Track mode through the loop body to verify the back-edge
            # arrives in `target_mode`. SEP/REP/PLP/RTI are the only
            # things that change runtime mode — we deliberately do not
            # consult `.ACCU` directives, which are advisory metadata
            # for WLA-DX, not the runtime CPU.
            body_start = sep_idx + 1
            current_mode = target_mode  # After the SEP/REP, we're in target_mode
            found_back_edge = False

            for k in range(body_start, len(nodes)):
                n = nodes[k]
                if not isinstance(n, Instruction):
                    continue

                # Track mode
                if n.opcode == Opcode.REP_IMMEDIATE and isinstance(n.operand, Immediate):
                    if isinstance(n.operand.value, int) and n.operand.value & M_FLAG:
                        current_mode = 16
                elif n.opcode == Opcode.SEP_IMMEDIATE and isinstance(n.operand, Immediate):
                    if isinstance(n.operand.value, int) and n.operand.value & M_FLAG:
                        current_mode = 8
                elif n.opcode in (Opcode.PLP, Opcode.RTI):
                    current_mode = None
                    break

                # Check for back-edge. The walk above follows text order,
                # which can run past a return into a block entered from
                # elsewhere in another mode, so the dataflow - every path
                # into the branch - must agree.
                if (n.opcode in BACK_EDGE_OPCODES and
                        hasattr(n.operand, 'value') and n.operand.value == label_name):
                    if current_mode == target_mode:
                        if info is None:
                            info = self._modes(nodes)
                        found_back_edge = info.unique_mode_at(k) == target_mode
                    break

                # If we hit an unconditional branch/jump to elsewhere, stop
                if n.opcode == Opcode.BRA or n.opcode in JUMP_OPCODES:
                    break

            if found_back_edge:
                hoists.append((i, sep_idx, target_mode))

        if not hoists:
            return nodes

        # Apply hoists: insert SEP/REP before label, remove from inside loop.
        hoist_set = {h[1] for h in hoists}  # SEP/REP indices to remove
        insert_before = {h[0]: nodes[h[1]] for h in hoists}

        # A label's first `.ACCU` is *anchored*: `normalize_mode_directives`
        # preserves it and seeds the dataflow with it, because codegen knows
        # things about block entry the asm-level dataflow can't see. Hoisting
        # the switch above the label changes the mode the label is entered in,
        # so the anchor has to move with it — otherwise the stale seed makes
        # WLA-DX size the block's immediates for the wrong width.
        # Walk to the anchor exactly the way `_classify_directives` does
        # when it decides what counts as anchored — same skip set, or the
        # two disagree about which directive is the seed and this retag
        # silently misses it.
        retag: dict[int, int] = {}
        for label_idx, _sep_idx, target_mode in hoists:
            j = label_idx + 1
            while j < len(nodes):
                nj = nodes[j]
                if isinstance(nj, (Comment, BlankLine)):
                    j += 1
                    continue
                if isinstance(nj, ModeChange):
                    if nj.flag == 'ACCU':
                        retag[j] = target_mode
                        break
                    j += 1
                    continue
                break

        optimized = []
        for i, node in enumerate(nodes):
            if i in insert_before:
                optimized.append(insert_before[i])
                self.stats.redundant_mode_changes_eliminated += 1
            if i in hoist_set:
                continue
            if i in retag:
                node = ModeChange('ACCU', retag[i], node.source_loc)
            optimized.append(node)

        return optimized

    def _eliminate_unreachable_code(self, nodes: List['AsmNode']) -> List['AsmNode']:
        """
        Eliminate unreachable code after unconditional control flow (RTS, RTI, RTL, BRA, JMP).

        Drops Instructions after an unconditional transfer until the next Label or
        non-instruction node, both of which are kept.
        """
        from r65.compiler.codegen.asm_nodes import Instruction, Label, Directive

        TERMINAL_OPCODES = ({Opcode.RTS, Opcode.RTI, Opcode.RTL, Opcode.BRA} |
                            JUMP_OPCODES)

        optimized = []
        skipping = False
        eliminated = 0

        for node in nodes:
            if skipping:
                if isinstance(node, Label):
                    # Stop skipping at any label — it may be referenced by
                    # expressions (e.g., LDA #>(__SCMP1 - 1)) not caught by
                    # simple operand scanning
                    skipping = False
                    optimized.append(node)
                    continue
                # Only skip Instructions — preserve Directives and Comments
                if isinstance(node, Instruction):
                    eliminated += 1
                    continue
                # Non-instruction node (Directive/Comment) — stop skipping
                skipping = False

            optimized.append(node)

            if isinstance(node, Instruction) and node.opcode in TERMINAL_OPCODES:
                skipping = True

        self.stats.unreachable_nodes_eliminated += eliminated
        return optimized

    def _eliminate_branch_to_next_label(self, nodes: List['AsmNode']) -> List['AsmNode']:
        """
        Eliminate BRA instructions that branch to the immediately following label.

        Pattern: BRA label; label: -> label:
        Skips mode directives, comments and other labels between the BRA and
        the label. RawAsm (inline asm, data directives) emits bytes the CPU
        would then execute, so it keeps the BRA.
        """
        from r65.compiler.codegen.asm_nodes import Instruction, Label, Address, RawAsm

        optimized = []
        i = 0

        while i < len(nodes):
            node = nodes[i]

            if (isinstance(node, Instruction) and
                node.opcode == Opcode.BRA and
                isinstance(node.operand, Address) and
                isinstance(node.operand.value, str)):

                target = node.operand.value

                # Look ahead for the target label. Labels between the BRA
                # and target contain no instructions, so falling through
                # them is equivalent to branching to the target.
                j = i + 1
                while j < len(nodes) and not isinstance(nodes[j], Instruction):
                    if isinstance(nodes[j], Label) and nodes[j].name == target:
                        break
                    if isinstance(nodes[j], RawAsm):
                        break
                    j += 1

                if (j < len(nodes) and
                    isinstance(nodes[j], Label) and
                    nodes[j].name == target):
                    # BRA to next label — skip the BRA, keep directives/comments between
                    for k in range(i + 1, j):
                        optimized.append(nodes[k])
                    # Label will be appended on next iteration
                    i = j
                    self.stats.branch_to_next_eliminated += 1
                    continue

            optimized.append(node)
            i += 1

        return optimized

    def _inline_branch_to_return(self, nodes: List['AsmNode']) -> List['AsmNode']:
        """
        Replace BRA to a return instruction with the return instruction itself.

        Pattern: BRA label / ... / label: RTS → RTS (inline the return)
        Saves the BRA (2 bytes and 3 cycles) and allows subsequent dead code
        elimination of the now-unreferenced label.
        """
        from r65.compiler.codegen.asm_nodes import Instruction, Label, RawAsm

        # Build label -> return opcode map
        label_to_return: dict[str, Opcode] = {}
        for i, node in enumerate(nodes):
            if isinstance(node, Label):
                # Find first instruction after label. Stop at RawAsm (inline
                # asm, data directives: it emits bytes) but skip
                # non-instruction nodes (ModeChange, comments).
                j = i + 1
                while j < len(nodes):
                    nj = nodes[j]
                    if isinstance(nj, Instruction):
                        break
                    if isinstance(nj, RawAsm):
                        break
                    j += 1
                if (j < len(nodes) and isinstance(nodes[j], Instruction) and
                        nodes[j].opcode in RETURN_OPCODES):
                    label_to_return[node.name] = nodes[j].opcode

        if not label_to_return:
            return nodes

        optimized = []
        for node in nodes:
            if (isinstance(node, Instruction) and
                    node.opcode == Opcode.BRA and
                    hasattr(node.operand, 'value') and
                    isinstance(node.operand.value, str) and
                    node.operand.value in label_to_return):
                # Replace BRA with the return instruction
                ret_opcode = label_to_return[node.operand.value]
                optimized.append(Instruction(ret_opcode, None, node.comment, node.source_loc))
                self.stats.branch_to_next_eliminated += 1
                continue
            optimized.append(node)

        return optimized

    def _eliminate_redundant_loads_tracked(self, nodes: List['AsmNode']) -> List['AsmNode']:
        """
        Eliminate redundant LDA instructions by tracking what A currently holds.

        Tracks the last LDA as (opcode, operand). When a subsequent LDA has
        the same opcode and operand, it's eliminated since A already holds
        that value — provided N/Z are dead after it, since an intervening
        CMP, BIT or TAX leaves the tracking valid but rewrites the flags
        the load would have set.

        Only tracks deterministic addressing modes (immediate, DP, absolute,
        stack-relative) — indexed and indirect loads depend on register values
        that may change between loads.

        State is cleared on: labels, inline asm blocks, mode changes,
        A-modifying instructions (except trackable LDA which updates tracking),
        stack pointer changes when tracking a stack-relative address, stores to
        the tracked address by other registers (STX/STY), and indirect stores.
        """
        from r65.compiler.codegen.asm_nodes import Instruction, Label, RawAsm, StackOffset

        optimized = []
        # Track last LDA: (opcode, operand) or None
        known_a = None

        for idx, node in enumerate(nodes):
            if not isinstance(node, Instruction):
                if isinstance(node, (Label, RawAsm)):
                    # Label = unknown incoming state
                    # RawAsm = inline asm!() may modify any register
                    known_a = None
                optimized.append(node)
                continue

            opcode = node.opcode

            # Check for redundant LDA before any state updates
            if opcode in LOAD_A_OPCODES:
                if (known_a is not None and
                    known_a[0] == opcode and
                    known_a[1] == node.operand and
                    not self._acc_live_after(nodes, idx + 1, ('NZ',))):
                    # A already holds this value — skip the LDA
                    self.stats.tracked_loads_eliminated += 1
                    continue
                # Update tracking only for deterministic addressing modes
                # Never track loads from hardware I/O registers (volatile)
                if opcode in TRACKABLE_LDA_OPCODES and not self._is_hardware_register(node.operand):
                    known_a = (opcode, node.operand)
                else:
                    # Non-deterministic load (indexed/indirect) — A changed
                    # but we can't track the value
                    known_a = None
                optimized.append(node)
                continue

            # Mode changes (REP/SEP affecting M flag) change A width
            if opcode in (Opcode.REP_IMMEDIATE, Opcode.SEP_IMMEDIATE):
                from r65.compiler.codegen.asm_nodes import Immediate
                if isinstance(node.operand, Immediate):
                    val = node.operand.value
                    if isinstance(val, int) and val & 0x20:
                        known_a = None
                optimized.append(node)
                continue

            # Instructions that modify A (but aren't LDA — those are handled above)
            if opcode in MODIFIES_A_OPCODES:
                known_a = None
                optimized.append(node)
                continue

            # Stack pointer modifications invalidate stack-relative tracking
            if opcode in STACK_MODIFYING_OPCODES:
                if known_a is not None and isinstance(known_a[1], StackOffset):
                    known_a = None
                optimized.append(node)
                continue

            # Any other write to the tracked address invalidates tracking
            # (STA to same address is fine — writes A back). An indexed
            # write may land on it.
            if opcode in MEMORY_WRITE_OPCODES and opcode not in STORE_A_OPCODES:
                if known_a is not None and (
                        node.operand == known_a[1]
                        or opcode.name.endswith(('_X', '_Y'))
                        or (known_a[0] != Opcode.LDA_IMMEDIATE
                            and self._is_hardware_register(node.operand))):
                    known_a = None
                optimized.append(node)
                continue

            # Indirect stores can alias any address
            if opcode in INDIRECT_STORE_OPCODES:
                known_a = None
                optimized.append(node)
                continue

            # Control flow (branches/jumps/calls) — clear tracking
            # Calls: callee may modify memory that A was loaded from
            # Branches/jumps: target may have different A state
            if opcode in CONTROL_FLOW_OPCODES:
                known_a = None
                optimized.append(node)
                continue

            # Stores to hw I/O registers may have side effects that change
            # the value at the tracked address (e.g., STA WRMPYB triggers
            # a multiply, changing RDMPYL/RDMPYH). But immediate loads are
            # unaffected — the constant value doesn't depend on memory.
            if opcode in STORE_A_OPCODES:
                if (known_a is not None and
                        known_a[0] not in (Opcode.LDA_IMMEDIATE,) and
                        self._is_hardware_register(node.operand)):
                    known_a = None
                optimized.append(node)
                continue

            # Other instructions (CMP, TAX, TAY, CLC, SEC, etc.)
            # don't modify A — tracking stays valid
            optimized.append(node)

        return optimized

    # ========================================================================
    # INC/DEC Accumulator Folding
    # ========================================================================

    def _carry_dead_after(self, nodes: List['AsmNode'], start: int) -> bool:
        """Check if the carry flag is dead (not used) after position start."""
        from r65.compiler.codegen.asm_nodes import Instruction, Label
        for j in range(start, min(start + 30, len(nodes))):
            n = nodes[j]
            if isinstance(n, Label):
                return False  # Conservative: unknown predecessors could use carry
            if not isinstance(n, Instruction):
                continue
            if n.opcode in CARRY_READ_OPCODES:
                return False  # Carry is used
            if n.opcode in CARRY_WRITE_OPCODES:
                return True  # Carry is overwritten before use
            if n.opcode in RETURN_OPCODES or n.opcode in JUMP_OPCODES or n.opcode == Opcode.BRA:
                return True  # Control flow exit, carry not used on this path
        return False  # Conservative

    def _fold_inc_dec_accumulator(self, nodes: List['AsmNode']) -> List['AsmNode']:
        """
        Fold CLC/ADC #$01 → INC A and SEC/SBC #$01 → DEC A.

        INC/DEC are 1 instruction (2 cycles) vs CLC/ADC or SEC/SBC (4 cycles).
        Only safe when carry flag is not used after (INC/DEC don't affect carry).
        """
        from r65.compiler.codegen.asm_nodes import Instruction, Immediate

        optimized = []
        i = 0
        while i < len(nodes):
            node = nodes[i]

            if isinstance(node, Instruction):
                # CLC / ADC #$01 → INC A
                if (node.opcode == Opcode.CLC and
                        i + 1 < len(nodes) and
                        isinstance(nodes[i + 1], Instruction) and
                        nodes[i + 1].opcode == Opcode.ADC_IMMEDIATE and
                        isinstance(nodes[i + 1].operand, Immediate) and
                        nodes[i + 1].operand.value == 1):
                    if self._carry_dead_after(nodes, i + 2):
                        optimized.append(Instruction(Opcode.INC, comment="INC A (folded CLC/ADC #1)"))
                        self.stats.inc_dec_folded += 1
                        i += 2
                        continue

                # SEC / SBC #$01 → DEC A
                if (node.opcode == Opcode.SEC and
                        i + 1 < len(nodes) and
                        isinstance(nodes[i + 1], Instruction) and
                        nodes[i + 1].opcode == Opcode.SBC_IMMEDIATE and
                        isinstance(nodes[i + 1].operand, Immediate) and
                        nodes[i + 1].operand.value == 1):
                    if self._carry_dead_after(nodes, i + 2):
                        optimized.append(Instruction(Opcode.DEC, comment="DEC A (folded SEC/SBC #1)"))
                        self.stats.inc_dec_folded += 1
                        i += 2
                        continue

            optimized.append(node)
            i += 1

        return optimized

    # ========================================================================
    # Redundant CMP #$00 Elimination
    # ========================================================================

    # Branches that only check Z or N flags (not carry or overflow)
    _Z_N_BRANCH_OPCODES = frozenset({
        Opcode.BEQ, Opcode.BNE, Opcode.BMI, Opcode.BPL,
    })

    def _eliminate_redundant_cmp_zero(self, nodes: List['AsmNode']) -> List['AsmNode']:
        """
        Eliminate CMP #$00 when the previous instruction already sets Z/N flags
        and the next branch only checks Z or N (not carry or overflow).

        CMP #$00 is redundant after LDA, AND, ORA, EOR, etc. when the only
        consumers of the flags are BEQ/BNE/BMI/BPL.
        """
        from r65.compiler.codegen.asm_nodes import Instruction, Immediate, Label, Directive

        optimized = []
        i = 0
        while i < len(nodes):
            node = nodes[i]

            if (isinstance(node, Instruction) and
                    node.opcode == Opcode.CMP_IMMEDIATE and
                    isinstance(node.operand, Immediate) and
                    node.operand.value == 0):
                # Check: previous real instruction sets Z/N
                prev_sets_zn = False
                for j in range(len(optimized) - 1, -1, -1):
                    prev = optimized[j]
                    if isinstance(prev, (Directive,)):
                        continue
                    if isinstance(prev, Label):
                        break
                    if isinstance(prev, Instruction):
                        prev_sets_zn = prev.opcode in NZ_OF_A_OPCODES
                        break
                    break

                # Check: ALL subsequent flag consumers only use Z/N flags.
                # Must scan past the first branch because signed comparisons
                # emit BNE (Z-only) followed by BVC/EOR (V-dependent).
                # Long branch fixup may insert JMP between them.
                # If ANY consumer needs V or C, the CMP is NOT redundant.
                next_only_zn = False
                if prev_sets_zn:
                    all_zn = True
                    for k in range(i + 1, len(nodes)):
                        n = nodes[k]
                        if isinstance(n, Directive):
                            continue
                        if isinstance(n, Label):
                            break  # Label = control flow merge, stop scanning
                        if isinstance(n, Instruction):
                            if n.opcode in self._Z_N_BRANCH_OPCODES:
                                continue  # This branch only uses Z/N, keep scanning
                            elif n.opcode in (Opcode.BVC, Opcode.BVS,
                                              Opcode.BCC, Opcode.BCS):
                                all_zn = False  # Needs V or C from CMP
                                break
                            elif n.opcode in (Opcode.EOR_IMMEDIATE,
                                              Opcode.JMP_ABSOLUTE, Opcode.JMP_LONG):
                                continue  # Part of signed compare / long branch pattern
                            elif (n.opcode in (Opcode.SEP_IMMEDIATE, Opcode.REP_IMMEDIATE)
                                    and isinstance(n.operand, Immediate)
                                    and isinstance(n.operand.value, int)
                                    and not n.operand.value & ~(M_FLAG | X_FLAG)):
                                continue  # A width switch before the branch leaves N/Z/C/V
                            else:
                                break  # Non-branch instruction, stop scanning
                    next_only_zn = all_zn

                if prev_sets_zn and next_only_zn:
                    self.stats.redundant_cmp_zero_eliminated += 1
                    i += 1
                    continue

            optimized.append(node)
            i += 1

        return optimized

    # ========================================================================
    # STZ Conversion
    # ========================================================================

    # Mapping from STA addressing modes to equivalent STZ opcodes
    _STA_TO_STZ = {
        Opcode.STA_DP: Opcode.STZ_DP,
        Opcode.STA_DP_X: Opcode.STZ_DP_X,
        Opcode.STA_ABSOLUTE: Opcode.STZ_ABSOLUTE,
        Opcode.STA_ABSOLUTE_X: Opcode.STZ_ABSOLUTE_X,
    }

    # Instructions that replace A without reading it first. Most writers of A
    # also read it (AND, ADC, ASL A, XBA, ...), so after one of those the
    # zero the LDA put in A is still needed.
    _LOADS_A_WITHOUT_READING = frozenset({'LDA', 'PLA', 'TXA', 'TYA', 'TDC', 'TSC'})

    def _convert_zero_stores_to_stz(self, nodes: List['AsmNode']) -> List['AsmNode']:
        """
        Convert LDA #$00 / STA addr sequences to STZ addr.

        When A is loaded with zero solely for storing to memory, STZ is more
        efficient: it stores zero directly without occupying A (1 fewer instr).

        Converts all consecutive STZ-compatible STAs after LDA #$00 to STZ.
        The LDA #$00 is removed if A is overwritten before next use, or kept
        if subsequent code depends on A=0.
        """
        from r65.compiler.codegen.asm_nodes import Instruction, Immediate, Label, Directive

        optimized = []
        i = 0
        while i < len(nodes):
            node = nodes[i]

            # Match LDA #$00 (immediate zero)
            if (isinstance(node, Instruction) and
                    node.opcode == Opcode.LDA_IMMEDIATE and
                    isinstance(node.operand, Immediate) and
                    node.operand.value == 0):

                # Scan forward: convert all consecutive STZ-compatible STAs
                j = i + 1
                stz_indices = []
                while j < len(nodes):
                    n = nodes[j]
                    if isinstance(n, (Directive, Label)):
                        break
                    if not isinstance(n, Instruction):
                        j += 1
                        continue
                    if n.opcode in self._STA_TO_STZ:
                        stz_indices.append(j)
                        j += 1
                        continue
                    break  # Non-STA instruction

                if not stz_indices:
                    optimized.append(node)
                    i += 1
                    continue

                # Check if A=0 is needed after the last converted STA.
                # Skip over mode switches (SEP/REP) and directives since
                # they don't use A's value.
                a_needed = True
                k = stz_indices[-1] + 1
                while k < len(nodes):
                    n = nodes[k]
                    if isinstance(n, Label):
                        break
                    if isinstance(n, Directive):
                        k += 1
                        continue
                    if isinstance(n, Instruction):
                        if mnemonic(n.opcode) in self._LOADS_A_WITHOUT_READING:
                            a_needed = False  # A is overwritten
                            break
                        # SEP/REP don't use A's value, skip them
                        if n.opcode in (Opcode.SEP_IMMEDIATE, Opcode.REP_IMMEDIATE):
                            k += 1
                            continue
                        break  # Other instruction uses A (or might)
                    k += 1

                # If A=0 is potentially needed, keep the LDA #$00
                if a_needed:
                    optimized.append(node)

                # Convert STAs to STZs
                for idx in stz_indices:
                    sta_node = nodes[idx]
                    stz_opcode = self._STA_TO_STZ[sta_node.opcode]
                    stz_node = Instruction(stz_opcode, sta_node.operand, sta_node.comment)
                    optimized.append(stz_node)
                    self.stats.stz_conversions += 1

                # Resume after the last converted STA
                i = stz_indices[-1] + 1
                continue

            optimized.append(node)
            i += 1

        return optimized


# ============================================================================
# Public API
# ============================================================================

def _split_regions(nodes: List['AsmNode'],
                   region_starts: Optional[List['AsmNode']]) -> List[List['AsmNode']]:
    """Cut `nodes` before each of `region_starts` (matched by identity).

    A start no longer in the list is skipped, merging its region into the
    previous one.
    """
    if not region_starts:
        return [nodes]
    position = {id(node): i for i, node in enumerate(nodes)}
    cuts = sorted({position[id(node)] for node in region_starts
                   if id(node) in position} - {0})
    bounds = [0] + cuts + [len(nodes)]
    return [nodes[a:b] for a, b in zip(bounds, bounds[1:])]


def optimize_nodes(nodes: List['AsmNode'], volatile_names: Set[str] = None,
                   volatile_addresses: Set[int] = None,
                   region_starts: Optional[List['AsmNode']] = None,
                   entry_modes: Optional[dict] = None,
                   call_exit_modes: Optional[dict] = None,
                   acc_abi: Optional[dict] = None,
                   ) -> Tuple[List['AsmNode'], int]:
    """
    Apply peephole optimizations to AsmNode list.

    Args:
        nodes: List of AsmNode objects
        volatile_names: Set of variable names that are volatile (from #[hw] attributes).
                       Stores to these locations will not be eliminated.
        volatile_addresses: Set of hardware register addresses (from #[hw] attributes).
                           Stores to these addresses will not be eliminated.
        region_starts: Optional first nodes of the regions (functions) to
                       optimize independently; see PeepholeOptimizer.optimize.
        entry_modes, call_exit_modes: Each function's entry and exit mode;
                       see PeepholeOptimizer.optimize.
        acc_abi: Each function's accumulator ABI; see
                       PeepholeOptimizer.optimize.

    Returns:
        Tuple of (optimized nodes, number of optimizations applied)
    """
    optimizer = PeepholeOptimizer(volatile_names, volatile_addresses)
    optimized = optimizer.optimize(nodes, region_starts, entry_modes, call_exit_modes,
                                   acc_abi)
    return optimized, optimizer.optimizations_applied
