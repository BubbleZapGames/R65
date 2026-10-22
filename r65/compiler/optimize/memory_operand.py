# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Memory-Operand Folding Pass.

Rewrites a load whose only reader is an operand of a binary operation or
the right-hand side of a comparison into that instruction's operand, so the
value is read by the ALU instruction itself instead of through a
temporary::

    %0 = Load [G_X]
    %1 = Load [G_VX]
    %2 = %0 + %1          ->     %2 = %0 + [G_VX]

The 65816's ADC/SBC/AND/ORA/EOR/CMP all take a memory operand in every
addressing mode LDA does, so the emitted code is the hand-written form::

    LDA G_X                      ; was: LDA G_X / STA $01,S
    CLC                          ;      LDA G_VX / STA $03,S
    ADC G_VX                     ;      LDA $01,S / CLC / ADC $03,S
    STA G_X

The win is mostly indirect: a materialized ``%1`` load clobbers A between
``%0``'s definition and its use, so hardware-register coalescence cannot
keep ``%0`` in A, and both values spill to stack slots. Deleting the load
restores ``%0``'s A residency and, in a function with nothing else to
spill, the frame too.

A load on the left goes to the right of a commutative operation whose
right operand is computed (``%2 = %1 & %0`` → ``%2 = %1 & [M]``), and stays
on the left of a subtract, which codegen then computes as
``EOR #$FF / SEC / ADC [M]`` with the right operand in A. Either way the
computed value no longer waits in a stack slot while A holds the load::

    %0 = Load [EXT,X]
    %1 = ~%m
    %2 = %0 & %1          ->     LDA m / EOR #$FF / AND EXT,X

Moving a read later is only legal if nothing between the load and its use
can change what it reads. The pass allows only loads, register-to-register
moves and ALU operations into vregs in between — no stores, calls, inline
asm or index-register writes — which covers how the MIR builder emits
``a op b`` (the left operand loaded, then the right one computed) without
any aliasing analysis. See `_may_pass`.
"""

from typing import Dict, List, Optional

from r65.compiler.mir.nodes import (
    BinaryOp, Compare, HardwareRegister, Load, MemoryLocation, MIRFunction,
    MIRProgram, Move, UnaryOp, VirtualRegister, count_vreg_reads,
)

# Operations whose right operand may be a memory location. Shifts, multiply
# and divide take constant amounts only.
_FOLDABLE_OPS = frozenset({'+', '-', '&', '|', '^'})
_COMMUTATIVE_OPS = frozenset({'+', '&', '|', '^'})

# Statics the compiler places itself. A hardware register is volatile and
# must be read exactly where written; anything else is not a plain static.
_FOLDABLE_STORAGE = frozenset({'zeropage', 'lowram', 'ram', 'rom'})


def _type_size(type_info) -> Optional[int]:
    size = getattr(type_info, 'size_bytes', None)
    return size if isinstance(size, int) else None


class MemoryOperandFolder:
    """Fold single-use loads into the ALU instruction that consumes them."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.folded = 0

    def optimize(self, mir_program: MIRProgram) -> int:
        """Rewrite every foldable load in the program. Returns the count."""
        self.folded = 0
        for func in mir_program.functions:
            self._optimize_function(func)
        return self.folded

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _optimize_function(self, func: MIRFunction) -> None:
        uses = count_vreg_reads(func)
        for block in func.blocks.values():
            block.instructions = self._rewrite_block(block.instructions, uses)

    def _rewrite_block(self, instrs: List, uses: Dict[int, int]) -> List:
        out: List = []
        # Loads still eligible to be folded, by the vreg that carries their
        # value, as the positions in `out` to delete on a fold: the load
        # itself, plus any same-size copy the value went through on the way
        # (`x as u8` on an `i8` is a Move that changes nothing but the type).
        pending: Dict[int, List[int]] = {}

        for instr in instrs:
            folded = self._try_fold(instr, out, pending)
            if folded is not None:
                instr, positions = folded
                for pos in positions:
                    out[pos] = None
                self.folded += 1
            elif self._is_alias(instr, uses, pending, out):
                # The copy's dest now carries the value; the copy goes with
                # the load if a fold happens, and stays if not.
                positions = pending.pop(instr.source.id)
                out.append(instr)
                pending[instr.dest.id] = positions + [len(out) - 1]
                continue

            # Anything that is not a load, a plain move or an ALU op into a
            # vreg ends every pending candidate: it might write memory or an
            # index register.
            if not self._may_pass(instr):
                pending.clear()
            elif pending:
                # An instruction that defines an index-hinted vreg ends the
                # indexed candidates (see _may_pass); the plain ones survive.
                # One that redefines a pending vreg ends that candidate: the
                # later reader sees the new value, not the load.
                dest = getattr(instr, 'dest', None)
                if isinstance(dest, VirtualRegister):
                    pending.pop(dest.id, None)
                    if dest.register_hint:
                        pending = {vid: positions for vid, positions in pending.items()
                                   if out[positions[0]].source.index_register is None}

            out.append(instr)
            if self._is_candidate(instr, uses):
                pending[instr.dest.id] = [len(out) - 1]

        return [i for i in out if i is not None]

    @staticmethod
    def _is_candidate(instr, uses: Dict[int, int]) -> bool:
        """A load the pass could delete if its one reader turns out to qualify."""
        if not isinstance(instr, Load) or not isinstance(instr.dest, VirtualRegister):
            return False
        if uses.get(instr.dest.id, 0) != 1:
            return False
        loc = instr.source
        if loc.is_volatile or loc.storage_type not in _FOLDABLE_STORAGE:
            return False
        if instr.dest.register_hint:
            return False
        return _type_size(instr.type_info) in (1, 2)

    @staticmethod
    def _is_alias(instr, uses: Dict[int, int], pending: Dict[int, List[int]],
                  out: List) -> bool:
        """A vreg-to-vreg Move of a pending value that keeps its width, whose
        dest is read exactly once — so the value can be followed through it."""
        if not (isinstance(instr, Move)
                and isinstance(instr.dest, VirtualRegister)
                and isinstance(instr.source, VirtualRegister)
                and instr.source.id in pending):
            return False
        if uses.get(instr.dest.id, 0) != 1 or instr.dest.register_hint:
            return False
        load = out[pending[instr.source.id][0]]
        size = _type_size(instr.type_info)
        return size is not None and size == _type_size(load.type_info)

    @staticmethod
    def _may_pass(instr) -> bool:
        """True if a pending load may be read *after* this instruction instead
        of before it.

        A load reads memory and writes a vreg; a move between vregs or from
        an immediate, or an ALU operation (computed in A), writes a vreg.
        None of them writes memory. A vreg write can
        still land in X or Y — a loop counter promoted there — which would
        change what an indexed location addresses; `_rewrite_block` drops the
        indexed candidates when the dest carries a register hint, and a
        hardware-register dest is refused here outright.
        """
        if isinstance(instr, Load):
            return isinstance(instr.dest, VirtualRegister)
        if isinstance(instr, Move):
            return (isinstance(instr.dest, VirtualRegister)
                    and not isinstance(instr.source, (HardwareRegister, MemoryLocation)))
        if isinstance(instr, (BinaryOp, UnaryOp)):
            return isinstance(instr.dest, VirtualRegister)
        return False

    def _try_fold(self, instr, out: List, pending: Dict[int, int]):
        """`(rewritten instr, positions to delete)` if *instr*'s right operand
        is a pending load, or its left one is and the right is computed (see
        the module docstring), else None."""
        if isinstance(instr, BinaryOp):
            if instr.op not in _FOLDABLE_OPS:
                return None
        elif not isinstance(instr, Compare):
            return None

        left_fold = False
        right = instr.right
        if not isinstance(right, VirtualRegister) or right.id not in pending:
            left = instr.left
            if not (isinstance(instr, BinaryOp)
                    and isinstance(left, VirtualRegister) and left.id in pending
                    and isinstance(right, VirtualRegister) and left.id != right.id):
                return None
            left_fold = True
        # An operation into or out of X/Y runs in 16-bit mode whatever its
        # type, so an 8-bit memory operand would be read as two bytes.
        for reg in (instr.left, instr.right, getattr(instr, 'dest', None)):
            if isinstance(reg, HardwareRegister) and reg.name in ('X', 'Y'):
                return None

        if left_fold:
            positions = pending.pop(instr.left.id)
            load = out[positions[0]]
            if _type_size(load.type_info) != _type_size(instr.type_info):
                pending[instr.left.id] = positions
                return None
            if self.verbose:
                print(f"  memory-operand: {load.dest} <- {load.source} folded into {instr}")
            if instr.op in _COMMUTATIVE_OPS:
                instr.left, instr.right = instr.right, load.source
            else:
                instr.left = load.source
            return instr, positions
        # `a op a` reads the vreg twice; the use count already excludes it,
        # but the left operand must not be the load either way.
        if isinstance(instr.left, VirtualRegister) and instr.left.id == right.id:
            return None

        positions = pending.pop(right.id)
        load = out[positions[0]]
        if _type_size(load.type_info) != _type_size(instr.type_info):
            pending[right.id] = positions
            return None

        if self.verbose:
            print(f"  memory-operand: {load.dest} <- {load.source} folded into {instr}")
        instr.right = load.source
        return instr, positions
