# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Constant far-pointer arguments as immediates.

A far pointer to a static passed on the stack - `u8_to_dec(&BUF[4] as far
*u8, n)`, what every format! formatter call is - was built in a 3-byte
stack slot, then pushed from it byte by byte::

    LDA #<$7E2004 / STA $01,S         ; build
    LDA #>$7E2004 / STA $02,S
    LDA #$7E      / STA $03,S
    LDA $03,S / PHA / LDA $03,S / PHA / LDA $03,S / PHA   ; push

about 50 cycles, and a frame slot. When the argument's vreg holds one
constant address - its only definition a Move of `#BUF+4` or of a number,
a near-to-far conversion of a ROM static's address (`&MSG as far *u8`,
whose bank is the label's), or a copy of one of these - the argument takes
the immediate itself, which
call selection pushes as `LDA #$7E / PHA / PEA $2004` (10 cycles), and the
Moves left with no reader go.

Only stack arguments with a declared 3-byte parameter type change: an
untyped immediate would be sized by its value, and a scratch-promoted
argument stores through a different path. So this runs after parameter
analysis. Static addresses only - a function's address is a far function
pointer, a local's is on the stack - and a parameter's vreg never
qualifies: its entry value is a definition no instruction shows.
"""

from typing import Dict

from r65.compiler.codegen.type_utils import get_type_size
from r65.compiler.mir.nodes import (
    ArgumentMechanism, CallBase, Immediate, MIRFunction, MIRProgram, Move,
    OperandRole, TypeConvert, VirtualRegister, count_vreg_reads, iter_operands,
)


def _static_or_number(imm: Immediate) -> bool:
    if imm.symbol is None:
        return isinstance(imm.value, int)
    from r65.compiler.hir.symbol_table import SymbolKind
    return getattr(imm.symbol, 'kind', None) is SymbolKind.STATIC_VAR


def _widens_alike(imm: Immediate) -> bool:
    """A near address whose near-to-far conversion gives the bank the push
    would: a ROM label's own (`:label`), or 0 for a number. Any other
    static widens to bank 0 whatever its address, which the push would not
    reproduce."""
    if imm.symbol is None:
        return isinstance(imm.value, int)
    return _static_or_number(imm) and bool(getattr(imm.symbol, 'rom_label', None))


class FarConstantArgFolder:
    """Pass constant far addresses to calls as immediates."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.folded = 0

    def optimize(self, mir_program: MIRProgram) -> int:
        """Rewrite every qualifying argument. Returns the count."""
        self.folded = 0
        for func in mir_program.functions:
            self._optimize_function(func)
        return self.folded

    def _optimize_function(self, func: MIRFunction) -> None:
        consts, droppable = self._constant_vregs(func)
        if not consts:
            return
        changed = False
        for block in func.blocks.values():
            for instr in block.instructions:
                if not isinstance(instr, CallBase):
                    continue
                for arg in getattr(instr, 'args', ()):
                    value = arg.value
                    if (arg.mechanism is ArgumentMechanism.STACK
                            and isinstance(value, VirtualRegister) and value.id in consts
                            and arg.param_type is not None
                            and get_type_size(arg.param_type) == 3):
                        imm = consts[value.id]
                        arg.value = Immediate(imm.value, symbol=imm.symbol)
                        self.folded += 1
                        changed = True
        if changed:
            self._drop_dead_moves(func, droppable)

    @staticmethod
    def _constant_vregs(func: MIRFunction):
        """`(consts, droppable)`: vreg id -> the constant far address it
        always holds, and the ids whose defining instruction has no effect
        but that value (the far ones and the near addresses they widen)."""
        defs: Dict[int, int] = {}
        for block in func.blocks.values():
            for instr in block.instructions:
                for _, v in iter_operands(instr, role=OperandRole.WRITE):
                    if isinstance(v, VirtualRegister):
                        defs[v.id] = defs.get(v.id, 0) + 1
        params = {v.id for v in func.param_to_vreg.values() if v is not None}
        if func.self_y_vreg is not None:
            params.add(func.self_y_vreg.id)

        def single_def(instr) -> bool:
            return (isinstance(instr.dest, VirtualRegister)
                    and defs.get(instr.dest.id) == 1 and instr.dest.id not in params)

        consts: Dict[int, Immediate] = {}
        near: Dict[int, Immediate] = {}
        widened = []
        copies = []
        for block in func.blocks.values():
            for instr in block.instructions:
                if isinstance(instr, Move) and single_def(instr):
                    size = get_type_size(instr.dest.type_info)
                    if isinstance(instr.source, Immediate):
                        if size == 3 and _static_or_number(instr.source):
                            consts[instr.dest.id] = instr.source
                        elif size == 2 and _widens_alike(instr.source):
                            near[instr.dest.id] = instr.source
                    elif isinstance(instr.source, VirtualRegister) and size == 3:
                        copies.append((instr.dest.id, instr.source.id))
                elif (isinstance(instr, TypeConvert) and single_def(instr)
                        and isinstance(instr.source, VirtualRegister)
                        and get_type_size(instr.source_type) == 2
                        and get_type_size(instr.target_type) == 3):
                    widened.append((instr.dest.id, instr.source.id))
        for dest, src in widened:
            if src in near:
                consts[dest] = near[src]
        changed = True
        while changed:
            changed = False
            for dest, src in copies:
                if dest not in consts and src in consts:
                    consts[dest] = consts[src]
                    changed = True
        return consts, set(consts) | set(near)

    @staticmethod
    def _drop_dead_moves(func: MIRFunction, droppable) -> None:
        """Remove the Moves and conversions defining constant vregs nothing
        reads any more, a copy chain at a time. Only those this pass proved
        constant, which have no other effect."""
        while True:
            reads = count_vreg_reads(func)
            dropped = False
            for block in func.blocks.values():
                kept = [i for i in block.instructions
                        if not (isinstance(i, (Move, TypeConvert))
                                and isinstance(i.dest, VirtualRegister)
                                and i.dest.id in droppable and not reads.get(i.dest.id))]
                if len(kept) != len(block.instructions):
                    block.instructions = kept
                    dropped = True
            if not dropped:
                return
