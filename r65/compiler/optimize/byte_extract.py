# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Byte-Extract Lowering Pass.

Rewrites `(x >> 8) as u8` on a value loaded from memory into a plain
byte load of that value's high byte.

The 65816 is little-endian, so the high byte of a 16-bit value at address
N lives at N+1 — reading it costs one instruction, no shift and no mode
switch. What the compiler emitted instead was the generic lowering of
each step in turn::

    REP #$20        ; 16-bit A for the load
    LDA $20         ; the whole word
    XBA             ; shift right 8
    AND #$FF        ; clear the high byte
    SEP #$20        ; 8-bit A for the truncated result
    STA $22

against::

    LDA $21         ; the high byte, directly
    STA $22

This cannot be done as an assembly peephole. `_emit_shift_right` in
instruction selection is handed only the shift count; it operates on
whatever happens to be in A and has no idea the value came from a known
address, so there is nothing there to rewrite the load into. Recognising
the whole `Load → Shr 8 → Trunc` chain needs a place where all three are
visible at once, which is MIR.

Folding the chain also deletes the two intermediate virtual registers,
which is where most of the win actually comes from: they were each
getting a stack slot and a spill.

Correctness note on signedness: an arithmetic shift right by 8 puts bits
15..8 into 7..0 and fills the top with sign, and a logical shift fills it
with zero. Either way the *low byte* of the result is the original high
byte, and the cast keeps only that low byte — so the rewrite holds for
`u16` and `i16` alike.
"""

from dataclasses import replace
from typing import Dict, List, Optional

from r65.compiler.codegen.type_utils import is_narrowing_convert
from r65.compiler.mir.nodes import (
    BinaryOp, Immediate, Load, MemoryLocation, MIRFunction, MIRProgram,
    OperandRole, TypeConvert, VirtualRegister, iter_operands,
)


def _type_size(type_info) -> Optional[int]:
    """Byte width of a TypeInfo, or None if it doesn't report one."""
    size = getattr(type_info, 'size_bytes', None)
    return size if isinstance(size, int) else None


def _addressed_by_label(loc: MemoryLocation) -> bool:
    """True if instruction selection reaches this location as ``label+offset``.

    `_resolve_operand` tests `rom_label` *before* it tests `address`, so a
    ROM symbol carrying a label is addressed through its offset no matter
    what `address` holds.
    """
    return bool(
        loc.symbol is not None
        and getattr(loc.symbol, 'rom_label', None)
        and loc.storage_type == 'rom'
    )


def _addressing_is_unambiguous(loc: MemoryLocation) -> bool:
    """True if it is clear which field addresses this location.

    Every location the compiler builds today is reached either by label or
    by number, never both — across a whole classickong build, every ROM
    location has a label and no address, and every location with an address
    has no label. `_high_byte_of` relies on that: it bumps one field and
    would silently read the wrong byte if the other were the live one.

    Rather than carry an untestable branch for a case that cannot occur,
    the pass checks the assumption and declines to fold when it does not
    hold. Worst case is a missed optimization instead of a wrong load.
    """
    return not (_addressed_by_label(loc) and loc.address is not None)


def _high_byte_of(loc: MemoryLocation) -> MemoryLocation:
    """The location one byte above `loc`.

    An explicit `address` already has any offset folded into it, so the
    bump goes there; a label-addressed or allocator-addressed location is
    reached as ``base + offset``, so the bump goes on the offset. Callers
    must have checked `_addressing_is_unambiguous` first.
    """
    if not _addressed_by_label(loc) and loc.address is not None:
        return replace(loc, address=loc.address + 1)
    return replace(loc, offset=loc.offset + 1)


class ByteExtractOptimizer:
    """Fold `(mem16 >> 8) as u8` into a load of the high byte."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.folded = 0

    def optimize(self, mir_program: MIRProgram) -> int:
        """Rewrite every foldable chain in the program. Returns the count."""
        self.folded = 0
        for func in mir_program.functions:
            self._optimize_function(func)
        return self.folded

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _optimize_function(self, func: MIRFunction) -> None:
        uses = self._count_vreg_reads(func)
        for block in func.blocks.values():
            block.instructions = self._rewrite_block(block.instructions, uses)

    @staticmethod
    def _count_vreg_reads(func: MIRFunction) -> Dict[int, int]:
        """How many times each vreg is read anywhere in the function.

        The chain's two intermediates are only dead once nothing else
        reads them — including a later block, which is why this counts
        across the whole function rather than per block. `iter_operands`
        is the registry every pass shares, and a coverage test keeps it
        exhaustive over MIR node types, so no reader goes uncounted.
        """
        counts: Dict[int, int] = {}
        for block in func.blocks.values():
            for instr in block.instructions:
                for _spec, value in iter_operands(instr, role=OperandRole.READ):
                    if isinstance(value, VirtualRegister):
                        counts[value.id] = counts.get(value.id, 0) + 1
        return counts

    def _rewrite_block(self, instrs: List, uses: Dict[int, int]) -> List:
        out: List = []
        i = 0
        n = len(instrs)
        while i < n:
            folded = self._try_fold(instrs, i, uses)
            if folded is not None:
                # The word load stays only if something other than the
                # shift still wants the whole 16 bits — a second read of
                # the same variable makes the builder share one load.
                if uses.get(instrs[i].dest.id, 0) > 1:
                    out.append(instrs[i])
                out.append(folded)
                self.folded += 1
                if self.verbose:
                    print(f"  byte-extract: {instrs[i].source} -> {folded.source}")
                i += 3
                continue
            out.append(instrs[i])
            i += 1
        return out

    def _try_fold(self, instrs: List, i: int, uses: Dict[int, int]) -> Optional[Load]:
        """The replacement Load for a chain at `i`, or None if it isn't one.

        The three instructions must be consecutive. That is how the MIR
        builder emits the expression, and requiring it means the pass
        needs no aliasing analysis: nothing can write the source memory
        between the load we are deleting and the load we are putting in
        its place.
        """
        if i + 2 >= len(instrs):
            return None
        load, shift, cast = instrs[i], instrs[i + 1], instrs[i + 2]

        if not (isinstance(load, Load)
                and isinstance(shift, BinaryOp)
                and isinstance(cast, TypeConvert)):
            return None

        # A volatile read has to happen exactly as written — narrowing a
        # 16-bit hardware read to its high byte would drop the low-byte
        # access the device may be counting on.
        if load.source.is_volatile:
            return None

        # `_high_byte_of` has to know which field addresses this location.
        if not _addressing_is_unambiguous(load.source):
            return None

        if not isinstance(load.dest, VirtualRegister):
            return None
        if _type_size(load.type_info) != 2:
            return None

        # ... >> 8, reading exactly the loaded word
        if shift.op != '>>':
            return None
        if not (isinstance(shift.left, VirtualRegister)
                and shift.left.id == load.dest.id):
            return None
        if not (isinstance(shift.right, Immediate) and shift.right.value == 8):
            return None
        if not isinstance(shift.dest, VirtualRegister):
            return None

        # ... truncated to a single byte. `is_narrowing_convert` is the same
        # predicate instruction selection and the register allocator use, so
        # all three agree on what counts as a two-to-one-byte cast (and it
        # rules out pointer conversions, which have their own emitter).
        if not (isinstance(cast.source, VirtualRegister)
                and cast.source.id == shift.dest.id):
            return None
        if not is_narrowing_convert(cast):
            return None
        # `as bool` normalizes to 0/1 rather than keeping the byte, so the
        # fold would change the value — unlike coalescing, which only moves
        # where the result lives and so needs no such guard. Today the builder
        # lowers `as bool` to a separate ToBool node and this cannot match
        # anyway, but TypeConvert's contract still covers boolean conversion.
        if getattr(cast.target_type, 'name', None) == 'bool':
            return None

        # The shift result must die at the cast: it is the one value the
        # rewrite actually destroys. The loaded word may have other
        # readers — `_rewrite_block` keeps its load when it does.
        if uses.get(shift.dest.id, 0) != 1:
            return None

        return Load(
            dest=cast.dest,
            source=_high_byte_of(load.source),
            type_info=cast.target_type,
            source_loc=load.source_loc,
            comment=load.comment,
        )
