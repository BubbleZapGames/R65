# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Byte-Widening Pass.

A u8 widened to u16 is loaded in m8 and then masked in m16::

    SEP #$20 / LDA ($38),Y / REP #$20 / AND #$00FF

The 16-bit load reads the byte after it too, and the mask clears it again,
so the same value comes from one m16 load: ``LDA ($38),Y / AND #$00FF``.
Two rewrites put the work in u16 on MIR, where the rest of the pipeline
(known bits, A residency, block widths) sees it:

  * a byte load whose one reader widens it loads a word and masks it::

        %v = Load [m] : u8             %t = Load [m] : u16
        %w = %v as u16          ->     %w = %t & $FF : u16

  * a byte masked by a constant below $100 and then widened is widened
    first, without clearing the high byte the mask clears anyway::

        %t = %v & K : u8               %u = %v as u16    (high byte unread)
        %w = %t as u16          ->     %w = %u & K : u16

The extra byte read is harmless except from I/O, where a read can have an
effect (`$2139` advances the VRAM address): never a `#[hw]` static, and
through a pointer only when the pointer provably holds the address of a
static that is not one. Signed widening sign-extends from bit 7 and is
left alone.
"""

from dataclasses import replace
from typing import Dict, List

from r65.compiler.mir.liveness import liveness_defs, liveness_uses
from r65.compiler.mir.nodes import (
    BinaryOp, Immediate, Load, LoadIndirect, MIRFunction, MIRProgram, Move,
    TypeConvert, VirtualRegister,
)
from r65.compiler.optimize.known_bits import _name, _widens


def _static_storage(symbol):
    """The storage class of the static *symbol* names, or None if it is not
    a static the compiler places (an extern label could be anything)."""
    from r65.compiler.hir.symbol_table import SymbolKind
    if getattr(symbol, 'kind', None) is not SymbolKind.STATIC_VAR:
        return None
    decl = symbol.definition
    if getattr(decl, 'is_extern', False):
        return None
    attr = getattr(decl, 'storage_attr', None)
    return attr.storage_kind.value if attr else 'ram'


class ByteWidener:
    """Widen bytes with one m16 load (or a masked m16 read) instead of an
    m8 load and a zero-extend."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.widened = 0

    def optimize(self, mir_program: MIRProgram) -> int:
        """Rewrite every eligible widening. Returns the count."""
        self.widened = 0
        for func in mir_program.functions:
            defs: Dict[int, List] = {}
            reads: Dict[int, int] = {}
            for block in func.blocks.values():
                for ins in block.instructions:
                    for d in liveness_defs(ins):
                        if type(d) is VirtualRegister:
                            defs.setdefault(d.id, []).append(ins)
                    for u in liveness_uses(ins):
                        if type(u) is VirtualRegister:
                            reads[u.id] = reads.get(u.id, 0) + 1
            for block in func.blocks.values():
                # A widened mask can leave a byte load feeding a widening.
                while self._rewrite_block(func, block, defs, reads):
                    pass
        return self.widened

    def _rewrite_block(self, func: MIRFunction, block, defs, reads) -> bool:
        """One pass over *block*. Returns whether anything changed."""
        instrs = block.instructions
        before = self.widened
        i = 0
        while i + 1 < len(instrs):
            first, conv = instrs[i], instrs[i + 1]
            if not (_widens(conv) and type(conv.source) is VirtualRegister
                    and getattr(first, 'dest', None) == conv.source
                    and reads.get(conv.source.id) == 1
                    and len(defs.get(conv.source.id, ())) == 1):
                i += 1
                continue
            word = conv.target_type
            if self._wide_load_ok(first, defs):
                loaded = replace(first.dest, type_info=word)
                instrs[i] = replace(first, dest=loaded, type_info=word)
                instrs[i + 1] = BinaryOp(dest=conv.dest, left=loaded, op='&',
                                         right=Immediate(0xFF), type_info=word,
                                         source_loc=conv.source_loc)
                self.widened += 1
            elif self._small_mask(first):
                wide = func.vreg_allocator.alloc(type_info=word,
                                                 hint=first.dest.hint)
                instrs[i] = TypeConvert(dest=wide, source=first.left,
                                        source_type=conv.source_type,
                                        target_type=word, high_undefined=True,
                                        source_loc=first.source_loc)
                instrs[i + 1] = BinaryOp(dest=conv.dest, left=wide, op='&',
                                         right=first.right, type_info=word,
                                         source_loc=conv.source_loc)
                defs[wide.id] = [instrs[i]]
                reads[wide.id] = 1
                self.widened += 1
            i += 2
        return self.widened != before

    @staticmethod
    def _small_mask(ins) -> bool:
        """`%t = %v & K : u8` with K a constant below $100."""
        return (type(ins) is BinaryOp and ins.op == '&'
                and _name(ins.type_info) == 'u8'
                and type(ins.left) is VirtualRegister
                and _name(ins.left.type_info) == 'u8'
                and type(ins.right) is Immediate and ins.right.symbol is None
                and isinstance(ins.right.value, int) and 0 <= ins.right.value < 0x100)

    def _wide_load_ok(self, ins, defs) -> bool:
        """A u8 load whose following byte may be read too."""
        if _name(getattr(ins, 'type_info', None)) != 'u8':
            return False
        if type(ins) is Load:
            storage = ins.source.storage_type
            if storage == 'unknown':
                storage = _static_storage(ins.source.symbol)
            return storage not in (None, 'hw')
        if type(ins) is LoadIndirect:
            return self._points_at_static(ins.pointer, defs, set())
        return False

    def _points_at_static(self, ptr, defs, seen) -> bool:
        """Whether every definition of *ptr* holds the address of a static
        that is not `#[hw]` - set from `&STATIC`, copied, or stepped from
        such a pointer."""
        if type(ptr) is not VirtualRegister:
            return False
        if ptr.id in seen:
            return True
        seen.add(ptr.id)
        sources = defs.get(ptr.id)
        if not sources:
            return False
        for ins in sources:
            if type(ins) is Move:
                src = ins.source
                if type(src) is Immediate:
                    if _static_storage(src.symbol) in (None, 'hw'):
                        return False
                elif not self._points_at_static(src, defs, seen):
                    return False
            elif (type(ins) is BinaryOp and ins.op in ('+', '-')
                    and type(ins.right) is Immediate and ins.right.symbol is None):
                if not self._points_at_static(ins.left, defs, seen):
                    return False
            else:
                return False
        return True
