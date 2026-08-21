# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""`(mem16 >> 8) as u8` should become a load of the high byte.

The 65816 is little-endian, so the high byte of a word at N is at N+1 and
reading it is one instruction. The compiler used to lower each step of
the expression in turn instead — load the word in m16, XBA, mask, switch
back to m8 — and spill an intermediate for each step:

    REP #$20 / LDA $20 / XBA / AND #$FF / STA $01,S / SEP #$20 / STA $22

The fold has to happen in MIR: `_emit_shift_right` in instruction
selection is handed only the shift count, operates on whatever is in A,
and cannot see that the value came from a known address.

These tests pin the pattern and, just as importantly, the cases that
must *not* fold.
"""

import re

from r65.compiler.main import compile_string


def _main_asm(source: str) -> str:
    asm = compile_string(source, cfg_options=["snes"])
    return asm.split("main:", 1)[1].split("__SCMP", 1)[0]


ZP_WORD = """
#[zeropage(0x20)] static mut W: u16;
#[zeropage(0x30)] static mut R: u8;
#[entry] fn main() {{ {body} }}
"""


def test_zeropage_word_folds_to_high_byte_load():
    """The whole expression is one byte load — no shift, no mode switch."""
    asm = _main_asm(ZP_WORD.format(body="R = (W >> 8) as u8;"))
    assert re.search(r"^\s*LDA \$21\b", asm, re.M)
    assert "XBA" not in asm
    assert "AND #$FF" not in asm


def test_no_frame_is_allocated():
    """Folding kills both intermediates, so the stack slots go too."""
    asm = _main_asm(ZP_WORD.format(body="R = (W >> 8) as u8;"))
    assert "Allocate frame" not in asm


def test_signed_word_folds():
    """Arithmetic or logical, the low byte of `x >> 8` is x's high byte."""
    asm = _main_asm("""
        #[zeropage(0x20)] static mut W: i16;
        #[zeropage(0x30)] static mut R: u8;
        #[entry] fn main() { R = (W >> 8) as u8; }
    """)
    assert re.search(r"^\s*LDA \$21\b", asm, re.M)


def test_indexed_element_keeps_its_index():
    """`TBL[Y] >> 8` reads the high byte of the *same* element."""
    asm = _main_asm("""
        #[ram] static mut TBL: [u16; 4];
        #[zeropage(0x30)] static mut R: u8;
        #[entry] fn main() { Y = 2; R = (TBL[Y] >> 8) as u8; }
    """)
    # TBL is the only RAM static, so it lands at $7E2000; the fold reads
    # the element's high byte through the same scaled index.
    assert re.search(r"^\s*LDA \$7E2001,X\b", asm, re.M)


def test_volatile_read_is_left_alone():
    """A hardware word read must keep touching both halves."""
    asm = _main_asm("""
        #[hw(0x4218)] static mut HWREG: u16;
        #[zeropage(0x30)] static mut R: u8;
        #[entry] fn main() { R = (HWREG >> 8) as u8; }
    """)
    assert re.search(r"^\s*LDA \$4218\b", asm, re.M)
    assert "XBA" in asm


def test_untruncated_shift_is_left_alone():
    """`W >> 8` kept as u16 is a real shift, not a byte extract."""
    asm = _main_asm("""
        #[zeropage(0x20)] static mut W: u16;
        #[zeropage(0x30)] static mut R: u16;
        #[entry] fn main() { R = W >> 8; }
    """)
    assert "XBA" in asm


def test_other_shift_amounts_are_left_alone():
    """Only 8 lands the high byte on a byte boundary."""
    asm = _main_asm(ZP_WORD.format(body="R = (W >> 4) as u8;"))
    assert not re.search(r"^\s*LDA \$21\b", asm, re.M)


def test_shared_word_load_survives_the_fold():
    """When the whole word is still wanted, keep its load and add the byte one."""
    asm = _main_asm("""
        #[zeropage(0x20)] static mut W: u16;
        #[zeropage(0x30)] static mut R: u8;
        #[zeropage(0x32)] static mut FULL: u16;
        #[entry] fn main() { R = (W >> 8) as u8; FULL = W; }
    """)
    assert re.search(r"^\s*LDA \$20\b", asm, re.M)   # word still loaded
    assert re.search(r"^\s*LDA \$21\b", asm, re.M)   # high byte folded
    assert "XBA" not in asm


def test_as_bool_is_left_alone():
    """`as bool` normalizes to 0/1 — it is not a byte copy."""
    asm = _main_asm("""
        #[zeropage(0x20)] static mut W: u16;
        #[zeropage(0x30)] static mut R: bool;
        #[entry] fn main() { R = (W >> 8) as bool; }
    """)
    assert not re.search(r"^\s*LDA \$21\b", asm, re.M)


def test_rom_array_element_addressed_through_its_label():
    """A ROM symbol is reached as `label+offset`, so the +1 lands there.

    `_high_byte_of` mirrors `_resolve_operand`'s three-way precedence —
    rom_label first, then an explicit address, then allocation+offset —
    and this is the branch that would silently rot if that ordering ever
    changed, since the other two are addressed numerically.
    """
    asm = _main_asm("""
        static TBL: [u16; 4] = [0x1122, 0x3344, 0x5566, 0x7788];
        #[zeropage(0x30)] static mut R: u8;
        #[entry] fn main() { Y = 2; R = (TBL[Y] >> 8) as u8; }
    """)
    assert re.search(r"^\s*LDA\.l __TBL_data\+1,X\b", asm, re.M)
    assert "XBA" not in asm


class TestAmbiguousAddressingIsDeclined:
    """`_high_byte_of` bumps one field, so it must know which one is live.

    `_resolve_operand` checks `rom_label` before `address`, so a location
    carrying both would be reached by label while the pass bumped the
    number — a silently wrong byte. No location the compiler builds today
    has both (measured over a whole classickong build: every ROM location
    has a label and no address, every addressed location has no label), so
    rather than carry an untestable branch the pass checks the assumption
    and declines. These build the impossible location by hand to pin that.
    """

    @staticmethod
    def _chain(source):
        """Load -> >>8 -> cast, the shape the pass folds."""
        from r65.compiler.hir.types import BasicTypeInfo
        from r65.compiler.mir.nodes import (
            BinaryOp, Immediate as MIRImm, Load, TypeConvert, VirtualRegister,
        )
        u16 = BasicTypeInfo('u16')
        u8 = BasicTypeInfo('u8')
        word = VirtualRegister(0, u16)
        shifted = VirtualRegister(1, u16)
        byte = VirtualRegister(2, u8)
        return [
            Load(dest=word, source=source, type_info=u16),
            BinaryOp(dest=shifted, left=word, right=MIRImm(8), op='>>', type_info=u16),
            TypeConvert(dest=byte, source=shifted, source_type=u16, target_type=u8),
        ]

    @staticmethod
    def _rom_symbol():
        class Sym:
            name = 'TBL'
            rom_label = '__TBL_data'
        return Sym()

    def _fold_count(self, source):
        from r65.compiler.optimize.byte_extract import ByteExtractOptimizer
        opt = ByteExtractOptimizer()
        instrs = self._chain(source)
        opt._rewrite_block(instrs, {0: 1, 1: 1, 2: 1})
        return opt.folded

    def test_label_only_folds(self):
        """The normal ROM shape — label, no address — still folds."""
        from r65.compiler.mir.nodes import MemoryLocation
        loc = MemoryLocation(storage_type='rom', address=None,
                             symbol=self._rom_symbol(), offset=0)
        assert self._fold_count(loc) == 1

    def test_address_only_folds(self):
        """The normal numeric shape still folds."""
        from r65.compiler.mir.nodes import MemoryLocation
        loc = MemoryLocation(storage_type='zeropage', address=0x20,
                             symbol=None, offset=0)
        assert self._fold_count(loc) == 1

    def test_label_and_address_together_is_declined(self):
        """Ambiguous: bail out rather than guess which field to bump."""
        from r65.compiler.mir.nodes import MemoryLocation
        loc = MemoryLocation(storage_type='rom', address=0x8000,
                             symbol=self._rom_symbol(), offset=0)
        assert self._fold_count(loc) == 0
