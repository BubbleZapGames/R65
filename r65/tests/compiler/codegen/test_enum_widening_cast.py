# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""`Dir as u16` on a fieldless enum reached the "should not happen" branch.

`codegen/type_utils.convert_operand_widths` sized a type by *name* — anything
outside `('u8','i8','bool')` counted as two bytes — so every enum looked
16-bit. A one-byte discriminant widened to `u16` therefore looked same-size and
fell through `select_type_convert` to:

    codegen error: Unexpected type conversion: Dir to u16

The MIR lowerer never agreed with that: it sizes the operands with
`_get_type_size`, which reads the enum's declared width, so it emitted a
widening `TypeConvert` that codegen then refused to select. `as u8` escaped the
bug only because MIR sees 1 -> 1 there and emits a `Move`, so no `TypeConvert`
reaches codegen at all.

The width test is deliberately shared with `is_narrowing_convert`, which the
register allocator uses to predict this emitter, so the fix belongs in that one
function rather than at either caller.
"""

from r65.compiler.main import compile_string


NARROW = """
enum Dir { North, East, South }
static SPEEDS: [u16; 3] = [10, 20, 30];
#[zeropage(0x40)] static mut o: u16;
#[zeropage(0x44)] static mut d: Dir;
"""

WIDE = """
enum Wide { A = 0, B = 300 }
#[zeropage(0x40)] static mut o: u8;
#[zeropage(0x44)] static mut w: Wide;
"""


def test_one_byte_enum_widens_to_u16():
    asm = compile_string(NARROW + """
        #[entry]
        fn main() { d = Dir::East; o = d as u16; }
    """, "test.r65")
    assert "AND #$FF" in asm, asm          # zero-extension, not an error


def test_widened_enum_indexes_a_table():
    """The shape this unblocks: a per-variant table addressed by the enum."""
    asm = compile_string(NARROW + """
        fn look() { o = SPEEDS[d as u16]; }
        #[entry]
        fn main() { d = Dir::South; look(); }
    """, "test.r65")
    assert "__SPEEDS_data" in asm, asm


def test_two_byte_enum_still_narrows_to_u8():
    """A discriminant past 255 is two bytes, so this stays a narrowing — the
    branch the name-based test happened to get right."""
    asm = compile_string(WIDE + """
        #[entry]
        fn main() { w = Wide::B; o = w as u8; }
    """, "test.r65")
    assert "codegen error" not in asm
    assert "STA $40" in asm, asm
