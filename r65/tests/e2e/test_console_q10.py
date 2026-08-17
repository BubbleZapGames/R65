# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""Printing a `Q10` through the `Console` interface.

This is the whole chain end to end: `Console::print!` expands to `format!` into
`__console_fmt_buf`, `format!`'s `{s}` dispatches through `__fmt_str`, the type
checker resolves that to `Q10::to_string` *by name* — which is the only way a
newtype can take part, since it cannot implement `ToString` — and `Console::print`
then maps the resulting ASCII to nametable tile indices.

In `Buffer` mode each cell is a 2-byte little-endian entry
`(ch - 32 + tile_base) | tile_attr`, so with `init(0)` the low byte of cell *n*
decodes straight back to ASCII.
"""

from pathlib import Path
from r65.tests.e2e import ExpectedState

STDLIB_DIR = Path(__file__).parent.parent.parent.parent / "stdlib"
# console.r65 pulls in sneslib, math and string itself.
CONSOLE_PATH = STDLIB_DIR / "console.r65"
Q10_PATH = STDLIB_DIR / "Q10.r65"

PRELUDE = f'''
            include!("{CONSOLE_PATH}")
            include!("{Q10_PATH}")
'''

NAMETABLE = 0x7E3000


def program(body: str) -> str:
    return f'''{PRELUDE}
            #[ram(0x7E3000)]
            static mut NAMETABLE: [u8; 2048];
            #[ram]
            static mut CON: Console;

            #[entry]
            fn main() {{
                CON.init(0);
                CON.set_buffer(&NAMETABLE as far *u8);
                CON.set_area(0, 0, 32, 28);
{body}
            }}
    '''


def read_row(cpu, length, row=0, col=0):
    """Decode `length` cells of a nametable row back to ASCII."""
    out = []
    for i in range(length):
        cell = NAMETABLE + ((row * 32 + col + i) * 2)
        out.append(chr(cpu.memory.read(cell) + 32))
    return ''.join(out)


class TestConsolePrintQ10:
    """A Q10 reaching the screen through Console."""

    def test_print_a_whole_value(self, e2e):
        r = e2e.run(program('                let q: Q10 = Q10::from_int(100);\n'
                            '                CON.print!("{s}", q);'),
                    ExpectedState(memory={}))
        assert r.success, f"{r.error} {r.failures}"
        assert read_row(r.cpu, 6) == "100.00"

    def test_print_with_surrounding_text(self, e2e):
        r = e2e.run(program('                let q: Q10 = Q10::from(12, 48);\n'
                            '                CON.print!("x={s}!", q);'),
                    ExpectedState(memory={}))
        assert r.success, f"{r.error} {r.failures}"
        assert read_row(r.cpu, 8) == "x=12.75!"

    def test_print_a_negative(self, e2e):
        """The '-' is ASCII 0x2D, which maps to tile 13 — it must survive the
        ASCII-to-tile mapping like any other character."""
        r = e2e.run(program('                let q: Q10 = Q10::from_int(0 - 7);\n'
                            '                CON.print!("{s}", q);'),
                    ExpectedState(memory={}))
        assert r.success, f"{r.error} {r.failures}"
        assert read_row(r.cpu, 5) == "-7.00"

    def test_print_a_fraction(self, e2e):
        r = e2e.run(program('                let q: Q10 = Q10::from(0, 32);\n'
                            '                CON.print!("{s}", q);'),
                    ExpectedState(memory={}))
        assert r.success, f"{r.error} {r.failures}"
        assert read_row(r.cpu, 4) == "0.50"

    def test_print_at_a_position(self, e2e):
        """print_at! places the text at an arbitrary cell, so the decode offset
        moves with it."""
        r = e2e.run(program('                let q: Q10 = Q10::from(3, 16);\n'
                            '                CON.print_at!(4, 2, "{s}", q);'),
                    ExpectedState(memory={}))
        assert r.success, f"{r.error} {r.failures}"
        assert read_row(r.cpu, 4, row=2, col=4) == "3.25"

    def test_two_values_in_one_call(self, e2e):
        """Two `{s}` fragments in sequence — the second must pick up where the
        first left the cursor, and neither may truncate the other."""
        r = e2e.run(program('                let a: Q10 = Q10::from(1, 32);\n'
                            '                let b: Q10 = Q10::from(2, 16);\n'
                            '                CON.print!("{s}/{s}", a, b);'),
                    ExpectedState(memory={}))
        assert r.success, f"{r.error} {r.failures}"
        assert read_row(r.cpu, 9) == "1.50/2.25"

    def test_println_advances_the_row(self, e2e):
        r = e2e.run(program('                let q: Q10 = Q10::from_int(5);\n'
                            '                CON.println!("{s}", q);\n'
                            '                CON.print!("{s}", q);'),
                    ExpectedState(memory={}))
        assert r.success, f"{r.error} {r.failures}"
        assert read_row(r.cpu, 4, row=0) == "5.00"
        assert read_row(r.cpu, 4, row=1) == "5.00"
