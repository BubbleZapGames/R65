# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""`m = m ± 1` on a static or plane folds into INC/DEC on memory.

The MIR fold (optimize/memory_incdec.py) turns `Load m; m ± 1; Store m` into
one read-modify-write, so the emitted code is the hand-written
`DEC heat,X` instead of `LDA heat,X; SEC; SBC #1; STA heat,X`. A following
`m == 0` / `m != 0` branch reuses the Z flag INC/DEC set.
"""

import pytest

from r65.compiler.main import compile_string


def _fn(asm: str, name: str) -> list:
    start = asm.index(f"\n{name}:")
    end = asm.find("; ---", start)
    body = asm[start:end if end != -1 else None]
    return [l.split(';')[0].strip() for l in body.splitlines()
            if l.startswith('    ') and not l.strip().startswith(('.', ';'))
            and l.split(';')[0].strip()]


def _compile(src: str) -> str:
    # `t` is small and called once; keep it out of line so it can be inspected.
    src = src.replace("fn t()", "#[inline(never)]\nfn t()")
    return compile_string(src + "\n#[entry]\nfn main() { t(); }\n", "t.r65")


SOA = '''
    #[soa] struct Spark { heat: u8 }
    #[soa] struct Flare { pos: u16 }
    enum Fx { Off, S(Spark), F(Flare) }
    #[soa] struct Particle { x: u8, fx: Fx }
    #[lowram] static mut PARTS: [Particle; 8];
    #[zeropage] static mut OUT: u8;
'''


class TestFolded:

    def test_lowram_static_decrement(self):
        code = _fn(_compile('''
            #[lowram] static mut LIVES: u8;
            fn t() { LIVES = LIVES - 1; }'''), "t")
        assert any(c.startswith("DEC $") for c in code), code
        assert not any(c.startswith(("SBC", "STA")) for c in code), code

    def test_zeropage_increment(self):
        code = _fn(_compile('''
            #[zeropage(0x40)] static mut C: u8;
            fn t() { C++; }'''), "t")
        assert "INC $40" in code, code

    def test_plane_decrement_and_zero_branch(self):
        code = _fn(_compile(SOA + '''
            fn t() {
                for i in 0..8 {
                    match PARTS[i].fx {
                        Fx::S(s) => { s.heat = s.heat - 1; if s.heat == 0 { OUT = 1; } }
                        _ => {}
                    };
                }
            }'''), "t")
        dec = [n for n, c in enumerate(code) if c.startswith("DEC $") and c.endswith(",X")]
        assert dec, code
        # The Z flag from DEC drives the branch: no reload, no compare.
        nxt = code[dec[0] + 1]
        assert nxt.startswith(("BEQ", "BNE")), code

    def test_u16_plane_increment(self):
        code = _fn(_compile(SOA + '''
            fn t() {
                for i in 0..8 {
                    match PARTS[i].fx { Fx::F(f) => { f.pos = f.pos + 1; } _ => {} };
                }
            }'''), "t")
        inc = [n for n, c in enumerate(code) if c.startswith("INC $") and c.endswith(",X")]
        assert inc, code
        assert "REP #$20" in code[:inc[0]], code     # a 16-bit INC

    def test_constant_index_folds_into_the_address(self):
        code = _fn(_compile(SOA + '''
            fn t() {
                match PARTS[1].fx { Fx::F(f) => { f.pos = f.pos + 1; } _ => {} };
            }'''), "t")
        assert any(c.startswith("INC $") and not c.endswith(",X") for c in code), code


class TestNotFolded:

    @pytest.mark.parametrize("decl", [
        "#[hw(0x4200)] static mut REG: u8;",          # volatile: one write, exactly
        "#[ram] static mut REG: u8;",                  # long addressing: no INC long
    ])
    def test_storage_without_an_inc_form(self, decl):
        code = _fn(_compile(decl + "\nfn t() { REG = REG - 1; }"), "t")
        assert not any(c.startswith("DEC $") for c in code), code

    def test_value_still_wanted_in_a(self):
        """The decremented value is also stored elsewhere, so it must exist in A."""
        code = _fn(_compile('''
            #[lowram] static mut N: u8;
            #[lowram] static mut COPY: u8;
            fn t() { let v: u8 = N - 1; N = v; COPY = v; }'''), "t")
        assert not any(c.startswith("DEC $") for c in code), code


class TestShiftFolded:
    """`m << 1` and an unsigned `m >> 1` fold into ASL/LSR on memory."""

    def test_shifts_on_statics_and_planes(self):
        code = _fn(_compile('''
            #[zeropage(0x40)] static mut BY: u8;
            #[lowram] static mut W: u16;
            #[lowram] static mut ARR: [u8; 8];
            #[zeropage] static mut I: u16;
            fn t() { BY = BY << 1; W = W >> 1; ARR[I] = ARR[I] >> 1; }'''), "t")
        assert "ASL $40" in code, code
        assert any(c.startswith("LSR $") and not c.endswith(",X") for c in code), code
        assert any(c.startswith("LSR $") and c.endswith(",X") for c in code), code
        assert not any(c.startswith("STA") for c in code), code

    def test_signed_shift_right_keeps_its_sign(self):
        code = _fn(_compile('''
            #[zeropage(0x40)] static mut V: i8;
            fn t() { V = V >> 1; }'''), "t")
        assert not any(c.startswith("LSR $") for c in code), code
        assert "ROR A" in code, code
