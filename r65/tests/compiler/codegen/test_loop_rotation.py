# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""Counted loops are rotated to test at the bottom.

`LoopRotator` (optimize/loop_rotate.py) enters a loop whose first test
always passes at its body, so `LDY #0; L: CPY #N; BCS exit; body; INY;
BRA L` becomes `LDY #0; L: body; INY; CPY #N; BCC L`; a counter nothing
else reads counts down to zero instead. The values are pinned on hardware
in tests/e2e/test_loop_rotation.py.
"""

from r65.compiler.main import compile_string


def _fn(asm: str, name: str) -> list:
    start = asm.index(f"\n{name}:")
    end = asm.find("; ---", start)
    body = asm[start:end if end != -1 else None]
    return [l.split(';')[0].strip() for l in body.splitlines()
            if l.startswith('    ') and not l.strip().startswith(('.', ';'))
            and l.split(';')[0].strip()]


def _compile(src: str) -> list:
    src = src.replace("fn t()", "#[inline(never)]\nfn t()")
    return _fn(compile_string(src + "\n#[entry]\nfn main() { OUT = t(); }\n", "t.r65"), "t")


def _back_edge(code: list, opcode: str = 'BCC') -> int:
    return next(k for k, line in enumerate(code) if line.startswith(opcode + ' '))


class TestRotation:

    def test_m16_loop_tests_at_the_bottom_in_its_width(self):
        # CPY reads no M: the test runs in the m16 the body ends in, so the
        # loop has no mode switch and no BRA.
        code = _compile('''
            #[lowram] static mut DATA: [u16; 256];
            #[lowram] static mut OUT: u16;
            fn t() -> u16 {
                let mut s: u16 = 0;
                let mut i: u16 = 0;
                while i < 256 {
                    s = s + DATA[i];
                    i = i + 1;
                }
                return s;
            }''')
        bcc = _back_edge(code)
        assert code[bcc - 2:bcc] == ['INY', 'CPY #$0100']
        loop = code[code.index('LDY #$00') + 1:bcc]
        assert not [l for l in loop if l.startswith(('SEP', 'REP', 'BRA'))]

    def test_exit_switches_back_to_m8(self):
        code = _compile('''
            #[lowram] static mut DATA: [u16; 16];
            #[zeropage] static mut G: u8;
            #[lowram] static mut OUT: u8;
            fn t() -> u8 {
                let mut s: u16 = 0;
                let mut i: u16 = 0;
                while i < 16 {
                    s = s + DATA[i];
                    i = i + 1;
                }
                G = G + 3;
                return (s as u8) + G;
            }''')
        bcc = _back_edge(code)
        assert code[bcc + 1] == 'SEP #$20'

    def test_dead_counter_counts_down(self):
        code = _compile('''
            #[zeropage] static mut G: u8;
            #[lowram] static mut OUT: u16;
            fn t() -> u16 {
                let mut i: u16 = 0;
                while i < 8 {
                    G = G + 2;
                    i = i + 1;
                }
                return 5;
            }''')
        bne = _back_edge(code, 'BNE')
        assert code[0] == 'LDY #$08'
        assert code[bne - 1] == 'DEY'
        assert not [l for l in code if l.startswith('CPY')]

    def test_counter_read_after_the_loop_keeps_counting_up(self):
        # Counting down would leave i = 0 instead of 8.
        code = _compile('''
            #[zeropage] static mut G: u8;
            #[lowram] static mut OUT: u16;
            fn t() -> u16 {
                let mut i: u16 = 0;
                while i < 8 {
                    G = G + 2;
                    i = i + 1;
                }
                return i;
            }''')
        assert 'LDY #$00' in code
        assert 'CPY #$08' in code

    def test_unknown_start_is_guarded(self):
        # The first test may fail: the entry keeps a copy of it as a guard
        # past the loop, and the body tests at the bottom with no BRA.
        code = _compile('''
            #[zeropage] static mut G: u8;
            #[lowram] static mut OUT: u16;
            fn t() -> u16 {
                let mut i: u16 = G as u16;
                while i < 8 {
                    G = G + 2;
                    i = i + 1;
                }
                return i;
            }''')
        guard = code.index('BCS t__L3')
        assert code[guard - 1] == 'CPY #$08'
        bcc = _back_edge(code)
        assert code[bcc - 2:bcc] == ['INY', 'CPY #$08']
        assert not [l for l in code if l.startswith('BRA')]

    def test_while_nonzero_is_guarded(self):
        # popcount's inner loop: the guard tests the loaded byte, the
        # back edge the shifted one.
        code = _compile('''
            #[lowram] static mut BUF: [u8; 16];
            #[lowram] static mut OUT: u16;
            fn t() -> u16 {
                let mut c: u16 = 0;
                let mut i: u16 = 0;
                while i < 16 {
                    let mut v: u8 = BUF[i];
                    while v != 0 {
                        c = c + ((v & 1) as u16);
                        v = v >> 1;
                    }
                    i = i + 1;
                }
                return c;
            }''')
        assert [l for l in code if l.startswith('BEQ')]
        assert [l for l in code if l.startswith('BNE')]
        assert not [l for l in code if l.startswith('BRA')]

    def test_counter_stepped_in_an_inner_loop_keeps_counting_up(self):
        # y steps 3 times per pass: counting down from 4 would step past 0.
        code = _compile('''
            #[zeropage] static mut G: u8;
            #[lowram] static mut OUT: u16;
            fn t() -> u16 {
                let mut y: u16 = 0;
                while y < 4 {
                    let mut x: u16 = 0;
                    while x < 3 {
                        x = x + 1;
                        y = y + 1;
                        G = G + 1;
                    }
                }
                return 5;
            }''')
        assert 'CPY #$04' in code
