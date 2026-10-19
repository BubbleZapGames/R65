# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""A widened byte is loaded in m16 and masked, except from I/O.

`ByteWidener` (optimize/byte_widen.py) turns `SEP / LDA m / REP / AND
#$00FF` into `LDA m / AND #$00FF`, which also reads the byte after m. That
read is only safe from memory: a `#[hw]` register, or a pointer not known
to hold a static's address, keeps its one-byte m8 read. Values are pinned
in tests/e2e/test_byte_widen.py.
"""

from r65.compiler.main import compile_string


def _fn(src: str, name: str) -> list:
    asm = compile_string(src, 't.r65')
    start = asm.index(f"\n{name}:")
    end = asm.find("; ---", start)
    body = asm[start:end if end != -1 else None]
    return [l.split(';')[0].strip() for l in body.splitlines()
            if l.startswith('    ') and not l.strip().startswith(('.', ';'))
            and l.split(';')[0].strip()]


def _width_at(code: list, k: int) -> int:
    """The accumulator width code[k] runs in: the last SEP/REP before it
    (the function is entered in m8)."""
    for line in reversed(code[:k]):
        if line.startswith('SEP #$20'):
            return 8
        if line.startswith('REP #$20'):
            return 16
    return 8


def test_pointer_sum_loop_reads_words():
    code = _fn('''
        #[lowram] static mut BUF: [u8; 64];
        #[lowram] static mut OUT: u16;
        #[inline(always)]
        fn sum_bytes(p: *u8, n: u16) -> u16 {
            let mut s: u16 = 0;
            let mut i: u16 = 0;
            while i < n { s = s + (p[i] as u16); i = i + 1; }
            return s;
        }
        #[inline(never)]
        fn t() -> u16 { return sum_bytes(&BUF as *u8, 64); }
        #[entry]
        fn main() { OUT = t(); }
    ''', 't')
    load = next(k for k, l in enumerate(code) if l.startswith('LDA (') and l.endswith(',Y'))
    assert code[load + 1] == 'AND #$FF'
    loop = code[load - 1:next(k for k, l in enumerate(code) if l.startswith('BCC'))]
    assert not [l for l in loop if l.startswith(('SEP', 'REP'))]


def test_hw_register_is_read_one_byte():
    code = _fn('''
        #[hw(0x4218)] static JOY1L: u8;
        #[lowram] static mut OUT: u16;
        #[inline(never)]
        fn t() -> u16 { return JOY1L as u16; }
        #[entry]
        fn main() { OUT = t(); }
    ''', 't')
    load = code.index('LDA $4218')
    assert _width_at(code, load) == 8


def test_pointer_of_unknown_origin_is_read_one_byte():
    # 0x2139 is VMDATALREAD: reading it advances the VRAM address.
    code = _fn('''
        #[lowram] static mut OUT: u16;
        #[inline(never)]
        fn t() -> u16 {
            let p: *u8 = 0x2139 as *u8;
            return p[0] as u16;
        }
        #[entry]
        fn main() { OUT = t(); }
    ''', 't')
    load = next(k for k, l in enumerate(code) if l.startswith('LDA (') or l.startswith('LDA $2139'))
    assert _width_at(code, load) == 8
