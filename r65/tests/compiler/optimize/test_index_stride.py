# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""A counter read only as an index scaled for 2- to 4-byte elements is
carried scaled (optimize/index_stride.py): `INX / INX; CPX #2N` instead of
rebuilding `TYA / ASL A / TAX` every pass. Values: tests/e2e/test_index_stride.py.
"""

from r65.compiler.main import compile_string


def _code(body: str, ret: str = 's') -> list:
    src = f'''
        struct E {{ pos: u16, hp: u8, kind: u8 }}
        #[lowram] static mut POOL: [E; 32];
        #[ram] static mut PTRS: [far *u8; 32];
        #[lowram] static mut W: [u16; 64];
        #[lowram] static mut V: [u16; 64];
        #[zeropage] static mut OUT: u16;
        #[inline(never)]
        fn t() -> u16 {{ let mut s: u16 = 0; {body} return {ret}; }}
        #[entry]
        fn main() {{ OUT = t(); }}
    '''
    asm = compile_string(src, 't.r65')
    start = asm.index("\nt:")
    end = asm.find("; ---", start)
    return [l.split(';')[0].strip() for l in asm[start:end].splitlines()
            if l.startswith('    ') and not l.strip().startswith(('.', ';'))
            and l.split(';')[0].strip()]


def test_counter_steps_by_two_in_x():
    code = _code('let mut i: u16 = 0; while i < 64 { s = s + W[i]; i = i + 1; }')
    assert 'ASL A' not in code, code
    bcc = next(k for k, l in enumerate(code) if l.startswith('BCC'))
    assert code[bcc - 3:bcc] == ['INX', 'INX', 'CPX #$80'], code


def test_one_double_serves_every_array():
    code = _code('let mut i: u16 = 0; while i < 64 { W[i] = s; V[i] = 7; i = i + 1; }')
    assert 'ASL A' not in code, code


def test_counter_read_after_the_loop_keeps_its_value():
    code = _code('let mut i: u16 = 0; while i < 64 { s = s + W[i]; i = i + 1; }',
                 ret='s + i')
    assert 'ASL A' in code, code


def test_bound_doubled_past_16_bits_is_left_alone():
    # 2 * 0x9000 wraps: `2i < 2N` would no longer be `i < N`. (Compiled only.)
    code = _code('let mut i: u16 = 0; while i < 0x9000 { s = s + W[i]; i = i + 1; }')
    assert 'ASL A' in code, code


def test_four_byte_struct_counter_steps_by_four_in_x():
    # The index is rebuilt in each block that reads POOL, the second time
    # under its own REP / SEP.
    code = _code('let mut i: u16 = 0; while i < 32 { POOL[i].pos = POOL[i].pos + 1; '
                 'if POOL[i].hp != 0 { POOL[i].kind = 3; } i = i + 1; }')
    assert 'ASL A' not in code, code
    cpx = code.index('CPX #$80')
    assert code[cpx - 4:cpx] == ['INX'] * 4, code


def test_counter_scaled_two_ways_is_left_alone():
    # `i << 1` for W and `i << 2` for POOL: one register cannot carry both.
    code = _code('let mut i: u16 = 0; while i < 32 { s = s + W[i]; POOL[i].hp = 1; i = i + 1; }')
    assert 'ASL A' in code, code


def test_far_pointer_counter_steps_by_three_in_x():
    # `i * 3` is `(i << 1) + i`: two reads of the counter, both scaling.
    code = _code('let mut i: u16 = 0; while i < 32 { let p: far *u8 = PTRS[i]; '
                 's = s + (*p as u16); i = i + 1; }')
    assert 'ASL A' not in code, code
    cpx = code.index('CPX #$60')
    assert code[cpx - 3:cpx] == ['INX'] * 3, code


def test_counter_also_read_unscaled_is_left_alone():
    # `(i << 1) + i` for PTRS, `i` itself for the sum.
    code = _code('let mut i: u16 = 0; while i < 32 { let p: far *u8 = PTRS[i]; '
                 's = s + (*p as u16) + i; i = i + 1; }')
    assert 'ASL A' in code, code
