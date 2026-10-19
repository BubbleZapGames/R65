# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""A counter read only as a doubled u16 index is carried doubled
(optimize/index_stride.py): `INX / INX; CPX #2N` instead of rebuilding
`TYA / ASL A / TAX` every pass. Values: tests/e2e/test_index_stride.py.
"""

from r65.compiler.main import compile_string


def _code(body: str, ret: str = 's') -> list:
    src = f'''
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
