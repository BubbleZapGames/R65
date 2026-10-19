# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""A value carried around a loop stays in A (slot_allocator
`_find_a_loop_vregs`): `s = s + W[i]` is `CLC / ADC $0200,X` with no load or
store of `s`, when nothing it is live across touches A. Values:
tests/e2e/test_loop_carried_a.py.
"""

from r65.compiler.main import compile_string


def _code(body: str, ret: str = 's', params: str = '', sig_ret: str = 'u16') -> list:
    src = f'''
        #[lowram] static mut W: [u16; 64];
        #[lowram] static mut V: [u8; 64];
        #[zeropage] static mut OUT: u16;
        #[zeropage] static mut OUT8: u8;
        #[inline(never)] fn g() {{ OUT8 = 1; }}
        #[inline(never)]
        fn t({params}) -> {sig_ret} {{ {body} return {ret}; }}
        #[entry]
        fn main() {{ OUT = t({'5' if params else ''}) as u16; }}
    '''
    asm = compile_string(src, 't.r65')
    start = asm.index("\nt:")
    end = asm.find("; ---", start)
    return [l.split(';')[0].strip() for l in asm[start:end].splitlines()
            if l.startswith('    ') and not l.strip().startswith(('.', ';'))
            and l.split(';')[0].strip()]


_SUM = 'let mut s: u16 = 0; let mut i: u16 = 0; while i < 64 { s = s + W[i]; i = i + 1; }'


def test_running_sum_stays_in_a():
    code = _code(_SUM)
    assert code == ['REP #$20', 'LDA #$00', 'LDX #$00', 'CLC', 'ADC $0200,X',
                    'INX', 'INX', 'CPX #$80', 'BCC t__L2', 'RTS'], code


def test_byte_sum_stays_in_a():
    code = _code('let mut s: u8 = 0; let mut i: u16 = 0; while i < 64 { s = s + V[i]; i = i + 1; }',
                 sig_ret='u8')
    assert 'STA' not in ' '.join(code) and 'PHX' not in code, code


def test_store_of_the_value_is_an_sta_from_a():
    code = _code('let mut s: u16 = 0; let mut i: u16 = 0; '
                 'while i < 64 { s = s + W[i]; OUT = s; i = i + 1; }')
    assert [l for l in code if l.startswith(('LDA', 'STA'))] == ['LDA #$00', 'STA $00'], code


def test_call_in_the_loop_keeps_it_in_memory():
    code = _code('let mut s: u16 = 0; let mut i: u16 = 0; '
                 'while i < 64 { s = s + W[i]; g(); i = i + 1; }')
    assert 'STA $01,S' in code, code


def test_other_value_through_a_keeps_it_in_memory():
    code = _code('let mut s: u16 = 0; let mut i: u16 = 0; '
                 'while i < 64 { s = s + W[i]; OUT8 = V[i]; i = i + 1; }')
    assert 'STA $01,S' in code, code


def test_parameter_in_a_keeps_it_in_memory():
    # `k` arrives in A and is read after the loop.
    code = _code(_SUM, ret='s + (k as u16)', params='k @ A: u8')
    assert 'STA $01,S' in code, code
