# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Signed `>>` is an arithmetic shift (rounds toward negative infinity) and
signed `/` by a power of two rounds toward zero, as in Rust. Both used to
compile to LSR, so -7 >> 1 and -7 / 2 gave 124.
"""

import pytest

from r65.tests.e2e import ExpectedState


def _trunc_div(a, b):
    q = abs(a) // abs(b)
    return q if (a < 0) == (b < 0) else -q


def _bytes(value, width):
    return [(value >> (8 * i)) & 0xFF for i in range(width)]


CASES = [
    # (type, op, amount, values)
    ('i8', '>>', 1, [-128, -7, -1, 0, 7, 127]),
    ('i8', '>>', 7, [-128, -1, 0, 127]),
    ('i8', '/', 2, [-128, -7, -1, 0, 7, 127]),
    ('i8', '/', 128, [-128, -127, -1, 0, 127]),
    ('i8', '/', 256, [-128, -1, 0, 127]),
    ('i16', '>>', 4, [-32768, -1000, -1, 0, 1000]),
    ('i16', '>>', 8, [-32768, -257, -256, -1, 0, 32767]),
    ('i16', '>>', 12, [-32768, -1, 0, 32767]),
    ('i16', '/', 16, [-32768, -1000, -15, -1, 0, 1000]),
    ('i16', '/', 256, [-32768, -257, -255, -1, 0]),
]


@pytest.mark.parametrize("ty,op,amount,values", CASES,
                         ids=[f"{c[0]}{c[1]}{c[2]}" for c in CASES])
def test_signed_shift_and_divide(e2e, ty, op, amount, values):
    width = 1 if ty == 'i8' else 2
    body = []
    expected = []
    for k, v in enumerate(values):
        body.append(f"IN = {v}; OUT[{k}] = IN {op} {amount};")
        result = v >> amount if op == '>>' else _trunc_div(v, amount)
        expected += _bytes(result, width)
    source = f'''
        #[lowram(0x0300)]
        static mut IN: {ty};
        #[lowram(0x0400)]
        static mut OUT: [{ty}; 8];
        #[entry]
        fn main() {{
            {" ".join(body)}
        }}
    '''
    result = e2e.run(source, ExpectedState(memory={0x0400: expected}),
                     max_instructions=20000)
    assert result.success, f"Failures: {result.failures}"
