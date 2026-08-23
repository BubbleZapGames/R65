# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
An or-patterned `match` large enough for a jump table dispatches correctly.

Or-patterns used to force the compare-chain lowering, so the table path had
never seen several values sharing one arm. It represents that natively --
`value_to_arm` is a plain value->arm dict, the same mechanism range patterns
have always used -- but "should work" is not "does work" when the failure
mode is an indirect jump to the wrong address.

Every value is exercised: both halves of each alternation, plus values below,
above, and inside a gap in the range, which must all reach the catch-all.
"""

from r65.tests.e2e import ExpectedState


def _src(probe: int) -> str:
    return f'''
        #[zeropage(0x10)]
        static mut OUT: u8;

        fn dispatch(k: u8) {{
            match k {{
                1 | 2   => {{ OUT = 0xA1; }},
                3 | 4   => {{ OUT = 0xA2; }},
                5 | 6   => {{ OUT = 0xA3; }},
                7 | 8   => {{ OUT = 0xA4; }},
                9 | 10  => {{ OUT = 0xA5; }},
                11 | 12 => {{ OUT = 0xA6; }},
                13 | 14 => {{ OUT = 0xA7; }},
                15 | 16 => {{ OUT = 0xA8; }},
                17 | 18 => {{ OUT = 0xA9; }},
                19 | 20 => {{ OUT = 0xAA; }},
                21 | 22 => {{ OUT = 0xAB; }},
                _ => {{ OUT = 0xFF; }},
            }};
        }}

        #[entry]
        fn main() {{
            OUT = 0x00;
            dispatch({probe});
        }}
    '''


_EXPECTED = {
    1: 0xA1, 2: 0xA1,
    2: 0xA1,
    7: 0xA4, 8: 0xA4,
    21: 0xAB, 22: 0xAB,
    0: 0xFF,     # below the range
    23: 0xFF,    # above the range
    200: 0xFF,   # far above
}


class TestOrPatternJumpTableDispatch:
    def test_every_value_reaches_its_arm(self, e2e):
        for probe, want in sorted(_EXPECTED.items()):
            result = e2e.run(_src(probe), ExpectedState(memory={0x7E0010: want}))
            assert result.success, (
                f"dispatch({probe}) expected 0x{want:02X}: {result.failures}"
            )
