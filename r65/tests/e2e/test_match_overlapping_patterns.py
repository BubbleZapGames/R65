# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
A jump table must honour `match`'s first-match-wins rule.

`value_to_arm` is built by walking arms in order, so a value named by two
arms was landing on the *last* one -- the opposite of the language rule, and
of what the compare-chain lowering does for the same source.

Sized past the pattern-count threshold so these actually lower to a table; a
shorter match uses a chain, which was always correct.
"""

import pytest

from r65.tests.e2e import ExpectedState


_TAIL = "\n".join(
    f"                {i} => {{ OUT = 0x{0xA0 + i:02X}; }}," for i in range(4, 13)
)


def _src(first: str, second: str, probe: int) -> str:
    return f'''
        #[zeropage(0x10)]
        static mut OUT: u8;

        fn dispatch(k: u8) {{
            match k {{
                {first} => {{ OUT = 0xA1; }},
                {second} => {{ OUT = 0xA2; }},
{_TAIL}
                _ => {{ OUT = 0xFF; }},
            }};
        }}

        #[entry]
        fn main() {{ dispatch({probe}); }}
    '''


class TestOverlappingPatternsPreferTheFirstArm:
    def test_shared_value_takes_the_earlier_arm(self, e2e):
        # 2 is covered by both arms; the first one wins.
        result = e2e.run(_src("1..=2", "2..=3", 2),
                         ExpectedState(memory={0x7E0010: 0xA1}))
        assert result.success, f"range overlap: {result.failures}"

    @pytest.mark.parametrize("probe,want", [
        (1, 0xA1),   # only the first arm covers 1
        (3, 0xA2),   # only the second covers 3
    ])
    def test_unshared_values_are_unaffected(self, probe, want, e2e):
        result = e2e.run(_src("1..=2", "2..=3", probe),
                         ExpectedState(memory={0x7E0010: want}))
        assert result.success, f"dispatch({probe}): {result.failures}"
