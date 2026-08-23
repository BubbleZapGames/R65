# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
A jump table must honour `match`'s first-match-wins rule.

`value_to_arm` is built by walking arms in order, so a value named by two
arms was landing on the *last* one -- the opposite of the language rule and
of what the compare-chain lowering does. Reachable through overlapping
ranges since ranges were first supported, and through the far more natural
`1 | 2` then `2 | 3` since or-patterns became table-eligible.

Sized past the dispatch cost model's break-even so these actually lower to a
table; a shorter match uses a chain, which was always correct.
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


_LUT_SRC = '''
    #[zeropage(0x10)]
    static mut OUT: u8;
    #[zeropage(0x11)]
    static mut K: u8;

    #[entry]
    fn main() {
        K = 2;
        let r: u8 = match K {
            1 | 2 => 0xA1,
            2 | 3 => 0xA2,
            4 => 0xA3,
            _ => 0xFF,
        };
        OUT = r;
    }
'''


class TestLookupTableHonoursFirstMatch:
    """The LookupTable path shares `value_to_arm` with the jump table.

    It is worth its own case because it is reached differently: a LUT is not
    subject to the dispatch cost model, so a *small* all-constant match uses
    one where a jump table would have lost to a compare chain. That makes it
    the more commonly reached of the two table lowerings.
    """

    def test_shared_value_takes_the_earlier_arm(self, e2e):
        result = e2e.run(_LUT_SRC, ExpectedState(memory={0x7E0010: 0xA1}))
        assert result.success, f"LUT overlap: {result.failures}"

    def test_this_really_is_the_lookup_table_path(self):
        from r65.compiler.main import compile_string
        asm = compile_string(_LUT_SRC, 'test.r65')
        assert '.DB' in asm, "expected a LookupTable (.DB data table)"
        assert 'JMP (' not in asm, "expected no jump-table dispatch"


class TestOverlappingPatternsPreferTheFirstArm:
    @pytest.mark.parametrize("first,second,label", [
        ("1 | 2", "2 | 3", "or-pattern"),
        ("1..=2", "2..=3", "range"),
    ])
    def test_shared_value_takes_the_earlier_arm(self, first, second, label, e2e):
        # 2 is named by both arms; the first one wins.
        result = e2e.run(_src(first, second, 2),
                         ExpectedState(memory={0x7E0010: 0xA1}))
        assert result.success, f"{label}: {result.failures}"

    @pytest.mark.parametrize("first,second,probe,want", [
        ("1 | 2", "2 | 3", 1, 0xA1),   # only the first arm names 1
        ("1 | 2", "2 | 3", 3, 0xA2),   # only the second names 3
        ("1..=2", "2..=3", 1, 0xA1),
        ("1..=2", "2..=3", 3, 0xA2),
    ])
    def test_unshared_values_are_unaffected(self, first, second, probe, want, e2e):
        result = e2e.run(_src(first, second, probe),
                         ExpectedState(memory={0x7E0010: want}))
        assert result.success, f"dispatch({probe}): {result.failures}"
