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


_JUMP_ARMS = "\n".join(
    f"                {i} => {{ OUT = 0x{0xA0 + i:02X}; }}," for i in range(9)
)


@pytest.mark.parametrize("match_body,probe,want", [
    # LookupTable: the catch-all hides inside an or-pattern.
    ("0 => 0xA0, 1 => 0xA1, 2 => 0xA2, 3 | _ => 0xEE", 7, 0xEE),
    # LookupTable: an arm after the catch-all is unreachable.
    ("0 => 0xA0, 1 => 0xA1, 2 => 0xA2, _ => 0xEE, 3 => 0xA3", 3, 0xEE),
    # Jump table: statement bodies, catch-all inside an or-pattern.
    (f"\n{_JUMP_ARMS}\n                9 | _ => {{ OUT = 0xEE; }},", 20, 0xEE),
], ids=["lut-or-catchall", "lut-arm-after-catchall", "jump-table-or-catchall"])
def test_catchall_arm_is_the_table_default(match_body, probe, want, e2e):
    """The first catch-all arm -- `_`, a binding, or an or-pattern holding
    one -- takes every value no earlier arm names."""
    value_arm = "let r: u8 = match K { %s }; OUT = r;" % match_body
    stmt = "match K { %s };" % match_body
    body = stmt if "OUT =" in match_body else value_arm
    src = f'''
        #[zeropage(0x10)]
        static mut OUT: u8;
        #[zeropage(0x11)]
        static mut K: u8;

        #[entry]
        fn main() {{ K = {probe}; {body} }}
    '''
    result = e2e.run(src, ExpectedState(memory={0x7E0010: want}))
    assert result.success, f"match K={probe}: {result.failures}"
