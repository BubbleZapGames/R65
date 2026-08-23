# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
When a `match` becomes a jump table, and when it stays a compare chain.

A table pays its full dispatch (~33 cycles: bounds check, scale, indirect
jump) on every match; a chain pays only for the arms it walks (~9 for the
first, ~5 for each after, since the redundant-load pass proves A still holds
the scrutinee). So a table only wins for a match with many arms.

The old rule -- three patterns -- got this backwards. Nothing in
classickong.r65 reached it until or-patterns became table-eligible, at which
point `particle_process` measured 100 cycles/frame *slower*: its four
particle slots are usually `PART_TYPE_NONE`, which is the first arm.
"""

import re

from r65.compiler.hir.nodes import (
    HIRLiteralPattern, HIROrPattern, HIRWildcardPattern,
)
from r65.compiler.main import compile_string
from r65.compiler.mir.lowerers.match import MatchLowerer


def _arms(n: int) -> str:
    return '\n'.join(f'            {i} => {{ OUT = {i}; }},' for i in range(1, n + 1))


def _src(arms: str) -> str:
    return f'''
        #[zeropage(0x10)]
        static mut OUT: u8;

        fn dispatch(k: u8) {{
            match k {{
{arms}
                _ => {{ OUT = 0xFF; }},
            }};
        }}

        #[entry]
        fn main() {{ dispatch(2); }}
    '''


def _uses_jump_table(asm: str) -> bool:
    return 'Jump table dispatch' in asm


class TestJumpTableProfitability:
    def test_small_match_stays_a_chain(self):
        # Four arms: a chain walks ~2.5 of them for ~17 cycles, well under
        # the table's 33.
        asm = compile_string(_src(_arms(4)), 'test.r65')
        assert not _uses_jump_table(asm), "4-arm match should not use a jump table"

    def test_large_match_uses_a_table(self):
        # Twelve arms: a chain walks ~6.5 for ~37 cycles, so the table wins.
        asm = compile_string(_src(_arms(12)), 'test.r65')
        assert _uses_jump_table(asm), "12-arm match should use a jump table"


class TestOrPatternsAreTableEligible:
    def test_or_pattern_does_not_block_the_table(self):
        # Or-patterns are just several values sharing one arm, which the
        # table represents natively. Before, any or-pattern forced a chain
        # regardless of size.
        arms = '\n'.join(
            f'            {2*i+1} | {2*i+2} => {{ OUT = {i}; }},' for i in range(11)
        )
        asm = compile_string(_src(arms), 'test.r65')
        assert _uses_jump_table(asm), "or-patterned arms should be table-eligible"

        # Both values of each alternation must reach the same arm.
        table = re.findall(r'^\s*\.DW (\w+)$', asm, re.M)
        assert len(table) >= 22, f"expected >=22 table slots, got {len(table)}"
        for i in range(0, 22, 2):
            assert table[i] == table[i + 1], (
                f"slots {i} and {i+1} are one or-pattern arm but point to "
                f"{table[i]} and {table[i+1]}"
            )


class TestPatternValueExtraction:
    """Unit-level checks for `_pattern_integer_values`.

    The empty-alternation case cannot be written in R65 source -- the parser
    will not produce it -- so it is pinned here rather than end-to-end.
    """

    @staticmethod
    def _values(pattern):
        lowerer = MatchLowerer.__new__(MatchLowerer)
        return lowerer._pattern_integer_values(pattern)

    def test_or_pattern_collects_every_alternative(self):
        pattern = HIROrPattern(patterns=[
            HIRLiteralPattern(value=3), HIRLiteralPattern(value=7),
        ])
        assert self._values(pattern) == [3, 7]

    def test_empty_or_pattern_is_not_a_catch_all(self):
        # An empty list returned here would be read by the caller as
        # "matches anything", which is the opposite of what an alternation
        # with no alternatives means.
        assert self._values(HIROrPattern(patterns=[])) is None

    def test_catch_all_inside_an_alternation_swallows_the_arm(self):
        pattern = HIROrPattern(patterns=[
            HIRLiteralPattern(value=1), HIRWildcardPattern(),
        ])
        assert self._values(pattern) == []

    def test_bool_literal_is_not_table_indexable(self):
        # `bool` subclasses `int` in Python, so an isinstance(v, int) test
        # alone would silently treat `true` as the integer 1.
        assert self._values(HIRLiteralPattern(value=True)) is None
