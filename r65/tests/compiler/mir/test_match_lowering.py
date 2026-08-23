# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""Tests for match expression lowering to MIR.

Tests that the match lowerer correctly chooses between:
- LookupTable for dense pure-constant match expressions (inline ROM table)
- JumpTable for dense non-constant match expressions (JMP (addr,X))
- Branch chain (CondBranch) for sparse or small matches
"""

import pytest
from r65.compiler.typeck.errors import TypeCheckError
from r65.compiler.mir.nodes import JumpTable, LookupTable, CondBranch
from r65.tests.language.common import build_mir, get_mir_instructions, type_check


def has_jump_table(instrs) -> bool:
    """Check if instruction list contains a JumpTable node."""
    return any(isinstance(i, JumpTable) for i in instrs)


def has_lookup_table(instrs) -> bool:
    """Check if instruction list contains a LookupTable node."""
    return any(isinstance(i, LookupTable) for i in instrs)


def has_table_optimization(instrs) -> bool:
    """Check if instruction list uses any table optimization (JumpTable or LookupTable)."""
    return any(isinstance(i, (JumpTable, LookupTable)) for i in instrs)


def count_cond_branches(instrs) -> int:
    """Count CondBranch instructions."""
    return sum(1 for i in instrs if isinstance(i, CondBranch))


def get_jump_table(instrs) -> JumpTable:
    """Get the first JumpTable instruction, or None."""
    for i in instrs:
        if isinstance(i, JumpTable):
            return i
    return None


def get_lookup_table(instrs) -> LookupTable:
    """Get the first LookupTable instruction, or None."""
    for i in instrs:
        if isinstance(i, LookupTable):
            return i
    return None


class TestJumpTableSelection:
    """Tests that the analyzer picks table optimization vs branch chain correctly."""

    def test_dense_3_patterns_uses_table_optimization(self):
        """3+ dense consecutive patterns should emit a table optimization node."""
        source = """
        fn classify(val @ A: u8) -> u8 {
            let result: u8 = match val {
                0 => 10,
                1 => 20,
                2 => 30,
                _ => 0
            };
            return result;
        }
        """
        mir = build_mir(source)
        instrs = get_mir_instructions(mir, "classify")

        assert has_table_optimization(instrs), "Dense 3-pattern match should use table optimization"
        assert count_cond_branches(instrs) == 0, "Table path should have no CondBranch"

    def test_only_2_patterns_no_jump_table(self):
        """2 patterns (below MIN_PATTERNS=3) should use branch chain."""
        source = """
        fn classify(val @ A: u8) -> u8 {
            let result: u8 = match val {
                0 => 10,
                1 => 20,
                _ => 0
            };
            return result;
        }
        """
        mir = build_mir(source)
        instrs = get_mir_instructions(mir, "classify")

        assert not has_table_optimization(instrs), "2-pattern match should NOT use table optimization"
        assert count_cond_branches(instrs) >= 2, "Should use CondBranch chain"

    def test_sparse_patterns_no_jump_table(self):
        """Sparse patterns (density < 50%) should use branch chain."""
        source = """
        fn classify(val @ A: u8) -> u8 {
            let result: u8 = match val {
                0 => 10,
                50 => 20,
                100 => 30,
                _ => 0
            };
            return result;
        }
        """
        mir = build_mir(source)
        instrs = get_mir_instructions(mir, "classify")

        assert not has_table_optimization(instrs), "Sparse match should NOT use table optimization"
        assert count_cond_branches(instrs) >= 3, "Should use CondBranch chain"


class TestJumpTableProperties:
    """Tests for JumpTable node properties when non-constant arms force JumpTable."""

    def test_non_constant_arm_uses_jump_table(self):
        """Match with a non-constant arm body should use JumpTable, not LookupTable.

        Sized past the dispatch cost model's break-even so the choice under
        test is JumpTable-vs-LookupTable rather than table-vs-chain; a short
        match lowers to a compare chain regardless of arm-body constness.
        """
        source = """
        #[ram]
        static mut VALS: [u8; 12] = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120];

        fn classify(val @ A: u8) -> u8 {
            let result: u8 = match val {
                0 => VALS[0],
                1 => VALS[1],
                2 => VALS[2],
                3 => VALS[3],
                4 => VALS[4],
                5 => VALS[5],
                6 => VALS[6],
                7 => VALS[7],
                8 => VALS[8],
                9 => VALS[9],
                10 => VALS[10],
                11 => VALS[11],
                _ => 0
            };
            return result;
        }
        """
        mir = build_mir(source)
        instrs = get_mir_instructions(mir, "classify")

        assert has_jump_table(instrs), "Non-constant arm should use JumpTable"
        assert not has_lookup_table(instrs), "Non-constant arm should NOT use LookupTable"

    def test_gap_filled_with_default(self):
        """A gap in the range should be filled with the default value."""
        source = """
        fn classify(val @ A: u8) -> u8 {
            let result: u8 = match val {
                0 => 10,
                1 => 20,
                3 => 40,
                _ => 0
            };
            return result;
        }
        """
        mir = build_mir(source)
        instrs = get_mir_instructions(mir, "classify")

        # 3 patterns in range 0..3 → range_size=4, density=3/4=75% ≥ 50%
        # All constant → LookupTable
        lut = get_lookup_table(instrs)
        assert lut is not None, "3 patterns with 75% density and constants should use LookupTable"
        assert len(lut.values) == 4, f"Range 0..3 should have 4 entries, got {len(lut.values)}"
        # The gap at index 2 should have the default value
        assert lut.values[2] == lut.default_value, \
            "Gap entry should have the default value"


class TestLookupTableSelection:
    """Tests that pure constant matches use LookupTable optimization."""

    def test_pure_constant_match_uses_lookup_table(self):
        """Match with all-constant arms emits LookupTable (not JumpTable)."""
        source = """
        fn classify(val @ A: u8) -> u8 {
            let result: u8 = match val {
                0 => 10,
                1 => 20,
                2 => 30,
                _ => 0
            };
            return result;
        }
        """
        mir = build_mir(source)
        instrs = get_mir_instructions(mir, "classify")

        assert has_lookup_table(instrs), "Pure constant match should emit LookupTable"
        assert not has_jump_table(instrs), "Pure constant match should NOT emit JumpTable"

    def test_lookup_table_properties(self):
        """LookupTable has correct values, default_value, and base_value."""
        source = """
        fn classify(val @ A: u8) -> u8 {
            let result: u8 = match val {
                0 => 10,
                1 => 20,
                2 => 30,
                _ => 0
            };
            return result;
        }
        """
        mir = build_mir(source)
        instrs = get_mir_instructions(mir, "classify")

        lut = get_lookup_table(instrs)
        assert lut is not None
        assert lut.base_value == 0
        assert lut.values == [10, 20, 30]
        assert lut.default_value == 0

    def test_non_zero_base_lookup_table(self):
        """LookupTable with non-zero base has correct base_value."""
        source = """
        fn classify(val @ A: u8) -> u8 {
            let result: u8 = match val {
                5 => 10,
                6 => 20,
                7 => 30,
                _ => 0
            };
            return result;
        }
        """
        mir = build_mir(source)
        instrs = get_mir_instructions(mir, "classify")

        lut = get_lookup_table(instrs)
        assert lut is not None, "Should emit LookupTable for dense constants"
        assert lut.base_value == 5, f"Expected base_value=5, got {lut.base_value}"
        assert lut.values == [10, 20, 30]
        assert len(lut.values) == 3

    def test_enum_constant_bodies_use_lookup_table(self):
        """Enum match with constant bodies should use LookupTable."""
        source = """
        enum Dir { North = 0, East = 1, South = 2, West = 3 }

        fn dir_cost(d @ A: u8) -> u8 {
            let result: u8 = match d {
                Dir::North => 1,
                Dir::East => 2,
                Dir::South => 3,
                Dir::West => 4,
                _ => 0
            };
            return result;
        }
        """
        mir = build_mir(source)
        instrs = get_mir_instructions(mir, "dir_cost")

        lut = get_lookup_table(instrs)
        assert lut is not None, "Dense enum with constant bodies should emit LookupTable"
        assert lut.base_value == 0
        assert lut.values == [1, 2, 3, 4]
        assert lut.default_value == 0

    def test_non_constant_arm_falls_back_to_jump_table(self):
        """One non-constant arm body forces fallback to JumpTable.

        Sized past the dispatch cost model's break-even, so what is under
        test is the LookupTable-to-JumpTable fallback rather than the
        separate table-vs-chain decision.
        """
        source = """
        fn identity(x @ A: u8) -> u8 { return x; }

        fn classify(val @ A: u8) -> u8 {
            let result: u8 = match val {
                0 => 10,
                1 => identity(20),
                2 => 30,
                3 => 40,
                4 => 50,
                5 => 60,
                6 => 70,
                7 => 80,
                8 => 90,
                9 => 100,
                10 => 110,
                11 => 120,
                _ => 0
            };
            return result;
        }
        """
        mir = build_mir(source)
        instrs = get_mir_instructions(mir, "classify")

        assert has_jump_table(instrs), "Non-constant arm should force JumpTable"
        assert not has_lookup_table(instrs), "Non-constant arm should NOT use LookupTable"


class TestRangePatternLowering:
    """Tests for range pattern lowering."""

    def test_range_pattern_uses_branch_chain(self):
        """Sparse range pattern uses branch chain (two CondBranch per range)."""
        source = """
        fn classify(val @ A: u8) -> u8 {
            let result: u8 = match val {
                0..=1 => 1,
                _ => 0
            };
            return result;
        }
        """
        mir = build_mir(source)
        instrs = get_mir_instructions(mir, "classify")

        # 0..=1 expands to 2 values, below MIN_PATTERNS=3, so uses branch chain
        # Range pattern emits two CondBranch instructions (>= and <=)
        assert count_cond_branches(instrs) == 2, \
            f"Range pattern should emit 2 CondBranch, got {count_cond_branches(instrs)}"
        assert not has_table_optimization(instrs), "Small range should not use table"

    def test_dense_range_uses_lookup_table(self):
        """Dense range pattern with constant bodies triggers LookupTable."""
        source = """
        fn classify(val @ A: u8) -> u8 {
            let result: u8 = match val {
                0..=2 => 10,
                3..=5 => 20,
                6..=8 => 30,
                _ => 0
            };
            return result;
        }
        """
        mir = build_mir(source)
        instrs = get_mir_instructions(mir, "classify")

        # 9 values (0..=8) across 3 arms, all constant → LookupTable
        lut = get_lookup_table(instrs)
        assert lut is not None, "Dense range with constant bodies should use LookupTable"
        assert lut.base_value == 0
        assert lut.values == [10, 10, 10, 20, 20, 20, 30, 30, 30]

    def test_range_mixed_with_literals_in_table(self):
        """Range patterns mixed with literal patterns trigger table optimization."""
        source = """
        fn classify(val @ A: u8) -> u8 {
            let result: u8 = match val {
                0 => 10,
                1..=3 => 20,
                4 => 30,
                _ => 0
            };
            return result;
        }
        """
        mir = build_mir(source)
        instrs = get_mir_instructions(mir, "classify")

        lut = get_lookup_table(instrs)
        assert lut is not None, "Dense mixed range+literal should use LookupTable"
        assert lut.values == [10, 20, 20, 20, 30]

    def test_single_value_range_optimizes_to_equality(self):
        """Single-value range (5..=5) should emit one CondBranch (equality)."""
        source = """
        fn classify(val @ A: u8) -> u8 {
            let result: u8 = match val {
                5..=5 => 1,
                _ => 0
            };
            return result;
        }
        """
        mir = build_mir(source)
        instrs = get_mir_instructions(mir, "classify")

        # Single-value range should emit one == CondBranch, not two >= <=
        assert count_cond_branches(instrs) == 1, \
            f"Single-value range should emit 1 CondBranch, got {count_cond_branches(instrs)}"

    def test_exclusive_range_pattern(self):
        """Exclusive range 0..3 covers values 0, 1, 2."""
        source = """
        fn classify(val @ A: u8) -> u8 {
            let result: u8 = match val {
                0..3 => 10,
                3..6 => 20,
                6..9 => 30,
                _ => 0
            };
            return result;
        }
        """
        mir = build_mir(source)
        instrs = get_mir_instructions(mir, "classify")

        lut = get_lookup_table(instrs)
        assert lut is not None, "Dense exclusive range should use LookupTable"
        assert lut.values == [10, 10, 10, 20, 20, 20, 30, 30, 30]



class TestRangePatternTypeCheck:
    """Tests for range pattern type checking errors."""

    def test_range_on_bool_scrutinee_errors(self):
        """Range pattern on bool scrutinee should produce type error."""
        source = """
        fn test(val @ A: u8) -> u8 {
            let b: bool = val == 0;
            let result: u8 = match b {
                0..=1 => 1,
                _ => 0
            };
            return result;
        }
        """
        with pytest.raises(TypeCheckError, match="Cannot use range pattern"):
            type_check(source)

    def test_empty_exclusive_range_errors(self):
        """Empty exclusive range (5..5) should produce type error."""
        source = """
        fn test(val @ A: u8) -> u8 {
            let result: u8 = match val {
                5..5 => 1,
                _ => 0
            };
            return result;
        }
        """
        with pytest.raises(TypeCheckError, match="Empty range"):
            type_check(source)

    def test_empty_inclusive_range_errors(self):
        """Inverted inclusive range (5..=3) should produce type error."""
        source = """
        fn test(val @ A: u8) -> u8 {
            let result: u8 = match val {
                5..=3 => 1,
                _ => 0
            };
            return result;
        }
        """
        with pytest.raises(TypeCheckError, match="Empty range"):
            type_check(source)
