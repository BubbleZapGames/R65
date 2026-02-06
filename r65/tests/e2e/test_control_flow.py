"""
End-to-end tests for control flow operations.

Tests while loops, nested if/else, enum comparisons, and boundary comparisons.
"""

import pytest
from r65.tests.e2e import E2ETest, ExpectedState


class TestWhileLoops:
    """Test while loop operations."""

    @pytest.fixture
    def e2e(self):
        return E2ETest()

    def test_while_loop_sum(self, e2e):
        """Test while loop with accumulation: sum 1..10 by 3s."""
        result = e2e.run('''
            #[zeropage(0x10)]
            static mut SUM: u8;
            #[zeropage(0x11)]
            static mut COUNT: u8;

            #[entry]
            fn main() {
                SUM = 0;
                COUNT = 0;
                while COUNT < 10 {
                    SUM = SUM + 3;
                    COUNT = COUNT + 1;
                }
            }
        ''', ExpectedState(memory={
            0x7E0010: 30,   # SUM = 3*10
            0x7E0011: 10,   # COUNT = 10
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_loop_with_break(self, e2e):
        """Test loop with break condition."""
        result = e2e.run('''
            #[entry]
            fn main() {
                X = 0;
                loop {
                    if X == 5 {
                        break;
                    }
                    X++;
                }
            }
        ''', ExpectedState(X=5))
        assert result.success, f"Failures: {result.failures}"

    def test_while_loop_decrement(self, e2e):
        """Test while loop counting down."""
        result = e2e.run('''
            #[zeropage(0x10)]
            static mut RESULT: u8;

            #[entry]
            fn main() {
                RESULT = 0;
                Y = 5;
                while Y != 0 {
                    RESULT = RESULT + 2;
                    Y--;
                }
                A = RESULT;
            }
        ''', ExpectedState(A=10, Y=0))
        assert result.success, f"Failures: {result.failures}"


class TestIfElse:
    """Test if/else branching."""

    @pytest.fixture
    def e2e(self):
        return E2ETest()

    def test_nested_if_else_chain(self, e2e):
        """Test multi-branch classify function."""
        result = e2e.run('''
            #[zeropage(0x10)]
            static mut R1: u8;
            #[zeropage(0x11)]
            static mut R2: u8;
            #[zeropage(0x12)]
            static mut R3: u8;
            #[zeropage(0x13)]
            static mut R4: u8;

            fn classify(val @ A: u8) -> u8 {
                if val < 10 {
                    return 1;
                } else if val < 50 {
                    return 2;
                } else if val < 100 {
                    return 3;
                } else {
                    return 4;
                }
            }

            #[entry]
            fn main() {
                R1 = classify(5);
                R2 = classify(25);
                R3 = classify(75);
                R4 = classify(200);
            }
        ''', ExpectedState(memory={
            0x7E0010: 1,
            0x7E0011: 2,
            0x7E0012: 3,
            0x7E0013: 4,
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_if_else_with_assignment(self, e2e):
        """Test if/else that assigns different values."""
        result = e2e.run('''
            #[zeropage(0x10)]
            static mut RESULT: u8;

            #[entry]
            fn main() {
                A = 50;
                if A > 100 {
                    RESULT = 1;
                } else {
                    RESULT = 0;
                }
                A = RESULT;
            }
        ''', ExpectedState(A=0))
        assert result.success, f"Failures: {result.failures}"


class TestEnumComparison:
    """Test enum variant comparisons."""

    @pytest.fixture
    def e2e(self):
        return E2ETest()

    def test_enum_comparison(self, e2e):
        """Test enum variants in if/else chain."""
        result = e2e.run('''
            enum Direction { North = 0, East = 1, South = 2, West = 3 }

            #[zeropage(0x10)]
            static mut DX: u8;
            #[zeropage(0x11)]
            static mut DY: u8;

            fn apply_direction(dir @ A: u8) {
                if dir == Direction::North as u8 {
                    DY = 1;
                } else if dir == Direction::East as u8 {
                    DX = 1;
                } else if dir == Direction::South as u8 {
                    DY = 255;
                } else {
                    DX = 255;
                }
            }

            #[entry]
            fn main() {
                DX = 0;
                DY = 0;
                apply_direction(Direction::East as u8);
            }
        ''', ExpectedState(memory={
            0x7E0010: 1,     # DX = 1 (East)
            0x7E0011: 0,     # DY = 0
        }))
        assert result.success, f"Failures: {result.failures}"


class TestUnsignedComparisons:
    """Test unsigned comparison edge cases."""

    @pytest.fixture
    def e2e(self):
        return E2ETest()

    def test_unsigned_comparison_boundary(self, e2e):
        """Test unsigned comparisons at byte boundaries."""
        result = e2e.run('''
            #[zeropage(0x10)]
            static mut R1: u8;
            #[zeropage(0x11)]
            static mut R2: u8;
            #[zeropage(0x12)]
            static mut R3: u8;

            #[entry]
            fn main() {
                // 0 < 255
                A = 0;
                if A < 255 {
                    R1 = 1;
                } else {
                    R1 = 0;
                }
                // 255 > 0
                A = 255;
                if A > 0 {
                    R2 = 1;
                } else {
                    R2 = 0;
                }
                // 128 > 127 (unsigned, not signed)
                A = 128;
                if A > 127 {
                    R3 = 1;
                } else {
                    R3 = 0;
                }
            }
        ''', ExpectedState(memory={
            0x7E0010: 1,
            0x7E0011: 1,
            0x7E0012: 1,
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_comparison_equal_values(self, e2e):
        """Test comparisons with equal values.

        Uses a zeropage variable so A clobbering doesn't affect later comparisons.
        """
        result = e2e.run('''
            #[zeropage(0x10)]
            static mut R1: u8;
            #[zeropage(0x11)]
            static mut R2: u8;
            #[zeropage(0x12)]
            static mut R3: u8;
            #[zeropage(0x13)]
            static mut VAL: u8;

            #[entry]
            fn main() {
                VAL = 42;
                // a == a
                A = VAL;
                if A == 42 {
                    R1 = 1;
                } else {
                    R1 = 0;
                }
                // !(a < a)
                A = VAL;
                if A < 42 {
                    R2 = 0;
                } else {
                    R2 = 1;
                }
                // !(a > a)
                A = VAL;
                if A > 42 {
                    R3 = 0;
                } else {
                    R3 = 1;
                }
            }
        ''', ExpectedState(memory={
            0x7E0010: 1,
            0x7E0011: 1,
            0x7E0012: 1,
        }))
        assert result.success, f"Failures: {result.failures}"
