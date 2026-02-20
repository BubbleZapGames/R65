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


class TestMatchExpression:
    """Test match expression compilation and runtime."""

    @pytest.fixture
    def e2e(self):
        return E2ETest()

    def test_match_basic(self, e2e):
        """Test match expression with multiple arms and wildcard."""
        result = e2e.run('''
            #[zeropage(0x10)]
            static mut R1: u8;
            #[zeropage(0x11)]
            static mut R2: u8;
            #[zeropage(0x12)]
            static mut R3: u8;

            fn classify(val @ A: u8) -> u8 {
                let result: u8 = match val {
                    0 => 10,
                    1 => 20,
                    _ => 30
                };
                return result;
            }

            #[entry]
            fn main() {
                R1 = classify(0);
                R2 = classify(1);
                R3 = classify(99);
            }
        ''', ExpectedState(memory={
            0x7E0010: 10,
            0x7E0011: 20,
            0x7E0012: 30,
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_match_as_initializer(self, e2e):
        """Test match used as let-binding initializer."""
        result = e2e.run('''
            #[zeropage(0x10)]
            static mut RESULT: u8;

            #[entry]
            fn main() {
                A = 2;
                let val: u8 = match A {
                    0 => 100,
                    1 => 200,
                    2 => 42,
                    _ => 0
                };
                RESULT = val;
            }
        ''', ExpectedState(memory={
            0x7E0010: 42,
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_match_with_constants(self, e2e):
        """Test match expression using named constants as patterns."""
        result = e2e.run('''
            const PLAYER: u8 = 1;
            const ENEMY: u8 = 2;
            const ITEM: u8 = 3;

            #[zeropage(0x10)]
            static mut R1: u8;
            #[zeropage(0x11)]
            static mut R2: u8;
            #[zeropage(0x12)]
            static mut R3: u8;
            #[zeropage(0x13)]
            static mut R4: u8;

            fn classify(id @ A: u8) -> u8 {
                let result: u8 = match id {
                    PLAYER => 10,
                    ENEMY => 20,
                    ITEM => 30,
                    _ => 0
                };
                return result;
            }

            #[entry]
            fn main() {
                R1 = classify(1);
                R2 = classify(2);
                R3 = classify(3);
                R4 = classify(99);
            }
        ''', ExpectedState(memory={
            0x7E0010: 10,
            0x7E0011: 20,
            0x7E0012: 30,
            0x7E0013: 0,
        }))
        assert result.success, f"Failures: {result.failures}"


class TestForLoops:
    """Test for loop compilation and runtime."""

    @pytest.fixture
    def e2e(self):
        return E2ETest()

    def test_for_loop_sum(self, e2e):
        """Test for loop summing: for i in 0..10 { SUM += 1; }."""
        result = e2e.run('''
            #[zeropage(0x10)]
            static mut SUM: u8;

            #[entry]
            fn main() {
                SUM = 0;
                for i in 0..10 {
                    SUM = SUM + 1;
                }
            }
        ''', ExpectedState(memory={
            0x7E0010: 10,
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_for_loop_array_fill(self, e2e):
        """Test for loop filling array: for i in 0..8 { BUF[i] = i as u8; }."""
        result = e2e.run('''
            #[zeropage(0x10)]
            static mut BUF: [u8; 8] = [0; 8];

            #[entry]
            fn main() {
                for i in 0..8 {
                    BUF[i] = i as u8;
                }
            }
        ''', ExpectedState(memory={
            0x7E0010: [0, 1, 2, 3, 4, 5, 6, 7],
        }))
        assert result.success, f"Failures: {result.failures}"


class TestInclusiveForLoops:
    """Test inclusive for loop (..) compilation and runtime."""

    @pytest.fixture
    def e2e(self):
        return E2ETest()

    def test_inclusive_for_loop_count(self, e2e):
        """Test inclusive for loop: for i in 0..=3 iterates 4 times."""
        result = e2e.run('''
            #[zeropage(0x10)]
            static mut COUNTER: u8;

            #[entry]
            fn main() {
                COUNTER = 0;
                for i in 0..=3 {
                    COUNTER = COUNTER + 1;
                }
            }
        ''', ExpectedState(memory={
            0x7E0010: 4,
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_inclusive_for_loop_sum(self, e2e):
        """Test inclusive for loop sum: 0+1+2+3+4+5 = 15."""
        result = e2e.run('''
            #[zeropage(0x10)]
            static mut SUM: u8;

            #[entry]
            fn main() {
                SUM = 0;
                for i in 0..=5 {
                    SUM = SUM + i as u8;
                }
            }
        ''', ExpectedState(memory={
            0x7E0010: 15,
        }))
        assert result.success, f"Failures: {result.failures}"


class TestLoopExpression:
    """Test loop expression with break values."""

    @pytest.fixture
    def e2e(self):
        return E2ETest()

    def test_loop_break_value(self, e2e):
        """Test loop expression: let x = loop { break 42; };"""
        result = e2e.run('''
            #[zeropage(0x10)]
            static mut RESULT: u8;

            #[entry]
            fn main() {
                let x: u8 = loop {
                    break 42;
                };
                RESULT = x;
            }
        ''', ExpectedState(memory={
            0x7E0010: 42,
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_loop_conditional_break(self, e2e):
        """Test loop expression with conditional break values."""
        result = e2e.run('''
            #[zeropage(0x10)]
            static mut RESULT: u8;
            #[zeropage(0x11)]
            static mut COUNTER: u8;

            #[entry]
            fn main() {
                COUNTER = 0;
                let x: u8 = loop {
                    COUNTER = COUNTER + 1;
                    if COUNTER == 3 {
                        break 30;
                    }
                    if COUNTER == 5 {
                        break 50;
                    }
                };
                RESULT = x;
            }
        ''', ExpectedState(memory={
            0x7E0010: 30,
            0x7E0011: 3,
        }))
        assert result.success, f"Failures: {result.failures}"


class TestRecursion:
    """Test recursive function calls with stack parameters.

    Uses tail-recursive (accumulator) style. See TestBinaryOpCallResult for
    non-tail-recursive patterns like `n + f(x)`.
    """

    @pytest.fixture
    def e2e(self):
        return E2ETest()

    def test_recursive_sum(self, e2e):
        """Test tail recursion: sum_acc(5, 0) = 5+4+3+2+1+0 = 15."""
        result = e2e.run('''
            #[zeropage(0x10)]
            static mut RESULT: u8;

            fn sum_acc(n: u8, acc: u8) -> u8 {
                if n == 0 { return acc; }
                return sum_acc(n - 1, acc + n);
            }

            #[entry]
            fn main() {
                RESULT = sum_acc(5, 0);
            }
        ''', ExpectedState(memory={
            0x7E0010: 15,
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_recursive_triangular(self, e2e):
        """Test recursion to depth 8: triangular number T(8) = 36."""
        result = e2e.run('''
            #[zeropage(0x10)]
            static mut RESULT: u8;

            fn triangular(n: u8, acc: u8) -> u8 {
                if n == 0 { return acc; }
                return triangular(n - 1, acc + n);
            }

            #[entry]
            fn main() {
                RESULT = triangular(8, 0);
            }
        ''', ExpectedState(memory={
            0x7E0010: 36,
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_recursive_countdown_to_zero(self, e2e):
        """Test recursion preserving separate accumulator: count(10,0) → acc=10."""
        result = e2e.run('''
            #[zeropage(0x10)]
            static mut RESULT: u8;

            fn count_down(depth: u8, acc: u8) -> u8 {
                if depth == 0 { return acc; }
                return count_down(depth - 1, acc + 1);
            }

            #[entry]
            fn main() {
                RESULT = count_down(10, 0);
            }
        ''', ExpectedState(memory={
            0x7E0010: 10,
        }))
        assert result.success, f"Failures: {result.failures}"


class TestBinaryOpCallResult:
    """Test binary operations where one operand is a call result.

    Verifies that the codegen doesn't clobber A (holding the call result)
    when loading the other operand for the binary operation.
    """

    @pytest.fixture
    def e2e(self):
        return E2ETest()

    def test_add_with_call_result(self, e2e):
        """Test expr + fn_call(): commutative swap path."""
        result = e2e.run('''
            #[zeropage(0x10)]
            static mut RESULT: u8;
            #[zeropage(0x20, register)]
            static mut SCRATCH0: u8;

            fn double(val @ A: u8) -> u8 {
                return val + val;
            }

            #[entry]
            fn main() {
                RESULT = 3 + double(5);
            }
        ''', ExpectedState(memory={
            0x7E0010: 13,   # 3 + double(5) = 3 + 10 = 13
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_subtract_with_call_result(self, e2e):
        """Test expr - fn_call(): non-commutative save path."""
        result = e2e.run('''
            #[zeropage(0x10)]
            static mut RESULT: u8;
            #[zeropage(0x20, register)]
            static mut SCRATCH0: u8;

            fn half(val @ A: u8) -> u8 {
                return val >> 1;
            }

            #[entry]
            fn main() {
                RESULT = 20 - half(6);
            }
        ''', ExpectedState(memory={
            0x7E0010: 17,   # 20 - half(6) = 20 - 3 = 17
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_non_tail_recursive_sum(self, e2e):
        """Test n + sum_to(n-1): the original bug pattern."""
        result = e2e.run('''
            #[zeropage(0x10)]
            static mut RESULT: u8;
            #[zeropage(0x20, register)]
            static mut SCRATCH0: u8;

            fn sum_to(n: u8) -> u8 {
                if n == 0 { return 0; }
                return n + sum_to(n - 1);
            }

            #[entry]
            fn main() {
                RESULT = sum_to(5);
            }
        ''', ExpectedState(memory={
            0x7E0010: 15,   # 5+4+3+2+1+0 = 15
        }))
        assert result.success, f"Failures: {result.failures}"


class TestShortCircuit:
    """Test short-circuit evaluation of && and ||."""

    @pytest.fixture
    def e2e(self):
        return E2ETest()

    def test_and_short_circuit(self, e2e):
        """Test && skips right side when left is false."""
        result = e2e.run('''
            #[zeropage(0x10)]
            static mut FLAG: u8;
            #[zeropage(0x11)]
            static mut FALSE_VAL: u8;

            fn side_effect() -> u8 {
                FLAG = 1;
                return 1;
            }

            #[entry]
            fn main() {
                FLAG = 0;
                FALSE_VAL = 0;
                if FALSE_VAL != 0 && side_effect() != 0 {
                    FLAG = 99;
                }
            }
        ''', ExpectedState(memory={
            0x7E0010: 0,
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_or_short_circuit(self, e2e):
        """Test || skips right side when left is true."""
        result = e2e.run('''
            #[zeropage(0x10)]
            static mut FLAG: u8;
            #[zeropage(0x11)]
            static mut TRUE_VAL: u8;

            fn side_effect() -> u8 {
                FLAG = 1;
                return 1;
            }

            #[entry]
            fn main() {
                FLAG = 0;
                TRUE_VAL = 1;
                if TRUE_VAL != 0 || side_effect() != 0 {
                    A = 1;
                }
            }
        ''', ExpectedState(memory={
            0x7E0010: 0,
        }))
        assert result.success, f"Failures: {result.failures}"
