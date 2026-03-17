# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Basic end-to-end tests for R65 compiler.

Tests simple programs to verify the compilation and execution pipeline.
"""

import pytest
from r65.tests.e2e import E2ETest, ExpectedState


class TestBasicOperations:
    """Test basic R65 operations compile and execute correctly."""

    @pytest.fixture
    def e2e(self):
        """Create E2ETest instance."""
        return E2ETest()

    def test_assign_accumulator(self, e2e):
        """Test simple accumulator assignment."""
        result = e2e.run('''
                        #[entry]
            fn main() {
                A = 0x42;
            }
        ''', ExpectedState(A=0x42))

        assert result.success, f"Failures: {result.failures}"

    def test_assign_x_register(self, e2e):
        """Test X register assignment."""
        result = e2e.run('''
                        #[entry]
            fn main() {
                X = 0x10;
            }
        ''', ExpectedState(X=0x10))

        assert result.success, f"Failures: {result.failures}"

    def test_assign_y_register(self, e2e):
        """Test Y register assignment."""
        result = e2e.run('''
                        #[entry]
            fn main() {
                Y = 0xFF;
            }
        ''', ExpectedState(Y=0xFF))

        assert result.success, f"Failures: {result.failures}"

    def test_assign_all_registers(self, e2e):
        """Test assignment to all registers."""
        result = e2e.run('''
                        #[entry]
            fn main() {
                A = 0x11;
                X = 0x22;
                Y = 0x33;
            }
        ''', ExpectedState(A=0x11, X=0x22, Y=0x33))

        assert result.success, f"Failures: {result.failures}"

    def test_register_copy(self, e2e):
        """Test copying between registers."""
        result = e2e.run('''
                        #[entry]
            fn main() {
                A = 0x55;
                X = A;
                Y = A;
            }
        ''', ExpectedState(A=0x55, X=0x55, Y=0x55))

        assert result.success, f"Failures: {result.failures}"


class TestArithmetic:
    """Test arithmetic operations."""

    @pytest.fixture
    def e2e(self):
        return E2ETest()

    def test_addition(self, e2e):
        """Test addition operation."""
        result = e2e.run('''
                        #[entry]
            fn main() {
                A = 10;
                A = A + 5;
            }
        ''', ExpectedState(A=15))

        assert result.success, f"Failures: {result.failures}"

    def test_subtraction(self, e2e):
        """Test subtraction operation."""
        result = e2e.run('''
                        #[entry]
            fn main() {
                A = 20;
                A = A - 8;
            }
        ''', ExpectedState(A=12))

        assert result.success, f"Failures: {result.failures}"

    def test_increment(self, e2e):
        """Test increment operation."""
        result = e2e.run('''
                        #[entry]
            fn main() {
                X = 0;
                X++;
                X++;
                X++;
            }
        ''', ExpectedState(X=3))

        assert result.success, f"Failures: {result.failures}"

    def test_decrement(self, e2e):
        """Test decrement operation."""
        result = e2e.run('''
                        #[entry]
            fn main() {
                Y = 10;
                Y--;
                Y--;
            }
        ''', ExpectedState(Y=8))

        assert result.success, f"Failures: {result.failures}"


class TestMemoryOperations:
    """Test memory read/write operations."""

    @pytest.fixture
    def e2e(self):
        return E2ETest()

    def test_zeropage_write(self, e2e):
        """Test writing to zero page memory."""
        result = e2e.run('''
            #[zeropage(0x10)]
            static mut VALUE: u8;

                        #[entry]
            fn main() {
                VALUE = 0xAB;
            }
        ''', ExpectedState(memory={0x7E0010: 0xAB}))

        assert result.success, f"Failures: {result.failures}"

    def test_zeropage_read(self, e2e):
        """Test reading from zero page memory."""
        result = e2e.run('''
            #[zeropage(0x10)]
            static mut VALUE: u8 = 0x99;

                        #[entry]
            fn main() {
                A = VALUE;
            }
        ''', ExpectedState(A=0x99))

        assert result.success, f"Failures: {result.failures}"


class TestFlags:
    """Test processor flag operations."""

    @pytest.fixture
    def e2e(self):
        return E2ETest()

    def test_zero_flag_set(self, e2e):
        """Test zero flag is set when result is zero."""
        result = e2e.run('''
                        #[entry]
            fn main() {
                A = 0;
            }
        ''', ExpectedState(A=0, flags={'Z': True}))

        assert result.success, f"Failures: {result.failures}"

    def test_zero_flag_clear(self, e2e):
        """Test zero flag is clear when result is non-zero."""
        result = e2e.run('''
                        #[entry]
            fn main() {
                A = 1;
            }
        ''', ExpectedState(A=1, flags={'Z': False}))

        assert result.success, f"Failures: {result.failures}"

    def test_negative_flag_set(self, e2e):
        """Test negative flag is set for values >= 0x80."""
        result = e2e.run('''
                        #[entry]
            fn main() {
                A = 0x80;
            }
        ''', ExpectedState(A=0x80, flags={'N': True}))

        assert result.success, f"Failures: {result.failures}"

    def test_negative_flag_clear(self, e2e):
        """Test negative flag is clear for values < 0x80."""
        result = e2e.run('''
                        #[entry]
            fn main() {
                A = 0x7F;
            }
        ''', ExpectedState(A=0x7F, flags={'N': False}))

        assert result.success, f"Failures: {result.failures}"


class Test16BitMode:
    """Test 16-bit mode operations."""

    @pytest.fixture
    def e2e(self):
        return E2ETest()

    def test_16bit_accumulator(self, e2e):
        """Test 16-bit accumulator assignment with automatic mode switching.

        In the new mode model, 16-bit values are loaded with temporary REP/SEP,
        then mode returns to m8. The full 16-bit value is in C accumulator but
        only low byte visible in m8 mode. This test verifies the low byte.
        """
        result = e2e.run('''
            #[entry]
            fn main() {
                A = 0x1234;
            }
        ''', ExpectedState(A=0x34))  # Low byte visible in m8 mode

        assert result.success, f"Failures: {result.failures}"

    def test_16bit_index_registers(self, e2e):
        """Test 16-bit index register assignment.

        X and Y are always in x16 mode (16-bit) by default.
        """
        result = e2e.run('''
            #[entry]
            fn main() {
                X = 0xABCD;
                Y = 0xEF01;
            }
        ''', ExpectedState(X=0xABCD, Y=0xEF01))

        assert result.success, f"Failures: {result.failures}"

    def test_16bit_addition(self, e2e):
        """Test 16-bit addition in A with automatic mode switching.

        16-bit operations on A are wrapped with REP/SEP. The result
        is in the full C accumulator but only low byte visible in m8.
        """
        result = e2e.run('''
            #[entry]
            fn main() {
                A = 0x1000;
                A = A + 0x0234;
            }
        ''', ExpectedState(A=0x34))  # Low byte of 0x1234

        assert result.success, f"Failures: {result.failures}"


class TestArrayOperations:
    """Test array operations including len()."""

    @pytest.fixture
    def e2e(self):
        return E2ETest()

    def test_array_len_small(self, e2e):
        """Test len() on a small array (no initializer)."""
        result = e2e.run('''
            #[ram]
            static mut BUFFER: [u8; 10];

                        #[entry]
            fn main() {
                A = BUFFER.len();
            }
        ''', ExpectedState(A=10))

        assert result.success, f"Failures: {result.failures}"

    def test_array_len_large(self, e2e):
        """Test len() on a larger array (no initializer).

        Direct register assignment `A = DATA.len()` assigns in current mode (m8),
        so only the low byte is visible (256 & 0xFF = 0).
        """
        result = e2e.run('''
            #[ram]
            static mut DATA: [u8; 256];

                        #[entry]
            fn main() {
                A = DATA.len();
            }
        ''', ExpectedState(A=0))  # Low byte of 256 in m8 mode

        assert result.success, f"Failures: {result.failures}"

    def test_array_len_large_explicit_u16(self, e2e):
        """Test len() with explicit u16 let binding preserves full 16-bit value.

        With `let x @ A : u16 = expr;`, the compiler keeps A in m16 mode
        since the binding type is u16 (persist_16bit_mode flag on Move instruction).
        """
        result = e2e.run('''
            #[ram]
            static mut DATA: [u8; 256];

                        #[entry]
            fn main() {
                let arrCount @ A : u16 = DATA.len();
            }
        ''', ExpectedState(A=256, flags={'M': False}))  # m16 mode, full value

        assert result.success, f"Failures: {result.failures}"

    def test_array_len_large_implicit_u16(self, e2e):
        """Test len() with implicit u16 let binding preserves full 16-bit value.

        Type inference infers u16 from DATA.len() return type,
        and the register binding keeps A in m16 mode (persist_16bit_mode).
        """
        result = e2e.run('''
            #[ram]
            static mut DATA: [u8; 256];

                        #[entry]
            fn main() {
                let arrCount @ A = DATA.len();
            }
        ''', ExpectedState(A=256, flags={'M': False}))  # m16 mode, full value

        assert result.success, f"Failures: {result.failures}"

    def test_array_len_in_expression(self, e2e):
        """Test len() used in arithmetic expression."""
        result = e2e.run('''
            #[ram]
            static mut ARR: [u8; 100];

                        #[entry]
            fn main() {
                A = ARR.len() - (50 as u16);
            }
        ''', ExpectedState(A=50))

        assert result.success, f"Failures: {result.failures}"


class TestStackArgumentDrift:
    """Test stack argument offset drift when pushing multiple arguments.

    Regression tests for bug where passing the same stack-relative variable
    as multiple arguments would use stale offsets after each push.
    """

    @pytest.fixture
    def e2e(self):
        return E2ETest()

    def test_same_variable_as_multiple_stack_args(self, e2e):
        """Test passing same variable as multiple stack arguments.

        This was a codegen bug where stack-relative source locations weren't
        adjusted for bytes already pushed by previous arguments.
        """
        result = e2e.run('''
            #[zeropage]
            static mut RESULT: u8;

            fn add_three(a: u8, b: u8, c: u8) -> u8 {
                A = a + b;
                A = A + c;
                return A;
            }

            #[entry]
            fn main() {
                let x: u8 = 10;
                // Pass x three times - tests stack offset drift
                RESULT = add_three(x, x, x);
                A = RESULT;
            }
        ''', ExpectedState(A=30))  # 10 + 10 + 10 = 30

        assert result.success, f"Failures: {result.failures}"

    def test_mixed_variable_sizes_stack_args(self, e2e):
        """Test passing variables of different sizes as stack arguments.

        Tests that offset adjustments account for different argument sizes.
        """
        result = e2e.run('''
            #[zeropage]
            static mut RESULT: u8;

            fn compute(a: u8, b: u16, c: u8) -> u8 {
                // Just return sum of u8 values
                A = a + c;
                return A;
            }

            #[entry]
            fn main() {
                let x: u8 = 5;
                let y: u16 = 100;
                // Mix of u8 and u16 arguments from stack variables
                RESULT = compute(x, y, x);
                A = RESULT;
            }
        ''', ExpectedState(A=10))  # 5 + 5 = 10

        assert result.success, f"Failures: {result.failures}"

    def test_variable_and_constant_stack_args(self, e2e):
        """Test mixing variable and constant stack arguments."""
        result = e2e.run('''
            #[zeropage]
            static mut RESULT: u8;

            fn add_pair(a: u8, b: u8) -> u8 {
                A = a + b;
                return A;
            }

            #[entry]
            fn main() {
                let x: u8 = 7;
                // Variable x followed by constant, then x again
                RESULT = add_pair(x, 3);
                A = RESULT;
            }
        ''', ExpectedState(A=10))  # 7 + 3 = 10

        assert result.success, f"Failures: {result.failures}"


class TestOffsetOf:
    """Test offset_of() built-in function end-to-end."""

    @pytest.fixture
    def e2e(self):
        return E2ETest()

    def test_offset_of_struct_field(self, e2e):
        """Test offset_of returns correct byte offset for struct field."""
        result = e2e.run('''
            struct Player { x: u8, y: u8, health: u16 }

            #[entry]
            fn main() {
                A = offset_of(Player, health);
            }
        ''', ExpectedState(A=2))

        assert result.success, f"Failures: {result.failures}"

    def test_offset_of_first_field(self, e2e):
        """Test offset_of first field is 0."""
        result = e2e.run('''
            struct Player { x: u8, y: u8, health: u16 }

            #[entry]
            fn main() {
                A = offset_of(Player, x);
            }
        ''', ExpectedState(A=0))

        assert result.success, f"Failures: {result.failures}"
