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
            #[mode(m8, x8)]
            #[entry]
            fn main() {
                A = 0x42;
            }
        ''', ExpectedState(A=0x42))

        assert result.success, f"Failures: {result.failures}"

    def test_assign_x_register(self, e2e):
        """Test X register assignment."""
        result = e2e.run('''
            #[mode(m8, x8)]
            #[entry]
            fn main() {
                X = 0x10;
            }
        ''', ExpectedState(X=0x10))

        assert result.success, f"Failures: {result.failures}"

    def test_assign_y_register(self, e2e):
        """Test Y register assignment."""
        result = e2e.run('''
            #[mode(m8, x8)]
            #[entry]
            fn main() {
                Y = 0xFF;
            }
        ''', ExpectedState(Y=0xFF))

        assert result.success, f"Failures: {result.failures}"

    def test_assign_all_registers(self, e2e):
        """Test assignment to all registers."""
        result = e2e.run('''
            #[mode(m8, x8)]
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
            #[mode(m8, x8)]
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
            #[mode(m8, x8)]
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
            #[mode(m8, x8)]
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
            #[mode(m8, x8)]
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
            #[mode(m8, x8)]
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

            #[mode(m8, x8)]
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

            #[mode(m8, x8)]
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
            #[mode(m8, x8)]
            #[entry]
            fn main() {
                A = 0;
            }
        ''', ExpectedState(A=0, flags={'Z': True}))

        assert result.success, f"Failures: {result.failures}"

    def test_zero_flag_clear(self, e2e):
        """Test zero flag is clear when result is non-zero."""
        result = e2e.run('''
            #[mode(m8, x8)]
            #[entry]
            fn main() {
                A = 1;
            }
        ''', ExpectedState(A=1, flags={'Z': False}))

        assert result.success, f"Failures: {result.failures}"

    def test_negative_flag_set(self, e2e):
        """Test negative flag is set for values >= 0x80."""
        result = e2e.run('''
            #[mode(m8, x8)]
            #[entry]
            fn main() {
                A = 0x80;
            }
        ''', ExpectedState(A=0x80, flags={'N': True}))

        assert result.success, f"Failures: {result.failures}"

    def test_negative_flag_clear(self, e2e):
        """Test negative flag is clear for values < 0x80."""
        result = e2e.run('''
            #[mode(m8, x8)]
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

    @pytest.mark.xfail(reason="16-bit mode transition not working - CPU stays in 8-bit mode")
    def test_16bit_accumulator(self, e2e):
        """Test 16-bit accumulator assignment."""
        result = e2e.run('''
            #[mode(m16, x16)]
            #[entry]
            fn main() {
                A = 0x1234;
            }
        ''', ExpectedState(A=0x1234))

        assert result.success, f"Failures: {result.failures}"

    @pytest.mark.xfail(reason="16-bit mode transition not working - CPU stays in 8-bit mode")
    def test_16bit_index_registers(self, e2e):
        """Test 16-bit index register assignment."""
        result = e2e.run('''
            #[mode(m16, x16)]
            #[entry]
            fn main() {
                X = 0xABCD;
                Y = 0xEF01;
            }
        ''', ExpectedState(X=0xABCD, Y=0xEF01))

        assert result.success, f"Failures: {result.failures}"

    @pytest.mark.xfail(reason="16-bit mode transition not working - CPU stays in 8-bit mode")
    def test_16bit_addition(self, e2e):
        """Test 16-bit addition."""
        result = e2e.run('''
            #[mode(m16, x16)]
            #[entry]
            fn main() {
                A = 0x1000;
                A = A + 0x0234;
            }
        ''', ExpectedState(A=0x1234))

        assert result.success, f"Failures: {result.failures}"
