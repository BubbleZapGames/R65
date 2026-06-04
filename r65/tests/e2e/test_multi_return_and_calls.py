# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
End-to-end tests for multi-return functions and nested calls.

Tests multi-return functions (return type lists, e.g. -> u8, u16), nested calls with spilling.
"""

from pathlib import Path
from r65.tests.e2e import ExpectedState

STDLIB_DIR = Path(__file__).parent.parent.parent.parent / "stdlib"
SNESLIB_PATH = STDLIB_DIR / "sneslib.r65"
MATH_PATH = STDLIB_DIR / "math.r65"


class TestMul8MultiReturn:
    """Test mul8 u16 return."""

    def test_mul8_small(self, e2e):
        """Test mul8 returning u16: 7*6=42."""
        source = f'''
            include!("{SNESLIB_PATH}")
            include!("{MATH_PATH}")

            #[zeropage(0x10)]
            static mut RESULT: [u8; 2];

            #[entry]
            fn main() {{
                let r: u16 = mul8(7, 6);
                RESULT[0] = r as u8;
                RESULT[1] = (r >> 8) as u8;
            }}
        '''
        result = e2e.run(source, ExpectedState(memory={
            0x7E0010: 42,   # lo = 42
            0x7E0011: 0,    # hi = 0
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_mul8_overflow(self, e2e):
        """Test mul8 with large result: 200*200=40000 (0x9C40)."""
        source = f'''
            include!("{SNESLIB_PATH}")
            include!("{MATH_PATH}")

            #[zeropage(0x10)]
            static mut RESULT: [u8; 2];

            #[entry]
            fn main() {{
                let r: u16 = mul8(200, 200);
                RESULT[0] = r as u8;
                RESULT[1] = (r >> 8) as u8;
            }}
        '''
        result = e2e.run(source, ExpectedState(memory={
            0x7E0010: 0x40,  # lo byte of 40000
            0x7E0011: 0x9C,  # hi byte of 40000
        }))
        assert result.success, f"Failures: {result.failures}"


class TestNestedCalls:
    """Test nested function calls."""

    def test_nested_calls(self, e2e):
        """Test double(add_one(5)) = 12."""
        result = e2e.run('''
            #[zeropage(0x02, register)]
            static mut SCRATCH0: u8;
            #[zeropage(0x04, register)]
            static mut SCRATCH1: u16;

            fn add_one(val @ A: u8) -> u8 {
                A = val + 1;
                return A;
            }

            fn double(val @ A: u8) -> u8 {
                A = val + val;
                return A;
            }

            #[entry]
            fn main() {
                A = add_one(5);
                A = double(A);
            }
        ''', ExpectedState(A=12))
        assert result.success, f"Failures: {result.failures}"

    def test_triple_nested(self, e2e):
        """Test inc(double(inc(3))) = 9."""
        result = e2e.run('''
            #[zeropage(0x02, register)]
            static mut SCRATCH0: u8;
            #[zeropage(0x04, register)]
            static mut SCRATCH1: u16;

            fn inc(val @ A: u8) -> u8 {
                A = val + 1;
                return A;
            }

            fn double(val @ A: u8) -> u8 {
                A = val + val;
                return A;
            }

            #[entry]
            fn main() {
                A = inc(3);
                A = double(A);
                A = inc(A);
            }
        ''', ExpectedState(A=9))
        assert result.success, f"Failures: {result.failures}"


class TestMultipleReturnPaths:
    """Test functions with multiple return paths."""

    def test_multiple_return_paths(self, e2e):
        """Test function with return in both if and else branches."""
        result = e2e.run('''
            #[zeropage(0x10)]
            static mut R1: u8;
            #[zeropage(0x11)]
            static mut R2: u8;

            fn max_val(a: u8, b: u8) -> u8 {
                if a > b {
                    return a;
                } else {
                    return b;
                }
            }

            #[entry]
            fn main() {
                R1 = max_val(10, 3);
                R2 = max_val(3, 10);
            }
        ''', ExpectedState(memory={
            0x7E0010: 10,
            0x7E0011: 10,
        }))
        assert result.success, f"Failures: {result.failures}"


class TestMixedParams:
    """Test mixed register and stack parameters."""

    def test_mixed_register_stack_params(self, e2e):
        """Test fn(a @ A: u8, b: u8) with mixed calling convention."""
        result = e2e.run('''
            #[zeropage(0x10)]
            static mut RESULT: u8;

            fn add_mixed(b: u8, a @ A: u8) -> u8 {
                A = a + b;
                return A;
            }

            #[entry]
            fn main() {
                RESULT = add_mixed(15, 27);
                A = RESULT;
            }
        ''', ExpectedState(A=42))
        assert result.success, f"Failures: {result.failures}"

    def test_call_preserving_x(self, e2e):
        """Test calling a function that preserves X."""
        result = e2e.run('''
            #[preserves(X)]
            fn process(val @ A: u8) -> u8 {
                X = 999;  // This gets saved/restored
                A = val + 1;
                return A;
            }

            #[entry]
            fn main() {
                X = 0x42;
                A = process(10);
            }
        ''', ExpectedState(A=11, X=0x42))
        assert result.success, f"Failures: {result.failures}"


class TestVariableBoundParams:
    """Test variable-bound parameter passing (param @ STATIC_VAR)."""

    def test_variable_bound_basic(self, e2e):
        """Test basic variable-bound parameter: fn process(temp @ TEMP: u8)."""
        result = e2e.run('''
            #[zeropage(0x10)]
            static mut TEMP: u8;
            #[zeropage(0x11)]
            static mut RESULT: u8;

            fn process(temp @ TEMP: u8) -> u8 {
                A = temp + 5;
                return A;
            }

            #[entry]
            fn main() {
                RESULT = process(10);
            }
        ''', ExpectedState(memory={
            0x7E0011: 15,
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_variable_bound_with_register(self, e2e):
        """Test mixing variable-bound and register params."""
        result = e2e.run('''
            #[zeropage(0x10)]
            static mut TEMP: u8;
            #[zeropage(0x11)]
            static mut RESULT: u8;

            fn compute(val @ TEMP: u8, factor @ A: u8) -> u8 {
                A = factor + val;
                return A;
            }

            #[entry]
            fn main() {
                RESULT = compute(10, 25);
            }
        ''', ExpectedState(memory={
            0x7E0011: 35,
        }))
        assert result.success, f"Failures: {result.failures}"


class TestBRegister:
    """Test B register parameter passing and (u8, u8) multi-returns."""

    def test_b_parameter_and_return(self, e2e):
        """Test B param passing and (u8,u8) return via A,B registers.

        Uses --cfg snes (always set by e2e framework) for XBA support.
        Verifies that B parameter is received correctly and can be
        returned alongside A using a `-> u8, u8` return type (the two u8
        values map to A and B automatically).
        """
        result = e2e.run('''
            #[zeropage(0x10)]
            static mut LO: u8;
            #[zeropage(0x11)]
            static mut HI: u8;

            fn add_and_keep(lo @ A: u8, hi @ B: u8) -> u8, u8 {
                A = lo + 1;
                return A, B;
            }

            #[entry]
            fn main() {
                let mut a_val: u8;
                let mut b_val: u8;
                a_val, b_val = add_and_keep(0x10, 0x55);
                LO = a_val;
                HI = b_val;
            }
        ''', ExpectedState(memory={
            0x7E0010: 0x11,
            0x7E0011: 0x55,
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_mixed_width_return(self, e2e):
        """A `-> u8, u16` return maps the u8 to A and the u16 to X."""
        result = e2e.run('''
            #[zeropage(0x10)]
            static mut LO: u8;
            #[zeropage(0x12)]
            static mut WORD: u16;

            fn split(seed @ A: u8) -> u8, u16 {
                A = seed + 1;
                X = 0x1234;
                return A, X;
            }

            #[entry]
            fn main() {
                let mut a_val: u8;
                let mut x_val: u16;
                a_val, x_val = split(0x20);
                LO = a_val;
                WORD = x_val;
            }
        ''', ExpectedState(memory={
            0x7E0010: 0x21,
            0x7E0012: 0x34,
            0x7E0013: 0x12,
        }))
        assert result.success, f"Failures: {result.failures}"
