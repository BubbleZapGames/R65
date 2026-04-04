# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
End-to-end test for const fn array support.

Verifies that const fn can generate lookup tables at compile time
and the resulting ROM data is correct when accessed at runtime.
"""

from r65.tests.e2e import ExpectedState


class TestConstFnArrayE2E:
    """Test const fn array generation produces correct ROM data."""

    def test_fibonacci_table(self, e2e):
        """Const fn generates Fibonacci table, copies to zeropage at runtime."""
        result = e2e.run('''
        const fn fibonacci(n: u8) -> u8 {
            if n <= 1 { return n; }
            let mut a: u8 = 0;
            let mut b: u8 = 1;
            for i in 2..n+1 {
                let tmp: u8 = b;
                b = a + b;
                a = tmp;
            }
            return b;
        }

        const fn generate_fib_table() -> [u8; 12] {
            let mut table: [u8; 12] = [0; 12];
            let mut i: u8 = 0;
            while i < 12 {
                table[i] = fibonacci(i);
                i = i + 1;
            }
            return table;
        }

        static FIB_TABLE: [u8; 12] = generate_fib_table();

        #[zeropage(0x10)]
        static mut RESULT: [u8; 12] = [0; 12];

        #[entry]
        fn main() {
            for i in 0..12 {
                RESULT[i] = FIB_TABLE[i];
            }
        }
        ''', ExpectedState(
            memory={
                0x7E0010: [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
            }
        ))

        assert result.success, f"Failures: {result.failures}"
