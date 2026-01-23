"""
End-to-end tests for 2x2 matrix multiplication.

Tests matrix multiplication using u8 arrays and validates the resulting matrix.
"""

import pytest
from pathlib import Path
from r65.tests.e2e import E2ETest, ExpectedState

# Absolute paths to stdlib files
# test_matrix.py is at r65/tests/e2e/test_matrix.py
# So we need 4 parents to get to R65 root: e2e -> tests -> r65 -> R65
STDLIB_DIR = Path(__file__).parent.parent.parent.parent / "stdlib"
SNESLIB_R65_PATH = STDLIB_DIR / "sneslib.r65"
MATH_R65_PATH = STDLIB_DIR / "math.r65"


class TestMatrixMultiplication:
    """Test 2x2 matrix multiplication operations."""

    @pytest.fixture
    def e2e(self):
        """Create E2ETest instance."""
        return E2ETest()

    def _make_matrix_source(self, mat_a: list, mat_b: list) -> str:
        """Generate R65 source for matrix multiplication test.

        Args:
            mat_a: 4-element list for matrix A [a00, a01, a10, a11]
            mat_b: 4-element list for matrix B [b00, b01, b10, b11]

        Returns:
            R65 source code string
        """
        return f'''
            // Include sneslib (provides hardware registers) then math library
            include!("{SNESLIB_R65_PATH}")
            include!("{MATH_R65_PATH}")

            // Input matrices stored in row-major order
            #[zeropage(0x10)]
            static mut MAT_A: [u8; 4] = [{mat_a[0]}, {mat_a[1]}, {mat_a[2]}, {mat_a[3]}];

            #[zeropage(0x14)]
            static mut MAT_B: [u8; 4] = [{mat_b[0]}, {mat_b[1]}, {mat_b[2]}, {mat_b[3]}];

            // Result matrix
            #[zeropage(0x18)]
            static mut RESULT: [u8; 4] = [0, 0, 0, 0];

            // 2x2 Matrix multiplication: RESULT = MAT_A * MAT_B
            // Uses global arrays directly (avoiding pointer params for now)
            //
            // Matrices stored as flat u8 arrays in row-major order:
            //   [[a, b], [c, d]] -> [a, b, c, d]
            //
            // Result:
            //   RESULT[0] = A00*B00 + A01*B10
            //   RESULT[1] = A00*B01 + A01*B11
            //   RESULT[2] = A10*B00 + A11*B10
            //   RESULT[3] = A10*B01 + A11*B11
            // Additional temps to work around register allocation issue

                        fn multiply_matrix() {{
                // RESULT[0] = MAT_A[0]*MAT_B[0] + MAT_A[1]*MAT_B[2]
                // mul8 returns (low, high) - we only need low byte
                let mut TEMP : u8;
                let mut DISCARD : u8;
                TEMP, DISCARD = mul8(MAT_A[0], MAT_B[0]);
                A, DISCARD = mul8(MAT_A[1], MAT_B[2]);
                RESULT[0] = A + TEMP;

                // RESULT[1] = MAT_A[0]*MAT_B[1] + MAT_A[1]*MAT_B[3]
                TEMP, DISCARD = mul8(MAT_A[0], MAT_B[1]);
                A, DISCARD = mul8(MAT_A[1], MAT_B[3]);
                RESULT[1] = A + TEMP;

                // RESULT[2] = MAT_A[2]*MAT_B[0] + MAT_A[3]*MAT_B[2]
                TEMP, DISCARD = mul8(MAT_A[2], MAT_B[0]);
                A, DISCARD = mul8(MAT_A[3], MAT_B[2]);
                RESULT[2] = A + TEMP;

                // RESULT[3] = MAT_A[2]*MAT_B[1] + MAT_A[3]*MAT_B[3]
                TEMP, DISCARD = mul8(MAT_A[2], MAT_B[1]);
                A, DISCARD = mul8(MAT_A[3], MAT_B[3]);
                RESULT[3] = A + TEMP;
            }}

                        #[entry]
            fn main() {{
                multiply_matrix();
            }}
        '''

    def test_matrix_2x2_multiply_identity(self, e2e):
        """Test multiplying a matrix by the identity matrix.

        A = [[2, 3], [4, 5]]
        I = [[1, 0], [0, 1]]
        A * I = [[2, 3], [4, 5]]
        """
        source = self._make_matrix_source(
            mat_a=[2, 3, 4, 5],
            mat_b=[1, 0, 0, 1]
        )
        result = e2e.run(source, ExpectedState(memory={
            0x7E0018: [2, 3, 4, 5]
        }))

        assert result.success, f"Failures: {result.failures}"

    def test_matrix_2x2_multiply_larger_values(self, e2e):
        """Test with larger values that still fit in u8.

        A = [[10, 5], [8, 12]]
        B = [[3, 2], [4, 6]]

        C[0,0] = 10*3 + 5*4 = 30 + 20 = 50
        C[0,1] = 10*2 + 5*6 = 20 + 30 = 50
        C[1,0] = 8*3 + 12*4 = 24 + 48 = 72
        C[1,1] = 8*2 + 12*6 = 16 + 72 = 88

        Result = [[50, 50], [72, 88]]
        """
        source = self._make_matrix_source(
            mat_a=[10, 5, 8, 12],
            mat_b=[3, 2, 4, 6]
        )
        result = e2e.run(source, ExpectedState(memory={
            0x7E0018: [50, 50, 72, 88]
        }))

        assert result.success, f"Failures: {result.failures}"
