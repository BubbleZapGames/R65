"""
End-to-end tests for 2x2 matrix multiplication.

Tests matrix multiplication using u8 arrays and validates the resulting matrix.
"""

import pytest
from pathlib import Path
from r65.tests.e2e import E2ETest, ExpectedState

# Absolute path to stdlib/math.r65
# test_matrix.py is at r65/tests/e2e/test_matrix.py
# So we need 4 parents to get to R65 root: e2e -> tests -> r65 -> R65
MATH_R65_PATH = Path(__file__).parent.parent.parent.parent / "stdlib" / "math.r65"


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
            // Include math library for mul8 function
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
            #[zeropage(0x1D)]
            static mut TEMP_A: u8 = 0;
            #[zeropage(0x1E)]
            static mut TEMP_X: u8 = 0;

            #[mode(m8, x8)]
            fn multiply_matrix() {{
                // RESULT[0] = MAT_A[0]*MAT_B[0] + MAT_A[1]*MAT_B[2]
                let mut TEMP : u8; 
                (TEMP) = mul8(MAT_A[0], MAT_B[0]);
                (A) = mul8(MAT_A[1], MAT_B[2]);
                RESULT[0] = A + TEMP;

                // RESULT[1] = MAT_A[0]*MAT_B[1] + MAT_A[1]*MAT_B[3]
                (TEMP) = mul8(MAT_A[0], MAT_B[1]);
                (A) = mul8(MAT_A[1], MAT_B[3]);
                RESULT[1] = A + TEMP;

                // RESULT[2] = MAT_A[2]*MAT_B[0] + MAT_A[3]*MAT_B[2]
                (TEMP) = mul8(MAT_A[2], MAT_B[0]);
                (A) = mul8(MAT_A[3], MAT_B[2]);
                RESULT[2] = A + TEMP;

                // RESULT[3] = MAT_A[2]*MAT_B[1] + MAT_A[3]*MAT_B[3]
                (TEMP) = mul8(MAT_A[2], MAT_B[1]);
                (A) = mul8(MAT_A[3], MAT_B[3]);
                RESULT[3] = A + TEMP;
            }}

            #[mode(m8, x8)]
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

    def test_matrix_2x2_multiply_simple(self, e2e):
        """Test 2x2 matrix multiplication with simple values.

        A = [[1, 2], [3, 4]]
        B = [[5, 6], [7, 8]]

        C = A * B:
        C[0,0] = 1*5 + 2*7 = 5 + 14 = 19
        C[0,1] = 1*6 + 2*8 = 6 + 16 = 22
        C[1,0] = 3*5 + 4*7 = 15 + 28 = 43
        C[1,1] = 3*6 + 4*8 = 18 + 32 = 50

        Result = [[19, 22], [43, 50]]
        """
        source = self._make_matrix_source(
            mat_a=[1, 2, 3, 4],
            mat_b=[5, 6, 7, 8]
        )
        result = e2e.run(source, ExpectedState(memory={
            0x7E0018: [19, 22, 43, 50]
        }))

        assert result.success, f"Failures: {result.failures}"

    def test_matrix_2x2_multiply_zeros(self, e2e):
        """Test multiplying by zero matrix.

        A = [[3, 7], [2, 5]]
        B = [[0, 0], [0, 0]]
        A * B = [[0, 0], [0, 0]]
        """
        source = self._make_matrix_source(
            mat_a=[3, 7, 2, 5],
            mat_b=[0, 0, 0, 0]
        )
        result = e2e.run(source, ExpectedState(memory={
            0x7E0018: [0, 0, 0, 0]
        }))

        assert result.success, f"Failures: {result.failures}"

    def test_matrix_2x2_multiply_diagonal(self, e2e):
        """Test multiplying two diagonal matrices.

        A = [[2, 0], [0, 3]]
        B = [[4, 0], [0, 5]]
        A * B = [[8, 0], [0, 15]]
        """
        source = self._make_matrix_source(
            mat_a=[2, 0, 0, 3],
            mat_b=[4, 0, 0, 5]
        )
        result = e2e.run(source, ExpectedState(memory={
            0x7E0018: [8, 0, 0, 15]
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
