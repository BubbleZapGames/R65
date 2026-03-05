"""
End-to-end test for const math builtins.

Verifies that fixed_sin/fixed_cos etc. generate correct ROM data
when used inside const fn for LUT generation.
"""

import pytest
from r65.tests.e2e import E2ETest, ExpectedState


class TestConstMathE2E:
    """Test const math builtins produce correct ROM data at runtime."""

    @pytest.fixture
    def e2e(self):
        return E2ETest()

    def test_sin_cos_table(self, e2e):
        """Generate sin/cos values via const fn, store individual results."""
        result = e2e.run('''
        const SIN_0: i16 = fixed_sin(0, 4, 127);
        const SIN_1: i16 = fixed_sin(1, 4, 127);
        const COS_0: i16 = fixed_cos(0, 4, 127);
        const LERP_MID: i16 = fixed_lerp(0, 200, 5, 10);

        #[zeropage(0x10)]
        static mut R0: i16;
        #[zeropage(0x12)]
        static mut R1: i16;
        #[zeropage(0x14)]
        static mut R2: i16;
        #[zeropage(0x16)]
        static mut R3: i16;

        #[entry]
        fn main() {
            R0 = SIN_0;
            R1 = SIN_1;
            R2 = COS_0;
            R3 = LERP_MID;
        }
        ''', ExpectedState(
            memory={
                # SIN_0 = sin(0) = 0
                0x7E0010: [0x00, 0x00],
                # SIN_1 = sin(pi/2) = 127
                0x7E0012: [0x7F, 0x00],
                # COS_0 = cos(0) = 127
                0x7E0014: [0x7F, 0x00],
                # LERP_MID = lerp(0, 200, 5, 10) = 100
                0x7E0016: [0x64, 0x00],
            }
        ))

        assert result.success, f"Failures: {result.failures}"
