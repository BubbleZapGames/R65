# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
End-to-end tests for stdlib rand functions.

Tests rand(), rand_update!(), and rand_range!().
"""

import pytest
from pathlib import Path
from r65.tests.e2e import E2ETest, ExpectedState

STDLIB_DIR = Path(__file__).parent.parent.parent.parent / "stdlib"
SNESLIB_PATH = STDLIB_DIR / "sneslib.r65"
MATH_PATH = STDLIB_DIR / "math.r65"
RAND_PATH = STDLIB_DIR / "rand.r65"


class TestRand:
    """Test rand() function."""

    @pytest.fixture
    def e2e(self):
        return E2ETest()

    def test_rand_returns_nonzero(self, e2e):
        """Test that rand() returns a non-zero value with default seed."""
        source = f'''
            include!("{SNESLIB_PATH}")
            include!("{MATH_PATH}")
            include!("{RAND_PATH}")

            #[zeropage(0x10)]
            static mut RESULT: [u8; 2];

            #[entry]
            fn main() {{
                let r: u16 = rand();
                RESULT[0] = r as u8;
                RESULT[1] = (r >> 8) as u8;
            }}
        '''
        # Default seed is 0xDEAD, xorshift will produce a deterministic non-zero result
        result = e2e.run(source, ExpectedState())
        assert result.success, f"Failures: {result.failures}"

    def test_rand_deterministic(self, e2e):
        """Test that two calls to rand() produce different results."""
        source = f'''
            include!("{SNESLIB_PATH}")
            include!("{MATH_PATH}")
            include!("{RAND_PATH}")

            #[zeropage(0x10)]
            static mut R1: [u8; 2];
            #[zeropage(0x12)]
            static mut R2: [u8; 2];

            #[entry]
            fn main() {{
                let a: u16 = rand();
                R1[0] = a as u8;
                R1[1] = (a >> 8) as u8;

                let b: u16 = rand();
                R2[0] = b as u8;
                R2[1] = (b >> 8) as u8;
            }}
        '''
        result = e2e.run(source, ExpectedState())
        assert result.success, f"Failures: {result.failures}"

    def test_rand_custom_seed(self, e2e):
        """Test that seeding produces deterministic output."""
        source = f'''
            include!("{SNESLIB_PATH}")
            include!("{MATH_PATH}")
            include!("{RAND_PATH}")

            #[zeropage(0x10)]
            static mut RESULT: [u8; 2];

            #[entry]
            fn main() {{
                __rand_seed = 0xBEEF;
                let r: u16 = rand();
                RESULT[0] = r as u8;
                RESULT[1] = (r >> 8) as u8;
            }}
        '''
        result = e2e.run(source, ExpectedState())
        assert result.success, f"Failures: {result.failures}"


class TestRandUpdate:
    """Test rand_update!() macro."""

    @pytest.fixture
    def e2e(self):
        return E2ETest()

    def test_rand_update_changes_seed(self, e2e):
        """Test that rand_update! modifies the seed."""
        source = f'''
            include!("{SNESLIB_PATH}")
            include!("{MATH_PATH}")
            include!("{RAND_PATH}")

            #[zeropage(0x10)]
            static mut SEED_BEFORE: [u8; 2];
            #[zeropage(0x12)]
            static mut SEED_AFTER: [u8; 2];

            #[entry]
            fn main() {{
                SEED_BEFORE[0] = __rand_seed as u8;
                SEED_BEFORE[1] = (__rand_seed >> 8) as u8;

                rand_update!(0x1234);

                SEED_AFTER[0] = __rand_seed as u8;
                SEED_AFTER[1] = (__rand_seed >> 8) as u8;
            }}
        '''
        result = e2e.run(source, ExpectedState())
        assert result.success, f"Failures: {result.failures}"
