"""
End-to-end tests for mixed u8/u16 mode switching.

Stress tests the automatic REP/SEP mode switching for correctness and
efficiency when alternating between 8-bit and 16-bit operations.
"""

import pytest
from r65.tests.e2e import E2ETest, ExpectedState


class TestModeSwitchingBasic:
    """Basic mode switching between u8 and u16 operations."""

    @pytest.fixture
    def e2e(self):
        return E2ETest()

    def test_u8_then_u16_accumulator(self, e2e):
        """Test u8 operation followed by u16 operation."""
        result = e2e.run('''
            #[zeropage(0x10)]
            static mut RESULT8: u8;
            #[zeropage(0x12)]
            static mut RESULT16: u16;

            #[entry]
            fn main() {
                // 8-bit operation (m8 mode)
                A = 0x42;
                RESULT8 = A;

                // 16-bit operation (needs REP/SEP)
                let val @ A : u16 = 0x1234;
                RESULT16 = A;
            }
        ''', ExpectedState(
            A=0x1234,
            memory={0x7E0010: 0x42, 0x7E0012: [0x34, 0x12]},
            flags={'M': False}  # Should be in m16 mode
        ))
        assert result.success, f"Failures: {result.failures}"

    def test_u16_then_u8_accumulator(self, e2e):
        """Test u16 operation followed by u8 operation.

        After u16 let binding, A is in m16 mode. But when we do an explicit 8-bit
        operation (A = 0x55), the compiler correctly switches back to m8 mode.
        """
        result = e2e.run('''
            #[zeropage(0x10)]
            static mut RESULT16: u16;
            #[zeropage(0x12)]
            static mut RESULT8: u8;

            #[entry]
            fn main() {
                // 16-bit operation first
                let wide @ A : u16 = 0xABCD;
                RESULT16 = A;

                // 8-bit operation - switches back to m8 mode
                A = 0x55;
                RESULT8 = A;
            }
        ''', ExpectedState(
            A=0x55,
            memory={0x7E0010: [0xCD, 0xAB], 0x7E0012: 0x55},
            flags={'M': True}  # Back to m8 for 8-bit operation
        ))
        assert result.success, f"Failures: {result.failures}"

    def test_alternating_u8_u16_three_times(self, e2e):
        """Test alternating u8/u16 operations three times."""
        result = e2e.run('''
            #[zeropage(0x10)]
            static mut R1: u8;
            #[zeropage(0x12)]
            static mut R2: u16;
            #[zeropage(0x14)]
            static mut R3: u8;
            #[zeropage(0x16)]
            static mut R4: u16;
            #[zeropage(0x18)]
            static mut R5: u8;
            #[zeropage(0x1A)]
            static mut R6: u16;

            #[entry]
            fn main() {
                A = 0x11;           // u8
                R1 = A;

                let v1 @ A : u16 = 0x2222;  // u16
                R2 = A;

                A = 0x33;           // u8
                R3 = A;

                let v2 @ A : u16 = 0x4444;  // u16
                R4 = A;

                A = 0x55;           // u8
                R5 = A;

                let v3 @ A : u16 = 0x6666;  // u16
                R6 = A;
            }
        ''', ExpectedState(
            memory={
                0x7E0010: 0x11,
                0x7E0012: [0x22, 0x22],
                0x7E0014: 0x33,
                0x7E0016: [0x44, 0x44],
                0x7E0018: 0x55,
                0x7E001A: [0x66, 0x66],
            }
        ))
        assert result.success, f"Failures: {result.failures}"


class TestModeSwitchingArithmetic:
    """Mode switching with arithmetic operations."""

    @pytest.fixture
    def e2e(self):
        return E2ETest()

    def test_u8_add_then_u16_add(self, e2e):
        """Test 8-bit addition followed by 16-bit addition."""
        result = e2e.run('''
            #[zeropage(0x10)]
            static mut SUM8: u8;
            #[zeropage(0x12)]
            static mut SUM16: u16;

            #[entry]
            fn main() {
                // 8-bit addition
                A = 100;
                A = A + 50;
                SUM8 = A;  // Should be 150

                // 16-bit addition
                let wide @ A : u16 = 1000;
                A = A + 234;
                SUM16 = A;  // Should be 1234
            }
        ''', ExpectedState(
            memory={0x7E0010: 150, 0x7E0012: [0xD2, 0x04]}  # 1234 = 0x04D2
        ))
        assert result.success, f"Failures: {result.failures}"

    def test_mixed_arithmetic_chain(self, e2e):
        """Test chain of mixed 8-bit and 16-bit arithmetic."""
        result = e2e.run('''
            #[zeropage(0x10)]
            static mut R1: u8;
            #[zeropage(0x12)]
            static mut R2: u16;
            #[zeropage(0x14)]
            static mut R3: u8;

            #[entry]
            fn main() {
                // Start with 8-bit
                A = 10;
                A = A + 5;      // 15
                R1 = A;

                // Switch to 16-bit
                let w @ A : u16 = 0x0100;
                A = A + 0x0050;  // 0x0150 = 336
                R2 = A;

                // Back to 8-bit
                A = 200;
                A = A - 100;    // 100
                R3 = A;
            }
        ''', ExpectedState(
            memory={
                0x7E0010: 15,
                0x7E0012: [0x50, 0x01],  # 0x0150
                0x7E0014: 100,
            }
        ))
        assert result.success, f"Failures: {result.failures}"

    def test_u16_subtraction_crossing_byte_boundary(self, e2e):
        """Test 16-bit subtraction that crosses byte boundary."""
        result = e2e.run('''
            #[zeropage(0x10)]
            static mut RESULT: u16;

            #[entry]
            fn main() {
                let val @ A : u16 = 0x0100;  // 256
                A = A - 1;                    // 255 = 0x00FF
                RESULT = A;
            }
        ''', ExpectedState(
            A=0x00FF,
            memory={0x7E0010: [0xFF, 0x00]}
        ))
        assert result.success, f"Failures: {result.failures}"


class TestModeSwitchingBitwise:
    """Mode switching with bitwise operations."""

    @pytest.fixture
    def e2e(self):
        return E2ETest()

    def test_u8_and_u16_bitwise_and(self, e2e):
        """Test 8-bit AND followed by 16-bit AND."""
        result = e2e.run('''
            #[zeropage(0x10)]
            static mut R8: u8;
            #[zeropage(0x12)]
            static mut R16: u16;

            #[entry]
            fn main() {
                // 8-bit AND
                A = 0xFF;
                A = A & 0x0F;
                R8 = A;  // 0x0F

                // 16-bit AND
                let w @ A : u16 = 0xFFFF;
                A = A & 0x0FF0;
                R16 = A;  // 0x0FF0
            }
        ''', ExpectedState(
            memory={0x7E0010: 0x0F, 0x7E0012: [0xF0, 0x0F]}
        ))
        assert result.success, f"Failures: {result.failures}"

    def test_u8_and_u16_bitwise_or(self, e2e):
        """Test 8-bit OR followed by 16-bit OR."""
        result = e2e.run('''
            #[zeropage(0x10)]
            static mut R8: u8;
            #[zeropage(0x12)]
            static mut R16: u16;

            #[entry]
            fn main() {
                // 8-bit OR
                A = 0x0F;
                A = A | 0xF0;
                R8 = A;  // 0xFF

                // 16-bit OR
                let w @ A : u16 = 0x00FF;
                A = A | 0xFF00;
                R16 = A;  // 0xFFFF
            }
        ''', ExpectedState(
            memory={0x7E0010: 0xFF, 0x7E0012: [0xFF, 0xFF]}
        ))
        assert result.success, f"Failures: {result.failures}"

    def test_mixed_bitwise_xor(self, e2e):
        """Test mixed XOR operations."""
        result = e2e.run('''
            #[zeropage(0x10)]
            static mut R8: u8;
            #[zeropage(0x12)]
            static mut R16: u16;

            #[entry]
            fn main() {
                // 8-bit XOR
                A = 0xAA;
                A = A ^ 0xFF;
                R8 = A;  // 0x55

                // 16-bit XOR
                let w @ A : u16 = 0xAAAA;
                A = A ^ 0xFFFF;
                R16 = A;  // 0x5555
            }
        ''', ExpectedState(
            memory={0x7E0010: 0x55, 0x7E0012: [0x55, 0x55]}
        ))
        assert result.success, f"Failures: {result.failures}"


class TestModeSwitchingMemory:
    """Mode switching with memory loads and stores."""

    @pytest.fixture
    def e2e(self):
        return E2ETest()

    def test_load_store_mixed_sizes(self, e2e):
        """Test loading and storing mixed 8-bit and 16-bit values."""
        result = e2e.run('''
            #[zeropage(0x10)]
            static mut VAL8: u8 = 0x42;
            #[zeropage(0x12)]
            static mut VAL16: u16 = 0x1234;
            #[zeropage(0x14)]
            static mut OUT8: u8;
            #[zeropage(0x16)]
            static mut OUT16: u16;

            #[entry]
            fn main() {
                // Load 8-bit, store 8-bit
                A = VAL8;
                OUT8 = A;

                // Load 16-bit, store 16-bit
                let w @ A : u16 = VAL16;
                OUT16 = A;
            }
        ''', ExpectedState(
            memory={
                0x7E0014: 0x42,
                0x7E0016: [0x34, 0x12],
            }
        ))
        assert result.success, f"Failures: {result.failures}"

    def test_interlaced_memory_operations(self, e2e):
        """Test interlaced 8-bit and 16-bit memory operations."""
        result = e2e.run('''
            #[zeropage(0x10)]
            static mut A8: u8 = 10;
            #[zeropage(0x12)]
            static mut B16: u16 = 1000;
            #[zeropage(0x14)]
            static mut C8: u8 = 20;
            #[zeropage(0x16)]
            static mut D16: u16 = 2000;

            #[zeropage(0x20)]
            static mut OUT1: u8;
            #[zeropage(0x22)]
            static mut OUT2: u16;
            #[zeropage(0x24)]
            static mut OUT3: u8;
            #[zeropage(0x26)]
            static mut OUT4: u16;

            #[entry]
            fn main() {
                A = A8;
                A = A + 5;
                OUT1 = A;  // 15

                let w1 @ A : u16 = B16;
                A = A + 234;
                OUT2 = A;  // 1234

                A = C8;
                A = A + 10;
                OUT3 = A;  // 30

                let w2 @ A : u16 = D16;
                A = A + 345;
                OUT4 = A;  // 2345
            }
        ''', ExpectedState(
            memory={
                0x7E0020: 15,
                0x7E0022: [0xD2, 0x04],  # 1234
                0x7E0024: 30,
                0x7E0026: [0x29, 0x09],  # 2345
            }
        ))
        assert result.success, f"Failures: {result.failures}"


class TestModeSwitchingIndexRegisters:
    """Mode switching involving X and Y registers (always 16-bit)."""

    @pytest.fixture
    def e2e(self):
        return E2ETest()

    def test_x_y_with_u8_a(self, e2e):
        """Test X/Y (16-bit) alongside u8 A operations."""
        result = e2e.run('''
            #[zeropage(0x10)]
            static mut OUT_A: u8;

            #[entry]
            fn main() {
                X = 0x1234;
                Y = 0x5678;
                A = 0x42;
                OUT_A = A;
            }
        ''', ExpectedState(
            A=0x42,
            X=0x1234,
            Y=0x5678,
            memory={0x7E0010: 0x42}
        ))
        assert result.success, f"Failures: {result.failures}"

    def test_x_y_with_u16_a(self, e2e):
        """Test X/Y (16-bit) alongside u16 A operations."""
        result = e2e.run('''
            #[zeropage(0x10)]
            static mut OUT_A: u16;

            #[entry]
            fn main() {
                X = 0xAAAA;
                Y = 0xBBBB;
                let val @ A : u16 = 0xCCCC;
                OUT_A = A;
            }
        ''', ExpectedState(
            A=0xCCCC,
            X=0xAAAA,
            Y=0xBBBB,
            memory={0x7E0010: [0xCC, 0xCC]}
        ))
        assert result.success, f"Failures: {result.failures}"

    def test_mixed_all_registers(self, e2e):
        """Test mixed operations on A (8/16), X, and Y."""
        result = e2e.run('''
            #[zeropage(0x10)]
            static mut R1: u8;
            #[zeropage(0x12)]
            static mut R2: u16;

            #[entry]
            fn main() {
                // Set index registers
                X = 0x100;
                Y = 0x200;

                // 8-bit A
                A = 0x55;
                R1 = A;

                // Modify index registers
                X = X + 0x50;
                Y = Y + 0x50;

                // 16-bit A
                let w @ A : u16 = 0x300;
                A = A + 0x50;
                R2 = A;
            }
        ''', ExpectedState(
            X=0x150,
            Y=0x250,
            A=0x350,
            memory={0x7E0010: 0x55, 0x7E0012: [0x50, 0x03]}
        ))
        assert result.success, f"Failures: {result.failures}"


class TestModeSwitchingEdgeCases:
    """Edge cases for mode switching."""

    @pytest.fixture
    def e2e(self):
        return E2ETest()

    def test_u16_value_fits_in_u8(self, e2e):
        """Test u16 binding with value that fits in u8."""
        result = e2e.run('''
            #[zeropage(0x10)]
            static mut RESULT: u16;

            #[entry]
            fn main() {
                // Even though value fits in u8, we're using u16 binding
                let val @ A : u16 = 0x0042;
                RESULT = A;
            }
        ''', ExpectedState(
            A=0x0042,
            memory={0x7E0010: [0x42, 0x00]},
            flags={'M': False}  # Should be in m16
        ))
        assert result.success, f"Failures: {result.failures}"

    def test_u8_max_value(self, e2e):
        """Test u8 at maximum value."""
        result = e2e.run('''
            #[zeropage(0x10)]
            static mut RESULT: u8;

            #[entry]
            fn main() {
                A = 0xFF;
                RESULT = A;
            }
        ''', ExpectedState(
            A=0xFF,
            memory={0x7E0010: 0xFF}
        ))
        assert result.success, f"Failures: {result.failures}"

    def test_u16_max_value(self, e2e):
        """Test u16 at maximum value."""
        result = e2e.run('''
            #[zeropage(0x10)]
            static mut RESULT: u16;

            #[entry]
            fn main() {
                let val @ A : u16 = 0xFFFF;
                RESULT = A;
            }
        ''', ExpectedState(
            A=0xFFFF,
            memory={0x7E0010: [0xFF, 0xFF]}
        ))
        assert result.success, f"Failures: {result.failures}"

    def test_consecutive_u16_no_redundant_switches(self, e2e):
        """Test consecutive u16 operations - mode should stay m16."""
        result = e2e.run('''
            #[zeropage(0x10)]
            static mut R1: u16;
            #[zeropage(0x12)]
            static mut R2: u16;
            #[zeropage(0x14)]
            static mut R3: u16;

            #[entry]
            fn main() {
                let v1 @ A : u16 = 0x1111;
                R1 = A;

                // These should ideally not need extra mode switches
                // since we're already in m16 from v1
                let v2 @ A : u16 = 0x2222;
                R2 = A;

                let v3 @ A : u16 = 0x3333;
                R3 = A;
            }
        ''', ExpectedState(
            A=0x3333,
            memory={
                0x7E0010: [0x11, 0x11],
                0x7E0012: [0x22, 0x22],
                0x7E0014: [0x33, 0x33],
            },
            flags={'M': False}  # Should stay in m16
        ))
        assert result.success, f"Failures: {result.failures}"

    def test_zero_values(self, e2e):
        """Test zero values in both modes."""
        result = e2e.run('''
            #[zeropage(0x10)]
            static mut R8: u8;
            #[zeropage(0x12)]
            static mut R16: u16;

            #[entry]
            fn main() {
                A = 0;
                R8 = A;

                let z @ A : u16 = 0;
                R16 = A;
            }
        ''', ExpectedState(
            A=0,
            memory={0x7E0010: 0, 0x7E0012: [0, 0]},
            flags={'Z': True}  # Zero flag should be set
        ))
        assert result.success, f"Failures: {result.failures}"


class TestModeSwitchingLoops:
    """Mode switching inside loops."""

    @pytest.fixture
    def e2e(self):
        return E2ETest()

    def test_u8_loop_counter_u16_accumulator(self, e2e):
        """Test u8 loop counter with u16 accumulator inside loop."""
        result = e2e.run('''
            #[zeropage(0x10)]
            static mut COUNTER: u8;
            #[zeropage(0x12)]
            static mut SUM: u16;

            #[entry]
            fn main() {
                COUNTER = 0;
                SUM = 0;

                loop {
                    // Increment 8-bit counter
                    A = COUNTER;
                    A = A + 1;
                    COUNTER = A;

                    // Add to 16-bit sum
                    let s @ A : u16 = SUM;
                    A = A + 100;
                    SUM = A;

                    // Check exit condition (back to 8-bit)
                    A = COUNTER;
                    if A == 5 {
                        break;
                    }
                }
            }
        ''', ExpectedState(
            memory={
                0x7E0010: 5,        # Counter = 5
                0x7E0012: [0xF4, 0x01],  # Sum = 500 = 0x01F4
            }
        ))
        assert result.success, f"Failures: {result.failures}"

    def test_alternating_in_loop(self, e2e):
        """Test alternating u8/u16 operations in a loop."""
        result = e2e.run('''
            #[zeropage(0x10)]
            static mut VAL8: u8;
            #[zeropage(0x12)]
            static mut VAL16: u16;
            #[zeropage(0x14)]
            static mut COUNT: u8;

            #[entry]
            fn main() {
                VAL8 = 0;
                VAL16 = 0;
                COUNT = 0;

                loop {
                    // 8-bit increment
                    A = VAL8;
                    A = A + 10;
                    VAL8 = A;

                    // 16-bit increment
                    let w @ A : u16 = VAL16;
                    A = A + 1000;
                    VAL16 = A;

                    // Loop counter (8-bit)
                    A = COUNT;
                    A = A + 1;
                    COUNT = A;

                    if A == 3 {
                        break;
                    }
                }
            }
        ''', ExpectedState(
            memory={
                0x7E0010: 30,       # 3 * 10
                0x7E0012: [0xB8, 0x0B],  # 3000 = 0x0BB8
                0x7E0014: 3,
            }
        ))
        assert result.success, f"Failures: {result.failures}"


class TestCrossModeFunctionCalls:
    """Tests for calling functions with different entry/exit modes."""

    @pytest.fixture
    def e2e(self):
        return E2ETest()

    def test_call_m8_function_from_m8(self, e2e):
        """Test calling an m8 function from m8 context (no mode switch needed)."""
        result = e2e.run('''
            #[zeropage(0x10)]
            static mut RESULT: u8;

            fn add_five(val @ A: u8) -> u8 {
                return A + 5;
            }

            #[entry]
            fn main() {
                A = 10;
                A = add_five(A);
                RESULT = A;
            }
        ''', ExpectedState(
            A=15,
            memory={0x7E0010: 15}
        ))
        assert result.success, f"Failures: {result.failures}"

    def test_call_m16_function_from_m8(self, e2e):
        """Test calling an m16 function (u16 @ A param) from m8 context.

        The caller should switch to m16 before the call, and the callee
        expects to receive the u16 argument in 16-bit mode.
        """
        result = e2e.run('''
            #[zeropage(0x10)]
            static mut RESULT: u16;

            fn add_hundred(val @ A: u16) -> u16 {
                return A + 100;
            }

            #[entry]
            fn main() {
                // Start in m8 mode
                let w @ A : u16 = 1000;
                A = add_hundred(A);  // Call m16 function
                RESULT = A;
            }
        ''', ExpectedState(
            memory={0x7E0010: [0x4C, 0x04]}  # 1100 = 0x044C in little endian
        ))
        assert result.success, f"Failures: {result.failures}"

    def test_u16_return_value(self, e2e):
        """Test function returning u16 - callee exits in m16 mode."""
        result = e2e.run('''
            #[zeropage(0x10)]
            static mut RESULT: u16;

            fn get_big_value() -> u16 {
                return 0xABCD;
            }

            #[entry]
            fn main() {
                let r @ A : u16 = get_big_value();
                RESULT = A;
            }
        ''', ExpectedState(
            memory={0x7E0010: [0xCD, 0xAB]}  # 0xABCD in little endian
        ))
        assert result.success, f"Failures: {result.failures}"

    def test_u8_return_after_u16_function(self, e2e):
        """Test that u8 operations work correctly after calling u16 function."""
        result = e2e.run('''
            #[zeropage(0x10)]
            static mut RESULT16: u16;
            #[zeropage(0x12)]
            static mut RESULT8: u8;

            fn get_u16() -> u16 {
                return 0x1234;
            }

            #[entry]
            fn main() {
                // Call u16 function, receive in m16 mode
                let r @ A : u16 = get_u16();
                RESULT16 = A;

                // After call, caller should be back in m8
                // Do an 8-bit operation
                A = 0x42;
                RESULT8 = A;
            }
        ''', ExpectedState(
            memory={
                0x7E0010: [0x34, 0x12],
                0x7E0012: 0x42
            }
        ))
        assert result.success, f"Failures: {result.failures}"

    def test_mixed_function_calls(self, e2e):
        """Test calling both u8 and u16 functions in sequence."""
        result = e2e.run('''
            #[zeropage(0x10)]
            static mut R8: u8;
            #[zeropage(0x12)]
            static mut R16: u16;

            fn inc_u8(val @ A: u8) -> u8 {
                return A + 1;
            }

            fn inc_u16(val @ A: u16) -> u16 {
                return A + 1;
            }

            #[entry]
            fn main() {
                // Call u8 function
                A = 10;
                A = inc_u8(A);
                R8 = A;

                // Call u16 function
                let w @ A : u16 = 1000;
                A = inc_u16(A);
                R16 = A;

                // Call u8 function again (should work after u16 call)
                A = R8;
                A = inc_u8(A);
                R8 = A;
            }
        ''', ExpectedState(
            memory={
                0x7E0010: 12,  # 10 + 1 + 1
                0x7E0012: [0xE9, 0x03]  # 1001 = 0x03E9
            }
        ))
        assert result.success, f"Failures: {result.failures}"
