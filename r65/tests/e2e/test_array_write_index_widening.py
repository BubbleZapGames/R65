# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Regression tests for u8-index widening on array/pointer *write* paths.

The read path (expression.py:_compute_index_offset) widens a u8/i8 index to
u16 before scaling by element_size, so `arr[i]` is correct even when
`i * element_size > 255`. The write paths in assignment.py used to reimplement
the scaling inline and skipped that widening, so `arr[i] = v` computed the
byte offset in 8-bit and wrapped:

    BUF: [u16; 200]; IDX: u8 = 150
    BUF[IDX] = 0xBEEF   # correct offset 150*2 = 300
                        # buggy offset (300 & 0xFF) = 44  -> wrong slot

These tests write at an index whose byte offset exceeds 255 and read it back
through the (already-correct) read path. If the write path overflows, the
value lands in the wrong slot and the read-back is 0.
"""

from r65.tests.e2e import ExpectedState


class TestArrayWriteIndexWidening:
    """u8 index into a multi-byte array on the assignment path."""

    def test_u16_array_write_high_index(self, e2e):
        """BUF[150] = v on a [u16; N] array: byte offset 300 must not wrap."""
        result = e2e.run('''
            #[zeropage(0x00, register)]
            static mut SCRATCH0: u8;
            #[zeropage(0x01, register)]
            static mut SCRATCH1: u8;
            #[zeropage(0x02, register)]
            static mut SCRATCH2: u8;
            #[zeropage(0x03, register)]
            static mut SCRATCH3: u8;

            #[ram]
            static mut BUF: [u16; 200] = [0; 200];

            #[zeropage(0x10)]
            static mut RESULT: u16;
            #[zeropage(0x12)]
            static mut WRONG_SLOT: u16;
            #[zeropage(0x14)]
            static mut IDX: u8;

            #[entry]
            fn main() {
                IDX = 150;
                BUF[IDX] = 0xBEEF;     // write path under test

                RESULT = BUF[IDX];     // read path (widened) -> 0xBEEF if write OK

                // (150 * 2) & 0xFF == 44 == slot 22. If the write overflowed
                // it landed here instead; it must be untouched (still 0).
                IDX = 22;
                WRONG_SLOT = BUF[IDX];
            }
        ''', ExpectedState(memory={
            0x7E0010: [0xEF, 0xBE],  # RESULT == 0xBEEF (write reached slot 150)
            0x7E0012: [0x00, 0x00],  # WRONG_SLOT == 0 (overflow slot untouched)
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_u16_array_write_max_u8_index(self, e2e):
        """BUF[200] would need u16; use index 200 in a [u16; 256] array."""
        result = e2e.run('''
            #[zeropage(0x00, register)]
            static mut SCRATCH0: u8;
            #[zeropage(0x01, register)]
            static mut SCRATCH1: u8;
            #[zeropage(0x02, register)]
            static mut SCRATCH2: u8;
            #[zeropage(0x03, register)]
            static mut SCRATCH3: u8;

            #[ram]
            static mut BUF: [u16; 256] = [0; 256];

            #[zeropage(0x10)]
            static mut RESULT: u16;
            #[zeropage(0x14)]
            static mut IDX: u8;

            #[entry]
            fn main() {
                IDX = 200;             // byte offset 400, wraps to 144 if buggy
                BUF[IDX] = 0x1234;
                RESULT = BUF[IDX];
            }
        ''', ExpectedState(memory={
            0x7E0010: [0x34, 0x12],  # RESULT == 0x1234
        }))
        assert result.success, f"Failures: {result.failures}"
