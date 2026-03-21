# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
End-to-end test for near pointer writes under D=S far pointer mode.

Bug: when a function has both a near pointer (*u8) and a far pointer
(far *u8) parameter, the compiler uses D=S mode for the far pointer.
Writes through the near pointer must use long addressing to avoid
DBR-relative writes going to the wrong bank.
"""

import pytest
from r65.tests.e2e import E2ETest, ExpectedState


class TestNearPtrDBR:
    """Test near pointer writes under D=S far pointer mode."""

    @pytest.fixture
    def e2e(self):
        return E2ETest()

    def test_near_ptr_write_with_far_ptr_param(self, e2e):
        """Near pointer write should work even when far pointer is present."""
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
            static mut dest_buf: [u8; 8];

            static src_data: [u8; 4] = [0xAA, 0xBB, 0xCC, 0xDD];

            fn copy_bytes(dst: *u8, src: far *u8, count: u8) {
                let i: u8 = 0;
                loop {
                    if i >= count {
                        break;
                    }
                    *dst = *src;
                    dst++;
                    src++;
                    i++;
                }
            }

            #[entry]
            fn main() {
                copy_bytes(&dest_buf as *u8, &src_data as far *u8, 4);
            }
        ''', ExpectedState(memory={
            0x7E2000: 0xAA,
            0x7E2001: 0xBB,
            0x7E2002: 0xCC,
            0x7E2003: 0xDD,
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_near_ptr_write_with_cast(self, e2e):
        """Near pointer write with explicit cast should also work.

        This reproduces the unrle bug: &map as *u8 has an explicit
        cast that may prevent auto-promotion to far *u8.
        """
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
            static mut dest_buf: [u8; 16];

            // RLE-like source data in ROM
            static src_data: [u8; 8] = [
                0xAA, 0x11, 0x22, 0x33, 0x44, 0xAA, 0x00,  // tag=0xAA, literals, tag+0=end
                0x00
            ];

            fn decompress(dst: *u8, src: far *u8) {
                let tag: u8 = *src;
                src++;
                loop {
                    let byte: u8 = *src;
                    src++;
                    if byte == tag {
                        let count: u8 = *src;
                        src++;
                        if count == 0 { break; }
                    } else {
                        *dst = byte;
                        dst++;
                    }
                }
            }

            #[entry]
            fn main() {
                // Pass with explicit cast to *u8 (like classickong's unrle call)
                decompress(&dest_buf as *u8, &src_data as far *u8);
            }
        ''', ExpectedState(memory={
            0x7E2000: 0x11,
            0x7E2001: 0x22,
            0x7E2002: 0x33,
            0x7E2003: 0x44,
        }))
        assert result.success, f"Failures: {result.failures}"
