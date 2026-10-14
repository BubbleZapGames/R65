# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
A `match` lookup table read from a function outside bank 0.

An all-constant `match` compiles to a table placed beside the code, so it
lands in the function's bank. It was read with `LDA.w table,X`, which takes
the bank from DBR (bank 0 here): in a bank-2 function that read whatever bank
0 held at the same offset. It is now long unless DBR holds the code's bank,
as a databank=inline prologue makes it.
"""

from r65.tests.e2e import ExpectedState


def test_lut_in_another_bank_reads_its_own_table(e2e):
    result = e2e.run('''
        #[zeropage(0x10)]
        static mut OUT: [u8; 4];

        #[bank(2)]

        #[inline(never)]
        far fn byte_for(i @ A: u8) -> u8 {
            return match i { 0 => 0x11, 1 => 0x22, 2 => 0x33, _ => 0 };
        }

        #[inline(never)]
        far fn word_for(i @ A: u8) -> u16 {
            return match i { 0 => 0x1111, 1 => 0x2222, 2 => 0xBEEF, _ => 0 };
        }

        #[bank(0)]

        #[entry]
        fn main() {
            OUT[0] = byte_for(1);
            OUT[1] = byte_for(2);
            let w: u16 = word_for(2);
            OUT[2] = w as u8;
            OUT[3] = (w >> 8) as u8;
        }
    ''', ExpectedState(memory={0x7E0010: [0x22, 0x33, 0xEF, 0xBE]}))
    assert result.success, f"Failures: {result.failures}"


def test_lut_in_a_databank_inline_function(e2e):
    """databank=inline points DBR at bank 2, so the absolute load is right.
    The hardware write keeps it from skipping that DBR setup."""
    result = e2e.run('''
        #[zeropage(0x10)]
        static mut OUT: u8;
        #[hw(0x4202)]
        static mut WRMPYA: u8;

        #[bank(2)]

        #[inline(never)]
        #[mode(databank=inline)]
        far fn byte_for(i @ A: u8) -> u8 {
            let v: u8 = match i { 0 => 0x11, 1 => 0x22, 2 => 0x33, _ => 0 };
            WRMPYA = v;
            return v;
        }

        #[bank(0)]

        #[entry]
        fn main() {
            OUT = byte_for(2);
        }
    ''', ExpectedState(memory={0x7E0010: [0x33]}))
    assert result.success, f"Failures: {result.failures}"
