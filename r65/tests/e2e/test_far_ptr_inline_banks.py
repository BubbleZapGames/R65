# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Inlining a far function with a far-pointer stack param reads the right bank.

The inliner used to refuse these outright, on the grounds that the body's
`(d,S),Y` / `[dp],Y` derefs depend on a prologue inlining drops. They do not —
codegen picks the far-deref path from whichever caller the body lands in, and
both outcomes are bank-explicit:

- a caller with no far-ptr params of its own gets no whole-function strategy, so
  each deref brackets itself with PHB / set DBR from the pointer's own bank byte
  / PLB;
- a caller that does have them sees a deref through a vreg that is not one of
  *its* params, which makes `_is_set_dbr_safe` veto SET_DBR and force D=S, whose
  `[dp],Y` is indirect long and carries its own bank.

Compiling is not the test. A wrong-bank read compiles perfectly and returns
whatever sits at the same offset in the wrong bank, so these run the ROM and
assert the bytes. The data deliberately lives outside bank 0, with a decoy at a
similar offset in the caller's bank: read the bank wrong and you get the decoy,
which is the failure this restriction existed to prevent.
"""

from r65.tests.e2e import ExpectedState


def test_inlined_far_helper_reads_the_pointers_bank(e2e):
    """Ordinary caller: no far-ptr params, so each deref sets DBR per access."""
    result = e2e.run('''
        #[zeropage(0x10)]
        static mut OUT: [u8; 2];

        #[bank(0)]
        static DECOY: [u8; 4] = "XX\\0";
        #[bank(1)]
        static REAL: [u8; 4] = "AB\\0";

        #[bank(2)]

        #[inline(always)]
        far fn byte_at(s: far *u8, i: u16) -> u8 {
            return s[i];
        }

        #[bank(0)]

        #[entry]
        fn main() {
            OUT[0] = byte_at(&REAL as far *u8, 0);
            OUT[1] = byte_at(&REAL as far *u8, 1);
        }
    ''', ExpectedState(memory={0x7E0010: [0x41, 0x42]}))  # 'A', 'B'
    assert result.success, f"Failures: {result.failures}"


def test_inlined_into_a_caller_that_has_its_own_far_ptr_param(e2e):
    """The caller derefs its own far-ptr param *and* an inlined one.

    This is the case that depends on `_is_set_dbr_safe` noticing the non-param
    deref and forcing D=S. Under SET_DBR the inlined deref would use the
    caller's DBR — set to the caller's own pointer bank, 3 — and read 'C' from
    OTHER instead of 'A' from REAL, summing to 0x86 rather than 0x84.

    The result comes back as a return value rather than through a global: a
    caller on D=S reaches zeropage by absolute addressing, which lands in
    whatever bank DBR happens to hold, and that is a separate question from the
    one under test.
    """
    result = e2e.run('''
        #[zeropage(0x10)]
        static mut OUT: u8;

        #[bank(0)]
        static DECOY: [u8; 4] = "XX\\0";
        #[bank(1)]
        static REAL: [u8; 4] = "AB\\0";
        #[bank(3)]
        static OTHER: [u8; 4] = "CD\\0";

        #[bank(2)]

        #[inline(always)]
        far fn byte_at(s: far *u8, i: u16) -> u8 {
            return s[i];
        }

        far fn mixed(own: far *u8) -> u8 {
            let mine: u8 = own[0];                              // 'C', bank 3
            let theirs: u8 = byte_at(&REAL as far *u8, 0);      // 'A', bank 1
            return mine + theirs;
        }

        #[bank(0)]

        #[entry]
        fn main() {
            OUT = mixed(&OTHER as far *u8);
        }
    ''', ExpectedState(memory={0x7E0010: [0x84]}))  # 'C' + 'A'
    assert result.success, f"Failures: {result.failures}"
