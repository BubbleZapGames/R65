# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""A register-bound parameter must survive its own use as an address.

`_count_param_usages` decides whether a register-bound parameter is saved
to a virtual register at entry or merely aliased to the hardware register.
An alias is only safe while nothing else touches that register — so a
parameter read *anywhere* has to be counted, including inside the target of
an assignment, where it is read to compute an address rather than written.

Missing those reads aliases the parameter to raw hardware X, and the first
instruction that rescales or reuses X destroys it with no diagnostic.
"""

from r65.tests.e2e import ExpectedState


class TestIndexParamSurvivesForeignRegisterUse:
    """`fn f(idx @ X)` whose body also needs X for another index."""

    def test_index_param_survives_another_array_index(self, e2e):
        """`DEST[idx] = TABLE[k]` — TABLE's index must not eat `idx`."""
        result = e2e.run('''
            #[zeropage(0x10)]
            static mut DEST: [u8; 4];
            #[zeropage(0x20)]
            static mut TABLE: [u8; 4];
            #[zeropage(0xB0)]
            static mut k: u16;

            fn store(idx @ X: u16) { DEST[idx] = TABLE[k]; }

            #[entry]
            fn main() {
                TABLE[3] = 0x5A;
                k = 3;
                store(1);
            }
        ''', ExpectedState(memory={
            0x7E0011: 0x5A,   # DEST[1]
            0x7E0010: 0x00,   # DEST[0] untouched
            0x7E0013: 0x00,   # DEST[3] untouched — the corrupt form writes here
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_write_only_index_param_is_still_a_read(self, e2e):
        """The parameter appears only in targets, never on a value side."""
        result = e2e.run('''
            #[zeropage(0x10)]
            static mut LO: [u8; 4];
            #[zeropage(0x20)]
            static mut HI: [u16; 4];

            fn init(idx @ X: u16) { HI[idx] = 0x1234; LO[idx] = 0x77; }

            #[entry]
            fn main() { init(2); }
        ''', ExpectedState(memory={
            0x7E0012: 0x77,            # LO[2]
            0x7E0024: [0x34, 0x12],    # HI[2]
        }))
        assert result.success, f"Failures: {result.failures}"
