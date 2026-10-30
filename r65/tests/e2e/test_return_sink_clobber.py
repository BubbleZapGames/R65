# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
End-to-end test for a returned load deferred past code that changes it.

Bug: the slot allocator "return-sank" a vreg whose only use was a Return,
emitting its Load at the return site instead of where it was written. It
never looked at what ran in between, so `let old = CUR; CUR = n; return
old;` returned n, and an indexed load read its table with whatever the
intervening code left in X. classickong's Rolling::process
(`let lift = barrel_landing_anim[self.land]; self.land++; return lift;`)
read the table at `self` instead of `self.land`. Return sinking was removed;
a returned load is now allocated like any other value.
"""

from r65.tests.e2e import ExpectedState


class TestReturnSinkClobber:
    def test_store_between_load_and_return(self, e2e):
        """The swap idiom: a store to the loaded static before the return."""
        result = e2e.run('''
            #[zeropage(0xB0)] static mut CUR: u8;
            #[zeropage(0xB1)] static mut H: u8;

            fn swap_out(n @ A: u8) -> u8 {
                let old = CUR;
                CUR = n;
                return old;
            }

            #[entry]
            fn main() {
                CUR = 7;
                H = swap_out(3);
            }
        ''', ExpectedState(memory={
            0x7E00B0: [0x03],   # CUR took the new value
            0x7E00B1: [0x07],   # the old one came back
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_index_register_rewritten_before_return(self, e2e):
        """An indexed load whose X is reloaded for another array's store."""
        result = e2e.run('''
            static TBL: [u8; 4] = [10, 20, 30, 40];
            #[zeropage(0xC0)] static mut SLOT: [u8; 8];
            #[zeropage(0xB1)] static mut H: u8;

            fn lookup_and_clear(i @ X: u16, k: u8) -> u8 {
                let v = TBL[k];
                SLOT[i] = 0;
                return v;
            }

            #[entry]
            fn main() {
                SLOT[1] = 0xFF;
                H = lookup_and_clear(1, 3);
            }
        ''', ExpectedState(memory={
            0x7E00B1: [40],     # TBL[3], not TBL[1]
            0x7E00C1: [0x00],   # SLOT[1] cleared
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_soa_method_post_increment(self, e2e):
        """Rolling::process's shape: read a table at a field, bump the field."""
        result = e2e.run('''
            static TBL: [u8; 4] = [10, 20, 30, 40];
            #[zeropage(0xB1)] static mut H: u8;
            #[zeropage(0xB2)] static mut F: u8;

            #[soa]
            struct Anim { frame: u8 }
            #[lowram] static mut ANIMS: [Anim; 4];

            impl Anim {
                near fn step(self) -> u8 {
                    let v = TBL[self.frame];
                    self.frame++;
                    return v;
                }
            }

            #[entry]
            fn main() {
                ANIMS[1].frame = 2;
                H = ANIMS[1].step();
                F = ANIMS[1].frame;
            }
        ''', ExpectedState(memory={
            0x7E00B1: [30],     # TBL[2], not TBL[1]
            0x7E00B2: [3],
        }))
        assert result.success, f"Failures: {result.failures}"
