# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
End-to-end tests for operator overloading.

Tier A (compound assignment): `a OP= b` on an aggregate dispatches to the
operator-trait method (`a.add_assign(&b)`), verified by running the ROM.
"""

from r65.tests.e2e import ExpectedState


class TestOperatorOverloadE2E:
    def test_add_assign_runs(self, e2e):
        """`SCORE += BONUS` invokes AddAssign::add_assign and mutates SCORE."""
        result = e2e.run('''
            struct Acc { lo: u8, hi: u8 }
            impl AddAssign for Acc {
                fn add_assign(*self, other: *Acc) {
                    self.lo = self.lo + other.lo;
                    self.hi = self.hi + other.hi;
                }
            }

            #[lowram] static mut SCORE: Acc;
            #[lowram] static mut BONUS: Acc;
            #[zeropage(0x20)] static mut RLO: u8;
            #[zeropage(0x21)] static mut RHI: u8;

            #[entry]
            fn main() {
                SCORE.lo = 1;
                SCORE.hi = 2;
                BONUS.lo = 10;
                BONUS.hi = 20;
                SCORE += BONUS;
                RLO = SCORE.lo;   // 11
                RHI = SCORE.hi;   // 22
            }
        ''', ExpectedState(memory={
            0x7E0020: 11,
            0x7E0021: 22,
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_sub_assign_runs(self, e2e):
        """A second operator (SubAssign) over the same machinery."""
        result = e2e.run('''
            struct Acc { lo: u8, hi: u8 }
            impl SubAssign for Acc {
                fn sub_assign(*self, other: *Acc) {
                    self.lo = self.lo - other.lo;
                    self.hi = self.hi - other.hi;
                }
            }

            #[lowram] static mut LHS: Acc;
            #[lowram] static mut RHS: Acc;
            #[zeropage(0x20)] static mut RLO: u8;
            #[zeropage(0x21)] static mut RHI: u8;

            #[entry]
            fn main() {
                LHS.lo = 50;
                LHS.hi = 99;
                RHS.lo = 8;
                RHS.hi = 9;
                LHS -= RHS;
                RLO = LHS.lo;   // 42
                RHI = LHS.hi;   // 90
            }
        ''', ExpectedState(memory={
            0x7E0020: 42,
            0x7E0021: 90,
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_comparison_eq_and_ord_run(self, e2e):
        """`a == b` -> PartialEq::eq; `a < b` -> PartialOrd::cmp compared to 0."""
        result = e2e.run('''
            struct Ord2 { hi: u8, lo: u8 }
            impl PartialEq for Ord2 {
                fn eq(*self, other: *Ord2) -> bool {
                    if self.hi != other.hi { return false; }
                    if self.lo != other.lo { return false; }
                    return true;
                }
            }
            impl PartialOrd for Ord2 {
                fn cmp(*self, other: *Ord2) -> i8 {
                    if self.hi < other.hi { return -1; }
                    if self.hi > other.hi { return 1; }
                    if self.lo < other.lo { return -1; }
                    if self.lo > other.lo { return 1; }
                    return 0;
                }
            }

            #[lowram] static mut P: Ord2;
            #[lowram] static mut Q: Ord2;
            #[zeropage(0x20)] static mut R_EQ: u8;
            #[zeropage(0x21)] static mut R_LT: u8;
            #[zeropage(0x22)] static mut R_GE: u8;

            #[entry]
            fn main() {
                P.hi = 1; P.lo = 5;
                Q.hi = 1; Q.lo = 9;
                if P == Q { R_EQ = 1; } else { R_EQ = 0; }   // not equal -> 0
                if P < Q  { R_LT = 1; } else { R_LT = 0; }   // 5 < 9    -> 1
                if P >= Q { R_GE = 1; } else { R_GE = 0; }   // not >=   -> 0
            }
        ''', ExpectedState(memory={
            0x7E0020: 0,
            0x7E0021: 1,
            0x7E0022: 0,
        }))
        assert result.success, f"Failures: {result.failures}"
