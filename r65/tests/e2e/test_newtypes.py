# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
End-to-end tests for newtypes.

A newtype is nominal at compile time and its payload at runtime, so what needs
proving on hardware is that the *machine* answers stay the payload's: the right
width, the right signedness, the right register. Several predicates in codegen
are written as `isinstance(t, BasicTypeInfo)` and each one that forgets to
unwrap is a silent miscompile rather than an error — these are the tests that
catch that class.
"""

import pytest
from r65.tests.e2e import ExpectedState


class TestNewtypeStorage:
    def test_static_round_trip(self, e2e):
        """A u8 newtype occupies exactly one byte and round-trips."""
        result = e2e.run('''
            struct TileId(u8);

            #[zeropage(0x10)]
            static mut TILE: TileId;
            #[zeropage(0x11)]
            static mut OUT: u8;

            #[entry]
            fn main() {
                TILE = TileId(0x5A);
                OUT = TILE.0;
            }
        ''', ExpectedState(memory={0x7E0010: 0x5A, 0x7E0011: 0x5A}))
        assert result.success, f"Failures: {result.failures}"

    def test_u16_payload_is_two_bytes(self, e2e):
        """A 2-byte newtype must store both bytes — the width predicate has to
        unwrap, or only the low byte lands."""
        result = e2e.run('''
            struct Addr(u16);

            #[zeropage(0x10)]
            static mut AD: Addr;

            #[entry]
            fn main() {
                AD = Addr(0x1234);
            }
        ''', ExpectedState(memory={0x7E0010: [0x34, 0x12]}))
        assert result.success, f"Failures: {result.failures}"

    def test_struct_field_and_array(self, e2e):
        """Newtypes compose as ordinary scalars in aggregates."""
        result = e2e.run('''
            struct TileId(u8);
            struct Holder { tag: u8, tile: TileId }

            #[zeropage(0x10)]
            static mut H: Holder;
            #[ram]
            static mut ARR: [TileId; 4];
            #[zeropage(0x20)]
            static mut OUT: u8;
            #[zeropage(0x21)]
            static mut OUT2: u8;

            #[entry]
            fn main() {
                H.tag = 1;
                H.tile = TileId(0x77);
                ARR[2] = TileId(0x33);
                OUT = H.tile.0;
                OUT2 = ARR[2].0;
            }
        ''', ExpectedState(memory={
            0x7E0010: 1,
            0x7E0011: 0x77,
            0x7E0020: 0x77,
            0x7E0021: 0x33,
        }))
        assert result.success, f"Failures: {result.failures}"


class TestNewtypeArithmetic:
    def test_inherited_arithmetic(self, e2e):
        result = e2e.run('''
            struct TileId(u8);

            #[zeropage(0x10)]
            static mut T: TileId;

            #[entry]
            fn main() {
                T = TileId(10);
                T = T + 5;
                T = T - 3;
                T = T << 1;
            }
        ''', ExpectedState(memory={0x7E0010: 24}))
        assert result.success, f"Failures: {result.failures}"

    def test_signed_comparison_takes_the_signed_branch(self, e2e):
        """`struct Q10(i16)` must compare with BMI/BPL. Miss the signedness
        unwrap and -5 reads as 65531, so the else branch runs."""
        result = e2e.run('''
            struct Q10(i16);

            #[zeropage(0x10)]
            static mut V: Q10;
            #[zeropage(0x14)]
            static mut FLAG: u8;

            #[entry]
            fn main() {
                V = Q10(0 - 5);
                if V < Q10(0) {
                    FLAG = 1;
                } else {
                    FLAG = 2;
                }
            }
        ''', ExpectedState(memory={0x7E0014: 1}))
        assert result.success, f"Failures: {result.failures}"

    def test_16bit_arithmetic_runs_in_m16(self, e2e):
        """A 16-bit newtype must compute in m16; in m8 the high byte is lost."""
        result = e2e.run('''
            struct Addr(u16);

            #[zeropage(0x10)]
            static mut AD: Addr;

            #[entry]
            fn main() {
                AD = Addr(0x00F0);
                AD = AD + 0x0020;
            }
        ''', ExpectedState(memory={0x7E0010: [0x10, 0x01]}))
        assert result.success, f"Failures: {result.failures}"


class TestNewtypeABI:
    def test_register_bound_parameter_and_return(self, e2e):
        result = e2e.run('''
            struct TileId(u8);

            #[zeropage(0x10)]
            static mut OUT: TileId;

            fn bump(t @ A: TileId) -> TileId {
                return t + 1;
            }

            #[entry]
            fn main() {
                OUT = bump(TileId(0x41));
            }
        ''', ExpectedState(memory={0x7E0010: 0x42}))
        assert result.success, f"Failures: {result.failures}"

    def test_u16_newtype_in_a_enters_m16(self, e2e):
        """Entry-mode inference must unwrap: a 2-byte newtype bound to A that
        stays in m8 truncates the argument to its low byte."""
        result = e2e.run('''
            struct Addr(u16);

            #[zeropage(0x10)]
            static mut OUT: Addr;

            fn add_one(a @ A: Addr) -> Addr {
                return a + 1;
            }

            #[entry]
            fn main() {
                OUT = add_one(Addr(0x0102));
            }
        ''', ExpectedState(memory={0x7E0010: [0x03, 0x01]}))
        assert result.success, f"Failures: {result.failures}"

    def test_stack_parameters(self, e2e):
        result = e2e.run('''
            struct TileId(u8);

            #[zeropage(0x10)]
            static mut OUT: TileId;

            fn combine(a: TileId, b: TileId) -> TileId {
                return a + b;
            }

            #[entry]
            fn main() {
                OUT = combine(TileId(0x20), TileId(0x03));
            }
        ''', ExpectedState(memory={0x7E0010: 0x23}))
        assert result.success, f"Failures: {result.failures}"

    def test_u8_newtype_multi_return_uses_b_slot(self, e2e):
        """The second 8-bit return value rides in B. The predicate choosing the
        return-register order has to unwrap or it picks X and both land wrong."""
        result = e2e.run('''
            struct TileId(u8);

            #[zeropage(0x10)]
            static mut FIRST: TileId;
            #[zeropage(0x11)]
            static mut SECOND: TileId;

            fn pair() -> TileId, TileId {
                return TileId(0x11), TileId(0x22);
            }

            #[entry]
            fn main() {
                let a, b = pair();
                FIRST = a;
                SECOND = b;
            }
        ''', ExpectedState(memory={0x7E0010: 0x11, 0x7E0011: 0x22}))
        assert result.success, f"Failures: {result.failures}"


class TestNewtypeMethods:
    def test_by_value_self(self, e2e):
        result = e2e.run('''
            struct TileId(u8);

            impl TileId {
                fn raw(self) -> u8 { return self.0; }
                fn bumped(self) -> TileId { return TileId(self.0 + 1); }
            }

            #[zeropage(0x10)]
            static mut TILE: TileId;
            #[zeropage(0x11)]
            static mut OUT: u8;

            #[entry]
            fn main() {
                TILE = TileId(0x40);
                TILE = TILE.bumped();
                OUT = TILE.raw();
            }
        ''', ExpectedState(memory={0x7E0010: 0x41, 0x7E0011: 0x41}))
        assert result.success, f"Failures: {result.failures}"

    def test_method_with_argument(self, e2e):
        """self in A plus a second argument — the argument must not clobber self."""
        result = e2e.run('''
            struct TileId(u8);

            impl TileId {
                fn plus(self, n: u8) -> TileId { return TileId(self.0 + n); }
            }

            #[zeropage(0x10)]
            static mut OUT: TileId;

            #[entry]
            fn main() {
                let t: TileId = 0x30;
                OUT = t.plus(5);
            }
        ''', ExpectedState(memory={0x7E0010: 0x35}))
        assert result.success, f"Failures: {result.failures}"

    def test_chained_methods(self, e2e):
        result = e2e.run('''
            struct TileId(u8);

            impl TileId {
                fn bumped(self) -> TileId { return TileId(self.0 + 1); }
            }

            #[zeropage(0x10)]
            static mut OUT: TileId;

            #[entry]
            fn main() {
                let t: TileId = 0x10;
                OUT = t.bumped().bumped().bumped();
            }
        ''', ExpectedState(memory={0x7E0010: 0x13}))
        assert result.success, f"Failures: {result.failures}"

    def test_u16_payload_method(self, e2e):
        """A 2-byte self in A must put the method in m16."""
        result = e2e.run('''
            struct Q10(i16);

            impl Q10 {
                fn doubled(self) -> Q10 { return Q10(self.0 + self.0); }
            }

            #[zeropage(0x10)]
            static mut OUT: Q10;

            #[entry]
            fn main() {
                let q: Q10 = 0x0123;
                OUT = q.doubled();
            }
        ''', ExpectedState(memory={0x7E0010: [0x46, 0x02]}))
        assert result.success, f"Failures: {result.failures}"

    def test_method_on_a_static_receiver(self, e2e):
        result = e2e.run('''
            struct TileId(u8);

            impl TileId {
                fn bumped(self) -> TileId { return TileId(self.0 + 1); }
            }

            #[zeropage(0x10)]
            static mut TILE: TileId;

            #[entry]
            fn main() {
                TILE = TileId(0x7F);
                TILE = TILE.bumped();
            }
        ''', ExpectedState(memory={0x7E0010: 0x80}))
        assert result.success, f"Failures: {result.failures}"


class TestNewtypeMatch:
    """`match` on a newtype, dispatching correctly on hardware.

    The type checker used to crash before codegen ever saw one of these
    (`match_validator` read `NewtypeTypeInfo.name`, which raises by design), so
    the lowering had never actually been exercised. It turned out to be correct
    already — these pin that.
    """

    SRC = '''
struct Tid(u8);
#[zeropage(0x10)] static mut OUT: u8;
#[zeropage(0x12)] static mut IN: Tid;
#[entry]
fn main() {
    IN = %d;
    let t: Tid = IN;
    match t {
        1 => { OUT = 10; },
        5 => { OUT = 50; },
        _ => { OUT = 99; }
    };
}
'''

    @pytest.mark.parametrize("value,expected", [
        (1, 10),      # first arm
        (5, 50),      # later arm
        (7, 99),      # wildcard
    ])
    def test_dispatches_through_the_wrapper(self, e2e, value, expected):
        """Through a static, so the scrutinee is a runtime value rather than a
        constant the folder could resolve."""
        r = e2e.run(self.SRC % value,
                    ExpectedState(memory={0x7E0010: expected}))
        assert r.success, f"Tid({value}): {r.error} {r.failures}"

    def test_range_pattern_dispatches(self, e2e):
        src = '''
struct Tid(u8);
#[zeropage(0x10)] static mut OUT: u8;
#[zeropage(0x12)] static mut IN: Tid;
#[entry]
fn main() {
    IN = 3;
    let t: Tid = IN;
    match t { 1..5 => { OUT = 1; }, _ => { OUT = 0; } };
}
'''
        r = e2e.run(src, ExpectedState(memory={0x7E0010: 1}))
        assert r.success, f"{r.error} {r.failures}"


class TestNewtypeTraitMethod:
    """A trait method on a newtype, dispatched statically, on hardware.

    Newtypes may implement traits now; only `*dyn` over one is rejected. Since
    the type checker had never let such an impl through, the lowering had not
    been exercised — this confirms `self` reaches the method by value as an
    inherent method's would.
    """

    SRC = '''
struct Tid(u8);
trait Bump { fn bump(self) -> u8; }
impl Bump for Tid { fn bump(self) -> u8 { return self.0 + 1; } }
#[zeropage(0x10)] static mut OUT: u8;
#[zeropage(0x12)] static mut IN: Tid;
#[entry]
fn main() { IN = 41; let t: Tid = IN; OUT = t.bump(); }
'''

    def test_static_dispatch(self, e2e):
        """Through a static, so `self` is a runtime value."""
        r = e2e.run(self.SRC, ExpectedState(memory={0x7E0010: 42}))
        assert r.success, f"{r.error} {r.failures}"
