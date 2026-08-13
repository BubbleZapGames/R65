# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
End-to-end tests for nested aggregate field access (`outer.inner.leaf`).

Nesting folds to a single constant offset at compile time, so the thing worth
checking at runtime is that the folded offset actually lands on the right bytes
for each base shape: static, array element, pointer auto-deref, explicit deref.
"""

from r65.tests.e2e import ExpectedState


class TestNestedFieldAccessE2E:
    def test_static_struct_three_levels(self, e2e):
        """outer.mid.inner.leaf round-trips through the correct offset."""
        result = e2e.run('''
            struct Inner { a: u8, w: u16 }
            struct Mid   { pad: u8, inner: Inner }
            struct Outer { tag: u8, mid: Mid }

            #[zeropage(0x40)]
            static mut O: Outer;
            #[zeropage(0x20)]
            static mut RW: u16;
            #[zeropage(0x22)]
            static mut RA: u8;

            #[entry]
            fn main() {
                O.tag = 1;
                O.mid.pad = 2;
                O.mid.inner.a = 3;
                O.mid.inner.w = 0x1234;
                RW = O.mid.inner.w;
                RA = O.mid.inner.a;
            }
        ''', ExpectedState(memory={
            0x7E0020: [0x34, 0x12],
            0x7E0022: 3,
            # Layout: tag@0, mid.pad@1, mid.inner.a@2, mid.inner.w@3
            0x7E0040: 1,
            0x7E0041: 2,
            0x7E0042: 3,
            0x7E0043: [0x34, 0x12],
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_nested_union_in_struct(self, e2e):
        """A union nested in a struct still overlays its own fields."""
        result = e2e.run('''
            union Word { raw: u16, bytes: [u8; 2] }
            struct Holder { tag: u8, w: Word }

            #[zeropage(0x40)]
            static mut H: Holder;
            #[zeropage(0x20)]
            static mut LO: u8;
            #[zeropage(0x21)]
            static mut HI: u8;

            #[entry]
            fn main() {
                H.tag = 9;
                H.w.raw = 0x1234;
                LO = H.w.bytes[0];
                HI = H.w.bytes[1];
            }
        ''', ExpectedState(memory={
            0x7E0020: 0x34,
            0x7E0021: 0x12,
            0x7E0040: 9,              # tag untouched by the union write
            0x7E0041: [0x34, 0x12],   # union at offset 1
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_array_element_nested_field(self, e2e):
        """arr[i].inner.leaf indexes and offsets together."""
        result = e2e.run('''
            struct Inner { a: u8, w: u16 }
            struct Mid   { pad: u8, inner: Inner }

            #[zeropage(0x40)]
            static mut ARR: [Mid; 3];
            #[zeropage(0x20)]
            static mut RW: u16;

            #[entry]
            fn main() {
                ARR[2].inner.w = 0x5678;
                RW = ARR[2].inner.w;
            }
        ''', ExpectedState(memory={
            0x7E0020: [0x78, 0x56],
            # element 2 starts at 0x40 + 2*4 = 0x48; inner.w is at +2
            0x7E004A: [0x78, 0x56],
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_array_reached_through_outer_field(self, e2e):
        """outer.arr[i].inner.leaf — both folding mechanisms compose.

        The array base is itself reached through a field, so `peel_field_chain`
        hands its folded offset to the array path, whose own base resolution
        folds `outer.arr`'s offset in turn.
        """
        result = e2e.run('''
            struct Inner { a: u8, w: u16 }
            struct Mid   { pad: u8, inner: Inner }
            struct Outer { tag: u8, arr: [Mid; 3] }

            #[zeropage(0x40)]
            static mut O: Outer;
            #[zeropage(0x20)]
            static mut RW: u16;

            #[entry]
            fn main() {
                O.tag = 1;
                O.arr[2].inner.w = 0x1234;
                RW = O.arr[2].inner.w;
            }
        ''', ExpectedState(memory={
            0x7E0020: [0x34, 0x12],
            0x7E0040: 1,
            # tag@0, arr@1, element 2 at +2*4, inner.w at +2 => 0x40+1+8+2
            0x7E004B: [0x34, 0x12],
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_address_of_nested_field(self, e2e):
        """&outer.inner yields the folded address, usable as a pointer."""
        result = e2e.run('''
            struct Inner { a: u8, w: u16 }
            struct Mid   { pad: u8, inner: Inner }

            #[zeropage(0x40)]
            static mut M: Mid;
            #[zeropage(0x20)]
            static mut RW: u16;

            fn get_w(p: *Inner) -> u16 { return p.w; }

            #[entry]
            fn main() {
                M.inner.w = 0x1234;
                RW = get_w(&M.inner);   // &M.inner == M + 1
            }
        ''', ExpectedState(memory={
            0x7E0020: [0x34, 0x12],
            0x7E0042: [0x34, 0x12],   # M.inner.w at offset 2
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_through_pointer_auto_deref(self, e2e):
        """p.inner.leaf where p is a pointer parameter."""
        result = e2e.run('''
            struct Inner { a: u8, w: u16 }
            struct Mid   { pad: u8, inner: Inner }

            #[zeropage(0x40)]
            static mut M: Mid;
            #[zeropage(0x20)]
            static mut RW: u16;

            fn set_and_get(p: *Mid) -> u16 {
                p.inner.w = 0xAABB;
                return p.inner.w;
            }

            #[entry]
            fn main() {
                RW = set_and_get(&M);
            }
        ''', ExpectedState(memory={
            0x7E0020: [0xBB, 0xAA],
            0x7E0042: [0xBB, 0xAA],   # M.inner.w at offset 2
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_through_explicit_deref(self, e2e):
        """(*p).inner.leaf uses the same folded offset."""
        result = e2e.run('''
            struct Inner { a: u8, w: u16 }
            struct Mid   { pad: u8, inner: Inner }

            #[zeropage(0x40)]
            static mut M: Mid;
            #[zeropage(0x20)]
            static mut RA: u8;

            fn set_and_get(p: *Mid) -> u8 {
                (*p).inner.a = 7;
                return (*p).inner.a;
            }

            #[entry]
            fn main() {
                RA = set_and_get(&M);
            }
        ''', ExpectedState(memory={
            0x7E0020: 7,
            0x7E0041: 7,   # M.inner.a at offset 1
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_local_union_fields_alias(self, e2e):
        """A local union must not be decomposed into independent registers."""
        result = e2e.run('''
            union W { a: u8, b: u8 }

            #[zeropage(0x20)]
            static mut R: u8;

            #[entry]
            fn main() {
                let w: W;
                w.a = 5;
                R = w.b;      // same byte as w.a
            }
        ''', ExpectedState(memory={0x7E0020: 5}))
        assert result.success, f"Failures: {result.failures}"
