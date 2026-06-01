# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Array indexing through a pointer-deref'd struct field.

Before the fix, the MIR builder rejected `self.arr[i]` and `(*p).arr[i]`
shapes with:
    MIR error: Array base must be a static array or struct field,
               got: <class 'r65.compiler.hir.nodes.HIRFieldAccess'>
                       or <class 'r65.compiler.hir.nodes.HIRDereference'>

`resolve_array_base_memloc` only handled bare statics and statically-located
struct fields; arrays reached through a `*self`-style auto-deref had no
static base address to fold into a MemoryLocation. The fix routes those
shapes through LoadIndirect / StoreIndirect against the underlying pointer,
with the outer field offset (offset of the array within the pointee struct)
plus any inner field offset (`.field` on the element) folded into Y or the
constant `offset` field.

Affects four lvalue/rvalue x element/element-field combinations. Idiom is
the typical 'method on a struct that holds an array', e.g.
`oam_buffer.hide(i)` where hide does `self.sprites[i].y = 240`.
"""

from r65.tests.e2e import ExpectedState


class TestPointerDerefArrayIndex:
    """Auto-deref `self.arr[i]` paths in both lvalue and rvalue positions."""

    def test_self_arr_index_field_assign(self, e2e):
        """`self.sprites[i].y = 240` — write a sub-field of an element."""
        result = e2e.run('''
            struct OamEntry { x: u8, y: u8, tile: u8, attr: u8 }
            struct OamBuffer { sprites: [OamEntry; 8] }
            #[lowram] static mut buf: OamBuffer;

            impl OamBuffer {
                #[inline]
                fn hide(*self, idx: u8) {
                    self.sprites[idx].y = 240;
                }
            }

            #[entry]
            fn main() {
                buf.sprites[3].x = 99;
                buf.sprites[3].y = 0;
                buf.hide(3);
            }
        ''', ExpectedState(memory={
            # buf at $0200; sprites[3] starts at $0200 + 3*4 = $020C.
            0x7E020C: 99,    # x untouched
            0x7E020D: 240,   # y set by hide()
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_self_arr_index_element_assign(self, e2e):
        """`self.bytes[i] = v` — write a primitive element through auto-deref."""
        result = e2e.run('''
            struct Buf { bytes: [u8; 8] }
            #[lowram] static mut b: Buf;

            impl Buf {
                #[inline]
                fn set(*self, idx: u8, v: u8) {
                    self.bytes[idx] = v;
                }
            }

            #[entry]
            fn main() {
                b.set(0, 0x11);
                b.set(7, 0x77);
            }
        ''', ExpectedState(memory={
            0x7E0200: 0x11,
            0x7E0207: 0x77,
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_self_arr_index_field_load(self, e2e):
        """`return self.sprites[i].tile` — read a sub-field of an element."""
        result = e2e.run('''
            struct OamEntry { x: u8, y: u8, tile: u8, attr: u8 }
            struct OamBuffer { sprites: [OamEntry; 4] }
            #[lowram] static mut buf: OamBuffer;
            #[lowram] static mut out: u8;

            impl OamBuffer {
                #[inline]
                fn tile_of(*self, idx: u8) -> u8 {
                    return self.sprites[idx].tile;
                }
            }

            #[entry]
            fn main() {
                buf.sprites[2].tile = 0x42;
                out = buf.tile_of(2);
            }
        ''', ExpectedState(memory={
            # buf at $0200, sprites[2].tile at $0200 + 2*4 + 2 = $020A.
            # out lives in lowram right after buf (16 bytes) -> $0210.
            0x7E020A: 0x42,
            0x7E0210: 0x42,
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_self_arr_index_element_load(self, e2e):
        """`return self.bytes[i]` — read a primitive element through auto-deref."""
        result = e2e.run('''
            struct Buf { bytes: [u8; 4] }
            #[lowram] static mut b: Buf;
            #[lowram] static mut out: u8;

            impl Buf {
                #[inline]
                fn get(*self, idx: u8) -> u8 {
                    return self.bytes[idx];
                }
            }

            #[entry]
            fn main() {
                b.bytes[2] = 0x5A;
                out = b.get(2);
            }
        ''', ExpectedState(memory={
            0x7E0202: 0x5A,
            # out at $0200 + 4 = $0204.
            0x7E0204: 0x5A,
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_explicit_deref_array_index(self, e2e):
        """`(*p)[i] = v;` and `(*p)[i]` — explicit-deref array shape.

        Exercises the HIRDereference branch of the fix (vs. auto_deref
        HIRFieldAccess for `self.arr[..]`). Same indirect-store/load path
        but reached through `*p` rather than an implicit deref.
        """
        result = e2e.run('''
            #[lowram] static mut arr: [u8; 8];
            #[lowram] static mut out: u8;

            #[entry]
            fn main() {
                let p: *[u8; 8] = &arr;
                (*p)[3] = 0xCD;
                out = (*p)[3];
            }
        ''', ExpectedState(memory={
            0x7E0203: 0xCD,
            # `out` lives in lowram after the 8-byte `arr` -> $0208.
            0x7E0208: 0xCD,
        }))
        assert result.success, f"Failures: {result.failures}"
