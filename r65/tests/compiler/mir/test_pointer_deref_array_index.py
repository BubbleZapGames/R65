# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Unit tests for MIRBuilder.try_pointer_deref_array_base and the array-index
lowering through a pointer-dereferenced struct field.

Before the fix, the MIR builder rejected `self.arr[i]` shapes with:

    MIR error: Array base must be a static array or struct field, got:
               HIRFieldAccess  /  HIRDereference

`resolve_array_base_memloc` only handled bare statics and statically-located
struct fields; arrays reached through an auto_deref HIRFieldAccess
(`self.arr[..]`), an explicit HIRDereference (`(*p)[..]`), or a chain of
field accesses where one ancestor is auto_deref (`self.outer.inner_arr[..]`)
had no static base to fold into a MemoryLocation. The fix walks the
HIRFieldAccess chain, accumulates each non-deref field's `field_offset`,
and at the first auto_deref'd field (or HIRDereference) returns the
underlying pointer plus the accumulated constant offset. Codegen emits
LoadIndirect / StoreIndirect with the offset folded into Y.

These tests exercise the contract at the MIR/codegen seam without spinning
up the emulator: each source program is one that pre-fix would raise
MIRLoweringError. Post-fix, all six shapes compile to indirect-indexed
addressing through the underlying pointer.
"""

import re

from r65.compiler.errors import MIRLoweringError
from r65.compiler.main import compile_string


# Indirect-indexed store/load against a stack-relative pointer slot:
#   near self:  STA (d,S),Y    LDA (d,S),Y
#   far self:   STA [d,S],Y    LDA [d,S],Y
# The slot offset varies — only the bracket shape and trailing ,Y matter.
_STORE_INDIRECT_INDEXED = re.compile(r'\bSTA\s+[\[\(]\$[0-9A-F]+,S[\]\)],Y\b')
_LOAD_INDIRECT_INDEXED = re.compile(r'\bLDA\s+[\[\(]\$[0-9A-F]+,S[\]\)],Y\b')


class TestPointerDerefArrayIndex:
    """Pointer-deref'd array bases reach codegen via indirect-indexed addressing."""

    def test_self_arr_index_field_assign(self):
        """`self.sprites[i].y = 240` — write a sub-field of an element.

        HIR shape: HIRFieldAccess(.y, base=HIRArrayIndex(array=HIRFieldAccess
        (sprites, auto_deref=True, base=self))). try_pointer_deref_array_base
        matches the auto_deref field directly and emits StoreIndirect with
        offsetof(y) folded into Y alongside (idx * 4).
        """
        asm = compile_string('''
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
            fn main() { buf.hide(3); }
        ''', cfg_options=['snes'])
        assert _STORE_INDIRECT_INDEXED.search(asm), (
            f"expected `STA (d,S),Y` indirect-indexed store, asm:\n{asm}"
        )

    def test_self_arr_index_element_assign(self):
        """`self.bytes[i] = v` — write a primitive element through auto-deref."""
        asm = compile_string('''
            struct Buf { bytes: [u8; 8] }
            #[lowram] static mut b: Buf;

            impl Buf {
                #[inline]
                fn set(*self, idx: u8, v: u8) {
                    self.bytes[idx] = v;
                }
            }

            #[entry]
            fn main() { b.set(0, 0x11); }
        ''', cfg_options=['snes'])
        assert _STORE_INDIRECT_INDEXED.search(asm), (
            f"expected `STA (d,S),Y` indirect-indexed store, asm:\n{asm}"
        )

    def test_self_arr_index_field_load(self):
        """`return self.sprites[i].tile` — read a sub-field of an element."""
        asm = compile_string('''
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
            fn main() { out = buf.tile_of(2); }
        ''', cfg_options=['snes'])
        assert _LOAD_INDIRECT_INDEXED.search(asm), (
            f"expected `LDA (d,S),Y` indirect-indexed load, asm:\n{asm}"
        )

    def test_self_arr_index_element_load(self):
        """`return self.bytes[i]` — read a primitive element through auto-deref."""
        asm = compile_string('''
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
            fn main() { out = b.get(2); }
        ''', cfg_options=['snes'])
        assert _LOAD_INDIRECT_INDEXED.search(asm), (
            f"expected `LDA (d,S),Y` indirect-indexed load, asm:\n{asm}"
        )

    def test_chained_auto_deref_array(self):
        """`self.inner.arr[i] = v` — pointer-deref at an *inner* field.

        HIR chain: arr (auto_deref=False, offset=0) wraps inner
        (auto_deref=True, offset=1) wraps HIRIdentifier(self, *Outer). The
        walker peels the outer wrapper, accumulates its offset (0), reaches
        the auto_deref'd inner, and returns (self, 1). The const_offset (1)
        plus the runtime `i * 1` is folded into Y. The emitted `ADC #$01`
        is the smoking gun that the inner field's offset rode along.
        """
        asm = compile_string('''
            struct Inner { arr: [u8; 8] }
            struct Outer { pad: u8, inner: Inner }
            #[lowram] static mut buf: Outer;

            impl Outer {
                #[inline]
                fn poke(*self, i: u8, v: u8) {
                    self.inner.arr[i] = v;
                }
            }

            #[entry]
            fn main() { buf.poke(3, 0xCD); }
        ''', cfg_options=['snes'])
        assert _STORE_INDIRECT_INDEXED.search(asm), (
            f"expected `STA (d,S),Y` indirect-indexed store, asm:\n{asm}"
        )
        # `inner` lives at offset 1 within Outer (after `pad: u8`). That
        # constant must be added to the scaled index before the store —
        # look for the ADC #$01 alongside the index TAY.
        assert re.search(r'ADC\s+#\$01\b', asm), (
            f"expected the inner-field offset (#$01) to be folded into Y, "
            f"asm:\n{asm}"
        )

    def test_explicit_deref_array_index(self):
        """`(*p)[i] = v; out = (*p)[i];` — explicit-deref array shape.

        Exercises the HIRDereference branch (vs auto_deref HIRFieldAccess
        for `self.arr[..]`). try_pointer_deref_array_base must match the
        HIRDereference node and lift its `.pointer` as the underlying
        pointer with zero base_field_offset.
        """
        asm = compile_string('''
            #[lowram] static mut arr: [u8; 8];
            #[lowram] static mut out: u8;

            #[entry]
            fn main() {
                let p: *[u8; 8] = &arr;
                (*p)[3] = 0xCD;
                out = (*p)[3];
            }
        ''', cfg_options=['snes'])
        assert _STORE_INDIRECT_INDEXED.search(asm), (
            f"expected `STA (d,S),Y` indirect-indexed store, asm:\n{asm}"
        )
        assert _LOAD_INDIRECT_INDEXED.search(asm), (
            f"expected `LDA (d,S),Y` indirect-indexed load, asm:\n{asm}"
        )


class TestStaticArrayBaseNotPerturbed:
    """Regression: a chain with no auto_deref still takes the static path."""

    def test_static_struct_array_field_unchanged(self):
        """`BUF.arr[i] = v` (purely static) must NOT route through indirect
        addressing — it should use absolute (long) indexed addressing
        through the assembler's resolved label.
        """
        asm = compile_string('''
            struct Buf { arr: [u8; 8] }
            #[lowram] static mut BUF: Buf;

            #[entry]
            fn main() {
                BUF.arr[3] = 0xAB;
            }
        ''', cfg_options=['snes'])
        # Must NOT emit indirect-indexed addressing for a purely static base.
        assert not _STORE_INDIRECT_INDEXED.search(asm), (
            f"static array base wrongly routed through indirect path:\n{asm}"
        )
