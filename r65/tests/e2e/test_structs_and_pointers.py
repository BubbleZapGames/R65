# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
End-to-end tests for struct and pointer runtime operations.

Field-offset/layout is unit-tested in compiler/hir/test_hir_builder.py.
This file covers runtime codegen: field reads/writes, pointer auto-deref,
struct initializers, arrays of structs, and pointer-loop indexed reads.
"""

from pathlib import Path

from r65.tests.e2e import ExpectedState

STDLIB_DIR = Path(__file__).parent.parent.parent.parent / "stdlib"
SNESLIB_PATH = STDLIB_DIR / "sneslib.r65"
MATH_PATH = STDLIB_DIR / "math.r65"


class TestStructFieldAccess:
    """Runtime read/write through struct field access."""

    def test_struct_with_u16_field(self, e2e):
        """Multi-byte field offset writes correct bytes."""
        result = e2e.run('''
            struct Entity { kind: u8, health: u16 }

            #[zeropage(0x10)]
            static mut ENT: Entity;

            #[entry]
            fn main() {
                ENT.kind = 0x01;
                ENT.health = 500;
            }
        ''', ExpectedState(memory={
            0x7E0010: 0x01,
            0x7E0011: [0xF4, 0x01],
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_struct_init_literal(self, e2e):
        """Static struct with literal initializer + runtime field reads."""
        result = e2e.run('''
            struct Pos { x: u8, y: u8 }

            #[zeropage(0x10)]
            static mut POS: Pos = Pos { x: 42, y: 99 };

            #[zeropage(0x20)]
            static mut RX: u8;
            #[zeropage(0x21)]
            static mut RY: u8;

            #[entry]
            fn main() {
                RX = POS.x;
                RY = POS.y;
            }
        ''', ExpectedState(memory={
            0x7E0010: 42, 0x7E0011: 99,
            0x7E0020: 42, 0x7E0021: 99,
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_struct_field_arithmetic(self, e2e):
        """Arithmetic on struct fields round-trips correctly."""
        result = e2e.run('''
            struct Pos { x: u8, y: u8 }

            #[zeropage(0x10)]
            static mut POS: Pos = Pos { x: 10, y: 20 };

            #[entry]
            fn main() {
                POS.x = POS.x + 5;
                A = POS.x;
            }
        ''', ExpectedState(A=15))
        assert result.success, f"Failures: {result.failures}"


class TestStructPointers:
    """Pointer auto-deref to struct fields."""

    def test_struct_pointer_read_nonzero_offset(self, e2e):
        """Reading the second field exercises non-zero offset codegen."""
        result = e2e.run('''
            struct Pos { x: u8, y: u8 }

            #[zeropage(0x10)]
            static mut POS: Pos = Pos { x: 55, y: 77 };

            fn read_y(ptr: *Pos) -> u8 { return ptr.y; }

            #[entry]
            fn main() {
                A = read_y(&POS);
            }
        ''', ExpectedState(A=77))
        assert result.success, f"Failures: {result.failures}"

    def test_struct_pointer_field_write(self, e2e):
        """Write through pointer; A-register binding."""
        result = e2e.run('''
            struct Pos { x: u8, y: u8 }

            #[zeropage(0x10)]
            static mut POS: Pos;

            fn set_x(ptr: *Pos, val @ A: u8) { ptr.x = val; }
            fn set_y(ptr: *Pos, val @ A: u8) { ptr.y = val; }

            #[entry]
            fn main() {
                set_x(&POS, 33);
                set_y(&POS, 44);
            }
        ''', ExpectedState(memory={
            0x7E0010: 33,
            0x7E0011: 44,
        }))
        assert result.success, f"Failures: {result.failures}"


class TestArrayOfStructs:
    """Array indexing × struct field offset."""

    def test_array_of_structs_write_read(self, e2e):
        """Array of 2-byte structs: indexed write and read."""
        result = e2e.run('''
            struct Item { id: u8, count: u8 }

            #[zeropage(0x10)]
            static mut ITEMS: [Item; 4];

            #[zeropage(0x20)]
            static mut RESULT: u8;

            #[entry]
            fn main() {
                ITEMS[0].id = 1;
                ITEMS[0].count = 10;
                ITEMS[1].id = 2;
                ITEMS[1].count = 20;
                ITEMS[2].id = 3;
                ITEMS[2].count = 30;
                ITEMS[3].id = 4;
                ITEMS[3].count = 40;
                RESULT = ITEMS[1].count;
                A = RESULT;
            }
        ''', ExpectedState(
            A=20,
            memory={
                0x7E0010: [1, 10, 2, 20, 3, 30, 4, 40],
            }
        ))
        assert result.success, f"Failures: {result.failures}"

    def test_large_struct_variable_index(self, e2e):
        """Regression: >16-byte struct array with a VARIABLE index.

        Structs larger than 16 bytes use the mul16() runtime multiply for
        the index. The scaled index (index * struct_size) is a byte offset
        that can exceed 255 (16 * 18 = 288), so it must be u16 — NOT the
        struct element type. Previously the offset vreg was sized as the
        struct, producing an 8-bit store / 16-bit load mismatch: the index
        high byte was garbage and the access landed at a random address.

        Mirrors the classickong enemy[i].field pattern (18-byte struct,
        u8 index parameter, compound read-modify-write, call with a
        field argument).
        """
        result = e2e.run(f'''
            include!("{SNESLIB_PATH}")
            include!("{MATH_PATH}")

            struct E {{
                a: u8, b: u8, c: u8, d: i8,
                e: u8, f: u8, g: u8, h: u8,
                i0: u16, i1: u16, i2: u16,
                j: u8, k: u8, l: u8, m: u8
            }}

            #[lowram(0x400)]
            static mut arr: [E; 16];

            #[lowram(0x300)]
            static mut OUT: [u8; 8];

            fn touch(v @ A: u8) -> u8 {{ return v + 1; }}

            fn proc(i: u8) {{
                arr[i].d = 5;
                arr[i].d = 0 - arr[i].d;        // compound RMW -> 251
                let t: u8 = touch(arr[i].a);    // call w/ indexed-field arg
                arr[i].b = t;
                arr[i].c = arr[i].b;            // field-to-field, two indexes
            }}

            #[entry]
            fn main() {{
                arr[15].a = 10;   // last slot: index*18 = 270 > 255 (needs u16)
                proc(15);
                OUT[0] = arr[15].d;
                OUT[1] = arr[15].b;
                OUT[2] = arr[15].c;
                OUT[3] = arr[15].a;
            }}
        ''', ExpectedState(memory={
            # arr base 0x7E0400, arr[15] = +15*18 = +270 => 0x7E050E
            0x7E0300: [251, 11, 11, 10],
            0x7E050E: 10,    # arr[15].a
            0x7E0511: 251,   # arr[15].d (offset 3)
        }), max_instructions=300000)
        assert result.success, f"Failures: {result.failures}"

    def test_sequential_same_index_reuses_x(self, e2e):
        """X-index reuse: consecutive accesses to the same (array, index)
        keep X live instead of recomputing index*size + Move-to-X.

        Correctness is the contract here (the codegen win is verified
        separately). Covers the cases the reuse cache must get right:
          - 3 sequential field stores reuse one scaled index
          - read-after-write on the same element
          - a call between accesses (clobbers X -> must recompute)
          - reassigning the index (i = i + 1 -> must recompute, hit new slot)
          - the 16-byte shift-and-add path, not just the mul16 path
        """
        result = e2e.run(f'''
            include!("{SNESLIB_PATH}")
            include!("{MATH_PATH}")

            struct E {{
                a: u8, b: u8, c: u8, d: u8, e: u8, f: u8, g: u8, h: u8,
                i0: u16, i1: u16, i2: u16, j: u8, k: u8, l: u8, m: u8
            }}
            #[lowram(0x400)] static mut arr: [E; 16];
            #[lowram(0x300)] static mut OUT: [u8; 8];

            fn bump(v @ A: u8) -> u8 {{ return v + 7; }}

            fn work(i: u8) {{
                arr[i].a = 11; arr[i].b = 22; arr[i].c = 33;  // reuse X x3
                arr[i].e = arr[i].a;                          // read-after-write
                let t: u8 = bump(arr[i].a);                   // call clobbers X
                arr[i].f = t;                                 // must recompute
                arr[i].h = 1; i = i + 1; arr[i].j = 2;        // index changed
            }}

            #[entry]
            fn main() {{
                work(3);                       // touches arr[3].* and arr[4].j
                OUT[0] = arr[3].a;             // 11
                OUT[1] = arr[3].c;             // 33
                OUT[2] = arr[3].e;             // 11 (== arr[3].a)
                OUT[3] = arr[3].f;             // 18 (bump(11))
                OUT[4] = arr[3].h;             // 1
                OUT[5] = arr[4].j;             // 2  (post-increment index)
            }}
        ''', ExpectedState(memory={
            0x7E0300: [11, 33, 11, 18, 1, 2],
        }), max_instructions=300000)
        assert result.success, f"Failures: {result.failures}"

    def test_pointer_loop_indexed_read(self, e2e):
        """Regression: LDA (dp),Y indirect indexed read with Y as loop counter.

        The compiler must not coalesce other variables into Y when
        LoadIndirect/StoreIndirect instructions exist in the same block,
        as the codegen uses Y for the index operand.
        """
        result = e2e.run('''
            #[zeropage(0x10)]
            static mut PTR: *u8;

            #[lowram(0x200)]
            static mut DATA: [u8; 4];

            #[lowram(0x300)]
            static mut RESULT: u8;

            #[entry]
            fn main() {
                DATA[0] = 10;
                DATA[1] = 20;
                DATA[2] = 30;
                DATA[3] = 40;
                PTR = &DATA as *u8;

                let mut acc: u8 = 0;
                let mut i: u16 = 0;
                loop {
                    if i == 4 { break; }
                    acc = acc + PTR[i];
                    i++;
                }
                RESULT = acc;
            }
        ''', ExpectedState(memory={0x7E0300: 100}))
        assert result.success, f"Failures: {result.failures}"
