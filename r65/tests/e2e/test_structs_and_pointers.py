# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
End-to-end tests for struct and pointer operations.

Tests struct field access, pointer auto-deref, multi-byte fields, and arrays of structs.
"""

from r65.tests.e2e import ExpectedState


class TestStructFieldAccess:
    """Test struct field read/write operations."""

    def test_struct_field_read_write(self, e2e):
        """Test writing and reading struct fields in zeropage.

        Note: X=field goes through A (65816 limitation), so load X first, then A.
        """
        result = e2e.run('''
            struct Pos { x: u8, y: u8 }

            #[zeropage(0x10)]
            static mut POS: Pos;

            #[zeropage(0x20)]
            static mut RX: u8;
            #[zeropage(0x21)]
            static mut RY: u8;

            #[entry]
            fn main() {
                POS.x = 100;
                POS.y = 200;
                // Store to memory to avoid A clobber issue
                RX = POS.x;
                RY = POS.y;
            }
        ''', ExpectedState(
            memory={
                0x7E0010: 100, 0x7E0011: 200,
                0x7E0020: 100, 0x7E0021: 200,
            }
        ))
        assert result.success, f"Failures: {result.failures}"

    def test_struct_with_u16_field(self, e2e):
        """Test struct with mixed u8/u16 fields."""
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
            0x7E0010: 0x01,            # kind
            0x7E0011: [0xF4, 0x01],    # health=500 LE
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_struct_init_literal(self, e2e):
        """Test struct initialization with literal values."""
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
        ''', ExpectedState(
            memory={
                0x7E0010: 42, 0x7E0011: 99,
                0x7E0020: 42, 0x7E0021: 99,
            }
        ))
        assert result.success, f"Failures: {result.failures}"

    def test_struct_field_arithmetic(self, e2e):
        """Test arithmetic on struct fields."""
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
    """Test struct access through pointers."""

    def test_struct_pointer_field_read(self, e2e):
        """Test reading struct fields through a pointer."""
        result = e2e.run('''
            struct Pos { x: u8, y: u8 }

            #[zeropage(0x10)]
            static mut POS: Pos = Pos { x: 55, y: 77 };

            fn read_x(ptr: *Pos) -> u8 {
                return ptr.x;
            }

            #[entry]
            fn main() {
                A = read_x(&POS);
            }
        ''', ExpectedState(A=55))
        assert result.success, f"Failures: {result.failures}"

    def test_struct_pointer_field_read_second(self, e2e):
        """Test reading second field through a pointer (non-zero offset)."""
        result = e2e.run('''
            struct Pos { x: u8, y: u8 }

            #[zeropage(0x10)]
            static mut POS: Pos = Pos { x: 55, y: 77 };

            fn read_y(ptr: *Pos) -> u8 {
                return ptr.y;
            }

            #[entry]
            fn main() {
                A = read_y(&POS);
            }
        ''', ExpectedState(A=77))
        assert result.success, f"Failures: {result.failures}"

    def test_struct_pointer_field_write(self, e2e):
        """Test writing struct fields through a pointer using A register."""
        result = e2e.run('''
            struct Pos { x: u8, y: u8 }

            #[zeropage(0x10)]
            static mut POS: Pos;

            fn set_x(ptr: *Pos, val @ A: u8) {
                ptr.x = val;
            }

            fn set_y(ptr: *Pos, val @ A: u8) {
                ptr.y = val;
            }

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
    """Test arrays of structs with field access."""

    def test_array_of_structs_write_read(self, e2e):
        """Test writing and reading from an array of structs."""
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
                // Read back second item count
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

    def test_struct_three_fields(self, e2e):
        """Test struct with 3 fields."""
        result = e2e.run('''
            struct RGB { r: u8, g: u8, b: u8 }

            #[zeropage(0x10)]
            static mut COLOR: RGB;

            #[entry]
            fn main() {
                COLOR.r = 0xFF;
                COLOR.g = 0x80;
                COLOR.b = 0x00;
            }
        ''', ExpectedState(memory={
            0x7E0010: [0xFF, 0x80, 0x00],
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_pointer_loop_indexed_read(self, e2e):
        """Regression: LoadIndirect via pointer uses Y as index register.

        LDA (dp),Y / LDA [dp],Y use Y as the index register. The compiler
        must not coalesce other variables into Y when LoadIndirect/StoreIndirect
        instructions exist in the same block, as the codegen uses Y for the
        index operand.
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

                let acc: u8 = 0;
                let i: u16 = 0;
                loop {
                    if i == 4 { break; }
                    acc = acc + PTR[i];
                    i++;
                }
                RESULT = acc;
            }
        ''', ExpectedState(
            memory={0x7E0300: 100}
        ))
        assert result.success, f"Failures: {result.failures}"
