# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
End-to-end tests for type aliases with pointer and function pointer types.

Verifies that type aliases resolve correctly through the full pipeline:
Source → Lexer → Parser → AST → HIR → TypeCheck → MIR → CodeGen → ASM → ROM.
"""

import pytest
from r65.tests.e2e import E2ETest, ExpectedState


class TestPointerTypeAliases:
    """Test type aliases for pointer types through full compilation."""

    @pytest.fixture
    def e2e(self):
        return E2ETest()

    def test_pointer_to_struct_alias(self, e2e):
        """type SpritePtr = *Sprite; — write through alias-typed pointer."""
        result = e2e.run('''
            struct Sprite { x: u8, y: u8 }

            type SpritePtr = *Sprite;

            #[zeropage(0x10)]
            static mut SPR: Sprite;

            fn set_pos(ptr: SpritePtr, xval @ A: u8) {
                ptr.x = xval;
            }

            #[entry]
            fn main() {
                set_pos(&SPR, 42);
            }
        ''', ExpectedState(memory={
            0x7E0010: 42,
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_pointer_to_u8_alias(self, e2e):
        """type BytePtr = *u8; — write through alias-typed pointer."""
        result = e2e.run('''
            type BytePtr = *u8;

            #[zeropage(0x10)]
            static mut VAL: u8;

            fn write_byte(ptr: BytePtr, val @ A: u8) {
                *ptr = val;
            }

            #[entry]
            fn main() {
                write_byte(&VAL, 0xAB);
            }
        ''', ExpectedState(memory={
            0x7E0010: 0xAB,
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_pointer_alias_read_struct_fields(self, e2e):
        """Read struct fields through alias-typed pointer."""
        result = e2e.run('''
            struct Pos { x: u8, y: u8 }

            type PosPtr = *Pos;

            #[zeropage(0x10)]
            static mut POS: Pos = Pos { x: 55, y: 77 };

            fn read_x(ptr: PosPtr) -> u8 {
                return ptr.x;
            }

            #[entry]
            fn main() {
                A = read_x(&POS);
            }
        ''', ExpectedState(A=55))
        assert result.success, f"Failures: {result.failures}"


class TestFunctionPointerTypeAliases:
    """Test type aliases for function pointer types with indirect calls."""

    @pytest.fixture
    def e2e(self):
        return E2ETest()

    def test_fn_pointer_alias_call(self, e2e):
        """type Callback = fn() -> u8; — call through alias-typed fn pointer."""
        result = e2e.run('''
            #[zeropage(0x02, register)]
            static mut SCRATCH0: u8;
            #[zeropage(0x04, register)]
            static mut SCRATCH1: u16;

            type Callback = fn() -> u8;

            #[zeropage(0x10)]
            static mut CB: Callback;

            fn get_answer() -> u8 {
                // asm! prevents inlining (keeps fn alive for DCE)
                asm!("NOP");
                return 42;
            }

            #[entry]
            fn main() {
                A = get_answer();
                CB = get_answer;
                A = CB();
            }
        ''', ExpectedState(A=42))
        assert result.success, f"Failures: {result.failures}"

    def test_fn_pointer_alias_param(self, e2e):
        """type Transform = fn() -> u8; — fn pointer passed as parameter."""
        result = e2e.run('''
            #[zeropage(0x02, register)]
            static mut SCRATCH0: u8;
            #[zeropage(0x04, register)]
            static mut SCRATCH1: u16;

            type Transform = fn() -> u8;

            fn get_value() -> u8 {
                // asm! prevents inlining (keeps fn alive for DCE)
                asm!("NOP");
                return 99;
            }

            fn apply(cb: Transform) -> u8 {
                return cb();
            }

            #[entry]
            fn main() {
                A = get_value();
                A = apply(get_value);
            }
        ''', ExpectedState(A=99))
        assert result.success, f"Failures: {result.failures}"


class TestTypeAliasInDeclarations:
    """Test type aliases used in various declaration contexts."""

    @pytest.fixture
    def e2e(self):
        return E2ETest()

    def test_alias_as_static_var_type(self, e2e):
        """type Word = u16; used as static variable type."""
        result = e2e.run('''
            type Word = u16;

            #[zeropage(0x10)]
            static mut VAL: Word;

            #[entry]
            fn main() {
                VAL = 0x1234;
            }
        ''', ExpectedState(memory={
            0x7E0010: [0x34, 0x12],  # Little-endian
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_alias_as_parameter_type(self, e2e):
        """type Byte = u8; used as function parameter type."""
        result = e2e.run('''
            type Byte = u8;

            fn add_bytes(a @ A: Byte, b: Byte) -> Byte {
                return a + b;
            }

            #[entry]
            fn main() {
                A = add_bytes(10, 20);
            }
        ''', ExpectedState(A=30))
        assert result.success, f"Failures: {result.failures}"

    def test_alias_as_return_type(self, e2e):
        """type Byte = u8; used as function return type."""
        result = e2e.run('''
            type Byte = u8;

            fn get_value() -> Byte {
                return 99;
            }

            #[entry]
            fn main() {
                A = get_value();
            }
        ''', ExpectedState(A=99))
        assert result.success, f"Failures: {result.failures}"

    def test_chained_alias(self, e2e):
        """type A = u8; type B = A; — chained aliases resolve correctly."""
        result = e2e.run('''
            type Byte = u8;
            type Octet = Byte;

            fn get_val() -> Octet {
                return 77;
            }

            #[entry]
            fn main() {
                A = get_val();
            }
        ''', ExpectedState(A=77))
        assert result.success, f"Failures: {result.failures}"

    def test_alias_struct_type_static(self, e2e):
        """type Player = PlayerData; used as static variable type."""
        result = e2e.run('''
            struct PlayerData { x: u8, y: u8 }
            type Player = PlayerData;

            #[zeropage(0x10)]
            static mut P: Player;

            #[entry]
            fn main() {
                P.x = 10;
                P.y = 20;
            }
        ''', ExpectedState(memory={
            0x7E0010: 10,
            0x7E0011: 20,
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_alias_enum_type(self, e2e):
        """type Dir = Direction; used in assignment."""
        result = e2e.run('''
            enum Direction { North = 0, East = 1, South = 2, West = 3 }
            type Dir = Direction;

            #[zeropage(0x10)]
            static mut FACING: Dir;

            #[entry]
            fn main() {
                FACING = Direction::South;
                A = FACING;
            }
        ''', ExpectedState(A=2))
        assert result.success, f"Failures: {result.failures}"
