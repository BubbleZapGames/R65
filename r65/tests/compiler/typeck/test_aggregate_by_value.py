"""
Tests for the restriction that arrays and structs cannot be passed or returned by value.
"""

import pytest
from r65.compiler.main import compile_string


class TestAggregateParameterRestriction:
    """Test that aggregate types (arrays, structs) cannot be passed by value."""

    def test_struct_parameter_by_value_error(self):
        """Passing a struct by value should produce a compile error."""
        source = """
        struct Player {
            x: u8,
            y: u8,
        }

                fn bad_func(player: Player) {
            A = player.x;
        }
        """
        with pytest.raises(Exception) as exc_info:
            compile_string(source)

        error_msg = str(exc_info.value)
        assert "Player" in error_msg
        assert "cannot be passed by value" in error_msg

    def test_array_parameter_by_value_error(self):
        """Passing an array by value should produce a compile error."""
        source = """
                fn bad_func(data: [u8; 256]) {
            A = data[0];
        }
        """
        with pytest.raises(Exception) as exc_info:
            compile_string(source)

        error_msg = str(exc_info.value)
        assert "[u8; 256]" in error_msg
        assert "cannot be passed by value" in error_msg

    def test_struct_pointer_parameter_ok(self):
        """Passing a struct by pointer should be allowed (type check passes)."""
        source = """
        struct Player {
            x: u8,
            y: u8,
        }

        #[zeropage(0x10)]
        static mut PTR: *Player;

                fn good_func(player @ PTR: *Player) {
            // Just verify the function signature is accepted
            A = 0;
        }
        """
        # Should compile without error
        result = compile_string(source)
        assert result is not None

    def test_array_pointer_parameter_ok(self):
        """Passing an array by pointer should be allowed (type check passes)."""
        source = """
        #[zeropage(0x10)]
        static mut PTR: *[u8];

                fn good_func(data @ PTR: *[u8]) {
            // Just verify the function signature is accepted
            A = 0;
        }
        """
        # Should compile without error
        result = compile_string(source)
        assert result is not None


class TestAggregateReturnRestriction:
    """Test that aggregate types (arrays, structs) cannot be returned by value."""

    def test_struct_return_by_value_error(self):
        """Returning a struct by value should produce a compile error."""
        source = """
        struct Player {
            x: u8,
            y: u8,
        }

                fn bad_func() -> Player {
            A = 0;
        }
        """
        with pytest.raises(Exception) as exc_info:
            compile_string(source)

        error_msg = str(exc_info.value)
        assert "Player" in error_msg
        assert "cannot be returned by value" in error_msg

    def test_array_return_by_value_error(self):
        """Returning an array by value should produce a compile error."""
        source = """
                fn bad_func() -> [u8; 8] {
            A = 0;
        }
        """
        with pytest.raises(Exception) as exc_info:
            compile_string(source)

        error_msg = str(exc_info.value)
        assert "[u8; 8]" in error_msg
        assert "cannot be returned by value" in error_msg

    def test_struct_pointer_return_ok(self):
        """Returning a struct pointer should be allowed."""
        source = """
        struct Player {
            x: u8,
            y: u8,
        }

        #[ram]
        static mut PLAYER: Player = Player { x: 0, y: 0 };

                fn good_func() -> *Player {
            return &PLAYER;
        }
        """
        # Should compile without error
        result = compile_string(source)
        assert result is not None


class TestFunctionPointerTypeRestriction:
    """Test that function pointer types cannot have aggregate params/returns."""

    def test_type_alias_with_struct_param_error(self):
        """Type alias for function with struct parameter should error."""
        source = """
        struct Player {
            x: u8,
            y: u8,
        }

        type BadCallback = fn(Player) -> u8;

        #[ram]
        static mut HANDLER: BadCallback;
        """
        with pytest.raises(Exception) as exc_info:
            compile_string(source)

        error_msg = str(exc_info.value)
        assert "Player" in error_msg
        assert "cannot be passed by value" in error_msg

    def test_type_alias_with_array_return_error(self):
        """Type alias for function returning array should error."""
        source = """
        type BadCallback = fn() -> [u8; 8];

        #[ram]
        static mut HANDLER: BadCallback;
        """
        with pytest.raises(Exception) as exc_info:
            compile_string(source)

        error_msg = str(exc_info.value)
        assert "[u8; 8]" in error_msg
        assert "cannot be returned by value" in error_msg

    def test_type_alias_with_pointer_params_ok(self):
        """Type alias for function with pointer parameters should be allowed."""
        source = """
        struct Player {
            x: u8,
            y: u8,
        }

        type GoodCallback = fn(*Player) -> u8;

        #[ram]
        static mut HANDLER: GoodCallback;
        """
        # Should compile without error
        result = compile_string(source)
        assert result is not None


class TestAggregateAssignmentRestriction:
    """Test that aggregate types (arrays, structs) cannot be assigned by value."""

    def test_struct_assignment_by_value_error(self):
        """Assigning a struct by value should produce a compile error."""
        source = """
        struct Entity {
            velocity_x: i8,
            velocity_y: i8,
        }

        #[ram]
        static mut PLAYER1: Entity;

        #[ram]
        static mut PLAYER2: Entity;

                fn main() {
            PLAYER1 = PLAYER2;
        }
        """
        with pytest.raises(Exception) as exc_info:
            compile_string(source)

        error_msg = str(exc_info.value)
        assert "Entity" in error_msg
        assert "cannot" in error_msg.lower()

    def test_array_assignment_by_value_error(self):
        """Assigning an array by value should produce a compile error."""
        source = """
        #[ram]
        static mut BUFFER1: [u8; 256];

        #[ram]
        static mut BUFFER2: [u8; 256];

                fn main() {
            BUFFER1 = BUFFER2;
        }
        """
        with pytest.raises(Exception) as exc_info:
            compile_string(source)

        error_msg = str(exc_info.value)
        assert "[u8; 256]" in error_msg
        assert "cannot" in error_msg.lower()

    def test_struct_field_assignment_ok(self):
        """Assigning individual struct fields should be allowed."""
        source = """
        struct Entity {
            x: u8,
            y: u8,
        }

        #[ram]
        static mut PLAYER1: Entity;

        #[ram]
        static mut PLAYER2: Entity;

                fn copy_entity() {
            PLAYER1.x = PLAYER2.x;
            PLAYER1.y = PLAYER2.y;
        }
        """
        result = compile_string(source)
        assert result is not None

    def test_array_element_assignment_ok(self):
        """Assigning individual array elements should be allowed."""
        source = """
        #[ram]
        static mut BUFFER1: [u8; 8];

        #[ram]
        static mut BUFFER2: [u8; 8];

                fn copy_element() {
            BUFFER1[0] = BUFFER2[0];
        }
        """
        result = compile_string(source)
        assert result is not None


class TestPrimitiveTypesStillWork:
    """Verify that primitive types are not affected by the restriction."""

    def test_u8_parameter_ok(self):
        """u8 parameter should work."""
        source = """
                fn add(a: u8, b: u8) -> u8 {
            A = a + b;
            return A;
        }
        """
        result = compile_string(source)
        assert result is not None

    def test_u16_return_ok(self):
        """u16 return should work."""
        source = """
                fn get_value() -> u16 {
            A = 1000;
            return A;
        }
        """
        result = compile_string(source)
        assert result is not None

    def test_register_parameter_ok(self):
        """Register parameter should work."""
        source = """
                fn set_value(value @ A: u8) {
            A = A + 1;
        }
        """
        result = compile_string(source)
        assert result is not None
