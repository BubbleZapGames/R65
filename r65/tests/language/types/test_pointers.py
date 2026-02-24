"""Tests for pointer types: *T (near) and far *T (far)."""

import pytest
from r65.compiler.frontend import ast
from r65.compiler.frontend.parser import ParseError
from r65.tests.language.common import parse_type, parse_static, parse_function


class TestPointerTypes:
    """Tests for pointer type parsing."""

    def test_near_pointer(self):
        """Test *T pointer type (implied near)."""
        t = parse_type("*u8")
        assert isinstance(t, ast.PointerType)
        assert t.is_far is False
        assert isinstance(t.pointee_type, ast.BasicType)
        assert t.pointee_type.name == "u8"

    def test_far_pointer(self):
        """Test far *T pointer type."""
        t = parse_type("far *u8")
        assert isinstance(t, ast.PointerType)
        assert t.is_far is True
        assert t.pointee_type.name == "u8"

    def test_explicit_near_pointer(self):
        """Test near *T pointer type (explicit near)."""
        t = parse_type("near *u8")
        assert isinstance(t, ast.PointerType)
        assert t.is_far is False
        assert t.pointee_type.name == "u8"

    def test_pointer_to_struct(self):
        """Test pointer to named type."""
        t = parse_type("*Player")
        assert isinstance(t, ast.PointerType)
        # Named types are parsed as BasicType
        assert isinstance(t.pointee_type, ast.BasicType)
        assert t.pointee_type.name == "Player"

    def test_pointer_to_array(self):
        """Test pointer to sized array type."""
        t = parse_type("*[u8; 256]")
        assert isinstance(t, ast.PointerType)
        assert t.is_far is False
        assert isinstance(t.pointee_type, ast.ArrayType)

    def test_pointer_to_slice_is_parse_error(self):
        """Test that *[u8] (unsized slice pointer) is now a parse error."""
        with pytest.raises(ParseError):
            parse_type("*[u8]")

    def test_far_pointer_to_slice_is_parse_error(self):
        """Test that far *[u8] (unsized slice pointer) is now a parse error."""
        with pytest.raises(ParseError):
            parse_type("far *[u8]")


class TestPointerOperations:
    """Tests for pointer operations."""

    def test_dereference(self):
        """Test pointer dereference with *."""
        func = parse_function("fn test() { let v: u8 = *ptr; }")
        let_stmt = func.body.statements[0]
        assert isinstance(let_stmt.initializer, ast.Dereference)

    def test_address_of(self):
        """Test address-of operator &."""
        func = parse_function("fn test() { let p: *u8 = &value; }")
        let_stmt = func.body.statements[0]
        assert isinstance(let_stmt.initializer, ast.AddressOf)
        # The type should be a pointer type
        assert isinstance(let_stmt.var_type, ast.PointerType)

    def test_indexed_pointer(self):
        """Test pointer indexing ptr[Y]."""
        func = parse_function("fn test() { let v: u8 = ptr[Y]; }")
        let_stmt = func.body.statements[0]
        # This could be ArrayIndex or IndexedPointer depending on implementation
        assert let_stmt.initializer is not None


class TestPointerStatics:
    """Tests for pointer static declarations with canonical name: *T syntax."""

    def test_pointer_static(self):
        """Test pointer in static declaration with name: *type syntax."""
        static = parse_static("#[zeropage] static mut PTR: *u8;")
        assert isinstance(static.var_type, ast.PointerType)
        assert static.var_type.is_far is False
        assert static.name == "PTR"

    def test_far_pointer_static(self):
        """Test far pointer in static declaration.

        The far modifier on the TYPE (far *u16) makes the pointer type far.
        """
        static = parse_static("#[ram] static mut FAR_PTR: far *u16;")
        assert isinstance(static.var_type, ast.PointerType)
        assert static.var_type.is_far is True
        assert static.name == "FAR_PTR"

    def test_near_pointer_static(self):
        """Test explicit near pointer in static declaration.

        The near modifier on the TYPE (near *u8) makes the pointer type near.
        """
        static = parse_static("#[lowram] static mut NEAR_PTR: near *u8;")
        assert isinstance(static.var_type, ast.PointerType)
        assert static.var_type.is_far is False
        assert static.name == "NEAR_PTR"


class TestPointerParameters:
    """Tests for pointer function parameters with canonical name: *T syntax."""

    def test_pointer_param(self):
        """Test pointer parameter with name: *type syntax."""
        func = parse_function("fn copy(src: *u8, dst: *u8) { }")
        assert len(func.params) == 2
        assert func.params[0].name == "src"
        assert isinstance(func.params[0].param_type, ast.PointerType)
        assert func.params[1].name == "dst"
        assert isinstance(func.params[1].param_type, ast.PointerType)

    def test_far_pointer_param(self):
        """Test far pointer parameter."""
        func = parse_function("fn read(data: far *u8) { }")
        assert func.params[0].name == "data"
        assert isinstance(func.params[0].param_type, ast.PointerType)
        assert func.params[0].param_type.is_far is True

    def test_pointer_param(self):
        """Test pointer parameter for array data uses *u8."""
        func = parse_function("fn process(data: far *u8) { }")
        assert func.params[0].name == "data"
        assert isinstance(func.params[0].param_type, ast.PointerType)
        assert func.params[0].param_type.is_far is True
        assert isinstance(func.params[0].param_type.pointee_type, ast.BasicType)


class TestPointerStructFields:
    """Tests for pointer struct fields with canonical name: *T syntax."""

    def test_pointer_field(self):
        """Test pointer field with name: *type syntax."""
        from r65.tests.language.common import parse_struct
        struct = parse_struct("struct Node { next: *Node, data: u8, }")
        assert len(struct.fields) == 2
        assert struct.fields[0].name == "next"
        assert isinstance(struct.fields[0].field_type, ast.PointerType)
        assert struct.fields[1].name == "data"

    def test_far_pointer_field(self):
        """Test far pointer field."""
        from r65.tests.language.common import parse_struct
        struct = parse_struct("struct FarRef { data: far *u8, }")
        assert struct.fields[0].name == "data"
        assert isinstance(struct.fields[0].field_type, ast.PointerType)
        assert struct.fields[0].field_type.is_far is True


class TestPointerLet:
    """Tests for pointer let statements with canonical name: *T syntax."""

    def test_pointer_let(self):
        """Test pointer let with name: *type syntax."""
        func = parse_function("fn test() { let ptr: *u8 = 0x2000; }")
        let_stmt = func.body.statements[0]
        assert let_stmt.name == "ptr"
        assert isinstance(let_stmt.var_type, ast.PointerType)
        assert let_stmt.var_type.is_far is False

    def test_far_pointer_let(self):
        """Test far pointer let."""
        func = parse_function("fn test() { let ptr: far *u8 = addr; }")
        let_stmt = func.body.statements[0]
        assert let_stmt.name == "ptr"
        assert isinstance(let_stmt.var_type, ast.PointerType)
        assert let_stmt.var_type.is_far is True

    def test_mut_pointer_let(self):
        """Test mutable pointer let."""
        func = parse_function("fn test() { let mut ptr: *u8 = 0x2000; }")
        let_stmt = func.body.statements[0]
        assert let_stmt.is_mut is True
        assert let_stmt.name == "ptr"
        assert isinstance(let_stmt.var_type, ast.PointerType)


class TestPointerCasting:
    """Tests for pointer type casting."""

    def test_cast_to_pointer(self):
        """Test casting to pointer type."""
        func = parse_function("fn test() { let x: u16 = addr as *u8; }")
        let_stmt = func.body.statements[0]
        assert isinstance(let_stmt.initializer, ast.TypeCast)
        assert isinstance(let_stmt.initializer.target_type, ast.PointerType)
        assert let_stmt.initializer.target_type.is_far is False

    def test_cast_to_far_pointer(self):
        """Test casting to far pointer type."""
        func = parse_function("fn test() { let x: u16 = addr as far *u8; }")
        let_stmt = func.body.statements[0]
        assert isinstance(let_stmt.initializer, ast.TypeCast)
        assert isinstance(let_stmt.initializer.target_type, ast.PointerType)
        assert let_stmt.initializer.target_type.is_far is True



class TestPatternSideForm:
    """Tests verifying that the pattern-side form (*name: T) still parses.

    Both forms are valid in R65:
    - Type-side (canonical): name: *T
    - Pattern-side: *name: T

    These tests ensure backwards compatibility for the pattern-side form.
    """

    def test_pattern_side_static(self):
        """Pattern-side *name: T in static declarations."""
        static = parse_static("#[zeropage] static mut *PTR: u8;")
        assert isinstance(static.var_type, ast.PointerType)
        assert static.name == "PTR"

    def test_pattern_side_param(self):
        """Pattern-side *name: T in function parameters."""
        func = parse_function("fn copy(*src: u8, *dst: u8) { }")
        assert len(func.params) == 2
        assert isinstance(func.params[0].param_type, ast.PointerType)
        assert isinstance(func.params[1].param_type, ast.PointerType)

    def test_pattern_side_far_param(self):
        """Pattern-side far *name: T in function parameters."""
        func = parse_function("fn read(far *data: u8) { }")
        assert isinstance(func.params[0].param_type, ast.PointerType)
        assert func.params[0].param_type.is_far is True

    def test_pattern_side_let(self):
        """Pattern-side let *name: T in let statements."""
        func = parse_function("fn test() { let *ptr: u8 = 0x2000; }")
        let_stmt = func.body.statements[0]
        assert let_stmt.name == "ptr"
        assert isinstance(let_stmt.var_type, ast.PointerType)

    def test_pattern_side_far_let(self):
        """Pattern-side let far *name: T in let statements."""
        func = parse_function("fn test() { let far *ptr: u8 = addr; }")
        let_stmt = func.body.statements[0]
        assert isinstance(let_stmt.var_type, ast.PointerType)
        assert let_stmt.var_type.is_far is True

    def test_pattern_side_struct_field(self):
        """Pattern-side *name: T in struct fields."""
        from r65.tests.language.common import parse_struct
        struct = parse_struct("struct Node { *next: Node, data: u8, }")
        assert isinstance(struct.fields[0].field_type, ast.PointerType)
        assert struct.fields[0].name == "next"

    def test_pattern_side_far_struct_field(self):
        """Pattern-side far *name: T in struct fields."""
        from r65.tests.language.common import parse_struct
        struct = parse_struct("struct FarRef { far *data: u8, }")
        assert isinstance(struct.fields[0].field_type, ast.PointerType)
        assert struct.fields[0].field_type.is_far is True

    def test_pattern_side_mut_let(self):
        """Pattern-side let mut *name: T in let statements."""
        func = parse_function("fn test() { let mut *ptr: u8 = 0x2000; }")
        let_stmt = func.body.statements[0]
        assert let_stmt.is_mut is True
        assert isinstance(let_stmt.var_type, ast.PointerType)
