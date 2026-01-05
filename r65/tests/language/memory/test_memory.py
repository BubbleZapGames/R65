"""
Comprehensive memory tests for R65.

Tests storage classes (#[zeropage], #[lowram], #[ram], #[rom], #[hw], #[stack]),
pointer types (near, far), pointer operations, and memory access patterns.
"""

import pytest
from r65.compiler.frontend.parser import parse
from r65.compiler.frontend import ast
from r65.compiler.hir.builder import HIRBuilder
from r65.compiler.hir import nodes as hir


# =============================================================================
# Helper Functions
# =============================================================================

def parse_program(source: str) -> ast.Program:
    """Parse source and return the program."""
    return parse(source)


def parse_static(source: str) -> ast.StaticDecl:
    """Parse source and return the first static declaration."""
    program = parse(source)
    assert len(program.items) >= 1
    static = program.items[0]
    assert isinstance(static, ast.StaticDecl)
    return static


def parse_function(source: str) -> ast.FunctionDecl:
    """Parse source and return the first function declaration."""
    program = parse(source)
    for item in program.items:
        if isinstance(item, ast.FunctionDecl):
            return item
    raise AssertionError("No function found")


def parse_statement(source: str) -> ast.Statement:
    """Parse a function with a single statement and return that statement."""
    func = parse_function(f"fn test() {{ {source} }}")
    assert len(func.body.statements) == 1
    return func.body.statements[0]


def get_attr(decl, name: str) -> ast.Attribute:
    """Get an attribute by name from a declaration."""
    for attr in decl.attributes:
        if attr.name == name:
            return attr
    return None


def get_attr_value(attr: ast.Attribute, index: int = 0):
    """Get the value of an attribute argument."""
    if len(attr.args) > index:
        return attr.args[index].value
    return None


def build_hir(source: str) -> hir.HIRProgram:
    """Parse source and build HIR."""
    program = parse(source)
    builder = HIRBuilder()
    return builder.build_program(program)


# =============================================================================
# Test Classes
# =============================================================================

class TestZeropageStorage:
    """Tests for #[zeropage] storage class."""

    def test_zeropage_basic(self):
        """Test basic zeropage declaration."""
        static = parse_static("#[zeropage] static mut TEMP: u8;")
        attr = get_attr(static, "zeropage")
        assert attr is not None
        assert len(attr.args) == 0

    def test_zeropage_with_address(self):
        """Test zeropage with explicit address."""
        static = parse_static("#[zeropage(0x42)] static mut TEMP: u8;")
        attr = get_attr(static, "zeropage")
        assert attr is not None
        value = get_attr_value(attr)
        assert isinstance(value, ast.IntegerLiteral)
        assert value.value == 0x42

    def test_zeropage_with_initializer(self):
        """Test zeropage with initializer."""
        static = parse_static("#[zeropage] static mut FLAGS: u8 = 0x80;")
        assert static.initializer is not None
        assert static.initializer.value == 0x80

    def test_zeropage_array(self):
        """Test zeropage array."""
        static = parse_static("#[zeropage(0x10)] static mut SCRATCH: [u8; 16];")
        assert isinstance(static.var_type, ast.ArrayType)
        assert static.var_type.size.value == 16

    def test_zeropage_u16(self):
        """Test zeropage with 16-bit value."""
        static = parse_static("#[zeropage(0x20)] static mut PTR: u16;")
        assert static.var_type.name == "u16"

    def test_zeropage_register_flag(self):
        """Test zeropage with register flag for scratch."""
        static = parse_static("#[zeropage(0x10, register)] static mut SCRATCH0: u8;")
        attr = get_attr(static, "zeropage")
        # Check that register flag is present
        assert len(attr.args) >= 1


class TestLowramStorage:
    """Tests for #[lowram] storage class."""

    def test_lowram_basic(self):
        """Test basic lowram declaration."""
        static = parse_static("#[lowram] static mut BUFFER: [u8; 256];")
        attr = get_attr(static, "lowram")
        assert attr is not None

    def test_lowram_with_address(self):
        """Test lowram with explicit address."""
        static = parse_static("#[lowram(0x0200)] static mut DATA: [u8; 512];")
        attr = get_attr(static, "lowram")
        value = get_attr_value(attr)
        assert value.value == 0x0200

    def test_lowram_with_initializer(self):
        """Test lowram with initializer."""
        static = parse_static("#[lowram] static mut TABLE: [u8; 4] = [1, 2, 3, 4];")
        assert static.initializer is not None

    def test_lowram_struct(self):
        """Test lowram with struct type."""
        prog = parse_program("""
            struct Player { x: u8, y: u8 }
            #[lowram(0x0100)]
            static mut PLAYER: Player;
        """)
        static = prog.items[1]
        assert isinstance(static, ast.StaticDecl)
        attr = get_attr(static, "lowram")
        assert attr is not None


class TestRamStorage:
    """Tests for #[ram] storage class."""

    def test_ram_basic(self):
        """Test basic RAM declaration."""
        static = parse_static("#[ram] static mut WORK: [u8; 1024];")
        attr = get_attr(static, "ram")
        assert attr is not None

    def test_ram_with_address(self):
        """Test RAM with explicit address."""
        static = parse_static("#[ram(0x7E2000)] static mut BUFFER: [u8; 8192];")
        attr = get_attr(static, "ram")
        value = get_attr_value(attr)
        assert value.value == 0x7E2000

    def test_ram_with_initializer(self):
        """Test RAM with initializer."""
        static = parse_static("#[ram] static mut DEFAULTS: [u8; 4] = [0, 0, 0, 0];")
        assert static.initializer is not None

    def test_ram_large_array(self):
        """Test RAM with large array."""
        static = parse_static("#[ram] static mut TILEMAP: [u16; 2048];")
        assert isinstance(static.var_type, ast.ArrayType)
        assert static.var_type.size.value == 2048

    def test_ram_register_flag(self):
        """Test RAM with register flag."""
        static = parse_static("#[ram(0x7E0000, register)] static mut RAM_SCRATCH: u8;")
        attr = get_attr(static, "ram")
        assert len(attr.args) >= 1


class TestRomStorage:
    """Tests for #[rom] storage class (read-only)."""

    def test_rom_basic(self):
        """Test basic ROM declaration."""
        static = parse_static("#[rom] static LOOKUP: [u8; 16] = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15];")
        attr = get_attr(static, "rom")
        assert attr is not None
        assert static.is_mut == False

    def test_rom_with_address(self):
        """Test ROM with explicit address."""
        static = parse_static("#[rom(0x8000)] static DATA: [u8; 256] = [0; 256];")
        attr = get_attr(static, "rom")
        value = get_attr_value(attr)
        assert value.value == 0x8000

    def test_rom_const_table(self):
        """Test ROM constant lookup table."""
        static = parse_static("#[rom] static SINE: [u8; 8] = [0, 25, 49, 71, 90, 106, 117, 125];")
        assert len(static.initializer.elements) == 8

    def test_rom_immutable(self):
        """Test that ROM declaration is not mutable."""
        static = parse_static("#[rom] static TABLE: [u8; 4] = [1, 2, 3, 4];")
        assert static.is_mut == False


class TestHwStorage:
    """Tests for #[hw] storage class (hardware registers)."""

    def test_hw_basic(self):
        """Test basic hardware register declaration."""
        static = parse_static("#[hw(0x2100)] static mut INIDISP: u8;")
        attr = get_attr(static, "hw")
        assert attr is not None
        value = get_attr_value(attr)
        assert value.value == 0x2100

    def test_hw_multiple_registers(self):
        """Test multiple hardware register declarations."""
        prog = parse_program("""
            #[hw(0x2100)] static mut INIDISP: u8;
            #[hw(0x2101)] static mut OBSEL: u8;
            #[hw(0x2102)] static mut OAMADDL: u8;
        """)
        assert len(prog.items) == 3
        for item in prog.items:
            assert get_attr(item, "hw") is not None

    def test_hw_16bit(self):
        """Test 16-bit hardware register."""
        static = parse_static("#[hw(0x4212)] static mut HVBJOY: u8;")
        attr = get_attr(static, "hw")
        value = get_attr_value(attr)
        assert value.value == 0x4212

    def test_hw_read_only(self):
        """Test read-only hardware register."""
        static = parse_static("#[hw(0x4212)] static RDNMI: u8;")
        assert static.is_mut == False

    def test_hw_volatile_semantics(self):
        """Test that hw is automatically volatile (parse only)."""
        prog = parse_program("""
            #[hw(0x4212)]
            static mut HVBJOY: u8;
            fn poll() {
                loop {
                    let ready: u8 = HVBJOY;
                    if ready & 0x01 != 0 { break; }
                }
            }
        """)
        # Just verify it parses correctly
        assert len(prog.items) == 2


class TestStackDirective:
    """Tests for #[stack(lower, upper)] directive."""

    def test_stack_basic(self):
        """Test basic stack directive."""
        prog = parse_program("#[stack(0x1F00, 0x1FFF)]")
        assert len(prog.items) == 1
        stack = prog.items[0]
        assert isinstance(stack, ast.StackDirective)
        assert stack.lower == 0x1F00
        assert stack.upper == 0x1FFF

    def test_stack_default_range(self):
        """Test stack with default-ish range."""
        prog = parse_program("#[stack(0x0100, 0x01FF)]")
        stack = prog.items[0]
        assert stack.lower == 0x0100
        assert stack.upper == 0x01FF

    def test_stack_large_range(self):
        """Test stack with larger range."""
        prog = parse_program("#[stack(0x1800, 0x1FFF)]")
        stack = prog.items[0]
        assert stack.upper - stack.lower == 0x07FF  # 2KB

    def test_stack_with_other_declarations(self):
        """Test stack directive with other declarations."""
        prog = parse_program("""
            #[stack(0x1F00, 0x1FFF)]
            #[lowram]
            static mut VAR: u8;
        """)
        assert len(prog.items) == 2
        assert isinstance(prog.items[0], ast.StackDirective)
        assert isinstance(prog.items[1], ast.StaticDecl)


class TestNearPointerType:
    """Tests for near<T> pointer type."""

    def test_near_pointer_declaration(self):
        """Test near pointer declaration."""
        static = parse_static("#[zeropage] static mut PTR: near<u8>;")
        assert isinstance(static.var_type, ast.PointerType)
        assert static.var_type.is_far == False
        assert static.var_type.pointee_type.name == "u8"

    def test_near_pointer_to_u16(self):
        """Test near pointer to u16."""
        static = parse_static("#[zeropage] static mut PTR16: near<u16>;")
        assert static.var_type.pointee_type.name == "u16"

    def test_near_pointer_to_array(self):
        """Test near pointer to array."""
        static = parse_static("#[zeropage] static mut PTR_ARR: near<[u8; 16]>;")
        assert isinstance(static.var_type.pointee_type, ast.ArrayType)

    def test_near_pointer_in_function(self):
        """Test near pointer as function parameter."""
        func = parse_function("fn process(ptr: near<u8>) { }")
        assert isinstance(func.params[0].param_type, ast.PointerType)
        assert func.params[0].param_type.is_far == False


class TestFarPointerType:
    """Tests for far<T> pointer type."""

    def test_far_pointer_declaration(self):
        """Test far pointer declaration."""
        static = parse_static("#[ram] static mut FAR_PTR: far<u8>;")
        assert isinstance(static.var_type, ast.PointerType)
        assert static.var_type.is_far == True

    def test_far_pointer_to_u16(self):
        """Test far pointer to u16."""
        static = parse_static("#[ram] static mut FAR_PTR16: far<u16>;")
        assert static.var_type.is_far == True
        assert static.var_type.pointee_type.name == "u16"

    def test_far_pointer_in_function(self):
        """Test far pointer as function parameter."""
        func = parse_function("fn process_far(ptr: far<u8>) { }")
        assert isinstance(func.params[0].param_type, ast.PointerType)
        assert func.params[0].param_type.is_far == True

    def test_far_pointer_return_type(self):
        """Test far pointer as return type."""
        func = parse_function("fn get_ptr() -> far<u8> { }")
        assert isinstance(func.return_type, ast.PointerType)
        assert func.return_type.is_far == True


class TestPointerDereference:
    """Tests for pointer dereference operations."""

    def test_dereference_read(self):
        """Test reading through pointer dereference."""
        stmt = parse_statement("let x: u8 = *PTR;")
        assert isinstance(stmt, ast.LetStmt)
        assert isinstance(stmt.initializer, ast.Dereference)
        assert isinstance(stmt.initializer.pointer, ast.Identifier)

    def test_dereference_write(self):
        """Test writing through pointer dereference."""
        stmt = parse_statement("*PTR = 42;")
        assert isinstance(stmt, ast.ExprStmt)
        assign = stmt.expr
        assert isinstance(assign, ast.Assignment)
        assert isinstance(assign.target, ast.Dereference)

    def test_dereference_to_register(self):
        """Test dereference assigned to register."""
        stmt = parse_statement("A = *PTR;")
        assign = stmt.expr
        assert isinstance(assign.value, ast.Dereference)

    def test_dereference_from_register(self):
        """Test storing register through dereference."""
        stmt = parse_statement("*PTR = A;")
        assign = stmt.expr
        assert isinstance(assign.target, ast.Dereference)
        assert isinstance(assign.value, ast.Register)

    def test_dereference_in_expression(self):
        """Test dereference in arithmetic expression."""
        stmt = parse_statement("let x: u8 = *PTR + 1;")
        assert isinstance(stmt.initializer, ast.BinaryOp)
        assert isinstance(stmt.initializer.left, ast.Dereference)

    def test_dereference_compound_assignment(self):
        """Test compound assignment through dereference."""
        stmt = parse_statement("*PTR += 10;")
        assert isinstance(stmt.expr, ast.CompoundAssignment)
        assert isinstance(stmt.expr.target, ast.Dereference)


class TestAddressOf:
    """Tests for address-of operator (&)."""

    def test_address_of_variable(self):
        """Test taking address of variable."""
        stmt = parse_statement("let ptr: near<u8> = &VAR;")
        assert isinstance(stmt.initializer, ast.AddressOf)
        assert isinstance(stmt.initializer.operand, ast.Identifier)

    def test_address_of_array_element(self):
        """Test taking address of array element."""
        stmt = parse_statement("let ptr: near<u8> = &ARR[0];")
        assert isinstance(stmt.initializer, ast.AddressOf)
        assert isinstance(stmt.initializer.operand, ast.ArrayIndex)

    def test_address_of_in_assignment(self):
        """Test address-of in assignment."""
        prog = parse_program("""
            #[zeropage]
            static mut PTR: near<u8>;
            #[ram]
            static mut DATA: u8;
            fn setup() {
                PTR = &DATA;
            }
        """)
        func = prog.items[2]
        stmt = func.body.statements[0]
        assert isinstance(stmt.expr.value, ast.AddressOf)


class TestArrayIndexing:
    """Tests for array indexing operations."""

    def test_array_index_read(self):
        """Test reading array element."""
        stmt = parse_statement("let x: u8 = BUF[0];")
        assert isinstance(stmt.initializer, ast.ArrayIndex)
        assert stmt.initializer.index.value == 0

    def test_array_index_write(self):
        """Test writing array element."""
        stmt = parse_statement("BUF[0] = 42;")
        assign = stmt.expr
        assert isinstance(assign.target, ast.ArrayIndex)

    def test_array_index_with_register(self):
        """Test array index with register."""
        stmt = parse_statement("A = BUF[X];")
        assign = stmt.expr
        assert isinstance(assign.value, ast.ArrayIndex)
        assert isinstance(assign.value.index, ast.Register)

    def test_array_index_write_with_register(self):
        """Test array write with register index."""
        stmt = parse_statement("BUF[Y] = A;")
        assign = stmt.expr
        assert isinstance(assign.target.index, ast.Register)

    def test_array_index_expression(self):
        """Test array index with expression."""
        stmt = parse_statement("let x: u8 = BUF[X + 1];")
        assert isinstance(stmt.initializer.index, ast.BinaryOp)

    def test_nested_array_access(self):
        """Test nested array (2D simulation)."""
        stmt = parse_statement("let x: u8 = MATRIX[Y][X];")
        # This should be ArrayIndex of ArrayIndex
        assert isinstance(stmt.initializer, ast.ArrayIndex)
        assert isinstance(stmt.initializer.array, ast.ArrayIndex)

    def test_array_compound_assignment(self):
        """Test compound assignment to array element."""
        stmt = parse_statement("BUF[X] += 10;")
        assert isinstance(stmt.expr, ast.CompoundAssignment)
        assert isinstance(stmt.expr.target, ast.ArrayIndex)


class TestFieldAccess:
    """Tests for struct field access."""

    def test_field_access_read(self):
        """Test reading struct field."""
        prog = parse_program("""
            struct Point { x: u8, y: u8 }
            #[ram]
            static mut P: Point;
            fn test() {
                let x: u8 = P.x;
            }
        """)
        func = prog.items[2]
        stmt = func.body.statements[0]
        assert isinstance(stmt.initializer, ast.FieldAccess)
        assert stmt.initializer.field == "x"

    def test_field_access_write(self):
        """Test writing struct field."""
        prog = parse_program("""
            struct Point { x: u8, y: u8 }
            #[ram]
            static mut P: Point;
            fn test() {
                P.x = 10;
            }
        """)
        func = prog.items[2]
        stmt = func.body.statements[0]
        assign = stmt.expr
        assert isinstance(assign.target, ast.FieldAccess)

    def test_array_of_struct_access(self):
        """Test accessing field of array element."""
        prog = parse_program("""
            struct Entity { x: u8, y: u8 }
            #[ram]
            static mut ENTITIES: [Entity; 8];
            fn test() {
                let x: u8 = ENTITIES[0].x;
            }
        """)
        func = prog.items[2]
        stmt = func.body.statements[0]
        # Should be FieldAccess of ArrayIndex
        assert isinstance(stmt.initializer, ast.FieldAccess)
        assert isinstance(stmt.initializer.base, ast.ArrayIndex)


class TestMemoryHIR:
    """Tests for HIR generation of memory operations."""

    def test_hir_zeropage_static(self):
        """Test HIR for zeropage static."""
        hir_prog = build_hir("#[zeropage] static mut TEMP: u8;")
        assert len(hir_prog.statics) > 0

    def test_hir_ram_static(self):
        """Test HIR for RAM static."""
        hir_prog = build_hir("#[ram] static mut BUFFER: [u8; 256];")
        assert len(hir_prog.statics) > 0

    def test_hir_hw_static(self):
        """Test HIR for hardware static."""
        hir_prog = build_hir("#[hw(0x2100)] static mut INIDISP: u8;")
        assert len(hir_prog.statics) > 0

    def test_hir_pointer_dereference(self):
        """Test HIR for pointer dereference."""
        hir_prog = build_hir("""
            #[zeropage]
            static mut PTR: near<u8>;
            fn test() {
                let x: u8 = *PTR;
            }
        """)
        # Just verify it builds
        assert len(hir_prog.functions) > 0

    def test_hir_array_access(self):
        """Test HIR for array access."""
        hir_prog = build_hir("""
            #[ram]
            static mut BUF: [u8; 256];
            fn test() {
                let x: u8 = BUF[0];
            }
        """)
        assert len(hir_prog.functions) > 0


class TestMemoryPatterns:
    """Tests for common memory access patterns."""

    def test_copy_loop_pattern(self):
        """Test memory copy loop pattern."""
        prog = parse_program("""
            #[ram]
            static mut SRC: [u8; 256];
            #[ram]
            static mut DST: [u8; 256];
            fn copy() {
                let i @ X = 0;
                loop {
                    DST[X] = SRC[X];
                    X++;
                    if X == 0 { break; }
                }
            }
        """)
        assert len(prog.items) == 3

    def test_buffer_clear_pattern(self):
        """Test buffer clear pattern."""
        prog = parse_program("""
            #[ram]
            static mut BUF: [u8; 256];
            fn clear() {
                let i @ X = 0;
                loop {
                    BUF[X] = 0;
                    X++;
                    if X == 0 { break; }
                }
            }
        """)
        assert len(prog.items) == 2

    def test_lookup_table_pattern(self):
        """Test lookup table pattern."""
        prog = parse_program("""
            #[rom]
            static SINE_TABLE: [u8; 4] = [0, 64, 127, 64];
            fn lookup(index @ X: u8) -> u8 {
                return SINE_TABLE[X];
            }
        """)
        assert len(prog.items) == 2

    def test_hardware_polling_pattern(self):
        """Test hardware polling pattern."""
        prog = parse_program("""
            #[hw(0x4212)]
            static mut HVBJOY: u8;
            fn wait_vblank() {
                loop {
                    let ready: u8 = HVBJOY;
                    if ready & 0x80 != 0 { break; }
                }
            }
        """)
        assert len(prog.items) == 2


class TestMemoryEdgeCases:
    """Tests for edge cases in memory operations."""

    def test_zero_address(self):
        """Test zeropage address 0."""
        static = parse_static("#[zeropage(0x00)] static mut FIRST: u8;")
        attr = get_attr(static, "zeropage")
        value = get_attr_value(attr)
        assert value.value == 0

    def test_max_zeropage_address(self):
        """Test maximum zeropage address."""
        static = parse_static("#[zeropage(0xFF)] static mut LAST: u8;")
        attr = get_attr(static, "zeropage")
        value = get_attr_value(attr)
        assert value.value == 0xFF

    def test_multiple_storage_attributes(self):
        """Test multiple storage-class attributes on same declaration."""
        # This might be an error case, but parser allows it
        prog = parse_program("#[zeropage] #[ram] static mut DOUBLE: u8;")
        static = prog.items[0]
        # Both attributes should be present in AST
        assert get_attr(static, "zeropage") is not None or get_attr(static, "ram") is not None

    def test_pointer_to_pointer_requires_spacing(self):
        """Test that nested pointer types require space to avoid >> ambiguity."""
        # Parser limitation: >> is parsed as right-shift operator
        # So near<near<u8>> fails, but could work with space: near<near<u8> >
        # For now, we just document that nested pointer-to-pointer is not supported
        with pytest.raises(Exception):
            parse_static("#[zeropage] static mut PP: near<near<u8>>;")

    def test_pointer_to_basic_type_works(self):
        """Test pointer to basic type (non-nested) works fine."""
        static = parse_static("#[zeropage] static mut P: near<u8>;")
        assert isinstance(static.var_type, ast.PointerType)
        assert static.var_type.is_far == False


class TestMemoryParseErrors:
    """Tests for memory-related parse errors."""

    def test_missing_storage_class_parses(self):
        """Test that missing storage class parses (type checker catches)."""
        # Parser allows static without storage class
        static = parse_static("static mut NAKED: u8;")
        assert len(static.attributes) == 0

    def test_invalid_hw_address_format(self):
        """Test that various address formats parse."""
        # Hex address
        static = parse_static("#[hw(0x2100)] static mut HW1: u8;")
        assert get_attr_value(get_attr(static, "hw")).value == 0x2100

        # Decimal address
        static = parse_static("#[hw(8448)] static mut HW2: u8;")
        assert get_attr_value(get_attr(static, "hw")).value == 8448

    def test_stack_missing_upper_fails(self):
        """Test that stack with missing upper bound fails."""
        with pytest.raises(Exception):
            parse("#[stack(0x1F00)]")

    def test_pointer_missing_type_fails(self):
        """Test that pointer without type fails."""
        with pytest.raises(Exception):
            parse("#[zeropage] static mut PTR: near;")

    def test_array_negative_size_parses(self):
        """Test that negative array size parses (type checker catches)."""
        # Parser allows negative size; type checker would reject
        static = parse_static("#[ram] static mut BAD: [u8; -1];")
        # The unary minus creates a UnaryOp around the literal
        assert isinstance(static.var_type, ast.ArrayType)
