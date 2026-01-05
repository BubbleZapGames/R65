"""
Comprehensive function tests for R65.

Tests all function-related constructs:
- Basic function declarations
- Function parameters (stack, register-aliased, variable-bound)
- Return types and values (single, multiple, implicit)
- Far functions (cross-bank calls)
- Register aliasing
- Register preservation (#[preserves(...)])
- Mode annotations (#[mode(...)])
- Entry functions (#[entry])
- Interrupt handlers (#[interrupt(...)])
- Function calls
- Function pointers

Each test validates:
1. Parsing succeeds and produces correct AST
2. Function structure is correctly represented
3. HIR is built correctly (where applicable)
"""

import pytest
from r65.compiler.frontend import parse, ParseError, ast
from r65.compiler.hir import HIRBuilder
from r65.compiler.hir import nodes as hir


# ============================================================================
# Test Helpers
# ============================================================================

def parse_program(source: str) -> ast.Program:
    """Parse source and return the program."""
    return parse(source)


def parse_function(source: str) -> ast.FunctionDecl:
    """Parse source and return the first function declaration."""
    program = parse(source)
    for item in program.items:
        if isinstance(item, ast.FunctionDecl):
            return item
    raise ValueError("No function found in source")


def build_hir(source: str) -> hir.HIRProgram:
    """Parse and build HIR from source."""
    program = parse(source)
    builder = HIRBuilder()
    return builder.build_program(program)


def get_hir_function(hir_prog: hir.HIRProgram, name: str) -> hir.HIRFunctionDecl:
    """Get a function by name from HIR program."""
    for func in hir_prog.functions:
        if func.name == name:
            return func
    raise KeyError(f"Function '{name}' not found")


# ============================================================================
# Basic Function Declaration Tests
# ============================================================================

class TestBasicFunctionDeclarations:
    """Tests for basic function declarations."""

    def test_empty_function(self):
        """Test empty function with no params or return."""
        func = parse_function("fn empty() { }")

        assert func.name == 'empty'
        assert len(func.params) == 0
        assert func.return_type is None
        assert func.is_far == False
        assert len(func.body.statements) == 0

    def test_function_with_body(self):
        """Test function with statements in body."""
        func = parse_function("""
            fn process() {
                A = 10;
                X = 20;
            }
        """)

        assert func.name == 'process'
        assert len(func.body.statements) == 2

    def test_function_with_return_type(self):
        """Test function with return type."""
        func = parse_function("fn get_value() -> u8 { return 42; }")

        assert func.name == 'get_value'
        assert isinstance(func.return_type, ast.BasicType)
        assert func.return_type.name == 'u8'

    def test_function_returns_u16(self):
        """Test function returning u16."""
        func = parse_function("fn get_word() -> u16 { return 1000; }")

        assert func.return_type.name == 'u16'

    def test_function_returns_bool(self):
        """Test function returning bool."""
        func = parse_function("fn is_ready() -> bool { return true; }")

        assert func.return_type.name == 'bool'

    def test_function_single_statement(self):
        """Test function with single statement."""
        func = parse_function("fn inc() { A = A + 1; }")

        assert len(func.body.statements) == 1

    def test_function_multiple_statements(self):
        """Test function with multiple statements."""
        func = parse_function("""
            fn multi() {
                let x: u8 = 0;
                x = x + 1;
                A = x;
                return A;
            }
        """)

        assert len(func.body.statements) == 4

    def test_function_name_with_underscore(self):
        """Test function with underscore in name."""
        func = parse_function("fn my_function() { }")

        assert func.name == 'my_function'

    def test_function_name_starting_with_underscore(self):
        """Test function starting with underscore."""
        func = parse_function("fn _internal() { }")

        assert func.name == '_internal'

    def test_function_name_with_numbers(self):
        """Test function with numbers in name."""
        func = parse_function("fn handler1() { }")

        assert func.name == 'handler1'


# ============================================================================
# Function Parameter Tests
# ============================================================================

class TestFunctionParameters:
    """Tests for function parameters."""

    def test_single_parameter(self):
        """Test function with single parameter."""
        func = parse_function("fn process(x: u8) { }")

        assert len(func.params) == 1
        assert func.params[0].name == 'x'
        assert func.params[0].param_type.name == 'u8'
        assert func.params[0].binding is None

    def test_two_parameters(self):
        """Test function with two parameters."""
        func = parse_function("fn add(a: u8, b: u8) -> u8 { return A; }")

        assert len(func.params) == 2
        assert func.params[0].name == 'a'
        assert func.params[1].name == 'b'

    def test_many_parameters(self):
        """Test function with many parameters."""
        func = parse_function("fn multi(a: u8, b: u8, c: u16, d: bool) { }")

        assert len(func.params) == 4
        assert func.params[0].param_type.name == 'u8'
        assert func.params[2].param_type.name == 'u16'
        assert func.params[3].param_type.name == 'bool'

    def test_parameter_mixed_types(self):
        """Test parameters with mixed types."""
        func = parse_function("fn mixed(byte: u8, word: u16, flag: bool) { }")

        assert func.params[0].param_type.name == 'u8'
        assert func.params[1].param_type.name == 'u16'
        assert func.params[2].param_type.name == 'bool'

    def test_parameter_signed_types(self):
        """Test parameters with signed types."""
        func = parse_function("fn signed(a: i8, b: i16) { }")

        assert func.params[0].param_type.name == 'i8'
        assert func.params[1].param_type.name == 'i16'

    def test_parameter_pointer_type(self):
        """Test parameter with pointer type."""
        func = parse_function("fn read(ptr: near<u8>) { }")

        assert isinstance(func.params[0].param_type, ast.PointerType)
        assert func.params[0].param_type.is_far == False

    def test_parameter_far_pointer(self):
        """Test parameter with far pointer type."""
        func = parse_function("fn read_far(ptr: far<u8>) { }")

        assert func.params[0].param_type.is_far == True

    def test_parameter_array_type(self):
        """Test parameter with array type."""
        func = parse_function("fn process(data: [u8; 16]) { }")

        assert isinstance(func.params[0].param_type, ast.ArrayType)

    def test_trailing_comma_in_params(self):
        """Test trailing comma in parameter list."""
        func = parse_function("fn test(a: u8, b: u8,) { }")

        assert len(func.params) == 2


# ============================================================================
# Register Aliasing Tests
# ============================================================================

class TestRegisterAliasing:
    """Tests for register aliasing in parameters."""

    def test_param_alias_a(self):
        """Test parameter aliased to A register."""
        func = parse_function("fn process(value @ A: u8) { }")

        assert func.params[0].name == 'value'
        assert isinstance(func.params[0].binding, ast.Register)
        assert func.params[0].binding.name == 'A'

    def test_param_alias_x(self):
        """Test parameter aliased to X register."""
        func = parse_function("fn index(idx @ X: u8) { }")

        assert func.params[0].binding.name == 'X'

    def test_param_alias_y(self):
        """Test parameter aliased to Y register."""
        func = parse_function("fn offset(off @ Y: u8) { }")

        assert func.params[0].binding.name == 'Y'

    def test_multiple_register_aliases(self):
        """Test multiple register-aliased parameters."""
        func = parse_function("fn coords(x @ X: u8, y @ Y: u8) { }")

        assert func.params[0].binding.name == 'X'
        assert func.params[1].binding.name == 'Y'

    def test_mixed_aliased_and_stack(self):
        """Test mixed aliased and stack parameters."""
        func = parse_function("fn mixed(stack_param: u8, reg @ A: u8) { }")

        assert func.params[0].binding is None
        assert func.params[1].binding.name == 'A'

    def test_param_alias_with_return(self):
        """Test register-aliased parameter with return type."""
        func = parse_function("fn double(value @ A: u8) -> u8 { return A; }")

        assert func.params[0].binding.name == 'A'
        assert func.return_type.name == 'u8'

    def test_let_register_alias(self):
        """Test register aliasing in let statement."""
        func = parse_function("""
            fn test() {
                let count @ X = 10;
            }
        """)

        let_stmt = func.body.statements[0]
        assert isinstance(let_stmt, ast.LetStmt)
        assert let_stmt.name == 'count'
        assert isinstance(let_stmt.binding, ast.Register)
        assert let_stmt.binding.name == 'X'

    def test_let_mut_register_alias(self):
        """Test mutable register alias in let."""
        func = parse_function("""
            fn test() {
                let mut total @ A = 0;
                total = total + 1;
            }
        """)

        let_stmt = func.body.statements[0]
        assert let_stmt.is_mut == True
        assert let_stmt.binding.name == 'A'

    def test_variable_binding(self):
        """Test parameter bound to variable (not register)."""
        program = parse_program("""
            #[zeropage]
            static mut TEMP: u8;

            fn process(value @ TEMP: u8) { }
        """)

        func = program.items[1]
        # When bound to a variable, binding is a string identifier
        assert func.params[0].name == 'value'
        # The binding could be parsed as identifier or kept as string
        assert func.params[0].binding is not None


# ============================================================================
# Return Value Tests
# ============================================================================

class TestReturnValues:
    """Tests for function return values."""

    def test_return_literal(self):
        """Test return with literal value."""
        func = parse_function("fn get() -> u8 { return 42; }")

        return_stmt = func.body.statements[0]
        assert isinstance(return_stmt, ast.ReturnStmt)
        assert len(return_stmt.values) == 1
        assert return_stmt.values[0].value == 42

    def test_return_register(self):
        """Test return with register value."""
        func = parse_function("fn get() -> u8 { return A; }")

        return_stmt = func.body.statements[0]
        assert isinstance(return_stmt.values[0], ast.Register)

    def test_return_variable(self):
        """Test return with variable."""
        func = parse_function("""
            fn get() -> u8 {
                let x: u8 = 10;
                return x;
            }
        """)

        return_stmt = func.body.statements[1]
        assert isinstance(return_stmt.values[0], ast.Identifier)

    def test_return_expression(self):
        """Test return with expression."""
        func = parse_function("fn calc() -> u8 { return x + y; }")

        return_stmt = func.body.statements[0]
        assert isinstance(return_stmt.values[0], ast.BinaryOp)

    def test_return_multiple_values(self):
        """Test return with multiple values."""
        # Note: R65 doesn't have tuple return type syntax, just multi-value return
        func = parse_function("fn get_both() { return A, X; }")

        return_stmt = func.body.statements[0]
        assert len(return_stmt.values) == 2
        assert return_stmt.values[0].name == 'A'
        assert return_stmt.values[1].name == 'X'

    def test_return_three_values(self):
        """Test return with three values."""
        func = parse_function("fn get_all() { return A, X, Y; }")

        return_stmt = func.body.statements[0]
        assert len(return_stmt.values) == 3

    def test_return_no_value(self):
        """Test return without value."""
        func = parse_function("fn done() { return; }")

        return_stmt = func.body.statements[0]
        assert len(return_stmt.values) == 0

    def test_implicit_return(self):
        """Test function with implicit return (no return statement)."""
        func = parse_function("""
            fn set_a() {
                A = 42;
            }
        """)

        # No explicit return, A is implicitly returned
        assert len(func.body.statements) == 1
        assert not isinstance(func.body.statements[0], ast.ReturnStmt)

    def test_early_return(self):
        """Test early return pattern."""
        func = parse_function("""
            fn validate(x: u8) -> u8 {
                if x == 0 {
                    return 0;
                }
                return x;
            }
        """)

        assert len(func.body.statements) == 2


# ============================================================================
# Far Function Tests
# ============================================================================

class TestFarFunctions:
    """Tests for far (cross-bank) functions."""

    def test_far_function_declaration(self):
        """Test basic far function declaration."""
        func = parse_function("far fn cross_bank() { }")

        assert func.is_far == True
        assert func.name == 'cross_bank'

    def test_far_function_with_params(self):
        """Test far function with parameters."""
        func = parse_function("far fn process(value @ A: u8) { }")

        assert func.is_far == True
        assert len(func.params) == 1

    def test_far_function_with_return(self):
        """Test far function with return type."""
        func = parse_function("far fn get_data() -> u8 { return 0; }")

        assert func.is_far == True
        assert func.return_type.name == 'u8'

    def test_far_function_with_bank_attr(self):
        """Test far function with bank attribute."""
        func = parse_function("""
            #[bank(1)]
            far fn in_bank1() { }
        """)

        assert func.is_far == True
        assert len(func.attributes) == 1
        assert func.attributes[0].name == 'bank'

    def test_far_function_bank_with_data_bank(self):
        """Test far function with data_bank option."""
        func = parse_function("""
            #[bank(1, data_bank=inline)]
            far fn with_dbr() { }
        """)

        attr = func.attributes[0]
        assert attr.name == 'bank'
        assert len(attr.args) == 2

    def test_near_function_default(self):
        """Test that functions are near by default."""
        func = parse_function("fn local() { }")

        assert func.is_far == False

    def test_far_with_complex_body(self):
        """Test far function with complex body."""
        func = parse_function("""
            far fn complex(x @ A: u8) -> u8 {
                if x > 10 {
                    return 10;
                }
                return x;
            }
        """)

        assert func.is_far == True
        assert len(func.body.statements) == 2


# ============================================================================
# Function Attribute Tests
# ============================================================================

class TestFunctionAttributes:
    """Tests for function attributes."""

    def test_mode_attribute_m8_x8(self):
        """Test mode attribute with m8, x8."""
        func = parse_function("""
            #[mode(m8, x8)]
            fn eight_bit() { }
        """)

        assert len(func.attributes) == 1
        assert func.attributes[0].name == 'mode'

    def test_mode_attribute_m16_x16(self):
        """Test mode attribute with m16, x16."""
        func = parse_function("""
            #[mode(m16, x16)]
            fn sixteen_bit() { }
        """)

        attr = func.attributes[0]
        assert attr.name == 'mode'

    def test_mode_with_transition(self):
        """Test mode attribute with transition option."""
        func = parse_function("""
            #[mode(m8, x8, transition=inline)]
            fn with_transition() { }
        """)

        attr = func.attributes[0]
        assert attr.name == 'mode'

    def test_preserves_attribute(self):
        """Test preserves attribute."""
        func = parse_function("""
            #[preserves(X, Y)]
            fn save_xy() { }
        """)

        assert len(func.attributes) == 1
        assert func.attributes[0].name == 'preserves'

    def test_preserves_single_register(self):
        """Test preserves with single register."""
        func = parse_function("""
            #[preserves(A)]
            fn save_a() { }
        """)

        attr = func.attributes[0]
        assert attr.name == 'preserves'

    def test_preserves_many_registers(self):
        """Test preserves with many registers."""
        func = parse_function("""
            #[preserves(A, X, Y, STATUS)]
            fn save_all() { }
        """)

        attr = func.attributes[0]
        assert attr.name == 'preserves'

    def test_entry_attribute(self):
        """Test entry attribute."""
        func = parse_function("""
            #[entry]
            fn main() -> ! {
                loop { }
            }
        """)

        assert len(func.attributes) == 1
        assert func.attributes[0].name == 'entry'

    def test_multiple_attributes(self):
        """Test function with multiple attributes."""
        func = parse_function("""
            #[mode(m8, x8)]
            #[preserves(X, Y)]
            fn complex() { }
        """)

        assert len(func.attributes) == 2
        assert func.attributes[0].name == 'mode'
        assert func.attributes[1].name == 'preserves'

    def test_bank_attribute(self):
        """Test bank attribute."""
        func = parse_function("""
            #[bank(2)]
            far fn in_bank2() { }
        """)

        assert func.attributes[0].name == 'bank'

    def test_zeropage_register_attribute(self):
        """Test zeropage with register flag."""
        program = parse_program("""
            #[zeropage(0x10, register)]
            static mut SCRATCH: u8;
        """)

        static = program.items[0]
        assert static.attributes[0].name == 'zeropage'


# ============================================================================
# Interrupt Handler Tests
# ============================================================================

class TestInterruptHandlers:
    """Tests for interrupt handlers."""

    def test_nmi_interrupt(self):
        """Test NMI interrupt handler."""
        func = parse_function("""
            #[interrupt(nmi)]
            fn vblank() { }
        """)

        assert len(func.attributes) == 1
        attr = func.attributes[0]
        assert attr.name == 'interrupt'

    def test_irq_interrupt(self):
        """Test IRQ interrupt handler."""
        func = parse_function("""
            #[interrupt(irq)]
            fn timer_handler() { }
        """)

        attr = func.attributes[0]
        assert attr.name == 'interrupt'

    def test_brk_interrupt(self):
        """Test BRK interrupt handler."""
        func = parse_function("""
            #[interrupt(brk)]
            fn break_handler() { }
        """)

        attr = func.attributes[0]
        assert attr.name == 'interrupt'

    def test_cop_interrupt(self):
        """Test COP interrupt handler."""
        func = parse_function("""
            #[interrupt(cop)]
            fn cop_handler() { }
        """)

        attr = func.attributes[0]
        assert attr.name == 'interrupt'

    def test_interrupt_with_body(self):
        """Test interrupt handler with body."""
        func = parse_function("""
            #[interrupt(nmi)]
            fn vblank() {
                A = 1;
                X = 2;
            }
        """)

        assert len(func.body.statements) == 2

    def test_interrupt_preserve_false(self):
        """Test interrupt with preserve=false."""
        func = parse_function("""
            #[interrupt(irq, preserve=false)]
            fn fast_irq() { }
        """)

        attr = func.attributes[0]
        assert attr.name == 'interrupt'

    def test_interrupt_with_preserves(self):
        """Test interrupt with manual preserves."""
        func = parse_function("""
            #[interrupt(nmi)]
            #[preserves(A)]
            fn custom_nmi() { }
        """)

        assert len(func.attributes) == 2


# ============================================================================
# Function Call Tests
# ============================================================================

class TestFunctionCalls:
    """Tests for function calls."""

    def test_call_no_args(self):
        """Test function call with no arguments."""
        func = parse_function("""
            fn test() {
                process();
            }
        """)

        expr_stmt = func.body.statements[0]
        call = expr_stmt.expr
        assert isinstance(call, ast.FunctionCall)
        assert len(call.args) == 0

    def test_call_single_arg(self):
        """Test function call with single argument."""
        func = parse_function("""
            fn test() {
                process(42);
            }
        """)

        call = func.body.statements[0].expr
        assert len(call.args) == 1
        assert call.args[0].value == 42

    def test_call_multiple_args(self):
        """Test function call with multiple arguments."""
        func = parse_function("""
            fn test() {
                add(10, 20);
            }
        """)

        call = func.body.statements[0].expr
        assert len(call.args) == 2

    def test_call_with_variables(self):
        """Test function call with variable arguments."""
        func = parse_function("""
            fn test() {
                process(x, y);
            }
        """)

        call = func.body.statements[0].expr
        assert isinstance(call.args[0], ast.Identifier)
        assert isinstance(call.args[1], ast.Identifier)

    def test_call_with_registers(self):
        """Test function call with register arguments."""
        func = parse_function("""
            fn test() {
                process(A, X);
            }
        """)

        call = func.body.statements[0].expr
        assert isinstance(call.args[0], ast.Register)
        assert isinstance(call.args[1], ast.Register)

    def test_call_with_expressions(self):
        """Test function call with expression arguments."""
        func = parse_function("""
            fn test() {
                process(x + 1, y * 2);
            }
        """)

        call = func.body.statements[0].expr
        assert isinstance(call.args[0], ast.BinaryOp)
        assert isinstance(call.args[1], ast.BinaryOp)

    def test_call_result_assignment(self):
        """Test assigning function call result."""
        func = parse_function("""
            fn test() {
                let result = get_value();
            }
        """)

        let_stmt = func.body.statements[0]
        assert isinstance(let_stmt.initializer, ast.FunctionCall)

    def test_nested_calls(self):
        """Test nested function calls."""
        func = parse_function("""
            fn test() {
                process(get_value());
            }
        """)

        call = func.body.statements[0].expr
        assert isinstance(call.args[0], ast.FunctionCall)

    def test_call_trailing_comma(self):
        """Test function call with trailing comma."""
        func = parse_function("""
            fn test() {
                process(a, b, c,);
            }
        """)

        call = func.body.statements[0].expr
        assert len(call.args) == 3

    def test_call_in_expression(self):
        """Test function call in expression."""
        func = parse_function("""
            fn test() {
                let x = get_a() + get_b();
            }
        """)

        let_stmt = func.body.statements[0]
        assert isinstance(let_stmt.initializer, ast.BinaryOp)
        assert isinstance(let_stmt.initializer.left, ast.FunctionCall)
        assert isinstance(let_stmt.initializer.right, ast.FunctionCall)


# ============================================================================
# Function Pointer Tests
# ============================================================================

class TestFunctionPointers:
    """Tests for function pointers."""

    def test_fn_pointer_type(self):
        """Test function pointer type declaration."""
        program = parse_program("static CALLBACK: fn();")

        static = program.items[0]
        assert isinstance(static.var_type, ast.FunctionType)
        assert static.var_type.is_far == False

    def test_fn_pointer_with_params(self):
        """Test function pointer with parameters."""
        program = parse_program("static HANDLER: fn(u8, u8);")

        static = program.items[0]
        assert len(static.var_type.param_types) == 2

    def test_fn_pointer_with_return(self):
        """Test function pointer with return type."""
        program = parse_program("static GETTER: fn() -> u8;")

        static = program.items[0]
        assert static.var_type.return_type.name == 'u8'

    def test_far_fn_pointer(self):
        """Test far function pointer."""
        program = parse_program("static FAR_CB: far fn();")

        static = program.items[0]
        assert static.var_type.is_far == True

    def test_fn_pointer_type_alias(self):
        """Test function pointer type alias."""
        program = parse_program("type Callback = fn(u8) -> u8;")

        alias = program.items[0]
        assert isinstance(alias.aliased_type, ast.FunctionType)

    def test_fn_pointer_in_param(self):
        """Test function pointer as parameter."""
        func = parse_function("fn call_it(callback: fn()) { }")

        assert isinstance(func.params[0].param_type, ast.FunctionType)

    def test_indirect_call(self):
        """Test indirect function call through pointer."""
        program = parse_program("""
            static mut HANDLER: fn(u8);

            fn test() {
                HANDLER(42);
            }
        """)

        func = program.items[1]
        call = func.body.statements[0].expr
        assert isinstance(call, ast.FunctionCall)


# ============================================================================
# HIR Building Tests
# ============================================================================

class TestFunctionHIR:
    """Tests for function HIR building."""

    def test_basic_function_hir(self):
        """Test basic function HIR building."""
        hir_prog = build_hir("""
            fn test() { }
        """)

        func = get_hir_function(hir_prog, 'test')
        assert func.name == 'test'
        assert func.is_far == False

    def test_function_params_hir(self):
        """Test function parameters in HIR."""
        hir_prog = build_hir("""
            fn add(a: u8, b: u8) -> u8 {
                return A;
            }
        """)

        func = get_hir_function(hir_prog, 'add')
        assert len(func.parameters) == 2

    def test_far_function_hir(self):
        """Test far function HIR building."""
        hir_prog = build_hir("""
            far fn cross_bank() { }
        """)

        func = get_hir_function(hir_prog, 'cross_bank')
        assert func.is_far == True

    def test_entry_function_hir(self):
        """Test entry function HIR building."""
        hir_prog = build_hir("""
            #[entry]
            fn main() -> ! {
                loop { }
            }
        """)

        func = get_hir_function(hir_prog, 'main')
        assert func.is_entry == True

    def test_function_with_mode_hir(self):
        """Test function with mode attribute in HIR."""
        hir_prog = build_hir("""
            #[mode(m8, x8)]
            fn eight_bit() { }
        """)

        func = get_hir_function(hir_prog, 'eight_bit')
        assert func.mode_attr is not None

    def test_function_with_preserves_hir(self):
        """Test function with preserves attribute in HIR."""
        hir_prog = build_hir("""
            #[preserves(X, Y)]
            fn preserve_xy() { }
        """)

        func = get_hir_function(hir_prog, 'preserve_xy')
        assert func.preserves_attr is not None


# ============================================================================
# Complex Function Tests
# ============================================================================

class TestComplexFunctions:
    """Tests for complex function patterns."""

    def test_game_loop_pattern(self):
        """Test game loop function pattern."""
        func = parse_function("""
            #[entry]
            fn main() -> ! {
                init();
                loop {
                    wait_vblank();
                    update();
                    render();
                }
            }
        """)

        assert len(func.attributes) == 1
        assert isinstance(func.return_type, ast.NeverType)

    def test_interrupt_handler_pattern(self):
        """Test interrupt handler pattern."""
        func = parse_function("""
            #[interrupt(nmi)]
            fn vblank() {
                frame_count = frame_count + 1;
                update_sprites();
            }
        """)

        assert func.attributes[0].name == 'interrupt'

    def test_register_efficient_function(self):
        """Test register-efficient function pattern."""
        func = parse_function("""
            #[mode(m8, x8)]
            #[preserves(Y)]
            fn fast_copy(src @ X: u8, dst @ Y: u8, count @ A: u8) {
            }
        """)

        assert len(func.attributes) == 2
        assert len(func.params) == 3

    def test_state_machine_function(self):
        """Test state machine update function."""
        program = parse_program("""
            enum State { Idle, Running, Paused }

            #[zeropage]
            static mut CURRENT: State;

            fn update() {
                if CURRENT == State::Idle {
                    handle_idle();
                } else if CURRENT == State::Running {
                    handle_running();
                } else {
                    handle_paused();
                }
            }
        """)

        func = program.items[2]
        assert func.name == 'update'


# ============================================================================
# Edge Cases
# ============================================================================

class TestFunctionEdgeCases:
    """Tests for function edge cases."""

    def test_function_no_space_before_paren(self):
        """Test function with no space before paren."""
        func = parse_function("fn test(){ }")
        assert func.name == 'test'

    def test_function_extra_whitespace(self):
        """Test function with extra whitespace."""
        func = parse_function("fn   test  (  )  {  }")
        assert func.name == 'test'

    def test_empty_function_one_line(self):
        """Test empty function on one line."""
        func = parse_function("fn empty() { }")
        assert len(func.body.statements) == 0

    def test_deeply_nested_body(self):
        """Test function with deeply nested body."""
        func = parse_function("""
            fn deep() {
                if a {
                    if b {
                        loop {
                            if c {
                                break;
                            }
                        }
                    }
                }
            }
        """)

        assert len(func.body.statements) == 1

    def test_many_parameters(self):
        """Test function with many parameters."""
        func = parse_function(
            "fn many(a: u8, b: u8, c: u8, d: u8, e: u8, f: u8, g: u8, h: u8) { }"
        )

        assert len(func.params) == 8


# ============================================================================
# Parse Error Tests
# ============================================================================

class TestFunctionParseErrors:
    """Tests for function parse errors."""

    def test_missing_fn_keyword(self):
        """Test missing fn keyword fails."""
        with pytest.raises(Exception):
            parse("test() { }")

    def test_missing_function_name(self):
        """Test missing function name fails."""
        with pytest.raises(Exception):
            parse("fn () { }")

    def test_missing_parens(self):
        """Test missing parentheses fails."""
        with pytest.raises(Exception):
            parse("fn test { }")

    def test_missing_body(self):
        """Test missing function body fails."""
        with pytest.raises(Exception):
            parse("fn test()")

    def test_missing_param_type(self):
        """Test missing parameter type fails."""
        with pytest.raises(Exception):
            parse("fn test(x) { }")

    def test_missing_colon_in_param(self):
        """Test missing colon in parameter fails."""
        with pytest.raises(Exception):
            parse("fn test(x u8) { }")

    def test_invalid_return_arrow(self):
        """Test invalid return arrow fails."""
        with pytest.raises(Exception):
            parse("fn test() - u8 { }")

    def test_missing_return_type_after_arrow(self):
        """Test missing return type after arrow fails."""
        with pytest.raises(Exception):
            parse("fn test() -> { }")


# ============================================================================
# Run tests directly
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
