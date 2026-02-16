"""
Common test utilities for R65 language tests.

Provides shared helper functions for parsing and HIR building.
"""

import pytest
from r65.compiler.frontend.parser import parse, ParseError
from r65.compiler.frontend import ast
from r65.compiler.hir.builder import HIRBuilder
from r65.compiler.hir import nodes as hir


# =============================================================================
# Parsing Helpers
# =============================================================================

def parse_program(source: str) -> ast.Program:
    """Parse source code and return the AST program."""
    return parse(source)


def parse_succeeds(source: str) -> ast.Program:
    """Parse source and assert it succeeds."""
    return parse(source)


def parse_fails(source: str) -> str:
    """Parse source, expecting it to fail. Returns error message."""
    with pytest.raises(Exception) as exc_info:
        parse(source)
    return str(exc_info.value)


def parse_function(source: str) -> ast.FunctionDecl:
    """Parse a function definition and return the FunctionDecl node."""
    prog = parse(source)
    for item in prog.items:
        if isinstance(item, ast.FunctionDecl):
            return item
    raise ValueError("No function found in source")


def parse_static(source: str) -> ast.StaticDecl:
    """Parse a static declaration and return the StaticDecl node."""
    prog = parse(source)
    for item in prog.items:
        if isinstance(item, ast.StaticDecl):
            return item
    raise ValueError("No static declaration found in source")


def parse_struct(source: str) -> ast.StructDecl:
    """Parse a struct definition and return the StructDecl node."""
    prog = parse(source)
    for item in prog.items:
        if isinstance(item, ast.StructDecl):
            return item
    raise ValueError("No struct found in source")


def parse_enum(source: str) -> ast.EnumDecl:
    """Parse an enum definition and return the EnumDecl node."""
    prog = parse(source)
    for item in prog.items:
        if isinstance(item, ast.EnumDecl):
            return item
    raise ValueError("No enum found in source")


def parse_const(source: str) -> ast.ConstDecl:
    """Parse a const definition and return the ConstDecl node."""
    prog = parse(source)
    for item in prog.items:
        if isinstance(item, ast.ConstDecl):
            return item
    raise ValueError("No const found in source")


def parse_type_alias(source: str) -> ast.TypeAlias:
    """Parse a type alias and return the TypeAlias node."""
    prog = parse(source)
    for item in prog.items:
        if isinstance(item, ast.TypeAlias):
            return item
    raise ValueError("No type alias found in source")


def parse_expr(source: str) -> ast.Expression:
    """Parse an expression within a let statement.

    Uses type annotation to disambiguate from register alias syntax.
    """
    func = parse_function(f"fn test() {{ let x: u8 = {source}; }}")
    let_stmt = func.body.statements[0]
    if isinstance(let_stmt, ast.LetStmt):
        return let_stmt.initializer
    raise ValueError("Failed to extract expression")


def parse_stmt(source: str) -> ast.Statement:
    """Parse a statement within a function body."""
    func = parse_function(f"fn test() {{ {source} }}")
    if func.body.statements:
        return func.body.statements[0]
    raise ValueError("No statement found")


def parse_type(source: str) -> ast.Type:
    """Parse a type annotation from a static declaration."""
    static = parse_static(f"#[ram] static mut X: {source};")
    return static.var_type


# =============================================================================
# HIR Building Helpers
# =============================================================================

def build_hir(source: str) -> hir.HIRProgram:
    """Parse source and build HIR."""
    program = parse(source)
    builder = HIRBuilder()
    return builder.build_program(program)


def build_hir_with_warnings(source: str) -> tuple[hir.HIRProgram, list[str]]:
    """Parse source and build HIR, returning (program, warnings)."""
    program = parse(source)
    builder = HIRBuilder()
    hir_program = builder.build_program(program)
    return hir_program, builder.warnings


def build_hir_function(source: str) -> hir.HIRFunctionDecl:
    """Parse and build HIR for a function, returning the HIR function."""
    hir_prog = build_hir(source)
    if hir_prog.functions:
        return hir_prog.functions[0]
    raise ValueError("No function in HIR")


# =============================================================================
# Attribute Helpers
# =============================================================================

def get_attr(node, name: str) -> ast.Attribute | None:
    """Get an attribute by name from a node with attributes."""
    if hasattr(node, 'attributes'):
        for attr in node.attributes:
            if attr.name == name:
                return attr
    return None


def get_attr_arg(attr: ast.Attribute, index: int = 0):
    """Get an attribute argument by index."""
    if attr.args and len(attr.args) > index:
        return attr.args[index].value
    return None


def get_attr_arg_by_name(attr: ast.Attribute, name: str):
    """Get an attribute argument by name."""
    for arg in attr.args:
        if arg.name == name:
            return arg.value
    return None


def get_attr_arg_names(attr: ast.Attribute) -> list:
    """Get all argument names/values from an attribute."""
    results = []
    for arg in attr.args:
        if isinstance(arg.value, ast.Register):
            results.append(arg.value.name)
        elif isinstance(arg.value, ast.Identifier):
            results.append(arg.value.name)
        elif hasattr(arg.value, 'value'):
            results.append(arg.value.value)
    return results


# =============================================================================
# Statement Extraction Helpers
# =============================================================================

def get_first_stmt(func: ast.FunctionDecl) -> ast.Statement:
    """Get the first statement from a function body."""
    return func.body.statements[0]


def get_stmt(func: ast.FunctionDecl, index: int) -> ast.Statement:
    """Get a statement by index from a function body."""
    return func.body.statements[index]


def get_if_stmt(func: ast.FunctionDecl) -> ast.IfStmt:
    """Get the first if statement from a function."""
    for stmt in func.body.statements:
        if isinstance(stmt, ast.IfStmt):
            return stmt
    raise ValueError("No if statement found")


def get_loop_stmt(func: ast.FunctionDecl) -> ast.LoopStmt:
    """Get the first loop statement from a function."""
    for stmt in func.body.statements:
        if isinstance(stmt, ast.LoopStmt):
            return stmt
    raise ValueError("No loop statement found")


def get_while_stmt(func: ast.FunctionDecl) -> ast.WhileStmt:
    """Get the first while statement from a function."""
    for stmt in func.body.statements:
        if isinstance(stmt, ast.WhileStmt):
            return stmt
    raise ValueError("No while statement found")
