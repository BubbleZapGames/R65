"""
Parser for R65 using Lark.

Transforms Lark parse trees into our custom AST.
"""
from pathlib import Path
from lark import Lark, Transformer, Token as LarkToken, Tree
from r65.compiler.frontend import ast
from typing import List, Union, Optional


# Load the grammar
GRAMMAR_PATH = Path(__file__).parent / "grammar.lark"
with open(GRAMMAR_PATH) as f:
    GRAMMAR = f.read()


class ASTBuilder(Transformer):
    """
    Lark Transformer that builds our AST from the parse tree.

    Each method corresponds to a grammar rule and transforms
    the matched nodes into AST nodes.
    """

    def _parse_integer(self, value: str) -> int:
        """Parse an integer literal."""
        value = value.replace('_', '')
        if value.startswith('0x') or value.startswith('0X'):
            return int(value, 16)
        elif value.startswith('0b') or value.startswith('0B'):
            return int(value, 2)
        else:
            return int(value, 10)

    def _filter_tokens(self, items, keep_types=None):
        """
        Filter out punctuation tokens, keeping only semantic content.

        Args:
            items: List of items from Lark
            keep_types: Optional set of token types to keep

        Returns:
            List with only AST nodes and relevant tokens
        """
        if keep_types is None:
            keep_types = {'IDENT', 'MUT', 'FAR', 'INTEGER', 'STRING', 'BOOLEAN', 'REGISTER'}

        result = []
        for item in items:
            if isinstance(item, LarkToken):
                if item.type in keep_types:
                    result.append(item)
            else:
                # Keep all AST nodes
                result.append(item)
        return result

    def _validate_identifier_not_register(self, identifier: str, token: LarkToken):
        """
        Validate that an identifier is not a wrong-case multi-character register name.

        Only validates multi-character register names to avoid false positives with
        common variable names like 'x', 'y', 'd', 's', etc.

        Multi-character registers that must be exact case (all uppercase):
        - DBR, PBR, STATUS

        Args:
            identifier: The identifier to validate
            token: The Lark token for error reporting

        Raises:
            ParseError: If identifier is a wrong-case register name
        """
        # Only validate multi-character register names
        # Single-letter identifiers (a, x, y, d, s) are common variable names
        multi_char_registers = {'DBR', 'PBR', 'STATUS'}

        # Check case-insensitive match
        for register in multi_char_registers:
            if identifier.lower() == register.lower() and identifier != register:
                raise ParseError(
                    f"Invalid register name '{identifier}' at line {token.line}, column {token.column}. "
                    f"Did you mean '{register}'? Register names are case-sensitive and must be uppercase."
                )

    def _collect_attributes(self, items: list, start_idx: int):
        """
        Collect attributes from items list starting at index.

        Args:
            items: Filtered items list
            start_idx: Starting index

        Returns:
            Tuple of (attributes_list, next_index)
        """
        attrs = []
        idx = start_idx
        while idx < len(items) and isinstance(items[idx], ast.Attribute):
            attrs.append(items[idx])
            idx += 1
        return attrs, idx

    # ========================================================================
    # Program
    # ========================================================================

    def start(self, items):
        """Start rule - returns a Program node."""
        return ast.Program(items=list(items))

    # ========================================================================
    # Declarations
    # ========================================================================

    def function_decl(self, items):
        """Function declaration."""
        items = self._filter_tokens(items)

        # Collect attributes
        attrs, idx = self._collect_attributes(items, 0)

        # Check for far
        is_far = False
        if idx < len(items) and isinstance(items[idx], LarkToken) and items[idx].type == 'FAR':
            is_far = True
            idx += 1

        name = items[idx]
        idx += 1

        # Params (optional)
        params = []
        if idx < len(items) and isinstance(items[idx], list):
            params = items[idx]
            idx += 1

        # Return type (optional)
        return_type = None
        if idx < len(items) and not isinstance(items[idx], ast.Block):
            return_type = items[idx]
            idx += 1

        # Body
        body = items[idx]

        return ast.FunctionDecl(
            attributes=attrs,
            is_far=is_far,
            name=name.value if isinstance(name, LarkToken) else name,
            params=params,
            return_type=return_type,
            body=body
        )

    def param_list(self, items):
        """Parameter list."""
        # Filter out commas, keep only Parameter nodes
        return [item for item in items if isinstance(item, ast.Parameter)]

    def param(self, items):
        """Function parameter."""
        items = self._filter_tokens(items)

        name = items[0].value if isinstance(items[0], LarkToken) else items[0]
        binding = None
        param_type = items[-1]  # Type is always last

        # Check for binding (@ register or variable)
        # If binding exists, items = [IDENT, binding_node, type]
        # If no binding, items = [IDENT, type]
        if len(items) > 2:
            binding_node = items[1]
            if isinstance(binding_node, ast.Register):
                binding = binding_node
            elif isinstance(binding_node, ast.Identifier):
                binding = binding_node
            else:
                binding = binding_node.value if isinstance(binding_node, LarkToken) else binding_node

        return ast.Parameter(name=name, binding=binding, param_type=param_type)

    def binding(self, items):
        """Binding for @ operator."""
        item = items[0]
        if item.type == 'REGISTER':
            return ast.Register(name=item.value)
        else:
            return ast.Identifier(name=item.value)

    def return_type(self, items):
        """Return type."""
        items = self._filter_tokens(items, keep_types={'EXCLAMATION'})
        if items:
            item = items[0]
            if isinstance(item, LarkToken) and item.value == '!':
                return ast.NeverType()
            return item
        return None

    def static_decl(self, items):
        """Static variable declaration."""
        items = self._filter_tokens(items)

        # Collect attributes
        attrs, idx = self._collect_attributes(items, 0)

        # Check for mut token
        is_mut = False
        if idx < len(items) and isinstance(items[idx], LarkToken) and items[idx].type == 'MUT':
            is_mut = True
            idx += 1

        # Name
        name = items[idx].value if isinstance(items[idx], LarkToken) else items[idx]
        idx += 1

        # Type
        var_type = items[idx]
        idx += 1

        # Initializer (optional)
        initializer = items[idx] if idx < len(items) else None

        return ast.StaticDecl(
            attributes=attrs,
            is_mut=is_mut,
            name=name,
            var_type=var_type,
            initializer=initializer
        )

    def const_decl(self, items):
        """Const declaration."""
        items = self._filter_tokens(items)
        name = items[0].value if isinstance(items[0], LarkToken) else items[0]
        const_type = items[1]
        value = items[2]
        return ast.ConstDecl(name=name, const_type=const_type, value=value)

    def struct_decl(self, items):
        """Struct declaration."""
        items = self._filter_tokens(items)
        name = items[0].value if isinstance(items[0], LarkToken) else items[0]
        # Filter to keep only StructField nodes
        fields = [item for item in items[1:] if isinstance(item, ast.StructField)]
        return ast.StructDecl(name=name, fields=fields)

    def struct_field(self, items):
        """Struct field."""
        items = self._filter_tokens(items)
        name = items[0].value if isinstance(items[0], LarkToken) else items[0]
        field_type = items[1]
        return ast.StructField(name=name, field_type=field_type)

    def enum_decl(self, items):
        """Enum declaration."""
        items = self._filter_tokens(items)
        name = items[0].value if isinstance(items[0], LarkToken) else items[0]
        # Filter to keep only EnumVariant nodes
        variants = [item for item in items[1:] if isinstance(item, ast.EnumVariant)]
        return ast.EnumDecl(name=name, variants=variants)

    def enum_variant(self, items):
        """Enum variant."""
        items = self._filter_tokens(items)
        name = items[0].value if isinstance(items[0], LarkToken) else items[0]
        value = items[1] if len(items) > 1 else None
        return ast.EnumVariant(name=name, value=value)

    def type_alias(self, items):
        """Type alias."""
        items = self._filter_tokens(items)
        name = items[0].value if isinstance(items[0], LarkToken) else items[0]
        aliased_type = items[1]
        return ast.TypeAlias(name=name, aliased_type=aliased_type)

    def include_stmt(self, items):
        """Include statement."""
        items = self._filter_tokens(items, keep_types={'STRING'})
        path = items[0].value.strip('"')  # Remove quotes
        return ast.IncludeStmt(path=path)

    # ========================================================================
    # Attributes
    # ========================================================================

    def attribute(self, items):
        """Attribute."""
        items = self._filter_tokens(items)
        return items[0]  # Just return the attribute_inner result

    def attribute_inner(self, items):
        """Attribute inner."""
        items = self._filter_tokens(items, keep_types={'IDENT'})
        name = items[0].value if isinstance(items[0], LarkToken) else items[0]
        # args will be a list from attribute_args if present
        args = items[1] if len(items) > 1 and isinstance(items[1], list) else []
        return ast.Attribute(name=name, args=args)

    def attribute_args(self, items):
        """Attribute arguments."""
        # Filter out comma tokens, keep only AttributeArg objects
        result = []
        for item in items:
            if not isinstance(item, LarkToken):
                result.append(item)
        return result

    def attribute_arg(self, items):
        """
        Attribute argument - can be named or positional.

        Grammar: (IDENT "=")? expr

        Returns AttributeArg with optional name.
        """
        items = self._filter_tokens(items, keep_types={'IDENT'})

        # Check if we have a name (IDENT token followed by expression)
        if len(items) == 2 and isinstance(items[0], LarkToken):
            # Named argument: name=value
            name = items[0].value
            value = items[1]
            return ast.AttributeArg(name=name, value=value)
        else:
            # Positional argument: just value
            value = items[0]
            return ast.AttributeArg(name=None, value=value)

    # ========================================================================
    # Statements
    # ========================================================================

    def block(self, items):
        """Block statement."""
        # Filter out brace tokens, keep only statement nodes
        statements = [item for item in items if not isinstance(item, LarkToken)]
        return ast.Block(statements=statements)

    def let_stmt(self, items):
        """Let statement."""
        items = self._filter_tokens(items)

        is_mut = False
        idx = 0

        # Check for mut
        if idx < len(items) and isinstance(items[idx], LarkToken) and items[idx].type == 'MUT':
            is_mut = True
            idx += 1

        name = items[idx].value
        idx += 1

        # Check for binding (register or identifier)
        binding = None
        if idx < len(items) and isinstance(items[idx], (ast.Register, ast.Identifier)):
            binding = items[idx]
            idx += 1

        # Type annotation (optional) - comes after binding
        var_type = None
        if idx < len(items) and isinstance(items[idx], ast.Type):
            var_type = items[idx]
            idx += 1

        # Initializer (always last)
        initializer = items[idx] if idx < len(items) else None

        return ast.LetStmt(
            is_mut=is_mut,
            name=name,
            binding=binding,
            var_type=var_type,
            initializer=initializer
        )

    def expr_stmt(self, items):
        """Expression statement."""
        items = self._filter_tokens(items)
        return ast.ExprStmt(expr=items[0])

    def return_stmt(self, items):
        """Return statement."""
        # Filter out 'return' keyword and semicolon, keep only expressions
        values = [item for item in items if not isinstance(item, LarkToken)]
        return ast.ReturnStmt(values=values)

    def break_stmt(self, items):
        """Break statement."""
        return ast.BreakStmt()

    def continue_stmt(self, items):
        """Continue statement."""
        return ast.ContinueStmt()

    def increment_stmt(self, items):
        """Increment statement (x++;) - desugars to x += 1;"""
        items = self._filter_tokens(items)
        lvalue = items[0]
        # Desugar to compound assignment: x++ becomes x += 1
        compound_assign = ast.CompoundAssignment(
            target=lvalue,
            operator='+',
            value=ast.IntegerLiteral(value=1)
        )
        return ast.ExprStmt(expr=compound_assign)

    def decrement_stmt(self, items):
        """Decrement statement (x--;) - desugars to x -= 1;"""
        items = self._filter_tokens(items)
        lvalue = items[0]
        # Desugar to compound assignment: x-- becomes x -= 1
        compound_assign = ast.CompoundAssignment(
            target=lvalue,
            operator='-',
            value=ast.IntegerLiteral(value=1)
        )
        return ast.ExprStmt(expr=compound_assign)

    def if_stmt(self, items):
        """If statement."""
        items = self._filter_tokens(items)
        condition = items[0]
        then_block = items[1]
        else_block = items[2] if len(items) > 2 else None
        return ast.IfStmt(condition=condition, then_block=then_block, else_block=else_block)

    def else_clause(self, items):
        """Else clause."""
        items = self._filter_tokens(items)
        return items[0]

    def loop_stmt(self, items):
        """Loop statement."""
        items = self._filter_tokens(items)
        return ast.LoopStmt(body=items[0])

    def while_stmt(self, items):
        """While statement."""
        items = self._filter_tokens(items)
        condition = items[0]
        body = items[1]
        return ast.WhileStmt(condition=condition, body=body)

    def asm_stmt(self, items):
        """Inline assembly statement."""
        # Keep only STRING tokens
        items = self._filter_tokens(items, keep_types={'STRING'})
        instructions = [item.value.strip('"') for item in items]
        return ast.AsmStmt(instructions=instructions)

    # ========================================================================
    # Expressions
    # ========================================================================

    def integer(self, items):
        """Integer literal."""
        value = self._parse_integer(items[0].value)
        return ast.IntegerLiteral(value=value)

    def boolean(self, items):
        """Boolean literal."""
        value = items[0].value == 'true'
        return ast.BooleanLiteral(value=value)

    def identifier(self, items):
        """Identifier."""
        token = items[0]
        identifier = token.value
        # Validate that this isn't a wrong-case register name
        self._validate_identifier_not_register(identifier, token)
        return ast.Identifier(name=identifier)

    def register_ref(self, items):
        """Register reference."""
        return ast.Register(name=items[0].value)

    def paren(self, items):
        """Parenthesized expression."""
        return items[0]  # Just return the inner expression

    # ========================================================================
    # Operation Handler Factories
    # ========================================================================

    @staticmethod
    def _make_binary_op_handler(operator: str):
        """
        Create a binary operation handler for a given operator.

        Args:
            operator: The operator string ('+', '-', '*', etc.)

        Returns:
            A handler function for Lark Transformer
        """
        def handler(self, items):
            items = self._filter_tokens(items)
            return ast.BinaryOp(op=operator, left=items[0], right=items[1])
        return handler

    @staticmethod
    def _make_unary_op_handler(operator: str):
        """
        Create a unary operation handler for a given operator.

        Args:
            operator: The operator string ('!', '~', '-')

        Returns:
            A handler function for Lark Transformer
        """
        def handler(self, items):
            items = self._filter_tokens(items)
            return ast.UnaryOp(op=operator, operand=items[0])
        return handler

    # ========================================================================
    # Binary Operations (generated via factory)
    # ========================================================================

    # Arithmetic operators
    add = _make_binary_op_handler('+')
    sub = _make_binary_op_handler('-')
    mul = _make_binary_op_handler('*')
    div = _make_binary_op_handler('/')
    mod = _make_binary_op_handler('%')

    # Bitwise operators
    bitand = _make_binary_op_handler('&')
    bitor = _make_binary_op_handler('|')
    bitxor = _make_binary_op_handler('^')
    lshift = _make_binary_op_handler('<<')
    rshift = _make_binary_op_handler('>>')

    # Comparison operators
    eq = _make_binary_op_handler('==')
    ne = _make_binary_op_handler('!=')
    lt = _make_binary_op_handler('<')
    le = _make_binary_op_handler('<=')
    gt = _make_binary_op_handler('>')
    ge = _make_binary_op_handler('>=')

    # Logical operators
    and_expr = _make_binary_op_handler('&&')
    or_expr = _make_binary_op_handler('||')

    # ========================================================================
    # Unary Operations (generated via factory)
    # ========================================================================

    not_expr = _make_unary_op_handler('!')
    bitnot = _make_unary_op_handler('~')
    neg = _make_unary_op_handler('-')

    def deref(self, items):
        items = self._filter_tokens(items)
        return ast.Dereference(pointer=items[0])

    # Postfix operations
    def call(self, items):
        """Function call."""
        items = self._filter_tokens(items)
        func = items[0]
        args = items[1] if len(items) > 1 and isinstance(items[1], list) else []
        return ast.FunctionCall(func=func, args=args)

    def arg_list(self, items):
        """Argument list."""
        # Filter out commas, parentheses - keep only expressions
        return [item for item in items if not isinstance(item, LarkToken)]

    def array_index(self, items):
        """Array indexing."""
        items = self._filter_tokens(items)
        return ast.ArrayIndex(array=items[0], index=items[1])

    def field_access(self, items):
        """Field access."""
        items = self._filter_tokens(items, keep_types={'IDENT'})
        base = items[0]
        field = items[1].value if isinstance(items[1], LarkToken) else items[1]
        return ast.FieldAccess(base=base, field=field)

    def type_cast(self, items):
        """Type cast."""
        items = self._filter_tokens(items)
        return ast.TypeCast(expr=items[0], target_type=items[1])

    # Assignment
    def assign(self, items):
        """Assignment."""
        items = self._filter_tokens(items)
        lvalue = items[0]
        value = items[1]

        # Convert lvalue to appropriate target
        if isinstance(lvalue, ast.Identifier):
            target = lvalue
        else:
            target = lvalue

        return ast.Assignment(target=target, value=value)

    def compound_assign(self, items):
        """Compound assignment (+=, -=, etc.)."""
        # Keep compound operator tokens
        items = self._filter_tokens(items, keep_types={
            'PLUSEQUAL', 'MINUSEQUAL', 'STAREQUAL', 'SLASHEQUAL', 'PERCENTEQUAL',
            'AMPEREQUAL', 'VBAREQUAL', 'CIRCUMFLEXEQUAL', 'LSHIFTEQUAL', 'RSHIFTEQUAL'
        })
        lvalue = items[0]
        # items[1] is the compound_op result (a token like PLUSEQUAL)
        compound_op_token = items[1]
        value = items[2]

        # Map compound operator token to binary operator
        op_map = {
            'PLUSEQUAL': '+',
            'MINUSEQUAL': '-',
            'STAREQUAL': '*',
            'SLASHEQUAL': '/',
            'PERCENTEQUAL': '%',
            'AMPEREQUAL': '&',
            'VBAREQUAL': '|',
            'CIRCUMFLEXEQUAL': '^',
            'LSHIFTEQUAL': '<<',
            'RSHIFTEQUAL': '>>'
        }

        # Get the operator type
        if isinstance(compound_op_token, LarkToken):
            op_type = compound_op_token.type
        else:
            # It might be wrapped; extract the actual token type
            op_type = str(compound_op_token)

        operator = op_map.get(op_type, '+')  # Default to '+' if unknown

        return ast.CompoundAssignment(target=lvalue, operator=operator, value=value)

    def compound_op(self, items):
        """Compound operator."""
        # Return the first token which is the compound operator
        # Keep all compound operator tokens
        items = self._filter_tokens(items, keep_types={
            'PLUSEQUAL', 'MINUSEQUAL', 'STAREQUAL', 'SLASHEQUAL', 'PERCENTEQUAL',
            'AMPEREQUAL', 'VBAREQUAL', 'CIRCUMFLEXEQUAL', 'LSHIFTEQUAL', 'RSHIFTEQUAL'
        })
        return items[0]

    def lvalue_ident(self, items):
        """Lvalue identifier."""
        items = self._filter_tokens(items)
        token = items[0]
        identifier = token.value if isinstance(token, LarkToken) else token
        # Validate that this isn't a wrong-case register name
        if isinstance(token, LarkToken):
            self._validate_identifier_not_register(identifier, token)
        return ast.Identifier(name=identifier)

    def lvalue_register(self, items):
        """Lvalue register."""
        items = self._filter_tokens(items, keep_types={'REGISTER'})
        return ast.Register(name=items[0].value if isinstance(items[0], LarkToken) else items[0])

    def lvalue_array(self, items):
        """Lvalue array index."""
        items = self._filter_tokens(items)
        return ast.ArrayIndex(array=items[0], index=items[1])

    def lvalue_field(self, items):
        """Lvalue field access."""
        items = self._filter_tokens(items, keep_types={'IDENT'})
        return ast.FieldAccess(base=items[0], field=items[1].value if isinstance(items[1], LarkToken) else items[1])

    # ========================================================================
    # Types
    # ========================================================================

    def type_basic(self, items):
        """Basic type."""
        items = self._filter_tokens(items, keep_types={'TYPE_NAME'})
        return ast.BasicType(name=items[0].value if isinstance(items[0], LarkToken) else items[0])

    def type_array(self, items):
        """Array type."""
        items = self._filter_tokens(items)
        element_type = items[0]
        size = items[1]
        return ast.ArrayType(element_type=element_type, size=size)

    def type_pointer(self, items):
        """Pointer type."""
        items = self._filter_tokens(items, keep_types={'FAR', 'TYPE_NAME'})
        is_far = items[0].value == 'far' if isinstance(items[0], LarkToken) and items[0].type == 'FAR' else False
        pointee_type = items[1] if len(items) > 1 else items[0]
        return ast.PointerType(is_far=is_far, pointee_type=pointee_type)

    def type_fn(self, items):
        """Function type."""
        items = self._filter_tokens(items, keep_types={'FAR'})
        is_far = False
        idx = 0

        if idx < len(items) and isinstance(items[idx], LarkToken) and items[idx].type == 'FAR':
            is_far = True
            idx += 1

        param_types = []
        return_type = None

        if idx < len(items):
            if isinstance(items[idx], list):
                param_types = items[idx]
                idx += 1

            if idx < len(items):
                return_type = items[idx]

        return ast.FunctionType(
            is_far=is_far,
            param_types=param_types,
            return_type=return_type
        )

    def type_list(self, items):
        """Type list."""
        # Filter out commas - keep only type nodes
        return [item for item in items if not isinstance(item, LarkToken)]


class Parser:
    """Parser for R65 source code."""

    def __init__(self):
        """Initialize the parser."""
        self.lark = Lark(GRAMMAR, parser='lalr', start='start', keep_all_tokens=True)
        self.transformer = ASTBuilder()

    def parse(self, source: str, filename: str = "<input>") -> ast.Program:
        """
        Parse source code into an AST.

        Args:
            source: Source code to parse
            filename: Name of the source file (for error messages)

        Returns:
            Program AST node

        Raises:
            ParseError: If parsing fails
        """
        try:
            tree = self.lark.parse(source)
            program = self.transformer.transform(tree)
            return program
        except Exception as e:
            raise ParseError(f"Parse error in {filename}: {e}") from e


class ParseError(Exception):
    """Exception raised when parsing fails."""
    pass


def parse(source: str, filename: str = "<input>") -> ast.Program:
    """
    Convenience function to parse source code.

    Args:
        source: Source code to parse
        filename: Name of the source file

    Returns:
        Program AST node
    """
    parser = Parser()
    return parser.parse(source, filename)
