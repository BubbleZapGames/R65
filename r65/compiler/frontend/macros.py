"""
Macro expander for R65.

Handles macro definition collection and invocation expansion.
Works at the AST level, expanding macro invocations into AST nodes.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Union, Callable, TypeVar
from copy import deepcopy

T = TypeVar('T')

from r65.compiler.frontend import ast
from r65.compiler.frontend.parser import parse
from r65.compiler.errors import MacroError, SourceLocation


@dataclass
class MacroDefinition:
    """Stored macro definition."""
    name: str
    params: List[ast.MacroParam]
    body_tokens: List[str]
    source_loc: Optional[SourceLocation] = None


class MacroExpander:
    """
    Expands macro invocations in an AST.

    The expander:
    1. Collects all macro definitions from the AST
    2. Replaces macro invocations with expanded code
    3. Handles parameter substitution and repetition
    """

    MAX_EXPANSION_DEPTH = 64

    # Hints for common Rust macros that don't exist in R65
    _RUST_MACRO_HINTS = {
        'println':       "use format!() to write to a buffer, then output via hardware",
        'print':         "use format!() to write to a buffer, then output via hardware",
        'eprintln':      "use format!() to write to a buffer",
        'dbg':           "use format!() to write to a buffer",
        'panic':         "use asm!(\"BRK\") for traps",
        'todo':          "use asm!(\"BRK\") for traps",
        'unimplemented': "use asm!(\"BRK\") for traps",
        'assert':        "use if !condition { asm!(\"BRK\"); }",
        'assert_eq':     "use if a != b { asm!(\"BRK\"); }",
        'assert_ne':     "use if a == b { asm!(\"BRK\"); }",
        'vec':           "use fixed-size array literals: [1, 2, 3]",
    }

    def __init__(self):
        self.macros: Dict[str, MacroDefinition] = {}
        self._impl_macros: Dict[str, Dict[str, MacroDefinition]] = {}  # struct_name → {macro_name → MacroDef}
        self.warnings: List[str] = []
        self._expansion_depth = 0
        self._expanding: Set[str] = set()  # Track currently expanding macros
        self._program_items: List[ast.Declaration] = []  # Store program declarations for symbol! resolution
        self._format_literal_counter = 0  # Unique ID for format string literal statics

    def _invoke_macro(
        self,
        name: str,
        args: List[str],
        source_loc: Optional[SourceLocation],
        parse_expanded: Callable[[str, str], T]
    ) -> T:
        """
        Common macro invocation logic shared by all expansion contexts.

        Handles: macro lookup, recursion check, depth check, parameter matching,
        token substitution, and string operation processing.

        Args:
            name: Macro name
            args: List of argument token strings
            source_loc: Source location of invocation
            parse_expanded: Function(expanded_source, macro_name) -> result

        Returns:
            Result from parse_expanded callback
        """
        # Check if macro exists
        if name not in self.macros:
            hint = self._RUST_MACRO_HINTS.get(name)
            error = MacroError(f"undefined macro: '{name}'", source_loc)
            if hint:
                error.hint = hint
            raise error

        macro = self.macros[name]

        # Check for recursive expansion
        if name in self._expanding:
            raise MacroError(f"recursive macro expansion: '{name}'", source_loc)

        # Check expansion depth
        self._expansion_depth += 1
        if self._expansion_depth > self.MAX_EXPANSION_DEPTH:
            raise MacroError(
                f"macro expansion depth exceeded ({self.MAX_EXPANSION_DEPTH} levels)",
                source_loc
            )

        self._expanding.add(name)

        try:
            # Match arguments to parameters
            bindings = self._match_params(macro, args, source_loc)

            # Substitute parameters in body
            expanded_tokens = self._substitute(macro.body_tokens, bindings)

            # Process string operations (stringify!, string concatenation)
            expanded_tokens = self._process_string_operations(expanded_tokens)

            # Join tokens and call the context-specific parser
            expanded_source = self._join_tokens(expanded_tokens)

            try:
                return parse_expanded(expanded_source, name)
            except MacroError:
                raise
            except Exception as e:
                raise MacroError(
                    f"error parsing expanded macro '{name}': {e}\n"
                    f"Expanded source: {expanded_source}",
                    source_loc
                )
        finally:
            self._expanding.remove(name)
            self._expansion_depth -= 1

    def expand(self, program: ast.Program) -> ast.Program:
        """
        Expand all macros in a program.

        Args:
            program: The parsed program AST

        Returns:
            A new program with macros expanded
        """
        # Store program items for symbol! resolution
        self._program_items = program.items

        # First pass: collect all macro definitions
        self._collect_macros(program)

        # Second pass: expand macro invocations in all declarations
        new_items = self._expand_declarations(program.items)

        return ast.Program(items=new_items, source_loc=program.source_loc)

    def _collect_macros(self, program: ast.Program):
        """Collect all macro definitions from the program."""
        for item in program.items:
            if isinstance(item, ast.MacroDecl):
                # Clean up body tokens - remove outer braces if present
                body = item.body_tokens
                if body and body[0] == '{' and body[-1] == '}':
                    body = body[1:-1]

                self.macros[item.name] = MacroDefinition(
                    name=item.name,
                    params=item.params,
                    body_tokens=body,
                    source_loc=item.source_loc
                )
            elif isinstance(item, ast.ImplDecl):
                # Collect impl macros
                for macro in item.macros:
                    self._register_impl_macro(item.struct_name, macro)

    def _expand_declarations(self, items: List[ast.Declaration]) -> List[ast.Declaration]:
        """Expand macro invocations in a list of declarations."""
        result: List[ast.Declaration] = []

        for item in items:
            if isinstance(item, ast.MacroDecl):
                # Skip macro definitions - they've been collected
                continue
            elif isinstance(item, ast.MacroInvocationStmt):
                # Expand the macro invocation at top level
                expanded = self._expand_top_level_invocation(
                    item.name, item.args, item.source_loc
                )
                result.extend(expanded)
            elif isinstance(item, ast.FunctionDecl):
                # Expand macros inside function body
                new_item = self._expand_function(item)
                result.append(new_item)
            elif isinstance(item, ast.StaticDecl):
                # Expand macros in static initializer
                new_init = None
                if item.initializer:
                    new_init = self._expand_expression(item.initializer)
                new_item = ast.StaticDecl(
                    attributes=item.attributes,
                    is_far=item.is_far,
                    is_mut=item.is_mut,
                    name=item.name,
                    var_type=item.var_type,
                    initializer=new_init,
                    source_loc=item.source_loc
                )
                result.append(new_item)
            elif isinstance(item, ast.ConstDecl):
                # Expand macros in const value
                new_value = self._expand_expression(item.value)
                new_item = ast.ConstDecl(
                    name=item.name,
                    const_type=item.const_type,
                    value=new_value,
                    source_loc=item.source_loc
                )
                result.append(new_item)
            elif isinstance(item, ast.ImplDecl):
                # Expand macros inside impl method bodies
                new_methods = []
                for method in item.methods:
                    new_body = self._expand_block(method.body)
                    new_methods.append(ast.ImplMethod(
                        attributes=method.attributes,
                        is_far=method.is_far,
                        name=method.name,
                        self_is_far=method.self_is_far,
                        params=method.params,
                        return_type=method.return_type,
                        body=new_body,
                        source_loc=method.source_loc,
                        is_const=method.is_const,
                    ))
                new_constants = []
                for const in item.constants:
                    new_value = self._expand_expression(const.value)
                    new_constants.append(ast.ImplConst(
                        name=const.name,
                        const_type=const.const_type,
                        value=new_value,
                        source_loc=const.source_loc,
                    ))
                # Impl macros are already collected; don't need to expand them
                result.append(ast.ImplDecl(
                    struct_name=item.struct_name,
                    is_far=item.is_far,
                    methods=new_methods,
                    constants=new_constants,
                    trait_name=item.trait_name,
                    macros=item.macros,
                    source_loc=item.source_loc,
                ))
            else:
                # Keep other declarations as-is
                result.append(item)

        return result

    def _expand_function(self, func: ast.FunctionDecl) -> ast.FunctionDecl:
        """Expand macros inside a function body."""
        new_body = self._expand_block(func.body)
        return ast.FunctionDecl(
            attributes=func.attributes,
            is_far=func.is_far,
            name=func.name,
            params=func.params,
            return_type=func.return_type,
            body=new_body,
            source_loc=func.source_loc,
            is_const=func.is_const
        )

    def _expand_block(self, block: ast.Block) -> ast.Block:
        """Expand macros inside a block."""
        new_statements = self._expand_statements(block.statements)
        return ast.Block(statements=new_statements, source_loc=block.source_loc)

    def _expand_statements(self, statements: List[ast.Statement]) -> List[ast.Statement]:
        """Expand macro invocations in a list of statements."""
        result: List[ast.Statement] = []

        for stmt in statements:
            if isinstance(stmt, ast.MacroInvocationStmtInner):
                # Expand the macro invocation
                expanded = self._expand_statement_invocation(
                    stmt.name, stmt.args, stmt.source_loc
                )
                result.extend(expanded)
            elif isinstance(stmt, ast.IfStmt):
                new_stmt = self._expand_if(stmt)
                result.append(new_stmt)
            elif isinstance(stmt, ast.LoopStmt):
                new_stmt = ast.LoopStmt(
                    body=self._expand_block(stmt.body),
                    label=getattr(stmt, 'label', None),
                    source_loc=stmt.source_loc
                )
                result.append(new_stmt)
            elif isinstance(stmt, ast.WhileStmt):
                new_stmt = ast.WhileStmt(
                    condition=self._expand_expression(stmt.condition),
                    body=self._expand_block(stmt.body),
                    label=getattr(stmt, 'label', None),
                    source_loc=stmt.source_loc
                )
                result.append(new_stmt)
            elif isinstance(stmt, ast.Block):
                new_stmt = self._expand_block(stmt)
                result.append(new_stmt)
            elif isinstance(stmt, ast.LetStmt):
                # Expand macros in let statement initializer
                new_init = self._expand_expression(stmt.initializer) if stmt.initializer else None
                new_stmt = ast.LetStmt(
                    is_mut=stmt.is_mut,
                    name=stmt.name,
                    binding=stmt.binding,
                    var_type=stmt.var_type,
                    initializer=new_init,
                    pattern=getattr(stmt, 'pattern', None),
                    source_loc=stmt.source_loc
                )
                result.append(new_stmt)
            elif isinstance(stmt, ast.ExprStmt):
                # Check if this is a macro invocation at statement level
                if isinstance(stmt.expr, ast.MacroInvocation):
                    # Expand as statements, not expression
                    expanded = self._expand_statement_invocation(
                        stmt.expr.name, stmt.expr.args, stmt.expr.source_loc
                    )
                    result.extend(expanded)
                elif isinstance(stmt.expr, ast.MethodMacro):
                    # Method macro invocation: receiver.name!(args)
                    expanded = self._expand_method_macro(stmt.expr)
                    result.extend(expanded)
                else:
                    # Expand macros in expression
                    new_expr = self._expand_expression(stmt.expr)
                    result.append(ast.ExprStmt(expr=new_expr, source_loc=stmt.source_loc))
            elif isinstance(stmt, ast.ReturnStmt):
                # Expand macros in return values
                new_values = [self._expand_expression(v) for v in stmt.values]
                result.append(ast.ReturnStmt(values=new_values, source_loc=stmt.source_loc))
            elif isinstance(stmt, ast.ForStmt):
                new_stmt = ast.ForStmt(
                    variable=stmt.variable,
                    start=self._expand_expression(stmt.start),
                    end=self._expand_expression(stmt.end),
                    body=self._expand_block(stmt.body),
                    label=getattr(stmt, 'label', None),
                    inclusive=getattr(stmt, 'inclusive', False),
                    source_loc=stmt.source_loc
                )
                result.append(new_stmt)
            elif isinstance(stmt, ast.BreakStmt):
                if stmt.value is not None:
                    new_value = self._expand_expression(stmt.value)
                    new_stmt = ast.BreakStmt(
                        label=stmt.label,
                        value=new_value,
                        source_loc=stmt.source_loc
                    )
                    result.append(new_stmt)
                else:
                    result.append(stmt)
            else:
                # Keep other statements as-is
                result.append(stmt)

        return result

    def _expand_expression(self, expr: ast.Expression) -> ast.Expression:
        """Recursively expand macro invocations in an expression."""
        if expr is None:
            return None

        if isinstance(expr, ast.MacroInvocation):
            # Handle compile_error! - raises error at compile time
            if expr.name == "compile_error":
                msg = ' '.join(expr.args) if expr.args else "explicit compile error"
                # Strip quotes if the message is a string literal
                if msg.startswith('"') and msg.endswith('"'):
                    msg = msg[1:-1]
                raise MacroError(msg, expr.source_loc)
            # Handle stringify! specially
            if expr.name == "stringify":
                return self._expand_stringify_expr(expr.args, expr.source_loc)
            # Handle symbol! specially
            if expr.name == "symbol":
                return self._expand_symbol_expr(expr.args, expr.source_loc)
            # Expand user-defined macro to expression
            return self._expand_expression_invocation(expr.name, expr.args, expr.source_loc)

        elif isinstance(expr, ast.BinaryOp):
            left = self._expand_expression(expr.left)
            right = self._expand_expression(expr.right)

            # Compile-time string concatenation with + operator
            if expr.op == '+' and isinstance(left, ast.StringLiteral) and isinstance(right, ast.StringLiteral):
                return ast.StringLiteral(
                    value=left.value + right.value,
                    source_loc=expr.source_loc
                )

            return ast.BinaryOp(
                op=expr.op,
                left=left,
                right=right,
                source_loc=expr.source_loc
            )

        elif isinstance(expr, ast.UnaryOp):
            return ast.UnaryOp(
                op=expr.op,
                operand=self._expand_expression(expr.operand),
                source_loc=expr.source_loc
            )

        elif isinstance(expr, ast.FunctionCall):
            new_args = [self._expand_expression(arg) for arg in expr.args]
            return ast.FunctionCall(
                func=self._expand_expression(expr.func),
                args=new_args,
                source_loc=expr.source_loc
            )

        elif isinstance(expr, ast.ArrayIndex):
            return ast.ArrayIndex(
                array=self._expand_expression(expr.array),
                index=self._expand_expression(expr.index),
                source_loc=expr.source_loc
            )

        elif isinstance(expr, ast.FieldAccess):
            return ast.FieldAccess(
                base=self._expand_expression(expr.base),
                field=expr.field,
                source_loc=expr.source_loc
            )

        elif isinstance(expr, ast.TypeCast):
            return ast.TypeCast(
                expr=self._expand_expression(expr.expr),
                target_type=expr.target_type,
                source_loc=expr.source_loc
            )

        elif isinstance(expr, ast.Dereference):
            return ast.Dereference(
                pointer=self._expand_expression(expr.pointer),
                source_loc=expr.source_loc
            )

        elif isinstance(expr, ast.AddressOf):
            return ast.AddressOf(
                operand=self._expand_expression(expr.operand),
                source_loc=expr.source_loc
            )

        elif isinstance(expr, ast.ArrayLiteralExpr):
            new_elements = [self._expand_expression(e) for e in expr.elements]
            return ast.ArrayLiteralExpr(
                elements=new_elements,
                source_loc=expr.source_loc
            )

        elif isinstance(expr, ast.ArrayFillExpr):
            return ast.ArrayFillExpr(
                value=self._expand_expression(expr.value),
                count=self._expand_expression(expr.count),
                source_loc=expr.source_loc
            )

        elif isinstance(expr, ast.StructLiteralExpr):
            new_fields = []
            for field in expr.fields:
                new_fields.append(ast.StructFieldInit(
                    name=field.name,
                    value=self._expand_expression(field.value),
                    source_loc=field.source_loc
                ))
            return ast.StructLiteralExpr(
                struct_name=expr.struct_name,
                fields=new_fields,
                source_loc=expr.source_loc
            )

        elif isinstance(expr, ast.MatchExpression):
            new_arms = []
            for arm in expr.arms:
                new_arms.append(ast.MatchArm(
                    pattern=arm.pattern,
                    body=self._expand_expression(arm.body),
                    source_loc=arm.source_loc
                ))
            return ast.MatchExpression(
                scrutinee=self._expand_expression(expr.scrutinee),
                arms=new_arms,
                source_loc=expr.source_loc
            )

        elif isinstance(expr, ast.LoopExpression):
            return ast.LoopExpression(
                body=self._expand_block(expr.body),
                label=getattr(expr, 'label', None),
                source_loc=expr.source_loc
            )

        elif isinstance(expr, ast.Assignment):
            return ast.Assignment(
                target=self._expand_expression(expr.target),
                value=self._expand_expression(expr.value),
                source_loc=expr.source_loc
            )

        elif isinstance(expr, ast.CompoundAssignment):
            return ast.CompoundAssignment(
                target=self._expand_expression(expr.target),
                operator=expr.operator,
                value=self._expand_expression(expr.value),
                source_loc=expr.source_loc
            )

        # Leaf expressions don't need transformation
        return expr

    def _expand_expression_invocation(
        self,
        name: str,
        args: List[str],
        source_loc: Optional[SourceLocation]
    ) -> ast.Expression:
        """Expand a macro invocation in expression context."""
        def parse_as_expression(expanded_source: str, macro_name: str) -> ast.Expression:
            # Wrap in a dummy function with expression statement
            wrapped = f"fn __macro_expr__() {{ let __result__: u16 = {expanded_source}; }}"
            program = parse(wrapped, f"<macro:{macro_name}>")

            # Extract the expression from the let statement
            if program.items and isinstance(program.items[0], ast.FunctionDecl):
                func = program.items[0]
                if func.body.statements:
                    stmt = func.body.statements[0]
                    if isinstance(stmt, ast.LetStmt) and stmt.initializer:
                        # Recursively expand any nested macro invocations
                        expanded_expr = self._expand_expression(stmt.initializer)
                        # Override source_loc to point to invocation site
                        self._override_source_loc(expanded_expr, source_loc)
                        return expanded_expr

            raise MacroError(
                f"macro '{macro_name}' did not expand to a valid expression",
                source_loc
            )

        return self._invoke_macro(name, args, source_loc, parse_as_expression)

    def _expand_if(self, stmt: ast.IfStmt) -> ast.IfStmt:
        """Expand macros inside an if statement."""
        new_condition = self._expand_expression(stmt.condition)
        new_then = self._expand_block(stmt.then_block)
        new_else = None
        if stmt.else_block:
            if isinstance(stmt.else_block, ast.IfStmt):
                new_else = self._expand_if(stmt.else_block)
            else:
                new_else = self._expand_block(stmt.else_block)

        return ast.IfStmt(
            condition=new_condition,
            then_block=new_then,
            else_block=new_else,
            source_loc=stmt.source_loc
        )

    # =========================================================================
    # Impl (Method) Macros
    # =========================================================================

    def _register_impl_macro(self, struct_name: str, macro: 'ast.ImplMacro'):
        """Register a macro defined inside an impl block."""
        body = macro.body_tokens
        if body and body[0] == '{' and body[-1] == '}':
            body = body[1:-1]

        if struct_name not in self._impl_macros:
            self._impl_macros[struct_name] = {}

        self._impl_macros[struct_name][macro.name] = MacroDefinition(
            name=macro.name,
            params=macro.params,
            body_tokens=body,
            source_loc=macro.source_loc
        )

    def _resolve_receiver_type(self, receiver: ast.Expression) -> Optional[str]:
        """Try to resolve receiver expression to a struct type name."""
        if isinstance(receiver, ast.Identifier):
            for item in self._program_items:
                if isinstance(item, ast.StaticDecl) and item.name == receiver.name:
                    if isinstance(item.var_type, ast.BasicType):
                        return item.var_type.name
        return None  # Can't resolve — fall back to name-only search

    def _resolve_impl_macro(
        self,
        receiver: ast.Expression,
        name: str,
        source_loc: Optional[SourceLocation]
    ) -> MacroDefinition:
        """Find the right impl macro for a method macro invocation."""
        # Try type-directed lookup first
        struct_type = self._resolve_receiver_type(receiver)
        if struct_type and struct_type in self._impl_macros:
            if name in self._impl_macros[struct_type]:
                return self._impl_macros[struct_type][name]

        # Fallback: search all impl macros by name
        matches = [(sname, m) for sname, macros in self._impl_macros.items()
                    for mname, m in macros.items() if mname == name]
        if len(matches) == 1:
            return matches[0][1]
        if len(matches) == 0:
            raise MacroError(f"no method macro '{name}' found", source_loc)
        struct_names = ', '.join(s for s, _ in matches)
        raise MacroError(
            f"ambiguous method macro '{name}' — defined in: {struct_names}",
            source_loc
        )

    def _ast_to_source(self, node: ast.Expression) -> str:
        """Convert a parsed AST expression back to source text for macro substitution."""
        if isinstance(node, ast.Identifier):
            return node.name
        elif isinstance(node, ast.Register):
            return node.name
        elif isinstance(node, ast.FieldAccess):
            return f"{self._ast_to_source(node.base)}.{node.field}"
        elif isinstance(node, ast.ArrayIndex):
            return f"{self._ast_to_source(node.array)}[{self._ast_to_source(node.index)}]"
        elif isinstance(node, ast.Dereference):
            return f"*{self._ast_to_source(node.pointer)}"
        elif isinstance(node, ast.AddressOf):
            return f"&{self._ast_to_source(node.operand)}"
        elif isinstance(node, ast.IntegerLiteral):
            if node.suffix:
                return f"{node.value}{node.suffix}"
            return str(node.value)
        elif isinstance(node, ast.FunctionCall):
            func_src = self._ast_to_source(node.func)
            args_src = ', '.join(self._ast_to_source(a) for a in node.args)
            return f"{func_src}({args_src})"
        elif isinstance(node, ast.TypeCast):
            return f"{self._ast_to_source(node.expr)} as {self._type_to_source(node.target_type)}"
        elif isinstance(node, ast.BinaryOp):
            return f"{self._ast_to_source(node.left)} {node.op} {self._ast_to_source(node.right)}"
        elif isinstance(node, ast.UnaryOp):
            return f"{node.op}{self._ast_to_source(node.operand)}"
        else:
            # Fallback for unknown nodes
            return repr(node)

    def _type_to_source(self, t: ast.Type) -> str:
        """Convert an AST type node back to source text."""
        if isinstance(t, ast.BasicType):
            return t.name
        elif isinstance(t, ast.PointerType):
            far = "far " if t.is_far else ""
            return f"{far}*{self._type_to_source(t.pointee_type)}"
        elif isinstance(t, ast.ArrayType):
            return f"[{self._type_to_source(t.element_type)}; {self._ast_to_source(t.size)}]"
        return str(t)

    def _expand_method_macro(self, mm: ast.MethodMacro) -> List[ast.Statement]:
        """Expand a method macro invocation: receiver.name!(args)."""
        receiver_text = self._ast_to_source(mm.receiver)

        # Resolve which impl macro to use
        macro = self._resolve_impl_macro(mm.receiver, mm.name, mm.source_loc)

        # Replace 'self' with receiver text in body tokens
        body_tokens = [receiver_text if t == 'self' else t for t in macro.body_tokens]

        # Build a temporary MacroDefinition with the substituted body
        temp_macro = MacroDefinition(
            name=mm.name,
            params=macro.params,
            body_tokens=body_tokens,
            source_loc=macro.source_loc
        )

        # Save the original macro, temporarily register this as a regular macro
        old = self.macros.get(mm.name)
        self.macros[mm.name] = temp_macro
        try:
            expanded = self._expand_statement_invocation(
                mm.name, mm.args, mm.source_loc
            )
        finally:
            # Restore the original macro (or remove if it didn't exist)
            if old is not None:
                self.macros[mm.name] = old
            else:
                del self.macros[mm.name]

        return expanded

    def _expand_top_level_invocation(
        self,
        name: str,
        args: List[str],
        source_loc: Optional[SourceLocation]
    ) -> List[ast.Declaration]:
        """Expand a top-level macro invocation into declarations."""
        def parse_as_declarations(expanded_source: str, macro_name: str) -> List[ast.Declaration]:
            # Parse directly as a program (not wrapped in a function)
            program = parse(expanded_source, f"<macro:{macro_name}>")

            # Extract declarations and recursively expand nested macros
            result: List[ast.Declaration] = []
            for item in program.items:
                if isinstance(item, ast.MacroInvocationStmt):
                    # Recursively expand nested top-level macro invocations
                    nested = self._expand_top_level_invocation(
                        item.name, item.args, item.source_loc
                    )
                    result.extend(nested)
                elif isinstance(item, ast.MacroDecl):
                    # Register any new macro definitions from expansion
                    body = item.body_tokens
                    if body and body[0] == '{' and body[-1] == '}':
                        body = body[1:-1]
                    self.macros[item.name] = MacroDefinition(
                        name=item.name,
                        params=item.params,
                        body_tokens=body,
                        source_loc=item.source_loc
                    )
                elif isinstance(item, ast.FunctionDecl):
                    # Expand macros inside function bodies
                    expanded_func = self._expand_function(item)
                    result.append(expanded_func)
                else:
                    result.append(item)

            return result

        return self._invoke_macro(name, args, source_loc, parse_as_declarations)

    def _expand_statement_invocation(
        self,
        name: str,
        args: List[str],
        source_loc: Optional[SourceLocation]
    ) -> List[ast.Statement]:
        """Expand a statement-level macro invocation."""
        # Handle built-in compile_error! macro - raises error at compile time
        if name == "compile_error":
            msg = ' '.join(args) if args else "explicit compile error"
            # Strip quotes if the message is a string literal
            if msg.startswith('"') and msg.endswith('"'):
                msg = msg[1:-1]
            raise MacroError(msg, source_loc)
        # Handle built-in const_assert! macro
        if name == "const_assert":
            return self._expand_const_assert(args, source_loc)
        # Handle built-in stringify! macro
        if name == "stringify":
            return self._expand_stringify(args, source_loc)
        # Handle built-in symbol! macro
        if name == "symbol":
            return self._expand_symbol(args, source_loc)
        # Handle built-in __format! macro (wrapped by format! in string.r65)
        if name == "__format":
            return self._expand_format_macro(args, source_loc)

        def parse_as_statements(expanded_source: str, macro_name: str) -> List[ast.Statement]:
            # Wrap in a dummy function to parse as statements
            wrapped = f"fn __macro_expand__() {{ {expanded_source} }}"
            program = parse(wrapped, f"<macro:{macro_name}>")

            # Extract the statements from the function body
            if program.items and isinstance(program.items[0], ast.FunctionDecl):
                func = program.items[0]
                # Recursively expand any nested macro invocations
                expanded_stmts = self._expand_statements(func.body.statements)
                # Override source_loc on expanded statements to point to invocation site
                for stmt in expanded_stmts:
                    self._override_source_loc(stmt, source_loc)
                return expanded_stmts
            return []

        return self._invoke_macro(name, args, source_loc, parse_as_statements)

    def _match_params(
        self,
        macro: MacroDefinition,
        args: List[str],
        source_loc: Optional[SourceLocation]
    ) -> Dict[str, List[str]]:
        """
        Match arguments to macro parameters.

        Args:
            macro: The macro definition
            args: List of argument token strings
            source_loc: Source location for error reporting

        Returns:
            Dictionary mapping parameter names to their values (as token lists)
        """
        bindings: Dict[str, List[str]] = {}

        # Handle simple case: no repetition
        simple_params = [p for p in macro.params if not p.is_repeated]
        repeated_params = [p for p in macro.params if p.is_repeated]

        if repeated_params:
            if len(repeated_params) > 1:
                raise MacroError(
                    f"macro '{macro.name}' has multiple repeated parameters (not supported)",
                    source_loc
                )
            # Bind leading simple params first, remaining args go to repeated param
            n_simple = len(simple_params)
            if len(args) < n_simple:
                raise MacroError(
                    f"macro '{macro.name}' expects at least {n_simple} arguments, got {len(args)}",
                    source_loc
                )
            for param, arg in zip(simple_params, args[:n_simple]):
                bindings[param.name] = [arg]
            bindings[repeated_params[0].name] = args[n_simple:]
        else:
            # Simple matching: each arg to each param
            if len(args) != len(simple_params):
                raise MacroError(
                    f"macro '{macro.name}' expects {len(simple_params)} arguments, got {len(args)}",
                    source_loc
                )
            for param, arg in zip(simple_params, args):
                bindings[param.name] = [arg]

        return bindings

    def _substitute(
        self,
        tokens: List[str],
        bindings: Dict[str, List[str]]
    ) -> List[str]:
        """
        Substitute parameters in macro body tokens.

        Handles:
        - Simple substitution: $name -> value
        - Repetition: $(...),* -> expanded for each value

        Args:
            tokens: Body tokens
            bindings: Parameter bindings

        Returns:
            Substituted token list
        """
        result: List[str] = []
        i = 0

        while i < len(tokens):
            token = tokens[i]

            if token == '$(':
                # Find matching close and handle repetition
                depth = 1
                j = i + 1
                rep_content = []

                while j < len(tokens) and depth > 0:
                    if tokens[j] == '(':
                        depth += 1
                        rep_content.append(tokens[j])
                    elif tokens[j] == ')':
                        depth -= 1
                        if depth > 0:
                            rep_content.append(tokens[j])
                    else:
                        rep_content.append(tokens[j])
                    j += 1

                # Check for separator after )
                separator = ''
                if j < len(tokens):
                    next_tok = tokens[j]
                    if next_tok == ',*':
                        separator = ','
                        j += 1
                    elif next_tok == '*':
                        separator = ''
                        j += 1

                # Expand the repetition
                expanded = self._expand_repetition(rep_content, bindings, separator)
                result.extend(expanded)
                i = j

            elif token.startswith('$') and len(token) > 1:
                # Simple parameter reference
                param_name = token[1:]  # Remove $
                if param_name in bindings:
                    values = bindings[param_name]
                    if values:
                        result.append(values[0])  # Use first value for simple substitution
                else:
                    result.append(token)  # Keep unbound references
                i += 1
            else:
                result.append(token)
                i += 1

        return result

    def _expand_const_assert(
        self,
        args: List[str],
        source_loc: Optional[SourceLocation]
    ) -> List[ast.Statement]:
        """
        Expand the built-in const_assert! macro.

        const_assert!(condition, "message") checks condition at compile time.
        If condition evaluates to false, compilation fails with the message.

        Args:
            args: List of argument strings (already comma-separated by parser)
            source_loc: Source location of invocation

        Returns:
            List containing a ConstAssertStmt
        """
        if not args:
            raise MacroError("const_assert! requires a condition", source_loc)

        # First arg is the condition, second (optional) is the message
        condition_str = args[0].strip()

        if len(args) >= 2:
            # Message provided
            message = args[1].strip()
            # Strip quotes if it's a string literal
            if message.startswith('"') and message.endswith('"'):
                message = message[1:-1]
        else:
            message = "const assertion failed"

        # Parse condition as an expression
        try:
            wrapped = f"fn __const_assert__() {{ let __cond: bool = {condition_str}; }}"
            program = parse(wrapped, "<const_assert>")

            if program.items and isinstance(program.items[0], ast.FunctionDecl):
                func = program.items[0]
                if func.body.statements:
                    let_stmt = func.body.statements[0]
                    if isinstance(let_stmt, ast.LetStmt) and let_stmt.initializer:
                        condition_expr = let_stmt.initializer
                        # Create ConstAssertStmt
                        return [ast.ConstAssertStmt(
                            condition=condition_expr,
                            message=message,
                            source_loc=source_loc
                        )]

            raise MacroError(f"failed to parse const_assert! condition: {condition_str}", source_loc)
        except MacroError:
            raise
        except Exception as e:
            raise MacroError(f"error parsing const_assert! condition: {e}", source_loc)

    def _expand_stringify(
        self,
        args: List[str],
        source_loc: Optional[SourceLocation]
    ) -> List[ast.Statement]:
        """
        Expand the built-in stringify! macro.

        stringify!(arg1, arg2, ...) converts its arguments to a string literal.
        The arguments are concatenated with spaces between them.

        Args:
            args: List of argument token strings
            source_loc: Source location of invocation

        Returns:
            List containing a single statement with the string literal
        """
        # Join arguments with spaces
        joined_args = ' '.join(args)
        
        # Create a string literal by wrapping in quotes and escaping
        escaped_args = self._escape_string_literal(joined_args)
        string_literal = f'"{escaped_args}"'
        
        # Parse as a simple expression statement
        try:
            wrapped = f"fn __stringify_expand__() {{ {string_literal}; }}"
            program = parse(wrapped, "<stringify>")
            
            if program.items and isinstance(program.items[0], ast.FunctionDecl):
                func = program.items[0]
                if func.body.statements:
                    return func.body.statements
            return []
        except Exception as e:
            raise MacroError(
                f"error expanding stringify!: {e}",
                source_loc
            )

    def _escape_string_literal(self, text: str) -> str:
        """
        Escape text for use in a string literal.
        
        Args:
            text: Text to escape
            
        Returns:
            Escaped text safe for string literal
        """
        # Basic escaping - handle quotes, backslashes, newlines, tabs
        text = text.replace('\\', '\\\\')  # Backslash first
        text = text.replace('"', '\\"')    # Double quotes
        text = text.replace('\n', '\\n')   # Newline
        text = text.replace('\t', '\\t')   # Tab
        text = text.replace('\r', '\\r')   # Carriage return
        return text

    def _resolve_assembler_symbol(self, name: str) -> str:
        """
        Resolve an R65 identifier to its WLA-DX assembler label.

        Rules:
        - Immutable static with include_bytes! initializer -> "{name}_data"
        - Immutable static with any other initializer -> "__{name}_data"
        - Mutable static, function, const, or unknown -> "{name}" (pass-through)
        """
        for item in self._program_items:
            if isinstance(item, ast.StaticDecl) and item.name == name:
                if not item.is_mut and item.initializer is not None:
                    if isinstance(item.initializer, ast.IncludeBytesExpr):
                        return f"{name}_data"
                    else:
                        return f"__{name}_data"
                return name
            elif isinstance(item, (ast.FunctionDecl, ast.ConstDecl)) and item.name == name:
                return name
        # Unknown identifier - pass through
        return name

    def _expand_stringify_expr(self, args: List[str], source_loc: Optional[SourceLocation]) -> ast.StringLiteral:
        """
        Expand stringify! macro in expression context.
        
        Args:
            args: List of argument token strings
            source_loc: Source location of invocation
            
        Returns:
            String literal expression
        """
        # Join arguments with spaces
        joined_args = ' '.join(args)
        
        # Escape the text for string literal
        escaped_args = self._escape_string_literal(joined_args)
        
        # Return a string literal expression
        return ast.StringLiteral(value=escaped_args, source_loc=source_loc)

    def _expand_symbol(
        self,
        args: List[str],
        source_loc: Optional[SourceLocation]
    ) -> List[ast.Statement]:
        """
        Expand the built-in symbol! macro in statement context.

        symbol!(name) resolves an R65 identifier to its WLA-DX assembler label.

        Args:
            args: List of argument token strings (expects exactly one identifier)
            source_loc: Source location of invocation

        Returns:
            List containing a single statement with the resolved string literal
        """
        if not args:
            raise MacroError("symbol! requires exactly one identifier argument", source_loc)
        name = args[0].strip()
        resolved = self._resolve_assembler_symbol(name)
        escaped = self._escape_string_literal(resolved)
        string_literal = f'"{escaped}"'

        try:
            wrapped = f"fn __symbol_expand__() {{ {string_literal}; }}"
            program = parse(wrapped, "<symbol>")

            if program.items and isinstance(program.items[0], ast.FunctionDecl):
                func = program.items[0]
                if func.body.statements:
                    return func.body.statements
            return []
        except Exception as e:
            raise MacroError(
                f"error expanding symbol!: {e}",
                source_loc
            )

    def _expand_symbol_expr(self, args: List[str], source_loc: Optional[SourceLocation]) -> ast.StringLiteral:
        """
        Expand symbol! macro in expression context.

        Args:
            args: List of argument token strings (expects exactly one identifier)
            source_loc: Source location of invocation

        Returns:
            String literal expression with resolved assembler label
        """
        if not args:
            raise MacroError("symbol! requires exactly one identifier argument", source_loc)
        name = args[0].strip()
        resolved = self._resolve_assembler_symbol(name)
        escaped = self._escape_string_literal(resolved)
        return ast.StringLiteral(value=escaped, source_loc=source_loc)

    # =========================================================================
    # Built-in format! Macro
    # =========================================================================

    def _expand_format_macro(
        self,
        args: List[str],
        source_loc: Optional[SourceLocation]
    ) -> List[ast.Statement]:
        """
        Expand the built-in format! macro.

        format!(buf, "fmt string {u8} {u16:x}", arg1, arg2)

        Generates code that writes formatted output into the buffer using
        pointer arithmetic and calls to string.r65 functions.

        Args:
            args: [buffer_ident, format_string_literal, ...format_args]
            source_loc: Source location of invocation

        Returns:
            List of statements implementing the format operation
        """
        import re as _re

        if len(args) < 2:
            raise MacroError(
                "format! requires at least a buffer and format string: "
                "format!(BUF, \"text {u8}\", value)",
                source_loc
            )

        buf = args[0].strip()
        fmt_raw = args[1].strip()

        # Strip quotes from format string literal
        if not (fmt_raw.startswith('"') and fmt_raw.endswith('"')):
            raise MacroError(
                "format! second argument must be a string literal",
                source_loc
            )
        fmt = fmt_raw[1:-1]

        format_args = args[2:]  # Remaining args are format arguments

        # Parse the format string into segments
        segments = self._parse_format_string(fmt, source_loc)

        # Count specifiers and validate against provided args
        spec_count = sum(1 for seg_type, _ in segments if seg_type == 'specifier')
        if spec_count != len(format_args):
            raise MacroError(
                f"format! has {spec_count} format specifier(s) but "
                f"{len(format_args)} argument(s) were provided",
                source_loc
            )

        # Check if output can fit in target buffer
        self._check_format_buffer_overflow(buf, segments, source_loc)

        # Generate R65 source code
        lines = [f"let mut __fmtptr: far *u8 = &{buf} as far *u8;"]
        arg_idx = 0
        var_idx = 0

        for seg_type, seg_data in segments:
            if seg_type == 'literal':
                text = seg_data['text']
                byte_len = self._compute_literal_byte_length(text)
                if byte_len > 0:
                    if byte_len <= 3:
                        # Inline byte writes for small literals
                        for b in self._literal_to_bytes(text):
                            lines.append(f'*__fmtptr = {hex(b)};')
                            lines.append('__fmtptr = __fmtptr + 1;')
                    else:
                        # Emit a static ROM array and for-loop copy
                        escaped = self._escape_format_literal(text)
                        lit_id = self._format_literal_counter
                        self._format_literal_counter += 1
                        static_name = f'__fmtstr_{lit_id}'
                        idx_var = f'__fmti{var_idx}'
                        # Inject static into program items
                        static_decl = ast.StaticDecl(
                            attributes=[],
                            is_far=False,
                            is_mut=False,
                            name=static_name,
                            var_type=ast.ArrayType(
                                element_type=ast.BasicType(name='u8'),
                                size=ast.IntegerLiteral(value=byte_len),
                            ),
                            initializer=ast.StringLiteral(value=escaped),
                        )
                        if source_loc:
                            static_decl.source_loc = source_loc
                        self._program_items.append(static_decl)
                        lines.append(
                            f'for {idx_var} in 0..{byte_len} {{ __fmtptr[{idx_var}] = {static_name}[{idx_var}]; }}'
                        )
                        lines.append(f'__fmtptr = __fmtptr + {byte_len};')
                        var_idx += 1
            elif seg_type == 'specifier':
                spec = seg_data
                arg = format_args[arg_idx].strip()

                if spec['type'] == 'u8' and spec.get('format') == 'd':
                    if spec.get('width'):
                        w = spec['width']
                        fill = '0x30' if spec.get('zero_pad') else '0x20'
                        lines.append(
                            f'u8_to_dec_pad(__fmtptr, {arg}, {w}, {fill});'
                        )
                        lines.append(f'__fmtptr = __fmtptr + {w};')
                    else:
                        lines.append(
                            f'let __fmtn{var_idx}: u8 = u8_to_dec(__fmtptr, {arg});'
                        )
                        lines.append(
                            f'__fmtptr = __fmtptr + __fmtn{var_idx} as u16;'
                        )
                        var_idx += 1
                elif spec['type'] == 'u16' and spec.get('format') == 'd':
                    if spec.get('width'):
                        w = spec['width']
                        fill = '0x30' if spec.get('zero_pad') else '0x20'
                        lines.append(
                            f'u16_to_dec_pad(__fmtptr, {arg}, {w}, {fill});'
                        )
                        lines.append(f'__fmtptr = __fmtptr + {w};')
                    else:
                        lines.append(
                            f'let __fmtn{var_idx}: u8 = u16_to_dec(__fmtptr, {arg});'
                        )
                        lines.append(
                            f'__fmtptr = __fmtptr + __fmtn{var_idx} as u16;'
                        )
                        var_idx += 1
                elif spec['type'] == 'u8' and spec.get('format') == 'x':
                    lines.append(f'u8_to_hex(__fmtptr, {arg});')
                    lines.append('__fmtptr = __fmtptr + 2;')
                elif spec['type'] == 'u16' and spec.get('format') == 'x':
                    lines.append(f'u16_to_hex(__fmtptr, {arg});')
                    lines.append('__fmtptr = __fmtptr + 4;')
                elif spec['type'] == 's':
                    # Cast strcpy return (u16) to u8 then back to u16.
                    # This forces a mode switch that prevents hw-coalescence
                    # of the return value to A, avoiding a clobber when the
                    # pointer address is loaded for the subsequent addition.
                    # String lengths >255 are not expected on SNES.
                    lines.append(
                        f'let __fmtn{var_idx}: u8 = strcpy(__fmtptr, {arg}) as u8;'
                    )
                    lines.append(
                        f'__fmtptr = __fmtptr + __fmtn{var_idx} as u16;'
                    )
                    var_idx += 1
                elif spec['type'] == 'c':
                    lines.append(f'*__fmtptr = {arg};')
                    lines.append('__fmtptr = __fmtptr + 1;')
                elif spec['type'] == 'bool':
                    lines.append(
                        f'if {arg} {{ *__fmtptr = 0x31; }}'
                        f' else {{ *__fmtptr = 0x30; }}'
                    )
                    lines.append('__fmtptr = __fmtptr + 1;')
                elif spec['type'] == 'i8' and spec.get('format') == 'd':
                    # Inline sign check: if negative, write '-' and negate
                    lines.append(
                        f'let __fmts{var_idx}: u8 = {arg} as u8;'
                    )
                    lines.append(
                        f'if __fmts{var_idx} & 0x80 != 0 {{'
                        f' *__fmtptr = 0x2D; __fmtptr = __fmtptr + 1;'
                        f' __fmts{var_idx} = 0 - __fmts{var_idx};'
                        f' }}'
                    )
                    lines.append(
                        f'let __fmtn{var_idx}: u8 = u8_to_dec(__fmtptr, __fmts{var_idx});'
                    )
                    lines.append(
                        f'__fmtptr = __fmtptr + __fmtn{var_idx} as u16;'
                    )
                    var_idx += 1
                elif spec['type'] == 'i16' and spec.get('format') == 'd':
                    # Inline sign check: if negative, write '-' and negate
                    lines.append(
                        f'let __fmts{var_idx}: u16 = {arg} as u16;'
                    )
                    lines.append(
                        f'if __fmts{var_idx} & 0x8000 != 0 {{'
                        f' *__fmtptr = 0x2D; __fmtptr = __fmtptr + 1;'
                        f' __fmts{var_idx} = 0 - __fmts{var_idx};'
                        f' }}'
                    )
                    lines.append(
                        f'let __fmtn{var_idx}: u8 = u16_to_dec(__fmtptr, __fmts{var_idx});'
                    )
                    lines.append(
                        f'__fmtptr = __fmtptr + __fmtn{var_idx} as u16;'
                    )
                    var_idx += 1

                arg_idx += 1

        # Null terminate
        lines.append("*__fmtptr = 0;")

        # Wrap in function, parse, extract statements
        source_code = ' '.join(lines)
        wrapped = f"fn __format_expand__() {{ {source_code} }}"

        try:
            program = parse(wrapped, "<format>")
            if program.items and isinstance(program.items[0], ast.FunctionDecl):
                func = program.items[0]
                expanded_stmts = self._expand_statements(func.body.statements)
                for stmt in expanded_stmts:
                    self._override_source_loc(stmt, source_loc)
                return expanded_stmts
            return []
        except MacroError:
            raise
        except Exception as e:
            raise MacroError(f"error expanding format!: {e}", source_loc)

    def _parse_format_string(
        self,
        fmt: str,
        source_loc: Optional[SourceLocation]
    ) -> List[tuple]:
        """
        Parse a format string into segments of literal text and specifiers.

        Args:
            fmt: Format string contents (without surrounding quotes)
            source_loc: Source location for error reporting

        Returns:
            List of (type, data) tuples where type is 'literal' or 'specifier'
        """
        segments = []
        i = 0
        literal: List[str] = []

        while i < len(fmt):
            ch = fmt[i]

            if ch == '\\':
                # Escape sequence - keep entirely as-is for re-emission
                literal.append('\\')
                i += 1
                if i < len(fmt):
                    next_ch = fmt[i]
                    literal.append(next_ch)
                    i += 1
                    if next_ch == 'x' and i + 1 < len(fmt):
                        # \xNN - two hex digits
                        literal.append(fmt[i])
                        literal.append(fmt[i + 1])
                        i += 2

            elif ch == '{':
                if i + 1 < len(fmt) and fmt[i + 1] == '{':
                    # Escaped brace: {{ -> literal {
                    literal.append('{')
                    i += 2
                else:
                    # Start of specifier - flush accumulated literal
                    if literal:
                        segments.append(('literal', {'text': ''.join(literal)}))
                        literal = []

                    # Find closing brace
                    end = fmt.find('}', i + 1)
                    if end == -1:
                        raise MacroError(
                            "unterminated '{' in format string",
                            source_loc
                        )

                    spec_str = fmt[i + 1:end]
                    spec = self._parse_specifier(spec_str, source_loc)
                    segments.append(('specifier', spec))
                    i = end + 1

            elif ch == '}':
                if i + 1 < len(fmt) and fmt[i + 1] == '}':
                    # Escaped brace: }} -> literal }
                    literal.append('}')
                    i += 2
                else:
                    raise MacroError(
                        "unmatched '}' in format string "
                        "(use '}}' for literal '}')",
                        source_loc
                    )
            else:
                literal.append(ch)
                i += 1

        # Flush remaining literal
        if literal:
            segments.append(('literal', {'text': ''.join(literal)}))

        return segments

    def _parse_specifier(
        self,
        spec_str: str,
        source_loc: Optional[SourceLocation]
    ) -> dict:
        """
        Parse a format specifier string.

        Valid specifiers: u8, u16, u8:x, u16:x, u16:Nd (1<=N<=10), s, c

        Args:
            spec_str: Specifier string (contents between { and })
            source_loc: Source location for error reporting

        Returns:
            Dict with 'type', optional 'format', optional 'width'
        """
        import re as _re

        spec_str = spec_str.strip()

        if spec_str == 'u8':
            return {'type': 'u8', 'format': 'd'}
        elif spec_str == 'u16':
            return {'type': 'u16', 'format': 'd'}
        elif spec_str == 'u8:x':
            return {'type': 'u8', 'format': 'x'}
        elif spec_str == 'u16:x':
            return {'type': 'u16', 'format': 'x'}
        elif spec_str == 's':
            return {'type': 's'}
        elif spec_str == 'c':
            return {'type': 'c'}
        elif spec_str == 'bool':
            return {'type': 'bool'}
        elif spec_str == 'i8':
            return {'type': 'i8', 'format': 'd'}
        elif spec_str == 'i16':
            return {'type': 'i16', 'format': 'd'}
        else:
            # Check for (u8|u16):(0?)Nd pattern
            m = _re.match(r'^(u8|u16):(0?)(\d+)d$', spec_str)
            if m:
                typ = m.group(1)
                zero_pad = m.group(2) == '0'
                width = int(m.group(3))
                if width < 1 or width > 10:
                    raise MacroError(
                        f"format specifier width must be 1-10, got {width}",
                        source_loc
                    )
                result = {'type': typ, 'format': 'd', 'width': width}
                if zero_pad:
                    result['zero_pad'] = True
                return result

            raise MacroError(
                f"unknown format specifier '{{{spec_str}}}'. "
                f"Valid: {{u8}}, {{u16}}, {{i8}}, {{i16}}, {{bool}}, "
                f"{{u8:x}}, {{u16:x}}, {{u8:Nd}}, {{u16:Nd}}, "
                f"{{u8:0Nd}}, {{u16:0Nd}}, {{s}}, {{c}}",
                source_loc
            )

    def _compute_literal_byte_length(self, text: str) -> int:
        """
        Count the number of bytes a literal string will occupy after
        escape sequence processing.

        Args:
            text: Raw literal text (with escape sequences in source form)

        Returns:
            Number of bytes
        """
        count = 0
        i = 0
        while i < len(text):
            if text[i] == '\\':
                # Escape sequence = 1 byte
                i += 1
                if i < len(text):
                    if text[i] == 'x':
                        i += 2  # skip two hex digits
                    i += 1
                count += 1
            else:
                count += 1
                i += 1
        return count

    def _literal_to_bytes(self, text: str) -> list:
        """
        Convert literal text (with escape sequences) to a list of byte values.

        Args:
            text: Raw literal text (with escape sequences in source form)

        Returns:
            List of integer byte values
        """
        result = []
        i = 0
        while i < len(text):
            if text[i] == '\\':
                i += 1
                if i < len(text):
                    ch = text[i]
                    if ch == 'n':
                        result.append(0x0A)
                    elif ch == 't':
                        result.append(0x09)
                    elif ch == 'r':
                        result.append(0x0D)
                    elif ch == '0':
                        result.append(0x00)
                    elif ch == '\\':
                        result.append(0x5C)
                    elif ch == 'x':
                        if i + 2 < len(text):
                            result.append(int(text[i + 1:i + 3], 16))
                            i += 2
                    else:
                        result.append(ord(ch))
                    i += 1
            else:
                result.append(ord(text[i]))
                i += 1
        return result

    def _escape_format_literal(self, text: str) -> str:
        """
        Prepare literal text for embedding in a generated R65 string literal.

        The text comes from inside a user-provided format string, so escape
        sequences are already in their raw form. We only need to handle
        characters that could break the generated string literal (e.g., if
        {{ / }} processing introduced raw braces, which are fine in strings).

        Args:
            text: Literal text segment

        Returns:
            Text safe for embedding between double quotes in R65 source
        """
        # The text already contains properly-formed escape sequences from the
        # original R65 string literal. Braces { } from {{ }} are valid in
        # R65 strings. No additional escaping needed.
        return text

    def _check_format_buffer_overflow(
        self,
        buf: str,
        segments: List[tuple],
        source_loc: Optional[SourceLocation]
    ):
        """
        Check if the formatted output can fit in the target buffer.

        Looks up the buffer in program declarations. If it's a fixed-size
        array with a literal size, computes the maximum possible output
        length and warns if it may exceed the buffer.

        Args:
            buf: Buffer identifier name
            segments: Parsed format string segments
            source_loc: Source location for warning
        """
        # Look up the buffer declaration
        buf_size = None
        for item in self._program_items:
            if isinstance(item, ast.StaticDecl) and item.name == buf:
                if isinstance(item.var_type, ast.ArrayType):
                    if isinstance(item.var_type.size, ast.IntegerLiteral):
                        buf_size = item.var_type.size.value
                break

        if buf_size is None:
            return

        # Compute maximum output length from segments
        # max_chars tracks known maximum; has_unbounded flags {s} specifiers
        max_chars = 0
        has_unbounded = False

        for seg_type, seg_data in segments:
            if seg_type == 'literal':
                max_chars += self._compute_literal_byte_length(seg_data['text'])
            elif seg_type == 'specifier':
                spec = seg_data
                if spec['type'] == 'u8' and spec.get('format') == 'd':
                    if spec.get('width'):
                        max_chars += spec['width']
                    else:
                        max_chars += 3    # "255"
                elif spec['type'] == 'u16' and spec.get('format') == 'd':
                    if spec.get('width'):
                        max_chars += spec['width']
                    else:
                        max_chars += 5    # "65535"
                elif spec['type'] == 'u8' and spec.get('format') == 'x':
                    max_chars += 2    # "FF"
                elif spec['type'] == 'u16' and spec.get('format') == 'x':
                    max_chars += 4    # "FFFF"
                elif spec['type'] == 'i8' and spec.get('format') == 'd':
                    max_chars += 4    # "-128"
                elif spec['type'] == 'i16' and spec.get('format') == 'd':
                    max_chars += 6    # "-32768"
                elif spec['type'] == 'bool':
                    max_chars += 1    # "0" or "1"
                elif spec['type'] == 's':
                    has_unbounded = True
                elif spec['type'] == 'c':
                    max_chars += 1

        # +1 for null terminator
        total = max_chars + 1

        if total > buf_size:
            loc_str = f" at {source_loc}" if source_loc else ""
            self.warnings.append(
                f"format!{loc_str}: output may overflow buffer '{buf}' "
                f"(max {total} bytes including null terminator, "
                f"buffer is {buf_size} bytes)"
            )
        elif has_unbounded and max_chars + 1 >= buf_size:
            # Known parts alone nearly fill the buffer, and {s} adds more
            loc_str = f" at {source_loc}" if source_loc else ""
            self.warnings.append(
                f"format!{loc_str}: output may overflow buffer '{buf}' "
                f"(at least {total} bytes before {{s}} content, "
                f"buffer is {buf_size} bytes)"
            )

    def _expand_repetition(
        self,
        content: List[str],
        bindings: Dict[str, List[str]],
        separator: str
    ) -> List[str]:
        """
        Expand a repetition pattern.

        Args:
            content: Tokens inside the repetition
            bindings: Parameter bindings
            separator: Separator between repetitions ('' or ',')

        Returns:
            Expanded token list
        """
        # Find which repeated parameter is used in the content
        repeated_param = None
        for token in content:
            if token.startswith('$') and len(token) > 1:
                param_name = token[1:]
                if param_name in bindings and len(bindings[param_name]) > 1:
                    repeated_param = param_name
                    break

        if not repeated_param or repeated_param not in bindings:
            # No repeated values, just substitute once
            return self._substitute(content, bindings)

        # Expand for each value
        result: List[str] = []
        values = bindings[repeated_param]

        for idx, value in enumerate(values):
            # Create binding with just this value
            single_binding = dict(bindings)
            single_binding[repeated_param] = [value]

            # Substitute
            expanded = self._substitute(content, single_binding)

            if idx > 0 and separator:
                result.append(separator)
            result.extend(expanded)

        return result

    def _process_string_operations(self, tokens: List[str]) -> List[str]:
        """
        Process compile-time string operations in token list.

        Handles:
        - stringify!(args) -> "args"
        - symbol!(name) -> "assembler_label"
        - "string1" + "string2" -> "string1string2"

        Args:
            tokens: List of tokens

        Returns:
            Processed token list with string operations resolved
        """
        if not tokens:
            return tokens

        result = list(tokens)
        changed = True

        # Keep processing until no more changes
        while changed:
            changed = False

            # First pass: expand stringify!(...) and symbol!(...)
            i = 0
            new_result = []
            while i < len(result):
                # Look for stringify ! ( ... ) or symbol ! ( ... )
                if (i + 3 < len(result) and
                    result[i] in ('stringify', 'symbol') and
                    result[i + 1] == '!' and
                    result[i + 2] == '('):
                    is_symbol = result[i] == 'symbol'
                    # Find matching close paren
                    depth = 1
                    j = i + 3
                    args = []
                    while j < len(result) and depth > 0:
                        if result[j] == '(':
                            depth += 1
                        elif result[j] == ')':
                            depth -= 1
                            if depth == 0:
                                break
                        if depth > 0:
                            args.append(result[j])
                        j += 1

                    if is_symbol:
                        # Resolve identifier to assembler label
                        name = args[0].strip() if args else ''
                        resolved = self._resolve_assembler_symbol(name)
                        escaped = self._escape_string_literal(resolved)
                    else:
                        # Create string literal from args
                        arg_str = ' '.join(args)
                        escaped = self._escape_string_literal(arg_str)
                    new_result.append(f'"{escaped}"')
                    i = j + 1  # Skip past closing paren
                    changed = True
                else:
                    new_result.append(result[i])
                    i += 1

            result = new_result

            # Second pass: concatenate adjacent string literals with +
            i = 0
            new_result = []
            while i < len(result):
                token = result[i]

                # Check if this is a string literal
                if (token.startswith('"') and token.endswith('"') and
                    i + 2 < len(result) and
                    result[i + 1] == '+' and
                    result[i + 2].startswith('"') and result[i + 2].endswith('"')):
                    # Concatenate the strings (remove quotes, join, re-quote)
                    left_str = token[1:-1]  # Remove quotes
                    right_str = result[i + 2][1:-1]  # Remove quotes
                    concatenated = f'"{left_str}{right_str}"'
                    new_result.append(concatenated)
                    i += 3  # Skip left, +, right
                    changed = True
                else:
                    new_result.append(token)
                    i += 1

            result = new_result

        return result

    def _join_tokens(self, tokens: List[str]) -> str:
        """
        Join tokens into source code string with smart spacing.

        Avoids inserting spaces where they would break token sequences like:
        - cfg!( -> should not become 'cfg ! ('
        - name!( -> macro invocations
        - a.b -> member access
        - fn() -> function calls
        """
        if not tokens:
            return ""

        # Tokens that should not have space before them
        no_space_before = {'!', '(', ')', '[', ']', '{', '}', ',', ';', '.', '::', ':', '++', '--'}
        # Tokens that should not have space after them
        no_space_after = {'!', '(', '[', '{', '.', '::', '@', '#'}
        # Identifiers and keywords that may precede ! for macros/builtins
        macro_like = {'cfg', 'stringify', 'symbol', 'include', 'include_bytes', 'asm', 'NOP', 'compile_error', 'const_assert', '__format'}

        result = [tokens[0]]

        for i in range(1, len(tokens)):
            prev = tokens[i - 1]
            curr = tokens[i]

            # Determine if we need a space
            need_space = True

            # No space before certain tokens
            if curr in no_space_before:
                need_space = False
            # No space after certain tokens
            elif prev in no_space_after:
                need_space = False
            # Special case: identifier followed by ! (macro invocation)
            elif curr == '!' and (prev.isidentifier() or prev in macro_like):
                need_space = False
            # Special case: ! followed by ( for macro/builtin calls
            elif prev == '!' and curr == '(':
                need_space = False

            if need_space:
                result.append(' ')
            result.append(curr)

        return ''.join(result)

    def _override_source_loc(self, node: ast.ASTNode, source_loc: Optional[SourceLocation]):
        """
        Override source_loc on an AST node and its children to point to the macro invocation site.

        This ensures expanded macro code has source_loc pointing to where the macro was called,
        not where it was defined.

        Args:
            node: AST node to update
            source_loc: Source location of the macro invocation
        """
        if source_loc is None:
            return

        # Override this node's source_loc
        if hasattr(node, 'source_loc'):
            node.source_loc = source_loc

        # Recursively process children based on node type
        if isinstance(node, ast.ExprStmt):
            self._override_source_loc(node.expr, source_loc)
        elif isinstance(node, ast.LetStmt):
            if node.initializer:
                self._override_source_loc(node.initializer, source_loc)
        elif isinstance(node, ast.IfStmt):
            if node.condition:
                self._override_source_loc(node.condition, source_loc)
            if node.then_block:
                self._override_source_loc(node.then_block, source_loc)
            if node.else_block:
                self._override_source_loc(node.else_block, source_loc)
        elif isinstance(node, ast.WhileStmt):
            if node.condition:
                self._override_source_loc(node.condition, source_loc)
            if node.body:
                self._override_source_loc(node.body, source_loc)
        elif isinstance(node, ast.LoopStmt):
            if node.body:
                self._override_source_loc(node.body, source_loc)
        elif isinstance(node, ast.LoopExpression):
            if node.body:
                self._override_source_loc(node.body, source_loc)
        elif isinstance(node, ast.BreakStmt):
            if node.value:
                self._override_source_loc(node.value, source_loc)
        elif isinstance(node, ast.ForStmt):
            if node.start:
                self._override_source_loc(node.start, source_loc)
            if node.end:
                self._override_source_loc(node.end, source_loc)
            if node.body:
                self._override_source_loc(node.body, source_loc)
        elif isinstance(node, ast.Block):
            for stmt in node.statements:
                self._override_source_loc(stmt, source_loc)
        elif isinstance(node, ast.BinaryOp):
            self._override_source_loc(node.left, source_loc)
            self._override_source_loc(node.right, source_loc)
        elif isinstance(node, ast.UnaryOp):
            self._override_source_loc(node.operand, source_loc)
        elif isinstance(node, ast.Assignment):
            self._override_source_loc(node.target, source_loc)
            self._override_source_loc(node.value, source_loc)
        elif isinstance(node, ast.CompoundAssignment):
            self._override_source_loc(node.target, source_loc)
            self._override_source_loc(node.value, source_loc)
        elif isinstance(node, ast.FunctionCall):
            if node.func:
                self._override_source_loc(node.func, source_loc)
            for arg in node.args:
                self._override_source_loc(arg, source_loc)
        elif isinstance(node, ast.ArrayIndex):
            self._override_source_loc(node.array, source_loc)
            self._override_source_loc(node.index, source_loc)
        elif isinstance(node, ast.FieldAccess):
            self._override_source_loc(node.base, source_loc)
        elif isinstance(node, ast.Dereference):
            self._override_source_loc(node.pointer, source_loc)


def expand_macros(program: ast.Program) -> ast.Program:
    """
    Convenience function to expand all macros in a program.

    Args:
        program: The parsed program AST

    Returns:
        A new program with macros expanded
    """
    import sys as _sys
    expander = MacroExpander()
    result = expander.expand(program)
    for warning in expander.warnings:
        print(f"warning: {warning}", file=_sys.stderr)
    return result
