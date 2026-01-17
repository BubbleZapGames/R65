"""
Macro expander for R65.

Handles macro definition collection and invocation expansion.
Works at the AST level, expanding macro invocations into AST nodes.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Union
from copy import deepcopy

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

    def __init__(self):
        self.macros: Dict[str, MacroDefinition] = {}
        self._expansion_depth = 0
        self._expanding: Set[str] = set()  # Track currently expanding macros

    def expand(self, program: ast.Program) -> ast.Program:
        """
        Expand all macros in a program.

        Args:
            program: The parsed program AST

        Returns:
            A new program with macros expanded
        """
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
            source_loc=func.source_loc
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
                    var_name=stmt.var_name,
                    start=self._expand_expression(stmt.start),
                    end=self._expand_expression(stmt.end),
                    body=self._expand_block(stmt.body),
                    label=getattr(stmt, 'label', None),
                    source_loc=stmt.source_loc
                )
                result.append(new_stmt)
            else:
                # Keep other statements as-is
                result.append(stmt)

        return result

    def _expand_expression(self, expr: ast.Expression) -> ast.Expression:
        """Recursively expand macro invocations in an expression."""
        if expr is None:
            return None

        if isinstance(expr, ast.MacroInvocation):
            # Handle stringify! specially
            if expr.name == "stringify":
                return self._expand_stringify_expr(expr.args, expr.source_loc)
            # Expand user-defined macro to expression
            return self._expand_expression_invocation(expr.name, expr.args, expr.source_loc)

        elif isinstance(expr, ast.BinaryOp):
            return ast.BinaryOp(
                op=expr.op,
                left=self._expand_expression(expr.left),
                right=self._expand_expression(expr.right),
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
        """
        Expand a macro invocation in expression context.

        Args:
            name: Macro name
            args: List of argument token strings
            source_loc: Source location of invocation

        Returns:
            Expanded expression
        """
        # Check if macro exists
        if name not in self.macros:
            raise MacroError(f"undefined macro: '{name}'", source_loc)

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

            # Parse the expanded tokens as an expression
            expanded_source = self._join_tokens(expanded_tokens)

            try:
                # Wrap in a dummy function with expression statement
                wrapped = f"fn __macro_expr__() {{ let __result__: u16 = {expanded_source}; }}"
                program = parse(wrapped, f"<macro:{name}>")

                # Extract the expression from the let statement
                if program.items and isinstance(program.items[0], ast.FunctionDecl):
                    func = program.items[0]
                    if func.body.statements:
                        stmt = func.body.statements[0]
                        if isinstance(stmt, ast.LetStmt) and stmt.initializer:
                            # Recursively expand any nested macro invocations
                            return self._expand_expression(stmt.initializer)

                raise MacroError(
                    f"macro '{name}' did not expand to a valid expression",
                    source_loc
                )
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

    def _expand_top_level_invocation(
        self,
        name: str,
        args: List[str],
        source_loc: Optional[SourceLocation]
    ) -> List[ast.Declaration]:
        """
        Expand a top-level macro invocation into declarations.

        Unlike statement-level expansion which wraps in a dummy function,
        top-level expansion parses the result as a complete program to
        extract declarations (functions, statics, structs, etc.).

        Args:
            name: Macro name
            args: List of argument token strings
            source_loc: Source location of invocation

        Returns:
            List of expanded declarations
        """
        # Check if macro exists
        if name not in self.macros:
            raise MacroError(f"undefined macro: '{name}'", source_loc)

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

            # Parse the expanded tokens as a complete program
            expanded_source = self._join_tokens(expanded_tokens)

            try:
                # Parse directly as a program (not wrapped in a function)
                program = parse(expanded_source, f"<macro:{name}>")

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
            except Exception as e:
                raise MacroError(
                    f"error parsing expanded macro '{name}': {e}\n"
                    f"Expanded source: {expanded_source}",
                    source_loc
                )
        finally:
            self._expanding.remove(name)
            self._expansion_depth -= 1

    def _expand_statement_invocation(
        self,
        name: str,
        args: List[str],
        source_loc: Optional[SourceLocation]
    ) -> List[ast.Statement]:
        """
        Expand a statement-level macro invocation.

        Args:
            name: Macro name
            args: List of argument token strings
            source_loc: Source location of invocation

        Returns:
            List of expanded statements
        """
        # Handle built-in stringify! macro
        if name == "stringify":
            return self._expand_stringify(args, source_loc)

        # Check if macro exists
        if name not in self.macros:
            raise MacroError(f"undefined macro: '{name}'", source_loc)

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

            # Parse the expanded tokens as statements
            expanded_source = self._join_tokens(expanded_tokens)

            try:
                # Wrap in a dummy function to parse as statements
                wrapped = f"fn __macro_expand__() {{ {expanded_source} }}"
                program = parse(wrapped, f"<macro:{name}>")

                # Extract the statements from the function body
                if program.items and isinstance(program.items[0], ast.FunctionDecl):
                    func = program.items[0]
                    # Recursively expand any nested macro invocations
                    expanded_stmts = self._expand_statements(func.body.statements)
                    return expanded_stmts
                return []
            except Exception as e:
                raise MacroError(
                    f"error parsing expanded macro '{name}': {e}\n"
                    f"Expanded source: {expanded_source}",
                    source_loc
                )
        finally:
            self._expanding.remove(name)
            self._expansion_depth -= 1

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
            # All args go to the repeated parameter
            if len(repeated_params) > 1:
                raise MacroError(
                    f"macro '{macro.name}' has multiple repeated parameters (not supported)",
                    source_loc
                )
            # Bind all args to the repeated param
            bindings[repeated_params[0].name] = args
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
        macro_like = {'cfg', 'stringify', 'include', 'include_bytes', 'asm', 'NOP'}

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


def expand_macros(program: ast.Program) -> ast.Program:
    """
    Convenience function to expand all macros in a program.

    Args:
        program: The parsed program AST

    Returns:
        A new program with macros expanded
    """
    expander = MacroExpander()
    return expander.expand(program)
