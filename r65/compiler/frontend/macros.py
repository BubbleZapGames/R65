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
from r65.compiler.hir.errors import SourceLocation


class MacroError(Exception):
    """Error during macro expansion."""
    def __init__(self, message: str, source_loc: Optional[SourceLocation] = None):
        self.message = message
        self.source_loc = source_loc
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        if self.source_loc:
            return f"{self.source_loc}: {self.message}"
        return self.message


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
                    source_loc=stmt.source_loc
                )
                result.append(new_stmt)
            elif isinstance(stmt, ast.WhileStmt):
                new_stmt = ast.WhileStmt(
                    condition=stmt.condition,
                    body=self._expand_block(stmt.body),
                    source_loc=stmt.source_loc
                )
                result.append(new_stmt)
            elif isinstance(stmt, ast.Block):
                new_stmt = self._expand_block(stmt)
                result.append(new_stmt)
            else:
                # Keep other statements as-is
                result.append(stmt)

        return result

    def _expand_if(self, stmt: ast.IfStmt) -> ast.IfStmt:
        """Expand macros inside an if statement."""
        new_then = self._expand_block(stmt.then_block)
        new_else = None
        if stmt.else_block:
            if isinstance(stmt.else_block, ast.IfStmt):
                new_else = self._expand_if(stmt.else_block)
            else:
                new_else = self._expand_block(stmt.else_block)

        return ast.IfStmt(
            condition=stmt.condition,
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
        """Expand a top-level macro invocation into declarations."""
        # For now, we don't support top-level macros that expand to statements
        # This would require different handling
        raise MacroError(
            f"top-level macro invocations not yet supported: '{name}'",
            source_loc
        )

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
            expanded_source = ' '.join(expanded_tokens)

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
