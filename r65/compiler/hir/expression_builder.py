# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Expression builder - builds HIR expressions from AST expressions.

Extracted from HIRBuilder to improve modularity and reduce file size.
"""

from typing import Optional
from pathlib import Path
from r65.compiler.frontend import ast

from r65.compiler.hir import nodes as hir
from r65.compiler.hir.symbol_table import Symbol, SymbolKind, ScopeKind
from r65.compiler.hir.types import BasicTypeInfo, ArrayTypeInfo
from r65.compiler.hir.errors import HIRError


class ExpressionBuilder:
    """Builds HIR expressions from AST expression nodes."""

    def __init__(self, symbol_table, const_evaluator, type_resolver,
                 cfg_evaluator, source_dir, include_paths):
        """
        Initialize expression builder.

        Args:
            symbol_table: Symbol table for name resolution
            const_evaluator: Const evaluator for compile-time evaluation
            type_resolver: Type resolver for type lookups
            cfg_evaluator: Cfg evaluator for conditional compilation
            source_dir: Source directory for resolving include paths
            include_paths: Additional include search directories
        """
        self.symbol_table = symbol_table
        self.const_evaluator = const_evaluator
        self.type_resolver = type_resolver
        self.cfg_evaluator = cfg_evaluator
        self.source_dir = source_dir
        self.include_paths = include_paths
        # Callback to build statements (set by HIRBuilder after construction)
        self.statement_builder = None

    def build_expression(self, expr: ast.Expression) -> hir.HIRExpression:
        """Build HIR expression from AST."""
        # Get source location from AST node
        src_loc = expr.source_loc

        if isinstance(expr, ast.IntegerLiteral):
            return hir.HIRIntegerLiteral(value=expr.value, suffix=expr.suffix, source_loc=src_loc)

        elif isinstance(expr, ast.BooleanLiteral):
            return hir.HIRBooleanLiteral(value=expr.value, source_loc=src_loc)

        elif isinstance(expr, ast.StringLiteral):
            return hir.HIRStringLiteral(value=expr.value, source_loc=src_loc)

        elif isinstance(expr, ast.Identifier):
            # Resolve identifier
            symbol = self.symbol_table.lookup(expr.name)
            if not symbol:
                raise HIRError(f"Undefined identifier: {expr.name}", source_loc=src_loc)
            return hir.HIRIdentifier(name=expr.name, symbol=symbol, source_loc=src_loc)

        elif isinstance(expr, ast.Register):
            # Resolve register
            symbol = self.symbol_table.lookup(expr.name)
            return hir.HIRRegister(name=expr.name, symbol=symbol, source_loc=src_loc)

        elif isinstance(expr, ast.IncludeBytesExpr):
            # Include binary data from file
            # Validate that the file exists and get file info
            resolved_path, file_size = self._validate_include_bytes_path(expr.path, expr.source_loc)
            return hir.HIRIncludeBytesExpr(
                path=expr.path,
                resolved_path=resolved_path,
                size=file_size,
                source_loc=src_loc
            )

        elif isinstance(expr, ast.ArrayFillExpr):
            # Array fill expression: [value; count]
            fill_value = self.build_expression(expr.value)
            # Count must be a constant - evaluate at compile time
            count = self.const_evaluator.eval(expr.count)
            return hir.HIRArrayFillExpr(fill_value=fill_value, count=count, source_loc=src_loc)

        elif isinstance(expr, ast.ArrayLiteralExpr):
            # Array literal expression: [a, b, c, ...]
            elements = [self.build_expression(e) for e in expr.elements]
            return hir.HIRArrayLiteralExpr(elements=elements, source_loc=src_loc)

        elif isinstance(expr, ast.StructLiteralExpr):
            # Struct literal expression: Player { x: 10, y: 20, health: 100 }
            return self._build_struct_literal(expr)

        elif isinstance(expr, ast.EnumVariantExpr):
            # Resolve enum variant or associated constant: Name::Variant / Name::CONST
            # First try qualified name lookup (covers both enum variants and impl consts)
            qualified_name = f"{expr.enum_name}::{expr.variant_name}"
            qualified_symbol = self.symbol_table.lookup(qualified_name)

            if qualified_symbol and qualified_symbol.kind == SymbolKind.IMPL_CONST:
                # Associated constant (e.g., Player::TYPE_ID, Player::WIDTH)
                return hir.HIRIntegerLiteral(
                    value=qualified_symbol.const_value,
                    source_loc=src_loc
                )

            # Otherwise resolve as enum variant
            enum_symbol = self.symbol_table.lookup(expr.enum_name)
            if not enum_symbol:
                raise HIRError(f"Undefined type: {expr.enum_name}", source_loc=src_loc)
            if enum_symbol.kind != SymbolKind.ENUM:
                raise HIRError(f"{expr.enum_name} is not an enum", source_loc=src_loc)

            if not qualified_symbol:
                raise HIRError(f"Undefined enum variant: {qualified_name}", source_loc=src_loc)
            if qualified_symbol.kind != SymbolKind.ENUM_VARIANT:
                raise HIRError(f"{qualified_name} is not an enum variant", source_loc=src_loc)

            variant_value = qualified_symbol.const_value
            return hir.HIREnumVariantExpr(
                enum_name=expr.enum_name,
                variant_name=expr.variant_name,
                value=variant_value,
                source_loc=src_loc
            )

        elif isinstance(expr, ast.BinaryOp):
            # Check for string concatenation
            if expr.op == '+':
                left_is_string = isinstance(expr.left, ast.StringLiteral)
                right_is_string = isinstance(expr.right, ast.StringLiteral)

                if left_is_string or right_is_string:
                    try:
                        # Use const evaluator on the AST directly
                        const_value = self.const_evaluator.eval(expr)
                        if isinstance(const_value, str):
                            return hir.HIRStringLiteral(value=const_value, source_loc=src_loc)
                    except HIRError:
                        # If constant evaluation fails, fall back to normal processing
                        pass

            left = self.build_expression(expr.left)
            right = self.build_expression(expr.right)
            return hir.HIRBinaryOp(op=expr.op, left=left, right=right, source_loc=src_loc)

        elif isinstance(expr, ast.UnaryOp):
            operand = self.build_expression(expr.operand)
            return hir.HIRUnaryOp(op=expr.op, operand=operand, source_loc=src_loc)

        elif isinstance(expr, ast.TypeCast):
            inner = self.build_expression(expr.expr)
            target = self.type_resolver.resolve_type(expr.target_type)
            return hir.HIRTypeCast(expr=inner, target_type=target, source_loc=src_loc)

        elif isinstance(expr, ast.NewtypeFieldAccess):
            # `t.0` — a retype of the operand. The target type is unknown until
            # the operand is typed, so the checker fills it in.
            inner = self.build_expression(expr.base)
            return hir.HIRTypeCast(expr=inner, target_type=None,
                                   newtype_field=expr.index, source_loc=src_loc)

        elif isinstance(expr, ast.FunctionCall):
            from r65.compiler.builtins import BuiltinRegistry

            # Associated function: `Q10::from_int(5)`. Parses as a call on a
            # `Name::member` path, and resolves to the same mangled symbol an
            # `impl` method gets — it is an ordinary function namespaced by the
            # type, with no receiver.
            if isinstance(expr.func, ast.EnumVariantExpr):
                mangled = f"{expr.func.enum_name}__{expr.func.variant_name}"
                assoc = self.symbol_table.lookup(mangled)
                if assoc is not None and assoc.kind == SymbolKind.METHOD:
                    func = hir.HIRIdentifier(name=mangled, symbol=assoc,
                                             source_loc=expr.func.source_loc)
                    args = [self.build_expression(a) for a in expr.args]
                    return hir.HIRFunctionCall(func=func, args=args, source_loc=src_loc)

            # Newtype construction: `TileId(x)` is a retype, not a call, so it
            # desugars to the same node an `as` cast uses and MIR never sees it.
            # It is *checked* more strictly than a cast — see `newtype_construct`.
            if isinstance(expr.func, ast.Identifier):
                sym = self.symbol_table.lookup(expr.func.name)
                if sym and sym.kind == SymbolKind.NEWTYPE:
                    if len(expr.args) != 1:
                        raise HIRError(
                            f"newtype '{expr.func.name}' takes exactly 1 value, "
                            f"got {len(expr.args)}",
                            source_loc=src_loc,
                            hint=f"write '{expr.func.name}(value)'")
                    inner = self.build_expression(expr.args[0])
                    target = self.type_resolver.resolve_named_type(expr.func.name)
                    return hir.HIRTypeCast(expr=inner, target_type=target,
                                           newtype_construct=True,
                                           source_loc=src_loc)

            # Check if this is a method call (e.g., value.rotate_left(3) or array.len())
            if isinstance(expr.func, ast.FieldAccess):
                method_name = expr.func.field
                if method_name in ['rotate_left', 'rotate_right']:
                    # This is a rotate method call
                    # Validate: must have exactly 1 argument
                    if len(expr.args) != 1:
                        raise HIRError(f"{method_name}() takes exactly 1 argument, got {len(expr.args)}", source_loc=src_loc)

                    # Build the base expression (the value being rotated)
                    base = self.build_expression(expr.func.base)

                    # Build the argument (rotation count)
                    count_arg = self.build_expression(expr.args[0])

                    # Return a special HIRMethodCall node for rotate methods
                    return hir.HIRMethodCall(
                        receiver=base,
                        method_name=method_name,
                        args=[count_arg],
                        source_loc=src_loc
                    )
                elif method_name == 'len':
                    # This is a len() method call on an array
                    # Validate: must have no arguments
                    if len(expr.args) != 0:
                        raise HIRError(f"len() takes no arguments, got {len(expr.args)}", source_loc=src_loc)

                    # Build the base expression (the array)
                    base = self.build_expression(expr.func.base)

                    # Try to const-evaluate if base is an identifier with known array type
                    if isinstance(expr.func.base, ast.Identifier):
                        symbol = self.symbol_table.lookup(expr.func.base.name)
                        if symbol and symbol.var_type and isinstance(symbol.var_type, ArrayTypeInfo):
                            # We know the array size at compile time
                            # len() returns u16 to hold array lengths up to 65535
                            result = hir.HIRIntegerLiteral(value=symbol.var_type.size, source_loc=src_loc)
                            result.expr_type = BasicTypeInfo('u16')
                            return result

                    # Return a special HIRMethodCall node for len method
                    # Type checker will validate receiver is an array
                    return hir.HIRMethodCall(
                        receiver=base,
                        method_name=method_name,
                        args=[],
                        source_loc=src_loc
                    )
                elif method_name == 'bank_byte':
                    # bank_byte() method on far pointers — extracts bank byte (byte 2)
                    if len(expr.args) != 0:
                        raise HIRError(f"bank_byte() takes no arguments, got {len(expr.args)}", source_loc=src_loc)

                    # Try const evaluation first
                    try:
                        value = self.const_evaluator.eval_method_call(expr.func.base, 'bank_byte')
                        result = hir.HIRIntegerLiteral(value=value, source_loc=src_loc)
                        result.expr_type = BasicTypeInfo('u8')
                        return result
                    except HIRError:
                        pass  # Not const-evaluable, fall through to runtime

                    base = self.build_expression(expr.func.base)
                    return hir.HIRMethodCall(
                        receiver=base,
                        method_name='bank_byte',
                        args=[],
                        source_loc=src_loc
                    )

            # Check if this is a built-in function call BEFORE trying to build func expression
            # This prevents "undefined identifier" errors for built-in function names
            builtin_name = None
            if isinstance(expr.func, ast.Identifier):
                func_name = expr.func.name
                if BuiltinRegistry.is_builtin(func_name):
                    builtin = BuiltinRegistry.get_builtin(func_name)

                    # Special handling for type_info builtins - const evaluate at HIR build time
                    if builtin and builtin.kind.value == "type_info" and func_name in ("size_of", "offset_of"):
                        try:
                            # Try to evaluate at compile time using const evaluator
                            const_value = self.const_evaluator.eval(expr)
                            if isinstance(const_value, int):
                                return hir.HIRIntegerLiteral(value=const_value, source_loc=src_loc)
                        except HIRError:
                            if func_name == "offset_of":
                                raise  # offset_of is const-only, propagate errors
                            # size_of: if const evaluation fails, fall back to runtime call
                            pass

                    # Const math builtins - try to fold if all args are const
                    if builtin and builtin.kind.value == "const_math":
                        try:
                            const_value = self.const_evaluator.eval(expr)
                            if isinstance(const_value, int):
                                return hir.HIRIntegerLiteral(value=const_value, source_loc=src_loc)
                        except HIRError:
                            pass  # Args may be runtime vars inside a const fn

                    # Mark this as a built-in function call
                    builtin_name = func_name
                elif func_name == '__fmt_str':
                    # Internal format! string-segment dispatch — type checker
                    # rewrites this to strcpy() or .to_string() based on arg type.
                    builtin_name = func_name

            # Try to const-fold calls to const fn with all-const arguments
            if isinstance(expr.func, ast.Identifier) and not builtin_name:
                symbol = self.symbol_table.lookup(expr.func.name)
                if (symbol and symbol.kind == SymbolKind.FUNCTION and
                        hasattr(symbol.definition, 'is_const') and symbol.definition.is_const):
                    try:
                        const_value = self.const_evaluator.eval(expr)
                        if isinstance(const_value, bool):
                            return hir.HIRBooleanLiteral(value=const_value, source_loc=src_loc)
                        elif isinstance(const_value, int):
                            return hir.HIRIntegerLiteral(value=const_value, source_loc=src_loc)
                        elif isinstance(const_value, list):
                            elements = [
                                hir.HIRIntegerLiteral(value=v, source_loc=src_loc)
                                for v in const_value
                            ]
                            return hir.HIRArrayLiteralExpr(elements=elements, source_loc=src_loc)
                    except HIRError:
                        pass  # Args not all const — fall through to runtime call

            # Build func expression
            # For built-ins, create a dummy symbol to avoid "undefined identifier" errors
            if builtin_name:
                # Create a dummy symbol for built-in function
                builtin_symbol = Symbol(
                    name=expr.func.name,
                    kind=SymbolKind.FUNCTION,
                    definition=None,
                    scope_id=0,  # Global scope
                    var_type=None
                )
                func = hir.HIRIdentifier(name=expr.func.name, symbol=builtin_symbol, source_loc=src_loc)
            else:
                func = self.build_expression(expr.func)

            args = [self.build_expression(a) for a in expr.args]

            return hir.HIRFunctionCall(func=func, args=args, builtin_name=builtin_name, source_loc=src_loc)

        elif isinstance(expr, ast.CfgFunctionCall):
            # cfg!(flag) evaluates to true/false based on compiler cfg options
            condition_name = None
            if isinstance(expr.condition, ast.Identifier):
                condition_name = expr.condition.name
            elif hasattr(expr.condition, 'value'):
                condition_name = str(expr.condition.value)

            # Check if this cfg flag is set
            if self.cfg_evaluator and condition_name:
                cond = ast.CfgIdentifier(name=condition_name)
                result = self.cfg_evaluator.evaluate(cond)
            else:
                result = False

            return hir.HIRBooleanLiteral(value=result, source_loc=src_loc)

        elif isinstance(expr, ast.ArrayIndex):
            # Try const-folding: CONST_ARRAY[0] → literal
            try:
                const_value = self.const_evaluator.eval(expr)
                if isinstance(const_value, bool):
                    return hir.HIRBooleanLiteral(value=const_value, source_loc=src_loc)
                elif isinstance(const_value, int):
                    return hir.HIRIntegerLiteral(value=const_value, source_loc=src_loc)
                # dict/list results: fall through (wrapping FieldAccess will resolve)
            except HIRError:
                pass

            array = self.build_expression(expr.array)
            index = self.build_expression(expr.index)
            return hir.HIRArrayIndex(array=array, index=index, original_ast=expr, source_loc=src_loc)

        elif isinstance(expr, ast.FieldAccess):
            # Try const-folding: CONST_STRUCT.field → literal
            try:
                const_value = self.const_evaluator.eval(expr)
                if isinstance(const_value, bool):
                    return hir.HIRBooleanLiteral(value=const_value, source_loc=src_loc)
                elif isinstance(const_value, int):
                    return hir.HIRIntegerLiteral(value=const_value, source_loc=src_loc)
                # dict/list results: fall through to normal handling
            except HIRError:
                pass

            base = self.build_expression(expr.base)

            # Check for STATUS.Flag pattern
            if isinstance(base, hir.HIRRegister) and base.name == 'STATUS':
                from r65.compiler.hir.status_flags import get_status_flag, get_all_flag_names
                flag = get_status_flag(expr.field)
                if flag:
                    return hir.HIRStatusFlagAccess(
                        flag_name=flag.name,
                        bit_position=flag.bit_position,
                        bit_mask=flag.bit_mask,
                        source_loc=src_loc
                    )
                else:
                    valid_flags = ', '.join(get_all_flag_names())
                    raise HIRError(
                        f"Unknown STATUS flag '{expr.field}'. "
                        f"Valid flags: {valid_flags}",
                        source_loc=src_loc
                    )

            # Normal field resolution happens in type checker
            return hir.HIRFieldAccess(base=base, field_name=expr.field, source_loc=src_loc)

        elif isinstance(expr, ast.Dereference):
            pointer = self.build_expression(expr.pointer)
            return hir.HIRDereference(pointer=pointer, source_loc=src_loc)

        elif isinstance(expr, ast.AddressOf):
            operand = self.build_expression(expr.operand)
            return hir.HIRAddressOf(operand=operand, source_loc=src_loc)

        elif isinstance(expr, ast.Assignment):
            target = self.build_expression(expr.target)
            value = self.build_expression(expr.value)
            return hir.HIRAssignment(target=target, value=value, source_loc=src_loc)

        elif isinstance(expr, ast.CompoundAssignment):
            # Desugar compound assignment: x += 5 becomes x = x + 5
            target = self.build_expression(expr.target)
            value = self.build_expression(expr.value)

            # Create binary operation: target op value
            binary_op = hir.HIRBinaryOp(
                op=expr.operator,
                left=target,  # Read from target
                right=value,
                source_loc=src_loc
            )

            # Create assignment: target = (target op value)
            # Tag with the base operator so the type checker can redirect an
            # aggregate compound-assign (a += b) to an operator-trait method
            # (a.add_assign(&b)) instead of the by-value primitive path.
            assignment = hir.HIRAssignment(target=target, value=binary_op, source_loc=src_loc)
            assignment.compound_op = expr.operator
            return assignment

        elif isinstance(expr, ast.MultiAssignment):
            # Multiple assignment: lo, hi = func()
            targets = [self.build_expression(t) for t in expr.targets]
            value = self.build_expression(expr.value)
            return hir.HIRMultiAssignment(targets=targets, value=value, source_loc=src_loc)

        elif isinstance(expr, ast.MatchExpression):
            # Build match expression
            return self._build_match_expression(expr)

        elif isinstance(expr, ast.BlockExpression):
            return self._build_block_expression(expr)

        elif isinstance(expr, ast.IfExpression):
            return self._build_if_expression(expr)

        elif isinstance(expr, ast.LoopExpression):
            return self._build_loop_expression(expr)

        else:
            raise HIRError(f"Unknown expression type: {type(expr).__name__}", source_loc=getattr(expr, 'source_loc', None))

    def _build_match_expression(self, expr: ast.MatchExpression) -> hir.HIRMatchExpression:
        """Build HIR match expression from AST."""
        # Build scrutinee
        scrutinee = self.build_expression(expr.scrutinee)

        # Build each arm
        arms = []
        for ast_arm in expr.arms:
            # Enter new scope for pattern bindings
            scope_id = self.symbol_table.enter_scope(ScopeKind.BLOCK)

            # Build pattern (may create bindings in scope)
            pattern = self._build_pattern(ast_arm.pattern)

            # Build body expression or statement (return/break/continue)
            if isinstance(ast_arm.body, (ast.ReturnStmt, ast.BreakStmt, ast.ContinueStmt, ast.Block)):
                body = self.statement_builder(ast_arm.body)
            else:
                body = self.build_expression(ast_arm.body)

            # Exit scope
            self.symbol_table.exit_scope()

            arms.append(hir.HIRMatchArm(pattern=pattern, body=body, scope_id=scope_id))

        return hir.HIRMatchExpression(scrutinee=scrutinee, arms=arms, source_loc=expr.source_loc)

    def _build_pattern(self, pattern: ast.Pattern) -> hir.HIRPattern:
        """Build HIR pattern from AST pattern."""
        if isinstance(pattern, ast.LiteralPattern):
            return hir.HIRLiteralPattern(value=pattern.value)

        elif isinstance(pattern, ast.EnumPattern):
            # Resolve enum variant value
            enum_symbol = self.symbol_table.lookup(pattern.enum_name)
            if not enum_symbol or enum_symbol.kind != SymbolKind.ENUM:
                raise HIRError(f"Undefined enum: {pattern.enum_name}", source_loc=getattr(pattern, 'source_loc', None))

            qualified_name = f"{pattern.enum_name}::{pattern.variant_name}"
            variant_symbol = self.symbol_table.lookup(qualified_name)
            if not variant_symbol or variant_symbol.kind != SymbolKind.ENUM_VARIANT:
                raise HIRError(f"Undefined enum variant: {qualified_name}", source_loc=getattr(pattern, 'source_loc', None))

            variant_value = variant_symbol.const_value
            return hir.HIREnumPattern(
                enum_name=pattern.enum_name,
                variant_name=pattern.variant_name,
                variant_value=variant_value
            )

        elif isinstance(pattern, ast.WildcardPattern):
            return hir.HIRWildcardPattern()

        elif isinstance(pattern, ast.IdentifierPattern):
            # Check if identifier refers to a constant
            existing = self.symbol_table.lookup(pattern.name)
            if existing and existing.kind == SymbolKind.CONST and existing.const_value is not None:
                # Resolve constant to literal pattern
                return hir.HIRLiteralPattern(value=existing.const_value)

            # Not a constant — create a new binding in current scope
            symbol = Symbol(
                name=pattern.name,
                kind=SymbolKind.LOCAL_VAR,
                definition=None,  # Pattern bindings don't have a separate definition
                scope_id=self.symbol_table.current_scope_id,
                var_type=None  # Will be set during type checking
            )
            self.symbol_table.declare(pattern.name, symbol)
            return hir.HIRIdentifierPattern(name=pattern.name, symbol=symbol)

        elif isinstance(pattern, ast.RangePattern):
            return hir.HIRRangePattern(
                start=pattern.start,
                end=pattern.end,
                inclusive=pattern.inclusive
            )

        elif isinstance(pattern, ast.OrPattern):
            patterns = [self._build_pattern(p) for p in pattern.patterns]
            return hir.HIROrPattern(patterns=patterns)

        else:
            raise HIRError(f"Unknown pattern type: {type(pattern).__name__}", source_loc=getattr(pattern, 'source_loc', None))

    def _build_struct_literal(self, expr: ast.StructLiteralExpr) -> hir.HIRStructLiteralExpr:
        """Build HIR struct literal from AST."""
        from r65.compiler.hir.unified_type_utils import get_unified_type_size

        # Lookup struct definition
        struct_symbol = self.symbol_table.lookup(expr.struct_name)
        if not struct_symbol:
            raise HIRError(f"Undefined struct: {expr.struct_name}", source_loc=expr.source_loc)
        if struct_symbol.kind != SymbolKind.STRUCT:
            raise HIRError(f"{expr.struct_name} is not a struct", source_loc=expr.source_loc)

        # Get the struct declaration to access field information
        struct_decl = struct_symbol.definition

        # Build field offsets mapping
        field_offsets = {}
        field_types = {}
        if isinstance(struct_decl, hir.HIRStructDecl):
            # Already built
            for field in struct_decl.fields:
                field_offsets[field.name] = field.offset
                field_types[field.name] = field.field_type
        elif isinstance(struct_decl, ast.StructDecl):
            # Need to calculate offsets
            from r65.compiler.hir.unified_type_utils import layout_fields
            resolved = [self.type_resolver.resolve_type(f.field_type) for f in struct_decl.fields]
            offsets, _ = layout_fields(
                [get_unified_type_size(t, self.symbol_table) for t in resolved],
                struct_decl.is_union
            )
            for field, field_type, offset in zip(struct_decl.fields, resolved, offsets):
                field_offsets[field.name] = offset
                field_types[field.name] = field_type

        # Build field initializers
        hir_fields = []
        for field_init in expr.fields:
            if field_init.name not in field_offsets:
                raise HIRError(f"Unknown field: {expr.struct_name}.{field_init.name}", source_loc=expr.source_loc)

            value_expr = self.build_expression(field_init.value)
            hir_field = hir.HIRStructFieldInit(
                name=field_init.name,
                value=value_expr,
                field_offset=field_offsets[field_init.name]
            )
            hir_fields.append(hir_field)

        # Create HIR struct literal
        # Note: struct_decl may be AST node (pass 1 not complete) or HIR node
        # Type checker will resolve this properly
        return hir.HIRStructLiteralExpr(
            struct_name=expr.struct_name,
            struct_decl=struct_decl if isinstance(struct_decl, hir.HIRStructDecl) else None,
            fields=hir_fields
        )

    def _build_block_expression(self, expr: ast.BlockExpression) -> hir.HIRBlockExpression:
        """Build HIR block expression from AST.

        Enters a new scope for the block, builds all statements,
        then builds the final expression.
        """
        scope_id = self.symbol_table.enter_scope(ScopeKind.BLOCK)

        # Build statements
        hir_stmts = []
        for stmt in expr.statements:
            if self.statement_builder is None:
                raise HIRError("Statement builder not configured for block expressions", source_loc=expr.source_loc)
            hir_stmt = self.statement_builder(stmt)
            hir_stmts.append(hir_stmt)

        # Build final expression (None for diverging blocks like { return 1; })
        final_expr = self.build_expression(expr.final_expr) if expr.final_expr is not None else None

        self.symbol_table.exit_scope()

        return hir.HIRBlockExpression(
            statements=hir_stmts,
            final_expr=final_expr,
            scope_id=scope_id,
            source_loc=expr.source_loc
        )

    def _build_if_expression(self, expr: ast.IfExpression) -> hir.HIRIfExpression:
        """Build HIR if expression from AST.

        Requires both then and else branches (parser validates this).
        """
        condition = self.build_expression(expr.condition)
        then_block = self._build_block_expression(expr.then_block)

        if isinstance(expr.else_block, ast.IfExpression):
            else_block = self._build_if_expression(expr.else_block)
        else:
            else_block = self._build_block_expression(expr.else_block)

        return hir.HIRIfExpression(
            condition=condition,
            then_block=then_block,
            else_block=else_block,
            source_loc=expr.source_loc
        )

    def _build_loop_expression(self, expr: ast.LoopExpression) -> hir.HIRLoopExpression:
        """Build HIR loop expression from AST.

        The body is a block that should contain break statements with values.
        """
        # Build body statements
        hir_stmts = []
        for stmt in expr.body.statements:
            if self.statement_builder is None:
                raise HIRError("Statement builder not configured for loop expressions", source_loc=expr.source_loc)
            hir_stmt = self.statement_builder(stmt)
            hir_stmts.append(hir_stmt)

        body = hir.HIRBlock(
            statements=hir_stmts,
            source_loc=expr.body.source_loc
        )

        return hir.HIRLoopExpression(
            body=body,
            label=expr.label,
            source_loc=expr.source_loc
        )

    def _resolve_include_bytes_path(self, path: str) -> Optional[Path]:
        """
        Resolve an include_bytes! path by searching source directory and include paths.

        Search order:
        1. Relative to the source file's directory
        2. Each directory in include_paths (-I options)

        Args:
            path: The path from the include_bytes! expression

        Returns:
            Resolved absolute path if found, None otherwise
        """
        # First try relative to the source file
        candidate = (self.source_dir / path).resolve()
        if candidate.exists() and candidate.is_file():
            return candidate

        # Then search include paths
        for inc_dir in self.include_paths:
            candidate = (inc_dir / path).resolve()
            if candidate.exists() and candidate.is_file():
                return candidate

        return None

    def _validate_include_bytes_path(self, path: str, source_loc) -> tuple:
        """
        Validate that the file path for include_bytes! exists and return its info.

        Args:
            path: The file path from the include_bytes! expression
            source_loc: Source location for error reporting

        Returns:
            Tuple of (resolved_path, file_size)

        Raises:
            HIRError: If the file does not exist
        """
        # Resolve path (searches source_dir and include paths)
        resolved_path = self._resolve_include_bytes_path(path)

        if resolved_path is None:
            searched_dirs = [str(self.source_dir)] + [str(p) for p in self.include_paths]
            raise HIRError(
                f"include_bytes!: file not found: '{path}'\n"
                f"  searched in: {', '.join(searched_dirs)}",
                source_loc=source_loc
            )

        # Get file size
        file_size = resolved_path.stat().st_size
        return str(resolved_path), file_size
