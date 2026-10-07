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
from r65.compiler.hir.types import (BasicTypeInfo, ArrayTypeInfo, NewtypeTypeInfo,
                                    EnumTypeInfo, StructTypeInfo, TypeInfo)
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
        # The static backing the `#[soa]` struct whose method body is being
        # built, or None outside one. Set by HIRBuilder around _build_block.
        self.soa_self_pool = None
        # Same, for the tagged enum whose method body is being built: `(enum
        # name, pool Symbol)`, or None outside one. Lets `match self` find the
        # planes `self` indexes.
        self.enum_self_pool = None
        # Shared with HIRBuilder — all three are filled before pass 2 builds any
        # body, so they answer the same regardless of whether an enum is declared
        # above or below the code that matches on it. Reading the enum's
        # `definition` instead would depend on that order, because it is still
        # the AST node until `_build_enum` runs.
        self.enum_storage = {}    # enum name -> (pool Symbol, source_loc, prefix)
        self.tagged_enums = {}    # tagged enum name -> all payloads are #[soa]
        self.enum_payloads = {}   # enum name -> variant name -> StructTypeInfo
        self.payload_structs = {} # #[soa] struct name -> (enum name, variant name)
        # Set around the body of a method on an `#[soa]` payload struct:
        # `(enum name, variant name)`. `self` is then the pool index, and
        # `self.field` reaches the variant's plane at that index.
        self.payload_self = None
        # Numbers the temporaries a variant construction reads its operands into.
        self._enum_temp_counter = 0

    def _payload_field_names(self, enum_name: str, variant_name: str):
        """The field names of *variant_name*'s payload struct, AST or HIR."""
        payload_type = self.enum_payloads[enum_name][variant_name]
        symbol = self.symbol_table.lookup(payload_type.name)
        definition = getattr(symbol, 'definition', None)
        return {f.name for f in getattr(definition, 'fields', ())}

    def _soa_payload_plane(self, enum_name: str, variant_name: str,
                           field_name: str, index_ast, loc):
        """`POOL__kind__Variant__field[index]` for an all-`#[soa]` enum.

        The variant's field planes are declared in pass 1.5 alongside tag and
        data, so this is an ordinary indexed static by the time anything
        downstream sees it — and it strides by the field, not the payload.
        """
        return self._enum_plane_access(
            enum_name, f"{variant_name}__{field_name}", index_ast, loc)

    def _soa_payload_access(self, expr: ast.FieldAccess, src_loc):
        """`f.field` on an `#[soa]` payload -> the variant's plane, else None.

        Two bases reach here: a payload binding `f` from a `match` arm, and
        `self` inside a method on the payload struct. Both are the pool index
        in disguise. A name that is not a field — a method — is left to the
        ordinary path, which builds the `data[i].Variant.method` shape the
        call validator already reads the index out of.
        """
        base = expr.base
        if not isinstance(base, ast.Identifier):
            return None

        if base.name == 'self' and self.payload_self is not None:
            enum_name, variant_name = self.payload_self
            if expr.field not in self._payload_field_names(enum_name, variant_name):
                return None
            symbol = self.symbol_table.lookup('self')
            index = lambda loc: hir.HIRIdentifier(name='self', symbol=symbol,
                                                  source_loc=loc)
            return self._soa_payload_plane(enum_name, variant_name, expr.field,
                                           index, src_loc)

        symbol = self.symbol_table.lookup(base.name)
        alias = getattr(symbol, 'enum_payload_alias', None)
        if alias is None:
            return None
        enum_name, index_ast, variant_name = alias
        if not self.tagged_enums.get(enum_name, False):
            return None
        if expr.field not in self._payload_field_names(enum_name, variant_name):
            return None
        return self._soa_payload_plane(enum_name, variant_name, expr.field,
                                       index_ast, src_loc)

    def _payload_self_ident(self, expr):
        """`self` in a payload struct's method, where it may appear, else None.

        Inside a method on an `#[soa]` payload struct `self` is an element of
        its variant, not a number, so a bare `self` is rejected (see the
        `ast.Identifier` branch). Two positions are exempt and build it here:
        the base of `self.field` / `self.method()`, and `self as u16`, the
        explicit way to ask for the pool index.
        """
        if (self.payload_self is None or not isinstance(expr, ast.Identifier)
                or expr.name != 'self'):
            return None
        return hir.HIRIdentifier(name='self', symbol=self.symbol_table.lookup('self'),
                                 source_loc=expr.source_loc)

    def _soa_rewrite_base(self, base, src_loc):
        """Turn an `#[soa]` field-access base into an explicit element index.

        `#[soa]` has exactly one legal element shape — `POOL[i]` — so rewriting
        the two implicit spellings into it here, before the type checker runs,
        lets `self.hp` and `PLAYER.hp` reuse the whole `arr[i].field` path:
        typecheck, the MIR funnel, the plane rewrite, compound assignment and
        method receivers all work with no further special cases.

          self     (inside an #[soa] method)  ->  POOL[self]
          PLAYER   (a single-instance static) ->  PLAYER[0]

        Anything else is returned unchanged.
        """
        if not isinstance(base, hir.HIRIdentifier):
            return base

        # `self` in a method on an #[soa] payload struct: the receiver of
        # `self.m()` takes the same `data[self].Variant` shape a payload
        # binding expands to, so the call validator finds the index the same
        # way for both.
        if self.payload_self is not None and base.name == 'self':
            enum_name, variant_name = self.payload_self
            symbol = base.symbol
            index = lambda loc: hir.HIRIdentifier(name='self', symbol=symbol,
                                                  source_loc=loc)
            return hir.HIRFieldAccess(
                base=self._enum_plane_access(enum_name, 'data', index, src_loc),
                field_name=variant_name,
                source_loc=src_loc,
            )

        # `self` in an #[soa] method is the element index into the pool.
        pool = self.soa_self_pool
        if pool is not None and base.name == 'self':
            return hir.HIRArrayIndex(
                array=hir.HIRIdentifier(name=pool.name, symbol=pool, source_loc=src_loc),
                index=base,
                source_loc=src_loc,
            )

        # A single instance is a pool of one.
        if getattr(getattr(base, 'symbol', None), 'soa_single', False):
            return hir.HIRArrayIndex(
                array=base,
                index=hir.HIRIntegerLiteral(value=0, source_loc=src_loc),
                source_loc=src_loc,
            )

        return base

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
            if expr.name == 'self' and self.payload_self is not None:
                enum_name, variant_name = self.payload_self
                struct_name = self.enum_payloads[enum_name][variant_name].name
                raise HIRError(
                    f"'self' in a method on #[soa] payload struct '{struct_name}' "
                    f"is not a value",
                    source_loc=src_loc,
                    hint=f"it is an element of '{enum_name}::{variant_name}', not an "
                         f"index: reach its own fields as `self.field`; code that "
                         f"needs the parent's fields belongs in an impl on the "
                         f"parent; write `self as u16` if the pool index itself is "
                         f"needed",
                )
            # Resolve identifier
            symbol = self.symbol_table.lookup(expr.name)
            if not symbol:
                raise HIRError(f"Undefined identifier: {expr.name}", source_loc=src_loc)
            alias = getattr(symbol, 'enum_payload_alias', None)
            if alias is not None:
                # A payload binding is a view of the pool, not a local holding a
                # copy: expand it to `POOL__data[i].Variant` at every use.
                enum_name, index_ast, variant_name = alias
                return hir.HIRFieldAccess(
                    base=self._enum_plane_access(enum_name, 'data', index_ast, src_loc),
                    field_name=variant_name,
                    source_loc=src_loc,
                )
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
                literal = hir.HIRIntegerLiteral(
                    value=qualified_symbol.const_value,
                    source_loc=src_loc
                )
                # Folding to a bare literal drops the declared type, which for a
                # newtype is the whole of its meaning: `Color::WHITE` would come
                # back a u16, so it would flow into any u16 and would not answer
                # to '.0'. Stamping the type keeps the constant as nominal as the
                # declaration that named it. The checker honours a pre-set
                # `expr_type` on a literal.
                if isinstance(qualified_symbol.var_type, NewtypeTypeInfo):
                    literal.expr_type = qualified_symbol.var_type
                return literal

            # Otherwise resolve as enum variant
            enum_symbol = self.symbol_table.lookup(expr.enum_name)
            if not enum_symbol:
                raise HIRError(f"Undefined type: {expr.enum_name}", source_loc=src_loc)
            if enum_symbol.kind != SymbolKind.ENUM:
                # `Name::member` also spells associated functions and constants.
                # Reaching here means the member did not resolve, so complain
                # about the member rather than about `Name` not being an enum —
                # which is what a cfg'd-out method or a typo looks like.
                if enum_symbol.kind in (SymbolKind.STRUCT, SymbolKind.NEWTYPE):
                    raise HIRError(
                        f"'{expr.enum_name}' has no associated function or "
                        f"constant named '{expr.variant_name}'",
                        source_loc=src_loc,
                        hint=f"check the spelling, or whether "
                             f"'{expr.enum_name}::{expr.variant_name}' is behind "
                             f"a #[cfg(...)] that is not enabled")
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
            target = self.type_resolver.resolve_type(expr.target_type)
            tag = self._tagged_enum_tag_cast(expr, target, src_loc)
            if tag is not None:
                return tag
            inner = (self._payload_self_ident(expr.expr)
                     or self.build_expression(expr.expr))
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

            plane = self._soa_payload_access(expr, src_loc)
            if plane is not None:
                return plane

            base = (self._payload_self_ident(expr.base)
                    or self.build_expression(expr.base))

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
            base = self._soa_rewrite_base(base, src_loc)
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

    def _tagged_match_target(self, scrutinee: ast.Expression):
        """`(enum_name, index_ast)` if *scrutinee* names a tagged enum element.

        A tagged enum has no contiguous element, so matching on one is really
        matching on its tag plane at some index. The three spellings that reach
        an element all reduce to an index here:

          POOL[i]  ->  i
          self     ->  self          (inside a tagged enum method)
          SHAPE    ->  0             (a single instance is a pool of one)

        Returns None for anything else, including a fieldless enum, which is an
        ordinary value and matches directly.
        """
        loc = getattr(scrutinee, 'source_loc', None)

        if isinstance(scrutinee, ast.ArrayIndex):
            base = scrutinee.array
            if isinstance(base, ast.Identifier):
                enum_name = self._tagged_enum_static(base.name)
                if enum_name is not None:
                    return enum_name, scrutinee.index
            return None

        # POOL[i].field — the enum is a field of an `#[soa]` element, so the
        # element is still reached by index and the planes are the pool's
        # nested ones. The index is the outer array's, exactly as above.
        if isinstance(scrutinee, ast.FieldAccess):
            base = getattr(scrutinee, 'base', None)
            if isinstance(base, ast.ArrayIndex) and isinstance(base.array, ast.Identifier):
                enum_name = self._nested_tagged_enum(base.array.name,
                                                     scrutinee.field)
                if enum_name is not None:
                    return enum_name, base.index
            # self.field inside a method on that `#[soa]` struct — the same
            # element, with `self` as the index. This is where code that needs
            # both the shared fields and a payload belongs: a payload struct's
            # own methods cannot reach the parent.
            pool = self.soa_self_pool
            if (pool is not None and isinstance(base, ast.Identifier)
                    and base.name == 'self'):
                enum_name = self._nested_tagged_enum(pool.name, scrutinee.field)
                if enum_name is not None:
                    return enum_name, base
            return None

        if isinstance(scrutinee, ast.Identifier):
            if scrutinee.name == 'self' and self.enum_self_pool is not None:
                return self.enum_self_pool[0], scrutinee
            enum_name = self._tagged_enum_static(scrutinee.name)
            if enum_name is not None:
                return enum_name, ast.IntegerLiteral(value=0, source_loc=loc)
        return None

    def _tagged_enum_tag_cast(self, expr: ast.TypeCast, target, loc):
        """`POOL[i] as u8` -> a read of the tag plane, else None.

        A tagged enum element has no contiguous value, so a cast of one is not a
        conversion of anything — it is a read of the discriminant, which is the
        tag plane's byte. That is the same rewrite `match` performs on the same
        spelling (`_tagged_match_target`), stopping at the load:

            LDA POOL__tag,X

        A wider target keeps the ordinary explicit u8 -> u16 widening on top,
        which is the path that zero-extends correctly. A non-integer target is
        left alone, so `POOL[i] as *u8` still hits the planed-value rejection in
        the type checker.
        """
        if not isinstance(target, BasicTypeInfo) or target.name == 'bool':
            return None
        tagged = self._tagged_match_target(expr.expr)
        if tagged is None:
            return None
        enum_name, index_ast = tagged
        tag = self._enum_plane_access(enum_name, 'tag', index_ast, loc)
        if target.name == 'u8':
            return tag
        return hir.HIRTypeCast(expr=tag, target_type=target, source_loc=loc)

    def _nested_tagged_enum(self, pool_name: str, field_name: str):
        """The tagged enum reached by `pool_name[i].field_name`, else None.

        Answered from the storage registry rather than the struct definition:
        the registry already records which pool and which field back each
        tagged enum, and it is filled before any body is built, so this gives
        the same answer whichever order the declarations appear in.
        """
        symbol = self.symbol_table.lookup(pool_name)
        if symbol is None:
            return None
        for enum_name, (pool_symbol, _loc, prefix) in self.enum_storage.items():
            if pool_symbol is symbol and prefix == f"{field_name}__":
                return enum_name
        return None

    def _tagged_enum_static(self, name: str):
        """The tagged enum name a static of *name* pools, else None."""
        symbol = self.symbol_table.lookup(name)
        var_type = getattr(symbol, 'var_type', None) if symbol else None
        if var_type is None:
            return None
        element = getattr(var_type, 'element_type', var_type)
        # isinstance first, deliberately: `NewtypeTypeInfo.name` raises rather
        # than answer a nominal question, and it raises through `getattr`.
        if not isinstance(element, EnumTypeInfo):
            return None
        return element.name if element.name in self.tagged_enums else None

    def _enum_plane_access(self, enum_name: str, plane: str, index_ast, loc):
        """`PLANE[index]` for one of a tagged enum's planes.

        The planes are ordinary statics — declared in pass 1.5 precisely so this
        can be built during pass 2 — so the result needs no further special
        handling anywhere downstream.
        """
        pool_symbol, _loc, prefix = self.enum_storage[enum_name]
        plane_symbol = self.symbol_table.lookup(f"{pool_symbol.name}__{prefix}{plane}")
        # The index is an AST node to rebuild per use, or — for `self` in a
        # payload struct's method, which has no AST — a factory of fresh HIR.
        index = index_ast(loc) if callable(index_ast) else self.build_expression(index_ast)
        return hir.HIRArrayIndex(
            array=hir.HIRIdentifier(name=plane_symbol.name, symbol=plane_symbol,
                                    source_loc=loc),
            index=index,
            source_loc=loc,
        )

    def build_enum_construction(self, expr: ast.Expression):
        """Lower `POOL[i] = Shape::C(Circle {...});` to the two plane stores.

        Returns an `HIRBlock` of the stores, or None if *expr* is not a tagged
        enum construction — in which case the caller builds it as an ordinary
        expression statement.

        A tagged enum value has no contiguous form, so there is no temporary to
        assemble and copy: the tag goes to the tag plane and each payload field
        to the payload plane, in place. A unit variant writes only the tag.
        """
        if not isinstance(expr, ast.Assignment):
            return None
        target = self._tagged_match_target(expr.target)
        if target is None:
            return None
        enum_name, index_ast = target
        loc = getattr(expr, 'source_loc', None)

        value = expr.value
        payload = None
        if isinstance(value, ast.EnumVariantExpr):
            variant_expr = value
        elif (isinstance(value, ast.FunctionCall)
              and isinstance(value.func, ast.EnumVariantExpr)):
            variant_expr = value.func
            if len(value.args) != 1:
                raise HIRError(
                    f"variant '{variant_expr.enum_name}::{variant_expr.variant_name}' "
                    f"takes one payload, got {len(value.args)}",
                    source_loc=loc,
                )
            payload = value.args[0]
        else:
            raise HIRError(
                f"cannot assign to a '{enum_name}' element",
                source_loc=loc,
                hint=f"a tagged enum element is written as a variant, e.g. "
                     f"`{enum_name}::Variant(Payload {{ ... }})`; it has no "
                     f"contiguous form to copy",
            )

        if variant_expr.enum_name != enum_name:
            raise HIRError(
                f"cannot assign a '{variant_expr.enum_name}' to a '{enum_name}' element",
                source_loc=loc,
            )

        variant_name = variant_expr.variant_name
        variant_symbol = self.symbol_table.lookup(f"{enum_name}::{variant_name}")
        if variant_symbol is None or variant_symbol.kind != SymbolKind.ENUM_VARIANT:
            raise HIRError(
                f"Undefined enum variant: {enum_name}::{variant_name}",
                source_loc=loc,
            )
        payload_type = self.enum_payloads.get(enum_name, {}).get(variant_name)
        if (payload_type is None) != (payload is None):
            if payload is None:
                raise HIRError(
                    f"variant '{enum_name}::{variant_name}' needs a payload",
                    source_loc=loc,
                    hint=f"write `{enum_name}::{variant_name}("
                         f"{payload_type} {{ ... }})`",
                )
            raise HIRError(
                f"variant '{enum_name}::{variant_name}' takes no payload",
                source_loc=loc,
                hint=f"write `{enum_name}::{variant_name}` on its own",
            )

        tag_store = hir.HIRExprStmt(
            expr=hir.HIRAssignment(
                target=self._enum_plane_access(enum_name, 'tag', index_ast, loc),
                value=hir.HIRIntegerLiteral(value=variant_symbol.const_value,
                                            source_loc=loc),
                source_loc=loc),
            source_loc=loc)

        stores = []
        if payload is not None and self.tagged_enums.get(enum_name, False):
            # No union slot to write into: each top-level field goes to its
            # own plane at the raw index. Anything but a literal would need a
            # contiguous source to copy from, which an #[soa] payload never has.
            if not isinstance(payload, ast.StructLiteralExpr):
                raise HIRError(
                    f"payload of '{enum_name}::{variant_name}' must be a struct "
                    f"literal",
                    source_loc=loc,
                    hint="an #[soa] payload is written one field per plane, so "
                         "there is no whole value to copy in; write "
                         f"`{enum_name}::{variant_name}({payload_type.name} {{ ... }})`",
                )
            for field in payload.fields:
                target = self._soa_payload_plane(enum_name, variant_name,
                                                 field.name, index_ast, loc)
                stores.extend(self._payload_stores(
                    target, field.value, self._field_type(payload_type, field.name), loc))
        elif payload is not None:
            slot = hir.HIRFieldAccess(
                base=self._enum_plane_access(enum_name, 'data', index_ast, loc),
                field_name=variant_name,
                source_loc=loc,
            )
            stores.extend(self._payload_stores(slot, payload, payload_type, loc))

        scope_id = self.symbol_table.enter_scope(ScopeKind.BLOCK)
        statements = self._stores_after_operands(stores, loc)
        self.symbol_table.exit_scope()
        statements.append(tag_store)
        return hir.HIRBlock(statements=statements, scope_id=scope_id, source_loc=loc)

    def _payload_stores(self, slot, payload: ast.Expression, slot_type, loc):
        """`[(slot, value AST, slot type)]`, one per leaf store of *payload*.

        A struct literal is written field by field — there is no temporary to
        build first. Anything else names an existing aggregate, which is copied
        the way any struct assignment is.

        A field that is itself a struct literal recurses, so
        `Unit::Tank(Tank { inst: UnitInstance { hp: 5 }, reload: 0 })` writes
        `slot.inst.hp` directly. Emitting `slot.inst = UnitInstance { .. }` for
        the inner one instead would be an aggregate assignment, which a bare
        `=` refuses — the whole point of writing the payload in place is that
        there is no aggregate to assign.
        """
        if isinstance(payload, ast.StructLiteralExpr):
            stores = []
            for field in payload.fields:
                target = hir.HIRFieldAccess(base=slot, field_name=field.name,
                                            source_loc=loc)
                stores.extend(self._payload_stores(
                    target, field.value, self._field_type(slot_type, field.name), loc))
            return stores
        return [(slot, payload, slot_type)]

    def _field_type(self, struct_type, field_name: str):
        """The resolved type of *struct_type*'s field, or None if unknown."""
        name = getattr(struct_type, 'name', None) if isinstance(struct_type, StructTypeInfo) else None
        symbol = self.symbol_table.lookup(name) if name else None
        for f in getattr(getattr(symbol, 'definition', None), 'fields', ()):
            if f.name == field_name:
                ft = f.field_type
                return ft if isinstance(ft, TypeInfo) else self.type_resolver.resolve_type(ft)
        return None

    def _stores_after_operands(self, stores, loc):
        """The payload stores, ordered so every operand is read before any write.

        The stores overwrite the element the operands may be reading: a variant
        rebuilt from the one it replaces (`Rolling { anim: self.anim, .. }` from a
        `Ladder`) reads a plane, or union bytes, that an earlier field store
        has already overlaid. So every non-constant operand but the last is
        evaluated into a temporary first, in source order; the last is stored
        straight from its evaluation, since nothing has been written yet; then
        the rest. The caller puts the tag store after all of them.

        A constant needs no temporary, and an aggregate operand cannot have
        one — it is copied in place, and is the payload's only store.
        """
        def is_const(value):
            try:
                self.const_evaluator.eval(value)
                return True
            except Exception:
                return False

        live = [k for k, (_slot, value, slot_type) in enumerate(stores)
                if not is_const(value)
                and not isinstance(slot_type, StructTypeInfo)]
        hoisted = {k: None for k in live[:-1]
                   if stores[k][2] is not None}

        statements = []
        for k in hoisted:
            _slot, value, slot_type = stores[k]
            self._enum_temp_counter += 1
            name = f"__variant_arg{self._enum_temp_counter}"
            symbol = Symbol(name=name, kind=SymbolKind.LOCAL_VAR, definition=None,
                            scope_id=self.symbol_table.current_scope_id,
                            var_type=slot_type, is_mutable=False)
            self.symbol_table.declare(name, symbol)
            statements.append(hir.HIRLetStmt(
                name=name, is_mutable=False, var_type=slot_type,
                initializer=self.build_expression(value), symbol=symbol,
                source_loc=loc))
            hoisted[k] = symbol

        def store(k):
            slot, value, _slot_type = stores[k]
            symbol = hoisted.get(k)
            built = (hir.HIRIdentifier(name=symbol.name, symbol=symbol, source_loc=loc)
                     if symbol is not None else self.build_expression(value))
            return hir.HIRExprStmt(
                expr=hir.HIRAssignment(target=slot, value=built, source_loc=loc),
                source_loc=loc)

        first = live[-1] if live else None
        if first is not None:
            statements.append(store(first))
        statements.extend(store(k) for k in range(len(stores)) if k != first)
        return statements

    def _build_match_expression(self, expr: ast.MatchExpression) -> hir.HIRMatchExpression:
        """Build HIR match expression from AST."""
        # A tagged enum scrutinee is really its tag: rewrite to a read of the
        # tag plane, and hand the arms the index so a payload binding can reach
        # the same element of the data plane.
        tagged = self._tagged_match_target(expr.scrutinee)
        if tagged is not None:
            enum_name, index_ast = tagged
            scrutinee = self._enum_plane_access(
                enum_name, 'tag', index_ast, getattr(expr, 'source_loc', None))
        else:
            enum_name = index_ast = None
            scrutinee = self.build_expression(expr.scrutinee)

        # Build each arm
        arms = []
        for ast_arm in expr.arms:
            # Enter new scope for pattern bindings
            scope_id = self.symbol_table.enter_scope(ScopeKind.BLOCK)

            # Build pattern (may create bindings in scope)
            pattern = self._build_pattern(ast_arm.pattern, enum_name, index_ast)

            # Build body expression or statement (return/break/continue)
            if isinstance(ast_arm.body, (ast.ReturnStmt, ast.BreakStmt, ast.ContinueStmt, ast.Block)):
                body = self.statement_builder(ast_arm.body)
            else:
                body = self.build_expression(ast_arm.body)

            # Exit scope
            self.symbol_table.exit_scope()

            arms.append(hir.HIRMatchArm(pattern=pattern, body=body, scope_id=scope_id))

        return hir.HIRMatchExpression(scrutinee=scrutinee, arms=arms,
                                      tagged_enum=enum_name,
                                      source_loc=expr.source_loc)

    def _declare_payload_binding(self, pattern: ast.EnumPattern, match_enum,
                                 match_index, loc):
        """Declare the local a payload pattern binds, aliased to the pool.

        The binding is not a copy. It resolves to `POOL__data[i].Variant` — the
        payload plane's element viewed as this variant — so `c.r` is one
        absolute-indexed load and writing through it writes the pool. The alias
        is stored as the *AST* index rather than a built expression so each use
        rebuilds fresh HIR nodes; sharing one subtree across uses would hand MIR
        the same node to lower more than once.
        """
        if match_enum is None:
            raise HIRError(
                f"pattern '{pattern.enum_name}::{pattern.variant_name}({pattern.binding})' "
                f"binds a payload, but the value being matched is not a tagged "
                f"enum element",
                source_loc=loc,
                hint="match a pool element, e.g. `match POOL[i] { ... }`, or drop "
                     "the binding",
            )
        if match_enum != pattern.enum_name:
            raise HIRError(
                f"pattern names enum '{pattern.enum_name}', but the value being "
                f"matched is a '{match_enum}'",
                source_loc=loc,
            )

        payload_type = self.enum_payloads.get(match_enum, {}).get(
            pattern.variant_name)
        if payload_type is None:
            raise HIRError(
                f"variant '{match_enum}::{pattern.variant_name}' has no payload to "
                f"bind",
                source_loc=loc,
                hint=f"write '{match_enum}::{pattern.variant_name}' with no "
                     f"parentheses",
            )

        symbol = Symbol(
            name=pattern.binding,
            kind=SymbolKind.LOCAL_VAR,
            definition=None,
            scope_id=self.symbol_table.current_scope_id,
            var_type=payload_type,
        )
        symbol.enum_payload_alias = (match_enum, match_index, pattern.variant_name)
        self.symbol_table.declare(pattern.binding, symbol)
        return symbol

    def _build_pattern(self, pattern: ast.Pattern, match_enum: str = None,
                       match_index=None) -> hir.HIRPattern:
        """Build HIR pattern from AST pattern.

        `match_enum` / `match_index` describe a tagged enum scrutinee: the enum
        being matched and the AST index of the element. A payload binding needs
        both, since it resolves to that element of the payload plane.
        """
        if isinstance(pattern, ast.LiteralPattern):
            return hir.HIRLiteralPattern(value=pattern.value)

        elif isinstance(pattern, ast.EnumPattern):
            loc = getattr(pattern, 'source_loc', None)
            # Resolve enum variant value
            enum_symbol = self.symbol_table.lookup(pattern.enum_name)
            if not enum_symbol or enum_symbol.kind != SymbolKind.ENUM:
                raise HIRError(f"Undefined enum: {pattern.enum_name}", source_loc=loc)

            qualified_name = f"{pattern.enum_name}::{pattern.variant_name}"
            variant_symbol = self.symbol_table.lookup(qualified_name)
            if not variant_symbol or variant_symbol.kind != SymbolKind.ENUM_VARIANT:
                raise HIRError(f"Undefined enum variant: {qualified_name}", source_loc=loc)

            variant_value = variant_symbol.const_value
            binding_symbol = None
            if pattern.binding is not None:
                binding_symbol = self._declare_payload_binding(
                    pattern, match_enum, match_index, loc)
            return hir.HIREnumPattern(
                enum_name=pattern.enum_name,
                variant_name=pattern.variant_name,
                variant_value=variant_value,
                binding=pattern.binding,
                binding_symbol=binding_symbol,
                source_loc=loc,
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
            # A payload binding cannot survive an alternation: which variant
            # matched decides how the payload plane's bytes are typed, and two
            # payload variants never share a type. Binding one and reaching the
            # arm through the other would read the wrong struct with no
            # diagnostic — the exact confusion the tag exists to prevent.
            for alternative in pattern.patterns:
                if (isinstance(alternative, ast.EnumPattern)
                        and alternative.binding is not None):
                    raise HIRError(
                        f"pattern '{alternative.enum_name}::"
                        f"{alternative.variant_name}({alternative.binding})' binds a "
                        f"payload inside an or-pattern",
                        source_loc=getattr(pattern, 'source_loc', None),
                        hint="each variant's payload is a different type, so a "
                             "binding shared across alternatives has no one type; "
                             "give the variant its own arm",
                    )
            patterns = [self._build_pattern(p, match_enum, match_index)
                        for p in pattern.patterns]
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
