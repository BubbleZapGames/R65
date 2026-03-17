# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Struct literal validation for R65 type checker.

Handles validation of struct literal expressions including field type checking.
"""

from typing import Callable
from r65.compiler.hir import HIRStructLiteralExpr, HIRStructDecl, SymbolKind, StructTypeInfo
from r65.compiler.hir.types import TypeInfo
from r65.compiler.typeck.errors import TypeCheckError
from r65.compiler.typeck.type_utils import TypeUtils


class StructValidator:
    """Validates struct literal expressions."""

    def __init__(self, symbol_table, const_evaluator,
                 check_expression_fn: Callable,
                 check_type_match_fn: Callable):
        """
        Initialize with dependencies.

        Args:
            symbol_table: Symbol table for looking up struct definitions
            const_evaluator: Const evaluator for type resolution
            check_expression_fn: Function to type check expressions
            check_type_match_fn: Function to check type matches
        """
        self.symbol_table = symbol_table
        self.const_evaluator = const_evaluator
        self.check_expression = check_expression_fn
        self.check_type_match = check_type_match_fn

    def check_struct_literal(self, expr: HIRStructLiteralExpr) -> TypeInfo:
        """Type check struct literal expression."""
        # Look up struct declaration
        struct_symbol = self.symbol_table.lookup(expr.struct_name)
        if not struct_symbol:
            raise TypeCheckError(
                f"undefined struct '{expr.struct_name}'",
                source_loc=expr.source_loc,
                hint="check spelling or add a struct declaration"
            )

        if struct_symbol.kind != SymbolKind.STRUCT:
            raise TypeCheckError(
                f"'{expr.struct_name}' is not a struct type",
                source_loc=expr.source_loc,
                hint=f"'{expr.struct_name}' is a {struct_symbol.kind.value}"
            )

        # Get struct definition
        struct_def = struct_symbol.definition
        if not struct_def:
            raise TypeCheckError(
                f"struct '{expr.struct_name}' has no definition",
                source_loc=expr.source_loc,
                hint="ensure the struct is defined before use"
            )

        # Build expected fields map from struct definition
        # Skip synthetic __type_id field — it's auto-initialized by the compiler
        expected_fields = {}
        if isinstance(struct_def, HIRStructDecl):
            for field in struct_def.fields:
                if field.name == '__type_id':
                    continue
                expected_fields[field.name] = field.field_type
        else:
            # AST struct definition - resolve types
            from r65.compiler.hir.types import TypeResolver
            type_resolver = TypeResolver(self.symbol_table, self.const_evaluator)
            for field in struct_def.fields:
                expected_fields[field.name] = type_resolver.resolve_type(field.field_type)

        # Check each field initializer
        provided_fields = set()
        for field_init in expr.fields:
            if field_init.name in provided_fields:
                raise TypeCheckError(
                    f"field '{field_init.name}' initialized multiple times",
                    source_loc=expr.source_loc,
                    hint="remove duplicate field initialization"
                )
            provided_fields.add(field_init.name)

            if field_init.name not in expected_fields:
                available = ', '.join(sorted(expected_fields.keys()))
                raise TypeCheckError(
                    f"struct '{expr.struct_name}' has no field '{field_init.name}'",
                    source_loc=expr.source_loc,
                    hint=f"available fields: {available}"
                )

            expected_type = expected_fields[field_init.name]
            actual_type = self.check_expression(field_init.value, expected_type)
            self.check_type_match(
                expected_type, actual_type, field_init.value,
                f"field '{field_init.name}'", expr.source_loc, use_compatible=True
            )

        # Check for missing fields
        missing_fields = set(expected_fields.keys()) - provided_fields
        if missing_fields:
            missing_list = ', '.join(sorted(missing_fields))
            raise TypeCheckError(
                f"missing field(s) in struct literal: {missing_list}",
                source_loc=expr.source_loc,
                hint=f"add: {missing_list}"
            )

        # Create struct type
        struct_type = StructTypeInfo(
            name=expr.struct_name,
            definition=struct_def if isinstance(struct_def, HIRStructDecl) else None
        )
        expr.expr_type = struct_type
        return struct_type
