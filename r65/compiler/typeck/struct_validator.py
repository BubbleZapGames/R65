# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Struct literal validation for R65 type checker.

Handles validation of struct literal expressions including field type checking.
"""

from typing import Callable
from r65.compiler.hir import HIRStructLiteralExpr, SymbolKind, StructTypeInfo
from r65.compiler.hir.types import TypeInfo
from r65.compiler.typeck.errors import TypeCheckError
from r65.compiler.typeck.type_utils import TypeUtils, widen_implicitly


class StructValidator:
    """Validates struct literal expressions."""

    def __init__(self, symbol_table,
                 check_expression_fn: Callable,
                 check_type_match_fn: Callable):
        """
        Initialize with dependencies.

        Args:
            symbol_table: Symbol table for looking up struct definitions
            check_expression_fn: Function to type check expressions
            check_type_match_fn: Function to check type matches
        """
        self.symbol_table = symbol_table
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

        # HIR building has replaced every struct symbol's AST definition with
        # its HIRStructDecl by the time type checking runs.
        # Build expected fields map from struct definition
        # Skip synthetic __type_id field — it's auto-initialized by the compiler
        expected_fields = {}
        for field in struct_def.fields:
            if field.name == '__type_id':
                continue
            expected_fields[field.name] = field.field_type

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
            field_init.value = widen_implicitly(field_init.value, actual_type, expected_type)

        # A union literal names exactly one field — the other fields alias the same
        # bytes, so initializing two of them would be contradictory. A struct literal
        # must name all of them.
        if getattr(struct_def, 'is_union', False):
            if len(provided_fields) != 1:
                raise TypeCheckError(
                    f"union literal for '{expr.struct_name}' must initialize exactly "
                    f"one field, found {len(provided_fields)}",
                    source_loc=expr.source_loc,
                    hint=f"union fields overlap; pick one of: {', '.join(sorted(expected_fields.keys()))}"
                )
        else:
            missing_fields = set(expected_fields.keys()) - provided_fields
            if missing_fields:
                missing_list = ', '.join(sorted(missing_fields))
                raise TypeCheckError(
                    f"missing field(s) in struct literal: {missing_list}",
                    source_loc=expr.source_loc,
                    hint=f"add: {missing_list}"
                )

        # Create struct type
        struct_type = StructTypeInfo(name=expr.struct_name, definition=struct_def)
        expr.expr_type = struct_type
        return struct_type
