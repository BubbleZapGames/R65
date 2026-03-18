# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Match expression validation for R65 type checker.

Handles validation of match expressions including pattern checking
and exhaustiveness analysis.
"""

from typing import Callable
from r65.compiler.hir import (
    HIRMatchExpression, HIRLiteralPattern, HIREnumPattern,
    HIRWildcardPattern, HIRIdentifierPattern, HIRRangePattern,
    HIROrPattern, BasicTypeInfo
)
from r65.compiler.hir.nodes import HIRStatement
from r65.compiler.hir.types import TypeInfo, EnumTypeInfo
from r65.compiler.typeck.errors import TypeCheckError
from r65.compiler.typeck.type_utils import TypeUtils


class MatchValidator:
    """Validates match expressions."""

    def __init__(self, check_expression_fn: Callable, check_statement_fn: Callable = None):
        """
        Initialize with expression checker callback.

        Args:
            check_expression_fn: Function to type check expressions (from TypeChecker)
            check_statement_fn: Function to type check statements (from TypeChecker)
        """
        self.check_expression = check_expression_fn
        self.check_statement = check_statement_fn

    def check_match_expression(self, expr: HIRMatchExpression, context_type=None) -> TypeInfo:
        """Type check match expression."""
        # Check scrutinee type
        scrutinee_type = self.check_expression(expr.scrutinee)

        # Check each arm
        arm_types = []
        has_wildcard = False

        # Determine the expected arm type: use context_type if provided,
        # otherwise infer from the first arm (done after the loop).
        expected_type = context_type

        for arm in expr.arms:
            # Check pattern matches scrutinee type and check for wildcard/identifier
            if self._check_pattern(arm.pattern, scrutinee_type):
                has_wildcard = True

            # Check arm body - statements (return/break/continue) have "never" type
            if isinstance(arm.body, HIRStatement):
                if self.check_statement:
                    self.check_statement(arm.body)
                arm_types.append(None)  # None = never type (diverging)
            else:
                # Check arm body with expected type context so integer literals
                # are inferred correctly (e.g., `0` as u16 when let binding is u16)
                body_type = self.check_expression(arm.body, expected_type)
                arm_types.append(body_type)

                # After checking the first arm, use its type as the expected type
                # for remaining arms (if no external context was provided)
                if expected_type is None:
                    expected_type = body_type

        # All arms must return compatible types
        if not arm_types:
            raise TypeCheckError(
                "Match expression must have at least one arm",
                source_loc=expr.source_loc
            )

        # Filter out None (never/diverging) types from statement arms
        expr_arm_types = [(i, t) for i, t in enumerate(arm_types) if t is not None]

        if expr_arm_types:
            # Use first non-diverging arm's type as the expected type
            result_type = expr_arm_types[0][1]
            for i, arm_type in expr_arm_types[1:]:
                if not TypeUtils.types_equal(result_type, arm_type):
                    raise TypeCheckError(
                        f"Match arm {i} returns type {arm_type}, expected {result_type}",
                        source_loc=expr.arms[i].body.source_loc
                    )
        else:
            # All arms are diverging (return/break/continue) - match has void type
            result_type = BasicTypeInfo(name='void')

        # Exhaustiveness check: must have wildcard/identifier pattern or cover all cases
        if not has_wildcard:
            self._check_match_exhaustiveness(expr, scrutinee_type)

        expr.expr_type = result_type
        return result_type

    def _check_match_exhaustiveness(self, expr: HIRMatchExpression, scrutinee_type: TypeInfo):
        """
        Check that match expression covers all possible values.

        For bool: must cover both true and false
        For enum: must cover all variants
        For integers: must have wildcard (too many values to enumerate)
        """
        # Collect all covered values/variants
        covered_values = set()
        covered_variants = set()

        def collect_patterns(pattern):
            """Recursively collect covered values from pattern."""
            if isinstance(pattern, HIRLiteralPattern):
                covered_values.add(pattern.value)
            elif isinstance(pattern, HIREnumPattern):
                covered_variants.add(pattern.variant_name)
            elif isinstance(pattern, HIRRangePattern):
                end = pattern.end + 1 if pattern.inclusive else pattern.end
                for v in range(pattern.start, end):
                    covered_values.add(v)
            elif isinstance(pattern, HIROrPattern):
                for subpat in pattern.patterns:
                    collect_patterns(subpat)

        for arm in expr.arms:
            collect_patterns(arm.pattern)

        # Check exhaustiveness based on scrutinee type
        if isinstance(scrutinee_type, BasicTypeInfo):
            if scrutinee_type.name == 'bool':
                # Bool must cover both true and false
                missing = []
                if True not in covered_values:
                    missing.append('true')
                if False not in covered_values:
                    missing.append('false')
                if missing:
                    raise TypeCheckError(
                        f"Non-exhaustive match: missing patterns for {', '.join(missing)}",
                        source_loc=expr.source_loc
                    )
            elif scrutinee_type.name in ('u8', 'i8', 'u16', 'i16'):
                # Integer types need wildcard - too many values to enumerate
                raise TypeCheckError(
                    f"Non-exhaustive match on {scrutinee_type}: "
                    f"add a wildcard pattern '_' to cover remaining values",
                    source_loc=expr.source_loc
                )

        elif isinstance(scrutinee_type, EnumTypeInfo):
            # Enum must cover all variants
            if scrutinee_type.definition:
                all_variants = {v.name for v in scrutinee_type.definition.variants}
                missing = all_variants - covered_variants
                if missing:
                    missing_list = ', '.join(sorted(missing))
                    raise TypeCheckError(
                        f"Non-exhaustive match on {scrutinee_type.name}: "
                        f"missing patterns for {missing_list}",
                        source_loc=expr.source_loc
                    )

    def _check_pattern(self, pattern, scrutinee_type: TypeInfo) -> bool:
        """
        Check if pattern is valid for scrutinee type.
        Returns True if pattern is a catch-all (wildcard or identifier).
        """
        if isinstance(pattern, HIRLiteralPattern):
            # Literal must match scrutinee type
            if isinstance(pattern.value, bool):
                if scrutinee_type.name != 'bool':
                    raise TypeCheckError(f"Cannot match bool literal against {scrutinee_type}", source_loc=pattern.source_loc)
            elif isinstance(pattern.value, int):
                if scrutinee_type.name not in ('u8', 'i8', 'u16', 'i16'):
                    raise TypeCheckError(f"Cannot match integer literal against {scrutinee_type}", source_loc=pattern.source_loc)
            return False

        elif isinstance(pattern, HIREnumPattern):
            # Enum pattern must match enum type
            # scrutinee should be the enum's underlying integer type
            return False

        elif isinstance(pattern, HIRWildcardPattern):
            # Wildcard always matches
            return True

        elif isinstance(pattern, HIRIdentifierPattern):
            # Identifier pattern always matches and binds the value
            # Set the symbol's type to the scrutinee type
            pattern.symbol.var_type = scrutinee_type
            return True

        elif isinstance(pattern, HIRRangePattern):
            # Range pattern only matches integer types
            if scrutinee_type.name not in ('u8', 'i8', 'u16', 'i16'):
                raise TypeCheckError(f"Cannot use range pattern against {scrutinee_type}", source_loc=pattern.source_loc)
            # Validate range is non-empty
            if pattern.inclusive:
                if pattern.start > pattern.end:
                    raise TypeCheckError("Empty range pattern", source_loc=pattern.source_loc)
            else:
                if pattern.start >= pattern.end:
                    raise TypeCheckError("Empty range pattern", source_loc=pattern.source_loc)
            return False

        elif isinstance(pattern, HIROrPattern):
            # Or pattern: check all sub-patterns
            is_catchall = False
            for subpat in pattern.patterns:
                if self._check_pattern(subpat, scrutinee_type):
                    is_catchall = True
            return is_catchall

        else:
            raise TypeCheckError(f"Unknown pattern type: {type(pattern).__name__}", source_loc=getattr(pattern, 'source_loc', None))
