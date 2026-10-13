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
from r65.compiler.hir.types import TypeInfo, EnumTypeInfo, strip_newtype, INTEGER_TYPE_NAMES
from r65.compiler.typeck.errors import TypeCheckError
from r65.compiler.typeck.type_utils import TypeUtils


class MatchValidator:
    """Validates match expressions."""

    def __init__(self, symbol_table, check_expression_fn: Callable,
                 check_statement_fn: Callable = None):
        """
        Initialize with expression checker callback.

        Args:
            symbol_table: Symbol table, for enum definitions by name
            check_expression_fn: Function to type check expressions (from TypeChecker)
            check_statement_fn: Function to type check statements (from TypeChecker)
        """
        self.symbol_table = symbol_table
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

        # Every arm of an enum match must name that enum. Variants are compared
        # by discriminant, and exhaustiveness by variant name, so nothing else
        # would notice a foreign variant: it would compile to a comparison
        # against an unrelated discriminant. A tagged enum's scrutinee is its
        # tag plane by now, so the enum comes from the match instead.
        enum_name = expr.tagged_enum
        if enum_name is None and isinstance(strip_newtype(scrutinee_type), EnumTypeInfo):
            enum_name = strip_newtype(scrutinee_type).name
        if enum_name is not None:
            self._check_arms_name_enum(expr, enum_name)

        # Exhaustiveness check: must have wildcard/identifier pattern or cover all cases
        if not has_wildcard:
            self._check_match_exhaustiveness(expr, scrutinee_type)

        expr.expr_type = result_type
        return result_type

    def _check_arms_name_enum(self, expr: HIRMatchExpression, enum_name: str):
        """Reject an arm naming a different enum than the one being matched."""
        def check(pattern):
            if isinstance(pattern, HIREnumPattern):
                if pattern.enum_name != enum_name:
                    raise TypeCheckError(
                        f"pattern names enum '{pattern.enum_name}', but the value "
                        f"being matched is a '{enum_name}'",
                        source_loc=pattern.source_loc
                    )
            elif isinstance(pattern, HIROrPattern):
                for subpat in pattern.patterns:
                    check(subpat)

        for arm in expr.arms:
            check(arm.pattern)

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

        # A tagged enum's scrutinee is its tag plane, so it *looks* like a plain
        # u8 here. Ask the enum instead: the values it can hold are its variants.
        if expr.tagged_enum is not None:
            self._check_enum_variants_covered(expr, expr.tagged_enum, covered_variants)
            return

        # Exhaustiveness is a question about the *values* a type can hold, so ask
        # it of the payload — a newtype over u8 needs a wildcard just as u8 does.
        # Without this the isinstance checks below simply miss it and the match is
        # accepted unchecked. `scrutinee_type` is kept for the messages, which
        # should name what the reader wrote.
        payload_type = strip_newtype(scrutinee_type)

        # Check exhaustiveness based on scrutinee type
        if isinstance(payload_type, BasicTypeInfo):
            if payload_type.name == 'bool':
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
            elif payload_type.name in INTEGER_TYPE_NAMES:
                # Integer types need wildcard - too many values to enumerate
                raise TypeCheckError(
                    f"Non-exhaustive match on {scrutinee_type}: "
                    f"add a wildcard pattern '_' to cover remaining values",
                    source_loc=expr.source_loc
                )

        elif isinstance(scrutinee_type, EnumTypeInfo):
            # Enum must cover all variants
            if scrutinee_type.definition:
                self._check_enum_variants_covered(
                    expr, scrutinee_type.name, covered_variants,
                    definition=scrutinee_type.definition)

    def _check_enum_variants_covered(self, expr, enum_name, covered, definition=None):
        """Every variant of *enum_name* must appear among *covered*."""
        if definition is None:
            symbol = self.symbol_table.lookup(enum_name)
            definition = symbol.definition if symbol else None
        if definition is None:
            return
        missing = {v.name for v in definition.variants} - covered
        if missing:
            raise TypeCheckError(
                f"Non-exhaustive match on {enum_name}: "
                f"missing patterns for {', '.join(sorted(missing))}",
                source_loc=expr.source_loc
            )

    def _check_pattern(self, pattern, scrutinee_type: TypeInfo) -> bool:
        """
        Check if pattern is valid for scrutinee type.
        Returns True if pattern is a catch-all (wildcard or identifier).
        """
        # Whether a literal can match is a machine-representation question, so ask
        # the payload: a newtype over u8 matches integer literals, transparently
        # in, like every other position that accepts one. Errors still name the
        # newtype rather than the payload, which is what the reader wrote.
        payload_type = strip_newtype(scrutinee_type)

        # Only a scalar payload has the `.name` the literal/range checks below
        # read, so reject a literal or range pattern against anything else.
        if (not isinstance(payload_type, BasicTypeInfo)
                and isinstance(pattern, (HIRLiteralPattern, HIRRangePattern))):
            raise TypeCheckError(
                f"cannot match on {scrutinee_type}",
                source_loc=pattern.source_loc,
                hint="match works on integers, bools, and enums; compare with "
                     "'==' instead, or match on a field of the value"
            )

        if isinstance(pattern, HIRLiteralPattern):
            # Literal must match scrutinee type
            if isinstance(pattern.value, bool):
                if payload_type.name != 'bool':
                    raise TypeCheckError(f"Cannot match bool literal against {scrutinee_type}", source_loc=pattern.source_loc)
            elif isinstance(pattern.value, int):
                if payload_type.name not in INTEGER_TYPE_NAMES:
                    raise TypeCheckError(f"Cannot match integer literal against {scrutinee_type}", source_loc=pattern.source_loc)
            return False

        elif isinstance(pattern, HIREnumPattern):
            # Which enum it names is checked per arm by `_check_arms_name_enum`.
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
            if payload_type.name not in INTEGER_TYPE_NAMES:
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
