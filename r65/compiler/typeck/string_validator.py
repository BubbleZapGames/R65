# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
String literal validation for R65 type checker.

Handles validation of string literals used as byte array initializers,
including escape sequence processing and Extended ASCII validation.
"""

from typing import Optional, List
from r65.compiler.hir import HIRStringLiteral, BasicTypeInfo
from r65.compiler.hir.types import ArrayTypeInfo, PointerTypeInfo, TypeInfo
from r65.compiler.typeck.errors import TypeCheckError
from r65.compiler.codegen.bank_size import LOROM_BANK_SIZE
from r65.compiler.frontend.tokens import decode_literal


class StringValidator:
    """Validates string literals for byte array initialization."""

    @staticmethod
    def check_string_literal(expr: HIRStringLiteral, context_type: Optional[TypeInfo]) -> TypeInfo:
        """
        Type check a string literal.

        A string literal is either a `[u8; N]` initializer (zero-padded) or,
        with no context or a `*u8` context, an anonymous ROM array whose
        pointer takes the context's far-ness.
        Code points up to U+00FF are stored as that byte; anything above is rejected.
        Escapes are decoded by frontend.tokens.decode_literal.

        Args:
            expr: String literal expression
            context_type: Expected type ([u8; N], *u8, far *u8, or None)

        Returns:
            ArrayTypeInfo with u8 element type, or PointerTypeInfo to u8 for an
            inline literal
        """
        # Check for inline string literal context (no context or *u8 pointer context).
        # Inline string literals are promoted to an anonymous ROM byte array and
        # evaluate to a pointer to that data. The pointer inherits the context's
        # far-ness: a `far *u8` parameter yields a 24-bit far pointer, a near
        # `*u8` (or no context) yields a near pointer.
        is_inline = False
        is_far = False
        if context_type is None:
            is_inline = True
        elif isinstance(context_type, PointerTypeInfo):
            if (isinstance(context_type.pointee_type, BasicTypeInfo) and
                context_type.pointee_type.name == 'u8'):
                is_inline = True
                is_far = context_type.is_far

        if is_inline:
            # Process escape sequences and validate characters
            byte_values = StringValidator.process_string_to_bytes(expr.value, expr.source_loc)
            expr.processed_bytes = byte_values
            ptr_type = PointerTypeInfo(is_far=is_far, pointee_type=BasicTypeInfo('u8'))
            expr.expr_type = ptr_type
            return ptr_type

        # Validate context: string literals in array context
        if not isinstance(context_type, ArrayTypeInfo):
            raise TypeCheckError(
                f"String literal cannot be assigned to non-array type '{context_type}'",
                source_loc=expr.source_loc
            )

        elem_type = context_type.element_type
        if not isinstance(elem_type, BasicTypeInfo) or elem_type.name != 'u8':
            raise TypeCheckError(
                f"String literal can only initialize [u8; N] arrays, not [{elem_type}; N]",
                source_loc=expr.source_loc
            )

        # Process escape sequences and validate characters
        byte_values = StringValidator.process_string_to_bytes(expr.value, expr.source_loc)

        string_len = len(byte_values)
        array_size = context_type.size

        # Validate size constraints
        if string_len > array_size:
            raise TypeCheckError(
                f"String literal ({string_len} bytes) is larger than array size ({array_size})",
                source_loc=expr.source_loc
            )

        # Store processed bytes for code generation (zero-padded if shorter)
        expr.processed_bytes = byte_values

        # Return array type matching context
        array_type = ArrayTypeInfo(element_type=BasicTypeInfo('u8'), size=array_size)
        expr.expr_type = array_type
        return array_type

    @staticmethod
    def process_string_to_bytes(raw_string: str, source_loc) -> List[int]:
        """
        Process a raw string into a list of byte values.

        Decodes escapes, rejects characters above 0xFF, and checks the result
        fits in one ROM bank.

        Args:
            raw_string: Raw string value (escape sequences not yet processed)
            source_loc: Source location for error reporting

        Returns:
            List of integer byte values (0x00-0xFF)
        """
        try:
            result = decode_literal(raw_string)
        except ValueError as e:
            raise TypeCheckError(str(e), source_loc=source_loc)

        # Validate that the string fits in a single ROM bank (LoROM = 32KB)
        if len(result) > LOROM_BANK_SIZE:
            raise TypeCheckError(
                f"String literal ({len(result)} bytes) exceeds maximum bank size "
                f"({LOROM_BANK_SIZE} bytes / {LOROM_BANK_SIZE // 1024}KB)",
                source_loc=source_loc
            )

        return result
