# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Type inference for R65.

Limited inference:
- Register aliases infer from current register type
- Integer literals infer from context
"""

from typing import Optional
from r65.compiler.hir import *


class TypeInference:
    """Limited type inference engine."""

    @staticmethod
    def infer_register_alias_type(register_name: str, mode: 'ProcessorMode') -> Optional[TypeInfo]:
        """
        Infer type for register alias binding.

        Example: let x @ A = ... → infer x is u8/u16 based on mode
        """
        return mode.get_register_type(register_name)

    @staticmethod
    def infer_integer_literal_type(value: int, context_type: Optional[TypeInfo], suffix: Optional[str] = None) -> TypeInfo:
        """
        Infer type for integer literal from context or suffix.

        Priority: suffix > context > default inference.
        If suffix is present, use it directly (validation catches overflow).
        If context provides a type, use it (if value fits).
        Otherwise, default to smallest type that fits.
        """
        from r65.compiler.typeck.type_utils import value_fits_type

        # Suffix takes priority - the programmer explicitly specified the type
        if suffix:
            return BasicTypeInfo(suffix)

        # If context provides a type, use it if value fits
        if context_type and isinstance(context_type, BasicTypeInfo):
            if value_fits_type(value, context_type.name):
                return context_type

        # Default inference: smallest type that fits the value
        # Positive values prefer unsigned types (u8, u16)
        # Negative values prefer signed types (i8, i16)
        if value >= 0:
            if value_fits_type(value, 'u8'):
                return BasicTypeInfo('u8')
            if value_fits_type(value, 'u16'):
                return BasicTypeInfo('u16')
        else:
            if value_fits_type(value, 'i8'):
                return BasicTypeInfo('i8')
            if value_fits_type(value, 'i16'):
                return BasicTypeInfo('i16')

        # Value too large - error will be caught later
        return BasicTypeInfo('u16')
