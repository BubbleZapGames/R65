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
    def infer_integer_literal_type(value: int, context_type: Optional[TypeInfo]) -> TypeInfo:
        """
        Infer type for integer literal from context.

        If context provides a type, use it (if value fits).
        Otherwise, default to smallest type that fits.
        """
        if context_type and isinstance(context_type, BasicTypeInfo):
            # Check if value fits in context type
            if context_type.name == 'u8' and 0 <= value <= 255:
                return context_type
            elif context_type.name == 'i8' and -128 <= value <= 127:
                return context_type
            elif context_type.name == 'u16' and 0 <= value <= 65535:
                return context_type
            elif context_type.name == 'i16' and -32768 <= value <= 32767:
                return context_type

        # Default inference: smallest unsigned type
        if 0 <= value <= 255:
            return BasicTypeInfo('u8')
        elif 0 <= value <= 65535:
            return BasicTypeInfo('u16')
        else:
            # Value too large - error will be caught later
            return BasicTypeInfo('u16')
