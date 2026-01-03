"""
Built-in function registry.

Defines all compiler built-in functions and their properties.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional, List


class BuiltinKind(Enum):
    """Categories of built-in functions."""
    PROCESSOR_CONTROL = "processor_control"  # wai, stp
    MODE_CONTROL = "mode_control"            # SEP, REP
    BLOCK_MOVE = "block_move"                # mvn, mvp
    ARITHMETIC = "arithmetic"                 # mul, div, mod
    SHIFT = "shift"                           # shl, shr


@dataclass
class BuiltinSignature:
    """Signature for a built-in function."""
    name: str
    kind: BuiltinKind
    param_count: int
    returns_value: bool
    description: str
    instruction: Optional[str] = None  # Direct 65816 instruction if applicable


class BuiltinRegistry:
    """
    Registry of all built-in functions.

    Provides lookup and validation for compiler built-ins.
    """

    # Built-in function definitions
    BUILTINS = {
        # Processor control (no parameters, no return value)
        'wai': BuiltinSignature(
            name='wai',
            kind=BuiltinKind.PROCESSOR_CONTROL,
            param_count=0,
            returns_value=False,
            description='Wait for interrupt',
            instruction='WAI'
        ),
        'stp': BuiltinSignature(
            name='stp',
            kind=BuiltinKind.PROCESSOR_CONTROL,
            param_count=0,
            returns_value=False,
            description='Stop processor',
            instruction='STP'
        ),

        # Mode control (1 parameter: flags, no return value)
        'SEP': BuiltinSignature(
            name='SEP',
            kind=BuiltinKind.MODE_CONTROL,
            param_count=1,
            returns_value=False,
            description='Set processor status bits',
            instruction='SEP'
        ),
        'REP': BuiltinSignature(
            name='REP',
            kind=BuiltinKind.MODE_CONTROL,
            param_count=1,
            returns_value=False,
            description='Reset processor status bits',
            instruction='REP'
        ),

        # Block moves (2 parameters: src_bank, dst_bank, no return value)
        'mvn': BuiltinSignature(
            name='mvn',
            kind=BuiltinKind.BLOCK_MOVE,
            param_count=2,
            returns_value=False,
            description='Move memory forward (MVN instruction)',
            instruction='MVN'
        ),
        'mvp': BuiltinSignature(
            name='mvp',
            kind=BuiltinKind.BLOCK_MOVE,
            param_count=2,
            returns_value=False,
            description='Move memory backward (MVP instruction)',
            instruction='MVP'
        ),

        # Arithmetic operations (2 parameters, returns value)
        'mul': BuiltinSignature(
            name='mul',
            kind=BuiltinKind.ARITHMETIC,
            param_count=2,
            returns_value=True,
            description='General multiplication (not power of 2)'
        ),
        'div': BuiltinSignature(
            name='div',
            kind=BuiltinKind.ARITHMETIC,
            param_count=2,
            returns_value=True,
            description='General division'
        ),
        'mod': BuiltinSignature(
            name='mod',
            kind=BuiltinKind.ARITHMETIC,
            param_count=2,
            returns_value=True,
            description='Modulo operation'
        ),

        # Shift operations (2 parameters, returns value)
        'shl': BuiltinSignature(
            name='shl',
            kind=BuiltinKind.SHIFT,
            param_count=2,
            returns_value=True,
            description='Variable left shift'
        ),
        'shr': BuiltinSignature(
            name='shr',
            kind=BuiltinKind.SHIFT,
            param_count=2,
            returns_value=True,
            description='Variable right shift'
        ),
    }

    @classmethod
    def is_builtin(cls, name: str) -> bool:
        """Check if a name is a built-in function."""
        return name in cls.BUILTINS

    @classmethod
    def get_builtin(cls, name: str) -> Optional[BuiltinSignature]:
        """Get built-in signature by name."""
        return cls.BUILTINS.get(name)

    @classmethod
    def get_all_names(cls) -> List[str]:
        """Get list of all built-in function names."""
        return list(cls.BUILTINS.keys())

    @classmethod
    def validate_call(cls, name: str, arg_count: int) -> tuple[bool, Optional[str]]:
        """
        Validate a built-in function call.

        Args:
            name: Function name
            arg_count: Number of arguments provided

        Returns:
            (is_valid, error_message)
        """
        builtin = cls.get_builtin(name)
        if not builtin:
            return False, f"Unknown built-in function: {name}"

        if arg_count != builtin.param_count:
            return False, f"{name}() takes {builtin.param_count} argument(s), got {arg_count}"

        return True, None
