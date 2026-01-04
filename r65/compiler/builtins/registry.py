"""
Built-in function registry.

Defines all compiler built-in functions and their properties.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional, List


class BuiltinKind(Enum):
    """Categories of built-in functions."""
    PROCESSOR_CONTROL = "processor_control"  # wai, stp, NOP, xba
    MODE_CONTROL = "mode_control"            # SEP, REP
    BLOCK_MOVE = "block_move"                # mvn, mvp
    ARITHMETIC = "arithmetic"                 # mul, div, mod
    SHIFT = "shift"                           # shl, shr
    ROTATE = "rotate"                         # rotate_left, rotate_right


@dataclass
class BuiltinSignature:
    """Signature for a built-in function."""
    name: str
    kind: BuiltinKind
    param_count: int
    returns_value: bool
    description: str
    instruction: Optional[str] = None  # Direct 65816 instruction if applicable
    max_param_count: Optional[int] = None  # Maximum params (if different from param_count)


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
        'NOP': BuiltinSignature(
            name='NOP',
            kind=BuiltinKind.PROCESSOR_CONTROL,
            param_count=0,
            max_param_count=1,
            returns_value=False,
            description='No operation (optionally repeated)',
            instruction='NOP'
        ),
        'xba': BuiltinSignature(
            name='xba',
            kind=BuiltinKind.PROCESSOR_CONTROL,
            param_count=0,
            returns_value=False,
            description='Exchange B and A registers (swap high/low bytes)',
            instruction='XBA'
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

        # Rotate operations (2 parameters: value and constant count 1-8, returns value)
        'rotate_left': BuiltinSignature(
            name='rotate_left',
            kind=BuiltinKind.ROTATE,
            param_count=2,
            returns_value=True,
            description='Rotate left (constant count 1-8)',
            instruction='ROL'
        ),
        'rotate_right': BuiltinSignature(
            name='rotate_right',
            kind=BuiltinKind.ROTATE,
            param_count=2,
            returns_value=True,
            description='Rotate right (constant count 1-8)',
            instruction='ROR'
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

        # Check if argument count is in valid range
        min_params = builtin.param_count
        max_params = builtin.max_param_count if builtin.max_param_count is not None else builtin.param_count

        if arg_count < min_params or arg_count > max_params:
            if min_params == max_params:
                return False, f"{name}() takes {min_params} argument(s), got {arg_count}"
            else:
                return False, f"{name}() takes {min_params}-{max_params} argument(s), got {arg_count}"

        return True, None
