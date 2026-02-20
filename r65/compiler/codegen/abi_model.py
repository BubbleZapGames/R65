"""
ABI model abstraction for the R65 compiler.

Defines selectable ABI policies that control calling convention decisions:
- Default: Traditional stack-based parameters with TSC/SBC/TCS frame allocation
- FixedStack: Zero-frame model with hw registers + scratch only, PHB-per-byte frames
"""

from enum import Enum


class ABIKind(Enum):
    DEFAULT = "Default"
    FIXED_STACK = "FixedStack"


class ABIModel:
    """Global ABI policy object controlling calling convention decisions.

    This is NOT per-function state — it's a compile-wide policy.
    Per-function ABI facts remain in ABIInfo from abi.py.
    """

    def __init__(self, kind: ABIKind):
        self.kind = kind

    def allows_stack_params(self) -> bool:
        """Whether functions may have stack-passed parameters."""
        return self.kind == ABIKind.DEFAULT

    def uses_tsc_frame(self) -> bool:
        """Whether to use TSC/SBC/TCS for large frame allocation."""
        return self.kind == ABIKind.DEFAULT

    def requires_mandatory_param_promotion(self) -> bool:
        """Whether ALL stack params must be promoted (compile error if not)."""
        return self.kind == ABIKind.FIXED_STACK

    def __repr__(self):
        return f"ABIModel({self.kind.value})"


# Singleton instances
ABI_DEFAULT = ABIModel(ABIKind.DEFAULT)
ABI_FIXED_STACK = ABIModel(ABIKind.FIXED_STACK)


def abi_model_from_string(name: str) -> ABIModel:
    """Create ABIModel from CLI string argument.

    Args:
        name: "Default" or "FixedStack"

    Returns:
        Corresponding ABIModel instance

    Raises:
        ValueError: If name is not recognized
    """
    if name == "Default":
        return ABI_DEFAULT
    elif name == "FixedStack":
        return ABI_FIXED_STACK
    else:
        raise ValueError(f"Unknown ABI model: {name!r}. Expected 'Default' or 'FixedStack'.")
