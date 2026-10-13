# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
The order call-argument setup loads arguments in.

Shared by call lowering (`CallSelector`) and the inliner, which emits the same
register setup in place of a call it removes, so both load A last.
"""

from r65.compiler.mir.nodes import ArgumentMechanism


def register_arg_rank(register_name: str) -> int:
    """Order among register arguments: B, then the other registers, then A.

    Every non-A load goes through A (B via XBA, DBR via PHA/PLB, D via TCD, a
    scratch or variable store via STA), so A is loaded after all of them.
    """
    if register_name == 'B':
        return 3
    if register_name == 'A':
        return 5
    return 4


def arg_sort_key(arg) -> int:
    """Sort key for argument setup: stack, scratch, variable, then registers.

    Docs: docs/abi-models.md, "Argument Setup Preserves an A-Resident Value".
    """
    if arg.mechanism == ArgumentMechanism.STACK:
        return 0
    if arg.mechanism == ArgumentMechanism.SCRATCH_PARAM:
        return 1
    if arg.mechanism == ArgumentMechanism.VARIABLE:
        return 2
    if arg.mechanism == ArgumentMechanism.REGISTER:
        return register_arg_rank(getattr(arg.location, 'name', str(arg.location)))
    return 6
