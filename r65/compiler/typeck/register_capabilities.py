"""
Register capability definitions for R65 compiler.

Defines which operators are valid for each 65816 hardware register based on
actual CPU instruction availability.
"""

# Register capabilities based on 65816 hardware instructions
#
# A register: Full ALU support
#   - ADC/SBC for add/subtract
#   - AND/ORA/EOR for bitwise ops
#   - ASL/LSR for shifts
#   - INC A/DEC A for increment/decrement
#
# X/Y registers: Limited operations
#   - INX/INY for increment
#   - DEX/DEY for decrement
#   - CPX/CPY for comparison
#   - Load immediate (LDX/LDY)
#   - Transfer (TAX/TAY/TXA/TYA/TXY/TYX)
#
# B register: No direct operations
#   - Accessed via XBA to swap with A
#   - No arithmetic or bitwise ops

REGISTER_CAPABILITIES = {
    'A': {
        'binary_ops': {'+', '-', '&', '|', '^', '<<', '>>'},
        'inc': True,
        'dec': True,
    },
    'X': {
        'binary_ops': set(),  # No binary ops - no ADD/SUB/AND/etc. instructions
        'inc': True,
        'dec': True,
    },
    'Y': {
        'binary_ops': set(),  # No binary ops - no ADD/SUB/AND/etc. instructions
        'inc': True,
        'dec': True,
    },
    'B': {
        'binary_ops': set(),  # Accessed via XBA swap only
        'inc': False,
        'dec': False,
    },
}

# Registers that are always valid (STATUS, D, DBR, PBR, S)
# These are control registers without arithmetic restrictions
UNRESTRICTED_REGISTERS = {'STATUS', 'D', 'DBR', 'PBR', 'S'}

# Index registers (X and Y) - these can only be compared against values,
# not against each other (no direct X vs Y comparison instruction)
INDEX_REGISTERS = {'X', 'Y'}


def get_register_capabilities(register_name: str) -> dict:
    """
    Get the capabilities for a register.

    Args:
        register_name: Name of the register (A, X, Y, B)

    Returns:
        Dict with 'binary_ops', 'inc', 'dec' keys, or None for unrestricted
    """
    if register_name in UNRESTRICTED_REGISTERS:
        return None  # No restrictions
    return REGISTER_CAPABILITIES.get(register_name)


def can_register_do_binary_op(register_name: str, op: str) -> bool:
    """
    Check if a register supports a binary operator.

    Args:
        register_name: Name of the register
        op: Binary operator (+, -, &, |, ^, <<, >>)

    Returns:
        True if the operation is supported
    """
    caps = get_register_capabilities(register_name)
    if caps is None:
        return True  # Unrestricted
    return op in caps['binary_ops']


def can_register_increment(register_name: str) -> bool:
    """Check if a register supports increment (++)."""
    caps = get_register_capabilities(register_name)
    if caps is None:
        return True
    return caps['inc']


def can_register_decrement(register_name: str) -> bool:
    """Check if a register supports decrement (--)."""
    caps = get_register_capabilities(register_name)
    if caps is None:
        return True
    return caps['dec']


def get_register_hint(register_name: str) -> str:
    """
    Get a helpful hint message for register restrictions.

    Args:
        register_name: Name of the register

    Returns:
        Hint string for error messages
    """
    if register_name in ('X', 'Y'):
        return (f"{register_name} only supports increment (++), decrement (--), "
                f"comparison, load, and transfer operations")
    elif register_name == 'B':
        return "B register is accessed via XBA swap with A; perform operations on A instead"
    return ""


def is_index_register(register_name: str) -> bool:
    """
    Check if a register is an index register (X or Y).

    Index registers cannot be directly compared against each other
    because there's no CPX Y or CPY X instruction.

    Args:
        register_name: Name of the register

    Returns:
        True if register is X or Y
    """
    return register_name in INDEX_REGISTERS
