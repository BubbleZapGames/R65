"""
Error message helpers for code generation.

Provides standardized error message formatting functions for
consistent error messages across the codegen module.
"""

from r65.compiler.errors import InstructionSelectionError


def unsupported_operation(operation: str, context: str = None, source_loc=None) -> InstructionSelectionError:
    """Create error for unsupported operations."""
    if context:
        return InstructionSelectionError(f"Unsupported {operation}: {context}", source_loc=source_loc)
    return InstructionSelectionError(f"Unsupported operation: {operation}", source_loc=source_loc)


def unsupported_addressing_mode(mnemonic: str, mode: str, source_loc=None) -> InstructionSelectionError:
    """Create error for unsupported addressing modes."""
    return InstructionSelectionError(f"{mnemonic} does not support {mode} addressing", source_loc=source_loc)


def cannot_use_register(action: str, register: str, source_loc=None) -> InstructionSelectionError:
    """Create error for register usage issues."""
    return InstructionSelectionError(f"Cannot {action}: {register}", source_loc=source_loc)


def unknown_value(category: str, value: str, source_loc=None) -> InstructionSelectionError:
    """Create error for unknown/unrecognized values."""
    return InstructionSelectionError(f"Unknown {category}: {value}", source_loc=source_loc)


def requires_constant(operation: str, source_loc=None) -> InstructionSelectionError:
    """Create error when operation requires a constant operand."""
    return InstructionSelectionError(f"{operation} requires constant operand", source_loc=source_loc)


def invalid_location(description: str, location, source_loc=None) -> InstructionSelectionError:
    """Create error for invalid memory/register locations."""
    return InstructionSelectionError(f"{description}: {location}", source_loc=source_loc)


def missing_allocation(symbol_name: str, source_loc=None) -> InstructionSelectionError:
    """Create error when a symbol has no memory allocation."""
    return InstructionSelectionError(f"No allocation for symbol: {symbol_name}", source_loc=source_loc)


def argument_count_error(func_name: str, expected: int, got: int, source_loc=None) -> InstructionSelectionError:
    """Create error for wrong number of arguments."""
    return InstructionSelectionError(
        f"{func_name}() expects {expected} argument{'s' if expected != 1 else ''}, "
        f"got {got}",
        source_loc=source_loc
    )
