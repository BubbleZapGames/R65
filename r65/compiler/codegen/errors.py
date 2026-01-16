"""
Error message helpers for code generation.

Provides standardized error message formatting functions for
consistent error messages across the codegen module.
"""

from r65.compiler.errors import InstructionSelectionError


def unsupported_operation(operation: str, context: str = None) -> InstructionSelectionError:
    """Create error for unsupported operations."""
    if context:
        return InstructionSelectionError(f"Unsupported {operation}: {context}")
    return InstructionSelectionError(f"Unsupported operation: {operation}")


def unsupported_addressing_mode(mnemonic: str, mode: str) -> InstructionSelectionError:
    """Create error for unsupported addressing modes."""
    return InstructionSelectionError(f"{mnemonic} does not support {mode} addressing")


def cannot_use_register(action: str, register: str) -> InstructionSelectionError:
    """Create error for register usage issues."""
    return InstructionSelectionError(f"Cannot {action}: {register}")


def unknown_value(category: str, value: str) -> InstructionSelectionError:
    """Create error for unknown/unrecognized values."""
    return InstructionSelectionError(f"Unknown {category}: {value}")


def requires_constant(operation: str) -> InstructionSelectionError:
    """Create error when operation requires a constant operand."""
    return InstructionSelectionError(f"{operation} requires constant operand")


def invalid_location(description: str, location) -> InstructionSelectionError:
    """Create error for invalid memory/register locations."""
    return InstructionSelectionError(f"{description}: {location}")


def missing_allocation(symbol_name: str) -> InstructionSelectionError:
    """Create error when a symbol has no memory allocation."""
    return InstructionSelectionError(f"No allocation for symbol: {symbol_name}")


def argument_count_error(func_name: str, expected: int, got: int) -> InstructionSelectionError:
    """Create error for wrong number of arguments."""
    return InstructionSelectionError(
        f"{func_name}() expects {expected} argument{'s' if expected != 1 else ''}, "
        f"got {got}"
    )
