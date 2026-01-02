"""
Register aliasing tracker for MIR.

Tracks register aliases from HIR (`let x @ A = ...`) to enable zero-cost aliasing.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Any
from r65.compiler.mir.nodes import *


@dataclass
class RegisterAlias:
    """
    Represents a register alias binding.

    Example: `let x @ A = value`
    - symbol: HIR Symbol for 'x'
    - hardware_reg: HardwareRegister('A')
    - scope_id: Lexical scope where alias is valid
    """
    symbol: Any  # HIR Symbol
    hardware_reg: HardwareRegister
    scope_id: int


class RegisterAliasTracker:
    """
    Tracks register aliasing through MIR.

    Register aliases (`let x @ A`) create zero-cost bindings where the
    symbol directly refers to a hardware register without allocating
    a virtual register.

    This is critical for matching hand-written assembly patterns where
    values are kept in hardware registers throughout a function.
    """

    def __init__(self):
        # Maps id(Symbol) → RegisterAlias (use id() since Symbol is not hashable)
        self.aliases: Dict[int, RegisterAlias] = {}

    def add_alias(self, symbol: Any, hw_reg: HardwareRegister, scope_id: int):
        """
        Register a new alias binding.

        Args:
            symbol: HIR Symbol being aliased
            hw_reg: Hardware register (A, X, Y, etc.)
            scope_id: Lexical scope ID
        """
        self.aliases[id(symbol)] = RegisterAlias(
            symbol=symbol,
            hardware_reg=hw_reg,
            scope_id=scope_id
        )

    def get_alias(self, symbol: Any) -> Optional[HardwareRegister]:
        """
        Get hardware register for aliased symbol.

        Args:
            symbol: HIR Symbol

        Returns:
            HardwareRegister if symbol is aliased, None otherwise
        """
        alias = self.aliases.get(id(symbol))
        return alias.hardware_reg if alias else None

    def is_aliased(self, symbol: Any) -> bool:
        """
        Check if symbol is aliased to a hardware register.

        Args:
            symbol: HIR Symbol

        Returns:
            True if symbol is aliased, False otherwise
        """
        return id(symbol) in self.aliases

    def remove_alias(self, symbol: Any):
        """
        Remove alias (when exiting scope).

        Args:
            symbol: HIR Symbol
        """
        symbol_id = id(symbol)
        if symbol_id in self.aliases:
            del self.aliases[symbol_id]

    def clear(self):
        """Clear all aliases (for new function)."""
        self.aliases.clear()
