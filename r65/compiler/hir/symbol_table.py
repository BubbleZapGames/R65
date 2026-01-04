"""
Symbol table implementation for R65 HIR.

Manages hierarchical scopes and name resolution.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from enum import Enum

from r65.compiler.hir.errors import *


class SymbolKind(Enum):
    """Categories of symbols."""
    FUNCTION = "function"
    STATIC_VAR = "static_var"
    CONST = "const"
    PARAMETER = "parameter"
    LOCAL_VAR = "local_var"
    STRUCT = "struct"
    ENUM = "enum"
    TYPE_ALIAS = "type_alias"
    ENUM_VARIANT = "enum_variant"
    REGISTER = "register"  # Hardware registers (A, X, Y, etc.)
    BUILTIN_FUNC = "builtin_func"  # Built-in functions (SEP, REP, etc.)


@dataclass
class Symbol:
    """Represents a named entity in the program."""
    name: str
    kind: SymbolKind
    definition: Optional[Any]  # Points to defining HIR node (None for built-ins)
    scope_id: int

    # For variables/parameters
    var_type: Optional[Any] = None  # Will be TypeInfo
    is_mutable: bool = False

    # For functions
    func_signature: Optional[Any] = None  # Will be function signature info

    # For constants
    const_value: Optional[Any] = None

    # For types (structs, enums, type aliases)
    type_info: Optional[Any] = None  # Will be TypeInfo


class ScopeKind(Enum):
    """Types of scopes."""
    GLOBAL = "global"
    FUNCTION = "function"
    BLOCK = "block"


@dataclass
class Scope:
    """Represents a lexical scope."""
    scope_id: int
    parent_id: Optional[int]  # None for global scope
    symbols: Dict[str, Symbol] = field(default_factory=dict)
    kind: ScopeKind = ScopeKind.BLOCK


class SymbolTable:
    """Global symbol table with hierarchical scopes."""

    def __init__(self):
        self.scopes: Dict[int, Scope] = {}
        self.next_scope_id = 0
        self.current_scope_id = 0

        # Initialize global scope with built-ins
        self._init_global_scope()

    def _init_global_scope(self):
        """Initialize global scope with hardware registers and built-in functions."""
        global_scope = Scope(
            scope_id=0,
            parent_id=None,
            symbols={},
            kind=ScopeKind.GLOBAL
        )

        # Add hardware registers
        # A, X, Y, B are mode-dependent; STATUS, D, DBR, PBR, S are fixed
        # B is only valid in m8 mode (type checker enforces this)
        for reg_name in ['A', 'X', 'Y', 'B', 'STATUS', 'D', 'DBR', 'PBR', 'S']:
            global_scope.symbols[reg_name] = Symbol(
                name=reg_name,
                kind=SymbolKind.REGISTER,
                definition=None,  # Built-in
                scope_id=0,
                is_mutable=(reg_name != 'PBR')  # PBR is read-only
            )

        # Add built-in functions
        builtin_funcs = [
            'SEP',   # Set processor status bits
            'REP',   # Reset processor status bits
            'mvn',   # Block move next (forward)
            'mvp',   # Block move previous (backward)
            'wai',   # Wait for interrupt
            'stp',   # Stop processor
            'mul',   # Multiply (general)
            'div',   # Divide (general)
            'mod',   # Modulo
            'shl',   # Shift left (variable)
            'shr',   # Shift right (variable)
        ]

        for func_name in builtin_funcs:
            global_scope.symbols[func_name] = Symbol(
                name=func_name,
                kind=SymbolKind.BUILTIN_FUNC,
                definition=None,  # Built-in
                scope_id=0
            )

        self.scopes[0] = global_scope
        self.next_scope_id = 1

    def enter_scope(self, kind: ScopeKind) -> int:
        """Create and enter a new scope. Returns the new scope ID."""
        new_scope = Scope(
            scope_id=self.next_scope_id,
            parent_id=self.current_scope_id,
            symbols={},
            kind=kind
        )
        self.scopes[self.next_scope_id] = new_scope
        self.current_scope_id = self.next_scope_id
        self.next_scope_id += 1
        return self.current_scope_id

    def exit_scope(self):
        """Exit current scope and return to parent."""
        current = self.scopes[self.current_scope_id]
        if current.parent_id is not None:
            self.current_scope_id = current.parent_id
        else:
            raise HIRError("Cannot exit global scope")

    def declare(self, name: str, symbol: Symbol):
        """Declare a symbol in the current scope."""
        current = self.scopes[self.current_scope_id]
        if name in current.symbols:
            raise HIRError(f"Redefinition of '{name}' in current scope")
        current.symbols[name] = symbol

    def lookup(self, name: str) -> Optional[Symbol]:
        """Look up a symbol, walking up the scope chain."""
        scope_id = self.current_scope_id
        while scope_id is not None:
            scope = self.scopes[scope_id]
            if name in scope.symbols:
                return scope.symbols[name]
            scope_id = scope.parent_id
        return None

    def lookup_current_scope(self, name: str) -> Optional[Symbol]:
        """Look up symbol only in the current scope (no parent search)."""
        return self.scopes[self.current_scope_id].symbols.get(name)

    def get_scope(self, scope_id: int) -> Scope:
        """Get a scope by ID."""
        return self.scopes.get(scope_id)

    def get_current_scope(self) -> Scope:
        """Get the current scope."""
        return self.scopes[self.current_scope_id]
