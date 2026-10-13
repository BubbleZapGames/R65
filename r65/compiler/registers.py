# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
The 65816 registers a program names, and what each one is.

The one table the frontend (macro fragments), HIR (the register symbols,
bindings, `#[preserves]`), typeck and codegen read, so a register's rules are
written once.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Register:
    name: str
    types: tuple          # the types a value in it may have
    writable: bool = True
    bindable: bool = True     # a parameter or let may be bound to it (`x @ A`)
    preservable: bool = False  # `#[preserves]` may save and restore it


REGISTERS = {r.name: r for r in (
    Register('A', ('u8', 'i8', 'u16', 'i16'), preservable=True),
    Register('B', ('u8', 'i8')),            # A's high byte, m8 only
    Register('X', ('u16', 'i16'), preservable=True),
    Register('Y', ('u16', 'i16'), preservable=True),
    Register('STATUS', ('u8',), bindable=False, preservable=True),
    Register('D', ('u16',), preservable=True),
    Register('DBR', ('u8',), preservable=True),
    Register('PBR', ('u8',), writable=False),
    Register('S', ('u16',)),                 # the saves live on the stack
)}

REGISTER_NAMES = frozenset(REGISTERS)
PRESERVABLE_REGISTERS = tuple(r.name for r in REGISTERS.values() if r.preservable)
