"""Built-in lint rules registry."""

from r65.compiler.lint.rules.dead_static import DeadStaticMut
from r65.compiler.lint.rules.empty_block import EmptyBlock
from r65.compiler.lint.rules.missed_shift import MissedShift
from r65.compiler.lint.rules.unreachable import UnreachableCode
from r65.compiler.lint.rules.unused_binding import UnusedBinding
from r65.compiler.lint.rules.unused_mut import UnusedMut

BUILTIN_RULES = [
    UnusedMut,       # L001
    UnusedBinding,   # L002
    UnreachableCode, # L003
    MissedShift,     # L004
    DeadStaticMut,   # L005
    EmptyBlock,      # L006
]

__all__ = [
    "BUILTIN_RULES",
    "UnusedMut",
    "UnusedBinding",
    "UnreachableCode",
    "MissedShift",
    "DeadStaticMut",
    "EmptyBlock",
]
