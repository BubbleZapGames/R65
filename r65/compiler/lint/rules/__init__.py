"""Built-in lint rules registry."""

from r65.compiler.lint.rules.dead_static import DeadStaticMut
from r65.compiler.lint.rules.empty_block import EmptyBlock
from r65.compiler.lint.rules.missed_shift import MissedShift
from r65.compiler.lint.rules.redundant_cast import RedundantCast
from r65.compiler.lint.rules.unreachable import UnreachableCode
from r65.compiler.lint.rules.unused_binding import UnusedBinding
from r65.compiler.lint.rules.unused_mut import UnusedMut
from r65.compiler.lint.rules.unused_param import UnusedParam
from r65.compiler.lint.rules.xy16_mode import Xy16Mode

BUILTIN_RULES = [
    UnusedMut,       # L001
    UnusedBinding,   # L002
    UnreachableCode, # L003
    MissedShift,     # L004
    DeadStaticMut,   # L005
    EmptyBlock,      # L006
    RedundantCast,   # L007
    UnusedParam,     # L008
    Xy16Mode,        # L009
]

__all__ = [
    "BUILTIN_RULES",
    "UnusedMut",
    "UnusedBinding",
    "UnreachableCode",
    "MissedShift",
    "DeadStaticMut",
    "EmptyBlock",
    "RedundantCast",
    "UnusedParam",
    "Xy16Mode",
]
