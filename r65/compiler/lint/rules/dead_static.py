"""L005: dead_static_mut — warn on `static mut` that is never read."""

from r65.compiler.hir import HIRIdentifier, HIRStaticDecl
from r65.compiler.hir.attributes import StorageKind
from r65.compiler.lint.rule import LintContext, LintRule


class DeadStaticMut(LintRule):
    """Warn when a ``static mut`` is written but never read (or totally
    unused). On a 128KB ROM target, dead RAM is worth catching early.

    Skips ``#[hw]`` statics: hardware registers are memory-mapped I/O, not
    storage, so "written but never read" is the common legitimate case
    (writing a PPU control register never has a corresponding read).

    Collection happens during the walk (``visit_static_decl`` records each
    mutable static, ``visit_identifier`` records reads of any symbol) and the
    emit phase runs in ``finalize`` after every function body has been seen.
    """

    def __init__(self):
        super().__init__(
            code="L005",
            name="dead_static_mut",
            description="`static mut` is never read",
        )
        self._declared: dict = {}  # id(symbol) -> (name, source_loc)
        self._read: set = set()     # id(symbol)

    def visit_static_decl(self, decl: HIRStaticDecl, ctx: LintContext) -> None:
        if not decl.is_mutable or decl.symbol is None:
            return
        if decl.storage_attr is not None and decl.storage_attr.storage_kind == StorageKind.HW:
            return  # Hardware registers are not storage.
        self._declared[id(decl.symbol)] = (decl.name, decl.source_loc)

    def visit_identifier(self, expr: HIRIdentifier, ctx: LintContext) -> None:
        if expr.symbol is not None:
            self._read.add(id(expr.symbol))

    def finalize(self, ctx: LintContext) -> None:
        for sym_id, (name, loc) in self._declared.items():
            if sym_id not in self._read:
                ctx.emit(
                    code=self.code,
                    message=f"`static mut {name}` is written but never read",
                    source_loc=loc,
                    hint="remove the declaration or read the value somewhere",
                )
        self._declared = {}
        self._read = set()
