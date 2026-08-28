"""
Rule kind: ``power_of_two_arrays``.

Flags arrays whose element **stride** or **length** is not a power of two.

Two different costs, so the rule checks whichever the project cares about:

``stride``
    ``arr[i]`` has to scale ``i`` by the element size. A power-of-two stride is
    a run of ``ASL``; anything else is a shift-and-add chain, and a large odd
    stride is a full multiply — the ~57 cycles that motivated ``#[soa]``
    (see docs/soa.md).

``length``
    A power-of-two length lets a ring index wrap with ``i = (i + 1) & (N - 1)``
    — one ``AND`` — instead of a compare and a branch.

    [[rule]]
    code = "C030"
    kind = "power_of_two_arrays"
    message = "array is not a power of two"
    check        = "stride"   # "stride" | "length" | "both"; default "stride"
    min_elements = 2          # optional: ignore arrays this short (default 2)

The diagnostic appends which quantity was measured (``'POOL' stride is 3``), so
with ``check = "both"`` keep `message` neutral — one that names "stride" reads
wrong on a length finding.

Only `static` arrays are examined. A one-element array is never indexed by a
runtime value, which is why `min_elements` defaults to 2.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from r65.compiler.hir import HIRStaticDecl
from r65.compiler.hir.types import ArrayTypeInfo
from r65.compiler.lint.rule import LintContext, LintRule
from r65.compiler.lint.rule_kinds._common import parse_severity


KIND_NAME = "power_of_two_arrays"

_CHECKS = ("stride", "length", "both")


def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _next_power_of_two(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


class PowerOfTwoArrays(LintRule):
    def __init__(
        self,
        code: str,
        message: str,
        check: str = "stride",
        min_elements: int = 2,
        severity_name: str = "warning",
        hint: Optional[str] = None,
    ):
        if check not in _CHECKS:
            raise ValueError(
                f"rule kind '{KIND_NAME}': `check` must be one of "
                f"{', '.join(_CHECKS)} (got {check!r})"
            )
        if min_elements < 0:
            raise ValueError(
                f"rule kind '{KIND_NAME}': `min_elements` must be >= 0"
            )
        super().__init__(code=code, name=KIND_NAME, description=message)
        self.message = message
        self.check = check
        self.min_elements = min_elements
        self.severity = parse_severity(severity_name)
        self.custom_hint = hint

    def setup(self, ctx: LintContext) -> None:
        for decl in ctx.program.declarations:
            if not isinstance(decl, HIRStaticDecl):
                continue
            # A planed parent (`#[soa]` struct or tagged enum) holds no bytes;
            # its planes are separate declarations and are checked on their own.
            if getattr(decl, "soa_planes", None) is not None:
                continue
            var_type = decl.var_type
            if not isinstance(var_type, ArrayTypeInfo):
                continue

            length = var_type.size
            if length < self.min_elements:
                continue

            if self.check in ("stride", "both"):
                stride = self._stride(decl, var_type)
                if stride is not None and not _is_power_of_two(stride):
                    self._report(ctx, decl, "stride", stride)

            if self.check in ("length", "both"):
                if not _is_power_of_two(length):
                    self._report(ctx, decl, "length", length)

    @staticmethod
    def _stride(decl: HIRStaticDecl, var_type: ArrayTypeInfo) -> Optional[int]:
        """Bytes per element, or None if the element has no computable size.

        A field plane's `soa_stride` is authoritative: it is already padded, so
        reading the element type instead would report a stride the emitted code
        never uses.
        """
        stride = getattr(decl, "soa_stride", None)
        if stride is not None:
            return stride
        from r65.compiler.hir.errors import HIRError
        try:
            return var_type.element_type.size_bytes
        except (AttributeError, NotImplementedError, HIRError):
            # An unsized element (a bare trait) or one whose definition never
            # resolved: not this rule's business to diagnose.
            return None

    def _report(self, ctx: LintContext, decl: HIRStaticDecl, what: str, value: int) -> None:
        nxt = _next_power_of_two(value)
        if what == "stride":
            default_hint = (
                f"indexing scales by {value}, which needs a shift-and-add; "
                f"padding the element to {nxt} bytes makes it "
                f"{nxt.bit_length() - 1} x ASL"
            )
        else:
            default_hint = (
                f"a length of {nxt} lets an index wrap with "
                f"`i = (i + 1) & {nxt - 1}` instead of a compare and branch"
            )
        ctx.emit(
            code=self.code,
            message=f"{self.message}: '{decl.name}' {what} is {value}",
            source_loc=decl.source_loc,
            hint=self.custom_hint or default_hint,
            severity=self.severity,
        )


def from_config(spec: Dict[str, Any]) -> PowerOfTwoArrays:
    code = spec["code"]
    message = spec["message"]

    check = spec.get("check", "stride")
    if not isinstance(check, str):
        raise ValueError(f"rule kind '{KIND_NAME}': `check` must be a string")

    min_elements = spec.get("min_elements", 2)
    if not isinstance(min_elements, int) or isinstance(min_elements, bool):
        raise ValueError(
            f"rule kind '{KIND_NAME}': `min_elements` must be an integer"
        )

    return PowerOfTwoArrays(
        code=code,
        message=message,
        check=check,
        min_elements=min_elements,
        severity_name=spec.get("severity", "warning"),
        hint=spec.get("hint"),
    )
