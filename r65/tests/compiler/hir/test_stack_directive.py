# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
`#[stack(lower, upper)]` names a region of low RAM, lowest address first.

The checks lived on an attribute path the grammar never reaches (`stack(`
always lexes as the directive), so the directive took any range.
"""

import pytest

from r65.compiler.frontend import parse
from r65.compiler.hir import HIRBuilder


def build(stack: str):
    source = f"#[stack({stack})]\n#[entry]\nfn main() {{ }}"
    return HIRBuilder().build_program(parse(source, "t.r65"))


@pytest.mark.parametrize("stack, message", [
    ("0x1F00, 0x2100", "within low RAM"),
    ("0x1FFF, 0x1F00", "must be <= upper bound"),
])
def test_rejected(stack, message):
    with pytest.raises(Exception) as exc:
        build(stack)
    assert message in str(exc.value)


def test_accepted():
    assert build("0x1F00, 0x1FFF").stack_attr.upper == 0x1FFF
