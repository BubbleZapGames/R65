# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
A built-in call takes the argument count its registry entry declares.

Nothing read the registry's counts, so `rotate_left(O)` reached codegen and
crashed there on the missing argument.
"""

import pytest

from r65.compiler.frontend import Parser
from r65.compiler.hir import HIRBuilder
from r65.compiler.typeck import TypeChecker


def check(body: str):
    source = "#[zeropage(0x10)] static mut O: u8;\nfn main() { " + body + " }"
    TypeChecker(HIRBuilder().build_program(Parser().parse(source))).check()


@pytest.mark.parametrize("body, message", [
    ("O = rotate_left(O);", "rotate_left() takes 2 argument(s), got 1"),
    ("NOP(1, 2);", "NOP() takes 0 to 1 argument(s), got 2"),
])
def test_wrong_count_rejected(body, message):
    with pytest.raises(Exception) as exc:
        check(body)
    assert message in str(exc.value)


@pytest.mark.parametrize("body", ["NOP();", "NOP(3);", "O = rotate_left(O, 1);"])
def test_declared_count_accepted(body):
    check(body)
