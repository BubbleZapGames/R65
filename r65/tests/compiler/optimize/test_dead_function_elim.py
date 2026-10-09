# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""Dead-function elimination keeps a #[used] function nothing calls.

A test harness reaches such a function from outside the program - a
debugger's `jump` - so the compiler cannot see the call.
"""

import pytest

from r65.compiler.hir import HIRError
from r65.compiler.main import compile_string


def _compile(src: str) -> str:
    return compile_string(src + "\n#[entry]\nfn main() { }\n", "t.r65")


def test_uncalled_function_is_dropped():
    asm = _compile("fn spare() -> u8 { return 5; }")
    assert "\nspare:" not in asm


def test_used_function_is_kept():
    asm = _compile("#[used]\nfn spare() -> u8 { return 5; }")
    assert "\nspare:" in asm


@pytest.mark.parametrize("src", [
    "#[used(1)]\nfn spare() { }",
    "#[used]\n#[ram] static mut SPARE: u8;",
])
def test_used_misplaced(src):
    with pytest.raises(HIRError, match=r"#\[used\]"):
        _compile(src)
