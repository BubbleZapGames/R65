# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Per-item lint suppression via `#[allow(...)]`.

Rust-style, lexically scoped: `#[allow(C001)]` on an item suppresses that lint
code for diagnostics arising inside the item (a function's body, a static's
decl), but not transitively through the functions it calls. `#[allow(all)]`
suppresses every code. Codes are not validated, matching Rust's leniency.
"""

import pytest

from r65.compiler.frontend.parser import parse
from r65.compiler.frontend.macros import expand_macros
from r65.compiler.hir.builder import HIRBuilder
from r65.compiler.typeck import TypeChecker
from r65.compiler.errors import HIRError
from r65.compiler.lint.config import LintConfig
from r65.compiler.lint.linter import run_lint
from r65.compiler.lint.rule_kinds.reachability_forbidden_access import from_config


def _hir(source: str):
    program = parse(source, '<test>')
    program = expand_macros(program)
    hir = HIRBuilder().build_program(program)
    TypeChecker(hir).check()
    return hir


def _c001(source: str):
    spec = {
        "code": "C001",
        "kind": "reachability_forbidden_access",
        "message": "PPU register write unsafe during active rendering",
        "entry_points": ["main"],
        "forbid_addrs": [0x2122],
    }
    config = LintConfig(enabled_codes={"C001"}, custom_rules=[from_config(spec)])
    diags = run_lint(_hir(source), config)
    src_lines = source.split('\n')
    return [src_lines[d.source_loc.line - 1].strip()
            for d in diags.diagnostics if d.code == "C001"]


HW = '#[hw(0x2122)]\nstatic mut CGDATA: u8;\n'


def _worker(decorator: str) -> str:
    return HW + f'''
{decorator}fn worker() {{
    loop {{
        CGDATA = 1;
    }}
}}
#[entry]
fn main() {{
    loop {{ worker(); }}
}}
'''


def test_no_allow_flags_the_write():
    assert _c001(_worker("")) == ["CGDATA = 1;"]


def test_allow_matching_code_suppresses():
    assert _c001(_worker("#[allow(C001)]\n")) == []


def test_allow_all_suppresses():
    assert _c001(_worker("#[allow(all)]\n")) == []


def test_allow_list_with_matching_code_suppresses():
    assert _c001(_worker("#[allow(L001, C001)]\n")) == []


def test_allow_unrelated_code_does_not_suppress():
    assert _c001(_worker("#[allow(L001)]\n")) == ["CGDATA = 1;"]


def test_allow_is_lexical_not_transitive():
    """`#[allow]` on a caller does not silence a callee's own writes."""
    flagged = _c001(HW + '''
fn helper() {
    CGDATA = 9;
}
#[allow(C001)]
fn worker() {
    loop { helper(); }
}
#[entry]
fn main() { loop { worker(); } }
''')
    assert flagged == ["CGDATA = 9;"]


def test_allow_does_not_leak_to_siblings():
    flagged = _c001(HW + '''
#[allow(C001)]
fn a() {
    loop {
        CGDATA = 1;
    }
}
fn b() {
    loop {
        CGDATA = 2;
    }
}
#[entry]
fn main() { loop { a(); b(); } }
''')
    assert flagged == ["CGDATA = 2;"]


def test_allow_requires_an_argument():
    with pytest.raises(HIRError, match="requires at least one lint code"):
        _hir("#[allow]\nfn f() {}")


def test_allow_rejects_named_arguments():
    with pytest.raises(HIRError, match="does not accept named arguments"):
        _hir("#[allow(code=C001)]\nfn f() {}")


def test_allow_on_static_suppresses_its_own_diagnostic():
    """A static may carry #[allow]; it suppresses diagnostics on that decl.

    Uses a naming_convention rule, which fires on the declaration itself."""
    from r65.compiler.lint.rule_kinds.naming_convention import from_config as nc_from_config
    spec = {
        "code": "L010",
        "kind": "naming_convention",
        "message": "static should be SCREAMING_SNAKE_CASE",
        "target": "statics",
        "pattern": "[A-Z][A-Z0-9_]*",
    }
    config = LintConfig(enabled_codes={"L010"}, custom_rules=[nc_from_config(spec)])

    def count(source):
        return sum(1 for d in run_lint(_hir(source), config).diagnostics
                   if d.code == "L010")

    bad = 'static badName: u8 = 0;\n'
    assert count(bad) == 1
    assert count("#[allow(L010)]\n" + bad) == 0
