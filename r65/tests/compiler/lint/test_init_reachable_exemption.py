# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Init-reachable auto-exemption for `reachability_forbidden_access` (C001).

A function reachable from an entry point *only* through calls that precede the
entry's first loop runs entirely during init, so its straight-line PPU writes
are setup writes and are exempt automatically — no `exclude_subtrees` entry,
and no guard (`INIDISP`) write required. The loop carve-out still applies: a
write inside a loop in such a function is still flagged. A function also
reachable through the game loop is not init-only and stays fully checked.
"""

from r65.compiler.frontend.parser import parse
from r65.compiler.frontend.macros import expand_macros
from r65.compiler.hir.builder import HIRBuilder
from r65.compiler.typeck import TypeChecker
from r65.compiler.lint.config import LintConfig
from r65.compiler.lint.linter import run_lint
from r65.compiler.lint.rule_kinds.reachability_forbidden_access import from_config


def _hir(source: str):
    program = parse(source, '<test>')
    program = expand_macros(program)
    hir = HIRBuilder().build_program(program)
    TypeChecker(hir).check()
    return hir


def _flagged(source: str):
    """Lines of source flagged by a bare C001 (no guard/exempt config)."""
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


HW = '''
#[hw(0x2122)]
static mut CGDATA: u8;
#[hw(0x2101)]
static mut FLAG: u8;
'''


def test_prologue_only_helper_is_exempt():
    """setup() is called before main's loop and writes straight-line — exempt,
    even though it never touches INIDISP."""
    flagged = _flagged(HW + '''
fn setup() {
    CGDATA = 1;
}
#[entry]
fn main() {
    setup();
    loop {}
}
''')
    assert flagged == [], f"prologue-only write should be exempt, got {flagged}"


def test_in_loop_write_in_init_helper_still_flagged():
    """The loop carve-out survives: a VRAM-writing loop in an init helper is a
    real bug and must stay flagged."""
    flagged = _flagged(HW + '''
fn setup() {
    CGDATA = 1;
    while FLAG != 0 {
        CGDATA = 2;
    }
}
#[entry]
fn main() {
    setup();
    loop {}
}
''')
    assert flagged == ["CGDATA = 2;"], (
        f"straight-line exempt, in-loop flagged; got {flagged}")


def test_game_loop_reachable_helper_is_checked():
    """work() is called from inside the game loop — not init, must be flagged."""
    flagged = _flagged(HW + '''
fn work() {
    CGDATA = 3;
}
#[entry]
fn main() {
    loop {
        work();
    }
}
''')
    assert flagged == ["CGDATA = 3;"], flagged


def test_helper_reached_both_ways_is_checked():
    """load() runs at boot AND in the loop — not init-only, stays flagged."""
    flagged = _flagged(HW + '''
fn load() {
    CGDATA = 4;
}
#[entry]
fn main() {
    load();
    loop {
        load();
    }
}
''')
    assert flagged == ["CGDATA = 4;"], flagged


def test_init_reachability_is_transitive():
    """b() is reached only through a(), reached only from main's prologue."""
    flagged = _flagged(HW + '''
fn b() {
    CGDATA = 6;
}
fn a() {
    b();
}
#[entry]
fn main() {
    a();
    loop {}
}
''')
    assert flagged == [], f"transitively prologue-only should be exempt, got {flagged}"


def test_entry_without_a_loop_grants_no_exemption():
    """A degenerate main with no game loop has no init/live boundary, so nothing
    is auto-exempt (conservative — never hide a would-be live write)."""
    flagged = _flagged(HW + '''
fn setup() {
    CGDATA = 7;
}
#[entry]
fn main() {
    setup();
}
''')
    assert flagged == ["CGDATA = 7;"], flagged
