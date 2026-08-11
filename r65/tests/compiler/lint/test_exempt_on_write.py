# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
`exempt_on_write_*`: a function that writes a guard register is skipped whole.

C001's canonical false positive is an init routine that drops the screen into
forced blank and then sets up VRAM and palette. Those writes are safe, but
forced blank is global mutable state any function can change, so proving it is
not on the table — and guessing would trade a sound rule for an unsound one.

Instead the writer is exempted: touching INIDISP ($2100) means the function is
managing blanking deliberately, so its PPU accesses are the programmer's call.
The exemption is whole-function for straight-line code (writes textually before
the guard write count too) and does not extend to callees. It deliberately stops
at loop bodies: forced blank is established once during init in straight-line
code, so a loop in the same function is the game loop running with the screen
on. Without that carve-out a `main` that blanks during setup — or anything that
fades brightness from its loop — would silence every PPU write in the loop, the
exact bug the rule exists to catch.
"""

from r65.compiler.frontend.parser import parse
from r65.compiler.frontend.macros import expand_macros
from r65.compiler.hir.builder import HIRBuilder
from r65.compiler.typeck import TypeChecker
from r65.compiler.lint.config import LintConfig
from r65.compiler.lint.linter import run_lint
from r65.compiler.lint.rule_kinds.reachability_forbidden_access import from_config


SOURCE = '''
#[hw(0x2100)]
static mut INIDISP: u8;
#[hw(0x2122)]
static mut CGDATA: u8;

// Writes CGDATA *before* the guard write - still exempt, the whole function is.
fn init_palette() {
    CGDATA = 1;
    INIDISP = 0x80;
    CGDATA = 2;
}

// No guard write - checked normally.
fn draw() {
    CGDATA = 3;
}

// Called by an exempt function, but not exempt itself.
fn helper() {
    CGDATA = 4;
}

fn init_with_helper() {
    INIDISP = 0x80;
    helper();
}

#[entry]
fn main() {
    init_palette();
    draw();
    init_with_helper();
}
'''


def _hir(source: str):
    program = parse(source, '<test>')
    program = expand_macros(program)
    hir = HIRBuilder().build_program(program)
    TypeChecker(hir).check()
    return hir


def _lint(exempt):
    spec = {
        "code": "C001",
        "kind": "reachability_forbidden_access",
        "message": "PPU register write unsafe during active rendering",
        "entry_points": ["main"],
        "forbid_addrs": [0x2122],
    }
    if exempt:
        spec["exempt_on_write_addrs"] = [0x2100]
    config = LintConfig(enabled_codes={"C001"}, custom_rules=[from_config(spec)])
    diags = run_lint(_hir(SOURCE), config)
    return sorted(d.source_loc.line for d in diags.diagnostics if d.code == "C001")


def test_without_guard_every_write_is_flagged():
    # 4 CGDATA writes: two in init_palette, one in draw, one in helper.
    assert len(_lint(exempt=False)) == 4


def test_guard_write_exempts_the_whole_function():
    lines = _lint(exempt=True)
    # init_palette's two writes are gone - including the one before the guard.
    # draw() and helper() remain.
    assert len(lines) == 2, f"expected draw + helper only, got lines {lines}"


def test_exemption_does_not_extend_to_callees():
    """helper() is called only from an exempt function but is still checked."""
    source_lines = SOURCE.split('\n')
    flagged = {source_lines[n - 1].strip() for n in _lint(exempt=True)}
    assert "CGDATA = 4;" in flagged, (
        f"callee of an exempt function should still be checked; flagged={flagged}"
    )


LOOP_SOURCE = '''
#[hw(0x2100)]
static mut INIDISP: u8;
#[hw(0x2122)]
static mut CGDATA: u8;

#[entry]
fn main() {
    INIDISP = 0x80;      // forced blank for init
    CGDATA = 1;          // straight-line, safe
    INIDISP = 0x0F;      // screen on
    loop {
        CGDATA = 2;      // active display - must stay flagged
    }
}
'''


def _lint_source(source, exempt=True):
    spec = {
        "code": "C001",
        "kind": "reachability_forbidden_access",
        "message": "PPU register write unsafe during active rendering",
        "entry_points": ["main"],
        "forbid_addrs": [0x2122],
    }
    if exempt:
        spec["exempt_on_write_addrs"] = [0x2100]
    config = LintConfig(enabled_codes={"C001"}, custom_rules=[from_config(spec)])
    diags = run_lint(_hir(source), config)
    return sorted(d.source_loc.line for d in diags.diagnostics if d.code == "C001")


def test_guard_does_not_exempt_writes_inside_a_loop():
    """A main that blanks during init must not go unchecked in its game loop."""
    lines = _lint_source(LOOP_SOURCE)
    source_lines = LOOP_SOURCE.split('\n')
    flagged = [source_lines[n - 1].strip() for n in lines]
    assert flagged == ["CGDATA = 2;      // active display - must stay flagged"], (
        f"expected only the in-loop write flagged, got {flagged}"
    )


# The exemption is anchored to the init prologue: it ends where the first loop
# begins. A write placed after the game loop starts is live-display and must be
# flagged even though `main` blanked during init — "OK at the very top, not
# once the loop begins".
AFTER_LOOP_SOURCE = '''
#[hw(0x2100)]
static mut INIDISP: u8;
#[hw(0x2122)]
static mut CGDATA: u8;
#[hw(0x2101)]
static mut FLAG: u8;

#[entry]
fn main() {
    INIDISP = 0x80;      // forced blank for init
    CGDATA = 1;          // prologue, before the loop - exempt
    while FLAG != 0 {
        CGDATA = 2;      // game loop - flagged
    }
    CGDATA = 3;          // after the loop - live display, must be flagged
}
'''


def test_exemption_ends_at_the_first_loop():
    lines = _lint_source(AFTER_LOOP_SOURCE)
    source_lines = AFTER_LOOP_SOURCE.split('\n')
    flagged = [source_lines[n - 1].strip() for n in lines]
    assert flagged == [
        "CGDATA = 2;      // game loop - flagged",
        "CGDATA = 3;          // after the loop - live display, must be flagged",
    ], f"prologue write should be exempt; loop and post-loop writes flagged; got {flagged}"
