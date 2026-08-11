# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Function-pointer calls are classified and traced for reachability.

A forbidden access (e.g. a PPU write) reached only through a function pointer
must not be a blind spot: the caller of an indirect call is widened to the set
of functions whose address is taken (a sound over-approximation, mirroring the
MIR-level call graph). This covers both a pointer stored in a local and a ROM
jump table — the canonical SNES dispatch pattern.
"""

from r65.compiler.frontend.parser import parse
from r65.compiler.frontend.macros import expand_macros
from r65.compiler.hir.builder import HIRBuilder
from r65.compiler.typeck import TypeChecker
from r65.compiler.lint.config import LintConfig
from r65.compiler.lint.linter import run_lint
from r65.compiler.lint.call_graph import build_call_graph
from r65.compiler.lint.rule_kinds.reachability_forbidden_access import from_config


def _hir(source: str):
    program = parse(source, '<test>')
    program = expand_macros(program)
    hir = HIRBuilder().build_program(program)
    TypeChecker(hir).check()
    return hir


def _graph(source: str):
    return build_call_graph(_hir(source))


def _c001_flagged(source: str):
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
    return {src_lines[d.source_loc.line - 1].strip()
            for d in diags.diagnostics if d.code == "C001"}


# ---------------------------------------------------------------- call graph

def test_function_used_as_value_is_address_taken():
    cg = _graph('''
        fn handler() {}
        #[entry]
        fn main() {
            let h: fn() = handler;
            h();
        }
    ''')
    assert "handler" in cg.address_taken
    assert "main" in cg.indirect_callers


def test_direct_callee_is_not_address_taken():
    """A plain direct call must not mark its callee as an indirect target."""
    cg = _graph('''
        fn helper() {}
        #[entry]
        fn main() { helper(); }
    ''')
    assert cg.address_taken == set()
    assert cg.indirect_callers == set()
    assert cg.get_callees("main") == {"helper"}


def test_indirect_caller_widened_to_address_taken():
    cg = _graph('''
        fn a() {}
        fn b() {}
        #[entry]
        fn main() {
            let h: fn() = a;
            let g: fn() = b;
            h();
        }
    ''')
    # main makes an indirect call, so its callees include every address-taken
    # function, a and b both.
    assert cg.get_callees("main") == {"a", "b"}
    assert cg.reachable_from(["main"]) == {"main", "a", "b"}


def test_address_taken_but_never_called_is_not_reachable():
    """Taking an address without ever calling through it invokes nothing."""
    cg = _graph('''
        fn never() {}
        #[entry]
        fn main() {
            let h: fn() = never;   // stored, never called
        }
    ''')
    assert "never" in cg.address_taken
    assert cg.indirect_callers == set()
    assert "never" not in cg.reachable_from(["main"])


# ---------------------------------------------------------------- C001 e2e

def test_c001_flags_write_reachable_only_via_fn_pointer():
    flagged = _c001_flagged('''
        #[hw(0x2122)]
        static mut CGDATA: u8;

        fn bad_handler() {
            CGDATA = 5;
        }

        #[entry]
        fn main() {
            let h: fn() = bad_handler;
            h();
        }
    ''')
    assert "CGDATA = 5;" in flagged


def test_c001_flags_jump_table_handlers():
    """`static TABLE: [fn(); N] = [...]` + indexed call reaches each handler."""
    flagged = _c001_flagged('''
        #[hw(0x2122)]
        static mut CGDATA: u8;

        fn h0() {
            CGDATA = 1;
        }
        fn h1() {
            CGDATA = 2;
        }
        static TABLE: [fn(); 2] = [h0, h1];

        #[entry]
        fn main() {
            let f: fn() = TABLE[0];
            f();
        }
    ''')
    assert flagged == {"CGDATA = 1;", "CGDATA = 2;"}


def test_c001_no_indirect_call_means_no_false_positive():
    """A safe direct-only program stays clean — the widening is gated on an
    actual indirect call, not merely on an address being taken."""
    flagged = _c001_flagged('''
        #[hw(0x2122)]
        static mut CGDATA: u8;

        fn writes_ppu() { CGDATA = 7; }   // address taken, never called

        #[entry]
        fn main() {
            let h: fn() = writes_ppu;
        }
    ''')
    assert flagged == set()
