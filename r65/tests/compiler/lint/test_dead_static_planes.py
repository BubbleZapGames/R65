# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""L005 on a planed pool: reading a plane is reading the pool.

An `#[soa]` struct pool and a tagged enum pool are never *named* in a read.
`POOL[i].hp` and `match POOL[i]` are both rewritten to read one of the
synthetic planes, so the parent appears only on the left of an assignment.
Without attributing a plane read back to its parent, every pool in the corpus
reads as "written but never read".
"""

from r65.compiler.frontend.parser import parse
from r65.compiler.frontend.macros import expand_macros
from r65.compiler.hir.builder import HIRBuilder
from r65.compiler.typeck import TypeChecker
from r65.compiler.lint.config import LintConfig
from r65.compiler.lint.linter import run_lint


TAGGED = '''
struct Tank { reload: u8 }
enum Kind { Free, Tank(Tank), Van }

#[lowram]
static mut unit_kind: [Kind; 8];
#[zeropage(0x40)]
static mut out: u8;
'''

SOA = '''
#[soa]
struct Enemy { kind: u8, hp: u8 }

#[lowram]
static mut POOL: [Enemy; 8];
#[zeropage(0x40)]
static mut out: u8;
'''


def _dead(source: str, name: str):
    """L005 messages naming *name*. Filtered by name so the scratch `out`
    static in these fixtures cannot answer for the pool."""
    program = expand_macros(parse(source, '<test>'))
    hir = HIRBuilder().build_program(program)
    TypeChecker(hir).check()
    diags = run_lint(hir, LintConfig(enabled_codes={"L005"}))
    return sorted(d.message for d in diags.diagnostics
                  if d.code == "L005" and f"static mut {name}`" in d.message)


def test_match_counts_as_reading_the_pool():
    assert _dead(TAGGED + '''
        #[entry]
        fn main() {
            unit_kind[0] = Kind::Van;
            out = match unit_kind[0] { Kind::Van => 1, _ => 0 };
        }
    ''', "unit_kind") == []


def test_discriminant_cast_counts_as_reading_the_pool():
    assert _dead(TAGGED + '''
        #[entry]
        fn main() {
            unit_kind[0] = Kind::Van;
            out = unit_kind[0] as u8;
        }
    ''', "unit_kind") == []


def test_soa_field_read_counts_as_reading_the_pool():
    assert _dead(SOA + '''
        #[entry]
        fn main() {
            POOL[0].hp = 5;
            out = POOL[0].hp;
        }
    ''', "POOL") == []


def test_an_untouched_pool_is_still_reported():
    """The rule still fires on a pool nothing uses at all.

    A pool that is only *written* is not reported, but neither is a plain
    `static mut BUF: [u8; 4]` that is only written — indexing an array names it
    in a read position either way, which is pre-existing behaviour for every
    array, not something the plane attribution introduced.
    """
    assert _dead(TAGGED + '''
        #[entry]
        fn main() { out = 1; }
    ''', "unit_kind") == ["`static mut unit_kind` is written but never read"]


class TestUnusedParamIgnoresSelf:
    """L008 does not flag an unread receiver.

    A method takes `self` because the calling convention says so, not because
    the author asked for it, and the rule's advice — "prefix with `_`" — cannot
    be followed for a receiver. An implementation that answers with a constant,
    `fn base(*self) -> *Base { return &TANK_BASE as *Base; }`, is a normal
    shape rather than an oversight.
    """

    def _unused(self, source: str):
        program = expand_macros(parse(source, '<test>'))
        hir = HIRBuilder().build_program(program)
        TypeChecker(hir).check()
        diags = run_lint(hir, LintConfig(enabled_codes={"L008"}))
        return sorted(d.message for d in diags.diagnostics if d.code == "L008")

    def test_unread_receiver_is_not_flagged(self):
        assert self._unused('''
            struct Base { speed: u16, pad: u16 }
            struct Tank { reload: u8 }
            static TANK_BASE: Base = Base { speed: 128, pad: 0 };
            trait HasBase { fn base(*self) -> *Base; }
            impl HasBase for Tank {
                fn base(*self) -> *Base { return &TANK_BASE as *Base; }
            }
            #[entry]
            fn main() {}
        ''') == []

    def test_an_ordinary_unused_parameter_is_still_flagged(self):
        assert self._unused('''
            #[zeropage(0x40)] static mut o: u8;
            fn f(used: u8, spare: u8) { o = used; }
            #[entry]
            fn main() { f(1, 2); }
        ''') == ["unused parameter `spare`"]
