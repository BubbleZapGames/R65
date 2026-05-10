# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Regression tests for `let x = y` vreg aliasing.

When a `let` binding initializes from another variable, the MIR builder must
allocate a fresh VirtualRegister for the new binding. If the same vreg were
reused, mutating one would mutate the other.

The buggy MIR for `let mut cy = height; cy--;` would look like:

    %0:height = Move A
    %0:cy = Move %0:height       # SAME vreg — wrong
    %1 = %0 - 1
    %0 = Move %1                  # decrements both height and cy

The fixed MIR uses distinct vregs for `height` and `cy`.
"""
from r65.compiler.frontend import Parser
from r65.compiler.hir import HIRBuilder
from r65.compiler.typeck import TypeChecker
from r65.compiler.mir import MIRBuilder
from r65.compiler.mir.nodes import VirtualRegister


def build_mir(source: str):
    parser = Parser()
    ast = parser.parse(source)
    hir = HIRBuilder().build_program(ast)
    TypeChecker(hir).check()
    return MIRBuilder().build_program(hir)


def find_function(mir, name: str):
    for fn in mir.functions:
        if fn.name == name:
            return fn
    raise AssertionError(f"function {name!r} not in MIR")


def collect_vregs(fn):
    """Map vreg.id → VirtualRegister for every vreg referenced in fn."""
    vregs = {}
    for blk in fn.blocks.values():
        for ins in blk.instructions:
            for attr in vars(ins):
                v = getattr(ins, attr, None)
                if isinstance(v, VirtualRegister):
                    vregs[v.id] = v
                elif isinstance(v, list):
                    for item in v:
                        if isinstance(item, VirtualRegister):
                            vregs[item.id] = item
    return vregs


def vregs_by_hint(fn):
    """Map hint name → list of VirtualRegisters with that hint."""
    out = {}
    for v in collect_vregs(fn).values():
        out.setdefault(v.hint, []).append(v)
    return out


class TestVregAliasingFresh:
    """`let x = y` produces a fresh vreg distinct from y's vreg."""

    def test_let_copy_distinct_vreg(self):
        """let mut cy = height; cy-- — cy and height are distinct vregs."""
        mir = build_mir('''
            #[zeropage(0x10)]
            static mut RESULT: [u8; 2];
            fn use_copy(height @ A: u8) {
                let mut cy: u8 = height;
                cy--;
                RESULT[0] = height;
                RESULT[1] = cy;
            }
        ''')
        fn = find_function(mir, 'use_copy')
        by_hint = vregs_by_hint(fn)

        # The 'height' parameter is saved into a vreg named 'saved_height'
        # (HIR inserts a save-on-entry); 'cy' is the local binding.
        assert 'cy' in by_hint, "expected a vreg named 'cy'"
        height_vregs = by_hint.get('saved_height') or by_hint.get('height') or []
        assert height_vregs, "expected a vreg for the height param"

        cy_ids = {v.id for v in by_hint['cy']}
        height_ids = {v.id for v in height_vregs}
        assert cy_ids.isdisjoint(height_ids), (
            f"cy and height share a vreg id: cy={cy_ids}, height={height_ids}"
        )

    def test_loop_bound_distinct_from_param(self):
        """let bound = n; n--; — bound stays distinct from n."""
        mir = build_mir('''
            #[zeropage(0x10)]
            static mut COUNT: u8;
            fn count_down(n @ A: u8) {
                let bound: u8 = n;
                let mut i: u8 = 0;
                loop {
                    if i >= bound { break; }
                    n--;
                    i++;
                }
                COUNT = i;
            }
        ''')
        fn = find_function(mir, 'count_down')
        by_hint = vregs_by_hint(fn)
        assert 'bound' in by_hint
        n_vregs = by_hint.get('saved_n') or by_hint.get('n') or []
        assert n_vregs
        bound_ids = {v.id for v in by_hint['bound']}
        n_ids = {v.id for v in n_vregs}
        assert bound_ids.isdisjoint(n_ids)

    def test_multiple_copies_each_distinct(self):
        """let a = val; let b = val; let c = val; — three distinct vregs."""
        mir = build_mir('''
            #[zeropage(0x10)]
            static mut RESULT: [u8; 3];
            fn triple_copy(val @ A: u8) {
                let mut a: u8 = val;
                let mut b: u8 = val;
                let mut c: u8 = val;
                a = a + 1;
                b = b + 2;
                c = c + 3;
                RESULT[0] = a;
                RESULT[1] = b;
                RESULT[2] = c;
            }
        ''')
        fn = find_function(mir, 'triple_copy')
        by_hint = vregs_by_hint(fn)
        # Each binding gets its own vreg
        for name in ('a', 'b', 'c'):
            assert name in by_hint, f"expected vreg {name!r}"
        a_ids = {v.id for v in by_hint['a']}
        b_ids = {v.id for v in by_hint['b']}
        c_ids = {v.id for v in by_hint['c']}
        assert a_ids.isdisjoint(b_ids)
        assert b_ids.isdisjoint(c_ids)
        assert a_ids.isdisjoint(c_ids)
