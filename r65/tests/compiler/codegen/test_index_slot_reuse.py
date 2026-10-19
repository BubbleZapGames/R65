# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""An index reloaded from its stack slot is skipped when X already holds it.

A loop counter that cannot live in a register (the u16 plane's scaled index
needs X, and Y is taken) sits in a stack slot, and every `units[j].field`
reloaded it: `LDA j,S; TAX` for a byte plane, `LDA j,S; ASL A; TAX` for a u16
plane. The emitter now tracks what X holds — a stack slot, shifted left k times
— so a reload that would give X the value it already has is not emitted.

The `*P = 0` keeps the counter in its slot: the store goes through Y, so loop
register promotion cannot give `j` to Y.
"""

from r65.compiler.main import compile_string

SOURCE = """struct Tank { reload: u8 }
enum UnitType { Free, Tank(Tank), Van }
#[soa]
struct Unit { x_px: u16, owner: u8, kind: UnitType }
#[lowram] static mut units: [Unit; 32];
#[zeropage] static mut P: *u8;

#[inline(never)]
fn target(self_i: u16, range: u16) -> u16 {
    let x: u16 = units[self_i].x_px;
    let mine: u8 = units[self_i].owner;
    let mut best: u16 = 32;
    let mut best_d: u16 = range;
    for j in 0..32 {
        *P = 0;
        match units[j].kind {
            UnitType::Free => {}
            _ => {
                if units[j].owner != mine {
                    let mut d: u16 = 0;
                    if units[j].x_px > x { d = units[j].x_px - x; }
                    if d < best_d { best_d = d; best = j; }
                }
            }
        };
    }
    return best;
}

#[entry]
fn main() { target(0, 100); }
"""


def _target():
    asm = compile_string(SOURCE, "t.r65")
    start = asm.index("\ntarget:")
    body = asm[start:asm.index("; ---", start)]
    return [l.split(';')[0].strip() for l in body.splitlines()
            if l.split(';')[0].strip() and not l.strip().startswith('.')]


def _loop_body():
    # The rotated loop runs from its body (L2) to its test, which falls into
    # the exit (L3); the match arms are laid out after the exit's return.
    code = _target()
    return (code[code.index("target__L2:"):code.index("target__L3:")]
            + code[code.index("target__L6:"):])


def test_one_load_of_each_index_shape():
    """In the loop: `j` into X once for the byte planes, `j*2` once for the u16
    plane — the repeats on the same path (owner after kind, x_px read twice)
    go."""
    body = _loop_body()
    assert body.count("TAX") == 2, body


def test_the_scaled_index_is_still_built_once():
    body = _loop_body()
    assert body.count("ASL A") == 1, body
