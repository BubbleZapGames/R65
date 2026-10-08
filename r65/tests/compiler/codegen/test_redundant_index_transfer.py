# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""A TYX is emitted only when X does not already hold Y's value.

A loop counter that shares its loop with a scaled index lives in Y and reaches
X by TYX for each `abs,X` access. Codegen emitted one per MIR `X = Move i`, so
a match dispatch's TYX was repeated at the top of every arm and before every
access inside one. The emitter now tracks "X == Y" through the instruction
stream and across blocks whose predecessors all agree.
"""

from r65.compiler.main import compile_string

SOURCE = '''
    #[soa] struct Spark { heat: u8 }
    #[soa] struct Ember { heat: u8, glow: u8 }
    enum Fx { Off, S(Spark), E(Ember) }
    #[soa] struct Particle { x: u8, pos: u16, fx: Fx }
    #[lowram] static mut PARTS: [Particle; 8];

    #[inline(never)]
    fn tick() {
        for i in 0..8 {
            match PARTS[i].fx {
                Fx::S(s) => { s.heat = s.heat - 1; }
                Fx::E(e) => { e.glow = e.glow + e.heat; PARTS[i].x = e.glow; }
                Fx::Off => { PARTS[i].pos = PARTS[i].pos + 1; PARTS[i].x = 0; }
            };
        }
    }

    #[entry]
    fn main() { tick(); }
'''


def _tick():
    asm = compile_string(SOURCE, "t.r65")
    start = asm.index("\ntick:")
    body = asm[start:asm.index("; ---", start)]
    return [l.split(';')[0].strip() for l in body.splitlines()
            if l.split(';')[0].strip() and not l.strip().startswith('.')]


def test_one_tyx_per_x_write():
    """The dispatch's TYX serves every arm; only the arm whose scaled index
    overwrote X (`TYA/ASL/TAX`) needs another."""
    code = _tick()
    assert code.count("TYX") == 2, code


def test_tyx_after_x_was_overwritten():
    code = _tick()
    tax = code.index("TAX")
    assert "TYX" in code[tax + 1:], code
