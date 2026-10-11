# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""Dead-code elimination keeps a read of a #[hw] register nothing uses.

The read is the side effect: $213F resets the H/V latch flip-flop, $2137
latches the counters, $4211 acknowledges the timer IRQ. -O1 used to drop
all three, so the V counter read 0 and the IRQ was never acknowledged.
"""

from r65.compiler.main import compile_source

SOURCE = """
#[hw(0x213F)] static mut STAT78: u8;
#[hw(0x2137)] static mut SLHV: u8;
#[hw(0x213D)] static mut OPVCT: u8;
#[hw(0x4211)] static mut TIMEUP: u8;
#[lowram] static mut PLAIN: u8;

#[inline(never)]
fn read_vcount() -> u8 {
    let junk: u8 = STAT78;
    let trigger: u8 = SLHV;
    let unused: u8 = PLAIN;
    return OPVCT;
}

#[interrupt(irq)]
fn irq() { let ack: u8 = TIMEUP; }

#[entry]
fn main() -> ! { let v = read_vcount(); loop {} }
"""


def _fn(asm: str, name: str) -> list:
    body = asm.split(f"\n{name}:", 1)[1].split("\n\n", 1)[0]
    return [l.split(';')[0].strip() for l in body.splitlines()
            if l.split(';')[0].strip() and not l.strip().startswith('.')]


def _compile(tmp_path) -> str:
    out = tmp_path / "t.asm"
    compile_source(SOURCE, "t.r65", output_file=str(out), quiet=True, opt_level=1)
    return out.read_text()


def test_unused_hw_reads_survive(tmp_path):
    code = _fn(_compile(tmp_path), "read_vcount")
    assert code[:3] == ["LDA $213F", "LDA $2137", "LDA $213D"], code


def test_unused_plain_read_is_still_removed(tmp_path):
    code = _fn(_compile(tmp_path), "read_vcount")
    assert not any("$0200" in c for c in code), code


def test_interrupt_acknowledge_survives(tmp_path):
    assert "BIT $4211" in _fn(_compile(tmp_path), "irq")
