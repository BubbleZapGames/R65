# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""The interrupt prologue's scratch pushes must mirror the epilogue's pops.

An NMI handler borrows the scratch-register pool from whatever code it
interrupted, so it stacks the pool on entry and puts it back on exit.
The prologue pushes a pair of adjacent 1-byte scratches as one `PEI`
(one instruction, two bytes of stack, and A untouched); the epilogue has
to pull that pair back as one 16-bit `PLA`. Get the grouping out of step
on either side and the handler returns with a shifted stack — every
local below it reads the wrong slot, and the interrupted code resumes
with scrambled scratch registers.

Both sides derive their sequence from
`register_alloc.group_interrupt_scratch_pushes`, so these tests pin the
property that keeps them honest: bytes pushed == bytes pulled, and each
scratch address is restored from the slot it was saved into.
"""

import re

from r65.compiler.main import compile_string


SOURCE = """
#[zeropage(0x00, register)] static mut S0: u8;
#[zeropage(0x01, register)] static mut S1: u8;
#[zeropage(0x02, register)] static mut S2: u8;
#[zeropage(0x03, register)] static mut S3: u8;

#[zeropage(0x20)]
static mut COUNTER: u16;

#[interrupt(nmi)]
fn nmi_handler() {
    COUNTER = COUNTER + 1;
}

#[entry]
fn main() {
    COUNTER = 0;
}
"""


def _handler_asm() -> str:
    asm = compile_string(SOURCE, cfg_options=["snes"])
    body = asm.split("nmi_handler:", 1)[1]
    return body.split("RTI", 1)[0]


def test_pushes_and_pulls_balance():
    """Every byte the prologue stacks for the pool is pulled back before RTI."""
    handler = _handler_asm()
    pushed = 2 * len(re.findall(r"^\s*PEI ", handler, re.M))
    # Count only the pool's pops — the trailing PLA restores the saved
    # accumulator, which the CPU-pushed frame accounts for, not the pool.
    pulled = 2 * len(re.findall(r"^\s*PLA\b\s*(?:;[^\n]*)?\n\s*STA \$00[0-9A-F]{2}\b",
                                handler, re.M))
    assert pushed and pushed == pulled


def test_scratch_pairs_ride_as_words():
    """Four adjacent 1-byte scratches become two PEI/PLA words, not eight ops."""
    handler = _handler_asm()
    assert re.search(r"^\s*PEI \(\$00\)", handler, re.M)
    assert re.search(r"^\s*PEI \(\$02\)", handler, re.M)
    # The old byte-at-a-time save is gone: no LDA/PHA of the pool.
    assert not re.search(r"^\s*LDA \$0[0-3]\b", handler, re.M)


def test_words_restored_to_their_own_addresses_in_reverse():
    """Pops mirror pushes: last word pushed is the first restored."""
    handler = _handler_asm()
    restores = re.findall(r"^\s*STA \$00(0[02])\b", handler, re.M)
    assert restores == ["02", "00"]


def test_direct_page_reset_uses_the_accumulator_path():
    """A is already saved 16-bit, so D=0 costs LDA/TCD, not PEA/PLD."""
    handler = _handler_asm()
    assert re.search(r"^\s*TCD\b", handler, re.M)
    assert not re.search(r"^\s*PLD\b", handler.split("PEI", 1)[0], re.M)


WIDE_SOURCE = """
#[zeropage(0x00, register)] static mut S0: u8;
#[zeropage(0x01, register)] static mut S1: u8;
#[zeropage(0x04, register)] static mut FP0: far *u8;

#[zeropage(0x20)]
static mut COUNTER: u16;

#[interrupt(nmi)]
fn nmi_handler() {
    COUNTER = COUNTER + 1;
}

#[entry]
fn main() {
    COUNTER = 0;
}
"""


EMPTY_BODY_SOURCE = """
#[zeropage(0x00, register)] static mut S0: u8;
#[zeropage(0x01, register)] static mut S1: u8;
#[zeropage(0x02, register)] static mut S2: u8;

#[interrupt(nmi)]
fn nmi_handler() { }

#[entry]
fn main() { }
"""


def _handler_of(source: str) -> str:
    asm = compile_string(source, cfg_options=["snes"])
    return asm.split("nmi_handler:", 1)[1].split("RTI", 1)[0]


def test_three_byte_scratch_is_saved_whole():
    """A far-pointer scratch is 3 bytes: a PEI word plus its odd byte.

    Splitting by width rather than special-casing sizes 1 and 2 matters —
    `stdlib/scratch_regs.r65` declares four 3-byte `far *u8` scratches,
    and the earlier size-keyed code saved none of their bytes at all.
    """
    handler = _handler_of(WIDE_SOURCE)
    assert re.search(r"^\s*PEI \(\$04\)", handler, re.M)      # FP0 bytes 0-1
    assert re.search(r"^\s*LDA \$06\b", handler, re.M)         # FP0 byte 2
    assert re.search(r"^\s*STA \$0006\b", handler, re.M)       # ...restored


def test_odd_width_pool_balances():
    """Bytes pushed equal bytes pulled when the pool doesn't pair evenly."""
    handler = _handler_of(WIDE_SOURCE)
    pushed = (2 * len(re.findall(r"^\s*PEI ", handler, re.M))
              + len(re.findall(r"^\s*PHA\b(?!.*full 16-bit)", handler, re.M)))
    pulled = (2 * len(re.findall(r"^\s*PLA\b[^\n]*\n\s*STA \$00[0-9A-F]{2}\b(?<!\$0006)",
                                 handler, re.M))
              + len(re.findall(r"^\s*PLA\b[^\n]*\n\s*STA \$0006\b", handler, re.M)))
    assert pushed == pulled == 5


def test_byte_save_never_runs_in_unknown_mode():
    """A pool forces m8 even when the body itself modifies nothing.

    `LDA dp`/`PHA` pushes one byte in m8 and two in m16, while the
    epilogue always pulls back one. Without the SEP, an interrupt taken
    from m16 code would RTI through a stack left a byte deep.
    """
    handler = _handler_of(EMPTY_BODY_SOURCE)
    sep = handler.find("SEP #$20")
    first_save = min(
        (m.start() for m in re.finditer(r"^\s*(PEI|LDA \$0)", handler, re.M)),
        default=-1,
    )
    assert sep != -1 and first_save != -1 and sep < first_save
