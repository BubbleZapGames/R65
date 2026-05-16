# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""Regression tests for `.bank_byte()` on a constant far pointer.

A constant far pointer (`&STATIC as far *u8`) is materialized into a
stack temp before its bytes are extracted; `.bank_byte()` lowers to a
read of that temp's bank byte (instruction_select.py select_bank_byte).

Fixed bug: dead-code elimination's read-operand map
(optimize/dead_code_elim.py `_READ_FIELDS`) omitted `BankByte`, so a
`BankByte %d = BankByte %src` was treated as reading nothing. When the
same `&STATIC as far *u8` was materialized twice in one expansion —

    A1T = (&tbl as far *u8) as u16;     // first materialization (%1)
    A1B = (&tbl as far *u8).bank_byte();// second materialization (%4)

DCE saw `%4` (the second far pointer) as unused, deleted its defining
`Move`/`TypeConvert`, and left `BankByte %5 = BankByte %4` dangling on
an undefined vreg. Codegen then resolved `%4` to a never-written stack
slot (`LDA $07,S` while the bank byte was stored to `$03,S`) → the bank
byte was uninitialised stack garbage. Address half was unaffected.

Found via the classickong `hdma_set_tables` rewrite: the doubly-
materialized form produced corrupted (rainbow-noise) HDMA gradients.

These tests assert peephole-stable invariants on the generated asm:
1. No `LDA $NN,S` reads a local stack slot the function never wrote
   (the exact corruption signature — a dangling/garbage read).
2. The doubly-materialized far pointer emits the bank-of-symbol
   immediate (`LDA #:...`) once per materialization (i.e. DCE no
   longer drops the second one).
"""

import re

from r65.compiler.main import compile_string


def _function_asm(asm: str, name: str) -> str:
    """Extract a single function's assembly (label .. RTS/RTL)."""
    out, in_fn = [], False
    for line in asm.split("\n"):
        if line.strip().startswith(f"{name}:"):
            in_fn = True
        if in_fn:
            out.append(line)
            s = line.strip()
            if s.startswith("RTS") or s.startswith("RTL"):
                break
    return "\n".join(out)


def _unwritten_slot_reads(func_asm: str) -> set:
    """`$NN,S` operands that are LDA-read but never STA-written here.

    `setup` builds its far pointers from scratch (no params), so every
    legitimate local-slot read is preceded by a store. A read of a slot
    with no store anywhere in the function is the dangling-vreg garbage
    read this regression guards against.
    """
    code = "\n".join(ln.split(";", 1)[0] for ln in func_asm.split("\n"))
    written = set(re.findall(r"STA\s+(\$[0-9A-Fa-f]+,S)", code))
    read = set(re.findall(r"LDA\s+(\$[0-9A-Fa-f]+,S)", code))
    return read - written


_DOUBLE_SRC = """
#[hw(0x4302)]
static mut A1T: u16;
#[hw(0x4304)]
static mut A1B: u8;
static tbl: [u8; 4] = [1, 2, 3, 4];

far fn setup() {
    A1T = (&tbl as far *u8) as u16;
    A1B = (&tbl as far *u8).bank_byte();
}

#[interrupt(nmi)]
fn vbl() { setup(); }

fn main() -> ! { loop { asm!("WAI"); } }
"""

_SINGLE_BIND_SRC = """
#[hw(0x4302)]
static mut A1T: u16;
#[hw(0x4304)]
static mut A1B: u8;
static tbl: [u8; 4] = [1, 2, 3, 4];

far fn setup() {
    let p: far *u8 = &tbl as far *u8;
    A1T = p as u16;
    A1B = p.bank_byte();
}

#[interrupt(nmi)]
fn vbl() { setup(); }

fn main() -> ! { loop { asm!("WAI"); } }
"""


def test_double_materialization_far_pointer_bank_byte_not_corrupted():
    """Doubly-materialized `&STATIC as far *u8` + `.bank_byte()` must not
    leave a dangling vreg whose bank byte reads an uninitialised slot."""
    setup_asm = _function_asm(compile_string(_DOUBLE_SRC, "test.r65"), "setup")

    assert _unwritten_slot_reads(setup_asm) == set(), (
        f"bank_byte() reads a never-written stack slot (dangling far "
        f"pointer — DCE dropped its materialization):\n{setup_asm}"
    )
    # Both `(&tbl as far *u8)` expressions must be materialized: each emits
    # the bank-of-symbol immediate. Pre-fix DCE eliminated the second one.
    assert setup_asm.count("LDA #:") == 2, (
        f"expected 2 far-pointer bank materializations (one per "
        f"`&tbl as far *u8`), got {setup_asm.count('LDA #:')}:\n{setup_asm}"
    )


def test_single_binding_far_pointer_bank_byte():
    """The known-good idiom (bind the far pointer once) must keep
    `.bank_byte()` reading a properly initialised slot."""
    setup_asm = _function_asm(compile_string(_SINGLE_BIND_SRC, "test.r65"), "setup")

    assert _unwritten_slot_reads(setup_asm) == set(), (
        f"single-binding regression: bank_byte() reads a never-written "
        f"stack slot:\n{setup_asm}"
    )
    assert setup_asm.count("LDA #:") == 1, (
        f"single binding should materialize the far pointer once, got "
        f"{setup_asm.count('LDA #:')}:\n{setup_asm}"
    )
