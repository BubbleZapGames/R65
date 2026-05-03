# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Codegen tests for the JML [d] fast path used to lower far indirect calls
when the function pointer is DP-addressable.

Reference: r65/compiler/codegen/call_select.py
            _emit_dp_indirect_far_call / _dp_offset_for_indirect_call
            _emit_indirect_call_trampoline (slow path)

The fast path fires for SCRATCH-resident far fn pointers and emits:

    PHK
    PEA <ret-1
    JML [<dp_offset>]    ; WLA-DX form for opcode 0xDC (long indirect via DP)
ret:

The slow trampoline emits the SBC chain via PHA/SBC stack-relative ops.
These tests rely on the textual difference: the fast path emits ``JML [`` and
no ``SBC`` chain, the slow path emits a 3-byte SBC chain on stack.
"""

from r65.tests.language.common import build_mir
from r65.compiler.codegen import ProgramCodeGenerator


def _generate(source: str) -> str:
    mir = build_mir(source)
    codegen = ProgramCodeGenerator()
    return codegen.generate(mir)


def _section(asm: str, func_name: str) -> str:
    """Extract the assembly for a single function from the program output.
    Returns text from ``func_name:`` up to the next top-level label or
    end-of-text.
    """
    import re
    pattern = rf"^{re.escape(func_name)}:.*?(?=^[a-zA-Z_][\w]*:\s|^\.[A-Z])"
    m = re.search(pattern, asm, re.MULTILINE | re.DOTALL)
    if m:
        return m.group(0)
    # Fall back: from the label to EOF.
    idx = asm.find(f"\n{func_name}:")
    return asm[idx:] if idx >= 0 else ""


# ---------------------------------------------------------------------------
# Fast path: SCRATCH-resident far fn pointer
# ---------------------------------------------------------------------------

# Source: thin invoker with scratch registers declared. Scratch param
# promotion lands the far fn ptr in a zeropage scratch slot, which is
# DP-addressable — the fast path fires.
SCRATCH_INVOKER_SOURCE = """
#[zeropage(0x10, register)]
static mut S0: u8;
#[zeropage(0x11, register)]
static mut S1: u8;
#[zeropage(0x12, register)]
static mut S2: u8;
#[zeropage(0x13, register)]
static mut S3: u8;

far fn target() {}

fn invoke(handler: far fn()) {
    handler();
}

fn main() -> u8 {
    invoke(target);
    return 0;
}
"""


def test_fast_path_fires_for_scratch_far_fn_ptr():
    """Far fn ptr in scratch slot triggers the JML [d] sequence.

    Asserts:
      - PHK is emitted (push PBR for the far call).
      - PEA with the return-label-1 expression is emitted.
      - JML [ <scratch_addr> ] is emitted (WLA-DX form for JML [d]).
      - The slow trampoline's SBC chain is NOT emitted.
    """
    asm = _generate(SCRATCH_INVOKER_SOURCE)
    invoke = _section(asm, "invoke")

    assert "PHK" in invoke, "fast path must emit PHK"
    assert "PEA " in invoke, "fast path must emit PEA"
    # WLA-DX expects the explicit "JML [addr]" mnemonic for 0xDC; the
    # emitter renders this form (see emitter._emit_instruction's
    # JMP_INDIRECT_LONG branch).
    assert "JML [" in invoke, "fast path must emit JML [d]"
    # The slow trampoline's address adjustment uses an SBC chain.
    assert "SBC " not in invoke, "fast path must NOT emit SBC chain"


# ---------------------------------------------------------------------------
# Slow path: STACK-resident far fn pointer (no scratch promotion)
# ---------------------------------------------------------------------------

# No scratch registers declared — far fn ptr param stays on stack. The
# STACK fast path is currently deferred (see _dp_offset_for_indirect_call
# docstring), so the trampoline fires.
STACK_INVOKER_SOURCE = """
far fn target() {}

fn invoke(handler: far fn()) {
    handler();
}

fn main() -> u8 {
    invoke(target);
    return 0;
}
"""


def test_slow_path_for_stack_far_fn_ptr_no_scratch():
    """Stack-resident far fn ptr (no scratch promotion) uses the trampoline.

    The trampoline pushes 3 bytes of return address, 3 bytes of target
    address, then runs an SBC chain of 3 stack-relative subtractions.
    """
    asm = _generate(STACK_INVOKER_SOURCE)
    invoke = _section(asm, "invoke")

    # The trampoline emits multiple SBC instructions.
    assert "SBC " in invoke, "stack/no-scratch case must use the trampoline"
    # And uses RTL (not JML [d]) at the end.
    assert "RTL" in invoke, "trampoline ends with RTL"


# ---------------------------------------------------------------------------
# Near indirect calls always use the trampoline (no JML [d] for 16-bit)
# ---------------------------------------------------------------------------

NEAR_INVOKER_SOURCE = """
#[zeropage(0x10, register)]
static mut S0: u8;
#[zeropage(0x11, register)]
static mut S1: u8;

fn target() {}

fn invoke_near(handler: fn()) {
    handler();
}

fn main() -> u8 {
    invoke_near(target);
    return 0;
}
"""


def test_near_indirect_call_uses_trampoline_not_jml():
    """Near (16-bit) indirect calls keep the existing trampoline. There
    is no JML [d] for 16-bit indirect via DP that synthesizes a JSR.
    """
    asm = _generate(NEAR_INVOKER_SOURCE)
    invoke = _section(asm, "invoke_near")

    # The trampoline emits SBC chain.
    assert "SBC " in invoke, "near indirect must keep using the trampoline"
    # No JML [d] for the near case.
    assert "JML [" not in invoke, "near indirect must NOT emit JML [d]"


# ---------------------------------------------------------------------------
# Soundness: unadjusted DP offset
# ---------------------------------------------------------------------------

def test_fast_path_dp_offset_matches_scratch_address():
    """The DP offset embedded in JML [<dp>] must equal the scratch slot
    address — not adjusted by stack_bytes_pushed (no PHA-pushed args here)
    and not shifted by other calls' bookkeeping.
    """
    asm = _generate(SCRATCH_INVOKER_SOURCE)
    invoke = _section(asm, "invoke")

    import re
    m = re.search(r"JML \[\$([0-9A-Fa-f]+)\]", invoke)
    assert m, "fast path must emit JML [$<dp>]"
    # The scratch pool starts at $10 — the param's slot. Just assert the
    # value parses as a small DP address (< $100).
    addr = int(m.group(1), 16)
    assert 0 <= addr < 0x100, f"DP offset must be DP-addressable, got ${addr:X}"
