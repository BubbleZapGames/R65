# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
D = S must be re-established after a call inside a far-pointer loop.

Root cause (fixed): a function that dereferences a `far *T` parameter sets
D = S so it can use `[dp],Y`. Around an inner call it emits PLD (restore the
caller's D) and afterwards PHD, optionally followed by TSC/TCD to put D back
at the frame. Whether the TSC/TCD pair is emitted came from
`_has_far_ptr_derefs_after_call`, which iterated `blocks.values()` linearly
and called everything textually after the call "after" it.

In the canonical string walk

    loop { let c = s[i]; if c == 0 { break; } putc(c); i++; }

the deref lives in the loop header, which precedes the call's block in the
block map but follows it across the back edge. The scan concluded "no further
derefs", D stayed at the caller's value, and iteration 2 read direct page
$0009 instead of the frame slot — the walk terminated after one character.
The scan is now a CFG reachability query.
"""

from r65.compiler.main import compile_string


WALK_WITH_INNER_CALL = """
static TEXT: [u8; 8] = "AIRLOCK\\0";

#[lowram] static mut SINK: u8;
#[lowram] static mut COUNT: u8;

far fn sink(x: u8) {
    SINK = x;
}

far fn walk(str: far *u8) {
    let s: far *u8 = str;
    let mut i: u16 = 0;
    loop {
        let c: u8 = s[i];
        if c == 0 { break; }
        sink(c);
        i++;
    }
    COUNT = i as u8;
}

#[entry]
fn main() {
    walk(&TEXT as far *u8);
}
"""


def _walk_body(asm: str) -> str:
    """Instructions belonging to `walk`, up to the next top-level function."""
    lines, out, active = asm.split('\n'), [], False
    for line in lines:
        if line.startswith('walk:'):
            active = True
        elif active and line and not line[0].isspace() and line.endswith(':') \
                and not line.startswith('walk'):
            break
        if active:
            out.append(line)
    return '\n'.join(out)


def test_d_is_reestablished_after_call_in_far_ptr_loop():
    body = _walk_body(compile_string(WALK_WITH_INNER_CALL, cfg_options=['snes']))

    # The call site must put D back at the frame, not just push it.
    assert 'PLD' in body, "expected D to be restored before the inner call"
    phd_index = body.index('PHD')
    after_phd = body[phd_index:]
    assert 'TCD' in after_phd, (
        "PHD after the inner call is not followed by TSC/TCD, so D stays at "
        f"the caller's value across the loop back edge:\n{body}"
    )

    # And the loop really does deref through the frame-relative pointer.
    assert '],Y' in body
