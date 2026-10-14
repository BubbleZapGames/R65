"""Compile every ```rust block in docs/AGENTS_R65.md so the primer cannot drift.

A block starting with ``#[snesrom`` is a complete program and compiles as is.
A block starting with ``// error:`` must fail to compile. Any other block is a
set of declarations, compiled after a minimal program prelude.
"""

import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
PRIMER = REPO / "docs" / "AGENTS_R65.md"

PRELUDE = """\
#[snesrom(name="PRIMER")]
#[stack(0x0100, 0x01FF)]
include!("lib/65816.r65")
include!("lib/sneslib.r65")
include!("lib/math.r65")
#[bank(0)]
#[entry]
fn init_reset() -> ! { snes_init(); main(); }
#[interrupt(nmi)]
fn nmi_handler() { clear_nmi!() }
"""

MAIN_STUB = "fn main() -> ! { loop { wait_nmi!(); } }\n"


def _blocks():
    text = PRIMER.read_text(encoding="utf-8")
    for m in re.finditer(r"^```rust\n(.*?)^```", text, re.S | re.M):
        line = text.count("\n", 0, m.start()) + 1
        yield pytest.param(m.group(1), id=f"line{line}")


def _compile(tmp_path, source):
    (tmp_path / "lib").symlink_to(REPO / "stdlib")
    src = tmp_path / "main.r65"
    src.write_text(source, encoding="utf-8")
    return subprocess.run(
        [sys.executable, "-m", "r65.compiler.main", str(src),
         "--cfg", "snes", "-o", str(tmp_path / "main.asm")],
        cwd=REPO, capture_output=True, text=True, timeout=120,
    )


@pytest.mark.parametrize("block", list(_blocks()))
def test_primer_block(tmp_path, block):
    if block.startswith("#[snesrom"):
        source = block
    else:
        source = PRELUDE + block
        if not re.search(r"^fn main\(", block, re.M):
            source += MAIN_STUB

    result = _compile(tmp_path, source)
    if block.startswith("// error:"):
        assert result.returncode != 0, "block was meant to be rejected"
    else:
        assert result.returncode == 0, result.stdout + result.stderr
