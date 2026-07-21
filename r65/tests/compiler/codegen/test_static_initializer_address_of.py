# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Regression tests for `&IMMUTABLE_STATIC` inside static initializers.

Root cause (fixed): `_extract_struct_literal_bytes` / `_emit_array_literal_init`
ran every initializer element through the const evaluator and substituted 0
whenever it returned None. An address-of is not a compile-time integer, so a
ROM table like

    static AREAS: [Area; 2] = [Area { text: &AREA0_TEXT as far *u8, .. }, ..];

linked cleanly with every pointer field silently emitted as $00,$00,$00 — the
canonical shape for a text adventure's area table. Fix: lower address-of to
WLA-DX `<label` / `>label` / `:label` bytes, and make any other non-constant
initializer a hard error rather than a silent zero.
"""

import pytest

from r65.compiler.main import compile_string
from r65.compiler.errors import MIRLoweringError


def _data_table(asm: str, label: str) -> str:
    """The `.db` lines belonging to `label`, up to the next label."""
    lines = asm.split('\n')
    out, active = [], False
    for line in lines:
        stripped = line.strip()
        if stripped == f'{label}:':
            active = True
            continue
        if active:
            if stripped.startswith('.db'):
                out.append(stripped)
            elif stripped and not stripped.startswith(';'):
                break
    return '\n'.join(out)


STRUCT_TABLE = """
static AREA0_TEXT: [u8; 8] = "AIRLOCK\\0";
static AREA1_TEXT: [u8; 8] = "GALLEY\\0";

struct Area {
    text: far *u8,
    exit_n: u8
}

static AREAS: [Area; 2] = [
    Area { text: &AREA0_TEXT as far *u8, exit_n: 1 },
    Area { text: &AREA1_TEXT as far *u8, exit_n: 0 }
];

#[zeropage(0x30)]
static mut OUT: u8;

#[entry]
fn main() {
    let a: far *Area = &AREAS[1] as far *Area;
    let t: far *u8 = a.text;
    OUT = t[0];
}
"""


def test_struct_field_address_of_emits_link_time_bytes():
    asm = compile_string(STRUCT_TABLE, cfg_options=['snes'])
    table = _data_table(asm, '__AREAS_data')

    # Each far-pointer field is three assembler-resolved bytes, not $00,$00,$00.
    for label in ('__AREA0_TEXT_data', '__AREA1_TEXT_data'):
        for prefix in ('<', '>', ':'):
            assert f'{prefix}{label}' in table, (
                f"missing {prefix}{label} in table:\n{table}"
            )

    # The u8 field still lands at its own offset (layout unchanged).
    assert '$01' in table and '$00' in table


ARRAY_OF_POINTERS = """
static MSG_A: [u8; 4] = "HI\\0";

static TABLE: [far *u8; 2] = [
    &MSG_A as far *u8,
    &MSG_A as far *u8
];

#[entry]
fn main() {
    let p: far *u8 = TABLE[0];
}
"""


def test_array_element_address_of_emits_link_time_bytes():
    asm = compile_string(ARRAY_OF_POINTERS, cfg_options=['snes'])
    table = _data_table(asm, '__TABLE_data')
    assert table.count('<__MSG_A_data') == 2
    assert table.count(':__MSG_A_data') == 2


MUTABLE_TARGET = """
#[ram]
static mut BUFFER: [u8; 4];

static TABLE: [far *u8; 1] = [&BUFFER as far *u8];

#[entry]
fn main() {
    let p: far *u8 = TABLE[0];
}
"""


def test_address_of_mutable_static_is_an_error_not_a_zero():
    """RAM addresses are assigned after MIR lowering, so this cannot link."""
    with pytest.raises(MIRLoweringError, match='mutable static'):
        compile_string(MUTABLE_TARGET, cfg_options=['snes'])
