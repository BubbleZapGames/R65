# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""A ROM bank whose code and data pass its end is a compile error
(codegen/bank_size.py check_bank_overflow), at the declaration that
crosses it, with a hint naming a bank with room. The limits match what
WLA-DX accepts: a bank's full size, and bank 0 up to the ROM header's
game code at $7FB2.
"""

import pytest

from r65.compiler.errors import CodegenError
from r65.compiler.main import compile_string

MAIN = '''
#[zeropage(0x10)] static mut R: u8;
#[entry] fn main() { R = 1; }
'''


def _compile(src: str):
    return compile_string(MAIN + src, cfg_options=["snes"])


def test_overflowing_bank_names_an_empty_one():
    with pytest.raises(CodegenError) as e:
        _compile('''
            #[bank(1)]
            static TA: [u8; 20000] = [1; 20000];
            #[bank(1)]
            static TB: [u8; 14000] = [2; 14000];
            #[used]
            far fn rd(i @ X: u16) -> u8 { return TA[i] + TB[i]; }
        ''')
    assert "ROM bank 1 overflows by" in e.value.message
    assert e.value.source_loc is not None
    assert "bank 2 is empty" in e.value.hint
    assert "TA 20,000 bytes, TB 14,000 bytes" in e.value.hint


def test_a_full_bank_fits_and_one_byte_more_does_not():
    _compile("#[bank(1)]\n#[used]\nstatic TA: [u8; 32768] = [1; 32768];")
    with pytest.raises(CodegenError, match="bank 1 overflows by 1 byte:"):
        _compile("#[bank(1)]\n#[used]\nstatic TA: [u8; 32769] = [1; 32769];")


def test_bank_zero_stops_at_the_rom_header():
    src = '''
        #[zeropage(0x10)] static mut R: u8;
        static TA: [u8; {n}] = [1; {n}];
        #[entry] fn main() {{ R = TA[3]; }}
    '''
    compile_string(src.format(n=32600), cfg_options=["snes"])
    with pytest.raises(CodegenError, match=r"header starts at \$7FB2"):
        compile_string(src.format(n=32700), cfg_options=["snes"])


def test_include_bytes_larger_than_a_bank(tmp_path, monkeypatch):
    """Sized by the frontend: the path is relative to the source file, and
    the compiler runs from the project root, as a project's Makefile does."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "gfx.bin").write_bytes(bytes(40000))
    monkeypatch.chdir(tmp_path)
    with pytest.raises(CodegenError) as e:
        compile_string(MAIN + '#[bank(1)]\n#[used]\nstatic GFX = include_bytes!("gfx.bin");',
                       filename=str(tmp_path / "src" / "main.r65"), cfg_options=["snes"])
    assert "more than a bank holds" in e.value.hint


def test_hirom_bank_zero_stops_at_the_vector_trampolines():
    """HiROM's trampolines go at $FF00 in bank 0, below the header."""
    src = '''
        #[snesrom(name = "HI", hirom)]
        #[zeropage(0x10)] static mut R: u8;
        static TA: [u8; {n}] = [1; {n}];
        #[entry] fn main() {{ R = TA[3]; }}
    '''
    compile_string(src.format(n=65000), cfg_options=["snes"])
    with pytest.raises(CodegenError, match=r"vector trampolines start at \$FF00"):
        compile_string(src.format(n=65300), cfg_options=["snes"])
