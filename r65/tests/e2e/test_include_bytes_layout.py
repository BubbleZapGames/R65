# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""include_bytes! data shares its bank with code without either landing on
the other: it is emitted in the bank's window with the code
(symbol_gen.include_bytes_by_bank). Opened on its own first, the bank was
re-entered by a bare `.BANK`, which WLA-DX continues at the previous bank's
offset - the code went over the start of the data."""

from r65.tests.e2e import ExpectedState


def test_code_and_include_bytes_in_one_bank(e2e, tmp_path):
    data = bytes(range(256)) * 16
    blob = tmp_path / "data.bin"
    blob.write_bytes(data)
    total = sum(data[:256])     # the bytes the code used to land on
    result = e2e.run(f'''
        #[zeropage(0xC0)] static mut SUM: u16;
        #[entry]
        fn main() {{ SUM = head_sum(); }}
        #[bank(1)]
        static DATA = include_bytes!("{blob}");
        far fn head_sum() -> u16 {{
            let mut s: u16 = 0;
            for i in 0..256 {{ s = s + (DATA[i] as u16); }}
            return s;
        }}
    ''', ExpectedState(memory={0x7E00C0: [total & 0xFF, total >> 8]}))
    assert result.success, f"Failures: {result.failures}"
