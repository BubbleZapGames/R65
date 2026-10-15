# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
The address of an include_bytes! static in a static initializer.

An include_bytes! static is labelled `NAME_data`, every other ROM static
`__NAME_data`. Codegen sets the include_bytes! label when it emits the
.INCBIN, after static initializers are lowered, so a ROM table taking its
address named `__NAME_data` and failed to link.
"""

from r65.tests.e2e import ExpectedState


def test_rom_table_of_include_bytes_addresses(e2e, tmp_path):
    blob = tmp_path / "blob.bin"
    blob.write_bytes(b"\x11\x22\x33\x44")
    result = e2e.run(f'''
        #[zeropage(0x10)]
        static mut OUT: [u8; 2];

        #[bank(1)]
        static BLOB: [u8; 4] = include_bytes!("{blob}");
        static PLAIN: [u8; 4] = [0x55, 0x66, 0x77, 0x88];

        #[bank(0)]
        struct Row {{ data: far *u8, size: u16 }}
        static ROWS: [Row; 2] = [
            Row {{ data: &BLOB, size: 4 }},
            Row {{ data: &PLAIN, size: 4 }},
        ];

        #[entry]
        fn main() {{
            let blob: far *u8 = ROWS[0].data;
            let plain: far *u8 = ROWS[1].data;
            OUT[0] = blob[2];
            OUT[1] = plain[2];
        }}
    ''', ExpectedState(memory={0x7E0010: [0x33, 0x77]}))
    assert result.success, f"Failures: {result.failures}"
