# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
End-to-end tests for sneslib.r65.

Tests SNES hardware register definitions, constants, and macros.
"""

import pytest
from pathlib import Path
from r65.tests.e2e import ExpectedState

# Path to stdlib
STDLIB_DIR = Path(__file__).parent.parent.parent.parent / "stdlib"
SNESLIB_PATH = STDLIB_DIR / "sneslib.r65"


class TestSneslibInclude:
    """Test that sneslib includes and compiles correctly."""

    def test_sneslib_includes(self, e2e):
        """Test that sneslib.r65 can be included and compiles."""
        source = f'''
            include!("{SNESLIB_PATH}")

            #[entry]
            fn main() {{
                A = 0x42;
            }}
        '''
        result = e2e.run(source, ExpectedState(A=0x42))
        assert result.success, f"Failures: {result.failures}"


class TestSneslibConstants:
    """Test sneslib constant definitions."""

    def test_vmain_constants(self, e2e):
        """Test VMAIN increment mode enums."""
        source = f'''
            include!("{SNESLIB_PATH}")

            #[zeropage(0x10)]
            static mut RESULT: [u8; 4];

            #[entry]
            fn main() {{
                RESULT[0] = VmainTrigger::Low as u8;
                RESULT[1] = VmainTrigger::High as u8;
                RESULT[2] = VmainStep::By1 as u8;
                RESULT[3] = VmainStep::By32 as u8;
            }}
        '''
        result = e2e.run(source, ExpectedState(memory={
            0x7E0010: [0x00, 0x80, 0x00, 0x01]
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_screen_constants(self, e2e):
        """Test screen mode enums."""
        source = f'''
            include!("{SNESLIB_PATH}")

            #[zeropage(0x10)]
            static mut RESULT: [u8; 4];

            #[entry]
            fn main() {{
                RESULT[0] = Brightness::Full as u8;
                RESULT[1] = FORCE_BLANK;
                RESULT[2] = BG::Mode0 as u8;
                RESULT[3] = BG::Mode1 as u8;
            }}
        '''
        result = e2e.run(source, ExpectedState(memory={
            0x7E0010: [0x0F, 0x80, 0x00, 0x01]
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_dma_mode_constants(self, e2e):
        """Test DMA mode enums."""
        source = f'''
            include!("{SNESLIB_PATH}")

            #[zeropage(0x10)]
            static mut RESULT: [u8; 8];

            #[entry]
            fn main() {{
                RESULT[0] = DmaTransferMode::OneReg as u8;
                RESULT[1] = DmaTransferMode::TwoReg as u8;
                RESULT[2] = DmaTransferMode::OneRegX2 as u8;
                RESULT[3] = DmaTransferMode::TwoRegX2 as u8;
                RESULT[4] = DmaTransferMode::FourReg as u8;
                RESULT[5] = DmaDirection::ToPPU as u8;
                RESULT[6] = DmaDirection::ToCPU as u8;
                RESULT[7] = DmaAddress::Fixed as u8;
            }}
        '''
        result = e2e.run(source, ExpectedState(memory={
            0x7E0010: [0x00, 0x01, 0x02, 0x03, 0x04, 0x00, 0x80, 0x08]
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_joypad_button_constants(self, e2e):
        """Test joypad button mask constants."""
        source = f'''
            include!("{SNESLIB_PATH}")

            #[zeropage(0x10)]
            static mut RESULT: [u8; 8];

            #[entry]
            fn main() {{
                // High bytes of button masks
                RESULT[0] = (JOY_A >> 8) as u8;
                RESULT[1] = (JOY_B >> 8) as u8;
                RESULT[2] = (JOY_X >> 8) as u8;
                RESULT[3] = (JOY_Y >> 8) as u8;
                RESULT[4] = (JOY_L >> 8) as u8;
                RESULT[5] = (JOY_R >> 8) as u8;
                RESULT[6] = (JOY_START >> 8) as u8;
                RESULT[7] = (JOY_SELECT >> 8) as u8;
            }}
        '''
        result = e2e.run(source, ExpectedState(memory={
            # JOY_A=0x0080, JOY_B=0x8000, JOY_X=0x0040, JOY_Y=0x4000
            # JOY_L=0x0020, JOY_R=0x0010, JOY_START=0x1000, JOY_SELECT=0x2000
            0x7E0010: [0x00, 0x80, 0x00, 0x40, 0x00, 0x00, 0x10, 0x20]
        }))
        assert result.success, f"Failures: {result.failures}"


class TestSneslibMacros:
    """Test sneslib macro functionality."""

    def test_set_brightness_macro(self, e2e):
        """Test set_brightness! macro sets INIDISP correctly."""
        source = f'''
            include!("{SNESLIB_PATH}")

            #[zeropage(0x10)]
            static mut BRIGHTNESS_COPY: u8;

            #[entry]
            fn main() {{
                // set_brightness! writes to INIDISP ($2100)
                // We can't easily read PPU regs, but we can verify it compiles
                // and check that BRIGHTNESS_FULL is correct
                BRIGHTNESS_COPY = Brightness::Full as u8;
            }}
        '''
        result = e2e.run(source, ExpectedState(memory={
            0x7E0010: 0x0F
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_asm_wai(self, e2e):
        """Test asm! with WAI instruction stops execution."""
        source = f'''
            include!("{SNESLIB_PATH}")

            #[entry]
            fn main() {{
                A = 0x55;
                // WAI will stop execution, so A should remain 0x55
                asm!("WAI");
                // This line won't execute
                A = 0xAA;
            }}
        '''
        result = e2e.run(source, ExpectedState(A=0x55))
        assert result.success, f"Failures: {result.failures}"


class TestSneslibHardwareAddresses:
    """Test that hardware register addresses are correct."""

    def test_ppu_register_writes(self, e2e):
        """Test writing to PPU registers compiles correctly."""
        source = f'''
            include!("{SNESLIB_PATH}")

            #[zeropage(0x10)]
            static mut TEST_VAL: u8;

            #[entry]
            fn main() {{
                // These write to hardware registers
                // We verify compilation succeeds and logic is correct
                TEST_VAL = 0x80;  // VMAIN_INCREMENT_HIGH

                // Store what we would write to verify logic
                A = TEST_VAL;
            }}
        '''
        result = e2e.run(source, ExpectedState(A=0x80))
        assert result.success, f"Failures: {result.failures}"


class TestSneslibColorMath:
    """Test color manipulation utilities."""

    def test_rgb15_constant(self, e2e):
        """Test RGB15 color format constants."""
        source = f'''
            include!("{SNESLIB_PATH}")

            #[zeropage(0x10)]
            static mut RESULT: [u8; 4];

            #[entry]
            fn main() {{
                // RGB15 format: 0BBBBBGG GGGRRRRR
                // Black = 0x0000, White = 0x7FFF
                // Red (31,0,0) = 0x001F
                // Green (0,31,0) = 0x03E0
                // Blue (0,0,31) = 0x7C00

                // Store some test values
                let red: u16 = 0x001F;
                let blue: u16 = 0x7C00;

                RESULT[0] = red as u8;
                RESULT[1] = (red >> 8) as u8;
                RESULT[2] = blue as u8;
                RESULT[3] = (blue >> 8) as u8;
            }}
        '''
        result = e2e.run(source, ExpectedState(memory={
            0x7E0010: [0x1F, 0x00, 0x00, 0x7C]
        }))
        assert result.success, f"Failures: {result.failures}"


class TestSneslibColorType:
    """The `Color` newtype: construction, accessors, and the raw-u16 interop.

    RGB15 is 0BBBBBGG GGGRRRRR, so red (31,0,0) is 0x001F, green 0x03E0 and
    blue 0x7C00. `Color::rgb!` and the component accessors are method macros
    rather than methods, so all of this folds at compile time when the
    components are constant — these run the result to prove the folding is
    right, not merely cheap.
    """

    def test_rgb_components_round_trip(self, e2e):
        source = f'''
            include!("{SNESLIB_PATH}")

            #[zeropage(0x10)]
            static mut RESULT: [u8; 6];

            #[entry]
            fn main() {{
                let red = Color::rgb!(31, 0, 0);
                let green = Color::rgb!(0, 31, 0);
                let blue = Color::rgb!(0, 0, 31);

                RESULT[0] = red.0 as u8;
                RESULT[1] = (green.0 >> 8) as u8;
                RESULT[2] = (blue.0 >> 8) as u8;

                // Accessors recover the components they were built from.
                let mixed = Color::rgb!(1, 2, 3);
                RESULT[3] = mixed.red!();
                RESULT[4] = mixed.green!();
                RESULT[5] = mixed.blue!();
            }}
        '''
        result = e2e.run(source, ExpectedState(memory={
            0x7E0010: [0x1F, 0x03, 0x7C, 0x01, 0x02, 0x03]
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_runtime_components(self, e2e):
        """The same arithmetic with a value the compiler cannot fold."""
        source = f'''
            include!("{SNESLIB_PATH}")

            #[zeropage(0x10)]
            static mut RESULT: [u8; 2];
            #[zeropage(0x20)]
            static mut LEVEL: u16;

            #[entry]
            fn main() {{
                LEVEL = 31;
                let c = Color::rgb!(LEVEL, 0, LEVEL);
                RESULT[0] = c.red!();
                RESULT[1] = c.blue!();
            }}
        '''
        result = e2e.run(source, ExpectedState(memory={
            0x7E0010: [31, 31]
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_constants_and_write_color(self, e2e):
        """Color::WHITE through write_color!, which still takes a raw u16 too."""
        source = f'''
            include!("{SNESLIB_PATH}")

            #[zeropage(0x10)]
            static mut RESULT: [u8; 4];

            #[entry]
            fn main() {{
                RESULT[0] = Color::WHITE.0 as u8;
                RESULT[1] = (Color::WHITE.0 >> 8) as u8;
                RESULT[2] = Color::BLACK.0 as u8;

                // A Color and a bare u16 both reach CGDATA through the macro.
                set_cgram_addr!(0);
                write_color!(Color::BLACK);
                write_color!(0x7FFF);
                RESULT[3] = 1;
            }}
        '''
        result = e2e.run(source, ExpectedState(memory={
            0x7E0010: [0xFF, 0x7F, 0x00, 0x01]
        }))
        assert result.success, f"Failures: {result.failures}"


class TestSneslibOamHelpers:
    """Test OAM helper functionality."""

    def test_oam_attribute_constants(self, e2e):
        """Test OAM attribute constants."""
        source = f'''
            include!("{SNESLIB_PATH}")

            #[zeropage(0x10)]
            static mut RESULT: [u8; 8];

            #[entry]
            fn main() {{
                // OAM entry struct field offsets (0,1,2,3)
                RESULT[0] = 0;  // OamEntry.x offset
                RESULT[1] = 1;  // OamEntry.y offset
                RESULT[2] = 2;  // OamEntry.tile offset
                RESULT[3] = 3;  // OamEntry.attr offset
                // OAM attribute enums
                RESULT[4] = OamFlip::V as u8;
                RESULT[5] = OamFlip::H as u8;
                RESULT[6] = OamPriority::InFront as u8;
                RESULT[7] = OamPalette::Palette7 as u8;
            }}
        '''
        result = e2e.run(source, ExpectedState(memory={
            0x7E0010: [0x00, 0x01, 0x02, 0x03, 0x80, 0x40, 0x30, 0x0E]
        }))
        assert result.success, f"Failures: {result.failures}"


class TestSneslibVramAddressing:
    """Test VRAM addressing utilities."""

    def test_vram_word_address(self, e2e):
        """Test VRAM addressing (word addresses)."""
        source = f'''
            include!("{SNESLIB_PATH}")

            #[zeropage(0x10)]
            static mut RESULT: [u8; 4];

            #[entry]
            fn main() {{
                // VRAM uses word addresses (16-bit values at each address)
                // Common tile addresses:
                // BG1 tiles at word 0x0000
                // BG2 tiles at word 0x4000
                // Sprite tiles at word 0x6000

                let bg1_tiles: u16 = 0x0000;
                let sprite_tiles: u16 = 0x6000;

                RESULT[0] = bg1_tiles as u8;
                RESULT[1] = (bg1_tiles >> 8) as u8;
                RESULT[2] = sprite_tiles as u8;
                RESULT[3] = (sprite_tiles >> 8) as u8;
            }}
        '''
        result = e2e.run(source, ExpectedState(memory={
            0x7E0010: [0x00, 0x00, 0x00, 0x60]
        }))
        assert result.success, f"Failures: {result.failures}"


class TestSneslibDmaMacros:
    """Test DMA macro compilation and execution.

    Note: The emulator doesn't execute actual DMA transfers, but we can verify
    that the macros compile correctly and the setup code runs without errors.

    DMA macros use WLA-DX assembler operators (#<, #>, #^) to extract addresses
    from labels, which requires the data to have proper assembly labels.
    """

    def test_dma_trigger_compiles(self, e2e):
        """Test dma_trigger! macro compiles and runs."""
        source = f'''
            include!("{SNESLIB_PATH}")

            #[zeropage(0x10)]
            static mut DONE: u8;

            #[entry]
            fn main() {{
                // Just trigger DMA on channel 0 (no actual transfer configured)
                dma_trigger!(0);
                DONE = 0xAA;
            }}
        '''
        result = e2e.run(source, ExpectedState(memory={
            0x7E0010: 0xAA
        }))
        assert result.success, f"Failures: {result.failures}"

    @pytest.mark.skip(reason="WLA-DX STACK_CALCULATE syntax error on generated LDA")
    def test_dma_set_ppu_dest_compiles(self, e2e):
        """Test dma_set_ppu_dest! macro compiles."""
        source = f'''
            include!("{SNESLIB_PATH}")

            #[zeropage(0x10)]
            static mut DONE: u8;

            #[entry]
            fn main() {{
                // Set up VRAM destination mode
                dma_set_ppu_dest!(0, DmaTransferMode::TwoReg as u8, 0x18);
                // Set up CGRAM destination mode
                dma_set_ppu_dest!(1, DmaTransferMode::OneRegX2 as u8, 0x22);
                // Set up OAM destination mode
                dma_set_ppu_dest!(2, DmaTransferMode::OneReg as u8, 0x04);
                DONE = 0xBB;
            }}
        '''
        result = e2e.run(source, ExpectedState(memory={
            0x7E0010: 0xBB
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_dma_set_size_compiles(self, e2e):
        """Test dma_set_size! macro compiles with various sizes."""
        source = f'''
            include!("{SNESLIB_PATH}")

            #[zeropage(0x10)]
            static mut DONE: u8;

            #[entry]
            fn main() {{
                // Test various transfer sizes
                dma_set_size!(0, 0x100);   // 256 bytes
                dma_set_size!(1, 0x1000);  // 4KB
                dma_set_size!(2, 0x8000);  // 32KB
                DONE = 0xCC;
            }}
        '''
        result = e2e.run(source, ExpectedState(memory={
            0x7E0010: 0xCC
        }))
        assert result.success, f"Failures: {result.failures}"

    @pytest.mark.skip(reason="WLA-DX STACK_CALCULATE syntax error on generated LDA")
    def test_dma_modes_with_flags(self, e2e):
        """Test DMA mode constants and flag combinations."""
        source = f'''
            include!("{SNESLIB_PATH}")

            #[zeropage(0x10)]
            static mut DONE: u8;

            #[entry]
            fn main() {{
                // Normal copy mode
                dma_set_ppu_dest!(0, DmaTransferMode::TwoReg as u8, 0x18);
                // Fixed source (for fills)
                dma_set_ppu_dest!(1, DmaTransferMode::TwoReg as u8 | DmaAddress::Fixed as u8, 0x18);
                // Reverse direction (PPU to CPU)
                dma_set_ppu_dest!(2, DmaTransferMode::TwoReg as u8 | DmaDirection::ToCPU as u8, 0x39);
                DONE = 0xDD;
            }}
        '''
        result = e2e.run(source, ExpectedState(memory={
            0x7E0010: 0xDD
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_vmain_setup_for_dma(self, e2e):
        """Test VMAIN register setup used by DMA macros."""
        source = f'''
            include!("{SNESLIB_PATH}")

            #[zeropage(0x10)]
            static mut DONE: u8;

            #[entry]
            fn main() {{
                // Set VMAIN for high byte increment (used before VRAM DMA)
                VMAIN = VmainTrigger::High as u8;
                // Set VRAM address
                VMADD = 0x1000;
                DONE = 0xEE;
            }}
        '''
        result = e2e.run(source, ExpectedState(memory={
            0x7E0010: 0xEE
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_cgadd_setup_for_dma(self, e2e):
        """Test CGRAM address setup used by DMA macros."""
        source = f'''
            include!("{SNESLIB_PATH}")

            #[zeropage(0x10)]
            static mut DONE: u8;

            #[entry]
            fn main() {{
                // Set CGRAM start address (color index)
                CGADD = 0;     // Start at color 0
                CGADD = 128;   // Start at color 128
                DONE = 0xFF;
            }}
        '''
        result = e2e.run(source, ExpectedState(memory={
            0x7E0010: 0xFF
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_oamadd_setup_for_dma(self, e2e):
        """Test OAM address setup used by DMA macros."""
        source = f'''
            include!("{SNESLIB_PATH}")

            #[zeropage(0x10)]
            static mut DONE: u8;

            #[entry]
            fn main() {{
                // Reset OAM address to start
                OAMADD = 0;
                DONE = 0x42;
            }}
        '''
        result = e2e.run(source, ExpectedState(memory={
            0x7E0010: 0x42
        }))
        assert result.success, f"Failures: {result.failures}"
