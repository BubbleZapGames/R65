"""
Integration tests for the emulator with compiled R65 programs.
"""

import pytest
import subprocess
import tempfile
import os
from pathlib import Path

from r65.emulator.cpu import CPU65816, StopExecution, WaitForInterrupt
from r65.emulator.memory import Memory


def compile_r65(source: str) -> bytes:
    """Compile R65 source code to a SNES ROM binary."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write source file
        src_path = Path(tmpdir) / "test.r65"
        asm_path = Path(tmpdir) / "test.asm"
        obj_path = Path(tmpdir) / "test.o"
        rom_path = Path(tmpdir) / "test.sfc"
        link_path = Path(tmpdir) / "linkfile"

        src_path.write_text(source)

        # Compile R65 to assembly
        result = subprocess.run(
            ["r65c", str(src_path), "-o", str(asm_path)],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            raise RuntimeError(f"R65 compilation failed: {result.stderr}")

        # Assemble with WLA-DX
        result = subprocess.run(
            ["wla-65816", "-o", str(obj_path), str(asm_path)],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            raise RuntimeError(f"Assembly failed: {result.stderr}")

        # Link
        link_path.write_text(f"[objects]\n{obj_path}\n")
        result = subprocess.run(
            ["wlalink", "-r", str(link_path), str(rom_path)],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            raise RuntimeError(f"Linking failed: {result.stderr}")

        return rom_path.read_bytes()


def run_program(rom_data: bytes, max_instructions: int = 1000) -> CPU65816:
    """Run a ROM and return the CPU state."""
    memory = Memory(rom_data, mapping="lorom")
    cpu = CPU65816(memory)

    # Start from reset vector
    cpu.PC = memory.get_reset_vector()
    cpu.PBR = 0x00

    instructions = 0
    try:
        while instructions < max_instructions:
            cpu.step()
            instructions += 1
    except (StopExecution, WaitForInterrupt):
        pass

    return cpu


@pytest.fixture
def has_toolchain():
    """Check if R65 compiler and WLA-DX are available."""
    import shutil
    tools = ["r65c", "wla-65816", "wlalink"]
    for tool in tools:
        if shutil.which(tool) is None:
            pytest.skip(f"Tool '{tool}' not available")
    return True


class TestCompiledPrograms:
    """Test emulator with compiled R65 programs."""

    def test_simple_assignment(self, has_toolchain):
        """Test basic register assignments."""
        source = '''
#[entry]
fn main() {
    A = 0x42;
    X = 0x10;
    Y = 0x20;
}
'''
        rom = compile_r65(source)
        cpu = run_program(rom)

        assert (cpu.A & 0xFF) == 0x42
        assert (cpu.X & 0xFF) == 0x10
        assert (cpu.Y & 0xFF) == 0x20

    def test_arithmetic(self, has_toolchain):
        """Test arithmetic operations."""
        source = '''
#[entry]
fn main() {
    A = 10;
    A = A + 5;   // A = 15
    A = A - 3;   // A = 12
}
'''
        rom = compile_r65(source)
        cpu = run_program(rom)

        assert (cpu.A & 0xFF) == 12

    def test_loop_countdown(self, has_toolchain):
        """Test loop with countdown."""
        source = '''
#[entry]
fn main() {
    X = 5;
    A = 0;
    loop {
        A++;
        X--;
        if X == 0 {
            break;
        }
    }
    // A should be 5 after 5 iterations
}
'''
        rom = compile_r65(source)
        cpu = run_program(rom)

        assert (cpu.A & 0xFF) == 5
        assert (cpu.X & 0xFF) == 0

    def test_conditional_branch(self, has_toolchain):
        """Test conditional branching."""
        source = '''
#[entry]
fn main() {
    A = 10;
    if A == 10 {
        X = 0xAA;
    } else {
        X = 0xBB;
    }
}
'''
        rom = compile_r65(source)
        cpu = run_program(rom)

        assert (cpu.X & 0xFF) == 0xAA

    def test_16bit_mode(self, has_toolchain):
        """Test 16-bit accumulator mode."""
        # Note: Entry functions use CLC;XCE to enter native mode, then need
        # REP #$20 to enable 16-bit A. X/Y are always 16-bit in R65.
        source = '''
#[entry]
fn main() {
    asm!("REP #$20");  // Enable 16-bit A
    A = 0x1234;
    X = 0x5678;
}
'''
        rom = compile_r65(source)
        cpu = run_program(rom)

        assert cpu.A == 0x1234
        assert cpu.X == 0x5678

    def test_memory_store_load(self, has_toolchain):
        """Test storing and loading from memory."""
        source = '''
#[zeropage(0x10)]
static mut TEMP: u8;

#[entry]
fn main() {
    TEMP = 0x42;
    A = TEMP;
}
'''
        rom = compile_r65(source)
        cpu = run_program(rom)

        assert (cpu.A & 0xFF) == 0x42
        # Verify memory was written
        assert cpu.memory.read(0x00, 0x10) == 0x42

    def test_subroutine_call(self, has_toolchain):
        """Test function call and return."""
        source = '''
fn set_value() {
    A = 0x99;
}

#[entry]
fn main() {
    A = 0;
    set_value();
    // A should be 0x99 after return
}
'''
        rom = compile_r65(source)
        cpu = run_program(rom)

        assert (cpu.A & 0xFF) == 0x99

    def test_compare_and_branch(self, has_toolchain):
        """Test comparison operations."""
        source = '''
#[entry]
fn main() {
    A = 5;
    X = 0;

    if A > 3 {
        X = 1;
    }

    if A < 10 {
        Y = 1;
    }
}
'''
        rom = compile_r65(source)
        cpu = run_program(rom)

        assert (cpu.X & 0xFF) == 1  # A > 3 was true
        assert (cpu.Y & 0xFF) == 1  # A < 10 was true

    def test_bitwise_operations(self, has_toolchain):
        """Test bitwise AND, OR, XOR."""
        source = '''
#[entry]
fn main() {
    A = 0xFF;
    A = A & 0x0F;  // A = 0x0F
    A = A | 0xF0;  // A = 0xFF
    A = A ^ 0x55;  // A = 0xAA
}
'''
        rom = compile_r65(source)
        cpu = run_program(rom)

        assert (cpu.A & 0xFF) == 0xAA
