#!/usr/bin/env python3
# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
End-to-end integration tests for the R65 compiler.

Tests the complete pipeline from R65 source code to linked SNES ROM:
1. Compile R65 source to WLA-DX assembly
2. Assemble with wla-65816
3. Link with wlalink

Requires wla-dx to be installed and available in PATH.
"""

import os
import subprocess
import tempfile
import pytest
from pathlib import Path

from r65.compiler.frontend import parse
from r65.compiler.hir import HIRBuilder
from r65.compiler.typeck import TypeChecker
from r65.compiler.mir import MIRBuilder
from r65.compiler.codegen import ProgramCodeGenerator
from r65.compiler.analysis import RecursionChecker


def check_wla_available():
    """Check if WLA-DX tools are available."""
    try:
        # wla-65816 doesn't support --version, so we just check if it exists
        # by running it with no args (shows usage and exits with error, but that's OK)
        result = subprocess.run(
            ["wla-65816"],
            capture_output=True,
            text=True,
            timeout=5
        )
        # If we get here, the command exists (even if it returns non-zero)
        return True
    except FileNotFoundError:
        return False
    except subprocess.SubprocessError:
        return False


def check_wlalink_available():
    """Check if wlalink is available."""
    try:
        result = subprocess.run(
            ["wlalink"],
            capture_output=True,
            text=True,
            timeout=5
        )
        # wlalink with no args shows usage and returns non-zero, but that's OK
        return True
    except FileNotFoundError:
        return False


WLA_AVAILABLE = check_wla_available()
WLALINK_AVAILABLE = check_wlalink_available()


def compile_r65_source(source: str) -> str:
    """
    Compile R65 source code to WLA-DX assembly.

    Args:
        source: R65 source code string

    Returns:
        Generated assembly code as string
    """
    # Parse
    ast = parse(source)

    # HIR
    hir_builder = HIRBuilder()
    hir_program = hir_builder.build_program(ast)

    # Type checking
    type_checker = TypeChecker(hir_program)
    type_checker.check()

    # MIR
    mir_builder = MIRBuilder()
    mir_program = mir_builder.build_program(hir_program)

    # Check for unsafe recursion
    recursion_checker = RecursionChecker(mir_program)
    recursion_checker.check()

    # Code generation
    codegen = ProgramCodeGenerator()
    assembly = codegen.generate(mir_program)

    return assembly


def assemble_and_link(assembly: str, workdir: Path) -> Path:
    """
    Assemble and link assembly code into a ROM.

    Args:
        assembly: WLA-DX assembly code
        workdir: Working directory for intermediate files

    Returns:
        Path to the generated ROM file
    """
    asm_file = workdir / "test.asm"
    obj_file = workdir / "test.o"
    link_file = workdir / "linkfile"
    rom_file = workdir / "test.sfc"

    # Write assembly
    asm_file.write_text(assembly)

    # Assemble
    result = subprocess.run(
        ["wla-65816", "-o", str(obj_file), str(asm_file)],
        capture_output=True,
        text=True,
        cwd=workdir,
        timeout=30
    )
    if result.returncode != 0:
        raise RuntimeError(f"Assembly failed:\n{result.stderr}\n{result.stdout}")

    # Create link file (RAM placement is handled by .ENUM directives
    # in the assembly, no [ramsections] block needed)
    link_file.write_text(f"[objects]\n{obj_file}\n")

    # Link
    result = subprocess.run(
        ["wlalink", "-r", str(link_file), str(rom_file)],
        capture_output=True,
        text=True,
        cwd=workdir,
        timeout=30
    )
    if result.returncode != 0:
        raise RuntimeError(f"Linking failed:\n{result.stderr}\n{result.stdout}")

    return rom_file


# =============================================================================
# Minimal Program Tests
# =============================================================================

MINIMAL_PROGRAM = '''
#[entry]
fn main() -> ! {
    A = 42;
}
'''


class TestEndToEnd:
    """End-to-end integration tests."""

    def test_compile_minimal_program(self):
        """Test that a minimal program compiles to valid assembly."""
        assembly = compile_r65_source(MINIMAL_PROGRAM)

        # Verify key elements are present
        assert ".65816" in assembly
        assert "main:" in assembly
        assert "WAI" in assembly
        assert ".SNESHEADER" in assembly
        assert "RESET main" in assembly

    @pytest.mark.skipif(not WLA_AVAILABLE, reason="wla-65816 not available")
    def test_assemble_minimal_program(self):
        """Test that the minimal program assembles with WLA-DX."""
        assembly = compile_r65_source(MINIMAL_PROGRAM)

        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)
            asm_file = workdir / "test.asm"
            obj_file = workdir / "test.o"

            asm_file.write_text(assembly)

            result = subprocess.run(
                ["wla-65816", "-o", str(obj_file), str(asm_file)],
                capture_output=True,
                text=True,
                cwd=workdir,
                timeout=30
            )

            assert result.returncode == 0, f"Assembly failed:\n{result.stderr}"
            assert obj_file.exists(), "Object file was not created"

    @pytest.mark.skipif(
        not (WLA_AVAILABLE and WLALINK_AVAILABLE),
        reason="wla-65816 or wlalink not available"
    )
    def test_link_minimal_program(self):
        """Test that the minimal program links into a valid ROM."""
        assembly = compile_r65_source(MINIMAL_PROGRAM)

        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)
            rom_file = assemble_and_link(assembly, workdir)

            assert rom_file.exists(), "ROM file was not created"

            # Verify ROM size (256KB for SNES ROMSIZE $08 - minimum size)
            rom_size = rom_file.stat().st_size
            assert rom_size == 262144, f"Expected 256KB ROM, got {rom_size} bytes"


# =============================================================================
# Program with Variables
# =============================================================================

PROGRAM_WITH_VARIABLES = '''
#[zeropage(0x10)]
static mut COUNTER: u8 = 0;

#[ram]
static mut BUFFER: [u8; 16] = [0; 16];

#[entry]
fn main() -> ! {
    COUNTER = 42;
    BUFFER[0] = COUNTER;

    loop {
        COUNTER = COUNTER + 1;
    }
}
'''


class TestProgramWithVariables:
    """Tests for programs with static variables."""

    def test_compile_with_variables(self):
        """Test compilation of program with static variables."""
        assembly = compile_r65_source(PROGRAM_WITH_VARIABLES)

        assert ".DEFINE COUNTER" in assembly
        assert "main:" in assembly
        assert "LDA #$2A" in assembly  # 42 decimal = $2A hex

    @pytest.mark.skipif(
        not (WLA_AVAILABLE and WLALINK_AVAILABLE),
        reason="wla-65816 or wlalink not available"
    )
    def test_link_with_variables(self):
        """Test that program with variables links successfully."""
        assembly = compile_r65_source(PROGRAM_WITH_VARIABLES)

        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)
            rom_file = assemble_and_link(assembly, workdir)

            assert rom_file.exists()
            assert rom_file.stat().st_size == 262144  # 256KB ROM (minimum)


# =============================================================================
# Program with Functions
# =============================================================================

PROGRAM_WITH_FUNCTIONS = '''
#[zeropage(0x10)]
static mut VALUE: u8 = 0;

fn increment() {
    VALUE = VALUE + 1;
}

fn add_ten() {
    VALUE = VALUE + 10;
}

#[entry]
fn main() -> ! {
    VALUE = 0;
    increment();
    add_ten();

    loop {
        increment();
        asm!("WAI");
    }
}
'''


class TestProgramWithFunctions:
    """Tests for programs with multiple functions."""

    def test_compile_with_functions(self):
        """Test compilation of program with functions."""
        assembly = compile_r65_source(PROGRAM_WITH_FUNCTIONS)

        # Main must always be present as entry point
        assert "main:" in assembly

        # increment is called multiple times, should not be inlined (unless very small)
        # Check that either the function exists OR it was inlined
        assert "increment:" in assembly or "JSR increment" not in assembly, \
            "increment should either exist as a function or be fully inlined"

        # add_ten is called once, so it may be inlined
        # Just verify the program compiles and produces some output
        assert len(assembly) > 0

    @pytest.mark.skipif(
        not (WLA_AVAILABLE and WLALINK_AVAILABLE),
        reason="wla-65816 or wlalink not available"
    )
    def test_link_with_functions(self):
        """Test that program with functions links successfully."""
        assembly = compile_r65_source(PROGRAM_WITH_FUNCTIONS)

        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)
            rom_file = assemble_and_link(assembly, workdir)

            assert rom_file.exists()
            assert rom_file.stat().st_size == 262144  # 256KB ROM (minimum)


# =============================================================================
# Program with Control Flow
# =============================================================================

PROGRAM_WITH_CONTROL_FLOW = '''
#[zeropage(0x10)]
static mut XPOS: u8 = 0;

#[zeropage(0x11)]
static mut YPOS: u8 = 0;

fn process() {
    if XPOS > 10 {
        YPOS = 1;
    } else {
        YPOS = 0;
    }

    while XPOS > 0 {
        XPOS = XPOS - 1;
    }
}

#[entry]
fn main() -> ! {
    XPOS = 20;
    process();
}
'''


class TestProgramWithControlFlow:
    """Tests for programs with control flow."""

    def test_compile_with_control_flow(self):
        """Test compilation of program with control flow."""
        assembly = compile_r65_source(PROGRAM_WITH_CONTROL_FLOW)

        # Main must always be present as entry point
        assert "main:" in assembly
        # process() is called once, so it may be inlined
        # Just verify the program compiles
        assert len(assembly) > 0
        # Should have branch instructions (either from process or inlined)
        assert "BEQ" in assembly or "BNE" in assembly or "BCC" in assembly or "BCS" in assembly

    @pytest.mark.skipif(
        not (WLA_AVAILABLE and WLALINK_AVAILABLE),
        reason="wla-65816 or wlalink not available"
    )
    def test_link_with_control_flow(self):
        """Test that program with control flow links successfully."""
        assembly = compile_r65_source(PROGRAM_WITH_CONTROL_FLOW)

        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)
            rom_file = assemble_and_link(assembly, workdir)

            assert rom_file.exists()
            assert rom_file.stat().st_size == 262144  # 256KB ROM (minimum)


# =============================================================================
# Standalone test runner
# =============================================================================

if __name__ == "__main__":
    print(f"WLA-65816 available: {WLA_AVAILABLE}")
    print(f"WLALINK available: {WLALINK_AVAILABLE}")

    if WLA_AVAILABLE and WLALINK_AVAILABLE:
        print("\nRunning full end-to-end test...")
        assembly = compile_r65_source(MINIMAL_PROGRAM)
        print(f"Generated {len(assembly)} bytes of assembly")

        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)
            rom_file = assemble_and_link(assembly, workdir)
            print(f"Created ROM: {rom_file} ({rom_file.stat().st_size} bytes)")
            print("SUCCESS!")
    else:
        print("\nWLA-DX tools not available, running compile-only test...")
        assembly = compile_r65_source(MINIMAL_PROGRAM)
        print(f"Generated {len(assembly)} bytes of assembly")
        print("Compile test passed!")
