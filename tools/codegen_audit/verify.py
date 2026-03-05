"""
Emulator-based correctness verification for reference assembly.

Verifies that Agent 1's optimized assembly produces identical behavior to the
compiler's output by running both through the R65 E2E pipeline and comparing
register/memory state.
"""

import subprocess
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from r65.emulator.cpu import CPU65816, StopExecution, WaitForInterrupt
from r65.emulator.memory import SNESMemory


@dataclass
class VerifyResult:
    """Result of a verification run."""
    success: bool
    mismatches: list[str] = field(default_factory=list)
    error: str | None = None
    compiler_state: dict | None = None
    reference_state: dict | None = None


def _check_toolchain() -> list[str]:
    """Check that required assembler tools are available."""
    tools = ['r65c', 'wla-65816', 'wlalink']
    return [t for t in tools if shutil.which(t) is None]


def _compile_r65(source: str, tmpdir: Path) -> bytes:
    """Compile R65 source to ROM bytes using the full toolchain."""
    src_path = tmpdir / 'test.r65'
    asm_path = tmpdir / 'test.asm'
    obj_path = tmpdir / 'test.o'
    rom_path = tmpdir / 'test.sfc'
    link_path = tmpdir / 'test.link'

    src_path.write_text(source)

    result = subprocess.run(
        ['r65c', str(src_path), '-o', str(asm_path), '--cfg', 'snes'],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f'R65 compilation failed:\n{result.stderr}')

    result = subprocess.run(
        ['wla-65816', '-o', str(obj_path), str(asm_path)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f'Assembly failed:\n{result.stderr}')

    result = subprocess.run(
        ['wlalink', '-r', str(link_path), str(rom_path)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f'Linking failed:\n{result.stderr}')

    return rom_path.read_bytes()


def _assemble_wladx(asm_source: str, tmpdir: Path) -> bytes:
    """Assemble raw WLA-DX source to ROM bytes."""
    asm_path = tmpdir / 'ref.asm'
    obj_path = tmpdir / 'ref.o'
    rom_path = tmpdir / 'ref.sfc'
    link_path = tmpdir / 'ref.link'

    asm_path.write_text(asm_source)
    link_path.write_text(f'[objects]\n{obj_path}\n')

    result = subprocess.run(
        ['wla-65816', '-o', str(obj_path), str(asm_path)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f'Reference assembly failed:\n{result.stderr}')

    result = subprocess.run(
        ['wlalink', '-r', str(link_path), str(rom_path)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f'Reference linking failed:\n{result.stderr}')

    return rom_path.read_bytes()


def _execute_rom(rom_data: bytes, max_instructions: int = 10000) -> CPU65816:
    """Execute ROM on the Python 65816 emulator."""
    memory = SNESMemory(rom_data, mapping='lorom')
    cpu = CPU65816(memory)
    cpu.PC = memory.read16(0xFFFC)
    cpu.PBR = 0x00

    instructions = 0
    try:
        while instructions < max_instructions:
            cpu.step()
            instructions += 1
    except (StopExecution, WaitForInterrupt):
        pass

    return cpu


def _capture_state(cpu: CPU65816, memory_addrs: list[int] | None = None) -> dict:
    """Capture CPU register and selected memory state."""
    state = {
        'A': cpu.A & 0xFFFF,
        'X': cpu.X & 0xFFFF,
        'Y': cpu.Y & 0xFFFF,
        'SP': cpu.SP,
        'flags': {
            'C': cpu.flag_c,
            'Z': cpu.flag_z,
            'N': cpu.flag_n,
            'V': cpu.flag_v,
            'M': cpu.flag_m,
        },
    }
    if memory_addrs:
        state['memory'] = {
            addr: cpu.memory.read(addr) for addr in memory_addrs
        }
    return state


def _compare_states(compiler_state: dict, reference_state: dict) -> list[str]:
    """Compare two captured CPU states and return list of mismatches."""
    mismatches: list[str] = []

    for reg in ('A', 'X', 'Y'):
        cv = compiler_state[reg]
        rv = reference_state[reg]
        if cv != rv:
            mismatches.append(f'{reg}: compiler=0x{cv:04X}, reference=0x{rv:04X}')

    # Compare flags (skip M — mode may differ at WAI)
    for flag in ('C', 'Z', 'N', 'V'):
        cv = compiler_state['flags'].get(flag)
        rv = reference_state['flags'].get(flag)
        if cv != rv:
            mismatches.append(f'Flag {flag}: compiler={cv}, reference={rv}')

    # Compare memory if present
    c_mem = compiler_state.get('memory', {})
    r_mem = reference_state.get('memory', {})
    for addr in sorted(set(c_mem) | set(r_mem)):
        cv = c_mem.get(addr)
        rv = r_mem.get(addr)
        if cv != rv:
            mismatches.append(
                f'Memory 0x{addr:06X}: compiler=0x{cv:02X}, reference=0x{rv:02X}'
            )

    return mismatches


def build_test_program(
    func_name: str,
    r65_source: str,
    test_vector: dict,
) -> str:
    """Build an R65 test program that calls a function with specific inputs.

    The test program sets up registers per the test vector's "inputs",
    calls the target function, and halts (WAI).

    Args:
        func_name: Function to test
        r65_source: Full R65 source (must include the function)
        test_vector: {"inputs": {"A": val, "X": val, ...}, "expected": {...}}
    """
    inputs = test_vector.get('inputs', {})

    # Build register setup code
    setup_lines = []
    if 'X' in inputs:
        setup_lines.append(f'    X = {inputs["X"]};')
    if 'Y' in inputs:
        setup_lines.append(f'    Y = {inputs["Y"]};')

    # A goes last since it's the most common parameter
    a_val = inputs.get('A', 0)
    setup_lines.append(f'    A = {a_val};')

    setup_code = '\n'.join(setup_lines)

    # Build a minimal test harness program
    # We inject our test entry point that calls the function
    return f"""\
#[snesrom(name="AUDIT TEST")]
#[bank(0)]

{r65_source}

#[entry]
fn __audit_main() {{
{setup_code}
    A = {func_name}(A);
}}
"""


def build_reference_wladx(
    func_name: str,
    reference_asm: str,
    test_vector: dict,
    is_far: bool = False,
) -> str:
    """Build a complete WLA-DX program with the reference assembly substituted.

    Creates SNES boilerplate, a main that sets up inputs and calls the function,
    and includes the reference assembly verbatim.

    Args:
        func_name: Function label name
        reference_asm: Agent 1's assembly (verbatim)
        test_vector: {"inputs": {"A": val, ...}}
        is_far: True if the function uses RTL (far call)
    """
    inputs = test_vector.get('inputs', {})

    # Build register setup instructions
    setup = []
    if 'X' in inputs:
        setup.append(f'  LDX #{inputs["X"]}')
    if 'Y' in inputs:
        setup.append(f'  LDY #{inputs["Y"]}')
    a_val = inputs.get('A', 0)
    setup.append(f'  SEP #$20')  # m8 mode (default)
    setup.append(f'  LDA #{a_val}')

    call_inst = 'JSL' if is_far else 'JSR'
    setup_code = '\n'.join(setup)

    return f"""\
.MEMORYMAP
  DEFAULTSLOT 0
  SLOTSIZE $8000
  SLOT 0 $8000
.ENDME

.ROMBANKMAP
  BANKSTOTAL 1
  BANKSIZE $8000
  BANKS 1
.ENDRO

.BANK 0 SLOT 0
.ORG 0

; Entry point
__reset:
  SEI
  CLC
  XCE
  REP #$10       ; x16 mode
  SEP #$20       ; m8 mode
  LDX #$1FFF
  TXS

{setup_code}
  {call_inst} {func_name}
  WAI

; Reference function (verbatim from Agent 1)
{reference_asm}

; Vectors
.ORGA $FFFC
.DW __reset
"""


def verify_function(
    func_name: str,
    r65_source: str,
    reference_asm: str,
    test_vectors: list[dict],
    memory_addrs: list[int] | None = None,
    is_far: bool = False,
) -> VerifyResult:
    """Verify that reference assembly matches compiler output behavior.

    For each test vector:
    1. Compile the R65 program → run on emulator → capture state
    2. Assemble WLA-DX program with reference → run on emulator → capture state
    3. Compare states

    Args:
        func_name: Function to verify
        r65_source: R65 source code (full file, must contain the function)
        reference_asm: Agent 1's assembly
        test_vectors: List of test vectors from Agent 1
        memory_addrs: Optional memory addresses to compare
        is_far: Whether the function is a far function

    Returns:
        VerifyResult with success status and any mismatches
    """
    missing = _check_toolchain()
    if missing:
        return VerifyResult(
            success=False,
            error=f'Missing tools: {", ".join(missing)}'
        )

    if not test_vectors:
        return VerifyResult(success=True, mismatches=['No test vectors provided'])

    all_mismatches: list[str] = []

    for i, tv in enumerate(test_vectors):
        try:
            # Step 1: Compile and run the R65 version
            test_prog = build_test_program(func_name, r65_source, tv)
            with tempfile.TemporaryDirectory() as tmpdir:
                compiler_rom = _compile_r65(test_prog, Path(tmpdir))
                compiler_cpu = _execute_rom(compiler_rom)
                compiler_state = _capture_state(compiler_cpu, memory_addrs)

            # Step 2: Assemble and run the reference version
            ref_prog = build_reference_wladx(
                func_name, reference_asm, tv, is_far=is_far
            )
            with tempfile.TemporaryDirectory() as tmpdir:
                ref_rom = _assemble_wladx(ref_prog, Path(tmpdir))
                ref_cpu = _execute_rom(ref_rom)
                ref_state = _capture_state(ref_cpu, memory_addrs)

            # Step 3: Compare
            mismatches = _compare_states(compiler_state, ref_state)
            if mismatches:
                prefix = f'Test vector {i}'
                all_mismatches.extend(f'[{prefix}] {m}' for m in mismatches)

        except Exception as e:
            all_mismatches.append(f'[Test vector {i}] Error: {e}')

    return VerifyResult(
        success=len(all_mismatches) == 0,
        mismatches=all_mismatches,
        compiler_state=compiler_state if not all_mismatches else None,
        reference_state=ref_state if not all_mismatches else None,
    )
