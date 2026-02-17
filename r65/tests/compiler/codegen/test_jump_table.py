"""Tests for jump table and lookup table code generation.

Tests that dense match expressions emit the correct assembly:
- LookupTable: LDA table,X with .DB/.DW for pure constant matches
- JumpTable: JMP (addr,X) with .DW for non-constant dense matches
- Branch chains: CMP/BEQ for sparse or small matches
"""

import pytest
from r65.compiler.main import compile_string


def get_function_asm(result: str, func_name: str) -> str:
    """Extract assembly lines for a single function (label to RTS/RTL)."""
    lines = result.split('\n')
    in_func = False
    func_lines = []
    for line in lines:
        if f'{func_name}:' in line and not in_func:
            in_func = True
            func_lines.append(line)
        elif in_func:
            func_lines.append(line)
            stripped = line.strip()
            if stripped.startswith('RTS') or stripped.startswith('RTL'):
                break
    return '\n'.join(func_lines)


def get_function_asm_with_data(result: str, func_name: str) -> str:
    """Extract assembly lines for a function, including trailing data after RTS.

    LookupTable emits .DB/.DW data after the final BRA, which may appear
    after the RTS in the output. We capture up to the next function label.
    """
    lines = result.split('\n')
    in_func = False
    past_rts = False
    func_lines = []
    for line in lines:
        if f'{func_name}:' in line and not in_func:
            in_func = True
            func_lines.append(line)
        elif in_func:
            stripped = line.strip()
            # Stop at next function label (not our own)
            if past_rts and stripped and not stripped.startswith('.') and not stripped.startswith(';') and ':' in stripped and not stripped.startswith('_'):
                break
            func_lines.append(line)
            if stripped.startswith('RTS') or stripped.startswith('RTL'):
                past_rts = True
    return '\n'.join(func_lines)


class TestLookupTableAssembly:
    """Tests that pure constant dense matches produce LookupTable assembly."""

    def test_u8_lut_emits_lda_table_x_and_db(self):
        """u8 LUT should emit LDA.w table,X with .DB entries."""
        source = """
        fn classify(val @ A: u8) -> u8 {
            let result: u8 = match val {
                0 => 10,
                1 => 20,
                2 => 30,
                _ => 0
            };
            return result;
        }
        """
        result = compile_string(source, "test.r65")
        func_asm = get_function_asm_with_data(result, "classify")

        assert "LDA.w" in func_asm, "LUT should emit LDA.w for absolute indexed"
        assert ",X" in func_asm, "Should use X-indexed addressing"
        assert ".DB" in func_asm, "u8 LUT should emit .DB table entries"
        assert "JMP (" not in func_asm, "LUT should NOT emit JMP (addr,X)"

    def test_u8_lut_omits_asl(self):
        """u8 LUT should not emit ASL (byte table, no index doubling needed)."""
        source = """
        fn classify(val @ A: u8) -> u8 {
            let result: u8 = match val {
                0 => 10,
                1 => 20,
                2 => 30,
                _ => 0
            };
            return result;
        }
        """
        result = compile_string(source, "test.r65")
        func_asm = get_function_asm_with_data(result, "classify")

        assert "ASL" not in func_asm, "u8 LUT should NOT emit ASL (byte-indexed)"

    def test_non_zero_base_emits_sec_sbc(self):
        """LUT with non-zero base should emit SEC + SBC."""
        source = """
        fn classify(val @ A: u8) -> u8 {
            let result: u8 = match val {
                5 => 10,
                6 => 20,
                7 => 30,
                _ => 0
            };
            return result;
        }
        """
        result = compile_string(source, "test.r65")
        func_asm = get_function_asm_with_data(result, "classify")

        assert "SEC" in func_asm, "Non-zero base should emit SEC"
        assert "SBC" in func_asm, "Non-zero base should emit SBC to subtract base"

    def test_zero_base_omits_sec_sbc(self):
        """LUT with zero base should not emit SEC/SBC."""
        source = """
        fn classify(val @ A: u8) -> u8 {
            let result: u8 = match val {
                0 => 10,
                1 => 20,
                2 => 30,
                _ => 0
            };
            return result;
        }
        """
        result = compile_string(source, "test.r65")
        func_asm = get_function_asm_with_data(result, "classify")

        assert "SEC" not in func_asm, "Zero base should not emit SEC"
        assert "SBC" not in func_asm, "Zero base should not emit SBC"

    def test_bounds_check_present(self):
        """LUT should emit CMP + BCS for upper bounds check."""
        source = """
        fn classify(val @ A: u8) -> u8 {
            let result: u8 = match val {
                0 => 10,
                1 => 20,
                2 => 30,
                _ => 0
            };
            return result;
        }
        """
        result = compile_string(source, "test.r65")
        func_asm = get_function_asm_with_data(result, "classify")

        assert "CMP" in func_asm, "Should emit CMP for bounds check"
        assert "BCS" in func_asm, "Should emit BCS to branch on out-of-bounds"

    def test_db_entry_count(self):
        """u8 LUT should have correct number of .DB entries."""
        source = """
        fn classify(val @ A: u8) -> u8 {
            let result: u8 = match val {
                0 => 10,
                1 => 20,
                2 => 30,
                3 => 40,
                4 => 50,
                _ => 0
            };
            return result;
        }
        """
        result = compile_string(source, "test.r65")
        func_asm = get_function_asm_with_data(result, "classify")

        db_count = func_asm.count(".DB")
        assert db_count == 5, f"Expected 5 .DB entries for 5-arm match, got {db_count}"

    def test_u16_lut_emits_dw_asl_rep_sep(self):
        """u16 LUT should emit .DW entries, ASL, REP/SEP mode switch."""
        source = """
        fn classify(val @ A: u8) -> u16 {
            let result: u16 = match val {
                0 => 1000,
                1 => 2000,
                2 => 3000,
                _ => 0
            };
            return result;
        }
        """
        result = compile_string(source, "test.r65")
        func_asm = get_function_asm_with_data(result, "classify")

        assert ".DW" in func_asm, "u16 LUT should emit .DW table entries"
        assert "ASL" in func_asm, "u16 LUT should emit ASL (word-indexed)"
        assert "REP" in func_asm, "u16 LUT should emit REP for m16 mode"
        assert "SEP" in func_asm, "u16 LUT should emit SEP to restore m8"

    def test_non_constant_arm_falls_back_to_jmp(self):
        """Non-constant arm should still use JMP (addr,X) jump table."""
        source = """
        fn identity(x @ A: u8) -> u8 { return x; }

        fn classify(val @ A: u8) -> u8 {
            let result: u8 = match val {
                0 => 10,
                1 => identity(20),
                2 => 30,
                _ => 0
            };
            return result;
        }
        """
        result = compile_string(source, "test.r65")
        func_asm = get_function_asm_with_data(result, "classify")

        assert "JMP (" in func_asm, "Non-constant arm should use JMP (addr,X)"
        assert ".DB" not in func_asm, "JumpTable should not emit .DB"


class TestJumpTableAssembly:
    """Tests that non-constant dense matches still produce JMP (addr,X) assembly."""

    def test_non_constant_dense_match_emits_jmp_indirect_x(self):
        """Dense non-constant match should emit JMP (label,X) with .DW table."""
        source = """
        fn identity(x @ A: u8) -> u8 { return x; }

        fn classify(val @ A: u8) -> u8 {
            let result: u8 = match val {
                0 => identity(10),
                1 => identity(20),
                2 => identity(30),
                _ => 0
            };
            return result;
        }
        """
        result = compile_string(source, "test.r65")
        func_asm = get_function_asm_with_data(result, "classify")

        assert "JMP (" in func_asm, "Dense non-constant match should emit JMP (addr,X)"
        assert ",X)" in func_asm, "Should use X-indexed indirect addressing"
        assert ".DW" in func_asm, "Should emit .DW table entries"
        assert "ASL" in func_asm, "Should emit ASL to multiply index by 2"

    def test_dw_entry_count(self):
        """JumpTable should have correct number of .DW entries."""
        source = """
        fn identity(x @ A: u8) -> u8 { return x; }

        fn classify(val @ A: u8) -> u8 {
            let result: u8 = match val {
                0 => identity(10),
                1 => identity(20),
                2 => identity(30),
                3 => identity(40),
                4 => identity(50),
                _ => 0
            };
            return result;
        }
        """
        result = compile_string(source, "test.r65")
        func_asm = get_function_asm_with_data(result, "classify")

        dw_count = func_asm.count(".DW")
        assert dw_count == 5, f"Expected 5 .DW entries for 5-arm match, got {dw_count}"


class TestBranchChainAssembly:
    """Tests that sparse/small matches use CMP/BEQ branch chains."""

    def test_sparse_match_uses_branch_chain(self):
        """Sparse patterns should NOT emit jump table, should use CMP/BEQ."""
        source = """
        fn classify(val @ A: u8) -> u8 {
            let result: u8 = match val {
                0 => 10,
                50 => 20,
                100 => 30,
                _ => 0
            };
            return result;
        }
        """
        result = compile_string(source, "test.r65")
        func_asm = get_function_asm(result, "classify")

        assert "JMP (" not in func_asm, "Sparse match should NOT emit JMP (addr,X)"
        assert ".DW" not in func_asm, "Sparse match should NOT emit .DW table"
        assert ".DB" not in func_asm, "Sparse match should NOT emit .DB table"
        assert "CMP" in func_asm, "Sparse match should use CMP comparisons"
        assert "BEQ" in func_asm or "BNE" in func_asm, "Sparse match should use conditional branches"

    def test_two_arm_match_uses_branch_chain(self):
        """2-arm match (below MIN_PATTERNS=3) should use branch chain."""
        source = """
        fn classify(val @ A: u8) -> u8 {
            let result: u8 = match val {
                0 => 10,
                1 => 20,
                _ => 0
            };
            return result;
        }
        """
        result = compile_string(source, "test.r65")
        func_asm = get_function_asm(result, "classify")

        assert "JMP (" not in func_asm, "2-arm match should NOT emit jump table"
        assert "CMP" in func_asm, "2-arm match should use CMP"
