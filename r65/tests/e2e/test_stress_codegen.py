# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Codegen stress tests targeting high-risk register/stack interactions.

Each test exercises a specific codegen weak point with hand-computed
expected values stored in zeropage statics for precise verification.
"""

from pathlib import Path

import pytest

from r65.tests.e2e import ExpectedState

STDLIB_DIR = Path(__file__).parent.parent.parent.parent / "stdlib"
SNESLIB_PATH = STDLIB_DIR / "sneslib.r65"
MATH_PATH = STDLIB_DIR / "math.r65"

SCRATCH_DECLS = '''
    #[zeropage(0x02, register)]
    static mut SCRATCH0: u8;
    #[zeropage(0x04, register)]
    static mut SCRATCH1: u16;
    #[zeropage(0x06, register)]
    static mut SCRATCH2: u16;
    #[zeropage(0x08, register)]
    static mut SCRATCH3: u8;
'''


class TestDeepCallChain:
    """4-level @A call chain forcing A spill/restore at every level."""

    def test_deep_call_chain(self, e2e):
        """main -> level3(10) -> level2 -> level1 -> leaf.
        leaf(x) = x+1, level1(x) = leaf(x)+2, level2(x) = level1(x)+4,
        level3(x) = level2(x)+8. level3(10) = 10+1+2+4+8 = 25.
        """
        result = e2e.run(f'''
            {SCRATCH_DECLS}

            #[zeropage(0x10)]
            static mut RESULT: u8;

            fn leaf(val @ A: u8) -> u8 {{
                return A + 1;
            }}

            fn level1(val @ A: u8) -> u8 {{
                A = leaf(val);
                return A + 2;
            }}

            fn level2(val @ A: u8) -> u8 {{
                A = level1(val);
                return A + 4;
            }}

            fn level3(val @ A: u8) -> u8 {{
                A = level2(val);
                return A + 8;
            }}

            #[entry]
            fn main() {{
                RESULT = level3(10);
            }}
        ''', ExpectedState(memory={
            0x7E0010: 25,  # 10+1+2+4+8
        }))
        assert result.success, f"Failures: {result.failures}"


class TestRegionSpillAcrossThreeCalls:
    """Live A value must survive across 3 function calls via region spilling."""

    def test_region_spill_across_three_calls(self, e2e):
        """value @ A: u8 = 42 must survive calls to fn_a, fn_b, fn_c."""
        result = e2e.run(f'''
            {SCRATCH_DECLS}

            #[zeropage(0x10)]
            static mut RET_A: u8;
            #[zeropage(0x11)]
            static mut RET_B: u8;
            #[zeropage(0x12)]
            static mut RET_C: u8;
            #[zeropage(0x13)]
            static mut ORIGINAL: u8;

            fn fn_a(x @ A: u8) -> u8 {{
                return A + 10;
            }}

            fn fn_b(x @ A: u8) -> u8 {{
                return A + 20;
            }}

            fn fn_c(x @ A: u8) -> u8 {{
                return A + 30;
            }}

            fn do_three_calls(value @ A: u8) {{
                // value must be spilled across all 3 calls
                RET_A = fn_a(5);
                RET_B = fn_b(6);
                RET_C = fn_c(7);
                ORIGINAL = value;
            }}

            #[entry]
            fn main() {{
                do_three_calls(42);
            }}
        ''', ExpectedState(memory={
            0x7E0010: 15,   # fn_a(5) = 15
            0x7E0011: 26,   # fn_b(6) = 26
            0x7E0012: 37,   # fn_c(7) = 37
            0x7E0013: 42,   # original value preserved
        }))
        assert result.success, f"Failures: {result.failures}"


class TestForLoopWithCalls:
    """X loop counter must survive function calls inside loop body."""

    def test_for_loop_with_calls(self, e2e):
        """for i in 0..5 { SUM += add_ten(i); } => 10+11+12+13+14 = 60."""
        result = e2e.run(f'''
            {SCRATCH_DECLS}

            #[zeropage(0x10)]
            static mut SUM: u8;
            #[zeropage(0x11)]
            static mut TEMP: u8;

            #[inline(never)]
            fn add_ten(val @ A: u8) -> u8 {{
                return A + 10;
            }}

            #[entry]
            fn main() {{
                SUM = 0;
                for i in 0..5 {{
                    TEMP = add_ten(i as u8);
                    SUM = SUM + TEMP;
                }}
            }}
        ''', ExpectedState(memory={
            0x7E0010: 60,  # 10+11+12+13+14
        }))
        assert result.success, f"Failures: {result.failures}"


class TestThreeStackOneRegisterParam:
    """3 stack params + 1 register param: codegen must track SP offset drift."""

    def test_three_stack_one_register_param(self, e2e):
        """combine(10, 20, 30, 40) = 10+20+30+40 = 100."""
        result = e2e.run(f'''
            {SCRATCH_DECLS}

            #[zeropage(0x10)]
            static mut RESULT: u8;

            fn combine(a: u8, b: u8, c: u8, factor @ A: u8) -> u8 {{
                A = a + b;
                A = A + c;
                A = A + factor;
                return A;
            }}

            #[entry]
            fn main() {{
                RESULT = combine(10, 20, 30, 40);
            }}
        ''', ExpectedState(memory={
            0x7E0010: 100,
        }))
        assert result.success, f"Failures: {result.failures}"


class TestIfElseOneBranchCalls:
    """Asymmetric spilling: call in one branch, none in the other."""

    def test_if_branch_with_call(self, e2e):
        """flag=1 path: A = double(21) = 42."""
        result = e2e.run(f'''
            {SCRATCH_DECLS}

            #[zeropage(0x10)]
            static mut RESULT: u8;
            #[zeropage(0x11)]
            static mut FLAG: u8;

            fn double(val @ A: u8) -> u8 {{
                return A + A;
            }}

            #[entry]
            fn main() {{
                FLAG = 1;
                if FLAG != 0 {{
                    RESULT = double(21);
                }} else {{
                    RESULT = 99;
                }}
            }}
        ''', ExpectedState(memory={
            0x7E0010: 42,
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_else_branch_no_call(self, e2e):
        """flag=0 path: RESULT = 99 (no call)."""
        result = e2e.run(f'''
            {SCRATCH_DECLS}

            #[zeropage(0x10)]
            static mut RESULT: u8;
            #[zeropage(0x11)]
            static mut FLAG: u8;

            fn double(val @ A: u8) -> u8 {{
                return A + A;
            }}

            #[entry]
            fn main() {{
                FLAG = 0;
                if FLAG != 0 {{
                    RESULT = double(21);
                }} else {{
                    RESULT = 99;
                }}
            }}
        ''', ExpectedState(memory={
            0x7E0010: 99,
        }))
        assert result.success, f"Failures: {result.failures}"


class TestChainedMul8Calls:
    """Two consecutive mul8() calls with B register + u16 returns."""

    def test_chained_mul8_calls(self, e2e):
        """mul8(10,20)=200; mul8(3,7)=21."""
        source = f'''
            include!("{SNESLIB_PATH}")
            include!("{MATH_PATH}")

            #[zeropage(0x10)]
            static mut R1: [u8; 2];
            #[zeropage(0x12)]
            static mut R2: [u8; 2];

            #[entry]
            fn main() {{
                let r1: u16 = mul8(10, 20);
                R1[0] = r1 as u8;
                R1[1] = (r1 >> 8) as u8;
                let r2: u16 = mul8(3, 7);
                R2[0] = r2 as u8;
                R2[1] = (r2 >> 8) as u8;
            }}
        '''
        result = e2e.run(source, ExpectedState(memory={
            0x7E0010: 0xC8,  # 200 low byte
            0x7E0011: 0x00,  # 200 high byte
            0x7E0012: 0x15,  # 21 low byte
            0x7E0013: 0x00,  # 21 high byte
        }))
        assert result.success, f"Failures: {result.failures}"


class TestTripleCallChainForwarding:
    """Return value in A flows through sequential calls: result-as-argument."""

    def test_triple_call_chain_forwarding(self, e2e):
        """add_three(7)=10, double(10)=20, add_five(20)=25."""
        result = e2e.run(f'''
            {SCRATCH_DECLS}

            #[zeropage(0x10)]
            static mut INTERMEDIATE: u8;
            #[zeropage(0x11)]
            static mut FINAL: u8;

            #[inline(never)]
            fn add_three(val @ A: u8) -> u8 {{
                return A + 3;
            }}

            #[inline(never)]
            fn double(val @ A: u8) -> u8 {{
                return A + A;
            }}

            #[inline(never)]
            fn add_five(val @ A: u8) -> u8 {{
                return A + 5;
            }}

            #[entry]
            fn main() {{
                A = add_three(7);   // A = 10
                A = double(A);      // A = 20
                INTERMEDIATE = A;
                A = add_five(INTERMEDIATE);  // add_five(20) = 25
                FINAL = A;
            }}
        ''', ExpectedState(memory={
            0x7E0010: 20,  # intermediate after double
            0x7E0011: 25,  # final after add_five
        }))
        assert result.success, f"Failures: {result.failures}"


class TestMultiReturnAXWithLiveY:
    """rA, rX multi-return with Y preserved across the call."""

    def test_multi_return_a_x_with_live_y(self, e2e):
        """fn returns (21, 0x1600) in A,X. Caller has Y=0xAAAA with #[preserves(Y)]."""
        result = e2e.run(f'''
            {SCRATCH_DECLS}

            #[zeropage(0x10)]
            static mut RET_A: u8;
            #[zeropage(0x11)]
            static mut RET_X: u16;

            #[preserves(Y)]
            fn make_pair(a_val @ A: u8) -> rA, rX {{
                X = 0x1600;
                Y = 0xFFFF;  // trash Y, should be restored by #[preserves]
                return A, X;
            }}

            #[entry]
            fn main() {{
                Y = 0xAAAA;
                let mut a: u8;
                let mut x: u16;
                a, x = make_pair(21);
                RET_A = a;
                RET_X = x;
            }}
        ''', ExpectedState(
            Y=0xAAAA,
            memory={
                0x7E0010: 21,
                0x7E0011: [0x00, 0x16],  # 0x1600 LE
            },
        ))
        assert result.success, f"Failures: {result.failures}"


class TestPerCallXYSpillReload:
    """
    Regression: per-call X/Y spill (PHY) with no corresponding PLY.

    When a vreg is allocated to Y and lives across a call, but the ClobberRegionAnalyzer
    finds no region (because it only tracks direct HardwareRegister usage, not vreg-to-hw
    allocations), the per-call fallback in _compute_hw_spills emits PHY. But
    _compute_hw_reloads only checked active regions for X/Y reload, never finding one.
    Result: PHY without PLY corrupts the stack, especially in loops where it accumulates.
    """

    def test_y_spill_in_loop_with_call(self, e2e):
        """Y-allocated loop variable must survive calls that clobber Y.

        put_num-style pattern: Y holds a value, loop calls mod16/div16 which
        clobber Y. Without per-call PLY, each iteration pushes 2 bytes onto
        the stack without popping, corrupting the return address.
        """
        result = e2e.run(f'''
            {SCRATCH_DECLS}

            #[zeropage(0x10)]
            static mut RESULT: u16;

            fn halve(val @ A: u16) -> u16 {{
                return A / 2;
            }}

            fn accumulate(count: u16, start: u16) {{
                let mut value: u16 = start;
                let mut i: u16 = count;
                let mut sum: u16 = 0;
                loop {{
                    if i == 0 {{ break; }}
                    i--;
                    // Call that clobbers Y — value (in Y) must be preserved
                    sum = sum + halve(value);
                    value = value + 2;
                }}
                RESULT = sum;
            }}

            #[entry]
            fn main() {{
                // halve(10)+halve(12)+halve(14)+halve(16) = 5+6+7+8 = 26
                accumulate(4, 10);
            }}
        ''', ExpectedState(memory={
            0x7E0010: [26, 0],  # sum = 26
        }))
        assert result.success, f"Failures: {result.failures}"


class TestThreeRegParamPrologue:
    """Regression: TAY in prologue clobbers Y parameter when all 3 regs have params.

    When a function has params in A, X, and Y, and the frame is large enough
    to require TSC/SBC/TCS allocation (which clobbers A), the compiler saves
    A via TAY before frame alloc. But TAY overwrites Y's parameter value.
    Fix: use push-based frame allocation (PHX/PHY) which doesn't clobber any register.
    """

    def test_three_reg_params_preserved(self, e2e):
        """fn(A=5, X=0x100, Y=0x200) must see all three values correctly."""
        result = e2e.run(f'''
            {SCRATCH_DECLS}

            #[zeropage(0x10)]
            static mut RES_A: u8;
            #[zeropage(0x11)]
            static mut RES_X: u16;
            #[zeropage(0x13)]
            static mut RES_Y: u16;

            far fn use_all_three(val @ A: u8, addr @ X: u16, size @ Y: u16) {{
                // Locals force a large frame (>8 bytes) to trigger TSC/SBC/TCS
                let a: u16 = 0;
                let b: u16 = 0;
                let c: u16 = 0;
                let d: u16 = 0;
                let e: u16 = 0;
                RES_A = val;
                RES_X = addr;
                RES_Y = size;
            }}

            #[entry]
            fn main() {{
                use_all_three(5, 0x100, 0x200);
            }}
        ''', ExpectedState(memory={
            0x7E0010: 5,
            0x7E0011: [0x00, 0x01],  # 0x100 LE
            0x7E0013: [0x00, 0x02],  # 0x200 LE
        }))
        assert result.success, f"Failures: {result.failures}"


class TestCallReturnValueAcrossCall:
    """Regression: return values from calls must survive across subsequent calls.

    The hw coalescence pass incorrectly treated Call instructions as no-ops
    in the two-pass mechanism, allowing both call results to be coalesced to A.
    The comparison then compared A with itself (always equal).
    """

    def test_two_calls_compare_greater(self, e2e):
        """Call results compared: double(3)=6 > double(2)=4 should be true."""
        result = e2e.run(f'''
            {SCRATCH_DECLS}

            #[zeropage(0x10)]
            static mut RESULT: u8;

            fn double(val @ A: u8) -> u8 {{
                return A + A;
            }}

            fn is_first_greater(a: u8, b: u8) -> u8 {{
                let key_a: u8 = double(a);
                let key_b: u8 = double(b);
                if key_a > key_b {{
                    return 1;
                }}
                return 0;
            }}

            #[entry]
            fn main() {{
                RESULT = is_first_greater(3, 2);
            }}
        ''', ExpectedState(memory={
            0x7E0010: 1,  # 6 > 4 = true
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_two_calls_compare_not_greater(self, e2e):
        """Call results compared: double(2)=4 > double(3)=6 should be false."""
        result = e2e.run(f'''
            {SCRATCH_DECLS}

            #[zeropage(0x10)]
            static mut RESULT: u8;

            fn double(val @ A: u8) -> u8 {{
                return A + A;
            }}

            fn is_first_greater(a: u8, b: u8) -> u8 {{
                let key_a: u8 = double(a);
                let key_b: u8 = double(b);
                if key_a > key_b {{
                    return 1;
                }}
                return 0;
            }}

            #[entry]
            fn main() {{
                RESULT = is_first_greater(2, 3);
            }}
        ''', ExpectedState(memory={
            0x7E0010: 0,  # 4 > 6 = false
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_call_result_survives_recursive_call(self, e2e):
        """Call result used after recursive call must survive on stack.

        Pattern: pivot = partition(), then recursive call, then use pivot.
        Tests that cross-block liveness correctly prevents coalescence.
        """
        result = e2e.run(f'''
            {SCRATCH_DECLS}

            #[zeropage(0x10)]
            static mut RESULT: u8;

            fn compute(val @ A: u8) -> u8 {{
                return A + 10;
            }}

            fn accumulate(n: u8) -> u8 {{
                if n == 0 {{
                    return 0;
                }}
                let base: u8 = compute(n);
                let rest: u8 = accumulate(n - 1);
                // base must survive across the recursive call
                return base + rest;
            }}

            #[entry]
            fn main() {{
                // accumulate(3) = compute(3) + compute(2) + compute(1) + 0
                //               = 13 + 12 + 11 + 0 = 36
                RESULT = accumulate(3);
            }}
        ''', ExpectedState(memory={
            0x7E0010: 36,
        }))
        assert result.success, f"Failures: {result.failures}"


class TestStackParamCopiedToMutLocal:
    """Regression: `let mut v: u16 = value;` with `value` as a stack param.

    Discovered while debugging the classickong port: put_num(pos: u16,
    value: u16, digits: u8) opens with `let mut v: u16 = value;` and then
    reads `v` inside a loop that calls mod16/div16. The compiler allocates
    `v` to a frame slot but skips the parameter→local copy at function
    entry, so `v` reads whatever the prologue's PHX pushed onto that slot.
    The displayed number ends up unrelated to the input value.

    Pattern: 3 stack params (u16, u16, u8), middle param copied into a
    mutable local, local used across calls in a loop.
    """

    def test_mut_local_from_stack_param_used_in_loop(self, e2e):
        """3-stack-arg function (u16, u16, u8) where the middle u16 is
        copied into `let mut v: u16 = value;` and then read in a loop
        whose body calls non-foldable helpers that take v as input.

        Regression: the slot allocator's vreg coalescer would replace
        the preassigned param vreg with a local that gets re-defined
        in the loop, orphaning the caller-set value at the param slot.
        v's frame slot was left as the prologue's PHX value instead of
        the parameter, so each RESULTS[d] write stored 0 instead of
        the expected halved cascade.
        """
        result = e2e.run(f'''
            #[zeropage(0x10)]
            static mut RESULTS: [u16; 4];
            #[zeropage(0x20)]
            static mut SINK: u8;

            // Side effect on SINK keeps the optimizer from folding these
            // away. The helpers don't touch A before the return so the
            // u16 parameter survives the call body.
            fn extract_low(val @ A: u16) -> u8 {{
                SINK = SINK + 1;
                return SINK;
            }}

            fn halve(val @ A: u16) -> u16 {{
                SINK = SINK + 1;
                return A >> 1;
            }}

            far fn write_halves(pos: u16, value: u16, digits: u8) {{
                let mut d: u8 = digits;
                let mut v: u16 = value;
                loop {{
                    if d == 0 {{ break; }}
                    d--;
                    let _digit: u8 = extract_low(v);
                    v = halve(v);
                    RESULTS[d as u16] = v;
                }}
            }}

            #[entry]
            fn main() -> ! {{
                write_halves(0, 0xBEEF, 3);
                loop {{ asm!("WAI"); }}
            }}
        ''', ExpectedState(memory={
            # Successive halves of 0xBEEF written right-to-left:
            #   d=2: v = 0xBEEF >> 1 = 0x5F77
            #   d=1: v = 0x5F77 >> 1 = 0x2FBB
            #   d=0: v = 0x2FBB >> 1 = 0x17DD
            0x7E0010: [0xDD, 0x17],   # RESULTS[0]
            0x7E0012: [0xBB, 0x2F],   # RESULTS[1]
            0x7E0014: [0x77, 0x5F],   # RESULTS[2]
        }))
        assert result.success, f"Failures: {result.failures}"
