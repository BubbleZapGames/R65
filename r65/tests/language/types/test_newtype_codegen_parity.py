# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""Differential codegen parity: a newtype must compile to its payload's code.

A newtype is nominal at compile time and its payload at runtime, so for any
program, substituting `struct N(P);` for `P` must not change a single emitted
instruction. That property is worth testing directly rather than case by case,
because the way it breaks is a *silent miscompile*: much of codegen asks machine
questions by duck-typing (`hasattr(t, 'name')`, `str(t) in ('u8', ...)`), and a
newtype that reaches such a site answers with its own name — never in the
expected set — so the site quietly concludes "8-bit" or "2 bytes" and emits the
wrong instruction rather than failing.

Each case below is written once, then rendered twice: `{T}` is the type name,
`{W}(x)` wraps a value, `{U}` unwraps one. The raw rendering substitutes the
payload, an empty wrap, and an empty unwrap; the newtype rendering substitutes
the newtype, its constructor, and `.0`. The two assembly outputs must match.

Every shape here was added because it caught a real bug, or guards a predicate
that was found wrong during review.
"""

import re
import pytest

from r65.compiler.frontend import parse
from r65.compiler.frontend.preprocessor import preprocess
from r65.compiler.frontend.macros import expand_macros
from r65.compiler.hir import HIRBuilder
from r65.compiler.typeck import TypeChecker
from r65.compiler.mir.builder import MIRBuilder
from r65.compiler.codegen.codegen import ProgramCodeGenerator
from r65.compiler.analysis import RecursionChecker


def compile_to_asm(source: str) -> str:
    program = expand_macros(preprocess(parse(source, "test.r65"), "test.r65"))
    hir_prog = HIRBuilder(source_file="test.r65").build_program(program)
    TypeChecker(hir_prog).check()
    mir_prog = MIRBuilder().build_program(hir_prog)
    RecursionChecker(mir_prog).check()
    return ProgramCodeGenerator().generate(mir_prog)


def instructions(asm: str):
    """Emitted instructions only, with generated label numbering canonicalized.

    Comments carry type names, and the global label counter is bumped by the
    extra `struct N(P);` declaration itself — neither is a codegen difference.
    """
    out = []
    for line in asm.splitlines():
        line = line.split(';')[0].strip()
        if not line or line.startswith('.'):
            continue
        out.append(re.sub(r'\s+', ' ', line))

    text = "\n".join(out)
    # Renumber compiler-generated labels by order of first appearance.
    mapping = {}
    for name in re.findall(r'\b__SCMP\d+\b', text):
        mapping.setdefault(name, f"__SCMP{len(mapping)}")
    for old, new in mapping.items():
        text = re.sub(rf'\b{old}\b', new, text)
    return text.split("\n")


def render(case: str, decl: str, type_name: str, wrap: str, unwrap: str) -> str:
    body = case.replace("{T}", type_name).replace("{W}", wrap).replace("{U}", unwrap)
    return decl + body


# `{T}` type, `{W}(v)` wrap, `{U}` unwrap. Payload-agnostic so each runs for
# every width; nothing here may depend on a specific payload's range.
CASES = {
    "static round trip": '''
        #[zeropage(0x10)] static mut V: {T};
        #[zeropage(0x20)] static mut O: {T};
        #[entry] fn main() { V = {W}(5); O = V; }
    ''',
    "arithmetic chain": '''
        #[zeropage(0x10)] static mut V: {T};
        #[entry] fn main() { V = {W}(5); V = V + 3; V = V - 1; V = V & 0x0F; V = V << 1; }
    ''',
    "unary operators": '''
        #[zeropage(0x10)] static mut V: {T};
        #[entry] fn main() { V = {W}(5); V = ~V; V = V ^ 0x03; }
    ''',
    # Guards codegen/compare_select.py::_is_signed_type and
    # control_flow_select.py::_is_signed_comparison. Comparing an all-ones value
    # against zero is what discriminates: signed takes BMI/BPL, unsigned BCC/BCS.
    # A comparison against a positive constant does not, and will not catch a
    # regression here.
    "signed comparison against zero": '''
        #[zeropage(0x10)] static mut V: {T};
        #[zeropage(0x30)] static mut F: u8;
        #[entry] fn main() {
            V = {W}(0);
            V = V - 1;
            if V < {W}(0) { F = 1; } else { F = 2; }
        }
    ''',
    "loop with compare": '''
        #[zeropage(0x10)] static mut V: {T};
        #[entry] fn main() { V = {W}(0); while V < {W}(10) { V = V + 1; } }
    ''',
    # Guards codegen/constants.py::_is_8bit_type — the second 8-bit return value
    # rides in B, and the register order is chosen from the type's width.
    "multi-return through the B slot": '''
        #[zeropage(0x10)] static mut FIRST: {T};
        #[zeropage(0x20)] static mut SECOND: {T};
        fn pair() -> {T}, {T} { return {W}(3), {W}(4); }
        #[entry] fn main() { let a, b = pair(); FIRST = a; SECOND = b; }
    ''',
    # Guards hir/builder.py::_infer_entry_mode / _infer_exit_mode.
    "register param and return": '''
        #[zeropage(0x10)] static mut O: {T};
        fn bump(a @ A: {T}) -> {T} { return a + 1; }
        #[entry] fn main() { O = bump({W}(7)); }
    ''',
    "stack params": '''
        #[zeropage(0x10)] static mut O: {T};
        fn add(a: {T}, b: {T}) -> {T} { return a + b; }
        #[entry] fn main() { O = add({W}(7), {W}(9)); }
    ''',
    "struct field": '''
        struct Holder { tag: u8, v: {T} }
        #[zeropage(0x10)] static mut H: Holder;
        #[zeropage(0x20)] static mut O: {T};
        #[entry] fn main() { H.tag = 1; H.v = {W}(4); O = H.v + 1; }
    ''',
    "array element": '''
        #[ram] static mut ARR: [{T}; 8];
        #[zeropage(0x10)] static mut O: {T};
        #[entry] fn main() { ARR[3] = {W}(9); O = ARR[3]; }
    ''',
    "array with a runtime index": '''
        #[ram] static mut ARR: [{T}; 8];
        #[zeropage(0x10)] static mut O: {T};
        #[zeropage(0x30)] static mut I: u16;
        #[entry] fn main() { I = 3; ARR[I] = {W}(9); O = ARR[I]; }
    ''',
    "compound assignment": '''
        #[zeropage(0x10)] static mut V: {T};
        #[entry] fn main() { V = {W}(4); V += 3; V -= 1; V <<= 1; V |= 1; V &= 0x7F; V ^= 2; }
    ''',
    "interrupt handler": '''
        #[zeropage(0x10)] static mut V: {T};
        #[interrupt(nmi)] fn on_vblank() { V = V + 1; }
        #[entry] fn main() { V = {W}(0); }
    ''',
    "far function call": '''
        #[zeropage(0x10)] static mut O: {T};
        #[bank(1)] far fn compute(a: {T}) -> {T} { return a + 1; }
        #[entry] fn main() { O = compute({W}(3)); }
    ''',
    "function pointer": '''
        #[zeropage(0x10)] static mut O: {T};
        fn inc(a: {T}) -> {T} { return a + 1; }
        #[entry] fn main() { let f: fn({T}) -> {T} = inc; O = f({W}(3)); }
    ''',
    "preserved registers": '''
        #[zeropage(0x10)] static mut O: {T};
        #[preserves(X, Y)] fn work(a: {T}) -> {T} { X = 1; Y = 2; return a + 1; }
        #[entry] fn main() { O = work({W}(3)); }
    ''',
    "static initializer data": '''
        static TABLE: [{T}; 4] = [{W}(1), {W}(2), {W}(3), {W}(4)];
        #[zeropage(0x10)] static mut O: {T};
        #[entry] fn main() { O = TABLE[2]; }
    ''',
    "nested call arguments": '''
        #[zeropage(0x10)] static mut O: {T};
        fn a1(v: {T}) -> {T} { return v + 1; }
        fn a2(v: {T}) -> {T} { return v + 2; }
        #[entry] fn main() { O = a1(a2({W}(1))); }
    ''',
}

PAYLOADS = [("u8", "Byte"), ("i8", "Sbyte"), ("u16", "Word"), ("i16", "Sword")]

# A few cases only render to *equivalent* programs at some widths. Returning a
# `u8` local from a `-> u16` function is accepted (return types are unchecked),
# but the newtype spelling `Word(a)` is a checked widening — so the two sources
# genuinely differ at 16 bits. Restrict rather than weaken the comparison.
CASE_PAYLOADS = {
    # A 2-byte second return value cannot ride in B.
    "multi-return through the B slot": {"u8", "i8"},
}


@pytest.mark.parametrize("payload,newtype", PAYLOADS, ids=[p for p, _ in PAYLOADS])
@pytest.mark.parametrize("case_name", sorted(CASES))
def test_newtype_matches_payload_codegen(case_name, payload, newtype):
    allowed = CASE_PAYLOADS.get(case_name)
    if allowed is not None and payload not in allowed:
        pytest.skip(f"{case_name!r} renders to different programs for {payload}")
    case = CASES[case_name]
    raw = compile_to_asm(render(case, "", payload, "", ""))
    wrapped = compile_to_asm(
        render(case, f"struct {newtype}({payload});\n", newtype, newtype, ".0"))

    raw_instrs, new_instrs = instructions(raw), instructions(wrapped)
    if raw_instrs != new_instrs:
        import difflib
        diff = "\n".join(difflib.unified_diff(
            raw_instrs, new_instrs, fromfile=payload, tofile=newtype, lineterm=""))
        pytest.fail(
            f"newtype '{newtype}' does not compile identically to '{payload}' "
            f"for case {case_name!r}:\n{diff}")


class TestReturnValueTransferredToIndexRegister:
    """A u8 return moved from A into X/Y must be masked first.

    `_return_value_sizes` measured the type by *name*, so a u8 newtype came out
    2 bytes and the `AND #$00FF` was skipped — leaving whatever the callee left
    in B as the high byte of the index register. Parity against the raw payload
    is not the right assertion here: the two renderings allocate registers
    differently, so the mask is checked directly.
    """

    SRC = '''
        struct TileId(u8);
        #[zeropage(0x10)] static mut O: u8;
        #[zeropage(0x30)] static mut SEED: u8;
        fn get() -> TileId { let mut a: u8 = SEED; while a > 3 { a = a - 1; } return TileId(a); }
        #[entry] fn main() {
            let mut t: TileId = get();
            while t < TileId(10) { t = t + 1; }
            O = t.0;
        }
    '''

    def test_high_byte_is_masked_before_the_transfer(self):
        instrs = instructions(compile_to_asm(self.SRC))
        assert any(i.startswith("TAY") or i.startswith("TAX") for i in instrs), \
            "expected the u8 return to be transferred into an index register"
        transfer = next(n for n, i in enumerate(instrs)
                        if i.startswith("TAY") or i.startswith("TAX"))
        assert "AND #$FF" in instrs[max(0, transfer - 3):transfer], \
            f"missing zero-extend before the transfer:\n" + "\n".join(instrs[:transfer + 1])


class TestEntryFunctionStaysInM16:
    """An `#[entry]` function whose A binding is 16-bit must not exit to m8.

    The check reads `.name` off the binding type, so a newtype silently looked
    8-bit and the entry emitted a `SEP #$20` the raw payload does not.
    """

    SRC = '''
        {D}
        #[ram] static mut DATA: [u8; 256];
        #[entry] fn main() { let n @ A : {T} = {W}(DATA.len()); }
    '''

    def test_matches_raw_u16_binding(self):
        raw = compile_to_asm(self.SRC.replace("{D}", "").replace("{T}", "u16").replace("{W}", ""))
        wrapped = compile_to_asm(self.SRC.replace("{D}", "struct Count(u16);")
                                 .replace("{T}", "Count").replace("{W}", "Count"))
        assert instructions(raw) == instructions(wrapped)


class TestShiftWidenedToWiderDestination:
    """A shift whose destination is wider than its operand must widen first.

    The guard that does this was gated on the destination being a BasicTypeInfo,
    so a newtype destination skipped it: `n << 2` computed in m8 and stored a
    stale high byte into two bytes — not even a deterministic truncation.
    """

    SRC = '''
        {D}
        #[zeropage(0x10)] static mut O: {T};
        #[zeropage(0x30)] static mut N: u8;
        #[entry] fn main() { N = 0x41; let n: u8 = N; O = n << 2; }
    '''

    def test_matches_raw_u16_destination(self):
        raw = compile_to_asm(self.SRC.replace("{D}", "").replace("{T}", "u16"))
        wrapped = compile_to_asm(
            self.SRC.replace("{D}", "struct Addr(u16);").replace("{T}", "Addr"))
        assert instructions(raw) == instructions(wrapped)


class TestBoolPayloadParity:
    """`bool` is not in PAYLOADS — it has no arithmetic — but its cast path has
    its own normalization step that a newtype must not skip."""

    SRC = '''
        {D}
        #[zeropage(0x10)] static mut N: u8;
        #[zeropage(0x11)] static mut F: {T};
        #[entry] fn main() { N = 5; F = N as {T}; }
    '''

    def test_cast_to_bool_normalizes(self):
        raw = compile_to_asm(self.SRC.replace("{D}", "").replace("{T}", "bool"))
        wrapped = compile_to_asm(
            self.SRC.replace("{D}", "struct Flag(bool);").replace("{T}", "Flag"))
        assert instructions(raw) == instructions(wrapped)
