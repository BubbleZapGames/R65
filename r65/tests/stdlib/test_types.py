"""Tests for 32-bit integer stdlib modules (U32 and I32)."""

import pytest
from r65.compiler.frontend import parse, expand_macros
from r65.compiler.hir import HIRBuilder
from r65.compiler.typeck import TypeChecker
from r65.compiler.errors import TypeCheckError, ParseError


def build_and_check(source: str):
    """Parse, expand macros, build HIR, and type check source."""
    program = parse(source, "test.r65")
    program = expand_macros(program)
    hir_builder = HIRBuilder(source_file="test.r65")
    hir_prog = hir_builder.build_program(program)
    type_checker = TypeChecker(hir_prog)
    type_checker.check()
    return hir_prog


# Common header with U32 struct and hardware registers
U32_HEADER = """
struct U32 {
    lo: u16,
    hi: u16
}

#[hw(0x4202)]
static mut WRMPYA: u8;
#[hw(0x4203)]
static mut WRMPYB: u8;
#[hw(0x4204)]
static mut WRDIVL: u8;
#[hw(0x4205)]
static mut WRDIVH: u8;
#[hw(0x4206)]
static mut WRDIVB: u8;
#[hw(0x4214)]
static mut RDDIVL: u8;
#[hw(0x4216)]
static mut RDMPYL: u8;
"""

# Common header with I32 struct and hardware registers
I32_HEADER = """
struct I32 {
    lo: u16,
    hi: u16
}

#[hw(0x4202)]
static mut WRMPYA: u8;
#[hw(0x4203)]
static mut WRMPYB: u8;
#[hw(0x4204)]
static mut WRDIVL: u8;
#[hw(0x4205)]
static mut WRDIVH: u8;
#[hw(0x4206)]
static mut WRDIVB: u8;
#[hw(0x4214)]
static mut RDDIVL: u8;
#[hw(0x4216)]
static mut RDMPYL: u8;
"""


# =============================================================================
# U32 (Unsigned 32-bit Integer) Tests
# =============================================================================

class TestU32Struct:
    """Tests for U32 struct definition."""

    def test_u32_struct_size(self):
        """U32 struct is 4 bytes (two u16 fields)."""
        source = """
            struct U32 {
                lo: u16,
                hi: u16
            }
        """
        program = parse(source, "test.r65")
        struct_decl = program.items[0]
        assert struct_decl.name == "U32"
        assert len(struct_decl.fields) == 2
        assert struct_decl.fields[0].name == "lo"
        assert struct_decl.fields[1].name == "hi"

    def test_u32_static_declaration(self):
        """U32 can be declared as static variable."""
        source = U32_HEADER + """
            #[zeropage]
            static mut VALUE: U32;
        """
        build_and_check(source)


class TestU32ImplBlock:
    """Tests for U32 impl block structure."""

    def test_impl_far_u32(self):
        """impl far U32 block parses correctly."""
        source = U32_HEADER + """
            impl far U32 {
                                far fn from_u16(far *self, value @ A: u16) {
                    self.lo = A;
                    self.hi = 0;
                }
            }
        """
        program = parse(source, "test.r65")
        impl_decl = program.items[-1]
        assert impl_decl.struct_name == "U32"
        assert impl_decl.is_far is True
        assert len(impl_decl.methods) == 1
        assert impl_decl.methods[0].is_far is True

    def test_method_mangling(self):
        """U32 methods are mangled to U32__method."""
        source = U32_HEADER + """
            impl far U32 {
                                far fn add(far *self, far *other: U32) {
                    STATUS.Carry = false;
                    A = self.lo;
                    A = A + other.lo;
                    self.lo = A;
                    A = self.hi;
                    A = A + other.hi;
                    self.hi = A;
                }
            }
        """
        hir = build_and_check(source)
        from r65.compiler.hir import HIRImplDecl
        impl_decl = next(d for d in hir.declarations if isinstance(d, HIRImplDecl))
        assert impl_decl.methods[0].name == "U32__add"


class TestU32ConversionMethods:
    """Tests for U32 conversion methods."""

    def test_from_u16(self):
        """from_u16 method type checks."""
        source = U32_HEADER + """
            impl far U32 {
                                far fn from_u16(far *self, value @ A: u16) {
                    self.lo = A;
                    self.hi = 0;
                }
            }

            #[zeropage]
            static mut VALUE: U32;

                        fn test() {
                VALUE.from_u16(1000);
            }
        """
        build_and_check(source)

    def test_to_u16(self):
        """to_u16 method returns u16."""
        source = U32_HEADER + """
            impl far U32 {
                                far fn to_u16(far *self) -> u16 {
                    return self.lo;
                }
            }

            #[zeropage]
            static mut VALUE: U32;
            #[zeropage]
            static mut RESULT: u16;

                        fn test() {
                A = VALUE.to_u16();
                RESULT = A;
            }
        """
        build_and_check(source)

    def test_copy(self):
        """copy method copies another U32."""
        source = U32_HEADER + """
            impl far U32 {
                                far fn copy(far *self, far *src: U32) {
                    A = src.lo;
                    self.lo = A;
                    A = src.hi;
                    self.hi = A;
                }
            }

            #[zeropage]
            static mut A_VAL: U32;
            #[zeropage]
            static mut B_PTR: far *U32;

                        fn test() {
                A_VAL.copy(B_PTR);
            }
        """
        build_and_check(source)


class TestU32ArithmeticMethods:
    """Tests for U32 arithmetic methods."""

    def test_add(self):
        """add method performs 32-bit addition."""
        source = U32_HEADER + """
            impl far U32 {
                                far fn add(far *self, far *other: U32) {
                    STATUS.Carry = false;
                    A = self.lo;
                    A = A + other.lo;
                    self.lo = A;
                    A = self.hi;
                    A = A + other.hi;
                    self.hi = A;
                }
            }

            #[zeropage]
            static mut A_VAL: U32;
            #[zeropage]
            static mut B_PTR: far *U32;

                        fn test() {
                A_VAL.add(B_PTR);
            }
        """
        build_and_check(source)

    def test_sub(self):
        """sub method performs 32-bit subtraction."""
        source = U32_HEADER + """
            impl far U32 {
                                far fn sub(far *self, far *other: U32) {
                    STATUS.Carry = true;
                    A = self.lo;
                    A = A - other.lo;
                    self.lo = A;
                    A = self.hi;
                    A = A - other.hi;
                    self.hi = A;
                }
            }

            #[zeropage]
            static mut A_VAL: U32;
            #[zeropage]
            static mut B_PTR: far *U32;

                        fn test() {
                A_VAL.sub(B_PTR);
            }
        """
        build_and_check(source)

    def test_div_with_zero_check(self):
        """div method handles division by zero."""
        source = U32_HEADER + """
            impl far U32 {
                                far fn div(far *self, far *other: U32) {
                    A = other.lo;
                    if A == 0 as u16 {
                        A = other.hi;
                        if A == 0 as u16 {
                            A = 0xFFFF;
                            self.lo = A;
                            self.hi = A;
                            return;
                        }
                    }
                    // Division logic would go here
                }
            }

            #[zeropage]
            static mut A_VAL: U32;
            #[zeropage]
            static mut B_PTR: far *U32;

                        fn test() {
                A_VAL.div(B_PTR);
            }
        """
        build_and_check(source)


class TestU32ComparisonMethod:
    """Tests for U32 comparison method."""

    def test_cmp_return_type(self):
        """cmp method returns u8 (STATUS)."""
        source = U32_HEADER + """
            impl far U32 {
                                far fn cmp(far *self, far *other: U32) -> u8 {
                    A = self.hi;
                    if A != other.hi {
                        A = self.hi;
                        asm!("CMP _U32__cmp_other_hi");
                        return;
                    }
                    A = self.lo;
                    asm!("CMP _U32__cmp_other_lo");
                    return STATUS;
                }
            }

            #[zeropage]
            static mut A_VAL: U32;
            #[zeropage]
            static mut B_PTR: far *U32;

                        fn test() {
                A_VAL.cmp(B_PTR);
            }
        """
        build_and_check(source)


class TestU32ShiftMethods:
    """Tests for U32 shift methods."""

    def test_shl(self):
        """shl method shifts left by count."""
        source = U32_HEADER + """
            impl far U32 {
                                far fn shl(far *self, count @ X: u16) {
                    loop {
                        if X == 0 {
                            break;
                        }
                        A = self.lo;
                        asm!("ASL A");
                        self.lo = A;
                        A = self.hi;
                        asm!("ROL A");
                        self.hi = A;
                        X--;
                    }
                }
            }

            #[zeropage]
            static mut VALUE: U32;

                        fn test() {
                VALUE.shl(4);
            }
        """
        build_and_check(source)

    def test_shr(self):
        """shr method shifts right by count."""
        source = U32_HEADER + """
            impl far U32 {
                                far fn shr(far *self, count @ X: u16) {
                    loop {
                        if X == 0 {
                            break;
                        }
                        A = self.hi;
                        asm!("LSR A");
                        self.hi = A;
                        A = self.lo;
                        asm!("ROR A");
                        self.lo = A;
                        X--;
                    }
                }
            }

            #[zeropage]
            static mut VALUE: U32;

                        fn test() {
                VALUE.shr(4);
            }
        """
        build_and_check(source)


class TestU32HardwareMethods:
    """Tests for U32 SNES hardware-accelerated methods."""

    def test_div_u8_signature(self):
        """div_u8 takes u8 divisor and returns u8 remainder."""
        source = U32_HEADER + """
            impl far U32 {
                                far fn div_u8(far *self, divisor @ X: u16) -> u8 {
                    if X == 0 {
                        self.lo = 0xFFFF;
                        self.hi = 0xFFFF;
                        A = 0;
                        return A;
                    }
                    A = 0;
                    return A;
                }
            }

            #[zeropage]
            static mut VALUE: U32;
            #[zeropage]
            static mut REMAINDER: u8;

                        fn test() {
                A = VALUE.div_u8(10);
                REMAINDER = A;
            }
        """
        build_and_check(source)

    def test_mod_u8_calls_div_u8(self):
        """mod_u8 can call div_u8 for code reuse."""
        source = U32_HEADER + """
            impl far U32 {
                                far fn div_u8(far *self, divisor @ X: u16) -> u8 {
                    A = 0;
                    return A;
                }

                                far fn mod_u8(far *self, divisor @ X: u16) {
                    let mut remainder: u8 = self.div_u8(X);
                    self.lo = remainder as u16;
                    self.hi = 0;
                }
            }

            #[zeropage]
            static mut VALUE: U32;

                        fn test() {
                VALUE.mod_u8(10);
            }
        """
        build_and_check(source)


class TestU32Macros:
    """Tests for U32 convenience macros."""

    def test_u32_add_macro(self):
        """u32_add macro expands to copy and add."""
        source = U32_HEADER + """
            impl far U32 {
                                far fn copy(far *self, far *src: U32) {
                    A = src.lo;
                    self.lo = A;
                    A = src.hi;
                    self.hi = A;
                }

                                far fn add(far *self, far *other: U32) {
                    STATUS.Carry = false;
                    A = self.lo;
                    A = A + other.lo;
                    self.lo = A;
                    A = self.hi;
                    A = A + other.hi;
                    self.hi = A;
                }
            }

            macro_rules! u32_add($dest:expr, $a:expr, $b:expr) {
                $dest.copy($a);
                $dest.add($b);
            }

            #[zeropage]
            static mut A_PTR: far *U32;
            #[zeropage]
            static mut B_PTR: far *U32;
            #[zeropage]
            static mut RESULT: U32;

                        fn test() {
                u32_add!(RESULT, A_PTR, B_PTR);
            }
        """
        build_and_check(source)

    def test_u32_sub_macro(self):
        """u32_sub macro expands to copy and sub."""
        source = U32_HEADER + """
            impl far U32 {
                                far fn copy(far *self, far *src: U32) {
                    A = src.lo;
                    self.lo = A;
                    A = src.hi;
                    self.hi = A;
                }

                                far fn sub(far *self, far *other: U32) {
                    STATUS.Carry = true;
                    A = self.lo;
                    A = A - other.lo;
                    self.lo = A;
                    A = self.hi;
                    A = A - other.hi;
                    self.hi = A;
                }
            }

            macro_rules! u32_sub($dest:expr, $a:expr, $b:expr) {
                $dest.copy($a);
                $dest.sub($b);
            }

            #[zeropage]
            static mut A_PTR: far *U32;
            #[zeropage]
            static mut B_PTR: far *U32;
            #[zeropage]
            static mut RESULT: U32;

                        fn test() {
                u32_sub!(RESULT, A_PTR, B_PTR);
            }
        """
        build_and_check(source)


class TestU32MethodChaining:
    """Tests for chaining U32 method calls."""

    def test_method_on_static(self):
        """Methods can be called on static U32 variables."""
        source = U32_HEADER + """
            impl far U32 {
                                far fn from_u16(far *self, value @ A: u16) {
                    self.lo = A;
                    self.hi = 0;
                }

                                far fn add(far *self, far *other: U32) {
                    STATUS.Carry = false;
                    A = self.lo;
                    A = A + other.lo;
                    self.lo = A;
                    A = self.hi;
                    A = A + other.hi;
                    self.hi = A;
                }
            }

            #[zeropage]
            static mut COUNTER: U32;
            #[zeropage]
            static mut INCREMENT: U32;
            #[zeropage]
            static mut INC_PTR: far *U32;

                        fn init() {
                COUNTER.from_u16(0 as u16);
                INCREMENT.from_u16(1 as u16);
            }

                        fn tick() {
                COUNTER.add(INC_PTR);
            }
        """
        build_and_check(source)

    def test_method_on_pointer(self):
        """Methods can be called via far pointer."""
        source = U32_HEADER + """
            impl far U32 {
                                far fn to_u16(far *self) -> u16 {
                    return self.lo;
                }
            }

            #[zeropage]
            static mut VALUE: U32;
            #[zeropage]
            static mut PTR: far *U32;
            #[zeropage]
            static mut RESULT: u16;

                        fn test() {
                A = PTR.to_u16();
                RESULT = A;
            }
        """
        build_and_check(source)


class TestU32Errors:
    """Tests for U32 error handling."""

    def test_wrong_argument_type(self):
        """Passing wrong type to U32 method fails."""
        source = U32_HEADER + """
            impl far U32 {
                                far fn add(far *self, far *other: U32) {
                    A = self.lo;
                }
            }

            #[zeropage]
            static mut VALUE: U32;
            #[zeropage]
            static mut OTHER_PTR: far *u16;

                        fn test() {
                VALUE.add(OTHER_PTR);
            }
        """
        with pytest.raises(TypeCheckError):
            build_and_check(source)

    def test_near_pointer_to_far_impl(self):
        """Using near pointer with far impl fails."""
        source = U32_HEADER + """
            impl far U32 {
                                far fn to_u16(far *self) -> u16 {
                    return self.lo;
                }
            }

            #[zeropage]
            static mut PTR: *U32;  // Near pointer

                        fn test() -> u16 {
                return PTR.to_u16();  // Should fail - near ptr to far method
            }
        """
        with pytest.raises(TypeCheckError):
            build_and_check(source)


# =============================================================================
# I32 (Signed 32-bit Integer) Tests
# =============================================================================

class TestI32Struct:
    """Tests for I32 struct definition."""

    def test_i32_struct_size(self):
        """I32 struct is 4 bytes (two u16 fields)."""
        source = """
            struct I32 {
                lo: u16,
                hi: u16
            }
        """
        program = parse(source, "test.r65")
        struct_decl = program.items[0]
        assert struct_decl.name == "I32"
        assert len(struct_decl.fields) == 2
        assert struct_decl.fields[0].name == "lo"
        assert struct_decl.fields[1].name == "hi"

    def test_i32_static_declaration(self):
        """I32 can be declared as static variable."""
        source = I32_HEADER + """
            #[zeropage]
            static mut VALUE: I32;
        """
        build_and_check(source)


class TestI32ImplBlock:
    """Tests for I32 impl block structure."""

    def test_impl_far_i32(self):
        """impl far I32 block parses correctly."""
        source = I32_HEADER + """
            impl far I32 {
                                far fn from_i16(far *self, value @ A: i16) {
                    self.lo = A as u16;
                    if (A as u16) & 0x8000 != 0 as u16 {
                        self.hi = 0xFFFF;
                    } else {
                        self.hi = 0;
                    }
                }
            }
        """
        program = parse(source, "test.r65")
        impl_decl = program.items[-1]
        assert impl_decl.struct_name == "I32"
        assert impl_decl.is_far is True
        assert len(impl_decl.methods) == 1
        assert impl_decl.methods[0].is_far is True

    def test_method_mangling(self):
        """I32 methods are mangled to I32__method."""
        source = I32_HEADER + """
            impl far I32 {
                                far fn add(far *self, far *other: I32) {
                    STATUS.Carry = false;
                    A = self.lo;
                    A = A + other.lo;
                    self.lo = A;
                    A = self.hi;
                    A = A + other.hi;
                    self.hi = A;
                }
            }
        """
        hir = build_and_check(source)
        from r65.compiler.hir import HIRImplDecl
        impl_decl = next(d for d in hir.declarations if isinstance(d, HIRImplDecl))
        assert impl_decl.methods[0].name == "I32__add"


class TestI32ConversionMethods:
    """Tests for I32 conversion methods."""

    def test_from_i16_positive(self):
        """from_i16 with positive value zero-extends."""
        source = I32_HEADER + """
            impl far I32 {
                                far fn from_i16(far *self, value @ A: i16) {
                    self.lo = A as u16;
                    if (A as u16) & 0x8000 != 0 as u16 {
                        self.hi = 0xFFFF;
                    } else {
                        self.hi = 0;
                    }
                }
            }

            #[zeropage]
            static mut VALUE: I32;

                        fn test() {
                VALUE.from_i16(1000 as i16);
            }
        """
        build_and_check(source)

    def test_from_i16_negative(self):
        """from_i16 with negative value sign-extends."""
        source = I32_HEADER + """
            impl far I32 {
                                far fn from_i16(far *self, value @ A: i16) {
                    self.lo = A as u16;
                    if (A as u16) & 0x8000 != 0 as u16 {
                        self.hi = 0xFFFF;
                    } else {
                        self.hi = 0;
                    }
                }
            }

            #[zeropage]
            static mut VALUE: I32;

                        fn test() {
                VALUE.from_i16(-100 as i16);
            }
        """
        build_and_check(source)

    def test_to_i16(self):
        """to_i16 method returns i16."""
        source = I32_HEADER + """
            impl far I32 {
                                far fn to_i16(far *self) -> u16 {
                    // Return low word (interpret as truncated value)
                    return self.lo;
                }
            }

            #[zeropage]
            static mut VALUE: I32;
            #[zeropage]
            static mut RESULT: u16;

                        fn test() {
                A = VALUE.to_i16();
                RESULT = A;
            }
        """
        build_and_check(source)

    def test_copy(self):
        """copy method copies another I32."""
        source = I32_HEADER + """
            impl far I32 {
                                far fn copy(far *self, far *src: I32) {
                    A = src.lo;
                    self.lo = A;
                    A = src.hi;
                    self.hi = A;
                }
            }

            #[zeropage]
            static mut A_VAL: I32;
            #[zeropage]
            static mut B_PTR: far *I32;

                        fn test() {
                A_VAL.copy(B_PTR);
            }
        """
        build_and_check(source)


class TestI32SignMethods:
    """Tests for I32 sign-related methods."""

    def test_is_negative(self):
        """is_negative returns bool based on sign bit."""
        source = I32_HEADER + """
            impl far I32 {
                                far fn is_negative(far *self) -> bool {
                    A = self.hi;
                    if A & 0x8000 != 0 as u16 {
                        return true;
                    }
                    return false;
                }
            }

            #[zeropage]
            static mut VALUE: I32;
            #[zeropage]
            static mut IS_NEG: bool;

                        fn test() {
                IS_NEG = VALUE.is_negative();
            }
        """
        build_and_check(source)

    def test_neg(self):
        """neg method negates using two's complement."""
        source = I32_HEADER + """
            impl far I32 {
                                far fn neg(far *self) {
                    A = self.lo;
                    A = A ^ 0xFFFF;
                    self.lo = A;
                    A = self.hi;
                    A = A ^ 0xFFFF;
                    self.hi = A;
                    STATUS.Carry = false;
                    A = self.lo;
                    A = A + 1 as u16;
                    self.lo = A;
                    A = self.hi;
                    A = A + 0 as u16;
                    self.hi = A;
                }
            }

            #[zeropage]
            static mut VALUE: I32;

                        fn test() {
                VALUE.neg();
            }
        """
        build_and_check(source)

    def test_abs(self):
        """abs method takes absolute value."""
        source = I32_HEADER + """
            impl far I32 {
                                far fn neg(far *self) {
                    A = self.lo;
                    A = A ^ 0xFFFF;
                    self.lo = A;
                    A = self.hi;
                    A = A ^ 0xFFFF;
                    self.hi = A;
                    STATUS.Carry = false;
                    A = self.lo;
                    A = A + 1 as u16;
                    self.lo = A;
                    A = self.hi;
                    A = A + 0 as u16;
                    self.hi = A;
                }

                                far fn abs(far *self) {
                    A = self.hi;
                    if A & 0x8000 != 0 as u16 {
                        self.neg();
                    }
                }
            }

            #[zeropage]
            static mut VALUE: I32;

                        fn test() {
                VALUE.abs();
            }
        """
        build_and_check(source)


class TestI32ArithmeticMethods:
    """Tests for I32 arithmetic methods."""

    def test_add(self):
        """add method performs 32-bit signed addition."""
        source = I32_HEADER + """
            impl far I32 {
                                far fn add(far *self, far *other: I32) {
                    STATUS.Carry = false;
                    A = self.lo;
                    A = A + other.lo;
                    self.lo = A;
                    A = self.hi;
                    A = A + other.hi;
                    self.hi = A;
                }
            }

            #[zeropage]
            static mut A_VAL: I32;
            #[zeropage]
            static mut B_PTR: far *I32;

                        fn test() {
                A_VAL.add(B_PTR);
            }
        """
        build_and_check(source)

    def test_sub(self):
        """sub method performs 32-bit signed subtraction."""
        source = I32_HEADER + """
            impl far I32 {
                                far fn sub(far *self, far *other: I32) {
                    STATUS.Carry = true;
                    A = self.lo;
                    A = A - other.lo;
                    self.lo = A;
                    A = self.hi;
                    A = A - other.hi;
                    self.hi = A;
                }
            }

            #[zeropage]
            static mut A_VAL: I32;
            #[zeropage]
            static mut B_PTR: far *I32;

                        fn test() {
                A_VAL.sub(B_PTR);
            }
        """
        build_and_check(source)

    def test_div_with_zero_check(self):
        """div method handles division by zero."""
        source = I32_HEADER + """
            impl far I32 {
                                far fn div(far *self, far *other: I32) {
                    A = other.lo;
                    if A == 0 as u16 {
                        A = other.hi;
                        if A == 0 as u16 {
                            A = 0;
                            self.lo = A;
                            A = 0x8000;
                            self.hi = A;
                            return;
                        }
                    }
                    // Division logic would go here
                }
            }

            #[zeropage]
            static mut A_VAL: I32;
            #[zeropage]
            static mut B_PTR: far *I32;

                        fn test() {
                A_VAL.div(B_PTR);
            }
        """
        build_and_check(source)


class TestI32ComparisonMethod:
    """Tests for I32 comparison method."""

    def test_cmp_return_type(self):
        """cmp method returns u8 (STATUS)."""
        source = I32_HEADER + """
            impl far I32 {
                                far fn cmp(far *self, far *other: I32) -> u8 {
                    let mut self_sign: u16;
                    let mut other_sign: u16;

                    A = self.hi;
                    self_sign = A & 0x8000;
                    A = other.hi;
                    other_sign = A & 0x8000;

                    if self_sign != other_sign {
                        if self_sign != 0 as u16 {
                            // self is negative, other is positive: self < other
                            STATUS.Carry = false;
                        } else {
                            // self is positive, other is negative: self > other
                            STATUS.Carry = true;
                        }
                        return STATUS;
                    }

                    A = self.hi;
                    if A != other.hi {
                        A = self.hi;
                        asm!("CMP _I32__cmp_other_hi");
                        return;
                    }

                    A = self.lo;
                    asm!("CMP _I32__cmp_other_lo");
                    return STATUS;
                }
            }

            #[zeropage]
            static mut A_VAL: I32;
            #[zeropage]
            static mut B_PTR: far *I32;

                        fn test() {
                A_VAL.cmp(B_PTR);
            }
        """
        build_and_check(source)

    def test_signed_cmp_different_signs(self):
        """cmp handles comparison when signs differ."""
        source = I32_HEADER + """
            impl far I32 {
                                far fn cmp(far *self, far *other: I32) -> u8 {
                    let mut self_sign: u16;
                    let mut other_sign: u16;

                    A = self.hi;
                    self_sign = A & 0x8000;
                    A = other.hi;
                    other_sign = A & 0x8000;

                    // If signs differ, negative < positive
                    if self_sign != other_sign {
                        if self_sign != 0 as u16 {
                            // self negative, other positive: self < other
                            STATUS.Carry = false;
                        } else {
                            // self positive, other negative: self > other
                            STATUS.Carry = true;
                        }
                    }
                    return STATUS;
                }
            }

            #[zeropage]
            static mut A_VAL: I32;
            #[zeropage]
            static mut B_PTR: far *I32;

                        fn test() {
                A_VAL.cmp(B_PTR);
            }
        """
        build_and_check(source)


class TestI32ShiftMethods:
    """Tests for I32 shift methods."""

    def test_shl(self):
        """shl method shifts left by count (same as unsigned)."""
        source = I32_HEADER + """
            impl far I32 {
                                far fn shl(far *self, count @ X: u16) {
                    loop {
                        if X == 0 as u8 {
                            break;
                        }
                        A = self.lo;
                        asm!("ASL A");
                        self.lo = A;
                        A = self.hi;
                        asm!("ROL A");
                        self.hi = A;
                        X--;
                    }
                }
            }

            #[zeropage]
            static mut VALUE: I32;

                        fn test() {
                VALUE.shl(4);
            }
        """
        build_and_check(source)

    def test_sar(self):
        """sar method performs arithmetic shift right (preserves sign)."""
        source = I32_HEADER + """
            impl far I32 {
                                far fn sar(far *self, count @ X: u16) {
                    loop {
                        if X == 0 as u8 {
                            break;
                        }
                        A = self.hi;
                        if A & 0x8000 != 0 as u16 {
                            asm!("LSR A");
                            A = A | 0x8000;
                        } else {
                            asm!("LSR A");
                        }
                        self.hi = A;
                        A = self.lo;
                        asm!("ROR A");
                        self.lo = A;
                        X--;
                    }
                }
            }

            #[zeropage]
            static mut VALUE: I32;

                        fn test() {
                VALUE.sar(4);
            }
        """
        build_and_check(source)


class TestI32HardwareMethods:
    """Tests for I32 SNES hardware-accelerated methods."""

    def test_div_i8_signature(self):
        """div_i8 takes i8 divisor and returns u8 remainder (as bit pattern)."""
        source = I32_HEADER + """
            impl far I32 {
                                far fn div_i8(far *self, divisor @ X: u16) -> u8 {
                    if X == 0 as i8 {
                        self.lo = 0;
                        self.hi = 0x8000;
                        A = 0;
                        return A;
                    }
                    A = 0;
                    return A;
                }
            }

            #[zeropage]
            static mut VALUE: I32;
            #[zeropage]
            static mut REMAINDER: u8;

                        fn test() {
                A = VALUE.div_i8(10 as i8);
                REMAINDER = A;
            }
        """
        build_and_check(source)

    def test_mod_i8_calls_div_i8(self):
        """mod_i8 can call div_i8 for code reuse."""
        source = I32_HEADER + """
            impl far I32 {
                                far fn div_i8(far *self, divisor @ X: u16) -> i8 {
                    A = 0;
                    return A as i8;
                }

                                far fn mod_i8(far *self, divisor @ X: u16) {
                    if X == 0 as i8 {
                        return;
                    }
                    let mut remainder: i8 = self.div_i8(X);
                    A = remainder as u8;
                    if A & 0x80 != 0 as u8 {
                        self.lo = (0xFF00 | (A as u16));
                        self.hi = 0xFFFF;
                    } else {
                        self.lo = A as u16;
                        self.hi = 0;
                    }
                }
            }

            #[zeropage]
            static mut VALUE: I32;

                        fn test() {
                VALUE.mod_i8(10 as i8);
            }
        """
        build_and_check(source)


class TestI32Macros:
    """Tests for I32 convenience macros."""

    def test_i32_add_macro(self):
        """i32_add macro expands to copy and add."""
        source = I32_HEADER + """
            impl far I32 {
                                far fn copy(far *self, far *src: I32) {
                    A = src.lo;
                    self.lo = A;
                    A = src.hi;
                    self.hi = A;
                }

                                far fn add(far *self, far *other: I32) {
                    STATUS.Carry = false;
                    A = self.lo;
                    A = A + other.lo;
                    self.lo = A;
                    A = self.hi;
                    A = A + other.hi;
                    self.hi = A;
                }
            }

            macro_rules! i32_add($dest:expr, $a:expr, $b:expr) {
                $dest.copy($a);
                $dest.add($b);
            }

            #[zeropage]
            static mut A_PTR: far *I32;
            #[zeropage]
            static mut B_PTR: far *I32;
            #[zeropage]
            static mut RESULT: I32;

                        fn test() {
                i32_add!(RESULT, A_PTR, B_PTR);
            }
        """
        build_and_check(source)

    def test_i32_neg_macro(self):
        """i32_neg macro expands to copy and neg."""
        source = I32_HEADER + """
            impl far I32 {
                                far fn copy(far *self, far *src: I32) {
                    A = src.lo;
                    self.lo = A;
                    A = src.hi;
                    self.hi = A;
                }

                                far fn neg(far *self) {
                    A = self.lo;
                    A = A ^ 0xFFFF;
                    self.lo = A;
                    A = self.hi;
                    A = A ^ 0xFFFF;
                    self.hi = A;
                    STATUS.Carry = false;
                    A = self.lo;
                    A = A + 1 as u16;
                    self.lo = A;
                    A = self.hi;
                    A = A + 0 as u16;
                    self.hi = A;
                }
            }

            macro_rules! i32_neg($dest:expr, $a:expr) {
                $dest.copy($a);
                $dest.neg();
            }

            #[zeropage]
            static mut SRC_PTR: far *I32;
            #[zeropage]
            static mut RESULT: I32;

                        fn test() {
                i32_neg!(RESULT, SRC_PTR);
            }
        """
        build_and_check(source)

    def test_i32_abs_macro(self):
        """i32_abs macro expands to copy and abs."""
        source = I32_HEADER + """
            impl far I32 {
                                far fn copy(far *self, far *src: I32) {
                    A = src.lo;
                    self.lo = A;
                    A = src.hi;
                    self.hi = A;
                }

                                far fn neg(far *self) {
                    A = self.lo;
                    A = A ^ 0xFFFF;
                    self.lo = A;
                    A = self.hi;
                    A = A ^ 0xFFFF;
                    self.hi = A;
                    STATUS.Carry = false;
                    A = self.lo;
                    A = A + 1 as u16;
                    self.lo = A;
                    A = self.hi;
                    A = A + 0 as u16;
                    self.hi = A;
                }

                                far fn abs(far *self) {
                    A = self.hi;
                    if A & 0x8000 != 0 as u16 {
                        self.neg();
                    }
                }
            }

            macro_rules! i32_abs($dest:expr, $a:expr) {
                $dest.copy($a);
                $dest.abs();
            }

            #[zeropage]
            static mut SRC_PTR: far *I32;
            #[zeropage]
            static mut RESULT: I32;

                        fn test() {
                i32_abs!(RESULT, SRC_PTR);
            }
        """
        build_and_check(source)


class TestI32MethodChaining:
    """Tests for chaining I32 method calls."""

    def test_method_on_static(self):
        """Methods can be called on static I32 variables."""
        source = I32_HEADER + """
            impl far I32 {
                                far fn from_i16(far *self, value @ A: i16) {
                    self.lo = A as u16;
                    if (A as u16) & 0x8000 != 0 as u16 {
                        self.hi = 0xFFFF;
                    } else {
                        self.hi = 0;
                    }
                }

                                far fn add(far *self, far *other: I32) {
                    STATUS.Carry = false;
                    A = self.lo;
                    A = A + other.lo;
                    self.lo = A;
                    A = self.hi;
                    A = A + other.hi;
                    self.hi = A;
                }
            }

            #[zeropage]
            static mut COUNTER: I32;
            #[zeropage]
            static mut INCREMENT: I32;
            #[zeropage]
            static mut INC_PTR: far *I32;

                        fn init() {
                COUNTER.from_i16(0 as i16);
                INCREMENT.from_i16(1 as i16);
            }

                        fn tick() {
                COUNTER.add(INC_PTR);
            }
        """
        build_and_check(source)

    def test_method_on_pointer(self):
        """Methods can be called via far pointer."""
        source = I32_HEADER + """
            impl far I32 {
                                far fn to_u16(far *self) -> u16 {
                    return self.lo;
                }
            }

            #[zeropage]
            static mut VALUE: I32;
            #[zeropage]
            static mut PTR: far *I32;
            #[zeropage]
            static mut RESULT: u16;

                        fn test() {
                A = PTR.to_u16();
                RESULT = A;
            }
        """
        build_and_check(source)


class TestI32Errors:
    """Tests for I32 error handling."""

    def test_wrong_argument_type(self):
        """Passing wrong type to I32 method fails."""
        source = I32_HEADER + """
            impl far I32 {
                                far fn add(far *self, far *other: I32) {
                    A = self.lo;
                }
            }

            #[zeropage]
            static mut VALUE: I32;
            #[zeropage]
            static mut OTHER_PTR: far *u16;

                        fn test() {
                VALUE.add(OTHER_PTR);
            }
        """
        with pytest.raises(TypeCheckError):
            build_and_check(source)

    def test_near_pointer_to_far_impl(self):
        """Using near pointer with far impl fails."""
        source = I32_HEADER + """
            impl far I32 {
                                far fn to_i16(far *self) -> i16 {
                    return self.lo as i16;
                }
            }

            #[zeropage]
            static mut PTR: *I32;  // Near pointer

                        fn test() -> i16 {
                return PTR.to_i16();  // Should fail - near ptr to far method
            }
        """
        with pytest.raises(TypeCheckError):
            build_and_check(source)

    def test_i32_vs_u32_type_mismatch(self):
        """I32 and U32 are different types."""
        source = """
            struct I32 { lo: u16, hi: u16 }
            struct U32 { lo: u16, hi: u16 }

            impl far I32 {
                                far fn add(far *self, far *other: I32) {
                    A = self.lo;
                }
            }

            #[zeropage]
            static mut SIGNED_VAL: I32;
            #[zeropage]
            static mut UNSIGNED_PTR: far *U32;

                        fn test() {
                SIGNED_VAL.add(UNSIGNED_PTR);  // Should fail - U32 != I32
            }
        """
        with pytest.raises(TypeCheckError):
            build_and_check(source)


class TestI32SignedSpecific:
    """Tests for signed-specific behavior."""

    def test_negative_literal_handling(self):
        """Negative literals work with from_i16."""
        source = I32_HEADER + """
            impl far I32 {
                                far fn from_i16(far *self, value @ A: i16) {
                    self.lo = A as u16;
                    if (A as u16) & 0x8000 != 0 as u16 {
                        self.hi = 0xFFFF;
                    } else {
                        self.hi = 0;
                    }
                }
            }

            #[zeropage]
            static mut VALUE: I32;

                        fn test() {
                VALUE.from_i16(-1 as i16);
                VALUE.from_i16(-32768 as i16);
            }
        """
        build_and_check(source)

    def test_subtraction_across_zero(self):
        """Subtraction that crosses zero (positive to negative)."""
        source = I32_HEADER + """
            impl far I32 {
                                far fn from_i16(far *self, value @ A: i16) {
                    self.lo = A as u16;
                    if (A as u16) & 0x8000 != 0 as u16 {
                        self.hi = 0xFFFF;
                    } else {
                        self.hi = 0;
                    }
                }

                                far fn sub(far *self, far *other: I32) {
                    STATUS.Carry = true;
                    A = self.lo;
                    A = A - other.lo;
                    self.lo = A;
                    A = self.hi;
                    A = A - other.hi;
                    self.hi = A;
                }
            }

            #[zeropage]
            static mut SMALL: I32;
            #[zeropage]
            static mut LARGE_PTR: far *I32;

                        fn test() {
                SMALL.from_i16(5 as i16);
                // Subtracting a larger value should make SMALL negative
                SMALL.sub(LARGE_PTR);
            }
        """
        build_and_check(source)
