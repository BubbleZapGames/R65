# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""End-to-end tests for tagged enums.

Runs the ROM. These cover what the assembly-level tests cannot: that the two
planes really are laid out and addressed as intended at runtime, that a payload
binding reads and writes the pool rather than a copy, and that an index `self`
survives a real call.

See docs/enums.md.
"""

from r65.tests.e2e import ExpectedState


SHAPES = '''
    struct Circle { r: u8 }
    struct Rect { w: u8, h: u8 }
    enum Shape { Empty, C(Circle), R(Rect) }

    #[zeropage(0x10)]
    static mut POOL: [Shape; 4];
'''


class TestLayout:
    """Storage splits into a tag plane then a payload plane."""

    def test_tag_and_payload_planes_are_separate(self, e2e):
        """POOL is 4 elements: tags at 0x10..0x13, payloads (2 bytes each) from
        0x14. An array-of-structs layout would interleave them instead.
        """
        result = e2e.run(SHAPES + '''
            #[entry]
            fn main() {
                POOL[0] = Shape::C(Circle { r: 0xB0 });
                POOL[1] = Shape::R(Rect { w: 0xC0, h: 0xC1 });
                POOL[2] = Shape::Empty;
            }
        ''', ExpectedState(memory={
            0x7E0010: [0x01, 0x02, 0x00],        # tags: C, R, Empty
            0x7E0014: [0xB0],                    # POOL__data[0].C.r
            0x7E0016: [0xC0, 0xC1],              # POOL__data[1].R.w, .h
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_variable_index(self, e2e):
        result = e2e.run(SHAPES + '''
            #[zeropage(0x30)]
            static mut i: u16;

            #[entry]
            fn main() {
                i = 3;
                POOL[i] = Shape::C(Circle { r: 0x5A });
            }
        ''', ExpectedState(memory={
            0x7E0013: [0x01],    # tag of element 3
            0x7E001A: [0x5A],    # payload of element 3
        }))
        assert result.success, f"Failures: {result.failures}"


class TestMatch:
    """Dispatch on the tag, and read the payload through a binding."""

    def test_each_variant_takes_its_own_arm(self, e2e):
        result = e2e.run(SHAPES + '''
            #[zeropage(0x40)]
            static mut OUT: [u8; 3];

            fn area(i: u16) -> u8 {
                return match POOL[i] {
                    Shape::C(c) => c.r,
                    Shape::R(r) => r.w,
                    Shape::Empty => 0xFF,
                };
            }

            #[entry]
            fn main() {
                POOL[0] = Shape::C(Circle { r: 0x11 });
                POOL[1] = Shape::R(Rect { w: 0x22, h: 0x33 });
                POOL[2] = Shape::Empty;
                OUT[0] = area(0);
                OUT[1] = area(1);
                OUT[2] = area(2);
            }
        ''', ExpectedState(memory={0x7E0040: [0x11, 0x22, 0xFF]}))
        assert result.success, f"Failures: {result.failures}"

    def test_binding_writes_through_to_the_pool(self, e2e):
        """The binding is a view: assigning to it must change the pool itself,
        not a copy that is then discarded."""
        result = e2e.run(SHAPES + '''
            fn bump(i: u16) {
                match POOL[i] {
                    Shape::C(c) => { c.r = 0x7E; }
                    _ => {}
                }
            }

            #[entry]
            fn main() {
                POOL[1] = Shape::C(Circle { r: 0x01 });
                bump(1);
            }
        ''', ExpectedState(memory={0x7E0016: [0x7E]}))
        assert result.success, f"Failures: {result.failures}"

    def test_second_payload_field(self, e2e):
        result = e2e.run(SHAPES + '''
            #[zeropage(0x40)]
            static mut OUT: u8;

            #[entry]
            fn main() {
                POOL[0] = Shape::R(Rect { w: 0x40, h: 0x9C });
                OUT = match POOL[0] {
                    Shape::R(r) => r.h,
                    _ => 0,
                };
            }
        ''', ExpectedState(memory={0x7E0040: [0x9C]}))
        assert result.success, f"Failures: {result.failures}"


class TestMethods:
    """`self` is the element index, passed in X."""

    def test_index_self_method(self, e2e):
        result = e2e.run(SHAPES + '''
            impl Shape {
                fn area(self) -> u8 {
                    return match self {
                        Shape::C(c) => c.r,
                        Shape::R(r) => r.w,
                        Shape::Empty => 0xFF,
                    };
                }
            }

            #[zeropage(0x40)]
            static mut OUT: [u8; 2];

            #[entry]
            fn main() {
                POOL[0] = Shape::C(Circle { r: 0x44 });
                POOL[3] = Shape::Empty;
                OUT[0] = POOL[0].area();
                OUT[1] = POOL[3].area();
            }
        ''', ExpectedState(memory={0x7E0040: [0x44, 0xFF]}))
        assert result.success, f"Failures: {result.failures}"

    def test_method_mutates_the_pool(self, e2e):
        result = e2e.run(SHAPES + '''
            impl Shape {
                fn grow(self) {
                    match self {
                        Shape::C(c) => { c.r = c.r + 1; }
                        _ => {}
                    }
                }
            }

            #[entry]
            fn main() {
                POOL[2] = Shape::C(Circle { r: 0x10 });
                POOL[2].grow();
                POOL[2].grow();
            }
        ''', ExpectedState(memory={0x7E0018: [0x12]}))
        assert result.success, f"Failures: {result.failures}"


class TestSingleInstance:
    """A lone static is a pool of one."""

    def test_single_instance_round_trip(self, e2e):
        result = e2e.run('''
            struct Circle { r: u8 }
            enum Shape { Empty, C(Circle) }

            #[zeropage(0x10)]
            static mut SHAPE: Shape;

            #[zeropage(0x40)]
            static mut OUT: u8;

            #[entry]
            fn main() {
                SHAPE = Shape::C(Circle { r: 0x6B });
                OUT = match SHAPE {
                    Shape::C(c) => c.r,
                    Shape::Empty => 0,
                };
            }
        ''', ExpectedState(memory={
            0x7E0010: [0x01],    # tag plane, one element
            0x7E0011: [0x6B],    # payload plane
            0x7E0040: [0x6B],
        }))
        assert result.success, f"Failures: {result.failures}"
