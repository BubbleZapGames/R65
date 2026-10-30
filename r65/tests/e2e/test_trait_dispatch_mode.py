# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""Code after a `*dyn` call is assembled for the method's exit width.

The dispatch wrapper accepts its caller in either width and enters every
method in m8; the method returns in its exit mode. The caller kept tracking
the width it reached the wrapper in, so an m16 caller went on assembling
3-byte immediates the CPU, back in m8, read as two - and ran off into the
operand bytes.
"""

from r65.tests.e2e import E2ETest, ExpectedState


def test_code_after_a_dispatch_runs_in_the_methods_exit_width():
    """Known to be in the exit width, codegen drops the REP/SEP it used to
    re-pin after a dispatch, so the peephole's directive rewrite must know
    the wrapper's exit mode too, or `ADC #$01` after the m16 `read` is
    assembled two bytes wide."""
    source = '''
        struct Node { val: u16 }
        trait Part {
            fn store(*self, v: u16);
            fn read(*self) -> u16;
            fn low(*self) -> u8;
        }
        impl Part for Node {
            fn store(*self, v: u16) { self.val = v; }
            fn read(*self) -> u16 { return self.val; }
            fn low(*self) -> u8 { return self.val as u8; }
        }

        #[lowram(0x0300)] static mut N: [Node; 2] = [Node { val: 0 }, Node { val: 0 }];
        #[lowram(0x0310)] static mut OUT: [u16; 3];
        #[lowram(0x0316)] static mut BYTES: [u8; 2];

        #[entry]
        fn main() {
            let a: *dyn Part = &N[0] as *dyn Part;
            let b: *dyn Part = &N[1] as *dyn Part;
            a.store(0x1234);          // reached in m16 (pointer setup), returns m8
            OUT[0] = N[0].val;        // m16 code after an m8 return
            b.store(0x5678);
            OUT[1] = a.read() + 1;    // returns m16
            BYTES[0] = 0x42;              // m8 code after an m16 return
            BYTES[1] = b.low();
            OUT[2] = 0xBEEF;
        }
    '''
    result = E2ETest().run(source, ExpectedState(memory={
        0x7E0310: [0x34, 0x12, 0x35, 0x12, 0xEF, 0xBE],
        0x7E0316: [0x42, 0x78],
    }))
    assert result.success, f"{result.error} {result.failures}"


def test_dispatch_reached_in_m16_resumes_in_m8():
    """With self in a scratch register, `LDY` loads it in m16 and the JSL
    goes out in m16; the void method comes back in m8."""
    source = '''
        #[zeropage(0x10, register)] static mut S0: u16;
        #[zeropage(0x12, register)] static mut S1: u16;
        #[zeropage(0x14, register)] static mut S2: u16;
        #[zeropage(0x16, register)] static mut S3: u16;

        struct Node { tag: u8, back: *Node, val: u16 }
        trait Linker { fn point_at_me(*self, other: *Node, v: u16); }
        impl Linker for Node {
            fn point_at_me(*self, other: *Node, v: u16) {
                other.back = self as *Node;
                other.val = v;
            }
        }

        #[lowram(0x0300)] static mut N: [Node; 3] = [
            Node { tag: 0, back: 0 as *Node, val: 0 },
            Node { tag: 0, back: 0 as *Node, val: 0 },
            Node { tag: 0, back: 0 as *Node, val: 0 },
        ];
        #[lowram(0x0320)] static mut OUT: [u16; 4];

        #[entry]
        fn main() {
            let a: *dyn Linker = &N[0] as *dyn Linker;
            let b: *dyn Linker = &N[1] as *dyn Linker;
            a.point_at_me(&N[2] as *Node, 0x1111);
            OUT[0] = N[2].back as u16;
            b.point_at_me(&N[0] as *Node, 0x2222);
            OUT[1] = N[0].back as u16;
            OUT[2] = N[2].val;
            OUT[3] = N[0].val;
        }
    '''
    result = E2ETest().run(source, ExpectedState(memory={
        0x7E0320: [0x00, 0x03, 0x06, 0x03, 0x11, 0x11, 0x22, 0x22],
    }))
    assert result.success, f"{result.error} {result.failures}"
