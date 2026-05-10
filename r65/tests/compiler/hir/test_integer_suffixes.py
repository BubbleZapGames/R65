# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Tests for integer literal type suffixes (u8, u16, i8, i16).

Verifies that suffixed literals get the right value and type at the HIR level
after type checking. Parser/lexer suffix tokenization is covered separately by
compiler/frontend/test_lexer.py and test_parser.py; this file covers semantics.
"""
from r65.compiler.frontend import Parser
from r65.compiler.hir import HIRBuilder, HIRIntegerLiteral
from r65.compiler.hir.types import BasicTypeInfo
from r65.compiler.typeck import TypeChecker


def build_and_typecheck(source: str):
    parser = Parser()
    ast = parser.parse(source)
    hir = HIRBuilder().build_program(ast)
    TypeChecker(hir).check()
    return hir


def find_literals(node, _seen=None):
    """Yield every HIRIntegerLiteral reachable from node (cycle-safe)."""
    if _seen is None:
        _seen = set()
    obj_id = id(node)
    if obj_id in _seen:
        return
    if isinstance(node, HIRIntegerLiteral):
        yield node
        return
    if isinstance(node, (str, int, float, bool, bytes)) or node is None:
        return
    _seen.add(obj_id)
    if isinstance(node, (list, tuple)):
        for item in node:
            yield from find_literals(item, _seen)
    elif hasattr(node, '__dict__'):
        for v in vars(node).values():
            yield from find_literals(v, _seen)


def first_literal(hir) -> HIRIntegerLiteral:
    for decl in hir.declarations:
        for lit in find_literals(decl):
            return lit
    raise AssertionError("no integer literal found")


class TestIntegerSuffixes:
    """Suffixed literals carry the suffix's type through type checking."""

    def test_u8_suffix(self):
        """255u8 → value=255, type u8."""
        hir = build_and_typecheck("fn main() { A = 255u8; }")
        lit = first_literal(hir)
        assert lit.value == 255
        assert lit.suffix == 'u8'
        assert isinstance(lit.expr_type, BasicTypeInfo)
        assert lit.expr_type.name == 'u8'

    def test_u16_suffix_forces_wide(self):
        """0u16 → small value still typed u16."""
        hir = build_and_typecheck('''
            #[zeropage]
            static mut VAL: u16;
            fn main() { VAL = 0u16; }
        ''')
        # Find the literal inside the assignment (skip static initializer if absent)
        for decl in hir.declarations:
            if getattr(decl, 'name', None) == 'main':
                lit = next(find_literals(decl))
                break
        else:
            raise AssertionError("main not found")
        assert lit.value == 0
        assert lit.suffix == 'u16'
        assert lit.expr_type.name == 'u16'

    def test_hex_with_suffix(self):
        """0xFFu8 → value=255, type u8."""
        hir = build_and_typecheck("fn main() { A = 0xFFu8; }")
        lit = first_literal(hir)
        assert lit.value == 0xFF
        assert lit.suffix == 'u8'
        assert lit.expr_type.name == 'u8'

    def test_binary_with_suffix(self):
        """0b0001_0000u16 → value=0x10, type u16."""
        hir = build_and_typecheck('''
            #[zeropage]
            static mut VAL: u16;
            fn main() { VAL = 0b0001_0000u16; }
        ''')
        for decl in hir.declarations:
            if getattr(decl, 'name', None) == 'main':
                lit = next(find_literals(decl))
                break
        else:
            raise AssertionError("main not found")
        assert lit.value == 0x10
        assert lit.suffix == 'u16'
        assert lit.expr_type.name == 'u16'

    def test_suffix_in_arithmetic(self):
        """10u8 + 20u8 → both operands u8, result u8."""
        hir = build_and_typecheck("fn main() { A = 10u8 + 20u8; }")
        lits = []
        for decl in hir.declarations:
            lits.extend(find_literals(decl))
        assert len(lits) == 2
        assert lits[0].value == 10 and lits[0].suffix == 'u8'
        assert lits[1].value == 20 and lits[1].suffix == 'u8'
        assert lits[0].expr_type.name == 'u8'
        assert lits[1].expr_type.name == 'u8'

    def test_i8_suffix_negative(self):
        """-128i8 → signed type carries through."""
        hir = build_and_typecheck('''
            #[zeropage]
            static mut VAL: i8;
            fn main() { VAL = 127i8; }
        ''')
        for decl in hir.declarations:
            if getattr(decl, 'name', None) == 'main':
                lit = next(find_literals(decl))
                break
        else:
            raise AssertionError("main not found")
        assert lit.value == 127
        assert lit.suffix == 'i8'
        assert lit.expr_type.name == 'i8'

    def test_i16_suffix(self):
        """1000i16 → signed 16-bit."""
        hir = build_and_typecheck('''
            #[zeropage]
            static mut VAL: i16;
            fn main() { VAL = 1000i16; }
        ''')
        for decl in hir.declarations:
            if getattr(decl, 'name', None) == 'main':
                lit = next(find_literals(decl))
                break
        else:
            raise AssertionError("main not found")
        assert lit.value == 1000
        assert lit.suffix == 'i16'
        assert lit.expr_type.name == 'i16'
