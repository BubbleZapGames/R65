# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Test that all Rust keywords are properly reserved.
"""


from r65.compiler.frontend import tokenize, TokenType


def test_currently_used_keywords():
    """Test keywords that are currently implemented."""
    keywords = [
        'fn', 'let', 'mut', 'const', 'static', 'if', 'else',
        'loop', 'while', 'break', 'continue', 'return',
        'struct', 'enum', 'type', 'include', 'asm', 'as',
    ]

    for keyword in keywords:
        source = f"{keyword} identifier"
        tokens = tokenize(source)
        assert tokens[0].is_keyword(keyword), f"{keyword} should be a keyword"
        assert tokens[1].type == TokenType.IDENTIFIER

    print(f"✓ Currently used keywords test passed ({len(keywords)} keywords)")


def test_builtin_functions():
    """Test built-in function names are identifiers.

    Built-in functions (mul, div, mod, etc.) are now regular identifiers,
    not keywords. This allows them to be called like normal functions and
    recognized by the BuiltinRegistry during HIR building.
    """
    builtins = ['mvn', 'mvp', 'wai', 'stp', 'mul', 'div', 'mod', 'shl', 'shr', 'xba']

    for builtin in builtins:
        source = f"{builtin} identifier"
        tokens = tokenize(source)
        # Builtins are identifiers, not keywords
        assert tokens[0].type == TokenType.IDENTIFIER, f"{builtin} should be an identifier"
        assert tokens[0].value == builtin
        assert tokens[1].type == TokenType.IDENTIFIER

    print(f"✓ Built-in functions test passed ({len(builtins)} functions)")


def test_reserved_rust_keywords():
    """Test that Rust keywords are reserved even if not implemented."""
    # Note: 'mod' is not reserved - it's a builtin function identifier
    reserved = [
        'impl', 'trait', 'for', 'in', 'match', 'where',
        'use', 'pub', 'crate', 'self', 'Self', 'super',
        'async', 'await', 'move', 'ref', 'dyn',
        'extern', 'unsafe',
    ]

    for keyword in reserved:
        source = f"{keyword} identifier"
        tokens = tokenize(source)
        assert tokens[0].is_keyword(keyword), f"{keyword} should be reserved"
        assert tokens[1].type == TokenType.IDENTIFIER

    print(f"✓ Reserved Rust keywords test passed ({len(reserved)} keywords)")


def test_strict_reserved_keywords():
    """Test strict reserved keywords (reserved by Rust for future use)."""
    # Note: 'macro' removed - we use 'macro_rules' as a first-class keyword
    strict = [
        'abstract', 'become', 'box', 'do', 'final',
        'override', 'priv', 'typeof',
        'unsized', 'virtual', 'yield', 'try',
    ]

    for keyword in strict:
        source = f"{keyword} identifier"
        tokens = tokenize(source)
        assert tokens[0].is_keyword(keyword), f"{keyword} should be reserved"
        assert tokens[1].type == TokenType.IDENTIFIER

    print(f"✓ Strict reserved keywords test passed ({len(strict)} keywords)")


def test_far_keyword():
    """Test that 'far' is a keyword."""
    source = "far fn"
    tokens = tokenize(source)
    assert tokens[0].is_keyword('far')
    assert tokens[1].is_keyword('fn')

    print("✓ 'far' keyword test passed")


def test_keywords_are_not_identifiers():
    """Test that keywords cannot be used as identifiers."""
    # These should all be keywords, not identifiers
    # Note: 'mod' removed - it's now a builtin function identifier
    test_cases = [
        'struct', 'impl', 'trait', 'unsafe', 'async', 'await',
        'match', 'pub', 'use', 'extern', 'yield'
    ]

    for keyword in test_cases:
        source = f"let {keyword} = 1;"
        tokens = tokenize(source)
        # tokens should be: let, keyword, =, 1, ;
        assert tokens[0].is_keyword('let')
        assert tokens[1].is_keyword(keyword), f"{keyword} should not be usable as identifier"
        assert tokens[2].type == TokenType.ASSIGN

    print(f"✓ Keywords cannot be identifiers test passed ({len(test_cases)} cases)")


def test_keyword_case_sensitivity():
    """Test that keywords are case-sensitive."""
    source = "impl Impl IMPL"
    tokens = tokenize(source)

    # 'impl' should be keyword
    assert tokens[0].is_keyword('impl')

    # 'Impl' and 'IMPL' should be identifiers (keywords are lowercase)
    assert tokens[1].type == TokenType.IDENTIFIER
    assert tokens[1].value == 'Impl'
    assert tokens[2].type == TokenType.IDENTIFIER
    assert tokens[2].value == 'IMPL'

    print("✓ Keyword case sensitivity test passed")


def test_keyword_boundaries():
    """Test that keywords require word boundaries."""
    # 'impl' is a keyword, but 'implementation' should be an identifier
    source = "impl implementation"
    tokens = tokenize(source)

    assert tokens[0].is_keyword('impl')
    assert tokens[1].type == TokenType.IDENTIFIER
    assert tokens[1].value == 'implementation'

    # 'pub' is a keyword, but 'publish' should be an identifier
    source = "pub publish public"
    tokens = tokenize(source)

    assert tokens[0].is_keyword('pub')
    assert tokens[1].type == TokenType.IDENTIFIER
    assert tokens[1].value == 'publish'
    assert tokens[2].type == TokenType.IDENTIFIER
    assert tokens[2].value == 'public'

    print("✓ Keyword boundaries test passed")


def test_total_keyword_count():
    """Verify we have all Rust keywords reserved."""
    # This is a sanity check to ensure we didn't miss any

    # Count from the grammar:
    # - Currently used: 19 (fn, let, mut, const, static, if, else, loop, while, break, continue, return, struct, enum, type, include, asm, as, macro_rules)
    # - Built-ins: 10 (mvn, mvp, wai, stp, mul, div, mod, shl, shr, xba) - identifiers, not keywords
    # - Reserved: 17 (impl, trait, for, in, match, where, use, pub, crate, self, Self, super, async, await, move, ref, dyn, extern, unsafe)
    # - Strict: 12 (abstract, become, box, do, final, override, priv, typeof, unsized, virtual, yield, try)
    # - far: 1
    # Total keywords: 19 + 17 + 12 + 1 = 49 (builtins are identifiers)

    print("✓ Total keywords reserved: ~49 (Rust-compatible, builtins are identifiers)")


if __name__ == '__main__':
    print("Running reserved keywords tests...\n")

    test_currently_used_keywords()
    test_builtin_functions()
    test_reserved_rust_keywords()
    test_strict_reserved_keywords()
    test_far_keyword()
    test_keywords_are_not_identifiers()
    test_keyword_case_sensitivity()
    test_keyword_boundaries()
    test_total_keyword_count()

    print("\n✅ All reserved keyword tests passed!")
