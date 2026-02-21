"""
Additional edge case tests for the R65 lexer.
"""


from r65.compiler.frontend import tokenize, TokenType, LexerError


def test_nested_block_comments_not_supported():
    """Test that nested block comments are not supported (as per spec)."""
    # After first */ comment ends, leaving "still" as code
    source = "/* outer /* inner */ still = 1;"
    tokens = tokenize(source)

    # Should end after first */ so we should see "still"
    # Find the "still" token
    found_still = any(t.value == 'still' for t in tokens if t.type == TokenType.IDENTIFIER)
    assert found_still, "Block comments should not nest - 'still' should be parsed as code"

    print("✓ Block comments don't nest (as expected)")


def test_operators_adjacent():
    """Test operators without spaces."""
    source = "a+b*c-d/e&f|g^h<<i>>j"
    tokens = tokenize(source)

    # Should tokenize correctly even without spaces
    assert tokens[0].value == 'a'
    assert tokens[1].type == TokenType.PLUS
    assert tokens[2].value == 'b'
    assert tokens[3].type == TokenType.STAR

    print("✓ Operators without spaces test passed")


def test_attribute_parsing():
    """Test various attribute syntaxes."""
    source = "#[hw(0x2100)] #[mode(m8, x8)] #[preserves(A, X, Y)]"
    tokens = tokenize(source)

    # Verify we have the right structure
    assert tokens[0].type == TokenType.HASH
    assert tokens[1].type == TokenType.LBRACKET
    assert tokens[2].value == 'hw'

    print("✓ Attribute parsing test passed")


def test_multiline_expressions():
    """Test expressions spanning multiple lines."""
    source = """
    let result = a +
                 b *
                 c;
    """
    tokens = tokenize(source)

    # Should handle multiline correctly
    assert tokens[0].is_keyword('let')
    result_idx = next(i for i, t in enumerate(tokens) if t.value == 'result')
    assert result_idx >= 0

    print("✓ Multiline expressions test passed")


def test_hex_and_binary_literals():
    """Test various number formats."""
    source = "0x00 0xFF 0xDEADBEEF 0b0 0b1111 0b1010_0101"
    tokens = tokenize(source)

    assert tokens[0].value == 0x00
    assert tokens[1].value == 0xFF
    assert tokens[2].value == 0xDEADBEEF
    assert tokens[3].value == 0b0
    assert tokens[4].value == 0b1111
    assert tokens[5].value == 0b10100101

    print("✓ Hex and binary literals test passed")


def test_underscores_in_numbers():
    """Test underscores in number literals."""
    source = "1_000_000 0xFF_FF 0b1111_0000"
    tokens = tokenize(source)

    assert tokens[0].value == 1000000
    assert tokens[1].value == 0xFFFF
    assert tokens[2].value == 0b11110000

    print("✓ Underscores in numbers test passed")


def test_invalid_hex():
    """Test error handling for invalid hex literal."""
    source = "0x"  # Invalid - no digits

    try:
        tokens = tokenize(source)
        assert False, "Should have raised LexerError"
    except LexerError as e:
        assert "Invalid hexadecimal" in e.message

    print("✓ Invalid hex error test passed")


def test_invalid_binary():
    """Test error handling for invalid binary literal."""
    source = "0b"  # Invalid - no digits

    try:
        tokens = tokenize(source)
        assert False, "Should have raised LexerError"
    except LexerError as e:
        assert "Invalid binary" in e.message

    print("✓ Invalid binary error test passed")


def test_register_case_sensitivity():
    """Test that registers are case-sensitive."""
    source = "A a X x DBR dbr"
    tokens = tokenize(source)

    # A, X, DBR should be registers
    assert tokens[0].type == TokenType.REGISTER
    assert tokens[0].value == 'A'

    # a, x, dbr should be identifiers
    assert tokens[1].type == TokenType.IDENTIFIER
    assert tokens[1].value == 'a'

    assert tokens[2].type == TokenType.REGISTER
    assert tokens[2].value == 'X'

    assert tokens[3].type == TokenType.IDENTIFIER
    assert tokens[3].value == 'x'

    print("✓ Register case sensitivity test passed")


def test_pointer_syntax():
    """Test pointer type syntax."""
    source = "*u8 far *u16"
    tokens = tokenize(source)

    # New pointer syntax: *T (implied near) or far *T
    assert tokens[0].type == TokenType.STAR  # *
    assert tokens[1].type == TokenType.TYPE  # u8

    assert tokens[2].is_keyword('far')
    assert tokens[3].type == TokenType.STAR  # *
    assert tokens[4].type == TokenType.TYPE  # u16

    print("✓ Pointer syntax test passed")


if __name__ == '__main__':
    print("Running edge case tests...\n")

    test_nested_block_comments_not_supported()
    test_operators_adjacent()
    test_attribute_parsing()
    test_multiline_expressions()
    test_hex_and_binary_literals()
    test_underscores_in_numbers()
    test_invalid_hex()
    test_invalid_binary()
    test_register_case_sensitivity()
    test_pointer_syntax()

    print("\n✅ All edge case tests passed!")
