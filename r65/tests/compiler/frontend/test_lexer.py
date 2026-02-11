"""
Tests for the R65 lexer.
"""


from r65.compiler.frontend import tokenize, TokenType, LexerError


def test_simple_tokens():
    """Test basic single-character tokens."""
    source = "+ - * / % & | ^ ~ < > ! = @ ( ) { } [ ] ; : , . #"
    tokens = tokenize(source)

    expected_types = [
        TokenType.PLUS, TokenType.MINUS, TokenType.STAR, TokenType.SLASH,
        TokenType.PERCENT, TokenType.AMPERSAND, TokenType.PIPE, TokenType.CARET,
        TokenType.TILDE, TokenType.LT, TokenType.GT, TokenType.NOT,
        TokenType.ASSIGN, TokenType.AT, TokenType.LPAREN, TokenType.RPAREN,
        TokenType.LBRACE, TokenType.RBRACE, TokenType.LBRACKET, TokenType.RBRACKET,
        TokenType.SEMICOLON, TokenType.COLON, TokenType.COMMA, TokenType.DOT,
        TokenType.HASH, TokenType.EOF
    ]

    assert len(tokens) == len(expected_types)
    for token, expected in zip(tokens, expected_types):
        assert token.type == expected, f"Expected {expected}, got {token.type}"

    print("✓ Simple tokens test passed")


def test_two_char_operators():
    """Test two-character operators."""
    source = "== != <= >= << >> && || ->"
    tokens = tokenize(source)

    expected_types = [
        TokenType.EQ, TokenType.NE, TokenType.LE, TokenType.GE,
        TokenType.LSHIFT, TokenType.RSHIFT, TokenType.AND, TokenType.OR,
        TokenType.ARROW, TokenType.EOF
    ]

    assert len(tokens) == len(expected_types)
    for token, expected in zip(tokens, expected_types):
        assert token.type == expected

    print("✓ Two-character operators test passed")


def test_integers():
    """Test integer literals."""
    source = "42 0xFF 0b1010 0x1A2B 123_456"
    tokens = tokenize(source)

    assert tokens[0].type == TokenType.INTEGER
    assert tokens[0].value == 42

    assert tokens[1].type == TokenType.INTEGER
    assert tokens[1].value == 0xFF

    assert tokens[2].type == TokenType.INTEGER
    assert tokens[2].value == 0b1010

    assert tokens[3].type == TokenType.INTEGER
    assert tokens[3].value == 0x1A2B

    assert tokens[4].type == TokenType.INTEGER
    assert tokens[4].value == 123456

    print("✓ Integer literals test passed")


def test_integer_suffixes():
    """Test integer literals with type suffixes."""
    source = "255u8 1000u16 0xFFu8 0b1010i8 42"
    tokens = tokenize(source)

    assert tokens[0].type == TokenType.INTEGER
    assert tokens[0].value == 255

    assert tokens[1].type == TokenType.INTEGER
    assert tokens[1].value == 1000

    assert tokens[2].type == TokenType.INTEGER
    assert tokens[2].value == 0xFF

    assert tokens[3].type == TokenType.INTEGER
    assert tokens[3].value == 0b1010

    assert tokens[4].type == TokenType.INTEGER
    assert tokens[4].value == 42

    print("✓ Integer suffixes lexer test passed")


def test_keywords():
    """Test keyword recognition."""
    source = "fn far let mut const static if else loop while break continue return"
    tokens = tokenize(source)

    expected = [
        'fn', 'far', 'let', 'mut', 'const', 'static', 'if', 'else',
        'loop', 'while', 'break', 'continue', 'return'
    ]

    for i, keyword in enumerate(expected):
        assert tokens[i].type == TokenType.KEYWORD
        assert tokens[i].value == keyword

    print("✓ Keywords test passed")


def test_registers():
    """Test hardware register recognition."""
    source = "A X Y STATUS D DBR PBR S"
    tokens = tokenize(source)

    expected = ['A', 'X', 'Y', 'STATUS', 'D', 'DBR', 'PBR', 'S']

    for i, register in enumerate(expected):
        assert tokens[i].type == TokenType.REGISTER
        assert tokens[i].value == register

    print("✓ Register recognition test passed")


def test_types():
    """Test type keyword recognition."""
    source = "u8 u16 i8 i16 bool"
    tokens = tokenize(source)

    expected = ['u8', 'u16', 'i8', 'i16', 'bool']

    for i, type_name in enumerate(expected):
        assert tokens[i].type == TokenType.TYPE
        assert tokens[i].value == type_name

    print("✓ Type keywords test passed")


def test_booleans():
    """Test boolean literals."""
    source = "true false"
    tokens = tokenize(source)

    assert tokens[0].type == TokenType.BOOLEAN
    assert tokens[0].value == True

    assert tokens[1].type == TokenType.BOOLEAN
    assert tokens[1].value == False

    print("✓ Boolean literals test passed")


def test_identifiers():
    """Test identifier recognition."""
    source = "my_var player_x _temp calculate123"
    tokens = tokenize(source)

    expected = ['my_var', 'player_x', '_temp', 'calculate123']

    for i, identifier in enumerate(expected):
        assert tokens[i].type == TokenType.IDENTIFIER
        assert tokens[i].value == identifier

    print("✓ Identifiers test passed")


def test_comments():
    """Test comment handling."""
    source = """
    // This is a line comment
    let x = 10; // inline comment
    /* This is a
       block comment */
    let y = 20;
    """
    tokens = tokenize(source)

    # Should only have: let x = 10 ; let y = 20 ; EOF
    assert tokens[0].is_keyword('let')
    assert tokens[1].value == 'x'
    assert tokens[2].type == TokenType.ASSIGN
    assert tokens[3].value == 10
    assert tokens[4].type == TokenType.SEMICOLON
    assert tokens[5].is_keyword('let')
    assert tokens[6].value == 'y'

    print("✓ Comments test passed")


def test_simple_function():
    """Test tokenizing a simple function."""
    source = """
    fn add(a: u8, b: u8) -> u8 {
        return a + b;
    }
    """
    tokens = tokenize(source)

    # Verify key tokens
    assert tokens[0].is_keyword('fn')
    assert tokens[1].value == 'add'
    assert tokens[2].type == TokenType.LPAREN
    assert tokens[3].value == 'a'
    assert tokens[4].type == TokenType.COLON
    assert tokens[5].value == 'u8'

    print("✓ Simple function test passed")


def test_register_aliasing():
    """Test register aliasing syntax."""
    source = "let hitpoints @ A = 100;"
    tokens = tokenize(source)

    assert tokens[0].is_keyword('let')
    assert tokens[1].value == 'hitpoints'
    assert tokens[2].type == TokenType.AT
    assert tokens[3].type == TokenType.REGISTER
    assert tokens[3].value == 'A'
    assert tokens[4].type == TokenType.ASSIGN
    assert tokens[5].value == 100

    print("✓ Register aliasing test passed")


def test_attribute_syntax():
    """Test attribute syntax."""
    source = "#[mode(m8, x8)]"
    tokens = tokenize(source)

    assert tokens[0].type == TokenType.HASH
    assert tokens[1].type == TokenType.LBRACKET
    assert tokens[2].value == 'mode'
    assert tokens[3].type == TokenType.LPAREN

    print("✓ Attribute syntax test passed")


def test_line_column_tracking():
    """Test that line and column numbers are tracked correctly."""
    source = "let\nx\n=\n10;"
    tokens = tokenize(source)

    assert tokens[0].line == 1 and tokens[0].column == 1  # let
    assert tokens[1].line == 2 and tokens[1].column == 1  # x
    assert tokens[2].line == 3 and tokens[2].column == 1  # =
    assert tokens[3].line == 4 and tokens[3].column == 1  # 10

    print("✓ Line and column tracking test passed")


def test_unterminated_block_comment():
    """Test handling of unterminated block comments."""
    # Note: Lark's C_COMMENT doesn't error on unterminated comments,
    # it just doesn't match them, so they become regular tokens
    source = "/* This comment never ends"

    tokens = tokenize(source)
    # Should tokenize as / * This comment never ends
    assert tokens[0].type == TokenType.SLASH
    assert tokens[1].type == TokenType.STAR

    print("✓ Unterminated block comment handling test passed")


def test_example_program():
    """Test tokenizing a complete example program."""
    source = """
    #[hw(0x2100)]
    static mut INIDISP: u8;

    #[zeropage(0x20)]
    static mut FRAME_COUNT: u16 = 0;

        #[preserves(X, Y)]
    fn wait_vblank() {
        loop {
            let flag @ A = VBLANK_FLAG;
            if flag != 0 {
                VBLANK_FLAG = 0;
                break;
            }
        }
    }
    """

    tokens = tokenize(source)

    # Just verify we can tokenize without errors
    assert tokens[-1].type == TokenType.EOF
    assert len(tokens) > 50  # Should have many tokens

    print("✓ Example program test passed")


if __name__ == '__main__':
    print("Running lexer tests...\n")

    test_simple_tokens()
    test_two_char_operators()
    test_integers()
    test_keywords()
    test_registers()
    test_types()
    test_booleans()
    test_identifiers()
    test_comments()
    test_simple_function()
    test_register_aliasing()
    test_attribute_syntax()
    test_line_column_tracking()
    test_unterminated_block_comment()
    test_example_program()

    print("\n✅ All lexer tests passed!")
