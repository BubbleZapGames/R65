# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
R65 Language Acid Test

A comprehensive test that exercises all major language features together
in a realistic program structure. This validates that features work in
combination, not just in isolation.
"""

import pytest
from r65.compiler.frontend.parser import parse
from r65.compiler.frontend.macros import expand_macros
from r65.compiler.frontend import ast
from r65.compiler.hir.builder import HIRBuilder


# The acid test source - a complete R65 program using all features
ACID_TEST_SOURCE = """
// R65 Acid Test - Comprehensive Language Feature Test

#[snesrom(name="R65 ACID TEST", version=0x01)]

// --- Constants ---
const SCREEN_WIDTH: u16 = 256;
const SCREEN_HEIGHT: u16 = 224;
const MAX_ENTITIES: u8 = 8;
const TILE_SIZE: u8 = 8;
const PLAYER_SPEED: u8 = 2;

// --- Const Functions ---
const fn tile_offset(x: u8, y: u8) -> u16 {
    return (y as u16) * 32 + (x as u16);
}

const fn clamp_offset(x: u8, y: u8) -> u16 {
    if x > 31 { return tile_offset(31, y); }
    if y > 27 { return tile_offset(x, 27); }
    return tile_offset(x, y);
}

const fn direction_dx(dir: u8) -> i8 {
    return match dir {
        0 => 0, 1 => 0, 2 => -1, 3 => 1, _ => 0
    };
}

const fn make_shift_table() -> [u16; 8] {
    let mut t: [u16; 8] = [0; 8];
    let mut i: u8 = 0;
    while i < 8 {
        t[i] = 1 << (i as u16);
        i = i + 1;
    }
    return t;
}

// --- Type Aliases ---
type Callback = fn(u8) -> u8;
type FarCallback = far fn() -> u8;
type Word = u16;
type Byte = u8;
type Ptr = *u8;

// --- Enums ---
enum State { Idle = 0, Running, Jumping, Falling, Dead }
enum Direction { Up = 0, Down, Left, Right }

// --- Structs ---
struct Point { x: u16, y: u16 }

struct Entity {
    pos: Point,
    velocity_x: i8,
    velocity_y: i8,
    state: u8,
    health: u8,
    flags: u8
}

struct GameState {
    frame_count: u16,
    player_score: u16,
    level: u8,
    paused: bool
}

struct Palette { colors: [u8; 16], count: u8 }
struct Handler { callback: fn(u8) -> u8, priority: u8 }

// --- Derived Constants (after struct defs for offset_of) ---
const PLAYER_TILE: u16 = tile_offset(5, 3);
const CLAMPED: u16 = clamp_offset(50, 10);
const DIR_DX: i8 = direction_dx(3);
const POINT_Y_OFFSET: u8 = offset_of(Point, y);

static SHIFT_TABLE: [u16; 8] = make_shift_table();
static SINE_TABLE: [u8; 256] = [0; 256];

// --- Hardware Registers ---
#[hw(0x2100)] static mut INIDISP: u8;
#[hw(0x2101)] static mut OBSEL: u8;
#[hw(0x4200)] static mut NMITIMEN: u8;
#[hw(0x4212)] static mut HVBJOY: u8;

// --- Memory ---
#[stack(0x1F00, 0x1FFF)]

#[zeropage(0x00)] static mut FRAME_COUNTER: u16;
#[zeropage(0x02)] static mut BUTTONS: u16;
#[zeropage(0x04)] static mut BUTTONS_PRESSED: u16;
#[zeropage(0x06, register)] static mut SCRATCH0: u8;
#[zeropage(0x08, register)] static mut SCRATCH1: u16;

#[lowram(0x0100)] static mut PLAYER: Entity;
#[lowram] static mut ENEMIES: [Entity; 8];

#[ram] static mut GAME: GameState;
#[ram] static mut TILE_BUFFER: [u8; 1024] = [0; 1024];
#[ram] static mut MESSAGE: [u8; 32] = "Game Over!\\0";

// --- Function Pointers and Pointer Types ---
#[ram] static mut UPDATE_HANDLER: fn(u8);
#[ram] static mut STATE_HANDLERS: [fn(u8) -> u8; 4];
#[ram] static mut HANDLER_TABLE: [Handler; 4];
#[zeropage(0x10)] static mut DATA_PTR: *u8;
#[zeropage(0x12)] static mut FAR_PTR: far *u8;
#[zeropage(0x15)] static mut STRUCT_PTR: *Entity;
#[ram] static mut CURRENT_PALETTE: Palette;
#[ram] static mut MSG_PTR: *u8;

// --- Entry Point ---
#[entry]
fn main() {
    INIDISP = 0x80;
    NMITIMEN = 0x00;

    GAME.frame_count = 0;
    GAME.player_score = 0;
    GAME.level = 1;
    GAME.paused = false;

    init_player();
    let i @ X = 0;
    while i < MAX_ENTITIES {
        init_entity(i);
        i = i + 1;
    }

    INIDISP = 0x0F;
    NMITIMEN = 0x81;

    loop {
        wait_vblank();
        update_game();
        draw_entity(PLAYER.pos.x, PLAYER.pos.y, 0);
    }
}

// --- Initialization ---
fn init_player() {
    PLAYER.pos.x = SCREEN_WIDTH / 2;
    PLAYER.pos.y = SCREEN_HEIGHT / 2;
    PLAYER.velocity_x = 0;
    PLAYER.velocity_y = 0;
    PLAYER.state = State::Idle as u8;
    PLAYER.health = 100;
    PLAYER.flags = 0;
}

fn init_entity(index @ X: u16) {
    ENEMIES[X].pos.x = 0;
    ENEMIES[X].pos.y = 0;
    ENEMIES[X].state = State::Dead as u8;
    ENEMIES[X].health = 0;
}

// --- Game Logic ---
fn update_game() {
    if GAME.paused { return; }
    GAME.frame_count = GAME.frame_count + 1;
    read_input();

    // Player movement
    if BUTTONS & 0x0100 != 0 {
        PLAYER.velocity_x = PLAYER_SPEED as i8;
        PLAYER.state = State::Running as u8;
    } else if BUTTONS & 0x0200 != 0 {
        PLAYER.velocity_x = -(PLAYER_SPEED as i8);
    } else {
        PLAYER.velocity_x = 0;
    }
    let new_x: u16 = (PLAYER.pos.x as i16 + PLAYER.velocity_x as i16) as u16;
    if new_x < SCREEN_WIDTH { PLAYER.pos.x = new_x; }

    // Update enemies
    let i @ X = 0;
    loop {
        if i >= MAX_ENTITIES { break; }
        if ENEMIES[X].state != State::Dead as u8 { update_enemy(i); }
        i = i + 1;
    }

    // Check collisions (for loop)
    X = 0;
    for j in 0..MAX_ENTITIES {
        if ENEMIES[X].state != State::Dead as u8 {
            if check_collision(X) { handle_collision(X); }
        }
        X++;
    }
}

fn update_enemy(index @ X: u16) {
    let state: u8 = ENEMIES[X].state;
    let behavior: u8 = match state {
        0 => 0, 1 => 1, _ => 2
    };
    if behavior == 1 {
        if ENEMIES[X].pos.x < PLAYER.pos.x {
            ENEMIES[X].pos.x = ENEMIES[X].pos.x + 1;
        } else if ENEMIES[X].pos.x > PLAYER.pos.x {
            ENEMIES[X].pos.x = ENEMIES[X].pos.x - 1;
        }
    }
}

fn check_collision(index @ X: u16) -> bool {
    let dx: i16 = PLAYER.pos.x as i16 - ENEMIES[X].pos.x as i16;
    let dy: i16 = PLAYER.pos.y as i16 - ENEMIES[X].pos.y as i16;
    if dx < 0 { dx = -dx; }
    if dy < 0 { dy = -dy; }
    return dx < TILE_SIZE as i16 && dy < TILE_SIZE as i16;
}

fn handle_collision(enemy_index @ X: u16) {
    if PLAYER.health > 10 {
        PLAYER.health = PLAYER.health - 10;
    } else {
        PLAYER.health = 0;
        PLAYER.state = State::Dead as u8;
    }
    GAME.player_score = GAME.player_score + 100;
}

fn read_input() {
    let prev: u16 = BUTTONS;
    BUTTONS = 0;
    BUTTONS_PRESSED = BUTTONS & ~prev;
}

// --- Utility ---
fn wait_vblank() {
    loop {
        let status: u8 = HVBJOY;
        if status & 0x80 != 0 { break; }
    }
}

fn draw_entity(x: u16, y: u16, tile @ A: u8) {
    let screen_x: u8 = x as u8;
    A = tile;
    X = screen_x;
    Y = y;
}

#[preserves(X, Y)]
fn multiply(a @ A: u8, b: u8) -> u8 {
    let result: u8 = 0;
    let count @ X = b;
    while count > 0 {
        result = result + A;
        count = count - 1;
    }
    return result;
}

// --- Cross-Bank Functions ---
#[bank(1)]
far fn load_level_data() { A = 0; }

#[bank(2)]
#[mode(databank=inline)]
far fn play_sound(sound_id @ A: u8) { X = sound_id; }

// --- Interrupt Handlers ---
#[interrupt(nmi)]
fn vblank_handler() {
    FRAME_COUNTER = FRAME_COUNTER + 1;
    asm!("LDA $4210");
}

#[interrupt(irq)]
fn irq_handler() { asm!("RTI"); }

// --- Macros ---
macro_rules! inc_twice($reg:reg) { $reg++; $reg++; }
macro_rules! set_value($dest:reg, $val:expr) { $dest = $val; }
macro_rules! repeat_inc($($reg:reg),*) { $($reg++;)* }

fn test_macros() {
    inc_twice!(X);
    set_value!(A, 42);
    repeat_inc!(X, Y);
}

// --- Hardware Operations (asm, mode, block move, cop) ---
fn test_hardware_ops() {
    asm!("CLI");
    asm!("SEI");
    asm!("WAI");
    STATUS.A16 = false;
    STATUS.A16 = true;
    xba();
    A = 255;
    X = 0x1000;
    Y = 0x2000;
    mvn(0x00, 0x7E);
    cop(0x00);
}

// --- Operators (compound assign, NOT, shifts, inc/dec) ---
fn test_operators() {
    let mut value: u8 = 10;
    value += 5;
    value -= 2;
    value &= 0x0F;
    value |= 0x80;
    value ^= 0x01;
    let mut shift_val: u16 = 1;
    shift_val <<= 4;
    shift_val >>= 2;
    let mut counter: u8 = 0;
    counter++;
    counter--;
    let inverted: u8 = ~value;
    let flag: bool = true;
    let negated: bool = !flag;
    if !GAME.paused && PLAYER.health > 0 { A = 1; }
    let shifted: u8 = value << 4;
    let tile_off: u16 = (A as u16) << 4;
}

// --- Pointer Operations ---
fn test_pointers() {
    let byte: u8 = *DATA_PTR;
    *DATA_PTR = 42;
    let indexed: u8 = DATA_PTR[Y];
    DATA_PTR[X] = 0xFF;
    STRUCT_PTR = &PLAYER;
}

// --- Function Pointer Dispatch ---
fn dummy_handler(input @ A: u8) -> u8 { return A; }

fn test_dispatch(input @ A: u8) -> u8 {
    STATE_HANDLERS[0] = dummy_handler;
    STATE_HANDLERS[1] = dummy_handler;
    HANDLER_TABLE[0].callback = dummy_handler;
    HANDLER_TABLE[0].priority = 10;
    let handler: fn(u8) -> u8 = STATE_HANDLERS[X];
    return handler(input);
}

// --- Multiple Return Values ---
fn test_returns() {
    return PLAYER.pos.x, PLAYER.pos.y;
}

// --- Indexing and Struct Array Fields ---
fn test_indexing() {
    let val: u8 = SINE_TABLE[4 + 2];
    ENEMIES[X + 1].health = 50;
    CURRENT_PALETTE.count = 8;
    CURRENT_PALETTE.colors[0] = 0x00;
    let color @ A = CURRENT_PALETTE.colors[X];
}

// --- Block Expressions and If Expressions ---
fn test_expressions() {
    let val: u8 = {
        let temp: u8 = 5;
        temp + 1
    };
    let sign: u8 = if PLAYER.velocity_x > 0 { 1 } else { 0 };
    let category: u8 = if PLAYER.health > 75 { 3 }
        else if PLAYER.health > 25 { 1 }
        else { 0 };
}

fn trailing_return_fn() -> u8 {
    let x: u8 = 42;
    x
}

// --- Range Patterns and Dense Match ---
fn test_match_patterns(val @ A: u8) -> u8 {
    let tier: u8 = match val {
        0..5 => 0, 5..10 => 1, 10..20 => 2, _ => 3
    };
    let priority: u8 = match val {
        0..=3 => 0, 4..=7 => 1, 8..=15 => 2, _ => 3
    };
    // Dense constant match (LookupTable optimization)
    return match val {
        0 => 0, 1 => 1, 2 => 1, 3 => 2, 4 => 2, 5 => 2, _ => 0
    };
}

// --- String Literals, offset_of ---
fn test_new_features() {
    let greeting: *u8 = "Hello SNES!\\0";
    MSG_PTR = "Game Over\\0";
    let off: u8 = offset_of(GameState, player_score);
    A = offset_of(Point, y);
}

// --- Never Return Type ---
fn game_loop() -> ! {
    loop { wait_vblank(); update_game(); }
}
"""


class TestAcidTest:
    """Acid test for comprehensive language feature coverage."""

    def test_acid_test_parses_and_structure(self):
        """Test that the acid test parses and contains expected declarations."""
        program = parse(ACID_TEST_SOURCE)
        assert isinstance(program, ast.Program)
        assert len(program.items) > 0

        counts = {
            'const': 0, 'type_alias': 0, 'enum': 0, 'struct': 0,
            'static': 0, 'function': 0, 'stack': 0, 'macro': 0,
        }

        for item in program.items:
            if isinstance(item, ast.ConstDecl):
                counts['const'] += 1
            elif isinstance(item, ast.TypeAlias):
                counts['type_alias'] += 1
            elif isinstance(item, ast.EnumDecl):
                counts['enum'] += 1
            elif isinstance(item, ast.StructDecl):
                counts['struct'] += 1
            elif isinstance(item, ast.StaticDecl):
                counts['static'] += 1
            elif isinstance(item, ast.FunctionDecl):
                counts['function'] += 1
            elif isinstance(item, ast.StackDirective):
                counts['stack'] += 1
            elif isinstance(item, ast.MacroDecl):
                counts['macro'] += 1

        assert counts['const'] >= 8, f"Expected at least 8 constants, got {counts['const']}"
        assert counts['type_alias'] >= 5, f"Expected at least 5 type aliases, got {counts['type_alias']}"
        assert counts['enum'] >= 2, f"Expected at least 2 enums, got {counts['enum']}"
        assert counts['struct'] >= 5, f"Expected at least 5 structs, got {counts['struct']}"
        assert counts['static'] >= 16, f"Expected at least 16 statics, got {counts['static']}"
        assert counts['function'] >= 28, f"Expected at least 28 functions, got {counts['function']}"
        assert counts['macro'] >= 3, f"Expected at least 3 macros, got {counts['macro']}"

    def test_acid_test_builds_hir(self):
        """Test that the acid test builds HIR successfully."""
        program = parse(ACID_TEST_SOURCE)
        program = expand_macros(program)
        builder = HIRBuilder()
        hir_program = builder.build_program(program)

        assert len(hir_program.functions) >= 28
        assert len(hir_program.statics) >= 16
        assert len(hir_program.structs) >= 5
        assert len(hir_program.enums) >= 2

    def test_acid_test_features(self):
        """Test specific language features are present."""
        program = parse(ACID_TEST_SOURCE)

        features_found = {
            'entry_attr': False,
            'interrupt_attr': False,
            'mode_attr': False,
            'preserves_attr': False,
            'bank_directive': False,
            'snesrom_directive': False,
            'hw_attr': False,
            'zeropage_attr': False,
            'ram_attr': False,
            'immutable_static': False,
            'far_function': False,
            'match_expr': False,
            'register_param': False,
            'register_alias': False,
            'compound_assign': False,
            'unary_not': False,
            'pointer_deref': False,
            'address_of': False,
            'fn_ptr_array': False,
            'multiple_return': False,
            'macro_decl': False,
            'macro_invocation': False,
            'const_fn': False,
            'block_expr': False,
            'if_expr': False,
            'range_pattern': False,
            'string_literal_ptr': False,
            'never_return_type': False,
            'for_loop': False,
            'type_alias': False,
        }

        for item in program.items:
            if isinstance(item, ast.BankDirective):
                features_found['bank_directive'] = True
            elif isinstance(item, ast.SnesRomDirective):
                features_found['snesrom_directive'] = True
            elif isinstance(item, ast.TypeAlias):
                features_found['type_alias'] = True
            elif isinstance(item, ast.FunctionDecl):
                for attr in item.attributes:
                    if attr.name == 'entry':
                        features_found['entry_attr'] = True
                    elif attr.name == 'interrupt':
                        features_found['interrupt_attr'] = True
                    elif attr.name == 'mode':
                        features_found['mode_attr'] = True
                    elif attr.name == 'preserves':
                        features_found['preserves_attr'] = True

                if item.is_far:
                    features_found['far_function'] = True
                if item.is_const:
                    features_found['const_fn'] = True
                if isinstance(item.return_type, ast.NeverType):
                    features_found['never_return_type'] = True

                for param in item.params:
                    if param.binding is not None:
                        features_found['register_param'] = True

                self._check_statements(item.body.statements, features_found)

            elif isinstance(item, ast.StaticDecl):
                for attr in item.attributes:
                    if attr.name == 'hw':
                        features_found['hw_attr'] = True
                    elif attr.name == 'zeropage':
                        features_found['zeropage_attr'] = True
                    elif attr.name == 'ram':
                        features_found['ram_attr'] = True
                if not item.is_mut:
                    features_found['immutable_static'] = True
                if isinstance(item.var_type, ast.ArrayType):
                    if isinstance(item.var_type.element_type, ast.FunctionType):
                        features_found['fn_ptr_array'] = True

            elif isinstance(item, ast.MacroDecl):
                features_found['macro_decl'] = True

        for feature, found in features_found.items():
            assert found, f"Feature '{feature}' not found in acid test"

    def _check_statements(self, statements, features_found):
        """Recursively check statements for features."""
        for stmt in statements:
            if isinstance(stmt, ast.MacroInvocationStmtInner):
                features_found['macro_invocation'] = True
            elif isinstance(stmt, ast.ExprStmt) and isinstance(stmt.expr, ast.MacroInvocation):
                features_found['macro_invocation'] = True
            elif isinstance(stmt, ast.LetStmt):
                if stmt.binding is not None:
                    features_found['register_alias'] = True
                if isinstance(stmt.initializer, ast.MatchExpression):
                    features_found['match_expr'] = True
                if stmt.initializer:
                    self._check_expression(stmt.initializer, features_found)
            elif isinstance(stmt, ast.ExprStmt):
                self._check_expression(stmt.expr, features_found)
            elif isinstance(stmt, ast.ReturnStmt):
                if stmt.values and len(stmt.values) > 1:
                    features_found['multiple_return'] = True
            elif isinstance(stmt, ast.IfStmt):
                self._check_expression(stmt.condition, features_found)
                self._check_statements(stmt.then_block.statements, features_found)
                if stmt.else_block:
                    if isinstance(stmt.else_block, ast.Block):
                        self._check_statements(stmt.else_block.statements, features_found)
                    elif isinstance(stmt.else_block, ast.IfStmt):
                        self._check_statements([stmt.else_block], features_found)
            elif isinstance(stmt, ast.WhileStmt):
                self._check_statements(stmt.body.statements, features_found)
            elif isinstance(stmt, ast.LoopStmt):
                self._check_statements(stmt.body.statements, features_found)
            elif isinstance(stmt, ast.ForStmt):
                features_found['for_loop'] = True
                self._check_statements(stmt.body.statements, features_found)

    def _check_expression(self, expr, features_found):
        """Check expression for features."""
        if isinstance(expr, ast.CompoundAssignment):
            features_found['compound_assign'] = True
        elif isinstance(expr, ast.UnaryOp):
            if expr.op in ('!', '~'):
                features_found['unary_not'] = True
        elif isinstance(expr, ast.Dereference):
            features_found['pointer_deref'] = True
        elif isinstance(expr, ast.AddressOf):
            features_found['address_of'] = True
        elif isinstance(expr, ast.BinaryOp):
            self._check_expression(expr.left, features_found)
            self._check_expression(expr.right, features_found)
        elif isinstance(expr, ast.Assignment):
            self._check_expression(expr.value, features_found)
        elif isinstance(expr, ast.BlockExpression):
            features_found['block_expr'] = True
        elif isinstance(expr, ast.IfExpression):
            features_found['if_expr'] = True
        elif isinstance(expr, ast.StringLiteral):
            features_found['string_literal_ptr'] = True
        elif isinstance(expr, ast.MatchExpression):
            features_found['match_expr'] = True
            for arm in expr.arms:
                if isinstance(arm.pattern, ast.RangePattern):
                    features_found['range_pattern'] = True
                elif isinstance(arm.pattern, ast.OrPattern):
                    for sub in arm.pattern.patterns:
                        if isinstance(sub, ast.RangePattern):
                            features_found['range_pattern'] = True
