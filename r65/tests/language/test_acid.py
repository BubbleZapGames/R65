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
// =============================================================================
// R65 Acid Test - Comprehensive Language Feature Test
// =============================================================================

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------
const SCREEN_WIDTH: u16 = 256;
const SCREEN_HEIGHT: u16 = 224;
const MAX_ENTITIES: u8 = 8;
const TILE_SIZE: u8 = 8;
const PLAYER_SPEED: u8 = 2;

// -----------------------------------------------------------------------------
// Type Aliases
// -----------------------------------------------------------------------------
type Callback = fn(u8) -> u8;
type FarCallback = far fn() -> u8;

// -----------------------------------------------------------------------------
// Enums
// -----------------------------------------------------------------------------
enum State { Idle = 0, Running, Jumping, Falling, Dead }
enum Direction { Up = 0, Down, Left, Right }

// -----------------------------------------------------------------------------
// Structs
// -----------------------------------------------------------------------------
struct Point {
    x: u16,
    y: u16
}

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

// Struct with array field
struct Palette {
    colors: [u8; 16],
    count: u8
}

// Struct for dispatch table
struct Handler {
    callback: fn(u8) -> u8,
    priority: u8
}

// -----------------------------------------------------------------------------
// Hardware Registers
// -----------------------------------------------------------------------------
#[hw(0x2100)] static mut INIDISP: u8;
#[hw(0x2101)] static mut OBSEL: u8;
#[hw(0x4200)] static mut NMITIMEN: u8;
#[hw(0x4212)] static mut HVBJOY: u8;

// -----------------------------------------------------------------------------
// Memory Declarations
// -----------------------------------------------------------------------------
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

#[rom(0x8000)] static SINE_TABLE: [u8; 256] = [0; 256];

// -----------------------------------------------------------------------------
// Function Pointers and Pointer Types
// -----------------------------------------------------------------------------
#[ram] static mut UPDATE_HANDLER: fn(u8);

// Array of function pointers (dispatch table)
#[ram] static mut STATE_HANDLERS: [fn(u8) -> u8; 4];

// Array of handlers with function pointer fields
#[ram] static mut HANDLER_TABLE: [Handler; 4];

// Pointer types
#[zeropage(0x10)] static mut *DATA_PTR: u8;
#[zeropage(0x12)] static far mut *FAR_PTR: u8;
#[zeropage(0x15)] static mut *STRUCT_PTR: Entity;

// -----------------------------------------------------------------------------
// Entry Point
// -----------------------------------------------------------------------------
#[entry]
fn main() {
    // Initialize hardware
    INIDISP = 0x80;  // Force blank
    NMITIMEN = 0x00; // Disable interrupts during init

    // Initialize game state
    GAME.frame_count = 0;
    GAME.player_score = 0;
    GAME.level = 1;
    GAME.paused = false;

    // Initialize player
    init_player();

    // Initialize enemies
    let i @ X = 0;
    while i < MAX_ENTITIES {
        init_entity(i);
        i = i + 1;
    }

    // Enable display
    INIDISP = 0x0F;
    NMITIMEN = 0x81;  // Enable NMI

    // Main loop
    loop {
        wait_vblank();
        update_game();
        render_frame();
    }
}

// -----------------------------------------------------------------------------
// Initialization Functions
// -----------------------------------------------------------------------------
fn init_player() {
    PLAYER.pos.x = SCREEN_WIDTH / 2;
    PLAYER.pos.y = SCREEN_HEIGHT / 2;
    PLAYER.velocity_x = 0;
    PLAYER.velocity_y = 0;
    PLAYER.state = State::Idle as u8;
    PLAYER.health = 100;
    PLAYER.flags = 0;
}

fn init_entity(index @ X: u8) {
    ENEMIES[X].pos.x = 0;
    ENEMIES[X].pos.y = 0;
    ENEMIES[X].state = State::Dead as u8;
    ENEMIES[X].health = 0;
}

// -----------------------------------------------------------------------------
// Game Logic
// -----------------------------------------------------------------------------
#[mode(m8, x8)]
fn update_game() {
    if GAME.paused {
        return;
    }

    GAME.frame_count = GAME.frame_count + 1;

    // Read input
    read_input();

    // Update player
    update_player();

    // Update enemies
    let i @ X = 0;
    loop {
        if i >= MAX_ENTITIES { break; }
        if ENEMIES[X].state != State::Dead as u8 {
            update_enemy(i);
        }
        i = i + 1;
    }

    // Check collisions
    check_collisions();
}

fn update_player() {
    // Handle movement based on input
    if BUTTONS & 0x0100 != 0 {  // Right
        PLAYER.velocity_x = PLAYER_SPEED as i8;
        PLAYER.state = State::Running as u8;
    } else if BUTTONS & 0x0200 != 0 {  // Left
        PLAYER.velocity_x = -(PLAYER_SPEED as i8);
        PLAYER.state = State::Running as u8;
    } else {
        PLAYER.velocity_x = 0;
        if PLAYER.state == State::Running as u8 {
            PLAYER.state = State::Idle as u8;
        }
    }

    // Apply velocity
    let new_x: u16 = (PLAYER.pos.x as i16 + PLAYER.velocity_x as i16) as u16;

    // Clamp to screen bounds
    if new_x < SCREEN_WIDTH {
        PLAYER.pos.x = new_x;
    }

    // Handle jumping
    if BUTTONS_PRESSED & 0x0080 != 0 && PLAYER.state != State::Jumping as u8 {
        PLAYER.velocity_y = -8;
        PLAYER.state = State::Jumping as u8;
    }
}

fn update_enemy(index @ X: u8) {
    // Simple enemy AI using match
    let state: u8 = ENEMIES[X].state;
    let behavior: u8 = match state {
        0 => 0,  // Idle - do nothing
        1 => 1,  // Running - chase player
        _ => 2   // Default - wander
    };

    if behavior == 1 {
        // Chase player
        if ENEMIES[X].pos.x < PLAYER.pos.x {
            ENEMIES[X].pos.x = ENEMIES[X].pos.x + 1;
        } else if ENEMIES[X].pos.x > PLAYER.pos.x {
            ENEMIES[X].pos.x = ENEMIES[X].pos.x - 1;
        }
    }
}

// -----------------------------------------------------------------------------
// Collision Detection
// -----------------------------------------------------------------------------
fn check_collisions() {
    let i @ X = 0;
    while i < MAX_ENTITIES {
        if ENEMIES[X].state != State::Dead as u8 {
            if check_entity_collision(i) {
                handle_collision(i);
            }
        }
        i = i + 1;
    }
}

fn check_entity_collision(index @ X: u8) -> bool {
    let dx: i16 = PLAYER.pos.x as i16 - ENEMIES[X].pos.x as i16;
    let dy: i16 = PLAYER.pos.y as i16 - ENEMIES[X].pos.y as i16;

    // Simple bounding box check
    if dx < 0 { dx = -dx; }
    if dy < 0 { dy = -dy; }

    return dx < TILE_SIZE as i16 && dy < TILE_SIZE as i16;
}

fn handle_collision(enemy_index @ X: u8) {
    // Damage player
    if PLAYER.health > 10 {
        PLAYER.health = PLAYER.health - 10;
    } else {
        PLAYER.health = 0;
        PLAYER.state = State::Dead as u8;
    }

    // Add score
    GAME.player_score = GAME.player_score + 100;
}

// -----------------------------------------------------------------------------
// Input Handling
// -----------------------------------------------------------------------------
fn read_input() {
    // Store previous buttons for edge detection
    let prev: u16 = BUTTONS;

    // Read new button state (simplified)
    BUTTONS = 0;

    // Detect newly pressed buttons
    BUTTONS_PRESSED = BUTTONS & ~prev;
}

// -----------------------------------------------------------------------------
// Rendering
// -----------------------------------------------------------------------------
fn render_frame() {
    // Draw player
    draw_entity(PLAYER.pos.x, PLAYER.pos.y, 0);

    // Draw enemies
    let i @ X = 0;
    while i < MAX_ENTITIES {
        if ENEMIES[X].state != State::Dead as u8 {
            draw_entity(ENEMIES[X].pos.x, ENEMIES[X].pos.y, 1);
        }
        i = i + 1;
    }
}

fn draw_entity(x: u16, y: u16, tile @ A: u8) {
    // Simplified sprite drawing
    let screen_x: u8 = x as u8;
    let screen_y: u8 = y as u8;

    // Would write to OAM here
    A = tile;
    X = screen_x;
    Y = screen_y;
}

// -----------------------------------------------------------------------------
// Utility Functions
// -----------------------------------------------------------------------------
fn wait_vblank() {
    // Wait for vblank
    loop {
        let status: u8 = HVBJOY;
        if status & 0x80 != 0 { break; }
    }
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

// -----------------------------------------------------------------------------
// Far Functions (Cross-Bank)
// -----------------------------------------------------------------------------
#[bank(1)]
far fn load_level_data() {
    // Load level data from ROM bank 1
    A = 0;
}

#[bank(2, data_bank=inline)]
far fn play_sound(sound_id @ A: u8) {
    // Play sound from audio bank
    X = sound_id;
}

// -----------------------------------------------------------------------------
// Interrupt Handlers
// -----------------------------------------------------------------------------
#[interrupt(nmi)]
fn vblank_handler() {
    FRAME_COUNTER = FRAME_COUNTER + 1;

    // Acknowledge NMI by reading RDNMI
    asm!("LDA $4210");
}

#[interrupt(irq)]
fn irq_handler() {
    // Handle IRQ
    asm!("RTI");
}

// -----------------------------------------------------------------------------
// Inline Assembly
// -----------------------------------------------------------------------------
fn enable_interrupts() {
    asm!("CLI");
}

fn disable_interrupts() {
    asm!("SEI");
}

fn halt() {
    asm!("WAI");
}

// -----------------------------------------------------------------------------
// Mode Control
// -----------------------------------------------------------------------------
#[mode(m16, x16)]
fn set_16bit_mode() {
    REP(0x30);
}

#[mode(m8, x8)]
fn set_8bit_mode() {
    SEP(0x30);
}

fn swap_accum_bytes() {
    xba();
}

// -----------------------------------------------------------------------------
// Macros
// -----------------------------------------------------------------------------
macro_rules! inc_twice($reg:reg) {
    $reg++;
    $reg++;
}

macro_rules! set_value($dest:reg, $val:expr) {
    $dest = $val;
}

macro_rules! repeat_inc($($reg:reg),*) {
    $($reg++;)*
}

fn test_macros() {
    // Simple macro invocation
    inc_twice!(X);

    // Macro with expression argument
    set_value!(A, 42);

    // Macro with repetition
    repeat_inc!(X, Y);
}

// -----------------------------------------------------------------------------
// Compound Assignment and Increment/Decrement
// -----------------------------------------------------------------------------
fn test_compound_ops() {
    let mut value: u8 = 10;
    value += 5;
    value -= 2;
    value &= 0x0F;
    value |= 0x80;
    value ^= 0x01;

    let mut shift_val: u16 = 1;
    shift_val <<= 4;
    shift_val >>= 2;

    // Increment/decrement
    let mut counter: u8 = 0;
    counter++;
    counter++;
    counter--;
}

// -----------------------------------------------------------------------------
// Bitwise and Logical NOT
// -----------------------------------------------------------------------------
fn test_not_operators() {
    let mask: u8 = 0x0F;
    let inverted: u8 = ~mask;  // Bitwise NOT

    let flag: bool = true;
    let negated: bool = !flag;  // Logical NOT

    // Complex boolean with NOT
    if !GAME.paused && PLAYER.health > 0 {
        A = 1;
    }
}

// -----------------------------------------------------------------------------
// Shift Operators
// -----------------------------------------------------------------------------
fn test_shifts() {
    let val: u8 = 1;
    let shifted_left: u8 = val << 4;   // 0x10
    let shifted_right: u8 = val >> 1;  // 0x00

    // Shift in expressions
    let tile_offset: u16 = (A as u16) << 4;
}

// -----------------------------------------------------------------------------
// Pointer Operations
// -----------------------------------------------------------------------------
fn test_pointers() {
    // Dereference pointer
    let byte: u8 = *DATA_PTR;
    *DATA_PTR = 42;

    // Indexed pointer access
    let indexed: u8 = DATA_PTR[Y];
    DATA_PTR[X] = 0xFF;

    // Pointer to struct
    STRUCT_PTR = &PLAYER;
}

// -----------------------------------------------------------------------------
// Function Pointer Dispatch
// -----------------------------------------------------------------------------
fn dummy_handler(input @ A: u8) -> u8 {
    return A;
}

fn init_handlers() {
    // Initialize array of function pointers
    STATE_HANDLERS[0] = dummy_handler;
    STATE_HANDLERS[1] = dummy_handler;

    // Initialize struct with function pointer
    HANDLER_TABLE[0].callback = dummy_handler;
    HANDLER_TABLE[0].priority = 10;
}

fn dispatch_handler(state @ X: u8, input @ A: u8) -> u8 {
    // Call through function pointer array
    let handler: fn(u8) -> u8 = STATE_HANDLERS[X];
    return handler(input);
}

// -----------------------------------------------------------------------------
// Multiple Return Values (via parenthesized tuple syntax)
// -----------------------------------------------------------------------------
fn get_position() {
    // Multiple values returned via registers (tuple syntax)
    return (PLAYER.pos.x, PLAYER.pos.y);
}

fn get_registers() {
    // Return all three registers (tuple syntax)
    return (A, X, Y);
}

// -----------------------------------------------------------------------------
// Complex Array Indexing
// -----------------------------------------------------------------------------
fn test_complex_indexing() {
    let base: u8 = 4;
    let offset: u8 = 2;

    // Array index with expression
    let val: u8 = SINE_TABLE[base + offset];

    // Chained access with computed index
    ENEMIES[X + 1].health = 50;
}

// -----------------------------------------------------------------------------
// Struct with Array Field
// -----------------------------------------------------------------------------
#[ram] static mut CURRENT_PALETTE: Palette;

fn init_palette() {
    CURRENT_PALETTE.count = 8;
    CURRENT_PALETTE.colors[0] = 0x00;
    CURRENT_PALETTE.colors[1] = 0x15;

    // Access array in struct with index
    let color @ A = CURRENT_PALETTE.colors[X];
}

// -----------------------------------------------------------------------------
// Block Move Operations
// -----------------------------------------------------------------------------
fn test_block_move() {
    // Setup for block move
    A = 255;        // count - 1
    X = 0x1000;     // src_addr
    Y = 0x2000;     // dest_addr
    mvn(0x00, 0x7E);  // Move forward

    A = 127;
    mvp(0x7E, 0x00);  // Move backward
}

// -----------------------------------------------------------------------------
// Software Interrupt
// -----------------------------------------------------------------------------
fn trigger_cop() {
    cop(0x00);  // COP with signature byte
}
"""


class TestAcidTest:
    """Acid test for comprehensive language feature coverage."""

    def test_acid_test_parses(self):
        """Test that the complete acid test program parses successfully."""
        program = parse(ACID_TEST_SOURCE)
        assert isinstance(program, ast.Program)
        assert len(program.items) > 0

    def test_acid_test_structure(self):
        """Test the acid test contains expected declarations."""
        program = parse(ACID_TEST_SOURCE)

        # Count declaration types
        counts = {
            'const': 0,
            'type_alias': 0,
            'enum': 0,
            'struct': 0,
            'static': 0,
            'function': 0,
            'stack': 0,
            'macro': 0,
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

        # Verify we have multiple of each type
        assert counts['const'] >= 5, f"Expected at least 5 constants, got {counts['const']}"
        assert counts['enum'] >= 2, f"Expected at least 2 enums, got {counts['enum']}"
        assert counts['struct'] >= 5, f"Expected at least 5 structs, got {counts['struct']}"
        assert counts['static'] >= 15, f"Expected at least 15 statics, got {counts['static']}"
        assert counts['function'] >= 25, f"Expected at least 25 functions, got {counts['function']}"
        assert counts['macro'] >= 3, f"Expected at least 3 macros, got {counts['macro']}"

    def test_acid_test_builds_hir(self):
        """Test that the acid test builds HIR successfully."""
        program = parse(ACID_TEST_SOURCE)
        program = expand_macros(program)  # Expand macros before HIR
        builder = HIRBuilder()
        hir_program = builder.build_program(program)

        # Verify HIR structure
        assert len(hir_program.functions) >= 25
        assert len(hir_program.statics) >= 15
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
            'bank_attr': False,
            'hw_attr': False,
            'zeropage_attr': False,
            'ram_attr': False,
            'rom_attr': False,
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
        }

        for item in program.items:
            if isinstance(item, ast.FunctionDecl):
                for attr in item.attributes:
                    if attr.name == 'entry':
                        features_found['entry_attr'] = True
                    elif attr.name == 'interrupt':
                        features_found['interrupt_attr'] = True
                    elif attr.name == 'mode':
                        features_found['mode_attr'] = True
                    elif attr.name == 'preserves':
                        features_found['preserves_attr'] = True
                    elif attr.name == 'bank':
                        features_found['bank_attr'] = True

                if item.is_far:
                    features_found['far_function'] = True

                # Check for register parameters
                for param in item.params:
                    if param.binding is not None:
                        features_found['register_param'] = True

                # Check function body for features
                self._check_statements(item.body.statements, features_found)

            elif isinstance(item, ast.StaticDecl):
                for attr in item.attributes:
                    if attr.name == 'hw':
                        features_found['hw_attr'] = True
                    elif attr.name == 'zeropage':
                        features_found['zeropage_attr'] = True
                    elif attr.name == 'ram':
                        features_found['ram_attr'] = True
                    elif attr.name == 'rom':
                        features_found['rom_attr'] = True

                # Check for array of function pointers
                if isinstance(item.var_type, ast.ArrayType):
                    if isinstance(item.var_type.element_type, ast.FunctionType):
                        features_found['fn_ptr_array'] = True

            elif isinstance(item, ast.MacroDecl):
                features_found['macro_decl'] = True

        # Verify all features found
        for feature, found in features_found.items():
            assert found, f"Feature '{feature}' not found in acid test"

    def _check_statements(self, statements, features_found):
        """Recursively check statements for features."""
        for stmt in statements:
            if isinstance(stmt, ast.MacroInvocationStmtInner):
                features_found['macro_invocation'] = True
            elif isinstance(stmt, ast.LetStmt):
                if stmt.binding is not None:
                    features_found['register_alias'] = True
                if isinstance(stmt.initializer, ast.MatchExpression):
                    features_found['match_expr'] = True
                # Check initializer for expressions
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
