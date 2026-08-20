# Traits and Dynamic Dispatch

## Overview

R65 traits provide **TypeId-based dynamic dispatch** for polymorphism. Any struct implementing a trait automatically gets a `TypeId` byte inserted at offset 0, enabling heterogeneous collections while keeping dispatch fast and predictable.

**Design Principles**:
- No `dyn` keyword - bare trait name for pointers: `*Drawable`
- Automatic TypeId insertion at struct offset 0
- Per-struct TypeId - same ID across all traits a struct implements
- O(1) dispatch via compiler-generated jump tables
- Near/far trait distinction - all methods in a trait must be same calling convention

---

## Trait Definition

### Basic Syntax

```rust
trait Drawable {
    fn draw(*self, x @ X: u16, y @ Y: u16);
    fn get_width(*self) -> u8;
}
```

**Rules**:
- All methods must take `*self` as the first parameter
- Methods can have additional parameters using standard R65 calling conventions
- Methods can return values
- No associated types

### Default Method Bodies

A trait method may supply a body instead of ending in `;`. Implementors that omit
the method inherit that body:

```rust
trait Drawable {
    fn get_width(*self) -> u8;
    fn get_height(*self) -> u8;

    // Default implementation — override it or inherit it
    fn get_perimeter(*self) -> u16 {
        return ((self.get_width() as u16) + (self.get_height() as u16)) << 1;
    }
}

impl Drawable for Player {
    fn get_width(*self) -> u8 { return 16; }
    fn get_height(*self) -> u8 { return 24; }
    // get_perimeter inherited
}

impl Drawable for Bullet {
    fn get_width(*self) -> u8 { return 4; }
    fn get_height(*self) -> u8 { return 4; }
    fn get_perimeter(*self) -> u16 { return 16; }   // overrides the default
}
```

**Rules**:
- The body is **copied into each implementor** that omits the method, then compiled
  as an ordinary method (`Player__get_perimeter`). There is no shared
  copy and no extra indirection — a defaulted method costs exactly what writing it
  out by hand costs.
- Because each copy is compiled against a concrete struct, `self.field` and
  `self.method()` inside a default body resolve to **that implementor's** fields and
  methods, statically. Calls in a default body are direct `JSR`/`JSL`, not dispatch.
- A method with no default body is still mandatory — omitting it is an error.
- Type errors in a default body are reported per implementor (a body that reads
  `self.hp` only compiles for structs that have an `hp` field).
- Defaulted methods behave normally under `*dyn` dispatch: each implementor's copy
  gets its own dispatch table entry.
- Default bodies carry no attributes; `far` and `far *self` are taken from the trait
  method's own declaration.

**Code size note**: N implementors inheriting one default body produce N copies of
that code. For a large default shared by many types, call a free function from the
default body so only the call site is duplicated.

### Associated Constants

Traits can declare compile-time constants that implementors must define:

```rust
trait Drawable {
    const WIDTH: u8;
    const HEIGHT: u8;
    fn draw(*self, x @ X: u16, y @ Y: u16);
}
```

**Rules**:
- Constants must be compile-time evaluable (same rules as `const` declarations)
- Only primitive types: `u8`, `u16`, `i8`, `i16`, `bool`
- No arrays or pointers in associated constants

**Implementing associated constants**:

```rust
impl Drawable for Player {
    const WIDTH: u8 = 16;
    const HEIGHT: u8 = 24;

    fn draw(*self, x @ X: u16, y @ Y: u16) {
        // Drawing logic
    }
}

impl Drawable for Bullet {
    const WIDTH: u8 = 4;
    const HEIGHT: u8 = 4;

    fn draw(*self, x @ X: u16, y @ Y: u16) { /* ... */ }
}
```

**Accessing associated constants**:

Associated constants are compile-time only. They can only be accessed via concrete types, not through trait pointers:

```rust
// Via concrete type - compile-time resolved
let w: u8 = Player::WIDTH;      // 16
let h: u8 = Bullet::HEIGHT;     // 4

// ERROR: Cannot access associated constant through trait pointer
let obj: *Drawable = &player;
let w: u8 = obj.WIDTH;          // Compile error!
```

If you need runtime access to type-specific values, use a trait method instead:

```rust
trait Drawable {
    const WIDTH: u8;            // Compile-time only
    fn get_width(*self) -> u8;  // Runtime accessible
}

impl Drawable for Player {
    const WIDTH: u8 = 16;
    fn get_width(*self) -> u8 { return 16; }
}
```

### Near Traits (Default)

Near traits use JSR/RTS calling convention. All methods are near functions:

```rust
trait Updatable {
    fn update(*self);
    fn reset(*self);
}
```

### Far Traits

Far traits use JSL/RTL calling convention for cross-bank calls. All methods must be declared with `far fn`:

```rust
trait Renderable {
    far fn render(*self);
    far fn get_bank(*self) -> u8;
}
```

**Constraint**: A trait must be entirely near or entirely far. Mixing is not allowed:

```rust
// ERROR: Cannot mix near and far methods in a trait
trait Invalid {
    fn near_method(*self);      // Near
    far fn far_method(*self);   // Far - ERROR
}
```

---

## Implementing Traits

### Basic Implementation

```rust
struct Player {
    x: u8,
    y: u8,
    health: u8
}

impl Drawable for Player {
    fn draw(*self, x @ X: u16, y @ Y: u16) {
        // Drawing logic
    }

    fn get_width(*self) -> u8 {
        return 16;
    }
}
```

### Implementation Rules

1. **All methods required** - Must implement every method in the trait that has no
   [default body](#default-method-bodies)
2. **Exact signature match** - Method signatures must match trait definition exactly
3. **TypeId insertion** - Compiler automatically adds TypeId field at offset 0

### Multiple Trait Implementation

A struct can implement multiple traits:

```rust
struct Enemy {
    x: u8,
    y: u8,
    damage: u8
}

impl Drawable for Enemy {
    fn draw(*self, x @ X: u16, y @ Y: u16) { /* ... */ }
    fn get_width(*self) -> u8 { return 8; }
}

impl Updatable for Enemy {
    fn update(*self) { /* ... */ }
    fn reset(*self) { /* ... */ }
}
```

**TypeId sharing**: The struct gets a single TypeId used for all trait dispatch.

### Near/Far Constraint

A struct cannot implement both near and far traits:

```rust
trait NearTrait { fn method(*self); }
trait FarTrait { far fn method(*self); }

struct MyStruct { data: u8 }

impl NearTrait for MyStruct { /* ... */ }  // OK
impl FarTrait for MyStruct { /* ... */ }   // ERROR: MyStruct already has near trait
```

---

## Trait Inheritance (Supertraits)

A trait can require ("inherit from") one or more **supertraits**, listed after a `:`:

```rust
trait Position { fn px(*self) -> u8; }
trait Sprite   { fn tile(*self) -> u8; }

trait Drawable: Position + Sprite {
    fn draw(*self);
}
```

### Requirement

Implementing a subtrait requires implementing every transitive supertrait. The
implementor still writes one `impl` block per trait:

```rust
struct Player { x: u8 }

impl Position for Player { fn px(*self) -> u8 { return self.x; } }
impl Sprite   for Player { fn tile(*self) -> u8 { return 7; } }
impl Drawable for Player { fn draw(*self) { /* ... */ } }
// Omitting impl Position or impl Sprite is a compile error.
```

### Calling Inherited Methods

A `*dyn Drawable` can call `Drawable`'s own methods **and** all inherited supertrait
methods. An inherited method dispatches through the supertrait's own jump table:

```rust
let d: *dyn Drawable = &PLAYER;
d.draw();   // Drawable's own method
d.px();     // inherited from Position (dispatched by TypeId)
```

### Upcasting

Because every implementor carries a single TypeId byte at offset 0, a `*dyn Sub` and a
`*dyn Super` to the same object are bit-identical. Upcasting is therefore a zero-cost,
representation-preserving coercion:

```rust
let d: *dyn Drawable = &PLAYER;
let p: *dyn Position = d;   // upcast, no runtime cost
p.px();
```

### Rules

- Supertraits are named only on the trait declaration, never on `impl`.
- The supertrait graph must be acyclic.
- A subtrait may not redeclare a method or constant name already declared by any
  transitive supertrait (keeps dispatch unambiguous).
- A trait and all its supertraits must share the same near/far calling convention.
- A subtrait's own behavior lives in its `impl` blocks (where `self` is the
  concrete type, so supertrait methods are callable directly), or in a default
  method body on the subtrait itself.

---

## TypeId System

### Automatic Insertion

When a struct implements any trait, the compiler inserts a `__type_id: u8` field at offset 0:

```rust
// Source code:
struct Player { x: u8, y: u8 }
impl Drawable for Player { /* ... */ }

// Actual memory layout:
// [TypeId=1, x, y]
// Offset 0: __type_id (1 byte)
// Offset 1: x (1 byte)
// Offset 2: y (1 byte)
// Total: 3 bytes
```

### TypeId Assignment

- TypeId 0 is reserved (invalid/null)
- Each struct with trait impls gets a unique TypeId (1, 2, 3, ...)
- TypeId is consistent across all traits the struct implements
- TypeId is assigned at compile time

### Size Impact

| Original Size | With Trait Impl |
|---------------|-----------------|
| N bytes | N + 1 bytes |

```rust
struct Small { a: u8 }              // 1 byte without traits
impl Drawable for Small { /* ... */ }  // 2 bytes with trait

struct Large { data: [u8; 100] }    // 100 bytes without traits
impl Drawable for Large { /* ... */ }  // 101 bytes with trait
```

---

## Type Introspection and Downcasting

### Accessing TypeId

The `type_id()` method is available on any trait pointer and returns the underlying `__type_id` field:

```rust
let obj: *Drawable = &player;
let id: u8 = obj.type_id();     // Returns Player's TypeId (e.g., 1)
```

This is equivalent to reading `obj.__type_id` directly. The `type_id()` syntax is preferred for clarity.

### Compile-Time TypeId Constants

Each struct that implements a trait has a `TYPE_ID` constant:

```rust
// Compiler-generated constants
Player::TYPE_ID     // e.g., 1
Enemy::TYPE_ID      // e.g., 2
Bullet::TYPE_ID     // e.g., 3
```

### Downcasting

Use `type_id()` comparison to safely downcast from trait pointer to concrete type:

```rust
fn handle_collision(obj: *Drawable) {
    if obj.type_id() == Player::TYPE_ID {
        // Safe to downcast - we know it's a Player
        let player: *Player = obj as *Player;
        player.health = player.health - 10;
    } else if obj.type_id() == Enemy::TYPE_ID {
        let enemy: *Enemy = obj as *Enemy;
        enemy.damage = enemy.damage + 1;
    }
}
```

### Unchecked Downcast

Casting without checking `type_id()` is allowed but unsafe:

```rust
// Dangerous - only do this if you're certain of the type
let player: *Player = obj as *Player;
```

If the cast is wrong, subsequent field accesses will read garbage or corrupt memory.

### Pattern: Type Switch

```rust
fn process_entity(e: *Entity) {
    let id: u8 = e.type_id();

    if id == Player::TYPE_ID {
        handle_player(e as *Player);
    } else if id == Enemy::TYPE_ID {
        handle_enemy(e as *Enemy);
    } else if id == Projectile::TYPE_ID {
        handle_projectile(e as *Projectile);
    }
}
```

### Code Generation

```asm
; obj.type_id() - just loads the TypeId byte
    LDA (obj)           ; TypeId is at offset 0

; Comparison: obj.type_id() == Player::TYPE_ID
    LDA (obj)
    CMP #Player__TYPE_ID
    BEQ _is_player
```

---

## Trait Pointers

### Near Trait Pointer

```rust
*Drawable           // 2-byte pointer to any Drawable
```

### Far Trait Pointer

```rust
far *Renderable     // 3-byte pointer to any Renderable
```

### Creating Trait Pointers

```rust
let player = Player { x: 10, y: 20, health: 100 };

// Cast concrete type to trait pointer
let drawable: *Drawable = &player as *Drawable;

// Direct assignment (type inferred from context)
let d: *Drawable = &player;
```

### Null Trait Pointers

```rust
// Null check
if ptr != 0 as *Drawable {
    ptr.draw(X, Y);
}

// Initialize to null
let target: *Damageable = 0 as *Damageable;
```

### Trait Pointers in Function Signatures

```rust
// Parameter
fn render(obj: *Drawable, x @ X: u16, y @ Y: u16) {
    obj.draw(X, Y);
}

// Return type
fn find_enemy() -> *Drawable {
    return &ENEMIES[0] as *Drawable;
}

// Far trait parameter
fn render_cross_bank(obj: far *Renderable) {
    obj.render();
}
```

### Trait Pointers in Data Structures

```rust
// Array of trait pointers
#[ram]
static mut ENTITIES: [*Drawable; 32];

// Struct containing trait pointer
struct Projectile {
    x: u8,
    y: u8,
    target: *Damageable    // 2-byte near trait pointer
}

struct CrossBankRef {
    renderer: far *Renderable  // 3-byte far trait pointer
}
```

### Static Initialization of Trait Pointers

Trait pointers can be initialized at compile time in static declarations:

```rust
#[ram]
static mut PLAYER: Player;

#[ram]
static mut ENEMY: Enemy;

// Initialize trait pointer to concrete instance
#[ram]
static mut CURRENT_TARGET: *Drawable = &PLAYER;

// Array with compile-time initialization
#[ram]
static mut DRAW_LIST: [*Drawable; 4] = [
    &PLAYER,
    &ENEMY,
    0 as *Drawable,    // Null
    0 as *Drawable
];
```

**Rules**:
- Target must be a `static` or `static mut` variable (not local)
- Target struct must implement the trait
- The `&` operator on a static yields a compile-time address
- Type coercion from `*ConcreteType` to `*Trait` happens implicitly

**Struct fields with trait pointer initializers**:

```rust
struct GameState {
    active_entity: *Drawable,
    collision_target: *Damageable
}

#[ram]
static mut STATE: GameState = GameState {
    active_entity: &PLAYER,
    collision_target: &ENEMY
};
```

**Far trait pointers**:

```rust
#[bank(1)]
#[ram]
static mut SPRITE: Sprite;

#[ram]
static mut RENDERER: far *Renderable = &SPRITE;
```

**Code generation**:

Static initialization generates ROM data that's copied to RAM during `__init_start()`:

```asm
; Initial values in ROM
CURRENT_TARGET_init:
    .dw PLAYER              ; Address of PLAYER

DRAW_LIST_init:
    .dw PLAYER              ; [0]
    .dw ENEMY               ; [1]
    .dw 0                   ; [2] null
    .dw 0                   ; [3] null
```

---

## Method Dispatch

### Calling Methods

```rust
let obj: *Drawable = &player as *Drawable;

// Method call - dispatches via jump table
obj.draw(X, Y);
let width: u8 = obj.get_width();
```

### Dispatch Mechanism

1. Load TypeId from offset 0 of the object
2. Use TypeId to index into trait's jump table
3. Jump to the correct implementation

### Dispatch Cost

| Trait Type | Dispatch Overhead |
|------------|-------------------|
| Near trait | ~10-12 cycles |
| Near trait, chained mid/end (Y-preserving impls) | ~3-5 cycles less than solo |
| Far trait | ~20-25 cycles |
| Far trait, chained mid/end (DBR coalesced only) | ~10 cycles less than solo |
| Far trait, chained mid/end (DBR + Y elision) | ~13-15 cycles less than solo |

Compare to direct call:
- Near function call: ~12 cycles (JSR/RTS)
- Far function call: ~18 cycles (JSL/RTL)

When a function makes back-to-back trait calls on the same self, the
compiler coalesces the redundant per-call setup. There are two
independent optimizations driven by independent predicates (see
[Register/Memory Configuration § 1.4](register_memory_config.md) for
the soundness rules):

- **DBR-bracket coalescing** (far self only): the
  `PHB / load bank / PHA / PLB ... PLB` bracket is emitted once at
  the chain start instead of per call. Saves ~10 cycles per chained
  far call.
- **Y-reload elision** (near and far): the LDY / TAY / TXY that
  reloads the self address is emitted once at the chain start
  instead of per call, when every impl in the trait method jump
  tables and every gap instruction provably preserves Y at exit.
  Saves ~3-5 cycles per chained call.

For far chains both fire when their predicates pass, totalling
~13-15 cycles per chained call. The chain pass can additionally
extend across simple `if`/`else` CFG diamonds when both arms pass
the chain predicate.

### Self Pointer Access

Trait methods always receive `*self` in `Y`. For `far *self`, the compiler picks per-method between a fast **DBR:Y** path (leaf methods only) and a **D=S** fallback (when the body has any call, ROM access, or HW access). See [Register/Memory Configuration](register_memory_config.md) for the analysis rules, prologue shapes, and addressing modes.

---

## Code Generation

### Jump Table (Near Traits)

For each near trait, the compiler generates a jump table:

```asm
; Jump table for Drawable::draw
Drawable__draw_table:
    .dw _trait_error        ; TypeId 0 (invalid)
    .dw Player__draw        ; TypeId 1
    .dw Enemy__draw         ; TypeId 2
    .dw Bullet__draw        ; TypeId 3
```

### Dispatch Code (Near)

```asm
; obj.draw(X, Y) where obj is *Drawable
    LDA (obj)               ; Load TypeId from offset 0
    ASL A                   ; ×2 for 16-bit table entries
    TAX
    JMP (Drawable__draw_table,X)
```

### Trampoline (Far Traits)

For far traits, the compiler generates a JML trampoline:

```asm
; Trampoline for Renderable::render
Renderable__render_trampoline:
    JML _trait_error        ; TypeId 0 (4 bytes)
    JML Sprite__render      ; TypeId 1 (4 bytes)
    JML Enemy__render       ; TypeId 2 (4 bytes)
```

### Dispatch Code (Far)

```asm
; obj.render() where obj is far *Renderable
    LDA [obj]               ; Load TypeId (24-bit pointer)
    ASL A
    ASL A                   ; ×4 for JML instruction size
    CLC
    ADC #<Renderable__render_trampoline
    STA _jmp_addr
    ; ... set up 24-bit address ...
    JSL [_jmp_addr]
```

### TypeId Initialization

When creating a struct instance that has trait impls:

```asm
; Player { x: 10, y: 20, health: 100 }
    LDA #1                  ; Player's TypeId
    STA player              ; Store at offset 0
    LDA #10
    STA player+1            ; x at offset 1
    LDA #20
    STA player+2            ; y at offset 2
    LDA #100
    STA player+3            ; health at offset 3
```

---

## Usage Patterns

### Heterogeneous Collections

```rust
#[ram]
static mut ENTITIES: [*Entity; 32];

fn update_all() {
    for i in 0..32 {
        let e: *Entity = ENTITIES[i];
        if e != 0 as *Entity {
            e.update();
        }
    }
}

fn add_entity(e: *Entity) {
    for i in 0..32 {
        if ENTITIES[i] == 0 as *Entity {
            ENTITIES[i] = e;
            return;
        }
    }
}
```

### Factory Pattern

```rust
fn spawn_enemy(enemy_type: u8, x: u8, y: u8) -> *Drawable {
    if enemy_type == 0 {
        GOOMBA.x = x;
        GOOMBA.y = y;
        return &GOOMBA as *Drawable;
    } else {
        KOOPA.x = x;
        KOOPA.y = y;
        return &KOOPA as *Drawable;
    }
}
```

### Callback System

```rust
trait EventHandler {
    fn on_collision(*self, other @ A: u8);
    fn on_destroy(*self);
}

#[ram]
static mut COLLISION_HANDLER: *EventHandler;

fn trigger_collision(id @ A: u8) {
    if COLLISION_HANDLER != 0 as *EventHandler {
        COLLISION_HANDLER.on_collision(A);
    }
}
```

### Component System

```rust
trait Component {
    fn update(*self);
    fn draw(*self, x @ X: u16, y @ Y: u16);
}

struct GameObject {
    x: u8,
    y: u8,
    component: *Component
}

fn update_object(obj: *GameObject) {
    if obj.component != 0 as *Component {
        obj.component.update();
    }
}
```

---

## Memory Considerations

### Table Size

| Factor | Size |
|--------|------|
| Near table entry | 2 bytes per TypeId |
| Far trampoline entry | 4 bytes per TypeId |
| TypeId per instance | 1 byte |

**Example**: 10 types implementing `Drawable` with 3 methods:
- 3 jump tables × 11 entries × 2 bytes = 66 bytes

### Bank Placement

- Jump tables must be accessible from calling code
- Near traits: tables in same bank as dispatch code
- Far traits: trampolines typically in bank 0 or a shared bank

### TypeId Limits

- Maximum 255 distinct struct types with trait impls (TypeId is u8)
- TypeId 0 reserved for null/invalid

---

## Comparison with Alternatives

### vs. Manual Function Pointers

```rust
// Manual approach
struct Entity {
    x: u8,
    y: u8,
    update_fn: fn(*Entity)
}

// Trait approach
trait Updatable { fn update(*self); }
struct Entity { x: u8, y: u8 }
impl Updatable for Entity { /* ... */ }
```

| Aspect | Function Pointer | Trait |
|--------|------------------|-------|
| Per-instance storage | 2 bytes per method | 1 byte (TypeId) |
| Dispatch cost | ~8 cycles | ~10-12 cycles |
| Type safety | None | Compile-time verified |
| Multiple methods | N × 2 bytes | 1 byte total |

### vs. Tagged Union

```rust
// Tagged union approach
enum EntityType { Player = 0, Enemy = 1 }
struct Entity { entity_type: EntityType, x: u8, y: u8 }

// Trait approach - similar but with compiler support
```

Traits provide:
- Automatic dispatch generation
- Type-safe method signatures
- Extensibility without modifying switch statements

---

## Limitations

1. **No generics** - Cannot parameterize traits with types
2. **No associated types** - Cannot define type aliases in traits (associated constants are supported)
3. **No trait bounds** - Cannot require traits in function signatures
4. **Near/far exclusivity** - Struct cannot mix near and far traits
5. **No `self` by value** - Methods must take `*self` pointer

Supertraits (trait inheritance) **are** supported — see [Trait Inheritance](#trait-inheritance-supertraits).
Default method bodies **are** supported — see [Default Method Bodies](#default-method-bodies).

---

## Error Messages

### Signature Mismatch

```rust
trait Drawable {
    fn draw(*self, x @ X: u16, y @ Y: u16);
}

impl Drawable for Player {
    fn draw(*self, x @ X: u8, y @ Y: u8) { }  // ERROR
}
```

```
error: method signature does not match trait
  --> game.r65:15:5
   |
15 |     fn draw(*self, x @ X: u8, y @ Y: u8) { }
   |        ^^^^ expected `x @ X: u16`, found `x @ X: u8`
```

### Missing Method

```rust
trait Drawable {
    fn draw(*self, x @ X: u16, y @ Y: u16);
    fn get_width(*self) -> u8;
}

impl Drawable for Player {
    fn draw(*self, x @ X: u16, y @ Y: u16) { }
    // Missing get_width
}
```

```
error: not all trait methods implemented
  --> game.r65:10:1
   |
10 | impl Drawable for Player {
   | ^^^^^^^^^^^^^^^^^^^^^^^^ missing `get_width`
```

### Missing Associated Constant

```rust
trait Drawable {
    const WIDTH: u8;
    const HEIGHT: u8;
    fn draw(*self, x @ X: u16, y @ Y: u16);
}

impl Drawable for Player {
    const WIDTH: u8 = 16;
    // Missing HEIGHT
    fn draw(*self, x @ X: u16, y @ Y: u16) { }
}
```

```
error: not all trait items implemented
  --> game.r65:10:1
   |
10 | impl Drawable for Player {
   | ^^^^^^^^^^^^^^^^^^^^^^^^ missing associated constant `HEIGHT`
```

### Near/Far Conflict

```rust
trait NearTrait { fn method(*self); }
trait FarTrait { far fn method(*self); }

impl NearTrait for MyStruct { /* ... */ }
impl FarTrait for MyStruct { /* ... */ }  // ERROR
```

```
error: struct cannot implement both near and far traits
  --> game.r65:20:1
   |
20 | impl FarTrait for MyStruct {
   | ^^^^^^^^^^^^^^^^^^^^^^^^^^ `MyStruct` already implements near trait `NearTrait`
```

---

## Complete Example

```rust
// Define traits
trait Drawable {
    fn draw(*self, x @ X: u16, y @ Y: u16);
}

trait Updatable {
    fn update(*self);
}

// Define structs
struct Player {
    x: u8,
    y: u8,
    sprite_id: u8
}

struct Enemy {
    x: u8,
    y: u8,
    health: u8
}

// Implement traits
impl Drawable for Player {
    fn draw(*self, x @ X: u16, y @ Y: u16) {
        // Draw player sprite
        draw_sprite(self.sprite_id, X, Y);
    }
}

impl Updatable for Player {
    fn update(*self) {
        // Handle input, update position
    }
}

impl Drawable for Enemy {
    fn draw(*self, x @ X: u16, y @ Y: u16) {
        // Draw enemy sprite
        draw_sprite(0x10, X, Y);
    }
}

impl Updatable for Enemy {
    fn update(*self) {
        // AI logic
        if self.health == 0 {
            self.x = 0xFF;  // Mark as dead
        }
    }
}

// Game state
#[ram]
static mut PLAYER: Player;

#[ram]
static mut ENEMIES: [Enemy; 8];

#[ram]
static mut DRAW_LIST: [*Drawable; 16];

// Main loop
fn game_update() {
    // Update all entities
    PLAYER.update();

    for i in 0..8 {
        ENEMIES[i].update();
    }

    // Draw all entities via trait dispatch
    for i in 0..16 {
        let d: *Drawable = DRAW_LIST[i];
        if d != 0 as *Drawable {
            d.draw(X, Y);
        }
    }
}
```

---

**STATUS**: Design Complete
**Last Updated**: 2026-01-22
