# Packaging System

How R65 code is shared between projects: a manifest (`r65.toml`), a lockfile
(`r65.lock`), and the `r65pkg` tool. Designed so that a single git repository can
hold several independently versioned packages.

*Status: design. Not yet implemented — see [Implementation](#implementation).*

---

## Motivation

R65 has no way to share code. `include!()` has exactly one resolution rule — try
the including file's directory, then each `-I` directory
(`r65/compiler/frontend/preprocessor.py::_resolve_include_path`) — and nothing
above it. There is no manifest, no version, and no dependency graph.

`stdlib/` is not even reachable from a project. Every project Makefile passes
exactly `-I src`, and nothing points at the repository's `stdlib/`. So
distribution happens by copying: `r65x init` physically copies 3 of the 14 stdlib
files into `<project>/src/lib/` (`r65/cli.py::_copy_stdlib`), and nothing ever
re-syncs them.

The result, measured across the projects in this repository:

- **four distinct `sneslib.r65` files** across six projects (pacman; astroids and
  bubblebobble; stdlib and rotor_raiders; textadv; kungfu) — 53 to 169 diff lines
  apart;
- `math.r65` stale in five of six projects;
- `rotor_raiders/src/lib/rand.r65` is a *forward* fork — it carries fixes stdlib
  does not, with no path back;
- the 11 stdlib files `init` does not copy (`console`, `string`, `Q10`, `F32`,
  `I32`, `U32`, …) were hand-copied in by users;
- 13 e2e test files reach stdlib by interpolating an absolute path into an
  `include!` string.

This is not drift in one direction that a sync script would fix; it is
bidirectional divergence. The goal is that a project *declares* what it depends
on instead of copying it.

## Scope

This is a **build-time system only**.

- **No language change.** No `mod`, `use`, or `pub` (all reserved and unused, but
  not spent here). No path syntax — `::` remains the two-deep type-qualified
  member selector it is today.
- **No symbol mangling.** The global namespace stays flat and names still reach
  WLA-DX verbatim. Duplicate symbols remain an error.
- **Source inclusion, not separate compilation.** A package resolves to
  directories on the include path; the build stays whole-program, so the existing
  dead-function elimination still prunes what a package doesn't use.

Packages are therefore a way to answer *"where does this file come from, and
which version is it"* — not a new unit of compilation.

---

## The manifest — `r65.toml`

A package is a directory containing an `r65.toml`. It is parsed with `tomllib`
and discovered by walking up from the root source file, the same way
`r65-lint.toml` is found today (`r65/compiler/lint/config.py::discover_config`).

A library:

```toml
# stdlib/r65.toml
[package]
name    = "stdlib"
version = "0.4.0"

[lib]
include_dir = "."          # dir added to the include path for dependents
```

An application:

```toml
# pacman/r65.toml
[package]
name    = "pacman"
version = "0.1.0"

[bin]
main = "src/main.r65"

[dependencies]
stdlib = { git = "https://github.com/BubbleZapGames/R65", tag = "stdlib-v0.4.0", package = "stdlib" }
q10    = { path = "../q10" }
```

The workspace root:

```toml
# /home/nathan/R65/r65.toml
[workspace]
members = ["stdlib", "pacman", "textadv", "rotor_raiders", "astroids", "bubblebobble", "kungfu.r65"]
```

### One repo, many versioned packages

This is the case the design exists for. Each workspace member carries its own
`[package] version`, and releases are tagged **per package** — `stdlib-v0.4.0`,
`q10-v0.2.0` — so cadences are independent.

A git dependency names three things:

| Key | Selects |
|---|---|
| `git` | the repository |
| `tag` (or `rev`, `branch`) | the commit |
| `package` | the member *within* that repository |

The resolver clones the repository, reads its root `[workspace]`, and picks the
member whose `[package] name` matches `package =`. Omit `package` and the
repository root itself must be a package.

### Workspace member override

When a git dependency names a package that is a member of the *current*
workspace, it resolves to the local path instead of a clone. Editing `stdlib/` is
immediately visible to `pacman/` inside this repository, while a consumer outside
gets the tagged release. Without this, monorepo development would mean committing
and tagging to test a one-line change.

---

## Resolution

### One version per package, always

Textual inclusion into a flat namespace means two versions of a package cannot
coexist — their symbols would collide at `SymbolTable.declare`. So, unlike Cargo,
the resolver **unifies each package name to exactly one version**. Incompatible
requirements are a hard error naming both requirers and both requirements; they
are never satisfied by instantiating the package twice.

This is a real restriction, and it is also what makes the rest work. Diamond
dependencies resolve to a single directory, so the preprocessor's existing
dedup-by-resolved-path makes a repeated `include!` a no-op with no extra
machinery.

### Include path synthesis

Resolution produces an ordered list of package include directories, appended to
`include_paths` *after* the application's own `-I` directories, ordered
deterministically (topological, then alphabetical). `include!("sneslib.r65")`
then simply works, and existing sources compile unchanged.

### Ambiguity is an error, not first-wins

Because dependencies are addressed by bare filename, two packages could each
provide `math.r65`. `_resolve_include_path` returns the first hit today; it must
instead collect every hit across *package* roots and fail when two distinct files
match:

```
error: include!: 'math.r65' is provided by two packages
  stdlib 0.4.0  -> .r65/deps/stdlib/math.r65
  mathx  0.1.0  -> .r65/deps/mathx/math.r65
  note: rename one, or shadow it with a local copy under -I
```

The including file's own directory and explicit `-I` directories still win
silently and first. That ordering is deliberate: it is how a project overrides a
single file from a package without forking it.

### Fetching — `.r65/deps/`

`r65pkg fetch` clones each git dependency into `<project>/.r65/deps/<name>/` at
the locked commit: a shallow clone at the tag, then `rev-parse` to pin. No shared
cache, no committed vendor tree; `.r65/` is gitignored.

`r65.lock` sits next to `r65.toml` and **is** committed:

```toml
[[package]]
name    = "stdlib"
version = "0.4.0"
source  = "git+https://github.com/BubbleZapGames/R65?tag=stdlib-v0.4.0#package=stdlib"
rev     = "0bf2f0f4a1..."
```

Path dependencies and workspace members are recorded without a rev. `r65c` fails
with an actionable message when the lockfile is stale or a dependency is missing,
rather than fetching implicitly — the compiler never touches the network.

---

## The `r65pkg` tool

A console script alongside `r65c`, `r65x` and `r65-emu`, and the **only** thing
that writes manifests, writes the lockfile, or runs `git`.

| Command | Does |
|---|---|
| `r65pkg init [--lib\|--bin] [NAME]` | Write an `r65.toml` in the current directory |
| `r65pkg add NAME --path ../q10` | Add a path dependency |
| `r65pkg add NAME --git URL --tag T [--package P]` | Add a git dependency, resolve it, update the lock |
| `r65pkg remove NAME` | Drop a dependency and its lock entry |
| `r65pkg fetch` | Materialize everything in `r65.lock` into `.r65/deps/` |
| `r65pkg update [NAME]` | Re-resolve (all dependencies, or one) and rewrite the lock |
| `r65pkg tree` | Print the dependency graph with versions and sources |
| `r65pkg list` | Resolved packages: version, source, include dir |
| `r65pkg check` | Validate manifests, lock freshness, version conflicts and cross-package include ambiguity — without compiling |
| `r65pkg path [--include-flags]` | Print resolved include dirs, or `-I a -I b`, for Makefiles |
| `r65pkg tag NAME` | Create the `NAME-v<version>` git tag from that member's manifest version |

`r65pkg path --include-flags` is what keeps Makefiles honest — a project appends
`$(shell r65pkg path --include-flags)` to both `R65C_FLAGS` and `WLA_FLAGS`
instead of hardcoding paths. `r65pkg tag` is the monorepo release helper: it
reads the member's `[package] version`, so a tag can never disagree with the
manifest it claims to release.

### Module layout

Split so the compiler links only against the read-only half:

```
r65/pkg/manifest.py   # read-only: Manifest, Dependency, load/discover
r65/pkg/resolve.py    # read-only: workspace, graph walk, unification
r65/pkg/lockfile.py   # read + write
r65/pkg/fetch.py      # git subprocess — CLI only
r65/pkg/cli.py        # r65pkg entry point
```

`r65c` imports `manifest` and `resolve` and nothing else, so it structurally
cannot shell out to git or mutate a manifest.

---

## Correctness fixes packaging depends on

These are not polish. Each is a latent silent-miscompile today that packaging
makes *reachable*, so each must land with the system.

### Bank state leaks across include boundaries

`#[bank(n)]` is a standalone directive that mutates a positional cursor
(`r65/compiler/hir/builder.py`, both declaration passes), not an attribute.
Because `include!` splices items into one flat list, a package containing a bank
directive **silently rebanks every declaration after its include site**, and a
package's own contents inherit whatever bank the includer happened to be in.

Fix in `Preprocessor._process_include`: when an included file emits any
`BankDirective`, append a synthesized `BankDirective` restoring the includer's
bank after the spliced items. Contained, and directly testable against the
emitted `.BANK` directives.

### Collision diagnostics carry no location

`SymbolTable.declare` raises `Redefinition of 'foo' in current scope` with
`source_loc=None` — no file, no line, no pointer at the original definition. With
packages the number of reachable collisions multiplies, so the error must name
both sites and both packages.

`SourceLocation` already chains `included_from` (`r65/compiler/errors.py`); add
an `origin_package` alongside it, set in the preprocessor, and pass
`symbol.definition.source_loc` into the error.

### Macro redefinition is silently last-wins

`MacroExpander` collects macros with a bare dict assignment. Two packages
defining `debug!` pick one with no diagnostic at all. Make cross-file macro
redefinition an error, using the same provenance.

---

## Known hazards, not addressed here

Recorded so they are not mistaken for oversights. Each is pre-existing, and each
gets worse as package count grows; see `docs/future-work.md`.

- **Statics are never dead-code eliminated.** Only functions are
  (`r65/compiler/optimize/dead_function_elim.py`). An unused package static still
  burns zeropage, RAM or ROM. "Pay for what you use" holds for code, not data.
- **Explicit addresses are not collision-checked.** Two packages both hardcoding
  `#[zeropage(0x40)]` overlap silently
  (`r65/compiler/codegen/memory_alloc.py`). Packages should use auto-allocation;
  the `zeropage_budget` lint is the only backstop.
- **`#[soa]` and tagged-enum types are program-wide singletons.** Each backs
  exactly one static in the whole program, so a package cannot ship one for
  general use.
- **`*dyn` TypeIds are a global counter** assigned in declaration order, so
  adding a package shifts the TypeIds of everything declared after it.

---

## Implementation

Small verified slices; each lands green on its own.

**0 — This document.** Written before the code, and kept current as the slices
land rather than describing an intent the code outgrew. Indexed in `CLAUDE.md`
and `README.md`; hazards recorded in `docs/future-work.md`.

**1 — Manifest and path dependencies (no network).**
`r65/pkg/manifest.py` (`Manifest`, `Dependency`, `load_manifest`,
`discover_manifest`, `PackageError`; unknown keys rejected) and
`r65/pkg/resolve.py` (workspace discovery, graph walk, single-version
unification, ordered include dirs). `r65/compiler/main.py` discovers and resolves
before `compile_source`, with `--no-manifest` and `--manifest PATH`.
`_resolve_include_path` gains the package-root list and the ambiguity error.

**2 — Provenance and diagnostics.** `origin_package` on `SourceLocation`, set in
`_process_include`; used by `SymbolTable.declare` and by the macro collector.

**3 — Bank scoping.** The `_process_include` restore, plus a test asserting that
declarations after an include site keep their original bank.

**4 — `r65pkg`, git dependencies and the lockfile.** `r65/pkg/lockfile.py`,
`r65/pkg/fetch.py`, `r65/pkg/cli.py`; the `r65pkg` entry point in `setup.py`;
`--offline`; `.r65/` in the template `.gitignore`.

**5 — Migration.** `stdlib/r65.toml` and the workspace root first. Then per
project: add `r65.toml` depending on stdlib by path, delete `src/lib/`.
**rotor_raiders last** — its `rand.r65` is a forward fork whose changes must be
merged *into* stdlib first, or they are lost. `r65x init` stops copying stdlib
and writes a manifest instead; the Makefile template gets a `deps:` target,
include dirs from `r65pkg path --include-flags`, and a dependency on the full
include graph rather than just `$(MAIN_SRC)`.

## Verification

- `pytest r65/tests/pkg/` — manifest parse, workspace discovery, version
  unification and its conflict error, include ambiguity, lockfile round-trip, and
  `r65pkg` subcommands driven against a throwaway local git repository holding
  two independently tagged members. That last one is the direct test that one
  repo can serve two versioned packages.
- **Byte-identical proof for slice 1.** For an unmigrated project, compile once
  through the manifest and once with the equivalent hand-written `-I`, and assert
  the two `.asm` outputs are identical. This proves the resolver changes only
  *how* include directories are computed, not what is compiled.
- **Migration diff, per project.** Build the ROM before migrating, migrate,
  rebuild, diff `build/main.asm`. Every hunk is stdlib drift being adopted — read
  it and confirm it is wanted before accepting. pacman is the first candidate,
  and its `play.sh` scripted-screenshot harness gives a runtime check on top of
  the diff.
- `pytest r65/tests/e2e/` stays green throughout: the 13 files interpolating an
  absolute `STDLIB_DIR` keep working because relative and `-I` resolution are
  unchanged.
- End to end: `r65x init /tmp/pkgtest && cd /tmp/pkgtest && make`
  succeeds with no `src/lib/`, and `r65pkg tree` / `r65pkg check` report clean.
