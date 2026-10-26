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
name        = "stdlib"
version     = "0.4.0"
r65-version = "0.2"        # the oldest r65c that builds it

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
cfg  = ["snes"]            # the build's --cfg conditions

[dependencies]
stdlib = { git = "https://github.com/BubbleZapGames/R65", version = "0.4" }
q10    = { path = "../q10" }
```

The workspace root:

```toml
# /home/nathan/R65/r65.toml
[workspace]
members = ["stdlib", "pacman", "textadv", "rotor_raiders", "astroids", "bubblebobble", "kungfu.r65"]

[patch]
stdlib = { path = "stdlib" }
```

### One repo, many versioned packages

This is the case the design exists for. Each workspace member carries its own
`[package] version`, and releases are tagged **per package** — `stdlib-v0.4.0`,
`q10-v0.2.0` — so cadences are independent.

A git dependency names two things:

| Key | Selects |
|---|---|
| `git` | the repository |
| `version` (or `tag`, `rev`, `branch`) | the release, or an exact commit |

The member is found by the dependency's own key: the resolver clones the
repository, reads its root `[workspace]`, and picks the member whose
`[package] name` is that key. With no `[workspace]`, the repository root itself
must be the package, under that name.

There is no `package =` key. In Cargo it renames a dependency, and R65 has no
renames: a package's name is its identity in the lockfile and in qualified
includes, so a dependency is always known by its own name.

### Version requirements

`version` is a requirement, not a pin, with Cargo's caret meaning: `"0.4"` is
`>= 0.4.0, < 0.5.0`, `"1.2"` is `>= 1.2.0, < 2.0.0`, and `"=0.4.1"` is exact.
The candidates are the repository's `NAME-vX.Y.Z` tags. `tag`, `rev` and
`branch` remain for an exact commit; the version such a dependency carries is
read from the member's manifest at that commit.

A tag is trusted only if it agrees with the manifest it points at: the resolver
reads `[package] version` at `stdlib-v0.4.0` and refuses the tag if it is not
`0.4.0`. Git tags can be moved and the lockfile pins the rev, but a tag that
disagrees with its own manifest is refused outright.

### Build requirements

A package states what it needs from the build, and the resolver checks it
against the application before anything is compiled:

```toml
[package]
name        = "fastmath"
version     = "0.1.0"
r65-version = "0.3"              # minimum r65c version
cfg         = ["snes"]           # cfg conditions it must be built with
```

- **`r65-version`** is a minimum, like Cargo's `rust-version`. R65 removes
  syntax without migration paths, so a package pinned at an old tag can stop
  building on a newer compiler, and one written for a newer compiler fails on an
  older one with an unrelated parse error. The check turns the second into
  `error: fastmath 0.1.0 requires r65c >= 0.3; this is r65c 0.2.0`. It only
  works if the compiler's own version (`setup.py`) is bumped when the language
  changes.
- **`cfg`** lists the conditions the package requires. A package that merely
  *offers* more under a cfg (`#[cfg(snes)]` in `math.r65`) does not list it.

The application's side is `[bin] cfg`. `r65c` uses it when `--cfg` is absent
and fails when a flag contradicts it, so the Makefile and the manifest cannot
drift.

The ABI is deliberately absent. It is a compile-time choice — `--abi` on the
`r65c` command line — and packaging has no say in it.

### Overriding a package — `[patch]`

`[patch]` in the root manifest — the workspace root, or the application when it
has none — replaces a package by name for the whole graph, as Cargo's does:

```toml
[patch]
stdlib = { path = "stdlib" }
```

This is how monorepo development works. Inside this repository, `pacman`'s
`stdlib = { git = ..., version = "0.4" }` resolves to `stdlib/`, so an edit there
is immediately visible to `pacman/`; a consumer outside, with no patch, gets the
tagged release. Without it, testing a one-line stdlib change would mean
committing and tagging.

The override is explicit rather than inferred from workspace membership, so a
member that wants the released stdlib simply isn't patched. A patched package
must still satisfy every requirement on it: bump `stdlib/` to 0.5.0 and
`pacman`'s `"0.4"` becomes an error naming both, rather than the patch being
quietly ignored as Cargo's is. `[patch]` in a dependency's manifest is ignored,
as Cargo ignores it — only the root decides.

---

## Resolution

### One version per package, always

Textual inclusion into a flat namespace means two versions of a package cannot
coexist — their symbols would collide at `SymbolTable.declare`. So, unlike Cargo,
the resolver **unifies each package name to exactly one version**: the highest
tagged release that satisfies every requirement on it. `stdlib "0.4"` from one
package and `stdlib "0.4.2"` from another unify to the newest 0.4.x at or above
0.4.2. An exact `tag`/`rev` unifies with a range when its version satisfies the
range; two different exact commits never unify. When nothing satisfies every
requirement, it is a hard error naming both requirers and both requirements —
the root resolves it with `[patch]` or by raising a requirement — and it is
never satisfied by instantiating the package twice.

This is a real restriction, and it is also what makes the rest work. Diamond
dependencies resolve to a single directory, so the preprocessor's existing
dedup-by-resolved-path makes a repeated `include!` a no-op with no extra
machinery.

### Include path synthesis

Resolution produces an ordered list of package include directories, appended to
`include_paths` *after* the application's own `-I` directories, ordered
deterministically (topological, then alphabetical). `include!("sneslib.r65")`
then simply works, and existing sources compile unchanged.

The same directory can arrive twice — from the manifest, and from
`r65pkg path --include-flags` in the Makefile — so the list is deduplicated by
resolved path before anything below looks at it.

### Package-qualified includes

`include!("stdlib/math.r65")` names the package as well as the file: when the
first path component is the name of a resolved package, the rest resolves
against that package's `include_dir`. This is the include-string counterpart of
Cargo's `stdlib::math` — a resolver rule, not a language change, since the
argument was always a path.

A qualified include cannot be ambiguous, so it is the form a **package** uses
for another package's files; `r65pkg check` warns when a package reaches a
dependency's file by bare name. Bare names stay for applications and existing
sources. Both forms resolve to the same file, so the preprocessor's
dedup-by-resolved-path still makes one of them a no-op when both appear.

Search order, first match wins except where noted:

1. the including file's directory;
2. each explicit `-I` directory;
3. a package-qualified path;
4. a bare name across package roots — an error if two packages match.

Local resolution stays first, so a project can still shadow one file of a
package, by either spelling, without forking it.

### Ambiguity is an error, not first-wins

Because a bare include names only a file, two packages could each provide
`math.r65`. `_resolve_include_path` returns the first hit today; it must instead
collect every hit across *package* roots and fail when two distinct files match:

```
error: include!: 'math.r65' is provided by two packages
  stdlib 0.4.0  -> .r65/deps/stdlib/math.r65
  mathx  0.1.0  -> .r65/deps/mathx/math.r65
  note: include it qualified ('stdlib/math.r65'), or shadow it with a local copy under -I
```

### Fetching — `.r65/deps/`

`r65pkg` lists a repository's `NAME-v*` tags with `git ls-remote --tags` to
resolve version requirements, then `r65pkg fetch` clones each git dependency
into `<project>/.r65/deps/<name>/` at the locked commit: a shallow clone at the
tag, then `rev-parse` to pin. No shared cache, no committed vendor tree; `.r65/`
is gitignored.

`r65.lock` sits next to `r65.toml` and **is** committed:

```toml
[[package]]
name    = "stdlib"
version = "0.4.2"
source  = "git+https://github.com/BubbleZapGames/R65?tag=stdlib-v0.4.2"
rev     = "0bf2f0f4a1..."
```

The lock records the version chosen, not the requirement, so `r65pkg update`
is the only thing that moves it to a newer 0.4.x. Path dependencies and patched
packages are recorded without a rev. Only the root's lockfile counts: a
dependency's own `r65.lock` is ignored, as Cargo ignores it. `r65c` fails with an
actionable message when the lockfile is stale or a dependency is missing, rather
than fetching implicitly — the compiler never touches the network.

---

## The `r65pkg` tool

A console script alongside `r65c`, `r65x` and `r65-emu`, and the **only** thing
that writes manifests, writes the lockfile, or runs `git`.

| Command | Does |
|---|---|
| `r65pkg init [--lib\|--bin] [NAME]` | Write an `r65.toml` in the current directory |
| `r65pkg add NAME --path ../q10` | Add a path dependency |
| `r65pkg add NAME --git URL [--version REQ \| --tag T \| --rev R \| --branch B]` | Add a git dependency (default: a caret requirement on the newest tag), resolve it, update the lock |
| `r65pkg remove NAME` | Drop a dependency and its lock entry |
| `r65pkg fetch` | Materialize everything in `r65.lock` into `.r65/deps/` |
| `r65pkg update [NAME]` | Re-resolve (all dependencies, or one) to the newest versions the requirements allow, and rewrite the lock |
| `r65pkg tree` | Print the dependency graph with versions and sources |
| `r65pkg list` | Resolved packages: version, source, include dir |
| `r65pkg check` | Validate manifests, lock freshness, version conflicts, build requirements (`r65-version`, `cfg`), tag/manifest agreement, cross-package include ambiguity, and bare-name includes of another package's files — without compiling |
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
before `compile_source`, with `--no-manifest` and `--manifest PATH`; it checks
the build requirements (`r65-version`, `cfg`) and takes `--cfg` from `[bin]`
when the flag is absent. `_resolve_include_path` gains the
package-root list, package-qualified paths and the ambiguity error.

**2 — Provenance and diagnostics.** `origin_package` on `SourceLocation`, set in
`_process_include`; used by `SymbolTable.declare` and by the macro collector.

**3 — Bank scoping.** The `_process_include` restore, plus a test asserting that
declarations after an include site keep their original bank.

**4 — `r65pkg`, git dependencies and the lockfile.** `r65/pkg/lockfile.py`,
`r65/pkg/fetch.py`, `r65/pkg/cli.py`; the `r65pkg` entry point in `setup.py`;
version requirements resolved against `NAME-v*` tags, the tag/manifest
agreement check, and `[patch]`; `--offline`; `.r65/` in the template
`.gitignore`.

**5 — Migration.** `stdlib/r65.toml` and the workspace root, with its
`[patch] stdlib = { path = "stdlib" }`, first. Then per project: add `r65.toml`
depending on stdlib by `git` and `version` — the patch resolves it to `stdlib/`
here — with `[bin] cfg` matching the Makefile, and delete `src/lib/`.
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
  repo can serve two versioned packages. Against the same repository: two
  caret requirements unifying to the newest tag both allow, an exact rev outside
  a range refused, a tag whose manifest version disagrees refused, and a
  `[patch]` that redirects a git dependency and fails when its version no longer
  satisfies the requirement.
- **Includes and build requirements.** A qualified include resolving into the
  named package while a bare one is ambiguous; a local file shadowing either
  spelling; a qualified and a bare include of one file compiling it once; and
  the `r65-version` and `cfg` errors, including a `--cfg` flag that
  contradicts `[bin]`.
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
