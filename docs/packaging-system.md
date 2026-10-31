# Packaging System

How R65 code is shared between projects: a manifest (`r65.toml`), a lockfile
(`r65.lock`), and the `r65pkg` tool. Designed so that a single git repository can
hold several independently versioned packages.

*Status: implemented (`r65/pkg/`). stdlib is a package, `r65x init` depends on
it, and five of this repository's projects are migrated — see
[Implementation](#implementation).*

---

## Motivation

Before packages, R65 had no way to share code. `include!()` had exactly one
resolution rule — try the including file's directory, then each `-I` directory
(`r65/compiler/frontend/preprocessor.py::_resolve_include_path`) — and nothing
above it. There was no manifest, no version, and no dependency graph.

`stdlib/` was not even reachable from a project. No project build pointed at the
repository's `stdlib/`: five Makefiles passed exactly `-I src`, classickong's
passed no `-I` at all, and astroids had no Makefile. So distribution happened by
copying: `r65x init` physically copied 3 of the 14 stdlib files (4 under
FixedStack) into `<project>/src/lib/`, and nothing ever re-synced them.

The result, measured across the projects in this repository before migration:

- **six distinct `sneslib.r65` files** across seven projects (astroids and
  bubblebobble; classickong; kungfu; pacman; textadv; rotor_raiders, the only
  one matching stdlib) — the five that differ are 4 to 175 diff lines from it;
- `math.r65` differs from stdlib in all seven, by 5 to 61 diff lines;
- the stdlib files `init` does not copy (`console`, `string`, `Q10`, `U32`, …)
  were hand-copied in by users and drift too: `string.r65` is 343 diff lines
  from stdlib in both rotor_raiders and textadv;
- 19 e2e test files reach stdlib by interpolating an absolute path into an
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
version     = "0.1.0"
r65-version = "0.2"        # the oldest r65c that builds it
repository  = "https://github.com/BubbleZapGames/R65"

[lib]
include_dir = "."          # dir added to the include path for dependents
```

`repository` is where the package is released from. `r65x init` reads stdlib's,
with its version, to write a new project's dependency: the release the installed
compiler ships, and anything compatible after it (see
[A new project's stdlib](#a-new-projects-stdlib)).

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
stdlib = { git = "https://github.com/BubbleZapGames/R65", version = "0.1" }
```

A path dependency is `q10 = { path = "../q10" }`, optionally with a `version`
requirement checked against that directory's manifest.

The workspace root:

```toml
# /home/nathan/R65/r65.toml
[workspace]
members = ["stdlib", "pacman", "astroids", "bubblebobble", "textadv", "kungfu.r65"]

[patch]
stdlib = { path = "stdlib" }
```

The example projects are not tracked in the R65 repository, so a fresh clone
has members that do not exist. A missing member is not an error: resolution only
asks whether a package's directory is listed, and `r65pkg check` warns.

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

The application's side is `[bin] cfg`: conditions every build of it is
compiled with. `--cfg` adds to them (a debug build's `--cfg debug`), and a flag
that gives a `key=value` condition a different value from `[bin]` is an error,
so the Makefile and the manifest cannot drift.

The ABI is chosen only by `--abi` on the `r65c` command line, so `[bin] cfg`
cannot set it. The compiler defines it as the cfg value `abi`, so a package
that works under one ABI alone lists `cfg = ["abi=FixedStack"]`; the error
then names `--abi FixedStack`.

### Overriding a package — `[patch]`

`[patch]` in the root manifest — the workspace root, or the application when it
has none — replaces a package by name for the whole graph, as Cargo's does. A
patch is a `path`:

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

The same directory can arrive twice — from `-I` and from a package — so the
list is deduplicated by resolved path before anything below looks at it.
`include_bytes!` searches the package roots after `-I` too.

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
`math.r65`. Rather than take the first hit, `_resolve_include_path` collects
every hit across *package* roots and fails when two distinct files match:

```
error: include!: 'math.r65' is provided by 2 packages
  stdlib 0.4.0     -> .r65/deps/stdlib/stdlib/math.r65
  mathx 0.1.0      -> .r65/deps/mathx/math.r65
  note: include it qualified ('stdlib/math.r65'), or shadow it with a local copy under -I
```

### Fetching — `.r65/deps/`

`r65pkg` lists a repository's `NAME-v*` tags with `git ls-remote --tags` to
resolve version requirements, then clones each git dependency into
`<project>/.r65/deps/<name>/` at the chosen commit: a shallow clone at the tag,
detached, then `rev-parse` to pin. A clone already at the locked commit is kept,
so a fetch with nothing new runs no network command. No shared cache, no
committed vendor tree; `.r65/` is gitignored.

`r65.lock` sits next to the application's `r65.toml` and **is** committed:

```toml
version = 1

[[package]]
name    = "stdlib"
version = "0.4.2"
source  = "git+https://github.com/BubbleZapGames/R65?tag=stdlib-v0.4.2"
rev     = "0bf2f0f4a1..."
```

### A new project's stdlib

`r65x init` does not clone. It copies the stdlib installed with r65 into
`.r65/deps/stdlib/` and locks it as the GitHub release it is — `source =
"git+https://github.com/BubbleZapGames/R65?tag=stdlib-v0.1.0"`, with no `rev`.
The first build needs no network, and the project still points at the
repository:

- `r65c` and `r65pkg fetch` take a rev-less git entry's copy as fetched while its
  manifest is at the locked version;
- a fetch with no copy there (a fresh checkout of the project, since `.r65/` is
  gitignored) clones the locked tag and pins its commit;
- `r65pkg update` resolves against the repository's tags like any git
  dependency and replaces the copy with a clone.

The lock records the version chosen, not the requirement, so `r65pkg update`
is the only thing that moves it to a newer 0.4.x. Path dependencies and patched
packages are recorded as `source = "path+../stdlib"`, without a rev; `r65c`
needs no lockfile for them. Only the root's lockfile counts: a
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
| `r65pkg fetch` | Lock any dependency not locked yet, then materialize everything in `r65.lock` into `.r65/deps/` |
| `r65pkg update [NAME]` | Re-resolve (all dependencies, or one) to the newest versions the requirements allow, and rewrite the lock |
| `r65pkg tree` | Print the dependency graph with versions and sources |
| `r65pkg list` | Resolved packages: version, source, include dir |
| `r65pkg check` | Validate manifests, lock freshness, version conflicts, build requirements (`r65-version`, `cfg`), tag/manifest agreement, cross-package include ambiguity, and bare-name includes of another package's files — without compiling |
| `r65pkg path [--include-flags]` | Print resolved include dirs, or `-I a -I b`, for tools other than `r65c` |
| `r65pkg tag NAME` | Create the `NAME-v<version>` git tag from that member's manifest version |

`r65pkg tag` is the monorepo release helper: it reads the member's `[package]
version`, refuses while that manifest differs from `HEAD`, and creates an
annotated tag, so a tag can never disagree with the manifest it claims to
release. Pushing the tag is left to you (`git push origin stdlib-v0.1.0`).
`--manifest PATH` before the command picks the package; `fetch`, `update` and
`add` take `--offline`.

### `r65c` and the Makefile

`r65c` reads `r65.toml` itself — the nearest one above the root source file,
or `--manifest PATH`; `--no-manifest` ignores it — so neither `R65C_FLAGS` nor
`WLA_FLAGS` names the package directories, and `include_bytes!` emits absolute
`.INCBIN` paths the assembler needs no `-I` for. A project Makefile runs
`r65pkg` as `$(PYTHON) -m r65.pkg.cli`, like `r65c`:

```make
$(BUILD_DIR)/main.asm: $(MAIN_SRC) | deps
	$(R65C) $(MAIN_SRC) -o $@ $(R65C_FLAGS) --depfile $(BUILD_DIR)/main.d

-include $(BUILD_DIR)/main.d

deps:
	$(R65PKG) fetch
```

`--depfile` writes a make rule naming every file the build read — each
included source, the manifests and the lockfile — so an edit to a stdlib file
or a `r65pkg update` rebuilds.

### Module layout

Split so the compiler links only against the read-only half:

```
r65/pkg/version.py    # versions and caret requirements
r65/pkg/manifest.py   # read-only: Manifest, Dependency, load/discover
r65/pkg/resolve.py    # read-only: workspace, graph walk, unification, LockedGit
r65/pkg/lockfile.py   # read + write
r65/pkg/fetch.py      # git subprocess (RemoteGit) — CLI only
r65/pkg/cli.py        # r65pkg entry point
```

`r65c` imports `version`, `manifest`, `lockfile` (to read) and `resolve`, and
nothing else, so it structurally cannot shell out to git or mutate a manifest.
The two resolvers share one graph walk and differ only in where a git
package's version comes from: `LockedGit` reads the lock and the clone,
`RemoteGit` lists tags and clones.

---

## Correctness fixes packaging depends on

These are not polish. Each was a latent silent miscompile or a silent choice
that packaging made *reachable*, so each landed with the system. They apply to
every program, packaged or not.

### Bank state leaks across include boundaries

`#[bank(n)]` is a standalone directive that mutates a positional cursor
(`r65/compiler/hir/builder.py`, both declaration passes), not an attribute.
Because `include!` splices items into one flat list, a package containing a bank
directive **silently rebanks every declaration after its include site**, and a
package's own contents inherit whatever bank the includer happened to be in.

Fix in `Preprocessor._expand_declarations`: when an included file emits any
`BankDirective`, a synthesized one restoring the includer's bank follows the
spliced items. No project in this repository relied on the leak: their
assembly was byte-identical before and after.

### Collision diagnostics carry no location

`SymbolTable.declare` raised `Redefinition of 'foo' in current scope` with
`source_loc=None` — no file, no line, no pointer at the original definition.
With packages the number of reachable collisions multiplies, so the error now
names both sites and their packages:

```
error: Redefinition of 'clash' in current scope
  first defined at .../one/a.r65:1:1 (one 0.1.0)
  --> .../app/main.r65:2:1
```

`SourceLocation` carries an `origin_package` beside `included_from`, set by the
parser for a file the preprocessor reached through a package.

### Macro redefinition is silently last-wins

`MacroExpander` collected macros with a bare dict assignment, so two packages
defining `debug!` picked one with no diagnostic at all. A macro defined in two
files is now an error naming the first; a redefinition within one file still
replaces it.

---

## Known hazards, not addressed here

Recorded so they are not mistaken for oversights. Each is pre-existing, and each
gets worse as package count grows; see `docs/future-work.md`.

- **`#[soa]` and tagged-enum types are program-wide singletons.** Each backs
  exactly one static in the whole program, so a package cannot ship one for
  general use.
- **`*dyn` TypeIds are a global counter** assigned in declaration order, so
  adding a package shifts the TypeIds of everything declared after it.

---

## Implementation

Landed:

- **`r65/pkg/`** — the modules above, and the `r65pkg` console script in
  `setup.py` (`pip install -e .` again to put it on `PATH`; Makefiles do not
  need it there).
- **`r65c`** — manifest discovery, `--manifest`, `--no-manifest`, `--depfile`,
  the build-requirement checks and `[bin] cfg`; the preprocessor's package
  roots, qualified includes and ambiguity error.
- **The correctness fixes** above.
- **stdlib** — `stdlib/r65.toml` at 0.1.0, and the workspace root with its
  `[patch]`.
- **`r65x init`** — writes `r65.toml` depending on the stdlib release it ships
  (`TemplateManager.stdlib_release`), seeded with the installed copy in
  `.r65/deps/stdlib/`, instead of copying files into `src/lib/`;
  the template includes `stdlib/...`, its Makefile has the `deps:` target and
  the depfile, and its `.gitignore` covers `.r65/`.
- **Migration** — pacman, astroids, bubblebobble, textadv and kungfu.r65: each
  has an `r65.toml` depending on stdlib by `git` and `version` (the root patch
  resolves it to `stdlib/`), `include!("stdlib/...")` in place of
  `include!("lib/...")`, and no `src/lib/`. None of their copies carried a fix
  stdlib lacked.

Not yet:

- **The first release.** No `stdlib-v*` tag exists. A new project builds from
  its installed copy regardless, but a fresh checkout of one, or `r65pkg
  update`, needs the tag: `r65pkg tag stdlib` after `stdlib/r65.toml` is
  committed, then `git push origin stdlib-v0.1.0`.
- **rotor_raiders and classickong** still carry `src/lib/` copies. They are
  separate repositories; migrating one is the same three steps.

## Verification

- `pytest r65/tests/pkg/` — manifest parse, version requirements, workspace
  discovery, unification and its conflict errors, `[patch]` (and its version
  check), build requirements, and, against throwaway local git repositories:
  one repository serving two independently tagged members, two caret ranges
  unifying to the newest release both allow, exact pins, a tag disagreeing with
  its manifest refused, the lockfile kept until `update` and stale for `r65c`
  after a requirement changes, and `r65pkg add` / `remove` / `tree` / `tag`.
- **Includes, provenance and build requirements**, through `r65c`: a qualified
  include into the named package while a bare one is ambiguous; a local file
  shadowing either spelling; a qualified and a bare include of one file
  compiling it once; the depfile; the `r65-version` and `cfg` errors and a
  `--cfg` clash with `[bin]`; both redefinition errors. Bank scoping is in
  `r65/tests/compiler/frontend/test_preprocessor.py`.
- **No change to unmigrated output.** pacman, bubblebobble, textadv,
  kungfu.r65, rotor_raiders and classickong compiled to byte-identical assembly
  before and after the compiler changes.
- **Migration.** Each migrated project's assembly is identical to a build with
  the current stdlib copied into its `src/lib/`, so the migration adopts
  stdlib's drift and nothing else. Run headless in Mesen against the
  pre-migration ROMs — idle, after Start, after holding Right — every
  screenshot matched and WRAM differed only in the stack page (return
  addresses that moved with the code).
- **End to end**: `r65x init` then `make` builds the ROM from the installed copy
  with no network and no `src/lib/` (the GitHub tag does not exist yet);
  `r65pkg tree` and `r65pkg check` report clean. In the tests, a missing copy
  is cloned and pinned and `update` replaces it with the newest release, against
  a local repository.
