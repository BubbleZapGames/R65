# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Preprocessor for R65 source files.

Handles include! statements by recursively parsing included files and
merging their declarations into a single AST Program.

An include resolves, first match winning: relative to the including file,
then each -I directory, then - with packages (docs/packaging-system.md) - a
package-qualified path (`stdlib/math.r65`), then a bare name across the
package roots, which is an error when two packages provide it.
"""

from pathlib import Path
from typing import Set, List, Optional, Sequence
from dataclasses import dataclass

from r65.compiler.frontend import ast
from r65.compiler.frontend.parser import parse as parse_source
from r65.compiler.errors import PreprocessorError, SourceLocation


@dataclass(frozen=True)
class PackageRoot:
    """A resolved package's include directory, as the preprocessor sees it."""
    name: str
    include_dir: Path
    label: str  # "stdlib 0.4.0", for diagnostics


@dataclass
class IncludeContext:
    """Tracks the include chain for error reporting and cycle detection."""
    file_path: Path
    source_loc: Optional[SourceLocation] = None  # Location of include! statement


class Preprocessor:
    """
    Preprocesses R65 source files by expanding include! statements.

    Handles:
    - Recursive file inclusion
    - Circular include detection
    - File existence validation
    - Source location tracking through include chains
    - Include path searching (-I option)
    """

    def __init__(self, source_file: str, include_paths: List[str] = None,
                 packages: Sequence[PackageRoot] = ()):
        """
        Initialize preprocessor.

        Args:
            source_file: Path to the main source file being compiled.
            include_paths: Additional directories to search for included files.
            packages: The resolved packages' include roots, searched after
                include_paths.
        """
        self.source_file = Path(source_file).resolve()
        self.source_dir = self.source_file.parent
        # Include paths for searching (resolved to absolute paths). A
        # directory can arrive twice (-I and a package), so dedup.
        self.include_paths = list(dict.fromkeys(Path(p).resolve() for p in (include_paths or [])))
        self.packages = [_absolute(p) for p in packages]
        self._packages_by_name = {p.name: p for p in self.packages}
        # The bank directive in force at the end of the items emitted so
        # far (None: the default, bank 0), restored after an include.
        self._bank: Optional[ast.BankDirective] = None
        # Track files currently being processed (for cycle detection)
        self._include_stack: List[IncludeContext] = []
        # Track all files that have been included (to avoid duplicate processing)
        self._included_files: Set[Path] = set()
        self.included_order: List[Path] = [self.source_file]

    def preprocess(self, program: ast.Program) -> ast.Program:
        """
        Preprocess AST program by expanding include! statements.

        Args:
            program: Parsed AST program

        Returns:
            New AST program with includes expanded
        """
        # Start with the main file
        self._include_stack.append(IncludeContext(file_path=self.source_file))
        self._included_files.add(self.source_file)

        try:
            expanded_items = self._expand_declarations(program.items, self.source_dir)
            return ast.Program(items=expanded_items, doc=program.doc,
                               source_loc=program.source_loc)
        finally:
            self._include_stack.pop()

    def _expand_declarations(
        self,
        items: List[ast.Declaration],
        base_dir: Path
    ) -> List[ast.Declaration]:
        """
        Expand declarations, replacing include! statements with included content.

        Args:
            items: List of declarations to process
            base_dir: Directory to resolve relative paths against

        Returns:
            Expanded list of declarations
        """
        result: List[ast.Declaration] = []

        for item in items:
            if isinstance(item, ast.IncludeStmt):
                # Expand this include. A bank directive inside it is scoped
                # to it: the includer's bank is restored after the spliced
                # items, so an included file cannot rebank what follows.
                bank = self._bank
                included_items = self._process_include(item, base_dir)
                result.extend(included_items)
                if any(isinstance(d, ast.BankDirective) for d in included_items):
                    result.append(ast.BankDirective(
                        bank_number=None if bank is not None and bank.is_auto
                        else (bank.bank_number if bank is not None else 0),
                        source_loc=item.source_loc))
                    self._bank = bank
            else:
                # Keep non-include declarations as-is
                if isinstance(item, ast.BankDirective):
                    self._bank = item
                result.append(item)

        return result

    def _resolve_include_path(self, include_path: str, base_dir: Path,
                              source_loc=None) -> Optional[Path]:
        """
        Resolve an include path, first match winning:

        1. Relative to the including file's directory (base_dir)
        2. Each directory in include_paths (-I options)
        3. A package-qualified path: `pkg/rest` with `pkg` a resolved package
        4. A bare name across package roots - an error if two packages match

        Returns:
            Resolved absolute path if found, None otherwise
        """
        # First try relative to the including file
        candidate = (base_dir / include_path).resolve()
        if candidate.exists() and candidate.is_file():
            return candidate

        # Then search include paths
        for inc_dir in self.include_paths:
            candidate = (inc_dir / include_path).resolve()
            if candidate.exists() and candidate.is_file():
                return candidate

        if not self.packages:
            return None

        first, _, rest = include_path.replace('\\', '/').partition('/')
        package = self._packages_by_name.get(first)
        if package is not None and rest:
            candidate = (package.include_dir / rest).resolve()
            if candidate.is_file():
                return candidate

        hits = {}
        for package in self.packages:
            candidate = (package.include_dir / include_path).resolve()
            if candidate.is_file():
                hits.setdefault(candidate, package)
        if len(hits) > 1:
            lines = "\n".join(f"  {p.label:<16} -> {path}" for path, p in hits.items())
            names = [p.name for p in hits.values()]
            raise PreprocessorError(
                f"include!: '{include_path}' is provided by {len(hits)} packages\n{lines}\n"
                f"  note: include it qualified ('{names[0]}/{include_path}'), "
                f"or shadow it with a local copy under -I",
                source_loc=source_loc)
        return next(iter(hits), None)

    def _package_of(self, path: Path) -> Optional[PackageRoot]:
        """The package whose include root holds `path` (the deepest one)."""
        owners = [p for p in self.packages if path.is_relative_to(p.include_dir)]
        return max(owners, key=lambda p: len(p.include_dir.parts), default=None)

    def _process_include(
        self,
        include_stmt: ast.IncludeStmt,
        base_dir: Path
    ) -> List[ast.Declaration]:
        """
        Process a single include! statement.

        Args:
            include_stmt: The include statement to process
            base_dir: Directory to resolve relative path against

        Returns:
            List of declarations from the included file
        """
        include_path = include_stmt.path
        source_loc = include_stmt.source_loc

        # Resolve the path (searches base_dir and include paths)
        resolved_path = self._resolve_include_path(include_path, base_dir, source_loc)

        # Validate file was found
        if resolved_path is None:
            searched_dirs = ([str(base_dir)] + [str(p) for p in self.include_paths]
                             + [f"{p.include_dir} ({p.label})" for p in self.packages])
            raise PreprocessorError(
                f"include!: file not found: '{include_path}'\n"
                f"  searched in: {', '.join(searched_dirs)}",
                source_loc=source_loc
            )

        # Check for circular includes
        for ctx in self._include_stack:
            if ctx.file_path == resolved_path:
                cycle = " -> ".join(str(c.file_path.name) for c in self._include_stack)
                cycle += f" -> {resolved_path.name}"
                raise PreprocessorError(
                    f"include!: circular include detected: {cycle}",
                    source_loc=source_loc
                )

        # Check if already included (skip duplicate includes)
        if resolved_path in self._included_files:
            # Already included - return empty to avoid duplicate declarations
            return []

        # Mark as included
        self._included_files.add(resolved_path)
        self.included_order.append(resolved_path)

        # Push onto include stack for cycle detection
        self._include_stack.append(IncludeContext(
            file_path=resolved_path,
            source_loc=source_loc
        ))

        try:
            # Read and parse the included file
            source_content = resolved_path.read_text()
            package = self._package_of(resolved_path)
            included_program = parse_source(
                source_content,
                str(resolved_path),
                included_from=source_loc,
                origin_package=package.label if package else None,
            )

            # Recursively expand any includes in the included file
            included_dir = resolved_path.parent
            return self._expand_declarations(included_program.items, included_dir)

        finally:
            self._include_stack.pop()


def _absolute(package: PackageRoot) -> PackageRoot:
    return PackageRoot(package.name, Path(package.include_dir).resolve(), package.label)


def preprocess(program: ast.Program, source_file: str, include_paths: List[str] = None,
               packages: Sequence[PackageRoot] = (),
               included_files: Optional[List[Path]] = None) -> ast.Program:
    """
    Preprocess an AST program by expanding include! statements.

    Args:
        program: Parsed AST program
        source_file: Path to the source file
        include_paths: Additional directories to search for included files (-I option)
        packages: Resolved packages' include roots (r65.toml)
        included_files: If given, receives every file read, the main one first

    Returns:
        New AST program with includes expanded

    Raises:
        PreprocessorError: If an include fails (file not found, circular, etc.)
    """
    preprocessor = Preprocessor(source_file, include_paths=include_paths, packages=packages)
    result = preprocessor.preprocess(program)
    if included_files is not None:
        included_files.extend(preprocessor.included_order)
    return result
