# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Preprocessor for R65 source files.

Handles include! statements by recursively parsing included files and
merging their declarations into a single AST Program.
"""

from pathlib import Path
from typing import Set, List, Optional
from dataclasses import dataclass

from r65.compiler.frontend import ast
from r65.compiler.frontend.parser import parse as parse_source
from r65.compiler.errors import PreprocessorError, SourceLocation


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

    def __init__(self, source_file: str, include_paths: List[str] = None):
        """
        Initialize preprocessor.

        Args:
            source_file: Path to the main source file being compiled.
            include_paths: Additional directories to search for included files.
        """
        self.source_file = Path(source_file).resolve()
        self.source_dir = self.source_file.parent
        # Include paths for searching (resolved to absolute paths)
        self.include_paths = [Path(p).resolve() for p in (include_paths or [])]
        # Track files currently being processed (for cycle detection)
        self._include_stack: List[IncludeContext] = []
        # Track all files that have been included (to avoid duplicate processing)
        self._included_files: Set[Path] = set()

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
            return ast.Program(items=expanded_items)
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
                # Expand this include
                included_items = self._process_include(item, base_dir)
                result.extend(included_items)
            else:
                # Keep non-include declarations as-is
                result.append(item)

        return result

    def _resolve_include_path(self, include_path: str, base_dir: Path) -> Optional[Path]:
        """
        Resolve an include path by searching base directory and include paths.

        Search order:
        1. Relative to the including file's directory (base_dir)
        2. Each directory in include_paths (-I options)

        Args:
            include_path: The path from the include! statement
            base_dir: Directory of the file containing the include!

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

        return None

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
        resolved_path = self._resolve_include_path(include_path, base_dir)

        # Validate file was found
        if resolved_path is None:
            searched_dirs = [str(base_dir)] + [str(p) for p in self.include_paths]
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

        # Push onto include stack for cycle detection
        self._include_stack.append(IncludeContext(
            file_path=resolved_path,
            source_loc=source_loc
        ))

        try:
            # Read and parse the included file
            source_content = resolved_path.read_text()
            included_program = parse_source(
                source_content,
                str(resolved_path),
                included_from=source_loc
            )

            # Recursively expand any includes in the included file
            included_dir = resolved_path.parent
            return self._expand_declarations(included_program.items, included_dir)

        finally:
            self._include_stack.pop()


def preprocess(program: ast.Program, source_file: str, include_paths: List[str] = None) -> ast.Program:
    """
    Preprocess an AST program by expanding include! statements.

    Args:
        program: Parsed AST program
        source_file: Path to the source file
        include_paths: Additional directories to search for included files (-I option)

    Returns:
        New AST program with includes expanded

    Raises:
        PreprocessorError: If an include fails (file not found, circular, etc.)
    """
    preprocessor = Preprocessor(source_file, include_paths=include_paths)
    return preprocessor.preprocess(program)
