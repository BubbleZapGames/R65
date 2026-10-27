#!/usr/bin/env python3
# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
R65 Project Tool

A CLI tool for creating R65 projects and generating assets.

Usage:
    r65x init my_game                    # Create new SNES project
    r65x init --abi FixedStack my_game   # ...with the FixedStack ABI
    r65x fontgen                         # Generate console font from TrueType
"""

import re
import shutil
import sys
import argparse
from pathlib import Path
from typing import List, Tuple


class R65XError(Exception):
    """Base exception for r65x operations."""
    pass


class ProjectInitError(R65XError):
    """Exception raised during project initialization."""
    pass


ABI_CHOICES = ['Default', 'FixedStack']


def prompt_abi() -> str:
    """Ask for the project's ABI model; empty input picks Default."""
    while True:
        try:
            answer = input("Select ABI model [Default, FixedStack] (default: Default): ").strip()
        except EOFError:
            print("\nNo input; using the Default ABI")
            return 'Default'
        if not answer:
            return 'Default'
        if answer.isdigit() and 1 <= int(answer) <= len(ABI_CHOICES):
            return ABI_CHOICES[int(answer) - 1]
        for choice in ABI_CHOICES:
            if answer.lower() == choice.lower():
                return choice
        print(f"Unknown ABI '{answer}'. Choose one of: {', '.join(ABI_CHOICES)}")


class TemplateManager:
    """Manages project templates."""

    def __init__(self):
        self.templates_dir = Path(__file__).parent / "templates"
        self.stdlib_dir = Path(__file__).parent.parent / "stdlib"
    
    def get_available_platforms(self) -> List[str]:
        """Get list of available platform templates."""
        if not self.templates_dir.exists():
            return []
        return [d.name for d in self.templates_dir.iterdir() if d.is_dir()]
    
    def copy_template(self, platform: str, target_dir: Path, project_name: str,
                      abi: str = 'Default'):
        """Copy template files to target directory."""
        template_dir = self.templates_dir / platform
        if not template_dir.exists():
            raise ProjectInitError(f"Platform '{platform}' not found. Available: {self.get_available_platforms()}")
        
        # Create target directory if it doesn't exist
        target_dir.mkdir(parents=True, exist_ok=True)

        # Create src directory for R65 source files
        src_dir = target_dir / "src"
        src_dir.mkdir(exist_ok=True)

        # Create build directory for compiled output
        build_dir = target_dir / "build"
        build_dir.mkdir(exist_ok=True)

        # The manifest: stdlib is a package dependency on its GitHub release,
        # seeded with the installed copy so the first build needs no network
        self._write_manifest(target_dir, project_name)
        self._seed_stdlib(target_dir)

        # Copy template files to src directory
        self._copy_directory(template_dir, src_dir, project_name, abi)
    
    def _copy_directory(self, src: Path, dst: Path, project_name: str, abi: str):
        """Copy directory with template substitution."""
        for item in src.iterdir():
            # Skip hidden files (vim swap files, .DS_Store, .git, etc.) —
            # they should never end up in a user's new project. Exception:
            # .gitattributes and .gitignore are real project files we ship.
            if item.name.startswith('.') and item.name not in ('.gitattributes', '.gitignore'):
                continue
            if item.is_file():
                content = self._render(item, project_name, abi)

                # Determine destination based on file type
                if item.suffix == '.md':
                    # Keep README.md in project root
                    dst_item = dst.parent / item.name
                elif item.name in ('Makefile', '.gitattributes', '.gitignore'):
                    # Keep these in project root
                    dst_item = dst.parent / item.name
                elif item.suffix == '.toml':
                    # Project-level config (r65-lint.toml, etc.) in project root
                    # so auto-discovery from src/*.r65 finds it by walking up.
                    dst_item = dst.parent / item.name
                else:
                    # R65 source files go to src directory
                    dst_item = dst / item.name

                dst_item.write_text(content, encoding='utf-8')
            elif item.is_dir():
                # Subdirectories (tests/, ...) keep their layout in the project root
                self._copy_tree(item, dst.parent / item.name, project_name, abi)

    def _copy_tree(self, src: Path, dst: Path, project_name: str, abi: str):
        """Copy a directory as-is, with template substitution."""
        dst.mkdir(exist_ok=True)
        for item in src.iterdir():
            if item.name.startswith('.'):
                continue
            if item.is_file():
                (dst / item.name).write_text(self._render(item, project_name, abi), encoding='utf-8')
            elif item.is_dir():
                self._copy_tree(item, dst / item.name, project_name, abi)

    def _render(self, item: Path, project_name: str, abi: str) -> str:
        """Read a template file and substitute template variables."""
        content = item.read_text(encoding='utf-8')
        content = content.replace("{{PROJECT_NAME}}", project_name)
        content = content.replace("{{PROJECT_NAME_UPPER}}", project_name.upper())
        content = content.replace("{{ABI}}", abi)
        # FixedStack passes every parameter in a register or a scratch slot,
        # so the project needs the stdlib's scratch block.
        abi_includes = ''
        if abi == 'FixedStack':
            abi_includes = ('// FixedStack ABI: direct-page scratch slots for parameters\n'
                            'include!("stdlib/scratch_regs.r65")\n')
        return content.replace("{{ABI_INCLUDES}}\n", abi_includes)

    def stdlib_release(self) -> Tuple[str, str]:
        """The stdlib this compiler ships: (repository, version). A new
        project requires that release - the newest this compiler is known to
        build - and anything compatible after it."""
        from r65.pkg.manifest import PackageError, load_manifest
        from r65.pkg.version import format_version
        try:
            stdlib = load_manifest(self.stdlib_dir / "r65.toml")
        except PackageError as e:
            raise ProjectInitError(f"cannot read the bundled stdlib manifest: {e}")
        if not stdlib.repository:
            raise ProjectInitError(f"{stdlib.path} has no [package] repository")
        return stdlib.repository, format_version(stdlib.version)

    def _seed_stdlib(self, target_dir: Path):
        """Copy the installed stdlib into `.r65/deps/stdlib/` and lock it as
        the GitHub release it is, with no commit: `r65pkg update`, or a fetch
        in a checkout without the copy, replaces it with a clone of that
        repository and pins the commit."""
        from r65.pkg.lockfile import LOCK_FILENAME, LockedPackage, git_source, write_lock
        from r65.pkg.resolve import DEPS_DIRNAME
        from r65.pkg.version import parse_version
        repository, version = self.stdlib_release()
        dest = target_dir / DEPS_DIRNAME / "stdlib"
        shutil.copytree(self.stdlib_dir, dest,
                        ignore=shutil.ignore_patterns('.*', '__pycache__'))
        source = git_source(repository, "tag", f"stdlib-v{version}")
        write_lock(target_dir / LOCK_FILENAME,
                   {"stdlib": LockedPackage("stdlib", parse_version(version), source)})

    def _write_manifest(self, target_dir: Path, project_name: str):
        from r65.pkg.cli import caret_of
        from r65.pkg.version import parse_version
        repository, version = self.stdlib_release()
        # A package name is letters, digits, '_' and '-'.
        name = re.sub(r'[^A-Za-z0-9_-]', '-', project_name).strip('-') or 'game'
        if name[0].isdigit():
            name = 'game-' + name
        (target_dir / "r65.toml").write_text(
            "[package]\n"
            f'name    = "{name}"\n'
            'version = "0.1.0"\n'
            "\n"
            "[bin]\n"
            'main = "src/main.r65"\n'
            'cfg  = ["snes"]\n'
            "\n"
            "[dependencies]\n"
            f'stdlib = {{ git = "{repository}", version = "{caret_of(parse_version(version))}" }}\n',
            encoding='utf-8')


def init_command(args):
    """Handle project initialization."""
    try:
        target_dir = Path(args.directory)
        
        # Check if directory already exists and has files
        if target_dir.exists() and any(target_dir.iterdir()):
            raise ProjectInitError(f"Directory '{target_dir}' already exists and is not empty")
        
        abi = args.abi or prompt_abi()

        # Initialize template manager and copy template
        template_mgr = TemplateManager()
        template_mgr.copy_template(args.platform, target_dir, target_dir.name, abi)
        
        print(f"✓ Initialized {args.platform} project in '{target_dir}' (ABI: {abi})")
        print(f"  Next steps:")
        print(f"    cd {target_dir}")
        print(f"    make             # Fetch stdlib and build the ROM")
        print(f"    make run         # Run with emulator")
        
    except R65XError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    """Main entry point for r65x CLI."""
    from r65.tools.fontgen import register_parser as fontgen_register, fontgen_command
    from r65.tools.packer import register_parser as packer_register, packer_command
    from r65.tools.bmp2chr import register_parser as bmp2chr_register, bmp2chr_command
    from r65.tools.brr import register_parser as brr_register, brr_command

    parser = argparse.ArgumentParser(
        prog='r65x',
        description='R65 Project Tool - Create projects and generate assets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  r65x init my_game                     Create new SNES project (prompts for the ABI)
  r65x init --abi FixedStack my_game    Create one with the FixedStack ABI
  r65x fontgen                          Generate console font (DejaVu Sans Mono Bold)
  r65x fontgen --font path/to/font.ttf  Generate from custom font
  r65x packer pack data.bin -o data.lz5 -x lz5
  r65x bmp2chr sprites.bmp -o sprites.chr -b4
  r65x brr encode sample.wav -o sample.brr
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # init command
    init_parser = subparsers.add_parser('init', help='Initialize new project')
    init_parser.add_argument('--platform',
                           default='snes',
                           choices=['snes'],
                           help='Target platform (default: snes)')
    init_parser.add_argument('--abi',
                           choices=ABI_CHOICES,
                           help='ABI model the project compiles with (prompted for if omitted)')
    init_parser.add_argument('directory',
                           help='Project directory name')

    # fontgen command
    fontgen_register(subparsers)

    # packer command
    packer_register(subparsers)

    # bmp2chr command
    bmp2chr_register(subparsers)

    # brr command
    brr_register(subparsers)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Route to appropriate command
    if args.command == 'init':
        init_command(args)
    elif args.command == 'fontgen':
        fontgen_command(args)
    elif args.command == 'packer':
        packer_command(args)
    elif args.command == 'bmp2chr':
        bmp2chr_command(args)
    elif args.command == 'brr':
        brr_command(args)


if __name__ == '__main__':
    main()