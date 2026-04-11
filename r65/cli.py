#!/usr/bin/env python3
# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
R65 Project Tool

A CLI tool for creating R65 projects and generating assets.

Usage:
    r65x init --platform=snes my_game    # Create new SNES project
    r65x fontgen                         # Generate console font from TrueType
"""

import sys
import argparse
from pathlib import Path
from typing import List


class R65XError(Exception):
    """Base exception for r65x operations."""
    pass


class ProjectInitError(R65XError):
    """Exception raised during project initialization."""
    pass


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
    
    def copy_template(self, platform: str, target_dir: Path, project_name: str):
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

        # Create lib directory and copy stdlib files
        lib_dir = src_dir / "lib"
        lib_dir.mkdir(exist_ok=True)
        self._copy_stdlib(lib_dir)

        # Copy template files to src directory
        self._copy_directory(template_dir, src_dir, project_name)
    
    def _copy_directory(self, src: Path, dst: Path, project_name: str):
        """Copy directory with template substitution."""
        for item in src.iterdir():
            # Skip hidden files (vim swap files, .DS_Store, .git, etc.) —
            # they should never end up in a user's new project.
            if item.name.startswith('.'):
                continue
            if item.is_file():
                # Read and substitute template variables
                content = item.read_text(encoding='utf-8')
                content = content.replace("{{PROJECT_NAME}}", project_name)
                content = content.replace("{{PROJECT_NAME_UPPER}}", project_name.upper())

                # Determine destination based on file type
                if item.suffix == '.md':
                    # Keep README.md in project root
                    dst_item = dst.parent / item.name
                elif item.name == 'Makefile':
                    # Keep Makefile in project root
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
                # Recursively copy subdirectories
                dst_subdir = dst / item.name
                dst_subdir.mkdir(exist_ok=True)
                self._copy_directory(item, dst_subdir, project_name)

    def _copy_stdlib(self, lib_dir: Path):
        """Copy standard library files to lib directory."""
        if not self.stdlib_dir.exists():
            return

        stdlib_files = ['sneslib.r65', 'math.r65', '65816.r65']
        for filename in stdlib_files:
            src_file = self.stdlib_dir / filename
            if src_file.exists():
                content = src_file.read_text(encoding='utf-8')
                (lib_dir / filename).write_text(content, encoding='utf-8')


def init_command(args):
    """Handle project initialization."""
    try:
        target_dir = Path(args.directory)
        
        # Check if directory already exists and has files
        if target_dir.exists() and any(target_dir.iterdir()):
            raise ProjectInitError(f"Directory '{target_dir}' already exists and is not empty")
        
        # Initialize template manager and copy template
        template_mgr = TemplateManager()
        template_mgr.copy_template(args.platform, target_dir, target_dir.name)
        
        print(f"✓ Initialized {args.platform} project in '{target_dir}'")
        print(f"  Next steps:")
        print(f"    cd {target_dir}")
        print(f"    make             # Build the ROM")
        print(f"    make run         # Run with emulator")
        
    except R65XError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    """Main entry point for r65x CLI."""
    from r65.tools.fontgen import register_parser as fontgen_register, fontgen_command
    from r65.tools.packer import register_parser as packer_register, packer_command
    from r65.tools.bmp2chr import register_parser as bmp2chr_register, bmp2chr_command

    parser = argparse.ArgumentParser(
        prog='r65x',
        description='R65 Project Tool - Create projects and generate assets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  r65x init --platform=snes my_game     Create new SNES project
  r65x fontgen                          Generate console font (DejaVu Sans Mono Bold)
  r65x fontgen --font path/to/font.ttf  Generate from custom font
  r65x packer pack data.bin -o data.lz5 -x lz5
  r65x bmp2chr sprites.bmp -o sprites.chr -b4
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # init command
    init_parser = subparsers.add_parser('init', help='Initialize new project')
    init_parser.add_argument('--platform',
                           required=True,
                           choices=['snes'],
                           help='Target platform')
    init_parser.add_argument('directory',
                           help='Project directory name')

    # fontgen command
    fontgen_register(subparsers)

    # packer command
    packer_register(subparsers)

    # bmp2chr command
    bmp2chr_register(subparsers)

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


if __name__ == '__main__':
    main()