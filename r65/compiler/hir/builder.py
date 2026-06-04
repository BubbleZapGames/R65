# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
HIR Builder - transforms AST to HIR with name resolution and desugaring.

Uses a two-pass algorithm:
1. First pass: Declare all top-level symbols
2. Second pass: Build HIR nodes with resolved references
"""

import re
from typing import List, Union, Optional, Dict, Set
from pathlib import Path
from r65.compiler.frontend import ast

from r65.compiler.hir import nodes as hir
from r65.compiler.hir.symbol_table import *
from r65.compiler.hir.types import *
from r65.compiler.hir.attributes import *
from r65.compiler.hir.ast_const_eval import *
from r65.compiler.hir.cfg import CfgEvaluator
from r65.compiler.hir.expression_builder import ExpressionBuilder
from r65.compiler.hir.errors import *


class HIRBuilder:
    """Builds HIR from AST with name resolution and desugaring."""

    def __init__(self, source_file: Optional[str] = None, cfg_evaluator: Optional[CfgEvaluator] = None,
                 include_paths: Optional[List[str]] = None):
        """
        Initialize HIR builder.

        Args:
            source_file: Path to source file being compiled.
                        Used for resolving relative paths in include_bytes!.
            cfg_evaluator: Optional cfg evaluator for conditional compilation.
            include_paths: Additional directories to search for include_bytes! files.
        """
        self.symbol_table = SymbolTable()
        self.cfg_evaluator = cfg_evaluator
        self.const_evaluator = ConstEvaluator(self.symbol_table, self.cfg_evaluator)
        self.type_resolver = TypeResolver(self.symbol_table, self.const_evaluator)
        self.attr_processor = AttributeProcessor()
        self.source_file = source_file
        self.source_dir = Path(source_file).parent if source_file else Path.cwd()
        # Include paths for searching (resolved to absolute paths)
        self.include_paths = [Path(p).resolve() for p in (include_paths or [])]
        self.current_bank = 0  # Current ROM bank for declarations (set by #[bank(n)])
        self.auto_bank_mode = False  # True when in #[bank(auto)] mode (NOT default for backward compatibility)
        self._pending_snesrom_config = None  # Store snesrom config when parsed as attribute
        self.expression_builder = ExpressionBuilder(
            self.symbol_table, self.const_evaluator, self.type_resolver,
            self.cfg_evaluator, self.source_dir, self.include_paths
        )
        # Wire up statement builder callback for block expressions
        self.expression_builder.statement_builder = self._build_statement
        self.warnings: list[str] = []  # Warnings collected during HIR building
        self.loop_depth = 0  # Track for loop nesting depth for register hints
        # Trait dispatch tracking
        self._next_type_id = 1  # TypeId 0 is reserved (invalid/error)
        self._struct_type_ids: Dict[str, int] = {}  # struct_name -> type_id
        self._trait_impls: Dict[str, List[str]] = {}  # trait_name -> [struct_name, ...] ordered by TypeId
        self._struct_trait_kind: Dict[str, str] = {}  # struct_name -> 'near' or 'far'
        self._dyn_used_traits: Set[str] = set()  # trait names used with *dyn

    def _process_snesrom_attribute(self, attr: ast.Attribute):
        """Process snesrom attribute that was mistakenly attached to a function."""
        # Extract values from attribute args
        name = None
        id = None
        cartridge_type = None
        sram_size = None
        country = None
        version = None
        lorom = True  # default
        hirom = False
        fastrom = False
        slowrom = True  # default

        for arg in attr.args:
            if arg.name == 'name' and isinstance(arg.value, ast.StringLiteral):
                name = arg.value.value
            elif arg.name == 'id' and isinstance(arg.value, ast.StringLiteral):
                id = arg.value.value
            elif arg.name == 'cartridge_type' and isinstance(arg.value, ast.IntegerLiteral):
                cartridge_type = arg.value.value
            elif arg.name == 'sram_size' and isinstance(arg.value, ast.IntegerLiteral):
                sram_size = arg.value.value
            elif arg.name == 'country' and isinstance(arg.value, ast.IntegerLiteral):
                country = arg.value.value
            elif arg.name == 'version' and isinstance(arg.value, ast.IntegerLiteral):
                version = arg.value.value
            elif arg.name is None and isinstance(arg.value, ast.Identifier):
                # Flag arguments like fastrom, slowrom, lorom, hirom
                flag = arg.value.name
                if flag == 'fastrom':
                    fastrom = True
                    slowrom = False
                elif flag == 'slowrom':
                    slowrom = True
                    fastrom = False
                elif flag == 'lorom':
                    lorom = True
                    hirom = False
                elif flag == 'hirom':
                    hirom = True
                    lorom = False

        self._pending_snesrom_config = hir.SnesRomConfig(
            name=name,
            id=id,
            cartridge_type=cartridge_type,
            sram_size=sram_size,
            country=country,
            version=version,
            lorom=lorom,
            hirom=hirom,
            fastrom=fastrom,
            slowrom=slowrom
        )

    def build_program(self, ast_program: ast.Program) -> hir.HIRProgram:
        """
        Build HIR from AST program.

        Args:
            ast_program: AST Program node

        Returns:
            HIRProgram with symbol table
        """
        # Track global attributes
        stack_attr = None
        snesrom_config = None

        # Pass 0: Scan for *dyn Trait usages to determine which traits need TypeIds.
        # TypeId injection changes struct layout, so it must only happen for traits
        # actually used with dynamic dispatch (*dyn), not static dispatch.
        self._dyn_used_traits = self._collect_dyn_traits(ast_program)

        # Pass 1: Declare all top-level symbols (filtered by cfg)
        # Also track bank directives to maintain ordering
        self.current_bank = 0  # Reset to default bank
        self.auto_bank_mode = False  # Default is explicit bank 0 for backward compatibility
        for decl in ast_program.items:
            if isinstance(decl, ast.StackDirective):
                # Create stack attribute from directive
                stack_attr = StackAttribute(
                    name='stack',
                    lower=decl.lower,
                    upper=decl.upper
                )
            elif isinstance(decl, ast.BankDirective):
                # Update current bank context
                if decl.is_auto:
                    # #[bank(auto)] - automatic placement mode
                    self.auto_bank_mode = True
                    self.current_bank = 0  # Will be determined at link time
                else:
                    # #[bank(n)] - explicit bank number
                    self.auto_bank_mode = False
                    self.current_bank = decl.bank_number
            elif isinstance(decl, ast.SnesRomDirective):
                # Create SNES ROM config from directive
                snesrom_config = hir.SnesRomConfig(
                    name=decl.name,
                    id=decl.id,
                    cartridge_type=decl.cartridge_type,
                    sram_size=decl.sram_size,
                    country=decl.country,
                    version=decl.version,
                    lorom=decl.lorom,
                    hirom=decl.hirom,
                    exhirom=decl.exhirom,
                    slowrom=decl.slowrom,
                    fastrom=decl.fastrom
                )
            else:
                if self._should_include_declaration(decl):
                    self._declare_toplevel(decl)

        # Pass 2: Build HIR nodes with resolved references (filtered by cfg)
        hir_decls = []
        self.current_bank = 0  # Reset for second pass
        self.auto_bank_mode = False  # Default is explicit bank 0 for backward compatibility
        for decl in ast_program.items:
            if isinstance(decl, ast.StackDirective):
                continue  # Skip stack directives, already processed
            if isinstance(decl, ast.BankDirective):
                # Update current bank context for following declarations
                if decl.is_auto:
                    self.auto_bank_mode = True
                    self.current_bank = 0
                else:
                    self.auto_bank_mode = False
                    self.current_bank = decl.bank_number
                continue
            if isinstance(decl, ast.SnesRomDirective):
                continue  # Skip snesrom directives, already processed
            if isinstance(decl, ast.IncludeAsmStmt):
                bank_num = None if self.auto_bank_mode else self.current_bank
                # Resolve path relative to the including .r65 file so WLA-DX
                # gets an unambiguous location regardless of where it's run.
                resolved = self._resolve_include_asm_path(decl.path)
                if resolved is None:
                    searched = [str(self.source_dir)] + [str(p) for p in self.include_paths]
                    raise HIRError(
                        f"include_asm!: file not found: '{decl.path}'\n"
                        f"  searched in: {', '.join(searched)}",
                        source_loc=decl.source_loc,
                    )
                hir_decls.append(hir.HIRIncludeAsm(
                    path=str(resolved),
                    bank_number=bank_num,
                    source_loc=decl.source_loc,
                ))
                continue
            if self._should_include_declaration(decl):
                hir_decl = self._build_declaration(decl)
                hir_decls.append(hir_decl)

        # Use pending snesrom config if it was set from an attribute
        if snesrom_config is None and self._pending_snesrom_config is not None:
            snesrom_config = self._pending_snesrom_config

        # Build trait dispatch info for codegen
        trait_dispatch_info = self._build_trait_dispatch_info()

        return hir.HIRProgram(
            declarations=hir_decls,
            symbol_table=self.symbol_table,
            stack_attr=stack_attr,
            snesrom_config=snesrom_config,
            trait_dispatch_info=trait_dispatch_info
        )

    def _collect_dyn_traits(self, program: ast.Program) -> Set[str]:
        """Pre-pass: find all trait names used in *dyn Trait positions.

        TypeId injection (which changes struct layout) must only happen for traits
        used with dynamic dispatch. This scan determines which traits need it.
        """
        dyn_traits: Set[str] = set()

        def scan_type(t) -> None:
            if t is None:
                return
            if isinstance(t, ast.PointerType):
                if t.is_dyn and isinstance(t.pointee_type, ast.BasicType):
                    dyn_traits.add(t.pointee_type.name)
                scan_type(t.pointee_type)
            elif isinstance(t, ast.ArrayType):
                scan_type(t.element_type)
            elif isinstance(t, ast.FunctionType):
                for pt in t.param_types:
                    scan_type(pt)
                scan_type(t.return_type)

        def scan_expr(e) -> None:
            if e is None:
                return
            if isinstance(e, ast.TypeCast):
                scan_type(e.target_type)
                scan_expr(e.expr)
            elif isinstance(e, ast.BinaryOp):
                scan_expr(e.left)
                scan_expr(e.right)
            elif isinstance(e, ast.UnaryOp):
                scan_expr(e.operand)
            elif isinstance(e, ast.FunctionCall):
                scan_expr(e.func)
                for arg in e.args:
                    scan_expr(arg)
            elif isinstance(e, ast.ArrayIndex):
                scan_expr(e.array)
                scan_expr(e.index)
            elif isinstance(e, ast.FieldAccess):
                scan_expr(e.base)
            elif isinstance(e, ast.Dereference):
                scan_expr(e.pointer)
            elif isinstance(e, ast.AddressOf):
                scan_expr(e.operand)
            elif isinstance(e, (ast.Assignment, ast.CompoundAssignment)):
                scan_expr(e.target)
                scan_expr(e.value)
            elif isinstance(e, ast.MultiAssignment):
                for t in e.targets:
                    scan_expr(t)
                scan_expr(e.value)
            elif isinstance(e, ast.ArrayLiteralExpr):
                for elem in e.elements:
                    scan_expr(elem)
            elif isinstance(e, ast.StructLiteralExpr):
                for fi in e.fields:
                    scan_expr(fi.value)
            elif isinstance(e, ast.BlockExpression):
                for stmt in e.statements:
                    scan_stmt(stmt)
                scan_expr(e.final_expr)
            elif isinstance(e, ast.IfExpression):
                scan_expr(e.condition)
                scan_expr(e.then_block)
                scan_expr(e.else_block)
            elif isinstance(e, ast.LoopExpression):
                scan_block(e.body)
            elif isinstance(e, ast.MatchExpression):
                scan_expr(e.scrutinee)
                for arm in e.arms:
                    scan_expr(arm.body)
            elif isinstance(e, ast.BreakStmt):
                scan_expr(e.value)

        def scan_block(block) -> None:
            if block is None:
                return
            for stmt in block.statements:
                scan_stmt(stmt)

        def scan_stmt(s) -> None:
            if s is None:
                return
            if isinstance(s, ast.Block):
                scan_block(s)
            elif isinstance(s, ast.LetStmt):
                scan_type(s.var_type)
                scan_expr(s.initializer)
            elif isinstance(s, ast.MultiLetStmt):
                scan_expr(s.initializer)
            elif isinstance(s, ast.ExprStmt):
                scan_expr(s.expr)
            elif isinstance(s, ast.ReturnStmt):
                for v in s.values:
                    scan_expr(v)
            elif isinstance(s, ast.IfStmt):
                scan_expr(s.condition)
                scan_block(s.then_block)
                scan_stmt(s.else_block)
            elif isinstance(s, ast.LoopStmt):
                scan_block(s.body)
            elif isinstance(s, ast.WhileStmt):
                scan_expr(s.condition)
                scan_block(s.body)
            elif isinstance(s, ast.ForStmt):
                scan_expr(s.start)
                scan_expr(s.end)
                scan_block(s.body)
            elif isinstance(s, ast.BreakStmt):
                scan_expr(s.value)
            elif isinstance(s, ast.ConstAssertStmt):
                scan_expr(s.condition)

        def scan_params(params) -> None:
            for p in params:
                scan_type(p.param_type)

        for decl in program.items:
            if isinstance(decl, ast.FunctionDecl):
                scan_params(decl.params)
                scan_type(decl.return_type)
                scan_block(decl.body)
            elif isinstance(decl, ast.StaticDecl):
                scan_type(decl.var_type)
            elif isinstance(decl, ast.StructDecl):
                for field in decl.fields:
                    scan_type(field.field_type)
            elif isinstance(decl, ast.TraitDecl):
                for method in decl.methods:
                    scan_params(method.params)
                    scan_type(method.return_type)
            elif isinstance(decl, ast.ImplDecl):
                for method in decl.methods:
                    scan_params(method.params)
                    scan_type(method.return_type)
                    scan_block(method.body)

        return dyn_traits

    def _should_include_declaration(self, decl: ast.Declaration) -> bool:
        """
        Check if a declaration should be included based on cfg attributes.

        Args:
            decl: AST declaration to check

        Returns:
            True if declaration should be included, False otherwise
        """
        # Get attributes from declaration (only FunctionDecl and StaticDecl have attributes)
        attributes = []
        if isinstance(decl, (ast.FunctionDecl, ast.StaticDecl)):
            attributes = decl.attributes

        # Check for cfg attributes
        cfg_attrs = [attr for attr in attributes if attr.name == 'cfg']

        if not cfg_attrs:
            return True  # No cfg attributes means include

        # Use provided evaluator or empty one (cfg conditions false by default)
        evaluator = self.cfg_evaluator or CfgEvaluator(set(), {})

        # If any cfg attribute evaluates to true, include the declaration
        for attr in cfg_attrs:
            processed_attr = self.attr_processor.process_attributes([attr], 'declaration')[0]
            if isinstance(processed_attr, CfgAttribute):
                if evaluator.evaluate(processed_attr.condition):
                    return True

        return False  # All cfg attributes evaluated to false

    def _evaluate_cfg_condition(self, condition: ast.CfgCondition) -> hir.HIRBooleanLiteral:
        """
        Evaluate a cfg condition and convert to boolean literal.

        Args:
            condition: AST cfg condition

        Returns:
            HIRBooleanLiteral with true/false value
        """
        # Use provided evaluator or empty one (cfg conditions false by default)
        evaluator = self.cfg_evaluator or CfgEvaluator(set(), {})
        result = evaluator.evaluate(condition)
        return hir.HIRBooleanLiteral(value=result)

    # =========================================================================
    # Pass 1: Declare Top-Level Symbols
    # =========================================================================

    def _declare_toplevel(self, decl: ast.Declaration):
        """First pass: declare top-level symbols."""
        if isinstance(decl, ast.FunctionDecl):
            # Create function symbol
            symbol = Symbol(
                name=decl.name,
                kind=SymbolKind.FUNCTION,
                definition=decl,
                scope_id=0
            )
            self.symbol_table.declare(decl.name, symbol)

        elif isinstance(decl, ast.StaticDecl):
            # Resolve type (or infer from include_bytes!)
            if decl.var_type is not None:
                var_type = self.type_resolver.resolve_type(decl.var_type)
            elif isinstance(decl.initializer, ast.IncludeBytesExpr):
                # Infer [u8; N] from file size
                _, file_size = self.expression_builder._validate_include_bytes_path(
                    decl.initializer.path, decl.source_loc
                )
                var_type = ArrayTypeInfo(
                    element_type=BasicTypeInfo(name='u8'),
                    size=file_size
                )
            else:
                raise HIRError(
                    f"static '{decl.name}' requires a type annotation",
                    source_loc=decl.source_loc,
                    hint="add ': Type' after the name, e.g., static NAME: u8 = ...;"
                )
            symbol = Symbol(
                name=decl.name,
                kind=SymbolKind.STATIC_VAR,
                definition=decl,
                scope_id=0,
                var_type=var_type,
                is_mutable=decl.is_mut
            )
            self.symbol_table.declare(decl.name, symbol)

        elif isinstance(decl, ast.ConstDecl):
            # Resolve type
            const_type = self.type_resolver.resolve_type(decl.const_type)

            # Evaluate const value
            const_value = self.const_evaluator.eval(decl.value)

            symbol = Symbol(
                name=decl.name,
                kind=SymbolKind.CONST,
                definition=decl,
                scope_id=0,
                var_type=const_type,
                const_value=const_value
            )
            self.symbol_table.declare(decl.name, symbol)

        elif isinstance(decl, ast.StructDecl):
            # Create struct type symbol
            symbol = Symbol(
                name=decl.name,
                kind=SymbolKind.STRUCT,
                definition=decl,
                scope_id=0
            )
            self.symbol_table.declare(decl.name, symbol)

        elif isinstance(decl, ast.EnumDecl):
            # Create enum type symbol
            enum_symbol = Symbol(
                name=decl.name,
                kind=SymbolKind.ENUM,
                definition=decl,
                scope_id=0
            )
            self.symbol_table.declare(decl.name, enum_symbol)

            # Declare variants in global scope with qualified names
            current_value = 0
            for variant in decl.variants:
                if variant.value is not None:
                    current_value = self.const_evaluator.eval(variant.value)

                variant_symbol = Symbol(
                    name=f"{decl.name}::{variant.name}",
                    kind=SymbolKind.ENUM_VARIANT,
                    definition=variant,
                    scope_id=0,
                    const_value=current_value
                )
                # Store with qualified name for resolution
                self.symbol_table.declare(f"{decl.name}::{variant.name}", variant_symbol)

                current_value += 1

        elif isinstance(decl, ast.TypeAlias):
            # Resolve aliased type
            aliased_type = self.type_resolver.resolve_type(decl.aliased_type)

            symbol = Symbol(
                name=decl.name,
                kind=SymbolKind.TYPE_ALIAS,
                definition=decl,
                scope_id=0,
                type_info=aliased_type
            )
            self.symbol_table.declare(decl.name, symbol)

        elif isinstance(decl, ast.TraitDecl):
            # Create trait symbol
            symbol = Symbol(
                name=decl.name,
                kind=SymbolKind.TRAIT,
                definition=decl,
                scope_id=0
            )
            self.symbol_table.declare(decl.name, symbol)

        elif isinstance(decl, ast.IncludeStmt):
            # Include statements are handled by preprocessing (not in this phase)
            pass

        elif isinstance(decl, ast.ImplDecl):
            # Declare impl block methods and constants
            self._declare_impl(decl)

    # =========================================================================
    # Pass 2: Build HIR Declarations
    # =========================================================================

    def _build_declaration(self, decl: ast.Declaration) -> hir.HIRDeclaration:
        """Build HIR declaration from AST."""
        if isinstance(decl, ast.FunctionDecl):
            return self._build_function(decl)
        elif isinstance(decl, ast.StaticDecl):
            return self._build_static(decl)
        elif isinstance(decl, ast.ConstDecl):
            return self._build_const(decl)
        elif isinstance(decl, ast.StructDecl):
            return self._build_struct(decl)
        elif isinstance(decl, ast.EnumDecl):
            return self._build_enum(decl)
        elif isinstance(decl, ast.TypeAlias):
            return self._build_type_alias(decl)
        elif isinstance(decl, ast.TraitDecl):
            return self._build_trait(decl)
        elif isinstance(decl, ast.ImplDecl):
            return self._build_impl(decl)
        else:
            raise HIRError(f"Unknown declaration type: {type(decl).__name__}", source_loc=getattr(decl, 'source_loc', None))

    def _resolve_include_asm_path(self, path: str) -> Optional[Path]:
        """Resolve an include_asm! path relative to the source file or -I paths."""
        candidate = (self.source_dir / path).resolve()
        if candidate.exists() and candidate.is_file():
            return candidate
        for inc_dir in self.include_paths:
            candidate = (inc_dir / path).resolve()
            if candidate.exists() and candidate.is_file():
                return candidate
        return None

    def _build_extern_function(self, func: ast.FunctionDecl) -> hir.HIRFunctionDecl:
        """Build HIR for an `extern fn` declaration (no body).

        The body lives in an asm file included via `include_asm!`. We still need
        a complete signature (params, return type, bank, mode) so call sites can
        be type-checked and lowered to JSR/JSL with correct argument passing.

        Defaults to all-clobbered; user opts into preservation with
        `#[preserves(...)]`. Near extern (`extern fn`) must inherit a concrete
        bank from `#[bank(n)]`; auto-bank requires `extern far fn`.
        """
        processed_attrs = self.attr_processor.process_attributes(
            func.attributes,
            context='function'
        )
        attrs = self._extract_attributes(processed_attrs)
        mode_attr = attrs['mode']
        preserves_attr = attrs['preserves']
        interrupt_attr = attrs['interrupt']
        is_entry = attrs['is_entry']

        if interrupt_attr is not None:
            raise HIRError(
                f"extern fn '{func.name}' cannot be declared `#[interrupt(...)]`",
                source_loc=func.source_loc,
                hint="interrupt handlers must have a body — implement the handler in R65 or use an entirely asm-side vector",
            )
        if is_entry:
            raise HIRError(
                f"extern fn '{func.name}' cannot be marked entry",
                source_loc=func.source_loc,
            )

        if self.auto_bank_mode:
            if not func.is_far:
                raise HIRError(
                    f"extern fn '{func.name}' in auto-bank mode must be declared `extern far fn`",
                    source_loc=func.source_loc,
                )
            bank_attr = BankAttribute(name='bank', bank_number=None)
        else:
            bank_attr = BankAttribute(name='bank', bank_number=self.current_bank)

        if mode_attr and mode_attr.databank != DataBankMode.NONE and not func.is_far:
            raise HIRError(
                f"extern fn '{func.name}' uses databank={mode_attr.databank.value} "
                "but is not a far function; DBR management requires `extern far fn`.",
                source_loc=func.source_loc,
            )

        self.symbol_table.enter_scope(ScopeKind.FUNCTION)
        try:
            hir_params = [self._build_parameter(p) for p in func.params]
            self._validate_no_duplicate_register_bindings(hir_params, func.name)
        finally:
            self.symbol_table.exit_scope()

        entry_m_mode = self._infer_entry_mode_and_validate(hir_params, func.name, func.source_loc)
        ret_type = None
        if func.return_type:
            ret_type = self._resolve_function_return_type(func.return_type, entry_m_mode)
        exit_m_mode = self._infer_exit_mode(ret_type)

        func_symbol = self.symbol_table.lookup(func.name)
        return hir.HIRFunctionDecl(
            name=func.name,
            is_far=func.is_far,
            is_const=False,
            is_extern=True,
            parameters=hir_params,
            return_type=ret_type,
            body=None,
            mode_attr=mode_attr,
            preserves_attr=preserves_attr,
            bank_attr=bank_attr,
            interrupt_attr=None,
            inline_attr=None,
            is_entry=False,
            symbol=func_symbol,
            returns_status_flag=None,
            entry_m_mode=entry_m_mode,
            exit_m_mode=exit_m_mode,
            source_loc=func.source_loc,
        )

    def _build_function(self, func: ast.FunctionDecl) -> hir.HIRFunctionDecl:
        """Build HIR function from AST."""
        if func.is_extern:
            return self._build_extern_function(func)
        # Filter out snesrom attributes (they're program-level, not function-level)
        # This can happen due to parser ambiguity when #[snesrom(...)] is followed by function
        func_attrs = []
        for attr in func.attributes:
            if attr.name == 'snesrom':
                # Convert attribute back to directive and process at program level
                self._process_snesrom_attribute(attr)
            else:
                func_attrs.append(attr)

        # Process attributes
        processed_attrs = self.attr_processor.process_attributes(
            func_attrs,
            context='function'
        )

        # Extract specific attributes
        attrs = self._extract_attributes(processed_attrs)
        mode_attr = attrs['mode']
        preserves_attr = attrs['preserves']
        interrupt_attr = attrs['interrupt']
        inline_attr = attrs['inline']
        is_entry = attrs['is_entry']

        # Bank comes from current bank context (set by #[bank(n)] directive)
        # In auto-bank mode, use None to indicate automatic placement
        if self.auto_bank_mode:
            bank_attr = BankAttribute(name='bank', bank_number=None)
            # Validate: functions in auto-bank mode must be far
            if not func.is_far:
                raise HIRError(
                    f"function '{func.name}' in auto-bank mode must be declared as 'far fn'",
                    source_loc=func.source_loc,
                    hint="use 'far fn " + func.name + "(...)' or place in explicit bank with #[bank(n)]"
                )
        else:
            bank_attr = BankAttribute(name='bank', bank_number=self.current_bank)

        # Enter function scope
        func_scope_id = self.symbol_table.enter_scope(ScopeKind.FUNCTION)

        # Process parameters
        hir_params = []
        for param in func.params:
            hir_param = self._build_parameter(param)
            hir_params.append(hir_param)

        # Validate no duplicate register bindings
        self._validate_no_duplicate_register_bindings(hir_params, func.name)

        # Process body
        hir_body = self._build_block(func.body)

        # Add implicit return A if needed (but not for interrupt handlers)
        self._add_implicit_return(hir_body, func.return_type, interrupt_attr)

        # Exit function scope
        self.symbol_table.exit_scope()

        # Infer entry mode from parameters and validate X/Y are u16
        # Must happen before return type resolution (MultiReturnType needs m_mode)
        entry_m_mode = self._infer_entry_mode_and_validate(hir_params, func.name, func.source_loc)

        # Resolve return type (MultiReturnType uses entry_m_mode to derive element types)
        ret_type = None
        if func.return_type:
            ret_type = self._resolve_function_return_type(func.return_type, entry_m_mode)

        # Get function symbol
        func_symbol = self.symbol_table.lookup(func.name)

        # Validate: DBR management modes require far functions
        if mode_attr and mode_attr.databank != DataBankMode.NONE:
            if not func.is_far:
                raise HIRError(
                    f"Function '{func.name}' uses databank={mode_attr.databank.value} "
                    f"but is not a far function. DBR management requires 'far fn'.",
                    source_loc=func.source_loc
                )

        # Detect STATUS flag return pattern for optimized branch generation at call sites
        returns_status_flag = self._detect_status_flag_return(hir_body)

        # Auto-detect trivial getters/setters and mark for inlining
        # Only if not already marked and not a far/interrupt/entry function
        if (inline_attr is None and
            not func.is_far and
            interrupt_attr is None and
            not is_entry and
            self._is_trivial_getter_or_setter(hir_body)):
            inline_attr = InlineAttribute(name='inline')

        # Infer exit mode from return type
        exit_m_mode = self._infer_exit_mode(ret_type)

        # Eagerly validate const fn body at definition time
        if func.is_const:
            self.const_evaluator.validate_const_fn(func, func.name)

        return hir.HIRFunctionDecl(
            name=func.name,
            is_far=func.is_far,
            is_const=func.is_const,
            parameters=hir_params,
            return_type=ret_type,
            body=hir_body,
            mode_attr=mode_attr,
            preserves_attr=preserves_attr,
            bank_attr=bank_attr,
            interrupt_attr=interrupt_attr,
            inline_attr=inline_attr,
            is_entry=is_entry,
            symbol=func_symbol,
            returns_status_flag=returns_status_flag,
            entry_m_mode=entry_m_mode,
            exit_m_mode=exit_m_mode,
            source_loc=func.source_loc  # Propagate source location from AST
        )

    def _detect_status_flag_return(self, body: hir.HIRBlock) -> Optional[str]:
        """
        Detect if function directly returns a STATUS flag.

        Checks if the last statement is `return STATUS.Flag` pattern.
        Returns the flag name if found, None otherwise.

        This enables call sites to use direct branch instructions instead of
        materializing the boolean return value.
        """
        if not body or not body.statements:
            return None

        last_stmt = body.statements[-1]

        # Check if last statement is a return
        if not isinstance(last_stmt, hir.HIRReturnStmt):
            return None

        # Check if return has exactly one value
        if len(last_stmt.values) != 1:
            return None

        return_value = last_stmt.values[0]

        # Check if return value is a STATUS flag access
        if isinstance(return_value, hir.HIRStatusFlagAccess):
            return return_value.flag_name

        return None

    def _is_simple_expression(self, expr: hir.HIRExpression, depth: int = 0) -> bool:
        """
        Check if expression is simple enough for auto-inlining.

        Simple expressions are those that compile to just a few instructions:
        - Literals (integers, booleans)
        - Identifiers (variables, parameters)
        - Field access (self.field)
        - Binary ops with simple operands (myVar & 0xF, x + 1)
        - Unary ops with simple operand (~flags, !condition)

        Args:
            expr: HIR expression to check
            depth: Current recursion depth (limits complexity)

        Returns:
            True if expression is simple enough.
        """
        # Limit recursion depth to avoid complex nested expressions
        if depth > 2:
            return False

        # Direct simple values
        if isinstance(expr, (hir.HIRIntegerLiteral, hir.HIRBooleanLiteral,
                             hir.HIRIdentifier, hir.HIRFieldAccess,
                             hir.HIRRegister)):
            return True

        # Binary op with simple operands: myVar & 0xF, x + 1, etc.
        if isinstance(expr, hir.HIRBinaryOp):
            return (self._is_simple_expression(expr.left, depth + 1) and
                    self._is_simple_expression(expr.right, depth + 1))

        # Unary op with simple operand: ~flags, !condition
        if isinstance(expr, hir.HIRUnaryOp):
            return self._is_simple_expression(expr.operand, depth + 1)

        return False

    def _is_trivial_getter_or_setter(self, body: hir.HIRBlock) -> bool:
        """
        Detect if function is a trivial getter or setter.

        Getter patterns (single return statement with simple expression):
        - return 15;              -> literal
        - return self.field;      -> field access
        - return variable;        -> identifier
        - return myVar & 0xF;     -> simple binary op
        - return ~flags;          -> simple unary op

        Setter patterns (single assignment, possibly with implicit return):
        - self.field = value;     -> HIRExprStmt with HIRAssignment to HIRFieldAccess
        - STATIC = value;         -> HIRExprStmt with HIRAssignment to HIRIdentifier

        These trivial functions should always be inlined to eliminate call overhead.

        Returns:
            True if function matches a getter/setter pattern.
        """
        if not body or not body.statements:
            return False

        stmts = body.statements

        # Getter: single return statement with a simple expression
        if len(stmts) == 1 and isinstance(stmts[0], hir.HIRReturnStmt):
            ret_stmt = stmts[0]
            if len(ret_stmt.values) == 1:
                if self._is_simple_expression(ret_stmt.values[0]):
                    return True

        # Setter: single assignment (possibly followed by implicit return)
        # After _add_implicit_return, a setter looks like: [HIRExprStmt(assignment), HIRReturnStmt]
        if len(stmts) in (1, 2):
            first_stmt = stmts[0]
            # Check first statement is an assignment
            if isinstance(first_stmt, hir.HIRExprStmt) and isinstance(first_stmt.expr, hir.HIRAssignment):
                assignment = first_stmt.expr
                # Target must be a field access or identifier (static variable)
                if isinstance(assignment.target, (hir.HIRFieldAccess, hir.HIRIdentifier)):
                    # Value must be a simple expression
                    if self._is_simple_expression(assignment.value):
                        # If there's a second statement, it must be an empty return
                        if len(stmts) == 1:
                            return True
                        elif len(stmts) == 2 and isinstance(stmts[1], hir.HIRReturnStmt):
                            return True

        return False

    def _infer_entry_mode_and_validate(self, params: List[hir.HIRParameter], func_name: str, source_loc) -> 'ModeState':
        """
        Infer entry mode from function parameters.

        Rules:
        - If any parameter is bound to A with type u16/i16 -> m16 entry
        - Otherwise -> m8 entry (default)

        Note: Register type validation (X/Y must be u16, etc.) is handled by
        _validate_register_binding_type() during parameter building.

        Args:
            params: List of HIR function parameters
            func_name: Function name for error messages
            source_loc: Source location for error messages

        Returns:
            ModeState for function entry (M8 or M16)
        """
        from r65.compiler.typeck.processor_mode import ModeState

        entry_mode = ModeState.M8  # Default
        a_param = None  # Track the A register parameter if 16-bit
        b_param = None  # Track any B register parameter

        for param in params:
            if isinstance(param.binding, hir.RegisterBinding):
                reg_name = param.binding.register_name

                if reg_name == "A":
                    # Check if A parameter is 16-bit
                    if param.param_type and isinstance(param.param_type, BasicTypeInfo):
                        if param.param_type.name in ('u16', 'i16'):
                            entry_mode = ModeState.M16
                            a_param = param
                elif reg_name == "B":
                    b_param = param

        # Validate: B register parameter is incompatible with 16-bit A parameter
        if entry_mode == ModeState.M16 and b_param is not None:
            a_type_name = a_param.param_type.name if a_param else 'u16'
            raise HIRError(
                f"Function '{func_name}' has a B register parameter '{b_param.name}' "
                f"but also has a 16-bit A parameter (@ A: {a_type_name}), which puts the "
                f"CPU in m16 mode where the B register does not exist",
                hint=(
                    f"In m16 mode, the accumulator is 16 bits wide and there is no "
                    f"separate B register.\n"
                    f"  Either remove the B parameter or change A to 8-bit: @ A: u8"
                ),
                source_loc=source_loc,
            )

        return entry_mode

    def _resolve_function_return_type(self, ast_return_type, entry_m_mode) -> 'TypeInfo':
        """Resolve function return type, handling MultiReturnType specially."""
        from r65.compiler.frontend import ast as ast_mod
        if isinstance(ast_return_type, ast_mod.MultiReturnType):
            return self._multi_return_to_tuple(ast_return_type, entry_m_mode)
        return self.type_resolver.resolve_type(ast_return_type)

    def _multi_return_to_tuple(self, mrt, entry_m_mode) -> 'TupleTypeInfo':
        """Convert a multi-return type list (e.g. `u8, u16`) to a TupleTypeInfo.

        Values are assigned to hardware registers in order — A, then B (m8 only)
        or X, then Y — by `get_return_registers`. This validates that the value
        count and each value's width are expressible in the chosen registers
        given the function's accumulator mode.
        """
        from r65.compiler.hir.types import TupleTypeInfo
        from r65.compiler.typeck.processor_mode import ModeState
        from r65.compiler.codegen.constants import get_return_registers

        loc = getattr(mrt, 'source_loc', None)
        element_types = [self.type_resolver.resolve_type(t) for t in mrt.element_types]

        if len(element_types) < 2:
            raise HIRError(
                "A multi-return type needs at least two types; "
                "use a single type for one return value",
                source_loc=loc,
            )

        tuple_info = TupleTypeInfo(element_types=element_types)
        regs = get_return_registers(tuple_info, entry_m_mode)
        is_m16 = entry_m_mode == ModeState.M16

        if len(element_types) > len(regs):
            raise HIRError(
                f"Too many return values: {len(element_types)} types declared but "
                f"only {len(regs)} return registers ({', '.join(regs)}) are "
                f"available. Four values are possible only when the second value "
                f"is 8-bit (u8/i8/bool) in m8 mode, which frees the B register; "
                f"otherwise the maximum is three.",
                source_loc=loc,
            )

        for et, reg in zip(element_types, regs):
            if reg == 'A':
                want = 2 if is_m16 else 1
            elif reg == 'B':
                want = 1
            else:  # X and Y are always 16-bit
                want = 2
            if et.size_bytes != want:
                raise HIRError(
                    f"Return value of type '{et}' ({et.size_bytes}-byte) does not "
                    f"fit return register {reg}, which holds {want} byte(s) in "
                    f"{'m16' if is_m16 else 'm8'} mode. Returns map to A "
                    f"(mode-width), then B (1-byte, m8 only) or X, then Y "
                    f"(both 2-byte).",
                    source_loc=loc,
                )

        return tuple_info

    def _infer_exit_mode(self, return_type: Optional[TypeInfo]) -> 'ModeState':
        """
        Infer exit mode from function return type.

        Rules:
        - If return type is u16/i16 (returns in A register) -> m16 exit
        - Otherwise -> m8 exit (default)

        Args:
            return_type: Function return type (or None for void)

        Returns:
            ModeState for function exit (M8 or M16)
        """
        from r65.compiler.typeck.processor_mode import ModeState

        if return_type is None:
            return ModeState.M8

        if isinstance(return_type, BasicTypeInfo):
            if return_type.name in ('u16', 'i16'):
                return ModeState.M16

        return ModeState.M8

    def _build_parameter(self, param: ast.Parameter) -> hir.HIRParameter:
        """Build HIR parameter from AST."""
        # Resolve parameter type
        param_type = self.type_resolver.resolve_type(param.param_type)

        # Process binding
        binding = None
        if param.binding:
            if isinstance(param.binding, ast.Register):
                binding = hir.RegisterBinding(register_name=param.binding.name)
            elif isinstance(param.binding, ast.Identifier):
                # Could be register or variable binding - resolve
                var_name = param.binding.name
                var_symbol = self.symbol_table.lookup(var_name)
                if not var_symbol:
                    raise HIRError(f"Undefined variable: {var_name}", source_loc=getattr(param, 'source_loc', None))
                if var_symbol.kind == SymbolKind.REGISTER:
                    # Register binding
                    binding = hir.RegisterBinding(register_name=var_name)
                elif var_symbol.kind == SymbolKind.STATIC_VAR:
                    # Variable binding
                    binding = hir.VariableBinding(
                        variable_name=var_name,
                        variable_symbol=var_symbol
                    )
                else:
                    raise HIRError(f"Parameter binding must be register or static variable, got {var_symbol.kind.value}", source_loc=getattr(param, 'source_loc', None))
            elif isinstance(param.binding, str):
                # Could be register or variable binding - resolve (legacy string support)
                var_symbol = self.symbol_table.lookup(param.binding)
                if not var_symbol:
                    raise HIRError(f"Undefined variable: {param.binding}", source_loc=getattr(param, 'source_loc', None))
                if var_symbol.kind == SymbolKind.REGISTER:
                    # Register binding
                    binding = hir.RegisterBinding(register_name=param.binding)
                elif var_symbol.kind == SymbolKind.STATIC_VAR:
                    # Variable binding
                    binding = hir.VariableBinding(
                        variable_name=param.binding,
                        variable_symbol=var_symbol
                    )
                else:
                    raise HIRError(f"Parameter binding must be register or static variable, got {var_symbol.kind.value}", source_loc=getattr(param, 'source_loc', None))

        # Validate register binding types
        if binding and isinstance(binding, hir.RegisterBinding):
            self._validate_register_binding_type(
                binding.register_name,
                param_type,
                param.name,
                "parameter"
            )

        # Declare parameter in function scope
        param_symbol = Symbol(
            name=param.name,
            kind=SymbolKind.PARAMETER,
            definition=param,
            scope_id=self.symbol_table.current_scope_id,
            var_type=param_type,
            is_mutable=True  # Parameters are mutable
        )
        self.symbol_table.declare(param.name, param_symbol)

        return hir.HIRParameter(
            name=param.name,
            param_type=param_type,
            binding=binding,
            symbol=param_symbol
        )

    def _validate_register_binding_type(
        self,
        register_name: str,
        bound_type: Optional[TypeInfo],
        name: str,
        context: str  # "parameter" or "variable"
    ) -> None:
        """
        Validate that a register binding type matches the register's supported types.

        Register type rules:
        - A: u8, i8, u16, i16 (accumulator, supports both 8-bit and 16-bit modes)
        - B: u8, i8 (high byte of accumulator, 8-bit only in m8 mode)
        - X: u16, i16 (index register, always 16-bit in R65)
        - Y: u16, i16 (index register, always 16-bit in R65)
        - D: u16 (direct page register, 16-bit)
        - S: u16 (stack pointer, 16-bit)
        - DBR: u8 (data bank register, 8-bit)
        - PBR: u8 (program bank register, 8-bit, read-only)
        """
        if bound_type is None:
            return

        if isinstance(bound_type, PointerTypeInfo):
            if bound_type.is_far:
                raise HIRError(
                    f"{context.capitalize()} '{name}' bound to register {register_name} "
                    f"has type {bound_type}, but far pointers are 24-bit and cannot fit in any register",
                    source_loc=None
                )
            # Near pointers are 16-bit, treat like u16 for register validation
            type_name = 'u16'
        elif not isinstance(bound_type, BasicTypeInfo):
            raise HIRError(
                f"{context.capitalize()} '{name}' bound to register {register_name} "
                f"must have a primitive type, got {bound_type}",
                source_loc=None
            )
        else:
            type_name = bound_type.name

        # Define allowed types for each register
        register_allowed_types = {
            'A': ('u8', 'i8', 'u16', 'i16'),
            'B': ('u8', 'i8'),
            'X': ('u16', 'i16'),
            'Y': ('u16', 'i16'),
            'D': ('u16',),
            'S': ('u16',),
            'DBR': ('u8',),
            'PBR': ('u8',),
        }

        allowed = register_allowed_types.get(register_name)
        if allowed is None:
            raise HIRError(
                f"{context.capitalize()} '{name}' bound to unknown register '{register_name}'",
                source_loc=None
            )

        if type_name not in allowed:
            # Provide helpful error messages based on register
            if register_name in ('X', 'Y'):
                hint = (
                    f"In R65, index registers X and Y are always 16-bit.\n"
                    f"  Change the {context} type to u16: {name} @ {register_name}: u16"
                )
            elif register_name == 'B':
                hint = (
                    f"The B register is the high byte of accumulator A in 8-bit mode.\n"
                    f"  Change the {context} type to u8 or i8: {name} @ B: u8"
                )
            elif register_name in ('D', 'S'):
                hint = (
                    f"The {register_name} register is always 16-bit.\n"
                    f"  Change the {context} type to u16: {name} @ {register_name}: u16"
                )
            elif register_name in ('DBR', 'PBR'):
                hint = (
                    f"The {register_name} register is an 8-bit bank register.\n"
                    f"  Change the {context} type to u8: {name} @ {register_name}: u8"
                )
            else:
                hint = f"Allowed types for {register_name}: {', '.join(allowed)}"

            raise HIRError(
                f"{context.capitalize()} '{name}' bound to {register_name} register "
                f"has type {type_name}, but {register_name} only supports: {', '.join(allowed)}",
                source_loc=None,
                hint=hint
            )

    def _validate_no_duplicate_register_bindings(
        self,
        params: list,
        func_name: str
    ) -> None:
        """
        Validate that no register or variable is bound to multiple parameters.

        Example of invalid code:
            fn bad(a @ A: u8, b @ A: u8) { }      // Error: A bound twice
            fn bad(a @ TEMP: u8, b @ TEMP: u8) { } // Error: TEMP bound twice
        """
        register_bindings: dict[str, str] = {}  # register_name -> param_name
        variable_bindings: dict[str, str] = {}  # variable_name -> param_name

        for param in params:
            if param.binding and isinstance(param.binding, hir.RegisterBinding):
                reg_name = param.binding.register_name
                if reg_name in register_bindings:
                    first_param = register_bindings[reg_name]
                    raise HIRError(
                        f"Register {reg_name} is bound to multiple parameters in function '{func_name}': "
                        f"'{first_param}' and '{param.name}'",
                        source_loc=None,
                        hint=f"Each hardware register can only be bound to one parameter per function.\n"
                             f"Consider using stack parameters for additional values."
                    )
                register_bindings[reg_name] = param.name
            elif param.binding and isinstance(param.binding, hir.VariableBinding):
                var_name = param.binding.variable_name
                if var_name in variable_bindings:
                    first_param = variable_bindings[var_name]
                    raise HIRError(
                        f"Variable {var_name} is bound to multiple parameters in function '{func_name}': "
                        f"'{first_param}' and '{param.name}'",
                        source_loc=None,
                        hint=f"Each variable can only be bound to one parameter per function.\n"
                             f"Use different variables or stack parameters for additional values."
                    )
                variable_bindings[var_name] = param.name

    def _validate_static_storage(
        self,
        static: ast.StaticDecl,
        storage_attr: Optional[StorageAttribute]
    ) -> Optional[StorageAttribute]:
        """Validate storage class based on mutability. Returns None for ROM."""
        if static.is_mut:
            # Mutable: must have explicit storage attribute
            if storage_attr is None:
                raise HIRError(
                    f"mutable static '{static.name}' requires explicit storage attribute",
                    source_loc=static.source_loc,
                    hint="add #[zeropage], #[lowram], #[ram], or #[hw(addr)]"
                )
            return storage_attr
        else:
            # Immutable: no storage attr = ROM, #[hw] allowed for read-only regs
            if storage_attr is None:
                return None  # Signals ROM (no StorageKind.ROM exists)
            if storage_attr.storage_kind == StorageKind.HW:
                return storage_attr  # Read-only HW registers OK
            # Any RAM-type storage on immutable is an error
            raise HIRError(
                f"immutable static '{static.name}' cannot use #{storage_attr.storage_kind.value} storage",
                source_loc=static.source_loc,
                hint="add 'mut' to make mutable, or remove attribute for ROM"
            )

    def _build_static(self, static: ast.StaticDecl) -> hir.HIRStaticDecl:
        """Build HIR static declaration from AST."""
        if static.is_extern:
            return self._build_extern_static(static)
        # Process attributes
        processed_attrs = self.attr_processor.process_attributes(
            static.attributes,
            context='static'
        )

        raw_storage_attr = None
        for attr in processed_attrs:
            if isinstance(attr, StorageAttribute):
                raw_storage_attr = attr

        # Validate storage class based on mutability
        storage_attr = self._validate_static_storage(static, raw_storage_attr)

        # Resolve type (or use inferred type from pass 1 for include_bytes!)
        if static.var_type is not None:
            var_type = self.type_resolver.resolve_type(static.var_type)
        else:
            # Type was inferred in pass 1 - retrieve from symbol table
            var_type = self.symbol_table.lookup(static.name).var_type

        # Reject include_bytes! on mutable (RAM) statics — no runtime copy support
        if isinstance(static.initializer, ast.IncludeBytesExpr) and static.is_mut:
            raise HIRError(
                f"include_bytes! cannot be used with 'static mut' (RAM variables)",
                source_loc=static.source_loc,
                hint=f"remove 'mut' to place in ROM: static {static.name} = include_bytes!(...)"
            )

        # Build initializer if present
        initializer = None
        if static.initializer and storage_attr and storage_attr.storage_kind == StorageKind.HW:
            self.warnings.append(
                f"volatile hardware register '{static.name}' cannot be initialized at startup; "
                f"initializer ignored"
            )
        elif static.initializer:
            initializer = self._build_expression(static.initializer)

        # Get static symbol
        static_symbol = self.symbol_table.lookup(static.name)

        # Bank applies only to ROM statics (storage_attr=None means ROM)
        bank_attr = None
        if storage_attr is None:  # ROM static
            if self.auto_bank_mode:
                bank_attr = BankAttribute(name='bank', bank_number=None)
                # Validate: ROM statics in auto-bank mode must be far
                if not static.is_far:
                    raise HIRError(
                        f"ROM static '{static.name}' in auto-bank mode must be declared as 'far static'",
                        source_loc=static.source_loc,
                        hint="use 'far static " + static.name + ": ...' or place in explicit bank with #[bank(n)]"
                    )
            else:
                bank_attr = BankAttribute(name='bank', bank_number=self.current_bank)

        # Create HIR node
        hir_static = hir.HIRStaticDecl(
            name=static.name,
            is_mutable=static.is_mut,
            var_type=var_type,
            initializer=initializer,
            storage_attr=storage_attr,
            bank_attr=bank_attr,
            symbol=static_symbol,
            source_loc=static.source_loc
        )

        # Update symbol's definition to point to HIR node (not AST node)
        static_symbol.definition = hir_static

        return hir_static

    def _build_extern_static(self, static: ast.StaticDecl) -> hir.HIRStaticDecl:
        """Build HIR for an `extern static` declaration.

        Carries the declared type so address-of, indexing, and pointer math
        type-check, but never gets allocated or emitted as data — the bytes
        live in an asm file included via `include_asm!`. Storage attributes
        (`#[ram]`, `#[zeropage]`, etc.) are not permitted: the asm file owns
        placement.
        """
        if static.attributes:
            raise HIRError(
                f"extern static '{static.name}' cannot have storage attributes "
                "— placement is owned by the included asm file",
                source_loc=static.source_loc,
            )
        if static.var_type is None:
            raise HIRError(
                f"extern static '{static.name}' requires a type annotation",
                source_loc=static.source_loc,
            )
        var_type = self.type_resolver.resolve_type(static.var_type)
        static_symbol = self.symbol_table.lookup(static.name)
        # The symbol resolves to a bare label in the included asm file —
        # codegen's label-based ROM addressing path is the right vehicle for
        # this whether the data is mutable or not.
        static_symbol.rom_label = static.name

        hir_static = hir.HIRStaticDecl(
            name=static.name,
            is_mutable=static.is_mut,
            is_extern=True,
            var_type=var_type,
            initializer=None,
            storage_attr=None,
            bank_attr=None,
            symbol=static_symbol,
            source_loc=static.source_loc,
        )
        static_symbol.definition = hir_static
        return hir_static

    def _build_const(self, const: ast.ConstDecl) -> hir.HIRConstDecl:
        """Build HIR const declaration from AST."""
        # Resolve type
        const_type = self.type_resolver.resolve_type(const.const_type)

        # Build value expression
        value_expr = self._build_expression(const.value)

        # Get const symbol (already evaluated in pass 1)
        const_symbol = self.symbol_table.lookup(const.name)
        evaluated_value = const_symbol.const_value if const_symbol else None

        return hir.HIRConstDecl(
            name=const.name,
            const_type=const_type,
            value=value_expr,
            evaluated_value=evaluated_value,
            symbol=const_symbol,
            source_loc=const.source_loc
        )

    def _build_struct(self, struct: ast.StructDecl) -> hir.HIRStructDecl:
        """Build HIR struct declaration from AST."""
        # Build fields with offsets
        hir_fields = []
        has_type_id = struct.name in self._struct_type_ids

        if has_type_id:
            # Insert synthetic __type_id: u8 field at offset 0
            type_id_field = hir.HIRStructField(
                name='__type_id',
                field_type=BasicTypeInfo(name='u8'),
                offset=0
            )
            hir_fields.append(type_id_field)
            current_offset = 1  # User fields start after TypeId byte
        else:
            current_offset = 0

        for field in struct.fields:
            field_type = self.type_resolver.resolve_type(field.field_type)

            hir_field = hir.HIRStructField(
                name=field.name,
                field_type=field_type,
                offset=current_offset
            )
            hir_fields.append(hir_field)

            # Calculate offset for next field (packed layout)
            current_offset += self._get_type_size(field_type)

        # Get struct symbol
        struct_symbol = self.symbol_table.lookup(struct.name)

        hir_struct = hir.HIRStructDecl(
            name=struct.name,
            fields=hir_fields,
            symbol=struct_symbol
        )

        # Update symbol to point to HIR definition (not AST)
        # This ensures type checking uses HIR types (BasicTypeInfo)
        # instead of AST types (BasicType) for field access
        struct_symbol.definition = hir_struct

        return hir_struct

    def _build_enum(self, enum: ast.EnumDecl) -> hir.HIREnumDecl:
        """Build HIR enum declaration from AST."""
        # Build variants with resolved values
        hir_variants = []
        current_value = 0
        max_value = 0

        for variant in enum.variants:
            if variant.value is not None:
                current_value = self.const_evaluator.eval(variant.value)

            # Get variant symbol
            variant_symbol = self.symbol_table.lookup(f"{enum.name}::{variant.name}")

            hir_variant = hir.HIREnumVariant(
                name=variant.name,
                value=current_value,
                symbol=variant_symbol
            )
            hir_variants.append(hir_variant)

            max_value = max(max_value, current_value)
            current_value += 1

        # Infer underlying type (u8 or u16)
        if max_value <= 255:
            underlying_type = BasicTypeInfo(name='u8')
        else:
            underlying_type = BasicTypeInfo(name='u16')

        # Get enum symbol
        enum_symbol = self.symbol_table.lookup(enum.name)

        hir_enum = hir.HIREnumDecl(
            name=enum.name,
            variants=hir_variants,
            underlying_type=underlying_type,
            symbol=enum_symbol
        )

        # Update symbol to point to HIR definition (not AST)
        enum_symbol.definition = hir_enum

        return hir_enum

    def _build_type_alias(self, alias: ast.TypeAlias) -> hir.HIRTypeAlias:
        """Build HIR type alias from AST."""
        # Resolve aliased type
        aliased_type = self.type_resolver.resolve_type(alias.aliased_type)

        # Get alias symbol
        alias_symbol = self.symbol_table.lookup(alias.name)

        hir_alias = hir.HIRTypeAlias(
            name=alias.name,
            aliased_type=aliased_type,
            symbol=alias_symbol
        )

        # Update symbol to point to HIR definition (not AST)
        alias_symbol.definition = hir_alias

        return hir_alias

    def _declare_impl(self, impl: ast.ImplDecl):
        """First pass: declare impl block methods and constants."""
        # Validate struct exists
        struct_symbol = self.symbol_table.lookup(impl.struct_name)
        if not struct_symbol:
            raise HIRError(
                f"impl block for undefined struct: {impl.struct_name}",
                source_loc=impl.source_loc
            )
        if struct_symbol.kind != SymbolKind.STRUCT:
            raise HIRError(
                f"impl block target '{impl.struct_name}' is not a struct",
                source_loc=impl.source_loc
            )

        # Handle trait impl validation and registration
        if impl.trait_name is not None:
            self._declare_trait_impl(impl)
            return

        # Declare associated constants with qualified names: StructName::CONSTANT
        for const in impl.constants:
            qualified_name = f"{impl.struct_name}::{const.name}"
            const_type = self.type_resolver.resolve_type(const.const_type)
            const_value = self.const_evaluator.eval(const.value)

            symbol = Symbol(
                name=qualified_name,
                kind=SymbolKind.IMPL_CONST,
                definition=const,
                scope_id=0,
                var_type=const_type,
                const_value=const_value
            )
            self.symbol_table.declare(qualified_name, symbol)

        # Declare methods with mangled names: StructName__method
        for method in impl.methods:
            mangled_name = f"{impl.struct_name}__{method.name}"

            # Create function symbol for method
            symbol = Symbol(
                name=mangled_name,
                kind=SymbolKind.METHOD,
                definition=method,
                scope_id=0
            )
            self.symbol_table.declare(mangled_name, symbol)

            # Also register method lookup: struct_name + method_name -> mangled_name
            # This allows type checker to resolve method calls
            method_key = f"{impl.struct_name}.{method.name}"
            method_info_symbol = Symbol(
                name=method_key,
                kind=SymbolKind.METHOD,
                definition=method,
                scope_id=0,
                # Store impl block info for self type resolution
                type_info={
                    'struct_name': impl.struct_name,
                    'mangled_name': mangled_name,
                    'impl_is_far': impl.is_far,
                    'method_self_is_far': method.self_is_far
                }
            )
            self.symbol_table.declare(method_key, method_info_symbol)

    def _declare_trait_impl(self, impl: ast.ImplDecl):
        """First pass: declare trait implementation methods and validate against trait definition."""
        trait_symbol = self.symbol_table.lookup(impl.trait_name)
        if not trait_symbol:
            raise HIRError(
                f"impl block for undefined trait: {impl.trait_name}",
                source_loc=impl.source_loc
            )
        if trait_symbol.kind != SymbolKind.TRAIT:
            raise HIRError(
                f"'{impl.trait_name}' is not a trait",
                source_loc=impl.source_loc
            )

        trait_ast = trait_symbol.definition  # ast.TraitDecl

        # Determine if trait methods are far or near
        trait_is_far = any(m.is_far for m in trait_ast.methods) if trait_ast.methods else False

        # Validate near/far consistency for this struct
        kind = 'far' if trait_is_far else 'near'
        if impl.struct_name in self._struct_trait_kind:
            existing_kind = self._struct_trait_kind[impl.struct_name]
            if existing_kind != kind:
                raise HIRError(
                    f"struct '{impl.struct_name}' cannot implement both near and far traits",
                    source_loc=impl.source_loc,
                    hint=f"previously implemented a {existing_kind} trait"
                )
        self._struct_trait_kind[impl.struct_name] = kind

        # Validate all trait methods are implemented
        trait_method_names = {m.name for m in trait_ast.methods}
        impl_method_names = {m.name for m in impl.methods}
        missing = trait_method_names - impl_method_names
        if missing:
            raise HIRError(
                f"impl '{impl.trait_name}' for '{impl.struct_name}' is missing methods: {', '.join(sorted(missing))}",
                source_loc=impl.source_loc
            )

        # Validate all trait constants are provided
        trait_const_names = {c.name for c in trait_ast.constants}
        impl_const_names = {c.name for c in impl.constants}
        missing_consts = trait_const_names - impl_const_names
        if missing_consts:
            raise HIRError(
                f"impl '{impl.trait_name}' for '{impl.struct_name}' is missing constants: {', '.join(sorted(missing_consts))}",
                source_loc=impl.source_loc
            )

        # Assign TypeId only for traits used with *dyn (dynamic dispatch).
        # TypeId injection inserts __type_id: u8 at offset 0, changing struct layout.
        # Traits used only for static dispatch (direct method calls) must not alter layout.
        if impl.trait_name in self._dyn_used_traits and impl.struct_name not in self._struct_type_ids:
            type_id = self._next_type_id
            self._next_type_id += 1
            self._struct_type_ids[impl.struct_name] = type_id

            # Register StructName::TYPE_ID as an associated constant
            type_id_name = f"{impl.struct_name}::TYPE_ID"
            type_id_symbol = Symbol(
                name=type_id_name,
                kind=SymbolKind.IMPL_CONST,
                definition=None,
                scope_id=0,
                var_type=BasicTypeInfo(name='u8'),
                const_value=type_id
            )
            self.symbol_table.declare(type_id_name, type_id_symbol)

        # Register trait impl tracking
        if impl.trait_name not in self._trait_impls:
            self._trait_impls[impl.trait_name] = []
        self._trait_impls[impl.trait_name].append(impl.struct_name)

        # Declare associated constants with qualified names
        for const in impl.constants:
            qualified_name = f"{impl.struct_name}::{const.name}"
            const_type = self.type_resolver.resolve_type(const.const_type)
            const_value = self.const_evaluator.eval(const.value)

            symbol = Symbol(
                name=qualified_name,
                kind=SymbolKind.IMPL_CONST,
                definition=const,
                scope_id=0,
                var_type=const_type,
                const_value=const_value
            )
            self.symbol_table.declare(qualified_name, symbol)

        # Declare methods with mangled names (same pattern as regular impl)
        for method in impl.methods:
            mangled_name = f"{impl.struct_name}__{method.name}"

            symbol = Symbol(
                name=mangled_name,
                kind=SymbolKind.METHOD,
                definition=method,
                scope_id=0
            )
            self.symbol_table.declare(mangled_name, symbol)

            # Register method lookup for struct.method
            method_key = f"{impl.struct_name}.{method.name}"
            # Check if already registered (from a non-trait impl)
            existing = self.symbol_table.lookup(method_key)
            if existing is None:
                method_info_symbol = Symbol(
                    name=method_key,
                    kind=SymbolKind.METHOD,
                    definition=method,
                    scope_id=0,
                    type_info={
                        'struct_name': impl.struct_name,
                        'mangled_name': mangled_name,
                        'impl_is_far': trait_is_far,
                        'method_self_is_far': method.self_is_far,
                        'trait_name': impl.trait_name
                    }
                )
                self.symbol_table.declare(method_key, method_info_symbol)

            # Register trait dispatch lookup: TraitName.method.StructName -> mangled_name
            dispatch_key = f"{impl.trait_name}.{method.name}.{impl.struct_name}"
            dispatch_symbol = Symbol(
                name=dispatch_key,
                kind=SymbolKind.METHOD,
                definition=method,
                scope_id=0,
                type_info={'mangled_name': mangled_name}
            )
            self.symbol_table.declare(dispatch_key, dispatch_symbol)

    def _build_trait(self, trait: ast.TraitDecl) -> hir.HIRTraitDecl:
        """Build HIR trait declaration from AST."""
        hir_methods = []
        for method in trait.methods:
            # Build parameter types
            hir_params = []
            for param in method.params:
                if param.binding is not None:
                    raise HIRError(
                        f"Trait method '{method.name}' parameter '{param.name}' "
                        f"cannot have a register binding — trait methods use stack-passing for dynamic dispatch",
                        source_loc=method.source_loc
                    )
                param_type = self.type_resolver.resolve_type(param.param_type)
                hir_param = hir.HIRParameter(
                    name=param.name,
                    param_type=param_type
                )
                hir_params.append(hir_param)

            ret_type = None
            if method.return_type:
                ret_type = self.type_resolver.resolve_type(method.return_type)

            hir_method = hir.HIRTraitMethod(
                is_far=method.is_far,
                name=method.name,
                self_is_far=method.self_is_far,
                params=hir_params,
                return_type=ret_type
            )
            hir_methods.append(hir_method)

        hir_constants = []
        for const in trait.constants:
            const_type = self.type_resolver.resolve_type(const.const_type)
            hir_const = hir.HIRTraitConst(
                name=const.name,
                const_type=const_type
            )
            hir_constants.append(hir_const)

        trait_symbol = self.symbol_table.lookup(trait.name)

        hir_trait = hir.HIRTraitDecl(
            name=trait.name,
            methods=hir_methods,
            constants=hir_constants,
            symbol=trait_symbol
        )

        # Update symbol to point to HIR definition
        trait_symbol.definition = hir_trait

        return hir_trait

    def _build_impl(self, impl: ast.ImplDecl) -> hir.HIRImplDecl:
        """Build HIR impl block from AST."""
        # Build associated constants
        hir_constants = []
        for const in impl.constants:
            qualified_name = f"{impl.struct_name}::{const.name}"
            const_type = self.type_resolver.resolve_type(const.const_type)
            value_expr = self._build_expression(const.value)
            const_symbol = self.symbol_table.lookup(qualified_name)
            evaluated_value = const_symbol.const_value if const_symbol else None

            hir_const = hir.HIRConstDecl(
                name=qualified_name,
                const_type=const_type,
                value=value_expr,
                evaluated_value=evaluated_value,
                symbol=const_symbol
            )
            hir_constants.append(hir_const)

        # Build methods as functions with mangled names
        hir_methods = []
        for method in impl.methods:
            hir_method = self._build_impl_method(impl, method)
            hir_methods.append(hir_method)

        return hir.HIRImplDecl(
            struct_name=impl.struct_name,
            is_far=impl.is_far,
            methods=hir_methods,
            constants=hir_constants,
            trait_name=impl.trait_name,
            source_loc=impl.source_loc
        )

    def _build_impl_method(self, impl: ast.ImplDecl, method: ast.ImplMethod) -> hir.HIRFunctionDecl:
        """Build HIR function from impl method."""
        mangled_name = f"{impl.struct_name}__{method.name}"

        # Process attributes
        processed_attrs = self.attr_processor.process_attributes(
            method.attributes,
            context='function'
        )

        # Extract specific attributes
        attrs = self._extract_attributes(processed_attrs)
        mode_attr = attrs['mode']
        preserves_attr = attrs['preserves']
        interrupt_attr = attrs['interrupt']
        inline_attr = attrs['inline']
        is_entry = attrs['is_entry']

        # Bank comes from current bank context
        if self.auto_bank_mode:
            bank_attr = BankAttribute(name='bank', bank_number=None)
            if not method.is_far:
                raise HIRError(
                    f"method '{impl.struct_name}::{method.name}' in auto-bank mode must be declared as 'far fn'",
                    source_loc=method.source_loc,
                    hint=f"use 'far fn {method.name}(...)' or place in explicit bank with #[bank(n)]"
                )
        else:
            bank_attr = BankAttribute(name='bank', bank_number=self.current_bank)

        # Enter function scope
        func_scope_id = self.symbol_table.enter_scope(ScopeKind.FUNCTION)

        # Determine self pointer type
        # impl far StructName -> far *self
        # impl StructName -> *self (near)
        # Method can override with explicit far *self or near *self
        self_is_far = impl.is_far or method.self_is_far

        # Build self parameter
        struct_type = StructTypeInfo(name=impl.struct_name)
        self_ptr_type = PointerTypeInfo(pointee_type=struct_type, is_far=self_is_far)

        self_param_symbol = Symbol(
            name='self',
            kind=SymbolKind.PARAMETER,
            definition=method,
            scope_id=self.symbol_table.current_scope_id,
            var_type=self_ptr_type,
            is_mutable=True
        )
        self.symbol_table.declare('self', self_param_symbol)

        hir_self_param = hir.HIRParameter(
            name='self',
            param_type=self_ptr_type,
            binding=None,  # Self is always stack-passed
            symbol=self_param_symbol
        )

        # Process additional parameters
        hir_params = [hir_self_param]
        for param in method.params:
            hir_param = self._build_parameter(param)
            hir_params.append(hir_param)

        # Process body
        hir_body = self._build_block(method.body)

        # Add implicit return A if needed
        self._add_implicit_return(hir_body, method.return_type, interrupt_attr)

        # Exit function scope
        self.symbol_table.exit_scope()

        # Infer entry mode from parameters first (MultiReturnType needs m_mode)
        entry_m_mode = self._infer_entry_mode_and_validate(hir_params, mangled_name, method.source_loc)

        # Resolve return type
        ret_type = None
        if method.return_type:
            ret_type = self._resolve_function_return_type(method.return_type, entry_m_mode)

        # Get method symbol
        method_symbol = self.symbol_table.lookup(mangled_name)

        # Detect STATUS flag return pattern
        returns_status_flag = self._detect_status_flag_return(hir_body)

        # Auto-detect trivial getters/setters and mark for inlining
        # Only if not already marked and not a far/interrupt/entry method
        if (inline_attr is None and
            not method.is_far and
            interrupt_attr is None and
            not is_entry and
            self._is_trivial_getter_or_setter(hir_body)):
            inline_attr = InlineAttribute(name='inline')

        # Infer exit mode from return type
        exit_m_mode = self._infer_exit_mode(ret_type)

        # Eagerly validate const fn body at definition time
        if method.is_const:
            self.const_evaluator.validate_const_fn(method, mangled_name)

        return hir.HIRFunctionDecl(
            name=mangled_name,
            is_far=method.is_far,
            is_const=method.is_const,
            is_trait_method=bool(impl.trait_name),
            parameters=hir_params,
            return_type=ret_type,
            body=hir_body,
            mode_attr=mode_attr,
            preserves_attr=preserves_attr,
            bank_attr=bank_attr,
            interrupt_attr=interrupt_attr,
            inline_attr=inline_attr,
            is_entry=is_entry,
            symbol=method_symbol,
            returns_status_flag=returns_status_flag,
            entry_m_mode=entry_m_mode,
            exit_m_mode=exit_m_mode,
            source_loc=method.source_loc
        )

    # =========================================================================
    # Build Statements
    # =========================================================================

    def _build_block(self, block: ast.Block) -> hir.HIRBlock:
        """Build HIR block from AST."""
        # Enter block scope
        block_scope_id = self.symbol_table.enter_scope(ScopeKind.BLOCK)

        # Build statements
        hir_stmts = []
        for stmt in block.statements:
            hir_stmt = self._build_statement(stmt)
            hir_stmts.append(hir_stmt)

        # Exit block scope
        self.symbol_table.exit_scope()

        return hir.HIRBlock(
            statements=hir_stmts,
            scope_id=block_scope_id,
            source_loc=block.source_loc
        )

    def _build_statement(self, stmt: ast.Statement) -> hir.HIRStatement:
        """Build HIR statement from AST."""
        if isinstance(stmt, ast.Block):
            return self._build_block(stmt)

        elif isinstance(stmt, ast.LetStmt):
            return self._build_let(stmt)

        elif isinstance(stmt, ast.MultiLetStmt):
            return self._build_multi_let(stmt)

        elif isinstance(stmt, ast.ExprStmt):
            return hir.HIRExprStmt(expr=self._build_expression(stmt.expr), source_loc=stmt.source_loc)

        elif isinstance(stmt, ast.ReturnStmt):
            values = [self._build_expression(v) for v in stmt.values]
            return hir.HIRReturnStmt(values=values, source_loc=stmt.source_loc)

        elif isinstance(stmt, ast.BreakStmt):
            value = self._build_expression(stmt.value) if stmt.value else None
            return hir.HIRBreakStmt(label=stmt.label, value=value, source_loc=stmt.source_loc)

        elif isinstance(stmt, ast.ContinueStmt):
            return hir.HIRContinueStmt(label=stmt.label, source_loc=stmt.source_loc)

        elif isinstance(stmt, ast.IfStmt):
            return self._build_if(stmt)

        elif isinstance(stmt, ast.WhileStmt):
            return self._build_while(stmt)

        elif isinstance(stmt, ast.LoopStmt):
            return self._build_loop(stmt)

        elif isinstance(stmt, ast.ForStmt):
            return self._build_for(stmt)

        elif isinstance(stmt, ast.AsmStmt):
            instructions = stmt.instructions
            # Process format string substitution if format_args provided
            if stmt.format_args:
                instructions = self._process_asm_format(instructions, stmt.format_args)
            return hir.HIRAsmStmt(instructions=instructions, source_loc=stmt.source_loc)

        elif isinstance(stmt, ast.ConstAssertStmt):
            # Evaluate the condition at compile time
            try:
                # Build the expression first to resolve identifiers
                hir_condition = self._build_expression(stmt.condition)
                # Now const-evaluate the original AST expression
                result = self.const_evaluator.eval(stmt.condition)
                if not isinstance(result, bool):
                    # Try to treat non-zero as true, zero as false (C-style)
                    if isinstance(result, int):
                        result = result != 0
                    else:
                        raise HIRError(
                            f"const_assert! condition must evaluate to bool, got {type(result).__name__}",
                            stmt.source_loc
                        )
                if not result:
                    raise HIRError(stmt.message, stmt.source_loc)
                # Assertion passed - return a no-op (empty block)
                return hir.HIRBlock(statements=[], source_loc=stmt.source_loc)
            except HIRError:
                raise
            except Exception as e:
                raise HIRError(f"const_assert! condition is not const-evaluable: {e}", stmt.source_loc)

        else:
            raise HIRError(f"Unknown statement type: {type(stmt).__name__}", source_loc=getattr(stmt, 'source_loc', None))

    def _process_asm_format(self, instructions: List[str], format_args: Dict[str, Union[str, int, ast.Expression]]) -> List[str]:
        """
        Process format string substitution in asm! instructions.

        Replaces {name} placeholders with values from format_args.
        String values are inserted directly, integers are formatted as decimal.
        Expressions are const-evaluated to integers.

        Example:
            asm!("LD{REG} #$01", REG="A")  -> "LDA #$01"
            asm!("LDA #{VAL}", VAL=42)     -> "LDA #42"
            asm!("LDA #{LEN}", LEN=buffer.len())  -> "LDA #256" (if buffer is [u8; 256])
        """
        result = []
        # Pattern matches {identifier}
        placeholder_pattern = re.compile(r'\{([A-Za-z_][A-Za-z0-9_]*)\}')

        for instruction in instructions:
            def replace_placeholder(match):
                name = match.group(1)
                if name not in format_args:
                    raise HIRError(f"Unknown format argument '{{{name}}}' in asm! statement", source_loc=None)
                value = format_args[name]
                if isinstance(value, str):
                    return value
                elif isinstance(value, int):
                    # Integer - format as decimal
                    return str(value)
                elif isinstance(value, ast.Expression):
                    # Expression - const-evaluate to integer
                    try:
                        evaluated = self.const_evaluator.eval(value)
                        if not isinstance(evaluated, int):
                            raise HIRError(f"asm! format argument '{name}' must evaluate to an integer, got {type(evaluated).__name__}", source_loc=None)
                        return str(evaluated)
                    except HIRError as e:
                        raise HIRError(f"Cannot const-evaluate asm! format argument '{name}': {e}", source_loc=None)
                else:
                    raise HIRError(f"Invalid asm! format argument type for '{name}': {type(value).__name__}", source_loc=None)

            processed = placeholder_pattern.sub(replace_placeholder, instruction)
            result.append(processed)

        return result

    def _build_let(self, let: ast.LetStmt) -> hir.HIRLetStmt:
        """Build HIR let statement from AST."""
        # Build initializer (may be None for uninitialized variables)
        initializer = None
        if let.initializer is not None:
            initializer = self._build_expression(let.initializer)

        # Resolve type (may be inferred)
        var_type = None
        if let.var_type:
            var_type = self.type_resolver.resolve_type(let.var_type)

        # Process binding
        binding = None
        if let.binding:
            if isinstance(let.binding, ast.Register):
                binding = hir.RegisterLetBinding(register_name=let.binding.name)
            elif isinstance(let.binding, str):
                # Variable binding
                var_symbol = self.symbol_table.lookup(let.binding)
                if not var_symbol:
                    raise HIRError(f"Undefined variable: {let.binding}", source_loc=let.source_loc)
                binding = hir.VariableLetBinding(
                    variable_name=let.binding,
                    variable_symbol=var_symbol
                )

        # Validate register binding types (only if type is explicitly specified)
        if binding and isinstance(binding, hir.RegisterLetBinding) and var_type:
            self._validate_register_binding_type(
                binding.register_name,
                var_type,
                let.name,
                "variable"
            )

        # Declare local variable in current scope
        local_symbol = Symbol(
            name=let.name,
            kind=SymbolKind.LOCAL_VAR,
            definition=let,
            scope_id=self.symbol_table.current_scope_id,
            var_type=var_type,
            is_mutable=let.is_mut
        )
        self.symbol_table.declare(let.name, local_symbol)

        return hir.HIRLetStmt(
            name=let.name,
            is_mutable=let.is_mut,
            var_type=var_type,
            initializer=initializer,
            binding=binding,
            symbol=local_symbol,
            source_loc=let.source_loc
        )

    def _build_multi_let(self, stmt: ast.MultiLetStmt) -> hir.HIRMultiLetStmt:
        """Build HIR multi-let statement: let [mut] a, b = multi_return_func();"""
        initializer = self._build_expression(stmt.initializer) if stmt.initializer is not None else None
        symbols = []

        for name in stmt.names:
            local_symbol = Symbol(
                name=name,
                kind=SymbolKind.LOCAL_VAR,
                definition=stmt,
                scope_id=self.symbol_table.current_scope_id,
                var_type=None,  # Inferred during type checking
                is_mutable=stmt.is_mut
            )
            self.symbol_table.declare(name, local_symbol)
            symbols.append(local_symbol)

        return hir.HIRMultiLetStmt(
            names=stmt.names,
            is_mutable=stmt.is_mut,
            var_types=[],  # Filled during type checking
            initializer=initializer,
            symbols=symbols,
            source_loc=stmt.source_loc
        )

    def _build_if(self, if_stmt: ast.IfStmt) -> hir.HIRIfStmt:
        """Build HIR if statement from AST."""
        condition = self._build_expression(if_stmt.condition)
        then_block = self._build_block(if_stmt.then_block)

        else_block = None
        if if_stmt.else_block:
            if isinstance(if_stmt.else_block, ast.Block):
                else_block = self._build_block(if_stmt.else_block)
            elif isinstance(if_stmt.else_block, ast.IfStmt):
                else_block = self._build_if(if_stmt.else_block)

        return hir.HIRIfStmt(
            condition=condition,
            then_block=then_block,
            else_block=else_block,
            source_loc=if_stmt.source_loc
        )

    def _build_while(self, while_stmt: ast.WhileStmt) -> hir.HIRWhileStmt:
        """Build HIR while statement from AST."""
        condition = self._build_expression(while_stmt.condition)
        body = self._build_block(while_stmt.body)

        return hir.HIRWhileStmt(
            condition=condition,
            body=body,
            is_infinite=False,
            label=while_stmt.label,
            source_loc=while_stmt.source_loc
        )

    def _build_loop(self, loop: ast.LoopStmt) -> hir.HIRWhileStmt:
        """Desugar loop to while true."""
        body = self._build_block(loop.body)

        return hir.HIRWhileStmt(
            condition=hir.HIRBooleanLiteral(value=True),
            body=body,
            is_infinite=True,
            label=loop.label,
            source_loc=loop.source_loc
        )

    def _build_loop_expression(self, loop: ast.LoopExpression) -> hir.HIRLoopExpression:
        """Build HIR loop expression from AST."""
        body = self._build_block(loop.body)

        return hir.HIRLoopExpression(
            body=body,
            label=loop.label,
            source_loc=loop.source_loc
        )

    def _build_for(self, for_stmt: ast.ForStmt) -> hir.HIRBlock:
        """Desugar for loop to while loop.

        for i in start..end { body }
        →
        {
            let mut i = start;
            while i < end {
                body
                i = i + 1;
            }
        }

        Loop variables are given register hints based on nesting depth:
        - Depth 1 (outermost): X register
        - Depth 2 (first nested): Y register
        - Depth 3+: no hint (use scratch/stack)
        """
        src_loc = for_stmt.source_loc

        # Increment loop depth
        self.loop_depth += 1

        # Assign register hint based on depth
        if self.loop_depth == 1:
            register_hint = 'X'  # Outer loop uses X
        elif self.loop_depth == 2:
            register_hint = 'Y'  # Inner loop uses Y
        else:
            register_hint = None  # 3+ nesting: use scratch/stack

        # Enter a new scope for the for-loop block
        self.symbol_table.enter_scope(ScopeKind.BLOCK)

        # Build start and end expressions
        start_expr = self._build_expression(for_stmt.start)
        end_expr = self._build_expression(for_stmt.end)

        # Determine loop variable type from range bounds
        # Try to const-evaluate both bounds to pick appropriate type
        loop_var_type = self._infer_for_loop_type(for_stmt.start, for_stmt.end, src_loc, for_stmt.inclusive)

        # Create symbol for loop variable (mutable, type inferred from range)
        loop_var_symbol = Symbol(
            name=for_stmt.variable,
            kind=SymbolKind.LOCAL_VAR,
            definition=for_stmt,
            scope_id=self.symbol_table.current_scope_id,
            var_type=loop_var_type,
            is_mutable=True,
            register_hint=register_hint  # Pass register hint
        )
        self.symbol_table.declare(for_stmt.variable, loop_var_symbol)

        # Create let statement: let mut i = start;
        let_stmt = hir.HIRLetStmt(
            name=for_stmt.variable,
            is_mutable=True,
            var_type=loop_var_type,
            initializer=start_expr,
            binding=None,
            symbol=loop_var_symbol,
            source_loc=src_loc
        )

        # Create condition: i < end (exclusive) or i <= end (inclusive)
        loop_var_ref = hir.HIRIdentifier(
            name=for_stmt.variable,
            symbol=loop_var_symbol,
            source_loc=src_loc
        )
        condition = hir.HIRBinaryOp(
            op='<=' if for_stmt.inclusive else '<',
            left=loop_var_ref,
            right=end_expr,
            source_loc=src_loc
        )

        # Build the original body statements
        body_statements = []
        for stmt in for_stmt.body.statements:
            body_statements.append(self._build_statement(stmt))

        # Create increment: i = i + 1;
        increment_expr = hir.HIRBinaryOp(
            op='+',
            left=hir.HIRIdentifier(
                name=for_stmt.variable,
                symbol=loop_var_symbol,
                source_loc=src_loc
            ),
            right=hir.HIRIntegerLiteral(value=1, source_loc=src_loc),
            source_loc=src_loc
        )
        increment_stmt = hir.HIRAssignment(
            target=hir.HIRIdentifier(
                name=for_stmt.variable,
                symbol=loop_var_symbol,
                source_loc=src_loc
            ),
            value=increment_expr,
            source_loc=src_loc
        )
        body_statements.append(increment_stmt)

        # Create while body block
        while_body = hir.HIRBlock(
            statements=body_statements,
            source_loc=for_stmt.body.source_loc
        )

        # Create while statement
        while_stmt = hir.HIRWhileStmt(
            condition=condition,
            body=while_body,
            is_infinite=False,
            label=for_stmt.label,
            source_loc=src_loc
        )

        # Exit the for-loop scope
        self.symbol_table.exit_scope()

        # Decrement loop depth
        self.loop_depth -= 1

        # Return block containing let and while
        return hir.HIRBlock(
            statements=[let_stmt, while_stmt],
            source_loc=src_loc
        )

    def _infer_for_loop_type(self, start: ast.Expression, end: ast.Expression, src_loc, inclusive: bool = False) -> TypeInfo:
        """Infer the appropriate type for a for loop variable based on range bounds.

        Examines both start and end values to determine the smallest type that
        can hold all values in the range. Falls back to u16 if bounds cannot
        be const-evaluated.

        Args:
            start: Start expression of the range
            end: End expression of the range (exclusive for .., inclusive for ..=)
            src_loc: Source location for error reporting
            inclusive: True for ..= ranges, False for .. ranges

        Returns:
            TypeInfo for the loop variable (u8, i8, u16, or i16)
        """
        # Try to const-evaluate the bounds
        start_val = self._try_const_eval(start)
        end_val = self._try_const_eval(end)

        # If we can't evaluate bounds, default to u16 (safest choice)
        if start_val is None or end_val is None:
            return BasicTypeInfo('u16')

        # Determine the range of values. The loop var type must hold BOTH the
        # iteration values AND the comparison value (end_val), since the loop
        # condition compares against end_val directly. For exclusive `0..256`,
        # iteration reaches 255 but comparison is `i < 256`, so type must hold 256.
        min_val = min(start_val, end_val)
        max_val = max(start_val, end_val)

        # Check if values fit in each type (prefer unsigned, smallest first)
        if min_val >= 0:
            # Unsigned range
            if max_val <= 255:
                return BasicTypeInfo('u8')
            elif max_val <= 65535:
                return BasicTypeInfo('u16')
        else:
            # Signed range (negative start)
            if min_val >= -128 and max_val <= 127:
                return BasicTypeInfo('i8')
            elif min_val >= -32768 and max_val <= 32767:
                return BasicTypeInfo('i16')

        # Default to u16 if nothing fits
        return BasicTypeInfo('u16')

    def _try_const_eval(self, expr: ast.Expression) -> Optional[int]:
        """Try to const-evaluate an expression, returning None if not possible."""
        try:
            if isinstance(expr, ast.IntegerLiteral):
                return expr.value
            elif isinstance(expr, ast.Identifier):
                # Look up const value if it's a constant
                symbol = self.symbol_table.lookup(expr.name)
                if symbol and symbol.kind == SymbolKind.CONST and hasattr(symbol, 'const_value'):
                    return symbol.const_value
            elif isinstance(expr, ast.BinaryOp):
                left = self._try_const_eval(expr.left)
                right = self._try_const_eval(expr.right)
                if left is not None and right is not None:
                    if expr.op == '+':
                        return left + right
                    elif expr.op == '-':
                        return left - right
                    elif expr.op == '*':
                        return left * right
                    elif expr.op == '/':
                        return left // right if right != 0 else None
                    elif expr.op == '<<':
                        return left << right
                    elif expr.op == '>>':
                        return left >> right
        except Exception:
            pass
        return None

    # =========================================================================
    # Build Expressions
    # =========================================================================

    def _build_expression(self, expr: ast.Expression) -> hir.HIRExpression:
        """Build HIR expression from AST. Delegates to ExpressionBuilder."""
        return self.expression_builder.build_expression(expr)

    # =========================================================================
    # Helpers
    # =========================================================================

    def _extract_attributes(self, processed_attrs: list) -> dict:
        """Extract specific attribute types from processed attributes list.

        Args:
            processed_attrs: List of processed attributes

        Returns:
            Dictionary with keys: mode, preserves, interrupt, is_entry
        """
        result = {
            'mode': None,
            'preserves': None,
            'interrupt': None,
            'inline': None,
            'is_entry': False
        }

        for attr in processed_attrs:
            if isinstance(attr, ModeAttribute):
                result['mode'] = attr
            elif isinstance(attr, PreservesAttribute):
                result['preserves'] = attr
            elif isinstance(attr, InterruptAttribute):
                result['interrupt'] = attr
            elif isinstance(attr, InlineAttribute):
                result['inline'] = attr
            elif isinstance(attr, EntryAttribute):
                result['is_entry'] = True

        return result

    def _add_implicit_return(self, hir_body: hir.HIRBlock, return_type, interrupt_attr=None):
        """Add implicit return A if function doesn't have explicit return.

        Args:
            hir_body: HIR block (function body)
            return_type: AST return type (or None)
            interrupt_attr: Interrupt attribute if this is an interrupt handler
        """
        # Interrupt handlers should not have implicit returns
        # They will get RTI (return from interrupt) instead
        if interrupt_attr:
            # Add explicit empty return for interrupt handlers
            if not hir_body.statements or not isinstance(hir_body.statements[-1], hir.HIRReturnStmt):
                hir_body.statements.append(
                    hir.HIRReturnStmt(values=[])  # Empty return
                )
            return

        if not hir_body.statements:
            # Empty body - add return A
            a_symbol = self.symbol_table.lookup('A')
            hir_body.statements.append(
                hir.HIRReturnStmt(values=[hir.HIRRegister(name='A', symbol=a_symbol)])
            )
            return

        last_stmt = hir_body.statements[-1]

        # Check if last statement is already a return
        if isinstance(last_stmt, hir.HIRReturnStmt):
            return

        # Trailing return expression: if the last statement is an ExprStmt
        # and the function has a return type, convert to return
        if isinstance(last_stmt, hir.HIRExprStmt) and return_type is not None:
            if not isinstance(return_type, ast.NeverType):
                hir_body.statements[-1] = hir.HIRReturnStmt(
                    values=[last_stmt.expr],
                    source_loc=last_stmt.source_loc
                )
                return

        # Add implicit return A (unless return type is !)
        if return_type is None or not isinstance(return_type, ast.NeverType):
            a_symbol = self.symbol_table.lookup('A')
            hir_body.statements.append(
                hir.HIRReturnStmt(values=[hir.HIRRegister(name='A', symbol=a_symbol)])
            )

    def _build_trait_dispatch_info(self) -> Optional[dict]:
        """Build trait dispatch info for codegen (jump tables)."""
        if not self._trait_impls:
            return None

        info = {}
        for trait_name, struct_names in self._trait_impls.items():
            trait_symbol = self.symbol_table.lookup(trait_name)
            trait_def = trait_symbol.definition  # HIRTraitDecl or ast.TraitDecl

            # Get method names from trait definition
            method_names = [m.name for m in trait_def.methods]

            # Determine if trait is far
            is_far = any(m.is_far for m in trait_def.methods) if trait_def.methods else False

            # Build implementor list sorted by TypeId.
            # Only include structs that have a TypeId (i.e., used with *dyn).
            implementors = []
            for struct_name in struct_names:
                if struct_name not in self._struct_type_ids:
                    continue  # static-dispatch only; no dispatch table entry needed
                type_id = self._struct_type_ids[struct_name]
                mangled_names = []
                for method_name in method_names:
                    dispatch_key = f"{trait_name}.{method_name}.{struct_name}"
                    dispatch_sym = self.symbol_table.lookup(dispatch_key)
                    mangled_names.append(dispatch_sym.type_info['mangled_name'])

                implementors.append({
                    'struct': struct_name,
                    'type_id': type_id,
                    'mangled': mangled_names
                })

            # Sort by type_id for correct table indexing
            implementors.sort(key=lambda x: x['type_id'])

            # Skip traits with no *dyn implementors (static dispatch only)
            if not implementors:
                continue

            info[trait_name] = {
                'is_far': is_far,
                'methods': method_names,
                'implementors': implementors
            }

        return info

    def _get_type_size(self, type_info) -> int:
        """Get size of a type in bytes. Delegates to unified_type_utils."""
        from .unified_type_utils import get_unified_type_size
        return get_unified_type_size(type_info, self.symbol_table)

