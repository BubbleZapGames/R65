# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
HIR Builder - transforms AST to HIR with name resolution and desugaring.

Passes: 0 collect `*dyn` traits; 1 declare top-level symbols; 1.5 resolve
tagged-enum payloads and declare plane symbols; 2 build HIR bodies; 3 pad
payload unions and expand planed statics into planes.
"""

import dataclasses
import copy
import re
from enum import Enum
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
from r65.compiler.registers import REGISTERS
from r65.compiler.hir.lang_items import (
    LANG_ITEM_TRAITS, CLONE_TRAIT, CLONE_METHOD, is_lang_item_trait,
)


class SelfForm(Enum):
    """How a method receives its receiver. One form per type, not per method."""
    POINTER = 'pointer'        # struct/union: `*self`, stack-passed (or SELF_Y)
    VALUE_IN_A = 'value_in_a'  # newtype or fieldless enum: the 1-2 byte value rides in A
    INDEX_IN_X = 'index_in_x'  # #[soa] struct, tagged enum, #[soa] payload: a u16 element index rides in X


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
        # Const fn bodies need declared types to wrap values at their width.
        self.const_evaluator.type_resolver = self.type_resolver
        # Forward-referenced consts (see _register_forward_consts)
        self._consts_declared = set()
        self._forward_consts = {}
        self._consts_in_progress = []
        self.attr_processor = AttributeProcessor()
        self.source_file = source_file
        self.source_dir = Path(source_file).parent if source_file else Path.cwd()
        # Include paths for searching (resolved to absolute paths)
        self.include_paths = [Path(p).resolve() for p in (include_paths or [])]
        # Bank context of the declaration being processed; set per item by
        # _items_in_bank_context.
        self.current_bank = 0
        self.auto_bank_mode = False
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
        # Names of `#[soa]` structs, and the single static that backs each.
        # Both are populated in pass 1, so pass 2 sees them complete.
        self._soa_structs: Set[str] = set()
        self._soa_storage: Dict[str, Any] = {}  # struct name -> (Symbol, source_loc)
        # Names of tagged enums, and the single static backing each. A tagged
        # enum is planed exactly like a `#[soa]` struct — see docs/enums.md — so
        # it carries the same one-static rule and the same storage validations.
        # Tagged enum name -> whether every payload struct is `#[soa]`. The key
        # is added in pass 1; the value is False until `_classify_enum_payloads`
        # settles it in pass 1.5, once every struct's attributes are known. An
        # all-#[soa] enum's data plane is a region every variant overlays with
        # one plane per field rather than an array of union elements — see
        # docs/enums.md, "#[soa] Payloads". Mixing is an error, so the value
        # is the whole story.
        self._tagged_enums: Dict[str, bool] = {}
        # enum name -> (pool Symbol, source_loc, plane prefix). The prefix is
        # '' for a pool of the enum itself and 'field__' when the enum is a
        # field of an #[soa] element, which is what names `units__kind__tag`.
        self._enum_storage: Dict[str, Any] = {}
        # Filled by `_setup_tagged_enums` between the passes.
        self._enum_payloads: Dict[str, Dict[str, Any]] = {}  # enum -> variant -> StructTypeInfo
        self._enum_payload_unions: Dict[str, Any] = {}  # enum -> HIRStructDecl
        # `#[soa]` struct name -> (enum name, variant name) for a struct that is
        # a payload. Such a struct's one storage is the enum's, so its methods
        # index the enum's pool and its fields are the variant's planes.
        self._payload_structs: Dict[str, tuple] = {}
        # The expression builder answers the same questions while building
        # bodies; share the objects rather than copying, so it always sees what
        # passes 1 and 1.5 filled in.
        self.expression_builder.enum_storage = self._enum_storage
        self.expression_builder.tagged_enums = self._tagged_enums
        self.expression_builder.enum_payloads = self._enum_payloads
        self._trait_impls: Dict[str, List[str]] = {}  # trait_name -> implementing struct names, in impl order
        self._trait_impl_locs: Dict[tuple, Any] = {}  # (trait_name, struct_name) -> source_loc
        self._struct_trait_kind: Dict[str, str] = {}  # struct_name -> 'near' or 'far'
        self._dyn_used_traits: Set[str] = set()  # trait names used with *dyn
        self._trait_supertraits: Dict[str, List[str]] = {}  # trait_name -> declared supertrait names

    @staticmethod
    def _stack_attribute(decl: ast.StackDirective) -> StackAttribute:
        """The `#[stack(lower, upper)]` region, which must lie in low RAM."""
        lower, upper = decl.lower, decl.upper
        if lower > upper:
            raise HIRError(f"Stack lower bound ${lower:04X} must be <= upper bound ${upper:04X}",
                           source_loc=decl.source_loc)
        if lower < 0x0000 or upper > 0x1FFF:
            raise HIRError(
                f"Stack region ${lower:04X}-${upper:04X} must be within low RAM ($0000-$1FFF)",
                source_loc=decl.source_loc)
        return StackAttribute(name='stack', lower=lower, upper=upper)

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

        # Map each trait to its declared supertraits (for inheritance validation
        # and supertrait-aware resolution).
        self._trait_supertraits = {
            d.name: list(d.supertraits)
            for d in ast_program.items if isinstance(d, ast.TraitDecl)
        }

        # Pass 0: Scan for *dyn Trait usages to determine which traits need TypeIds.
        # TypeId injection changes struct layout, so it must only happen for traits
        # actually used with dynamic dispatch (*dyn), not static dispatch.
        #
        # Supertraits are deliberately NOT force-marked dyn-used. A struct implementing a
        # *dyn-used subtrait already gets a TypeId from that subtrait, and
        # _build_trait_dispatch_info builds a supertrait's table/wrapper from any implementor
        # that has a TypeId — so inherited dispatch and upcast work without it. Marking
        # supertraits dyn-used would only inject a TypeId byte (a layout change) into structs
        # that implement *only* a supertrait and are never dynamically dispatched.
        self._dyn_used_traits = self._collect_dyn_traits(ast_program)

        # Consts and const fns may be used above their declaration: register
        # them so const evaluation can reach one that pass 1 has not declared.
        self._register_forward_consts(ast_program)

        # Pass 1: Declare all top-level symbols (filtered by cfg). Statics,
        # consts, newtypes, aliases and impls resolve a type as they are
        # declared, so the types go first and any of them may be named above
        # its declaration: structs, enums and traits by name, then newtypes
        # and aliases, which resolve a type themselves.
        type_groups = ((ast.StructDecl, ast.EnumDecl, ast.TraitDecl),
                       (ast.NewtypeDecl, ast.TypeAlias))
        for group in type_groups:
            for decl in self._items_in_bank_context(ast_program):
                if isinstance(decl, group) and self._should_include_declaration(decl):
                    self._declare_toplevel(decl)
        type_decls = tuple(t for group in type_groups for t in group)
        for decl in self._items_in_bank_context(ast_program):
            if isinstance(decl, type_decls):
                continue
            if isinstance(decl, ast.StackDirective):
                stack_attr = self._stack_attribute(decl)
            elif isinstance(decl, ast.SnesRomDirective):
                snesrom_config = decl
            else:
                if self._should_include_declaration(decl):
                    self._declare_toplevel(decl)

        # Validate trait inheritance now that all traits and impls are declared.
        self._validate_trait_hierarchy()
        self._validate_supertrait_impls()

        # Pass 1.5: resolve tagged enum payloads and declare their plane symbols.
        # Between the passes on purpose. Payload resolution needs every struct
        # declared, so it cannot run during pass 1 without forcing the payload
        # struct to be written above the enum; and the plane symbols have to
        # exist before pass 2 builds any function body, because that is where a
        # `match` desugars into reads of those very planes.
        self._setup_tagged_enums(ast_program)

        # Pass 2: Build HIR nodes with resolved references (filtered by cfg)
        hir_decls = []
        for decl in self._items_in_bank_context(ast_program):
            if isinstance(decl, ast.StackDirective):
                continue  # Skip stack directives, already processed
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

        # Pass 3: pad payload unions, then expand every planed static (#[soa]
        # struct or tagged enum) into its planes. Runs after the loop above so
        # every element struct's HIRStructDecl exists, and so the whole program
        # is in hand.
        self._pad_enum_payloads()
        self._expand_soa_statics(hir_decls)
        self._point_type_definitions_at_hir(hir_decls)


        # Build trait dispatch info for codegen
        trait_dispatch_info = self._build_trait_dispatch_info()

        return hir.HIRProgram(
            declarations=hir_decls,
            symbol_table=self.symbol_table,
            stack_attr=stack_attr,
            snesrom_config=snesrom_config,
            trait_dispatch_info=trait_dispatch_info
        )

    def _point_type_definitions_at_hir(self, hir_decls) -> None:
        """Re-point every StructTypeInfo/EnumTypeInfo.definition at its HIR decl.

        A type resolved in pass 1 captured the AST declaration, since that is
        what the symbol held then; pass 2 re-points the symbols but not the
        TypeInfos. Walks the HIR declarations and the symbol table (only HIR
        objects, never into AST nodes) so later passes see one kind.
        """
        hir_mod = ('r65.compiler.hir.nodes', 'r65.compiler.hir.types',
                   'r65.compiler.hir.symbol_table')
        global_symbols = self.symbol_table.scopes[0].symbols
        seen: Set[int] = set()
        stack: list = [hir_decls, list(self.symbol_table.scopes.values())]
        while stack:
            obj = stack.pop()
            if id(obj) in seen:
                continue
            seen.add(id(obj))
            if isinstance(obj, (list, tuple)):
                stack.extend(obj)
                continue
            if isinstance(obj, dict):
                stack.extend(obj.values())
                continue
            if not dataclasses.is_dataclass(obj) or type(obj).__module__ not in hir_mod:
                continue
            if isinstance(obj, (StructTypeInfo, EnumTypeInfo)):
                symbol = obj.symbol or global_symbols.get(obj.name)
                decl = getattr(symbol, 'definition', None)
                if isinstance(decl, (hir.HIRStructDecl, hir.HIREnumDecl)):
                    obj.definition = decl
            for f in dataclasses.fields(obj):
                stack.append(getattr(obj, f.name))

    def _items_in_bank_context(self, program: ast.Program):
        """Yield the program's items other than `#[bank]` directives, with
        `current_bank` / `auto_bank_mode` set to the directive context each
        item is declared in. Declarations start in explicit bank 0; under
        `#[bank(auto)]` the bank is chosen at link time. Passes 1 and 2 both
        walk the items through this, so they see the same bank context."""
        self.current_bank = 0
        self.auto_bank_mode = False
        for decl in program.items:
            if isinstance(decl, ast.BankDirective):
                self.auto_bank_mode = decl.is_auto
                self.current_bank = 0 if decl.is_auto else decl.bank_number
                continue
            yield decl

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
                    name = t.pointee_type.name
                    if is_lang_item_trait(name):
                        raise HIRError(
                            f"'{name}' is a built-in operator/clone trait and cannot be "
                            f"used as a trait object (*dyn {name})",
                            source_loc=getattr(t, 'source_loc', None)
                        )
                    dyn_traits.add(name)
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
        # Get attributes from declaration (only these node types carry them)
        attributes = []
        if isinstance(decl, (ast.FunctionDecl, ast.StaticDecl, ast.ImplMethod,
                             ast.StructDecl, ast.NewtypeDecl)):
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


    # =========================================================================
    # Pass 1: Declare Top-Level Symbols
    # =========================================================================

    def _register_forward_consts(self, program: ast.Program):
        """Hand the const evaluator every top-level const and const fn.

        Pass 1 declares items in source order, so a const whose value uses a
        const or const fn written below it would find nothing. The evaluator
        asks `_declare_forward_const` for a const it cannot find, which
        declares it there and then; a const fn needs no symbol to be run.
        """
        self._consts_declared = set()
        self._forward_consts = {}
        self._consts_in_progress = []
        const_fns = {}
        for decl in program.items:
            if not (isinstance(decl, ast.Declaration) and self._should_include_declaration(decl)):
                continue
            if isinstance(decl, ast.ConstDecl):
                self._forward_consts[decl.name] = decl
            elif isinstance(decl, ast.FunctionDecl) and decl.is_const:
                const_fns[decl.name] = decl
            elif isinstance(decl, ast.ImplDecl):
                for method in decl.methods:
                    if method.is_const:
                        const_fns[f"{decl.struct_name}__{method.name}"] = method
        self.const_evaluator.forward_const_fns = const_fns
        self.const_evaluator.declare_forward_const = self._declare_forward_const

    def _declare_forward_const(self, name: str) -> bool:
        """Declare the const *name* ahead of its turn in pass 1; whether
        there is such a const."""
        decl = self._forward_consts.get(name)
        if name in self._consts_in_progress:
            cycle = " -> ".join(self._consts_in_progress + [name])
            raise HIRError(f"const '{name}' depends on itself: {cycle}",
                           source_loc=getattr(decl, 'source_loc', None))
        if decl is None or name in self._consts_declared:
            return False
        self._declare_toplevel(decl)
        return True

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
            # A bare static of a planed type — a `#[soa]` struct or a tagged
            # enum — is a pool of one; wrap it before the symbol is built so
            # every consumer sees the array form.
            planed_single = (not isinstance(var_type, ArrayTypeInfo)
                             and (self._soa_element_of(var_type) is not None
                                  or self._tagged_enum_of(var_type) is not None))
            if planed_single:
                var_type = self._soa_wrap_single(var_type)
            symbol = Symbol(
                name=decl.name,
                kind=SymbolKind.STATIC_VAR,
                definition=decl,
                scope_id=0,
                var_type=var_type,
                is_mutable=decl.is_mut
            )
            symbol.soa_single = planed_single
            self._bind_soa_storage(decl, var_type, symbol)
            self._bind_enum_storage(decl, var_type, symbol)
            self.symbol_table.declare(decl.name, symbol)

        elif isinstance(decl, ast.ConstDecl):
            if decl.name in self._consts_declared:
                return  # Declared early, on first use (see _declare_forward_const)
            self._consts_declared.add(decl.name)
            # Resolve type
            const_type = self.type_resolver.resolve_type(decl.const_type)

            # Evaluate const value
            self._consts_in_progress.append(decl.name)
            try:
                const_value = self.const_evaluator.eval(decl.value)
            finally:
                self._consts_in_progress.pop()

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
            # Create struct/union type symbol
            if decl.is_union and not decl.fields:
                raise HIRError(
                    f"union '{decl.name}' must declare at least one field",
                    source_loc=decl.source_loc,
                    hint="a union's size is that of its largest field; with no fields it has no layout"
                )
            processed = self.attr_processor.process_attributes(
                decl.attributes, context='struct')
            decl.is_soa = any(isinstance(a, SoaAttribute) for a in processed)
            if decl.is_soa:
                self._validate_soa_struct(decl)
                self._soa_structs.add(decl.name)
            symbol = Symbol(
                name=decl.name,
                kind=SymbolKind.STRUCT,
                definition=decl,
                scope_id=0
            )
            self.symbol_table.declare(decl.name, symbol)

        elif isinstance(decl, ast.NewtypeDecl):
            # Processed for the diagnostic only — a newtype is a scalar, so no
            # attribute currently applies to one.
            self.attr_processor.process_attributes(decl.attributes, context='newtype')
            inner_type = self.type_resolver.resolve_type(decl.inner_type)
            self._validate_newtype_inner(decl.name, inner_type, decl.source_loc)
            symbol = Symbol(
                name=decl.name,
                kind=SymbolKind.NEWTYPE,
                definition=decl,
                scope_id=0,
                type_info=inner_type
            )
            self.symbol_table.declare(decl.name, symbol)

        elif isinstance(decl, ast.EnumDecl):
            # Recorded in pass 1 so a static declared later can ask whether its
            # element type is planed, the way `_soa_structs` is.
            if self._enum_is_tagged(decl):
                self._tagged_enums[decl.name] = False
            # Create enum type symbol
            enum_symbol = Symbol(
                name=decl.name,
                kind=SymbolKind.ENUM,
                definition=decl,
                scope_id=0
            )
            self.symbol_table.declare(decl.name, enum_symbol)

            # Declare variants in global scope with qualified names
            for variant, value in self._enum_discriminants(decl):
                variant_symbol = Symbol(
                    name=f"{decl.name}::{variant.name}",
                    kind=SymbolKind.ENUM_VARIANT,
                    definition=variant,
                    scope_id=0,
                    const_value=value
                )
                # Store with qualified name for resolution
                self.symbol_table.declare(f"{decl.name}::{variant.name}", variant_symbol)

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
        elif isinstance(decl, ast.NewtypeDecl):
            return self._build_newtype(decl)
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

    # Payload types a newtype may wrap: scalars that fit in a register, so the
    # value can ride in A/X/Y and `self` can be passed by value.
    _NEWTYPE_SCALAR_KINDS = (BasicTypeInfo, EnumTypeInfo, PointerTypeInfo, FunctionTypeInfo)

    def _validate_newtype_inner(self, name: str, inner: TypeInfo, source_loc):
        """Reject newtype payloads that are not register-sized scalars."""
        hint = (f"newtypes wrap a scalar of at most 2 bytes; "
                f"use a struct with a named field: 'struct {name} {{ value: ... }}'")

        if isinstance(inner, BasicTypeInfo) and inner.name == 'void':
            raise HIRError(f"newtype '{name}' cannot wrap 'void'",
                           source_loc=source_loc, hint=hint)

        if isinstance(inner, NewtypeTypeInfo):
            raise HIRError(
                f"newtype '{name}' cannot wrap another newtype '{inner.newtype_name}'",
                source_loc=source_loc,
                hint=f"wrap the payload type directly: 'struct {name}({inner.inner});'")

        if not isinstance(inner, self._NEWTYPE_SCALAR_KINDS):
            raise HIRError(
                f"newtype '{name}' wraps '{inner}', which is not a scalar",
                source_loc=source_loc, hint=hint)

        size = inner.size_bytes
        if size > 2:
            raise HIRError(
                f"newtype '{name}' wraps '{inner}', which is {size} bytes",
                source_loc=source_loc, hint=hint)

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

    def _bank_attr_for(self, is_far: bool, not_far_error) -> BankAttribute:
        """The bank a declaration lands in: the current #[bank(n)] context, or
        None (placed at link time) under #[bank(auto)] - where it must not be
        `near`, else *not_far_error()* is raised."""
        if self.auto_bank_mode:
            if not is_far:
                raise not_far_error()
            return BankAttribute(name='bank', bank_number=None)
        return BankAttribute(name='bank', bank_number=self.current_bank)

    def _check_vector_bank(self, name: str, is_entry: bool, interrupt_attr, loc):
        """The reset and interrupt vectors are 16-bit addresses the CPU reads
        in bank $00, so whatever they name must be there. A #[bank] directive
        carries on to every later declaration, the entry too."""
        if (is_entry or interrupt_attr) and (self.auto_bank_mode or self.current_bank != 0):
            where = ("#[bank(auto)]" if self.auto_bank_mode
                     else f"bank {self.current_bank}")
            what = "entry function" if is_entry else "interrupt handler"
            raise HIRError(
                f"{what} '{name}' must be in bank 0, but an earlier #[bank] "
                f"directive places it in {where}",
                source_loc=loc,
                hint="the CPU's vectors point into bank 0: declare it before the "
                     "#[bank] directive, or put #[bank(0)] above it"
            )

    def _check_databank_far(self, name: str, mode_attr, is_far: bool, loc, keyword: str = "far fn"):
        """DBR management saves and restores the bank across a JSL/RTL pair."""
        if mode_attr and mode_attr.databank != DataBankMode.NONE and not is_far:
            raise HIRError(
                f"'{name}' uses databank={mode_attr.databank.value} but is not a far "
                f"function; DBR management requires '{keyword}'.",
                source_loc=loc
            )

    def _signature_modes(self, hir_params, return_type, name: str, loc):
        """(entry_m_mode, return type, exit_m_mode). The entry mode comes first:
        a multi-value return type is laid out by it."""
        entry_m_mode = self._infer_entry_mode_and_validate(hir_params, name, loc)
        ret_type = (self._resolve_function_return_type(return_type, entry_m_mode)
                    if return_type else None)
        return entry_m_mode, ret_type, self._infer_exit_mode(ret_type)

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
        interrupt_attr = attrs['interrupt']
        is_entry = attrs['is_entry']

        if attrs['preserves'] is not None:
            raise HIRError(
                f"extern fn '{func.name}' cannot be declared `#[preserves(...)]`",
                source_loc=func.source_loc,
            )
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

        bank_attr = self._bank_attr_for(func.is_far, lambda: HIRError(
            f"extern fn '{func.name}' in auto-bank mode cannot be `extern near fn`",
            source_loc=func.source_loc))
        self._check_databank_far(func.name, mode_attr, func.is_far, func.source_loc,
                                 keyword="extern far fn")

        self.symbol_table.enter_scope(ScopeKind.FUNCTION)
        try:
            hir_params = [self._build_parameter(p) for p in func.params]
            self._validate_no_duplicate_register_bindings(hir_params, func.name)
        finally:
            self.symbol_table.exit_scope()

        entry_m_mode, ret_type, exit_m_mode = self._signature_modes(
            hir_params, func.return_type, func.name, func.source_loc)

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
            clobbers_attr=attrs['clobbers'],
            bank_attr=bank_attr,
            interrupt_attr=None,
            inline_attr=None,
            is_entry=False,
            allow_lints=attrs['allow'],
            symbol=func_symbol,
            entry_m_mode=entry_m_mode,
            exit_m_mode=exit_m_mode,
            source_loc=func.source_loc,
        )

    def _build_function(self, func: ast.FunctionDecl) -> hir.HIRFunctionDecl:
        """Build HIR function from AST."""
        if func.is_extern:
            return self._build_extern_function(func)
        # Process attributes
        processed_attrs = self.attr_processor.process_attributes(
            func.attributes,
            context='function'
        )

        # Extract specific attributes
        attrs = self._extract_attributes(processed_attrs)
        mode_attr = attrs['mode']
        preserves_attr = attrs['preserves']
        interrupt_attr = attrs['interrupt']
        inline_attr = attrs['inline']
        is_entry = attrs['is_entry']

        self._check_vector_bank(func.name, is_entry, interrupt_attr, func.source_loc)
        bank_attr = self._bank_attr_for(func.is_far, lambda: HIRError(
            f"function '{func.name}' in auto-bank mode cannot be 'near fn'",
            source_loc=func.source_loc,
            hint="drop 'near', or place it in an explicit bank with #[bank(n)]"))
        # The vectors reach the entry and the interrupt handlers without a
        # call, so the far default means nothing for them: keep them near.
        is_far = func.is_far and not is_entry and interrupt_attr is None

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
        self._add_implicit_return(hir_body, func.return_type, interrupt_attr, func.body)

        # Exit function scope
        self.symbol_table.exit_scope()

        entry_m_mode, ret_type, exit_m_mode = self._signature_modes(
            hir_params, func.return_type, func.name, func.source_loc)
        func_symbol = self.symbol_table.lookup(func.name)
        self._check_databank_far(func.name, mode_attr, is_far, func.source_loc)

        inline_attr = self._auto_inline_hint(
            inline_attr, interrupt_attr, is_entry, hir_body)

        # Eagerly validate const fn body at definition time
        if func.is_const:
            self.const_evaluator.validate_const_fn(func, func.name)

        return hir.HIRFunctionDecl(
            name=func.name,
            is_far=is_far,
            is_const=func.is_const,
            parameters=hir_params,
            return_type=ret_type,
            body=hir_body,
            mode_attr=mode_attr,
            preserves_attr=preserves_attr,
            clobbers_attr=attrs['clobbers'],
            bank_attr=bank_attr,
            interrupt_attr=interrupt_attr,
            inline_attr=inline_attr,
            is_entry=is_entry,
            is_used=attrs['is_used'],
            allow_lints=attrs['allow'],
            symbol=func_symbol,
            entry_m_mode=entry_m_mode,
            exit_m_mode=exit_m_mode,
            source_loc=func.source_loc  # Propagate source location from AST
        )

    def _auto_inline_hint(self, inline_attr, interrupt_attr, is_entry: bool,
                          body: hir.HIRBlock):
        """The inline attribute a function or method gets: its own, or else an
        `#[inline]` hint when it is a trivial getter/setter that is neither an
        interrupt handler nor the entry point. Far functions qualify: the
        inliner decides per call site whether the caller's bank makes the body
        safe to splice in (`InlinabilityChecker._bank_compatible`)."""
        if (inline_attr is None and
                interrupt_attr is None and
                not is_entry and
                self._is_trivial_getter_or_setter(body)):
            return InlineAttribute(name='inline')
        return inline_attr

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

        # Cast with a simple operand: `x as u16`, and the newtype wrap/unwrap
        # pair `TileId(v)` / `self.0`, which are pure retypes costing nothing.
        # Without this a newtype accessor is a JSR to a bare RTS.
        if isinstance(expr, hir.HIRTypeCast):
            return self._is_simple_expression(expr.expr, depth + 1)

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

        These get a #[inline] hint (size-gated, not `always`).

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
                    # Check if A parameter is 16-bit. Mode is a question about the
                    # machine width, so a newtype answers for its payload — miss
                    # this and a 2-byte newtype in A silently runs in m8.
                    a_type = strip_newtype(param.param_type) if param.param_type else None
                    if isinstance(a_type, BasicTypeInfo):
                        if a_type.name in ('u16', 'i16'):
                            entry_mode = ModeState.M16
                            a_param = param
                elif reg_name == "B":
                    b_param = param

        # Validate: B register parameter is incompatible with 16-bit A parameter.
        # Only a 16-bit A parameter sets M16, so a_param is set; the message
        # names its declared type (str() — a newtype has no `.name`).
        if entry_mode == ModeState.M16 and b_param is not None:
            a_type_name = str(a_param.param_type)
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

    def _multi_return_to_tuple(self, mrt, entry_m_mode) -> 'MultiReturnTypeInfo':
        """Convert a multi-return type list (e.g. `u8, u16`) to a MultiReturnTypeInfo.

        Values are assigned to hardware registers in order — A, then B (m8 only)
        or X, then Y — by `get_return_registers`. This validates that the value
        count and each value's width are expressible in the chosen registers
        given the function's accumulator mode.
        """
        from r65.compiler.hir.types import MultiReturnTypeInfo
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

        tuple_info = MultiReturnTypeInfo(element_types=element_types)
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
        - If the value returned in A is two bytes wide -> m16 exit
        - Otherwise -> m8 exit (default)

        The question is the *machine width* of what lands in A, not the
        spelling of the type:

        - a newtype answers for its payload, so returning `Q10` leaves the CPU
          in m16 exactly as `i16` does;
        - a near pointer and a near `fn()` pointer are two bytes in A; the
          caller reads this field via `callee_exit_m_mode`, so both sides must
          agree. Far forms are 3 bytes and the type checker rejects them first;
        - an enum is one byte until a variant exceeds 255, and two after.

        A multi-return keeps m8: its values are mapped to A/B/X/Y by
        `get_return_registers`, which owns that decision.

        Args:
            return_type: Function return type (or None for void)

        Returns:
            ModeState for function exit (M8 or M16)
        """
        from r65.compiler.typeck.processor_mode import ModeState
        from r65.compiler.hir.types import (EnumTypeInfo, FunctionTypeInfo,
                                            PointerTypeInfo)

        if return_type is None:
            return ModeState.M8

        payload = strip_newtype(return_type)
        if isinstance(payload, (BasicTypeInfo, PointerTypeInfo, FunctionTypeInfo,
                                EnumTypeInfo)):
            if payload.size_bytes == 2:
                return ModeState.M16

        return ModeState.M8

    def _build_parameter(self, param: ast.Parameter) -> hir.HIRParameter:
        """Build HIR parameter from AST."""
        # Resolve parameter type
        param_type = self.type_resolver.resolve_type(param.param_type)
        self._reject_soa_type(param_type, 'parameter', param.name,
                              getattr(param, 'source_loc', None))

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

        Allowed types come from `registers.REGISTERS[name].types`; the hints
        below explain the common mistakes.
        """
        if bound_type is None:
            return

        # A newtype is its payload in a register — that is the whole point of the
        # type. `bound_type` stays unstripped so diagnostics name what was written.
        payload = strip_newtype(bound_type)

        if isinstance(payload, PointerTypeInfo):
            if payload.is_far:
                raise HIRError(
                    f"{context.capitalize()} '{name}' bound to register {register_name} "
                    f"has type {bound_type}, but far pointers are 24-bit and cannot fit in any register",
                    source_loc=None
                )
            # Near pointers are 16-bit, treat like u16 for register validation
            type_name = 'u16'
        elif isinstance(payload, EnumTypeInfo):
            # An enum is an integer of its own width. `self` in an enum method
            # arrives in A this way (docs/enums.md).
            type_name = 'u16' if payload.size_bytes == 2 else 'u8'
        elif not isinstance(payload, BasicTypeInfo):
            raise HIRError(
                f"{context.capitalize()} '{name}' bound to register {register_name} "
                f"must have a primitive type, got {bound_type}",
                source_loc=None
            )
        else:
            type_name = payload.name

        reg = REGISTERS.get(register_name)
        if reg is None or not reg.bindable:
            raise HIRError(
                f"{context.capitalize()} '{name}' bound to unknown register '{register_name}'",
                source_loc=None
            )
        allowed = reg.types

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
            # Mutable with no explicit attribute → auto-allocated RAM ($7E2000+).
            # Equivalent to writing `#[ram]`: `&` of it is a far pointer and
            # access uses long addressing. An explicit attribute is still the way
            # to get zeropage/lowram speed or near pointers.
            if storage_attr is None:
                return StorageAttribute(
                    name='ram', storage_kind=StorageKind.RAM, address=None)
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

    def _soa_element_of(self, var_type):
        """The `#[soa]` element `StructTypeInfo` of *var_type*, else None.

        Accepts both `[Enemy; N]` and a bare `Enemy` — a single instance is a
        one-element pool (see `_soa_wrap_single`).
        """
        inner = var_type.element_type if isinstance(var_type, ArrayTypeInfo) else var_type
        inner = strip_newtype(inner)
        if not isinstance(inner, StructTypeInfo):
            return None
        return inner if self._type_is_soa(inner) else None

    def _type_is_soa(self, struct_type) -> bool:
        """True if *struct_type* names a `#[soa]` struct.

        Reads the name rather than the definition: `_soa_structs` is filled in
        pass 1, whereas `Symbol.definition` is an `ast.StructDecl` in pass 1 and
        an `HIRStructDecl` in pass 2, and callers run in both.
        """
        return getattr(struct_type, 'name', None) in self._soa_structs

    def _soa_wrap_single_if_soa(self, var_type):
        """Wrap *var_type* into `[T; 1]` only if it is a bare planed type.

        A `#[soa]` struct or a tagged enum — both are pools, so a lone instance
        is a pool of one and every downstream path stays the array path.
        """
        if isinstance(var_type, ArrayTypeInfo):
            return var_type
        if (self._soa_element_of(var_type) is None
                and self._tagged_enum_of(var_type) is None):
            return var_type
        return self._soa_wrap_single(var_type)

    @staticmethod
    def _soa_wrap_single(var_type):
        """Wrap a bare planed type (`#[soa]` struct or tagged enum) into a one-element array.

        `static mut PLAYER: Player;` is a pool of one, so every downstream path
        — plane synthesis, `PLAYER.hp`, `PLAYER.method()` — is the array path
        with index 0, and none of them needs a singleton special case.

        Returns a new type; never mutates the AST, which is shared with the
        preprocessor and would otherwise be wrapped twice on a second build.
        """
        if isinstance(var_type, ArrayTypeInfo):
            return var_type
        return ArrayTypeInfo(element_type=var_type, size=1)

    def _bind_soa_storage(self, decl: ast.StaticDecl, var_type, symbol):
        """Record the single static backing a `#[soa]` struct, or reject a second.

        Called from pass 1, where the struct is guaranteed already declared: a
        static cannot forward-reference its type, because `resolve_type` above
        is eager. So `_soa_structs` is complete for anything this can see.
        """
        element = self._soa_element_of(var_type)
        if element is None:
            return
        self._bind_planed_storage(
            decl, symbol, element.name, self._soa_storage,
            noun=f"#[soa] struct '{element.name}'",
            static_noun="#[soa] static",
            duplicate_hint="a #[soa] struct backs exactly one array — its methods "
                           "index that array's field planes by an address fixed at "
                           "compile time, so a second storage has nowhere to point; "
                           "merge the pools, or drop #[soa] and write the parallel "
                           "arrays by hand",
            allow_initializer=True,
        )

    def _bind_planed_storage(self, decl: ast.StaticDecl, symbol, type_name: str,
                             registry: dict, *, noun: str, static_noun: str,
                             duplicate_hint: str, prefix: str = None,
                             allow_initializer: bool = False) -> None:
        """Record the single static backing a planed type, or reject a second.

        Shared by `#[soa]` structs and tagged enums. Both split their storage
        into planes addressed by a compile-time-fixed label, so both admit
        exactly one backing static and both rule out the storage forms that
        would leave the planes unplaced: `extern` (placement belongs to the asm
        file) and `#[hw]` (a fixed hardware layout).

        An `#[soa]` struct's array literal transposes into one literal per
        plane (`_transpose_soa_initializer`), so with *allow_initializer* the
        static may carry one, and may then be immutable — a ROM table per
        plane. A tagged enum takes no initializer, so its static must be
        `static mut`: immutable, it could only ever be undefined bytes.
        """
        previous = registry.get(type_name)
        if previous is not None:
            # A tagged enum's entry carries a third element, the plane prefix.
            prev_symbol = previous[0]
            raise HIRError(
                f"{noun} is already stored by static '{prev_symbol.name}'",
                source_loc=decl.source_loc,
                hint=duplicate_hint,
            )

        if decl.is_extern:
            raise HIRError(
                f"extern static '{decl.name}' cannot have {noun} element type",
                source_loc=decl.source_loc,
                hint="placement is owned by the included asm file, which has no "
                     "way to express one array per field",
            )
        for attr in decl.attributes:
            if attr.name == 'hw':
                raise HIRError(
                    f"{static_noun} '{decl.name}' cannot use #[hw] storage",
                    source_loc=decl.source_loc,
                    hint="hardware registers have a fixed layout the compiler "
                         "does not place",
                )
        if decl.initializer is not None and not allow_initializer:
            raise HIRError(
                f"{static_noun} '{decl.name}' cannot have an initializer",
                source_loc=decl.source_loc,
                hint="a tagged enum has no literal form; leave the static "
                     "uninitialized and write every element before matching on it",
            )
        if not decl.is_mut and decl.initializer is None:
            raise HIRError(
                f"{static_noun} '{decl.name}' must be `static mut`",
                source_loc=decl.source_loc,
                hint="an immutable static is ROM, so without an initializer it "
                     "could only ever be undefined bytes; add 'mut', or give it "
                     "an array literal to make it a ROM table per plane"
                     if allow_initializer else
                     "an immutable static is ROM, and a tagged enum static cannot "
                     "carry an initializer — so it could only ever be undefined "
                     "bytes; add 'mut'",
            )

        # A tagged enum's entry carries the plane-name prefix as well; an
        # `#[soa]` struct's does not, and passes None.
        registry[type_name] = ((symbol, decl.source_loc) if prefix is None
                               else (symbol, decl.source_loc, prefix))

    def _tagged_enum_of(self, var_type):
        """The tagged `EnumTypeInfo` of *var_type*, else None.

        Accepts `[Shape; N]` and a bare `Shape` alike, mirroring
        `_soa_element_of`: a single instance is a pool of one.
        """
        inner = var_type.element_type if isinstance(var_type, ArrayTypeInfo) else var_type
        inner = strip_newtype(inner)
        if not isinstance(inner, EnumTypeInfo):
            return None
        return inner if inner.name in self._tagged_enums else None

    def _bind_enum_storage(self, decl: ast.StaticDecl, var_type, symbol) -> None:
        """Record the single static backing a tagged enum, or reject a second."""
        element = self._tagged_enum_of(var_type)
        if element is None:
            return
        self._bind_planed_storage(
            decl, symbol, element.name, self._enum_storage, prefix='',
            noun=f"tagged enum '{element.name}'",
            static_noun="tagged enum static",
            duplicate_hint="a tagged enum backs exactly one array — its methods "
                           "index that array's tag and payload planes by an address "
                           "fixed at compile time, so a second storage has nowhere "
                           "to point; merge the pools",
        )

    @staticmethod
    def _soa_stride(size: int) -> int:
        """Bytes per element in a field plane.

        Rounded up to a power of two when that costs at most one byte per
        element: a 3-byte `far *T` scales by `ASL A / ASL A` instead of a spill,
        a shift, a CLC and an ADC. A 5- or 6-byte field would need 3 or 2 bytes
        of padding to reach 8, which is not worth one add, so it keeps the
        shift-and-add.
        """
        return 4 if size == 3 else 8 if size == 7 else size

    def _pad_enum_payloads(self) -> None:
        """Round each payload union up to a power-of-two stride where it is cheap.

        A 3-byte element scales by two ASLs; 3 itself needs a shift, a CLC and an
        ADC. `_soa_stride` draws the same line for `#[soa]` field planes, and
        padding a *union* takes one field, since its size is its largest member.

        Runs in pass 3, the first point at which every payload struct is HIR and
        so has a real size.
        """
        for enum_name, union in self._enum_payload_unions.items():
            natural = max(f.field_type.size_bytes for f in union.fields)
            if self._tagged_enums.get(enum_name, False):
                # The data plane is a region every variant overlays with its
                # field planes, grouped into one lane per stride so planes only
                # overlay planes of their own stride — see _soa_payload_layout.
                # Its per-element size can exceed the packed size, since each
                # plane pads 3 -> 4 and 7 -> 8 on its own.
                _offsets, stride = self._soa_payload_layout(enum_name)
            else:
                stride = self._soa_stride(natural)
            if stride > natural:
                union.fields.append(hir.HIRStructField(
                    name='__pad',
                    field_type=ArrayTypeInfo(element_type=BasicTypeInfo(name='u8'),
                                             size=stride),
                    offset=0))

    def _expand_soa_statics(self, hir_decls: list) -> None:
        """Replace each planed static's storage with one array per plane.

        A `#[soa]` struct contributes one plane per field; a tagged enum
        contributes exactly two, a tag plane and a payload plane. Both are laid
        out identically from here down, which is the point — the plane, its
        stride, its synthetic symbol and its allocation are one mechanism.

        The parent stays in the declaration list — type checking and the MIR
        funnel both resolve through it — but is allocated no bytes; the planes
        it points at hold the data.
        """
        planes_added = []
        for decl in hir_decls:
            if not isinstance(decl, hir.HIRStaticDecl) or not decl.is_soa:
                continue

            count = decl.var_type.size
            explicit = (decl.storage_attr.address
                        if decl.storage_attr is not None else None)
            columns = self._transpose_soa_initializer(decl)
            planes = []
            children = {}
            running = explicit
            for field_decl, aos_offset, overlay in self._plane_fields(decl):
                size = self._get_type_size(field_decl.field_type)
                stride = self._soa_stride(size)
                if overlay is not None:
                    # A variant's field plane over the enum's data region. It
                    # reserves no bytes and is not in `soa_planes` — nothing
                    # reaches it by folded offset, only by name — so it neither
                    # advances the explicit layout nor counts toward a budget.
                    region_name, elem_offset = overlay
                    child = self._make_soa_plane(decl, field_decl, count, stride, None)
                    child.soa_overlay = (children[region_name], count * elem_offset)
                    planes_added.append(child)
                    continue
                child = self._make_soa_plane(decl, field_decl, count, stride, running)
                if columns is not None:
                    child.initializer = hir.HIRArrayLiteralExpr(
                        elements=columns[field_decl.name],
                        source_loc=decl.initializer.source_loc,
                    )
                children[field_decl.name] = child
                planes.append(hir.SoaPlane(field_decl.name, aos_offset, child))
                planes_added.append(child)
                if running is not None:
                    running += count * stride

            decl.soa_planes = planes
            # The planes now carry the data; the parent is allocated no bytes,
            # so nothing may try to initialize it.
            decl.initializer = None

        hir_decls.extend(planes_added)

    def _transpose_soa_initializer(self, decl: 'hir.HIRStaticDecl'):
        """Split an `#[soa]` static's array literal into one column per field.

        `[E { a: 1, b: 2 }, E { a: 3, b: 4 }]` becomes `{a: [1, 3], b: [2, 4]}`;
        each column then initializes that field's plane through the ordinary
        array-literal path, which for a ROM static emits one table per plane
        and for a RAM static one block copy per plane.

        Struct-literal checking runs after the static is built and would see
        only the columns, so the literal's shape is checked here: an element
        per slot, each a literal of the element struct naming every field once.
        (An unknown field name never gets this far — the expression builder
        rejects it.) Field *values* are still type-checked against the plane's
        element type downstream. Returns None when there is no initializer.
        """
        init = decl.initializer
        if init is None:
            return None
        element = strip_newtype(decl.var_type.element_type)
        if isinstance(element, EnumTypeInfo):
            # `_bind_planed_storage` already refuses this; keep the invariant
            # visible here rather than transposing a tag plane by accident.
            raise HIRError(
                f"tagged enum static '{decl.name}' cannot have an initializer",
                source_loc=init.source_loc,
            )
        definition = element.symbol.definition
        field_names = [f.name for f in definition.fields]
        count = decl.var_type.size
        loc = init.source_loc

        # A single instance (`static mut P: One`) is a pool of one, but it was
        # declared as a struct, so it takes the struct literal.
        if getattr(decl.symbol, 'soa_single', False):
            if not isinstance(init, hir.HIRStructLiteralExpr):
                raise HIRError(
                    f"#[soa] static '{decl.name}' is a single `{element.name}`, "
                    f"so it takes a `{element.name} {{ .. }}` literal",
                    source_loc=loc,
                )
            init = hir.HIRArrayLiteralExpr(elements=[init], source_loc=loc)

        if not isinstance(init, hir.HIRArrayLiteralExpr):
            raise HIRError(
                f"#[soa] static '{decl.name}' must be initialized with an "
                f"array literal",
                source_loc=loc,
                hint=f"write one `{element.name} {{ .. }}` per element; a fill "
                     f"`[.. ; N]` has no per-plane form yet",
            )
        if len(init.elements) != count:
            raise HIRError(
                f"#[soa] static '{decl.name}' has {count} element(s) but its "
                f"initializer has {len(init.elements)}",
                source_loc=loc,
            )

        columns = {name: [] for name in field_names}
        for index, elem in enumerate(init.elements):
            if (not isinstance(elem, hir.HIRStructLiteralExpr)
                    or elem.struct_name != element.name):
                raise HIRError(
                    f"element {index} of #[soa] static '{decl.name}' must be a "
                    f"`{element.name} {{ .. }}` literal",
                    source_loc=getattr(elem, 'source_loc', loc),
                )
            seen = set()
            for field_init in elem.fields:
                if field_init.name in seen:
                    raise HIRError(
                        f"field '{field_init.name}' initialized multiple times",
                        source_loc=elem.source_loc,
                        hint="remove duplicate field initialization",
                    )
                seen.add(field_init.name)
                columns[field_init.name].append(field_init.value)
            missing = [name for name in field_names if name not in seen]
            if missing:
                raise HIRError(
                    f"missing field(s) in struct literal: {', '.join(missing)}",
                    source_loc=elem.source_loc,
                    hint=f"add: {', '.join(missing)}",
                )
        return columns

    def _planed_field_size(self, field_type) -> int:
        """Bytes a field occupies in the packed layout the plane offsets index.

        A tagged enum's `size_bytes` is its *discriminant* width — that is what
        the tag plane wants — but as a field it stands for both of its planes,
        tag and payload. Sizing it at the discriminant alone would put the next
        field's offset on top of the payload plane's, and `_soa_plane_for`
        picks a plane by which range contains the summed offset.
        """
        enum = self._tagged_enum_of(field_type)
        if enum is not None:
            definition = enum.symbol.definition if enum.symbol else enum.definition
            payload = getattr(definition, 'payload_union', None)
            if payload is not None:
                return (definition.underlying_type.size_bytes
                        + StructTypeInfo(name=payload.name, definition=payload,
                                         symbol=payload.symbol).size_bytes)
        return self._get_type_size(field_type)

    def _plane_fields(self, decl: 'hir.HIRStaticDecl'):
        """The `(field, aos_offset, overlay)` triples *decl* splits into planes.

        *overlay* is None or `(region plane name, element offset)`.

        `aos_offset` is where the plane would start in a packed
        array-of-structs layout. MIR inverts it by range to find the plane
        owning a folded field chain, so the values only have to be ordered and
        non-overlapping, which the packed layout guarantees.
        """
        element = strip_newtype(decl.var_type.element_type)
        definition = element.symbol.definition if element.symbol else None

        if isinstance(element, EnumTypeInfo):
            if not isinstance(definition, hir.HIREnumDecl):
                raise HIRError(
                    f"tagged enum static '{decl.name}': enum '{element.name}' "
                    f"was never built",
                    source_loc=decl.source_loc,
                )
            return self._enum_plane_pair(definition, '', 0)

        if not isinstance(definition, hir.HIRStructDecl):
            raise HIRError(
                f"#[soa] static '{decl.name}': element struct "
                f"'{element.name}' was never built",
                source_loc=decl.source_loc,
            )

        planes = []
        for f in definition.fields:
            # A field that is itself a tagged enum contributes that enum's two
            # planes rather than one of its own, named for the path that
            # reaches it: `units__kind__tag` and `units__kind__data`. The list
            # stays flat, so everything downstream — the stride rule, the
            # synthetic symbol, the allocation — is the same mechanism it
            # already was for a standalone pool.
            nested = self._tagged_enum_of(f.field_type)
            if nested is not None:
                nested_defn = (nested.symbol.definition if nested.symbol
                               else nested.definition)
                planes.extend(self._enum_plane_pair(
                    nested_defn, f"{f.name}__", f.offset))
            else:
                planes.append((f, f.offset, None))
        return planes

    def _enum_plane_pair(self, definition, prefix: str, base_offset: int):
        """A tagged enum's `[(tag, offset, None), (data, offset, None), ...]`.

        The tag leads so that a tag read is `LDA POOL__tag,X` with the raw index
        in X — no scaling — which is the whole reason for the split. *prefix*
        names the path when the enum is a field of the element rather than the
        element itself, and *base_offset* is where that field starts in the
        packed layout `_soa_plane_for` inverts by range.

        An all-`#[soa]` enum appends one entry per variant field, each carrying
        an overlay `(data plane name, element offset)`: the plane sits over the
        data region at `count * element offset`, the offset coming from
        `_soa_payload_layout` — variants overlay within one lane per stride,
        which is what makes it a union without one element's payload landing
        on another's.
        """
        tag = hir.HIRStructField(
            name=f'{prefix}tag', field_type=definition.underlying_type,
            offset=base_offset)
        payload = definition.payload_union
        data = hir.HIRStructField(
            name=f'{prefix}data',
            field_type=StructTypeInfo(name=payload.name,
                                      definition=payload,
                                      symbol=payload.symbol),
            offset=base_offset + tag.field_type.size_bytes)
        planes = [(tag, tag.offset, None), (data, data.offset, None)]

        if self._tagged_enums.get(definition.name, False):
            offsets, _size = self._soa_payload_layout(definition.name)
            for variant, field_name, ftype in self._soa_payload_fields(definition.name):
                plane = hir.HIRStructField(
                    name=f'{prefix}{variant}__{field_name}', field_type=ftype,
                    offset=data.offset)
                planes.append((plane, data.offset,
                               (data.name, offsets[(variant, field_name)])))
        return planes

    def _declare_plane_symbol(self, name: str, element_type, count: int,
                              is_mutable: bool = True) -> Symbol:
        """Declare the static symbol for one plane: `[element_type; count]`.
        Its `HIRStaticDecl` is built in pass 3 by `_expand_soa_statics`."""
        symbol = Symbol(
            name=name,
            kind=SymbolKind.STATIC_VAR,
            definition=None,
            scope_id=0,
            var_type=ArrayTypeInfo(element_type=element_type, size=count),
            is_mutable=is_mutable,
        )
        self.symbol_table.declare(name, symbol)
        return symbol

    def _make_soa_plane(self, parent, field_decl, count, stride, address):
        """Build one field plane of an `#[soa]` static, and declare its symbol."""
        name = f"{parent.name}__{field_decl.name}"
        plane_type = ArrayTypeInfo(element_type=field_decl.field_type, size=count)

        storage_attr = parent.storage_attr
        if storage_attr is not None:
            storage_attr = StorageAttribute(
                name=storage_attr.name,
                storage_kind=storage_attr.storage_kind,
                address=address,
                is_register=False,
            )

        # A tagged enum's planes were declared in pass 1.5 so function bodies
        # could already name them; adopt that symbol rather than shadowing it,
        # or the body and the allocation would refer to two different objects.
        symbol = self.symbol_table.lookup(name)
        if symbol is not None and symbol.definition is None:
            symbol.var_type = plane_type
            symbol.is_mutable = parent.is_mutable
        else:
            symbol = self._declare_plane_symbol(
                name, field_decl.field_type, count, parent.is_mutable)
        child = hir.HIRStaticDecl(
            name=name,
            is_mutable=parent.is_mutable,
            var_type=plane_type,
            storage_attr=storage_attr,
            bank_attr=parent.bank_attr,
            is_synthetic=True,
            soa_stride=stride,
            symbol=symbol,
            source_loc=parent.source_loc,
        )
        symbol.definition = child
        return child

    def _reject_soa_type(self, var_type, what: str, name: str, source_loc) -> None:
        """Refuse a declaration whose type is (or contains) a planed type.

        A `#[soa]` struct or a tagged-enum value exists only as an element of
        its pool — its bytes live in one array per plane, so there is nothing to
        copy into a local, push as an argument, or embed in another aggregate.
        Callers pass the index.
        """
        from r65.compiler.typeck.type_utils import soa_struct_of

        planed_name = soa_struct_of(var_type)
        if planed_name is None:
            return
        tagged = planed_name in self._tagged_enums
        kind = "tagged enum" if tagged else "#[soa] type"
        if what == 'field' and tagged:
            # There is a way to have this field — the containing struct has to
            # be planed too, so the enum can contribute its own two planes
            # rather than needing a contiguous form inside an element.
            hint = (f"mark the containing struct `#[soa]`: a tagged enum can be "
                    f"a field of a planed element, where it contributes its tag "
                    f"and payload planes")
        else:
            hint = ("a planed value lives only as an element of its pool; hold "
                    "the element index instead, e.g. `let i: u16 = 0;`")
        raise HIRError(
            f"{what} '{name}' cannot have {kind} '{planed_name}'",
            source_loc=source_loc,
            hint=hint,
        )

    def _validate_soa_struct(self, decl: ast.StructDecl) -> None:
        """Reject `#[soa]` on an aggregate that cannot be split into field planes.

        Runs in pass 1 on the AST, so nothing here may consult field *types* —
        they are not resolved yet. Type-dependent rules (a field of SoA type, a
        second backing static) live with the declarations that can see them.
        """
        if decl.is_union:
            raise HIRError(
                f"#[soa] cannot apply to union '{decl.name}'",
                source_loc=decl.source_loc,
                hint="a union's fields overlay at offset 0; splitting them into "
                     "separate arrays would destroy the aliasing that makes it a union",
            )
        if not decl.fields:
            raise HIRError(
                f"#[soa] struct '{decl.name}' has no fields",
                source_loc=decl.source_loc,
                hint="there is nothing to split into planes",
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
        allow_lints = frozenset()
        is_used = False
        for attr in processed_attrs:
            if isinstance(attr, StorageAttribute):
                raw_storage_attr = attr
            elif isinstance(attr, AllowAttribute):
                allow_lints = attr.codes
            elif isinstance(attr, UsedAttribute):
                is_used = True

        # Validate storage class based on mutability
        storage_attr = self._validate_static_storage(static, raw_storage_attr)

        # Resolve type (or use inferred type from pass 1 for include_bytes!)
        if static.var_type is not None:
            var_type = self.type_resolver.resolve_type(static.var_type)
            # Pass 1 wrapped a bare planed type into a one-element array; this
            # re-resolves from the same AST, so wrap it again to match.
            var_type = self._soa_wrap_single_if_soa(var_type)
        else:
            # Type was inferred in pass 1 - retrieve from symbol table
            var_type = self.symbol_table.lookup(static.name).var_type

        # `is_soa` — "this static is split into planes" — is derived from the
        # element type, not from an attribute on the static. A `#[soa]` struct
        # carries the attribute on the struct; a tagged enum is planed by virtue
        # of having a payload variant at all.
        is_soa = (self._soa_element_of(var_type) is not None
                  or self._tagged_enum_of(var_type) is not None)

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
            bank_attr = self._bank_attr_for(static.is_far, lambda: HIRError(
                f"ROM static '{static.name}' in auto-bank mode must be declared as 'far static'",
                source_loc=static.source_loc,
                hint="use 'far static " + static.name + ": ...' or place in explicit bank with #[bank(n)]"))

        # Create HIR node
        hir_static = hir.HIRStaticDecl(
            name=static.name,
            is_mutable=static.is_mut,
            var_type=var_type,
            initializer=initializer,
            storage_attr=storage_attr,
            bank_attr=bank_attr,
            allow_lints=allow_lints,
            is_soa=is_soa,
            is_used=is_used,
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
        """Build HIR struct or union declaration from AST."""
        from .unified_type_utils import layout_fields

        # Build fields with offsets. Unions never carry a TypeId: the byte would sit
        # at offset 0 on top of field data, which is why unions cannot implement
        # traits (see _declare_trait_impl).
        hir_fields = []
        # An #[soa] element has no offset 0 to hold a TypeId, so `*dyn` over one
        # is rejected (docs/soa.md, Known limits).
        has_type_id = (not struct.is_union
                       and not getattr(struct, 'is_soa', False)
                       and struct.name in self._struct_type_ids)

        if has_type_id:
            # Insert synthetic __type_id: u8 field at offset 0
            type_id_field = hir.HIRStructField(
                name='__type_id',
                field_type=BasicTypeInfo(name='u8'),
                offset=0
            )
            hir_fields.append(type_id_field)
            base_offset = 1  # User fields start after TypeId byte
        else:
            base_offset = 0

        field_types = [self.type_resolver.resolve_type(f.field_type) for f in struct.fields]
        offsets, _ = layout_fields(
            [self._planed_field_size(t) for t in field_types],
            struct.is_union
        )

        for field, field_type, offset in zip(struct.fields, field_types, offsets):
            # A tagged enum is allowed as a field of an `#[soa]` struct: the
            # outer static splits into planes anyway, so the enum contributes
            # its own two (see `_plane_fields`) and the tag read stays
            # stride-1. In a plain array-of-structs it would have to be
            # contiguous — a tag byte with the payload inline — which puts the
            # element stride multiply back on every tag read, so it stays
            # rejected there. Only a bare enum: an array of them would need a
            # pair of planes per element.
            if (struct.is_soa and isinstance(field_type, ArrayTypeInfo)
                    and self._tagged_enum_of(field_type) is not None):
                raise HIRError(
                    f"field '{field.name}' of #[soa] struct '{struct.name}' is an "
                    f"array of tagged enum '{field_type.element_type}'",
                    source_loc=struct.source_loc,
                    hint="a tagged enum field contributes one tag plane and one "
                         "payload plane; use a field per element, or give the "
                         "enum its own pool",
                )
            if not (struct.is_soa and self._tagged_enum_of(field_type) is not None):
                self._reject_soa_type(field_type, 'field', field.name,
                                      struct.source_loc)
            hir_fields.append(hir.HIRStructField(
                name=field.name,
                field_type=field_type,
                offset=base_offset + offset
            ))

        # Get struct symbol
        struct_symbol = self.symbol_table.lookup(struct.name)

        hir_struct = hir.HIRStructDecl(
            name=struct.name,
            fields=hir_fields,
            is_union=struct.is_union,
            is_soa=getattr(struct, 'is_soa', False),
            symbol=struct_symbol
        )

        # Update symbol to point to HIR definition (not AST)
        # This ensures type checking uses HIR types (BasicTypeInfo)
        # instead of AST types (BasicType) for field access
        struct_symbol.definition = hir_struct

        return hir_struct

    @staticmethod
    def _enum_is_tagged(enum: ast.EnumDecl) -> bool:
        """True if any variant of *enum* carries a payload (an AST-level query)."""
        return any(v.payload_type is not None for v in enum.variants)

    def _enum_discriminants(self, enum: ast.EnumDecl):
        """Yield `(variant, discriminant)` for every variant, in order.

        Both passes need these numbers (pass 1 for the `Enum::Variant` symbols,
        pass 2 for the HIR variants); one generator keeps the rule, including
        the tagged-enum ban on explicit discriminants, in one place.
        """
        tagged = self._enum_is_tagged(enum)
        current = 0
        for variant in enum.variants:
            if variant.value is not None:
                if tagged:
                    raise HIRError(
                        f"variant '{enum.name}::{variant.name}' cannot have an "
                        f"explicit discriminant",
                        source_loc=enum.source_loc,
                        hint="a tagged enum numbers its variants from 0 so the tag "
                             "indexes a jump table directly; drop the '= ...'",
                    )
                current = self.const_evaluator.eval(variant.value)
            yield variant, current
            current += 1

    def _build_enum(self, enum: ast.EnumDecl) -> hir.HIREnumDecl:
        """Build HIR enum declaration from AST."""
        # Build variants with resolved values
        hir_variants = []
        max_value = 0

        for variant, value in self._enum_discriminants(enum):
            # Get variant symbol
            variant_symbol = self.symbol_table.lookup(f"{enum.name}::{variant.name}")

            payload_type = self._enum_payloads.get(enum.name, {}).get(variant.name)

            hir_variant = hir.HIREnumVariant(
                name=variant.name,
                value=value,
                payload_type=payload_type,
                symbol=variant_symbol
            )
            hir_variants.append(hir_variant)

            max_value = max(max_value, value)

        underlying_type = discriminant_type(max_value)

        # Get enum symbol
        enum_symbol = self.symbol_table.lookup(enum.name)

        hir_enum = hir.HIREnumDecl(
            name=enum.name,
            variants=hir_variants,
            underlying_type=underlying_type,
            symbol=enum_symbol
        )
        hir_enum.payload_union = self._enum_payload_unions.get(enum.name)

        # Update symbol to point to HIR definition (not AST)
        enum_symbol.definition = hir_enum

        return hir_enum

    def _resolve_enum_payload(self, enum: ast.EnumDecl, variant: ast.EnumVariant):
        """Resolve and validate a variant payload, which must name a plain struct.

        The payload becomes a field of the synthetic payload union, so it has to
        be something that can *be* a union field with a stable layout. Each
        rejection below names why the type in particular cannot.
        """
        where = f"payload of '{enum.name}::{variant.name}'"
        ast_type = variant.payload_type

        if not isinstance(ast_type, ast.BasicType):
            raise HIRError(
                f"{where} must name a struct",
                source_loc=enum.source_loc,
                hint="an enum payload is a declared struct — arrays, pointers and "
                     "type lists have no name to match on; wrap it in a struct",
            )

        if ast_type.name in BUILTIN_TYPE_NAMES:
            raise HIRError(
                f"{where} is '{ast_type.name}', not a struct",
                source_loc=enum.source_loc,
                hint=f"a payload must be a named struct so a pattern binding has "
                     f"fields to reach; wrap it, e.g. "
                     f"`struct {variant.name}Data {{ value: {ast_type.name} }}`",
            )

        symbol = self.symbol_table.lookup(ast_type.name)
        if symbol is None:
            raise HIRError(
                f"{where} names undefined type '{ast_type.name}'",
                source_loc=enum.source_loc,
            )
        if symbol.kind != SymbolKind.STRUCT:
            kinds = {
                SymbolKind.NEWTYPE: ("a newtype",
                                     "a newtype is a scalar, so it has no fields to "
                                     "reach through a pattern binding; use its payload "
                                     "type in a struct"),
                SymbolKind.ENUM: ("an enum",
                                  "a nested tag has nowhere to live — the payload "
                                  "plane holds bytes, not a second discriminant"),
                SymbolKind.TRAIT: ("a trait", "traits are unsized"),
            }
            noun, hint = kinds.get(
                symbol.kind, (f"a {symbol.kind.value}", "use a declared struct"))
            raise HIRError(
                f"{where} is {noun}, not a struct",
                source_loc=enum.source_loc,
                hint=hint,
            )

        struct_decl = symbol.definition
        if getattr(struct_decl, 'is_union', False):
            raise HIRError(
                f"{where} is a union, not a struct",
                source_loc=enum.source_loc,
                hint="the payload plane is already a union across variants; "
                     "overlaying a second one inside it hides which read is live",
            )
        # An `#[soa]` payload is allowed, but only if every payload of the
        # enum is: the all-or-nothing check lives in `_setup_tagged_enums`,
        # which can see all the variants at once.
        return self.type_resolver.resolve_named_type(ast_type.name)

    def _setup_tagged_enums(self, ast_program) -> None:
        """Resolve tagged enum payloads and declare their plane symbols.

        Runs between the declaration pass and the body pass. See the call site
        for why neither pass can host it.
        """
        for decl in ast_program.items:
            if not isinstance(decl, ast.EnumDecl):
                continue
            if not self._should_include_declaration(decl):
                continue
            if decl.name not in self._tagged_enums:
                continue

            payloads = {}
            for variant in decl.variants:
                if variant.payload_type is not None:
                    payloads[variant.name] = self._resolve_enum_payload(decl, variant)
            self._enum_payloads[decl.name] = payloads
            self._classify_enum_payloads(decl, payloads)
            self._enum_payload_unions[decl.name] = self._build_enum_payload_union(
                decl.name, payloads)

        self._bind_nested_enum_storage(ast_program)

        for enum_name, (pool_symbol, source_loc, prefix) in self._enum_storage.items():
            self._declare_enum_planes(enum_name, pool_symbol, source_loc, prefix)

    def _classify_enum_payloads(self, decl: ast.EnumDecl, payloads: dict) -> None:
        """Decide whether *decl*'s data plane is a union array or overlaid planes.

        A tagged enum's data plane has one layout. With plain payloads it is an
        array of union elements; with `#[soa]` payloads it is a region that
        every variant overlays with one plane per field, so an arm body and a
        payload struct's methods index by the raw element index. The two cannot
        share a plane — they would give the enum two stride disciplines, and
        make a struct's layout depend on where else it happens to be used — so
        an enum is all one or all the other.

        An `#[soa]` payload struct has no storage of its own: the enum's pool is
        its one static, which is what lets its methods hard-code plane
        addresses. That rules out a static of the struct, and a second enum or
        variant naming it.
        """
        soa = {name: t for name, t in payloads.items() if t.name in self._soa_structs}
        if not soa:
            return
        if len(soa) != len(payloads):
            plain = next(name for name in payloads if name not in soa)
            planed = next(iter(soa))
            raise HIRError(
                f"enum '{decl.name}' mixes #[soa] and plain payloads: "
                f"'{decl.name}::{planed}' is #[soa], '{decl.name}::{plain}' is not",
                source_loc=decl.source_loc,
                hint="a tagged enum's data plane has one layout — either an array "
                     "of union elements or a set of overlaid field planes; mark "
                     "every payload struct #[soa], or none of them",
            )

        self._tagged_enums[decl.name] = True
        for variant_name, payload_type in soa.items():
            struct_name = payload_type.name
            previous = self._payload_structs.get(struct_name)
            if previous is not None:
                prev_enum, prev_variant = previous
                raise HIRError(
                    f"#[soa] struct '{struct_name}' is already the payload of "
                    f"'{prev_enum}::{prev_variant}'",
                    source_loc=decl.source_loc,
                    hint="an #[soa] payload struct's fields are planes of that "
                         "enum's pool, addressed at compile time, so it can be "
                         "the payload of exactly one variant; declare a second "
                         "struct for the other",
                )
            stored = self._soa_storage.get(struct_name)
            if stored is not None:
                raise HIRError(
                    f"#[soa] struct '{struct_name}' is the payload of "
                    f"'{decl.name}::{variant_name}' but is also stored by static "
                    f"'{stored[0].name}'",
                    source_loc=stored[1],
                    hint="an #[soa] struct backs exactly one storage; as a "
                         "payload that storage is the enum's pool, so drop the "
                         "standalone static or use a plain struct as the payload",
                )
            self._payload_structs[struct_name] = (decl.name, variant_name)

    def _soa_payload_layout(self, enum_name: str):
        """`({(variant, field): element offset}, bytes per element)` for the region.

        Every plane of the region has `count` elements of its own stride, so a
        plane at element offset `o` puts element i at `count * o + i * stride`.
        Two planes may share bytes only if they have the same stride *and* the
        same offset — then element i of one is element i of the other. A wider
        plane laid over narrower ones would put its element i on their elements
        `2i` and `2i + 1`, so one element's payload would rewrite another's.

        So the region is cut into lanes, one per stride, narrowest first. A
        variant's k-th field of stride s sits at `lane(s) + k * s`, and lane s
        is as wide as the variant with the most fields of that stride needs.
        Variants still overlay — within each lane — so the region is the sum of
        the per-stride maxima, at least the largest variant and usually equal.
        """
        fields = [(variant, name, self._soa_stride(ftype.size_bytes))
                  for variant, name, ftype in self._soa_payload_fields(enum_name)]
        per_lane: Dict[int, Dict[str, int]] = {}
        for variant, _name, stride in fields:
            counts = per_lane.setdefault(stride, {})
            counts[variant] = counts.get(variant, 0) + 1
        lane_base, running = {}, 0
        for stride in sorted(per_lane):
            lane_base[stride] = running
            running += stride * max(per_lane[stride].values())

        offsets, used = {}, {}
        for variant, name, stride in fields:
            k = used.get((variant, stride), 0)
            used[(variant, stride)] = k + 1
            offsets[(variant, name)] = lane_base[stride] + k * stride
        return offsets, running

    def _soa_payload_fields(self, enum_name: str):
        """`[(variant, field name, field type)]` for an all-`#[soa]` enum.

        Variant order, then field order — the order the overlaid planes are
        laid out in. Works in both pass 1.5, where a payload struct's symbol
        still points at its AST node and the field types have to be resolved
        here, and pass 3, where it is HIR with resolved types already.
        """
        out = []
        for variant_name, payload_type in self._enum_payloads[enum_name].items():
            symbol = self.symbol_table.lookup(payload_type.name)
            definition = symbol.definition if symbol else None
            if isinstance(definition, hir.HIRStructDecl):
                fields = [(f.name, f.field_type) for f in definition.fields]
            else:
                fields = [(f.name, self.type_resolver.resolve_type(f.field_type))
                          for f in definition.fields]
            out.extend((variant_name, name, ftype) for name, ftype in fields)
        return out

    def _bind_nested_enum_storage(self, ast_program) -> None:
        """Back a tagged enum that is a *field* of an `#[soa]` element.

        Such an enum has no static of its own — the static is the containing
        pool — so nothing in pass 1 registers it. Its planes are the pool's
        nested ones, `units__kind__tag` and `units__kind__data`, which is what
        the recorded prefix names.

        Runs after pass 1 so every enum is known to be tagged and every
        `#[soa]` static is bound, and before the plane symbols are declared
        just above.
        """
        structs = {d.name: d for d in ast_program.items
                   if isinstance(d, ast.StructDecl)}
        for struct_name, (pool_symbol, source_loc) in self._soa_storage.items():
            struct = structs.get(struct_name)
            if struct is None:
                continue
            for field in struct.fields:
                enum_name = getattr(field.field_type, 'name', None)
                if enum_name not in self._tagged_enums:
                    continue
                previous = self._enum_storage.get(enum_name)
                if previous is not None:
                    raise HIRError(
                        f"tagged enum '{enum_name}' backs both "
                        f"'{struct_name}.{field.name}' and a static of its own",
                        source_loc=source_loc,
                        hint=f"a tagged enum backs exactly one array — its methods "
                             f"index that array's planes by an address fixed at "
                             f"compile time, so a second storage has nowhere to "
                             f"point; drop the standalone `[{enum_name}; N]` static "
                             f"and keep the field, or the other way round",
                    )
                self._enum_storage[enum_name] = (pool_symbol, source_loc,
                                                 f"{field.name}__")

    def _build_enum_payload_union(self, enum_name: str, payloads: dict) -> hir.HIRStructDecl:
        """Synthesize the `Enum__Payload` union backing a tagged enum's data plane.

        One field per payload variant, all at offset 0 — which is exactly a union,
        so `layout_fields` computes the plane's element size with no new layout
        code, and a payload access is an ordinary union-field access.

        Sizes are deliberately *not* consulted here. This runs before pass 2, so
        each payload struct's symbol still points at its AST node and asking for
        a size would read AST field types. The stride rounding that needs a size
        happens in `_pad_enum_payloads`, once every struct is HIR.
        """
        fields = [
            hir.HIRStructField(name=name, field_type=payload_type, offset=0)
            for name, payload_type in payloads.items()
        ]
        name = f"{enum_name}__Payload"
        decl = hir.HIRStructDecl(name=name, fields=fields, is_union=True)
        symbol = Symbol(name=name, kind=SymbolKind.STRUCT, definition=decl, scope_id=0)
        decl.symbol = symbol
        self.symbol_table.declare(name, symbol)
        return decl

    def _declare_enum_planes(self, enum_name: str, pool_symbol, source_loc,
                             prefix: str = '') -> None:
        """Declare the `POOL__tag` and `POOL__data` symbols for a tagged enum.

        Only the symbols: the `HIRStaticDecl`s behind them are built in pass 3
        by `_expand_soa_statics`, alongside every other plane. Declaring them
        here is what lets a `match` in pass 2 desugar straight into ordinary
        array indexing — the planes are plain statics, so nothing downstream
        needs to know a tagged enum was involved.
        """
        count = pool_symbol.var_type.size
        union = self._enum_payload_unions[enum_name]
        # A tagged enum numbers its variants densely from 0 — explicit
        # discriminants are rejected — so the largest is `count - 1` and the
        # same rule `_build_enum` applies gives the same answer. Pass 3 restamps
        # this from the built enum anyway; deriving it the one way keeps the two
        # from ever disagreeing in between.
        ast_enum = self.symbol_table.lookup(enum_name).definition
        tag_elem = discriminant_type(len(ast_enum.variants) - 1)

        for suffix, element_type in (
            ('tag', tag_elem),
            ('data', StructTypeInfo(name=union.name, definition=union,
                                    symbol=union.symbol)),
        ):
            self._declare_plane_symbol(
                f"{pool_symbol.name}__{prefix}{suffix}", element_type, count)

        # An all-#[soa] enum also has one plane per variant field, laid over
        # the data region. Declared here for the same reason as tag and data:
        # so a payload binding's `f.field` can be rewritten to `PLANE[i]` in
        # pass 2, before the type checker sees it.
        if self._tagged_enums.get(enum_name, False):
            for variant_name, field_name, field_type in self._soa_payload_fields(enum_name):
                suffix = f"{variant_name}__{field_name}"
                self._declare_plane_symbol(
                    f"{pool_symbol.name}__{prefix}{suffix}", field_type, count)

    def _build_newtype(self, decl: ast.NewtypeDecl) -> hir.HIRNewtypeDecl:
        """Build HIR newtype declaration from AST."""
        symbol = self.symbol_table.lookup(decl.name)
        hir_newtype = hir.HIRNewtypeDecl(
            name=decl.name,
            inner_type=symbol.type_info,
            symbol=symbol
        )
        # Point the symbol at the HIR definition, matching structs/aliases.
        symbol.definition = hir_newtype
        return hir_newtype

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
        # Apply #[cfg(...)] to the methods before anything reads them. Done once,
        # in place, because impl.methods is walked from several places —
        # filtering at each would risk them disagreeing about what exists.
        impl.methods = [m for m in impl.methods
                        if self._should_include_declaration(m)]

        # Validate struct exists
        struct_symbol = self.symbol_table.lookup(impl.struct_name)
        if not struct_symbol:
            raise HIRError(
                f"impl block for undefined struct: {impl.struct_name}",
                source_loc=impl.source_loc
            )
        if struct_symbol.kind not in (SymbolKind.STRUCT, SymbolKind.NEWTYPE,
                                      SymbolKind.ENUM):
            raise HIRError(
                f"impl block target '{impl.struct_name}' is not a struct",
                source_loc=impl.source_loc
            )

        # A newtype may implement a trait, but only for static dispatch. `*dyn`
        # reads a TypeId byte at offset 0 of the pointee, and a newtype has no
        # such byte — every byte is payload — so forming a dyn pointer to one is
        # rejected at the cast instead (type_checker.check_type_cast).
        #
        # In practice the two rarely meet: a trait whose methods take `self` by
        # value can only be implemented by a newtype, and one taking `*self` only
        # by a struct, so a trait is naturally either dyn-able or newtype-able.
        #
        # Clone stays rejected: it is redundant rather than impossible, since a
        # newtype copies with a plain assignment.
        if struct_symbol.kind == SymbolKind.NEWTYPE and impl.trait_name is not None:
            if impl.trait_name == CLONE_TRAIT:
                raise HIRError(
                    f"newtype '{impl.struct_name}' cannot implement Clone",
                    source_loc=impl.source_loc,
                    hint="newtypes are copied by plain assignment: 'let b = a;'"
                )

        # Handle trait impl validation and registration
        if impl.trait_name is not None:
            # Clone means a contiguous byte copy, which an #[soa] element is not:
            # its bytes are spread one per plane. Other traits are fine — the
            # receiver form follows the implementing type, so their methods take
            # the element index like any other #[soa] method.
            if (impl.trait_name == 'Clone'
                    and impl.struct_name in self._tagged_enums):
                raise HIRError(
                    f"tagged enum '{impl.struct_name}' cannot implement Clone",
                    source_loc=impl.source_loc,
                    hint="Clone copies a contiguous block, and a tagged enum "
                         "element's bytes are split between the tag plane and the "
                         "payload plane; write the variant you want instead, e.g. "
                         "`POOL[i] = Enum::Variant(...)`",
                )
            if (impl.trait_name == 'Clone'
                    and impl.struct_name in self._soa_structs):
                raise HIRError(
                    f"#[soa] struct '{impl.struct_name}' cannot implement Clone",
                    source_loc=impl.source_loc,
                    hint="Clone copies a contiguous block, and an #[soa] element's "
                         "bytes are spread across one array per field; copy the "
                         "fields you need explicitly",
                )
            # Compiler lang-item traits (Clone, operator traits) need no source
            # `trait` declaration and are static-dispatch only.
            if is_lang_item_trait(impl.trait_name):
                self._declare_lang_item_impl(impl)
                return
            self._declare_trait_impl(impl)
            return

        self._declare_impl_consts(impl)
        for method in impl.methods:
            self._register_method(impl.struct_name, method, overwrite=True)

    def _declare_impl_consts(self, impl):
        """Declare an impl's associated constants as `StructName::CONSTANT`."""
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

    def _register_method(self, struct_name: str, method, trait_name: str = None,
                         overwrite: bool = False) -> str:
        """Declare `Struct__method` and the `Struct.method` lookup the type
        checker resolves calls through; returns the mangled name.

        An inherent impl's method takes the lookup (*overwrite*); a trait's
        sets it only when nothing has, so an inherent method of the same name
        wins.
        """
        mangled_name = f"{struct_name}__{method.name}"
        self.symbol_table.declare(mangled_name, Symbol(
            name=mangled_name, kind=SymbolKind.METHOD, definition=method, scope_id=0))
        method_key = f"{struct_name}.{method.name}"
        if overwrite or self.symbol_table.lookup(method_key) is None:
            info = {
                'struct_name': struct_name,
                'mangled_name': mangled_name,
                # Near/far comes from the method's own self pointer.
                'method_self_is_far': method.self_is_far,
            }
            if trait_name is not None:
                info['trait_name'] = trait_name
            self.symbol_table.declare(method_key, Symbol(
                name=method_key, kind=SymbolKind.METHOD, definition=method,
                scope_id=0, type_info=info))
        return mangled_name

    def _is_newtype(self, name: str) -> bool:
        """True if `name` names a newtype."""
        sym = self.symbol_table.lookup(name)
        return sym is not None and sym.kind == SymbolKind.NEWTYPE

    def _is_enum(self, name: str) -> bool:
        """True if `name` names an enum."""
        sym = self.symbol_table.lookup(name)
        return sym is not None and sym.kind == SymbolKind.ENUM

    def _impl_self_form(self, impl, method) -> 'SelfForm':
        """Decide the self form for an impl method, rejecting mismatched pairings.

        One self form per type, never per method:

          POINTER    struct/union — addressed through `*self`
          VALUE_IN_A newtype, enum — the 1-2 byte payload rides in A
          INDEX_IN_X #[soa]       — a u16 element index rides in X

        The written form must agree with the type. A trait declared with `*self`
        is still implemented by a newtype or an SoA struct with bare `self`: the
        receiver form follows the implementing type, not the declaration.
        """
        if impl.struct_name in self._tagged_enums:
            # A tagged enum element has no address — its bytes are split across
            # the tag and payload planes — so `self` is the index, as for #[soa].
            form = SelfForm.INDEX_IN_X
        elif self._is_newtype(impl.struct_name) or self._is_enum(impl.struct_name):
            form = SelfForm.VALUE_IN_A
        elif impl.struct_name in self._soa_structs:
            form = SelfForm.INDEX_IN_X
        else:
            form = SelfForm.POINTER

        wrote_by_value = getattr(method, 'self_by_value', False)

        # An associated function declares no self, so there is no written form
        # to check; return the type's form for the caller's (unused) binding.
        if not getattr(method, 'has_self', True):
            return form

        if form is SelfForm.VALUE_IN_A and not wrote_by_value:
            ret = f" -> {impl.struct_name}" if method.return_type else ""
            noun = "enum" if self._is_enum(impl.struct_name) else "newtype"
            article = "an" if noun == "enum" else "a"
            raise HIRError(
                f"method '{impl.struct_name}::{method.name}' takes '*self', but "
                f"'{impl.struct_name}' is {article} {noun}",
                source_loc=method.source_loc,
                hint=f"{noun} methods take 'self' by value: "
                     f"'fn {method.name}(self){ret}' — return a new value instead of "
                     f"mutating in place"
            )

        if form is SelfForm.INDEX_IN_X and not wrote_by_value:
            tagged = impl.struct_name in self._tagged_enums
            what = "a tagged enum" if tagged else "#[soa]"
            why = ("a tagged enum element has no address — its tag and payload "
                   "live in separate arrays"
                   if tagged else
                   "an #[soa] element has no address — its fields live in "
                   "separate arrays")
            raise HIRError(
                f"method '{impl.struct_name}::{method.name}' takes '*self', but "
                f"'{impl.struct_name}' is {what}",
                source_loc=method.source_loc,
                hint=f"{why}. Take 'self' by value: "
                     f"'fn {method.name}(self, ...)'; inside the body 'self' is the "
                     f"element index"
            )

        if wrote_by_value and form is SelfForm.POINTER:
            raise HIRError(
                f"method '{impl.struct_name}::{method.name}' takes 'self' by value, "
                f"but '{impl.struct_name}' is not a newtype or enum",
                source_loc=method.source_loc,
                hint=f"use '*self': 'fn {method.name}(*self, ...)' — structs and unions "
                     f"are passed by reference"
            )

        return form

    def _validate_self_register_conflict(self, impl, method, params, self_binding):
        """Reject a parameter that binds the register a by-value `self` occupies.

        *self_binding* is None unless `self` rides in a register (so for a
        pointer `self` and for an associated function); when set, params[0] is
        that `self`. A by-value `self` rides in A (newtype/enum) or X (planed
        index). In A, neither A nor B is available to a parameter: B is the
        accumulator's high byte, not a register of its own. Sharing either
        with `self` is an entanglement the author did not ask for, and codegen
        routes B/X/Y argument setup *through* A, so it is also a live hazard.

        `_validate_no_duplicate_register_bindings` catches the A case too, but
        its message talks about two parameters — and `self` is not one the author
        wrote a binding for, so the conflict needs naming directly.
        """
        if not isinstance(self_binding, hir.RegisterBinding):
            return
        reg = self_binding.register_name
        if reg == 'X':
            receiver = "an #[soa] method receives 'self' as an element index"
        elif self._is_enum(impl.struct_name):
            receiver = "an enum method receives 'self' by value"
        else:
            receiver = "a newtype method receives 'self' by value"
        # A and B are one physical register; `self` in A claims both.
        blocked = {reg, 'B'} if reg == 'A' else {reg}
        # Name the registers that are actually still free, not a fixed pair.
        free = " or ".join(r for r in ('A', 'X', 'Y') if r not in blocked)

        for param in params[1:]:
            if not isinstance(param.binding, hir.RegisterBinding):
                continue
            bound = param.binding.register_name
            if bound not in blocked:
                continue
            if bound == reg:
                conflict = f"binds {reg}, which holds 'self'"
            else:
                conflict = (f"binds B, which is the high byte of A — "
                            f"and A holds 'self'")
            raise HIRError(
                f"parameter '{param.name}' of '{impl.struct_name}::{method.name}' "
                f"{conflict}",
                source_loc=method.source_loc,
                hint=f"{receiver} in {reg}; bind '{param.name}' to "
                     f"{free}, or pass it on the stack"
            )

    def _supertrait_closure(self, trait_name: str) -> List[str]:
        """Transitive supertraits of `trait_name`, de-duplicated, excluding itself.

        Cycle-safe (a `seen` guard prevents infinite loops); cycle *reporting* is
        handled separately by _validate_trait_hierarchy.
        """
        result: List[str] = []
        seen: Set[str] = set()
        stack = list(self._trait_supertraits.get(trait_name, []))
        while stack:
            s = stack.pop(0)
            if s in seen:
                continue
            seen.add(s)
            result.append(s)
            stack.extend(self._trait_supertraits.get(s, []))
        return result

    def _validate_trait_hierarchy(self):
        """Validate supertrait references, acyclicity, near/far uniformity, and no shadowing."""
        # 1. Every supertrait must exist and be a trait.
        for name, supers in self._trait_supertraits.items():
            for s in supers:
                sym = self.symbol_table.lookup(s)
                if sym is None:
                    raise HIRError(f"trait '{name}' lists unknown supertrait '{s}'",
                                   source_loc=self._trait_decl_loc(name))
                if sym.kind != SymbolKind.TRAIT:
                    raise HIRError(f"trait '{name}': supertrait '{s}' is not a trait",
                                   source_loc=self._trait_decl_loc(name))

        # 2. The supertrait graph must be acyclic.
        WHITE, GRAY, BLACK = 0, 1, 2
        color: Dict[str, int] = {}

        def dfs(n: str, path: List[str]):
            color[n] = GRAY
            for s in self._trait_supertraits.get(n, []):
                if color.get(s) == GRAY:
                    cycle = ' -> '.join(path + [s])
                    raise HIRError(f"trait inheritance cycle: {cycle}",
                                   source_loc=self._trait_decl_loc(n))
                if color.get(s, WHITE) == WHITE:
                    dfs(s, path + [s])
            color[n] = BLACK

        for n in self._trait_supertraits:
            if color.get(n, WHITE) == WHITE:
                dfs(n, [n])

        # 3 & 4. Per trait: uniform near/far across the hierarchy, and no method/const
        #        name appears in more than one trait of the closure (incl. the trait itself).
        for name in self._trait_supertraits:
            trait_ast = self.symbol_table.lookup(name).definition  # ast.TraitDecl
            chain = [name] + self._supertrait_closure(name)

            own_far = self._trait_far_kind(trait_ast)
            seen_methods: Dict[str, str] = {}
            seen_consts: Dict[str, str] = {}
            for t in chain:
                t_ast = self.symbol_table.lookup(t).definition
                t_far = self._trait_far_kind(t_ast)
                if own_far is not None and t_far is not None and own_far != t_far:
                    raise HIRError(
                        f"trait '{name}' is {'far' if own_far else 'near'} but supertrait "
                        f"'{t}' is {'far' if t_far else 'near'}; a trait and its supertraits "
                        f"must share the same calling convention",
                        source_loc=self._trait_decl_loc(name))
                for m in t_ast.methods:
                    if m.name in seen_methods and seen_methods[m.name] != t:
                        raise HIRError(
                            f"trait '{name}' inherits conflicting method '{m.name}' from both "
                            f"'{seen_methods[m.name]}' and '{t}'",
                            source_loc=self._trait_decl_loc(name))
                    seen_methods[m.name] = t
                for c in t_ast.constants:
                    if c.name in seen_consts and seen_consts[c.name] != t:
                        raise HIRError(
                            f"trait '{name}' inherits conflicting const '{c.name}' from both "
                            f"'{seen_consts[c.name]}' and '{t}'",
                            source_loc=self._trait_decl_loc(name))
                    seen_consts[c.name] = t

    def _validate_supertrait_impls(self):
        """Every struct implementing a trait must also implement all transitive supertraits."""
        for trait_name, struct_names in list(self._trait_impls.items()):
            supers = self._supertrait_closure(trait_name)
            if not supers:
                continue
            for struct in struct_names:
                for s in supers:
                    if struct not in self._trait_impls.get(s, []):
                        raise HIRError(
                            f"impl '{trait_name}' for '{struct}' requires '{struct}' to also "
                            f"implement supertrait '{s}'",
                            source_loc=self._trait_impl_locs.get((trait_name, struct)))

    def _trait_far_kind(self, trait_ast) -> Optional[bool]:
        """True if the trait's methods are far, False if near, None if it has no methods.

        A trait has one dispatch wrapper kind, so its methods must agree:
        a `*dyn` call through a far wrapper reaches the implementor with a
        3-byte return address, and a near implementor's `RTS` would pop
        only two of them.
        """
        if not trait_ast.methods:
            return None
        kinds = {m.is_far for m in trait_ast.methods}
        if len(kinds) > 1:
            near = sorted(m.name for m in trait_ast.methods if not m.is_far)
            raise HIRError(
                f"trait '{trait_ast.name}' mixes near and far methods "
                f"(near: {', '.join(near)}); a trait must be entirely near or "
                f"entirely far",
                source_loc=self._trait_decl_loc(trait_ast.name),
                hint="methods are far by default; mark every method 'near fn' or none")
        return kinds.pop()

    def _trait_decl_loc(self, trait_name: str):
        """Source location of a trait declaration, if available."""
        sym = self.symbol_table.lookup(trait_name)
        defn = sym.definition if sym else None
        return getattr(defn, 'source_loc', None)

    def _declare_lang_item_impl(self, impl: ast.ImplDecl):
        """First pass: declare a compiler lang-item trait impl (Clone, operator traits).

        Lang-item traits are known to the compiler and need no source `trait`
        declaration. They are static-dispatch only: no TypeId, no `*dyn`, and they
        do NOT participate in the near/far trait-consistency constraint (each method
        is called directly with its own calling convention). Methods are registered
        exactly like any other impl method so that explicit calls
        (`recv.method(args)`) resolve through the existing static-dispatch path.
        """
        # Clone is the one trait a union may implement: it is a bitwise copy of
        # size_bytes, so it neither reads fields nor changes layout. The operator
        # traits are rejected — arithmetic on an untagged overlay has no meaning.
        if impl.trait_name != CLONE_TRAIT:
            self._reject_union_trait_impl(impl)

        if impl.trait_name == CLONE_TRAIT:
            self._declare_clone_impl(impl)

        for method in impl.methods:
            self._register_method(impl.struct_name, method, impl.trait_name)

    def _declare_clone_impl(self, impl: ast.ImplDecl):
        """Validate and record a `Clone` impl. Empty body => auto bitwise copy."""
        for method in impl.methods:
            if method.name != CLONE_METHOD:
                raise HIRError(
                    f"impl Clone for '{impl.struct_name}': only a '{CLONE_METHOD}' "
                    f"method is allowed, found '{method.name}'",
                    source_loc=method.source_loc
                )
            if len(method.params) != 1:
                raise HIRError(
                    f"Clone::{CLONE_METHOD} for '{impl.struct_name}' must take exactly "
                    f"one parameter (the source) besides self",
                    source_loc=method.source_loc
                )
        is_auto = len(impl.methods) == 0

        # Register a marker symbol so the type checker (which shares this symbol
        # table) can resolve "does T implement Clone, and is it auto or manual".
        marker_key = f"{impl.struct_name}$Clone"
        if self.symbol_table.lookup(marker_key) is None:
            self.symbol_table.declare(marker_key, Symbol(
                name=marker_key,
                kind=SymbolKind.METHOD,
                definition=None,
                scope_id=0,
                type_info={'clone_auto': is_auto}
            ))

    def _declare_trait_impl(self, impl: ast.ImplDecl):
        """First pass: declare trait implementation methods and validate against trait definition."""
        self._reject_union_trait_impl(impl)

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
        trait_is_far = bool(self._trait_far_kind(trait_ast))

        # Each impl method must use the trait's calling convention: the
        # dispatch wrapper is built per trait, but the impl's own RTS/RTL
        # comes from the method as written.
        trait_methods_by_name = {m.name: m for m in trait_ast.methods}
        for method in impl.methods:
            trait_method = trait_methods_by_name.get(method.name)
            if trait_method is not None and method.is_far != trait_method.is_far:
                kind = 'far' if method.is_far else 'near'
                want = 'far' if trait_method.is_far else 'near'
                raise HIRError(
                    f"method '{method.name}' in impl '{impl.trait_name}' for "
                    f"'{impl.struct_name}' is {kind} but trait '{impl.trait_name}' "
                    f"declares it {want}",
                    source_loc=method.source_loc,
                    hint=("drop 'near' from the impl method" if trait_method.is_far
                          else "declare it 'near fn' to match the trait"))

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

        # Copy in default bodies for trait methods this impl omits. Each implementor
        # gets its own copy of the body, so from here on a defaulted method is
        # indistinguishable from one written out in the impl block.
        trait_method_names = {m.name for m in trait_ast.methods}
        impl_method_names = {m.name for m in impl.methods}
        for trait_method in trait_ast.methods:
            if trait_method.name in impl_method_names or trait_method.body is None:
                continue
            impl.methods.append(self._instantiate_default_method(trait_method))
            impl_method_names.add(trait_method.name)

        # Validate all remaining trait methods are implemented
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
        # Never for a newtype: it has no byte to put one in, and no dispatch table
        # entry can name it (see the dyn-cast rejection in the type checker).
        impl_symbol = self.symbol_table.lookup(impl.struct_name)
        is_newtype_impl = (impl_symbol is not None
                           and impl_symbol.kind == SymbolKind.NEWTYPE)
        if (impl.trait_name in self._dyn_used_traits
                and not is_newtype_impl
                and impl.struct_name not in self._struct_type_ids):
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
        self._trait_impl_locs[(impl.trait_name, impl.struct_name)] = impl.source_loc

        self._declare_impl_consts(impl)
        for method in impl.methods:
            mangled_name = self._register_method(impl.struct_name, method, impl.trait_name)

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

    def _is_union(self, type_name: str) -> bool:
        """True if *type_name* names a union (AST or HIR definition)."""
        symbol = self.symbol_table.lookup(type_name)
        definition = getattr(symbol, 'definition', None) if symbol else None
        return bool(getattr(definition, 'is_union', False))

    def _reject_union_trait_impl(self, impl: ast.ImplDecl):
        """Unions and tagged enums cannot implement traits.

        Dynamic dispatch injects a `__type_id: u8` at offset 0 to identify the
        concrete type. In a union offset 0 is live field data, and a tagged enum
        keeps its discriminant there — either way the TypeId and the data would
        alias. Inherent `impl` blocks are fine; they add no layout.
        """
        if self._is_union(impl.struct_name):
            raise HIRError(
                f"union '{impl.struct_name}' cannot implement trait '{impl.trait_name}'",
                source_loc=impl.source_loc,
                hint="trait dispatch stores a __type_id byte at offset 0, which would "
                     "overlap union field data; use an inherent `impl` block instead"
            )
        if impl.struct_name in self._tagged_enums:
            raise HIRError(
                f"tagged enum '{impl.struct_name}' cannot implement trait "
                f"'{impl.trait_name}'",
                source_loc=impl.source_loc,
                hint="trait dispatch stores a __type_id byte at offset 0, which is "
                     "where a tagged enum keeps its discriminant; match on the "
                     "variant in an inherent `impl` block instead"
            )

    def _instantiate_default_method(self, trait_method: ast.ImplMethod) -> ast.ImplMethod:
        """An implementor's copy of a trait method's default.

        Deep-copied because each implementor lowers its own instance
        (mangled StructName__method) and the HIR/typeck passes annotate AST nodes.
        """
        return copy.deepcopy(trait_method)

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
            supertraits=list(trait.supertraits),
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

        self._check_vector_bank(mangled_name, is_entry, interrupt_attr, method.source_loc)
        bank_attr = self._bank_attr_for(method.is_far, lambda: HIRError(
            f"method '{impl.struct_name}::{method.name}' in auto-bank mode cannot be 'near fn'",
            source_loc=method.source_loc,
            hint="drop 'near' (methods are far by default) or place it in an explicit bank with #[bank(n)]"))
        self._check_databank_far(f"{impl.struct_name}::{method.name}", mode_attr,
                                 method.is_far, method.source_loc)

        # Enter function scope
        func_scope_id = self.symbol_table.enter_scope(ScopeKind.FUNCTION)

        # Determine self pointer type from the method's own self pointer:
        # `far *self` -> far, `*self` -> near.
        self_is_far = method.self_is_far

        # An associated function declares no self, so none is synthesized: it is
        # an ordinary function that happens to be namespaced by the type. A
        # synthetic self here would take a parameter slot the caller never
        # passes, and (bound to A) would drag a 2-byte payload into m16 entry.
        has_self = getattr(method, 'has_self', True)

        # Build self parameter. A newtype takes `self` by value in the
        # accumulator — the payload is one or two bytes, so there is nothing to
        # point at and a trivial accessor costs nothing. An #[soa] struct takes
        # an element index in X, which is where indexed addressing wants it.
        self_form = self._impl_self_form(impl, method)
        soa_pool = None
        if self_form is SelfForm.VALUE_IN_A:
            self_type = self.type_resolver.resolve_named_type(impl.struct_name)
            self_binding = hir.RegisterBinding(register_name='A')
        elif self_form is SelfForm.INDEX_IN_X:
            tagged = impl.struct_name in self._tagged_enums
            payload_of = self._payload_structs.get(impl.struct_name)
            if payload_of is not None:
                # An #[soa] payload struct's one storage is its enum's pool;
                # `self` indexes that pool, and `self.field` reaches the
                # variant's plane there.
                registry, key = self._enum_storage, payload_of[0]
            else:
                registry = self._enum_storage if tagged else self._soa_storage
                key = impl.struct_name
            stored = registry.get(key)
            if stored is None and has_self:
                what = "tagged enum" if tagged else "#[soa] struct"
                if payload_of is not None:
                    hint = (f"'{impl.struct_name}' is the payload of "
                            f"'{payload_of[0]}::{payload_of[1]}', so its storage is "
                            f"the enum's; declare one, e.g. "
                            f"`#[lowram] static mut POOL: [{payload_of[0]}; 16];`")
                else:
                    hint = (f"declare one, e.g. "
                            f"`#[lowram] static mut POOL: [{impl.struct_name}; 16];`")
                raise HIRError(
                    f"{what} '{impl.struct_name}' has no static, so its "
                    f"methods have no array to index",
                    source_loc=method.source_loc,
                    hint=hint,
                )
            soa_pool = stored[0] if stored else None
            # A plain u16: `self` is genuinely an index, so `self == 0` and
            # `other(self)` work. Field access never sees it — the base rewrite
            # in ExpressionBuilder turns `self.f` into `POOL[self].f` first.
            self_type = BasicTypeInfo(name='u16')
            self_binding = hir.RegisterBinding(register_name='X')
        else:
            struct_type = StructTypeInfo(name=impl.struct_name)
            self_type = PointerTypeInfo(pointee_type=struct_type, is_far=self_is_far)
            self_binding = None  # Pointer self: stack-passed, or Y (SELF_Y) for a trait impl method

        self_param_symbol = Symbol(
            name='self',
            kind=SymbolKind.PARAMETER,
            definition=method,
            scope_id=self.symbol_table.current_scope_id,
            var_type=self_type,
            is_mutable=True
        )
        hir_params = []
        if has_self:
            self.symbol_table.declare('self', self_param_symbol)
            hir_params.append(hir.HIRParameter(
                name='self',
                param_type=self_type,
                binding=self_binding,
                symbol=self_param_symbol
            ))

        # Process additional parameters
        for param in method.params:
            hir_param = self._build_parameter(param)
            hir_params.append(hir_param)

        # A by-value self occupies A (or X), so a parameter binding that
        # register would silently share it.
        self._validate_self_register_conflict(
            impl, method, hir_params, self_binding if has_self else None)
        self._validate_no_duplicate_register_bindings(hir_params, mangled_name)

        # Process body. `_soa_rewrite_base` reads `soa_self_pool` to turn
        # `self.f` into `POOL[self].f`, and the match desugar reads
        # `enum_self_pool` to turn `match self` into a read of the tag plane at
        # index `self`. Restore rather than clear, so a future nested build
        # cannot lose an outer pool.
        tagged_self = impl.struct_name in self._tagged_enums
        payload_self = self._payload_structs.get(impl.struct_name)
        previous_pool = self.expression_builder.soa_self_pool
        previous_enum_pool = self.expression_builder.enum_self_pool
        previous_payload_self = self.expression_builder.payload_self
        self.expression_builder.soa_self_pool = (
            None if tagged_self or payload_self else soa_pool)
        self.expression_builder.enum_self_pool = (
            (impl.struct_name, soa_pool) if tagged_self and soa_pool else None)
        self.expression_builder.payload_self = (
            payload_self if payload_self and soa_pool else None)
        try:
            hir_body = self._build_block(method.body)
        finally:
            self.expression_builder.soa_self_pool = previous_pool
            self.expression_builder.enum_self_pool = previous_enum_pool
            self.expression_builder.payload_self = previous_payload_self

        # Add implicit return A if needed
        self._add_implicit_return(hir_body, method.return_type, interrupt_attr, method.body)

        # Exit function scope
        self.symbol_table.exit_scope()

        entry_m_mode, ret_type, exit_m_mode = self._signature_modes(
            hir_params, method.return_type, mangled_name, method.source_loc)

        # Get method symbol
        method_symbol = self.symbol_table.lookup(mangled_name)

        inline_attr = self._auto_inline_hint(
            inline_attr, interrupt_attr, is_entry, hir_body)

        # Eagerly validate const fn body at definition time
        if method.is_const:
            self.const_evaluator.validate_const_fn(method, mangled_name)

        return hir.HIRFunctionDecl(
            name=mangled_name,
            is_far=method.is_far,
            is_const=method.is_const,
            # Lang-item traits (Clone, operator traits) are static-dispatch only:
            # self is a normal stack arg, not the dyn-dispatch self-in-Y convention.
            is_trait_method=bool(impl.trait_name) and not is_lang_item_trait(impl.trait_name),
            parameters=hir_params,
            return_type=ret_type,
            body=hir_body,
            mode_attr=mode_attr,
            preserves_attr=preserves_attr,
            clobbers_attr=attrs['clobbers'],
            bank_attr=bank_attr,
            interrupt_attr=interrupt_attr,
            inline_attr=inline_attr,
            is_entry=is_entry,
            is_used=attrs['is_used'],
            allow_lints=attrs['allow'],
            symbol=method_symbol,
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
            constructed = self.expression_builder.build_enum_construction(stmt.expr)
            if constructed is not None:
                return constructed
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
            self._reject_soa_type(var_type, 'local', let.name, let.source_loc)

        # Process binding
        binding = None
        if let.binding:
            if not isinstance(let.binding, ast.Register):
                raise HIRError(
                    f"`let {let.name} @ {let.binding.name}`: a let binds only a register",
                    source_loc=let.source_loc)
            binding = hir.RegisterBinding(register_name=let.binding.name)

        # Validate register binding types (only if type is explicitly specified)
        if binding and var_type:
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

        `..=` with const bounds is the same with `<=` (the counter type holds
        `end + 1`); with a runtime bound it tests after the body, so an `end`
        at the counter type's maximum still exits:
        `let mut i = start; if i <= end { loop { body; if i == end { break; } i = i + 1; } }`

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

        # Build start and end expressions. A bound that const-evaluates is
        # folded to a literal here, so that `0..N` with `const N: u16 = 8`
        # compares the u8 counter against an immediate rather than against a
        # u16 identifier - the widening that would otherwise be inserted is
        # a TypeConvert use of the counter, which disqualifies it from the
        # X/Y loop promotion a literal `0..8` gets.
        start_expr = self._fold_for_bound(for_stmt.start)
        end_expr = self._fold_for_bound(for_stmt.end)

        # Determine loop variable type from range bounds
        # Try to const-evaluate both bounds to pick appropriate type
        loop_var_type = self._infer_for_loop_type(for_stmt.start, for_stmt.end, src_loc, for_stmt.inclusive)
        bounds_const = (self._try_const_eval(for_stmt.start) is not None
                        and self._try_const_eval(for_stmt.end) is not None)

        # A counter that indexes an array is a u16 whatever its range: an
        # index is 16 bits on this CPU, so a u8 counter would be widened at
        # every `arr[i]` - a TypeConvert use that keeps it out of X/Y and
        # costs a load, a mask and two mode switches per access instead of
        # one transfer.
        if loop_var_type.name == 'u8' and self._for_var_indexes_array(for_stmt):
            loop_var_type = BasicTypeInfo('u16')

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
        def increment_stmt():
            return hir.HIRAssignment(
                target=hir.HIRIdentifier(
                    name=for_stmt.variable,
                    symbol=loop_var_symbol,
                    source_loc=src_loc
                ),
                value=hir.HIRBinaryOp(
                    op='+',
                    left=hir.HIRIdentifier(
                        name=for_stmt.variable,
                        symbol=loop_var_symbol,
                        source_loc=src_loc
                    ),
                    right=hir.HIRIntegerLiteral(value=1, source_loc=src_loc),
                    source_loc=src_loc
                ),
                source_loc=src_loc
            )

        if for_stmt.inclusive and not bounds_const:
            # A runtime `..=` end may be the counter type's maximum, where
            # `i <= end` never fails. Test after the body instead, so the
            # counter never steps past `end`:
            #   if i <= end { loop { body; if i == end { break; } i = i + 1; } }
            def advance():
                exit_check = hir.HIRIfStmt(
                    condition=hir.HIRBinaryOp(
                        op='==',
                        left=hir.HIRIdentifier(name=for_stmt.variable,
                                               symbol=loop_var_symbol, source_loc=src_loc),
                        right=self._fold_for_bound(for_stmt.end),
                        source_loc=src_loc),
                    then_block=hir.HIRBlock(
                        statements=[hir.HIRBreakStmt(label=for_stmt.label, source_loc=src_loc)],
                        source_loc=src_loc),
                    source_loc=src_loc)
                return hir.HIRBlock(statements=[exit_check, increment_stmt()],
                                    source_loc=src_loc)
        else:
            advance = increment_stmt

        # The advance ends the body, so a `continue` - which jumps to the
        # loop's head - would skip it and never advance the counter. Each one
        # that targets this loop advances first.
        body_statements = [
            self._increment_before_continue(stmt, for_stmt.label, advance, False)
            for stmt in body_statements
        ]
        body_statements.append(advance())

        while_body = hir.HIRBlock(
            statements=body_statements,
            source_loc=for_stmt.body.source_loc
        )

        if advance is increment_stmt:
            loop_stmt = hir.HIRWhileStmt(
                condition=condition,
                body=while_body,
                is_infinite=False,
                label=for_stmt.label,
                source_loc=src_loc
            )
        else:
            loop_stmt = hir.HIRIfStmt(
                condition=condition,
                then_block=hir.HIRBlock(statements=[hir.HIRWhileStmt(
                    condition=hir.HIRBooleanLiteral(value=True),
                    body=while_body,
                    is_infinite=True,
                    label=for_stmt.label,
                    source_loc=src_loc
                )], source_loc=src_loc),
                source_loc=src_loc
            )

        # Exit the for-loop scope
        self.symbol_table.exit_scope()

        # Decrement loop depth
        self.loop_depth -= 1

        # Return block containing let and while
        return hir.HIRBlock(
            statements=[let_stmt, loop_stmt],
            source_loc=src_loc
        )

    def _increment_before_continue(self, node, label, increment_stmt, inner: bool):
        """Return *node* with every `continue` that targets the `for` loop being
        built replaced by `{ i = i + 1; continue; }`.

        An unlabeled `continue` targets it unless it sits in a nested loop
        (*inner*); a labeled one targets it by its label from any depth. A
        nested `for` was built first, so its own continues are already done.
        Walks HIR statements and expressions by their fields, not into
        declarations (a call's callee) or symbols.
        """
        if isinstance(node, hir.HIRContinueStmt):
            if (node.label is None and not inner) or (node.label is not None and node.label == label):
                return hir.HIRBlock(statements=[increment_stmt(), node], source_loc=node.source_loc)
            return node
        if not isinstance(node, (hir.HIRStatement, hir.HIRExpression, hir.HIRMatchArm)):
            return node
        if isinstance(node, (hir.HIRWhileStmt, hir.HIRLoopExpression)):
            inner = True
        for f in dataclasses.fields(node):
            value = getattr(node, f.name)
            if isinstance(value, list):
                new = [self._increment_before_continue(v, label, increment_stmt, inner) for v in value]
                if any(a is not b for a, b in zip(new, value)):
                    setattr(node, f.name, new)
            elif isinstance(value, (hir.HIRStatement, hir.HIRExpression, hir.HIRMatchArm)):
                new = self._increment_before_continue(value, label, increment_stmt, inner)
                if new is not value:
                    setattr(node, f.name, new)
        return node

    def _infer_for_loop_type(self, start: ast.Expression, end: ast.Expression, src_loc, inclusive: bool = False) -> TypeInfo:
        """Infer the appropriate type for a for loop variable based on range bounds.

        Picks the smallest type that holds every value the counter takes,
        including the exit value (`end` for `..`, `end + 1` for `..=`). Falls
        back to u16 if bounds cannot be const-evaluated; a const inclusive
        range whose exit value exceeds 16 bits is an error.

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

        # The counter must hold every value it takes, including the one that
        # fails the test: `i < end` exits at `end`, `i <= end` exits at
        # `end + 1`. So `0..256` needs a u16, and so does `0..=255` — a u8
        # counter would wrap to 0 and `i <= 255` would never fail.
        last_val = end_val + 1 if inclusive else end_val
        min_val = min(start_val, last_val)
        max_val = max(start_val, last_val)

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

        # An inclusive range ending at the 16-bit maximum has no counter that
        # can step past its end, so the loop would never exit.
        if inclusive:
            raise HIRError(
                f"for-loop range {start_val}..={end_val} never terminates: its "
                f"counter would have to pass {end_val}, beyond 16 bits",
                source_loc=src_loc,
            )
        return BasicTypeInfo('u16')

    def _for_var_indexes_array(self, for_stmt: ast.ForStmt) -> bool:
        """Does the loop body use the counter as an array index?

        Walks the body's AST for `ArrayIndex` nodes whose index *is* the
        counter, or is arithmetic over it (`arr[i + 1]`, `arr[i << 1]`).
        A nested loop that rebinds the name shadows it and is not searched.
        """
        from dataclasses import fields, is_dataclass

        name = for_stmt.variable

        def mentions(expr) -> bool:
            if isinstance(expr, ast.Identifier):
                return expr.name == name
            if isinstance(expr, ast.BinaryOp):
                return mentions(expr.left) or mentions(expr.right)
            if isinstance(expr, ast.UnaryOp):
                return mentions(expr.operand)
            return False

        def walk(node) -> bool:
            if isinstance(node, ast.ForStmt) and node is not for_stmt and node.variable == name:
                return False
            if isinstance(node, ast.ArrayIndex) and mentions(node.index):
                return True
            if isinstance(node, list):
                return any(walk(n) for n in node)
            if is_dataclass(node):
                return any(walk(getattr(node, f.name)) for f in fields(node)
                           if f.name != 'source_loc')
            return False

        return walk(for_stmt.body)

    def _fold_for_bound(self, expr: ast.Expression) -> hir.HIRExpression:
        """Build a `for` range bound, as a literal when it is a constant."""
        value = self._try_const_eval(expr)
        if value is not None and not isinstance(expr, ast.IntegerLiteral):
            return hir.HIRIntegerLiteral(value=value, source_loc=expr.source_loc)
        return self._build_expression(expr)

    def _try_const_eval(self, expr: ast.Expression) -> Optional[int]:
        """Try to const-evaluate an expression, returning None if not possible.

        Deliberately narrower than ConstEvaluator: a for-loop counter's type is
        inferred from this, and folding `N as u16` would type an explicitly
        u16 range as u8.
        """
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
            Dictionary with keys: mode, preserves, clobbers, interrupt, inline, is_entry, is_used, allow
        """
        result = {
            'mode': None,
            'preserves': None,
            'clobbers': None,
            'interrupt': None,
            'inline': None,
            'is_entry': False,
            'is_used': False,
            'allow': frozenset(),
        }

        for attr in processed_attrs:
            if isinstance(attr, ModeAttribute):
                result['mode'] = attr
            elif isinstance(attr, PreservesAttribute):
                result['preserves'] = attr
            elif isinstance(attr, ClobbersAttribute):
                result['clobbers'] = attr
            elif isinstance(attr, InterruptAttribute):
                result['interrupt'] = attr
            elif isinstance(attr, InlineAttribute):
                result['inline'] = attr
            elif isinstance(attr, EntryAttribute):
                result['is_entry'] = True
            elif isinstance(attr, UsedAttribute):
                result['is_used'] = True
            elif isinstance(attr, AllowAttribute):
                result['allow'] = attr.codes

        return result

    @staticmethod
    def _desugared_match(stmt) -> Optional[hir.HIRMatchExpression]:
        """The match *stmt* holds, if the parser desugared it from `if let` or
        `let ... else`; a snapshotted index wraps it in a block expression."""
        expr = getattr(stmt, 'expr', None) if isinstance(stmt, hir.HIRExprStmt) else None
        if isinstance(expr, hir.HIRBlockExpression):
            expr = expr.final_expr
        if isinstance(expr, hir.HIRMatchExpression) and expr.origin is not None:
            return expr
        return None

    def _add_implicit_return(self, hir_body: hir.HIRBlock, return_type, interrupt_attr=None,
                             ast_body=None):
        """Add implicit return A if function doesn't have explicit return.

        Args:
            hir_body: HIR block (function body)
            return_type: AST return type (or None)
            interrupt_attr: Interrupt attribute if this is an interrupt handler
            ast_body: The body's AST block, to see whether its tail diverges
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

        # A tail that never falls through — a `loop` with no break out, an
        # if/else or match whose branches all return — needs no return after
        # it. One would be unreachable `return A`, and A's u8 is no `-> bool`.
        if (ast_body is not None and ast_body.statements
                and self.expression_builder.diverges(ast_body.statements[-1])):
            return

        # A match desugared from `if let` is a statement, like `if`: it falls
        # through to the implicit return below. One from `let ... else` holds
        # the rest of the body in its first arm (its other arm diverges), so
        # that arm's tail is the function's tail.
        desugared = self._desugared_match(last_stmt)
        if desugared is not None:
            if desugared.origin == 'let_else':
                self._add_implicit_return(desugared.arms[0].body, return_type)
                return
        # Trailing return expression: if the last statement is an ExprStmt
        # and the function has a return type, convert to return
        elif isinstance(last_stmt, hir.HIRExprStmt) and return_type is not None:
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

