"""
HIR Builder - transforms AST to HIR with name resolution and desugaring.

Uses a two-pass algorithm:
1. First pass: Declare all top-level symbols
2. Second pass: Build HIR nodes with resolved references
"""

from typing import List, Union, Optional
from pathlib import Path
from r65.compiler.frontend import ast

from r65.compiler.hir import nodes as hir
from r65.compiler.hir.symbol_table import *
from r65.compiler.hir.types import *
from r65.compiler.hir.attributes import *
from r65.compiler.hir.const_eval import *
from r65.compiler.hir.cfg import CfgEvaluator
from r65.compiler.hir.errors import *


class HIRBuilder:
    """Builds HIR from AST with name resolution and desugaring."""

    def __init__(self, source_file: Optional[str] = None, cfg_evaluator: Optional[CfgEvaluator] = None):
        """
        Initialize HIR builder.

        Args:
            source_file: Path to source file being compiled.
                        Used for resolving relative paths in include_bytes!.
            cfg_evaluator: Optional cfg evaluator for conditional compilation.
        """
        self.symbol_table = SymbolTable()
        self.cfg_evaluator = cfg_evaluator
        self.const_evaluator = ConstEvaluator(self.symbol_table, self.cfg_evaluator)
        self.type_resolver = TypeResolver(self.symbol_table, self.const_evaluator)
        self.attr_processor = AttributeProcessor()
        self.source_file = source_file
        self.source_dir = Path(source_file).parent if source_file else Path.cwd()
        self.current_bank = 0  # Current ROM bank for declarations (set by #[bank(n)])
        self.auto_bank_mode = False  # True when in #[bank(auto)] mode (NOT default for backward compatibility)
        self._pending_snesrom_config = None  # Store snesrom config when parsed as attribute

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
            if self._should_include_declaration(decl):
                hir_decl = self._build_declaration(decl)
                hir_decls.append(hir_decl)

        # Use pending snesrom config if it was set from an attribute
        if snesrom_config is None and self._pending_snesrom_config is not None:
            snesrom_config = self._pending_snesrom_config

        return hir.HIRProgram(
            declarations=hir_decls,
            symbol_table=self.symbol_table,
            stack_attr=stack_attr,
            snesrom_config=snesrom_config
        )

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
            # Resolve type and create static variable symbol
            var_type = self.type_resolver.resolve_type(decl.var_type)
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

            # Validate: const only supports primitive types (no arrays or structs)
            if isinstance(const_type, ArrayTypeInfo):
                raise HIRError(
                    f"const '{decl.name}' cannot have array type",
                    hint="use 'static' for array constants (immutable statics are ROM)"
                )
            if isinstance(const_type, StructTypeInfo):
                raise HIRError(
                    f"const '{decl.name}' cannot have struct type",
                    hint="use 'static' for struct constants (immutable statics are ROM)"
                )

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
        elif isinstance(decl, ast.ImplDecl):
            return self._build_impl(decl)
        else:
            raise HIRError(f"Unknown declaration type: {type(decl).__name__}")

    def _build_function(self, func: ast.FunctionDecl) -> hir.HIRFunctionDecl:
        """Build HIR function from AST."""
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

        # Process body
        hir_body = self._build_block(func.body)

        # Add implicit return A if needed (but not for interrupt handlers)
        self._add_implicit_return(hir_body, func.return_type, interrupt_attr)

        # Exit function scope
        self.symbol_table.exit_scope()

        # Resolve return type
        ret_type = None
        if func.return_type:
            ret_type = self.type_resolver.resolve_type(func.return_type)

        # Get function symbol
        func_symbol = self.symbol_table.lookup(func.name)

        # Validate: DBR management modes require far functions
        if mode_attr and mode_attr.databank != DataBankMode.NONE:
            if not func.is_far:
                raise HIRError(
                    f"Function '{func.name}' uses databank={mode_attr.databank.value} "
                    f"but is not a far function. DBR management requires 'far fn'."
                )

        # Detect STATUS flag return pattern for optimized branch generation at call sites
        returns_status_flag = self._detect_status_flag_return(hir_body)

        # Infer entry mode from parameters and validate X/Y are u16
        entry_m_mode = self._infer_entry_mode_and_validate(hir_params, func.name, func.source_loc)

        return hir.HIRFunctionDecl(
            name=func.name,
            is_far=func.is_far,
            parameters=hir_params,
            return_type=ret_type,
            body=hir_body,
            mode_attr=mode_attr,
            preserves_attr=preserves_attr,
            bank_attr=bank_attr,
            interrupt_attr=interrupt_attr,
            is_entry=is_entry,
            symbol=func_symbol,
            returns_status_flag=returns_status_flag,
            entry_m_mode=entry_m_mode,
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

    def _infer_entry_mode_and_validate(self, params: List[hir.HIRParameter], func_name: str, source_loc) -> 'ModeState':
        """
        Infer entry mode from function parameters and validate X/Y are u16.

        Rules:
        - If any parameter is bound to A with type u16/i16 -> m16 entry
        - Otherwise -> m8 entry (default)
        - X/Y parameters must be u16/i16 (compiler error if u8/i8)

        Args:
            params: List of HIR function parameters
            func_name: Function name for error messages
            source_loc: Source location for error messages

        Returns:
            ModeState for function entry (M8 or M16)
        """
        from r65.compiler.typeck.processor_mode import ModeState

        entry_mode = ModeState.M8  # Default

        for param in params:
            if isinstance(param.binding, hir.RegisterBinding):
                reg_name = param.binding.register_name

                if reg_name == "A":
                    # Check if A parameter is 16-bit
                    if param.param_type and isinstance(param.param_type, BasicTypeInfo):
                        if param.param_type.name in ('u16', 'i16'):
                            entry_mode = ModeState.M16

                elif reg_name in ("X", "Y"):
                    # Validate X/Y parameters are 16-bit
                    if param.param_type and isinstance(param.param_type, BasicTypeInfo):
                        if param.param_type.name in ('u8', 'i8'):
                            raise HIRError(
                                f"Function '{func_name}': Parameter '{param.name}' bound to {reg_name} "
                                f"has type {param.param_type.name}, but X/Y registers are always 16-bit (u16).\n"
                                f"  In R65, index registers X and Y are always 16-bit.\n"
                                f"  Change the parameter type to u16: {param.name} @ {reg_name}: u16",
                                source_loc=source_loc
                            )

        return entry_mode

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
                    raise HIRError(f"Undefined variable: {var_name}")
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
                    raise HIRError(f"Parameter binding must be register or static variable, got {var_symbol.kind.value}")
            elif isinstance(param.binding, str):
                # Could be register or variable binding - resolve (legacy string support)
                var_symbol = self.symbol_table.lookup(param.binding)
                if not var_symbol:
                    raise HIRError(f"Undefined variable: {param.binding}")
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
                    raise HIRError(f"Parameter binding must be register or static variable, got {var_symbol.kind.value}")

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
                hint="add 'mut' to make mutable, or remove attribute for ROM"
            )

    def _build_static(self, static: ast.StaticDecl) -> hir.HIRStaticDecl:
        """Build HIR static declaration from AST."""
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

        # Resolve type
        var_type = self.type_resolver.resolve_type(static.var_type)

        # Build initializer if present
        initializer = None
        if static.initializer:
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
            symbol=static_symbol
        )

        # Update symbol's definition to point to HIR node (not AST node)
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
            symbol=const_symbol
        )

    def _build_struct(self, struct: ast.StructDecl) -> hir.HIRStructDecl:
        """Build HIR struct declaration from AST."""
        # Build fields with offsets
        hir_fields = []
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

        # Resolve return type
        ret_type = None
        if method.return_type:
            ret_type = self.type_resolver.resolve_type(method.return_type)

        # Get method symbol
        method_symbol = self.symbol_table.lookup(mangled_name)

        # Detect STATUS flag return pattern
        returns_status_flag = self._detect_status_flag_return(hir_body)

        # Infer entry mode from parameters and validate X/Y are u16
        entry_m_mode = self._infer_entry_mode_and_validate(hir_params, mangled_name, method.source_loc)

        return hir.HIRFunctionDecl(
            name=mangled_name,
            is_far=method.is_far,
            parameters=hir_params,
            return_type=ret_type,
            body=hir_body,
            mode_attr=mode_attr,
            preserves_attr=preserves_attr,
            bank_attr=bank_attr,
            interrupt_attr=interrupt_attr,
            is_entry=is_entry,
            symbol=method_symbol,
            returns_status_flag=returns_status_flag,
            entry_m_mode=entry_m_mode,
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

        elif isinstance(stmt, ast.ExprStmt):
            return hir.HIRExprStmt(expr=self._build_expression(stmt.expr), source_loc=stmt.source_loc)

        elif isinstance(stmt, ast.ReturnStmt):
            values = [self._build_expression(v) for v in stmt.values]
            return hir.HIRReturnStmt(values=values, source_loc=stmt.source_loc)

        elif isinstance(stmt, ast.BreakStmt):
            return hir.HIRBreakStmt(label=stmt.label, source_loc=stmt.source_loc)

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
            return hir.HIRAsmStmt(instructions=stmt.instructions, source_loc=stmt.source_loc)

        else:
            raise HIRError(f"Unknown statement type: {type(stmt).__name__}")

    def _build_let(self, let: ast.LetStmt) -> Union[hir.HIRLetStmt, hir.HIRTupleLetStmt]:
        """Build HIR let statement from AST."""
        # Build initializer (may be None for uninitialized variables)
        initializer = None
        if let.initializer is not None:
            initializer = self._build_expression(let.initializer)

        # Handle tuple destructuring pattern
        if let.pattern:
            return self._build_tuple_let(let, initializer)

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
                    raise HIRError(f"Undefined variable: {let.binding}")
                binding = hir.VariableLetBinding(
                    variable_name=let.binding,
                    variable_symbol=var_symbol
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

    def _build_tuple_let(self, let: ast.LetStmt, initializer: hir.HIRExpression) -> hir.HIRTupleLetStmt:
        """Build HIR tuple let statement for destructuring patterns."""
        names = let.pattern.names
        symbols = []

        # Create a symbol for each binding
        # Types will be inferred during type checking from the initializer's tuple type
        for name in names:
            local_symbol = Symbol(
                name=name,
                kind=SymbolKind.LOCAL_VAR,
                definition=let,
                scope_id=self.symbol_table.current_scope_id,
                var_type=None,  # Will be inferred
                is_mutable=let.is_mut
            )
            self.symbol_table.declare(name, local_symbol)
            symbols.append(local_symbol)

        return hir.HIRTupleLetStmt(
            names=names,
            is_mutable=let.is_mut,
            var_types=[],  # Will be filled during type checking
            initializer=initializer,
            symbols=symbols,
            source_loc=let.source_loc
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
        """
        src_loc = for_stmt.source_loc

        # Enter a new scope for the for-loop block
        self.symbol_table.enter_scope(ScopeKind.BLOCK)

        # Build start and end expressions
        start_expr = self._build_expression(for_stmt.start)
        end_expr = self._build_expression(for_stmt.end)

        # Create symbol for loop variable (mutable, type inferred from start)
        loop_var_symbol = Symbol(
            name=for_stmt.variable,
            kind=SymbolKind.LOCAL_VAR,
            definition=for_stmt,
            scope_id=self.symbol_table.current_scope_id,
            var_type=None,  # Will be inferred during type checking
            is_mutable=True
        )
        self.symbol_table.declare(for_stmt.variable, loop_var_symbol)

        # Create let statement: let mut i = start;
        let_stmt = hir.HIRLetStmt(
            name=for_stmt.variable,
            is_mutable=True,
            var_type=None,  # Inferred
            initializer=start_expr,
            binding=None,
            symbol=loop_var_symbol,
            source_loc=src_loc
        )

        # Create condition: i < end
        loop_var_ref = hir.HIRIdentifier(
            name=for_stmt.variable,
            symbol=loop_var_symbol,
            source_loc=src_loc
        )
        condition = hir.HIRBinaryOp(
            op='<',
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

        # Return block containing let and while
        return hir.HIRBlock(
            statements=[let_stmt, while_stmt],
            source_loc=src_loc
        )

    # =========================================================================
    # Build Expressions
    # =========================================================================

    def _build_expression(self, expr: ast.Expression) -> hir.HIRExpression:
        """Build HIR expression from AST."""
        # Get source location from AST node
        src_loc = expr.source_loc

        if isinstance(expr, ast.IntegerLiteral):
            return hir.HIRIntegerLiteral(value=expr.value, source_loc=src_loc)

        elif isinstance(expr, ast.BooleanLiteral):
            return hir.HIRBooleanLiteral(value=expr.value, source_loc=src_loc)

        elif isinstance(expr, ast.StringLiteral):
            return hir.HIRStringLiteral(value=expr.value, source_loc=src_loc)

        elif isinstance(expr, ast.Identifier):
            # Resolve identifier
            symbol = self.symbol_table.lookup(expr.name)
            if not symbol:
                raise HIRError(f"Undefined identifier: {expr.name}", source_loc=src_loc)
            return hir.HIRIdentifier(name=expr.name, symbol=symbol, source_loc=src_loc)

        elif isinstance(expr, ast.Register):
            # Resolve register
            symbol = self.symbol_table.lookup(expr.name)
            return hir.HIRRegister(name=expr.name, symbol=symbol, source_loc=src_loc)

        elif isinstance(expr, ast.IncludeBytesExpr):
            # Include binary data from file
            # Validate that the file exists and get file info
            resolved_path, file_size = self._validate_include_bytes_path(expr.path, expr.source_loc)
            return hir.HIRIncludeBytesExpr(
                path=expr.path,
                resolved_path=resolved_path,
                size=file_size,
                source_loc=src_loc
            )

        elif isinstance(expr, ast.ArrayFillExpr):
            # Array fill expression: [value; count]
            fill_value = self._build_expression(expr.value)
            # Count must be a constant - evaluate at compile time
            count = self.const_evaluator.eval(expr.count)
            return hir.HIRArrayFillExpr(fill_value=fill_value, count=count, source_loc=src_loc)

        elif isinstance(expr, ast.ArrayLiteralExpr):
            # Array literal expression: [a, b, c, ...]
            elements = [self._build_expression(e) for e in expr.elements]
            return hir.HIRArrayLiteralExpr(elements=elements, source_loc=src_loc)

        elif isinstance(expr, ast.StructLiteralExpr):
            # Struct literal expression: Player { x: 10, y: 20, health: 100 }
            return self._build_struct_literal(expr)

        elif isinstance(expr, ast.EnumVariantExpr):
            # Resolve enum variant to HIREnumVariantExpr (preserves type info)
            # Lookup enum type
            enum_symbol = self.symbol_table.lookup(expr.enum_name)
            if not enum_symbol:
                raise HIRError(f"Undefined enum: {expr.enum_name}", source_loc=src_loc)
            if enum_symbol.kind != SymbolKind.ENUM:
                raise HIRError(f"{expr.enum_name} is not an enum", source_loc=src_loc)

            # Lookup variant with qualified name
            qualified_name = f"{expr.enum_name}::{expr.variant_name}"
            variant_symbol = self.symbol_table.lookup(qualified_name)
            if not variant_symbol:
                raise HIRError(f"Undefined enum variant: {qualified_name}", source_loc=src_loc)
            if variant_symbol.kind != SymbolKind.ENUM_VARIANT:
                raise HIRError(f"{qualified_name} is not an enum variant", source_loc=src_loc)

            # Get variant value from symbol (stored in const_value field)
            variant_value = variant_symbol.const_value
            return hir.HIREnumVariantExpr(
                enum_name=expr.enum_name,
                variant_name=expr.variant_name,
                value=variant_value,
                source_loc=src_loc
            )

        elif isinstance(expr, ast.BinaryOp):
            # Check for string concatenation
            if expr.op == '+':
                left_is_string = isinstance(expr.left, ast.StringLiteral)
                right_is_string = isinstance(expr.right, ast.StringLiteral)

                if left_is_string or right_is_string:
                    try:
                        # Use const evaluator on the AST directly
                        const_value = self.const_evaluator.eval(expr)
                        if isinstance(const_value, str):
                            return hir.HIRStringLiteral(value=const_value, source_loc=src_loc)
                    except Exception:
                        # If constant evaluation fails, fall back to normal processing
                        pass

            left = self._build_expression(expr.left)
            right = self._build_expression(expr.right)
            return hir.HIRBinaryOp(op=expr.op, left=left, right=right, source_loc=src_loc)

        elif isinstance(expr, ast.UnaryOp):
            operand = self._build_expression(expr.operand)
            return hir.HIRUnaryOp(op=expr.op, operand=operand, source_loc=src_loc)

        elif isinstance(expr, ast.TypeCast):
            inner = self._build_expression(expr.expr)
            target = self.type_resolver.resolve_type(expr.target_type)
            return hir.HIRTypeCast(expr=inner, target_type=target, source_loc=src_loc)

        elif isinstance(expr, ast.FunctionCall):
            from r65.compiler.builtins import BuiltinRegistry

            # Check if this is a method call (e.g., value.rotate_left(3) or array.len())
            if isinstance(expr.func, ast.FieldAccess):
                method_name = expr.func.field
                if method_name in ['rotate_left', 'rotate_right']:
                    # This is a rotate method call
                    # Validate: must have exactly 1 argument
                    if len(expr.args) != 1:
                        raise HIRError(f"{method_name}() takes exactly 1 argument, got {len(expr.args)}", source_loc=src_loc)

                    # Build the base expression (the value being rotated)
                    base = self._build_expression(expr.func.base)

                    # Build the argument (rotation count)
                    count_arg = self._build_expression(expr.args[0])

                    # Return a special HIRMethodCall node for rotate methods
                    return hir.HIRMethodCall(
                        receiver=base,
                        method_name=method_name,
                        args=[count_arg],
                        source_loc=src_loc
                    )
                elif method_name == 'len':
                    # This is a len() method call on an array
                    # Validate: must have no arguments
                    if len(expr.args) != 0:
                        raise HIRError(f"len() takes no arguments, got {len(expr.args)}", source_loc=src_loc)

                    # Build the base expression (the array)
                    base = self._build_expression(expr.func.base)

                    # Try to const-evaluate if base is an identifier with known array type
                    if isinstance(expr.func.base, ast.Identifier):
                        symbol = self.symbol_table.lookup(expr.func.base.name)
                        if symbol and symbol.var_type and isinstance(symbol.var_type, ArrayTypeInfo):
                            # We know the array size at compile time
                            # len() returns u16 to hold array lengths up to 65535
                            result = hir.HIRIntegerLiteral(value=symbol.var_type.size, source_loc=src_loc)
                            result.expr_type = BasicTypeInfo('u16')
                            return result

                    # Return a special HIRMethodCall node for len method
                    # Type checker will validate receiver is an array
                    return hir.HIRMethodCall(
                        receiver=base,
                        method_name=method_name,
                        args=[],
                        source_loc=src_loc
                    )

            # Check if this is a built-in function call BEFORE trying to build func expression
            # This prevents "undefined identifier" errors for built-in function names
            builtin_name = None
            if isinstance(expr.func, ast.Identifier):
                func_name = expr.func.name
                if BuiltinRegistry.is_builtin(func_name):
                    builtin = BuiltinRegistry.get_builtin(func_name)

                    # Special handling for size_of - try const evaluation first
                    if builtin and builtin.kind.value == "type_info" and func_name == "size_of":
                        try:
                            # Try to evaluate at compile time using const evaluator
                            const_value = self.const_evaluator.eval(expr)
                            if isinstance(const_value, int):
                                return hir.HIRIntegerLiteral(value=const_value, source_loc=src_loc)
                        except Exception:
                            # If const evaluation fails, fall back to runtime call
                            pass

                    # Mark this as a built-in function call
                    builtin_name = func_name

            # Build func expression
            # For built-ins, create a dummy symbol to avoid "undefined identifier" errors
            if builtin_name:
                # Create a dummy symbol for built-in function
                builtin_symbol = Symbol(
                    name=expr.func.name,
                    kind=SymbolKind.FUNCTION,
                    definition=None,
                    scope_id=0,  # Global scope
                    var_type=None
                )
                func = hir.HIRIdentifier(name=expr.func.name, symbol=builtin_symbol, source_loc=src_loc)
            else:
                func = self._build_expression(expr.func)

            args = [self._build_expression(a) for a in expr.args]

            return hir.HIRFunctionCall(func=func, args=args, builtin_name=builtin_name, source_loc=src_loc)

        elif isinstance(expr, ast.CfgFunctionCall):
            # cfg!(flag) evaluates to true/false based on compiler cfg options
            condition_name = None
            if isinstance(expr.condition, ast.Identifier):
                condition_name = expr.condition.name
            elif hasattr(expr.condition, 'value'):
                condition_name = str(expr.condition.value)

            # Check if this cfg flag is set
            if self.cfg_evaluator and condition_name:
                cond = ast.CfgIdentifier(name=condition_name)
                result = self.cfg_evaluator.evaluate(cond)
            else:
                result = False

            return hir.HIRBooleanLiteral(value=result, source_loc=src_loc)

        elif isinstance(expr, ast.ArrayIndex):
            array = self._build_expression(expr.array)
            index = self._build_expression(expr.index)
            return hir.HIRArrayIndex(array=array, index=index, original_ast=expr, source_loc=src_loc)

        elif isinstance(expr, ast.FieldAccess):
            base = self._build_expression(expr.base)

            # Check for STATUS.Flag pattern
            if isinstance(base, hir.HIRRegister) and base.name == 'STATUS':
                from r65.compiler.hir.status_flags import get_status_flag, get_all_flag_names
                flag = get_status_flag(expr.field)
                if flag:
                    return hir.HIRStatusFlagAccess(
                        flag_name=flag.name,
                        bit_position=flag.bit_position,
                        bit_mask=flag.bit_mask,
                        source_loc=src_loc
                    )
                else:
                    valid_flags = ', '.join(get_all_flag_names())
                    raise HIRError(
                        f"Unknown STATUS flag '{expr.field}'. "
                        f"Valid flags: {valid_flags}",
                        source_loc=src_loc
                    )

            # Normal field resolution happens in type checker
            return hir.HIRFieldAccess(base=base, field_name=expr.field, source_loc=src_loc)

        elif isinstance(expr, ast.Dereference):
            pointer = self._build_expression(expr.pointer)
            return hir.HIRDereference(pointer=pointer, source_loc=src_loc)

        elif isinstance(expr, ast.AddressOf):
            operand = self._build_expression(expr.operand)
            return hir.HIRAddressOf(operand=operand, source_loc=src_loc)

        elif isinstance(expr, ast.Assignment):
            target = self._build_expression(expr.target)
            value = self._build_expression(expr.value)
            return hir.HIRAssignment(target=target, value=value, source_loc=src_loc)

        elif isinstance(expr, ast.CompoundAssignment):
            # Desugar compound assignment: x += 5 becomes x = x + 5
            target = self._build_expression(expr.target)
            value = self._build_expression(expr.value)

            # Create binary operation: target op value
            binary_op = hir.HIRBinaryOp(
                op=expr.operator,
                left=target,  # Read from target
                right=value,
                source_loc=src_loc
            )

            # Create assignment: target = (target op value)
            return hir.HIRAssignment(target=target, value=binary_op, source_loc=src_loc)

        elif isinstance(expr, ast.MultiAssignment):
            # Multiple assignment: lo, hi = func()
            targets = [self._build_expression(t) for t in expr.targets]
            value = self._build_expression(expr.value)
            return hir.HIRMultiAssignment(targets=targets, value=value, source_loc=src_loc)

        elif isinstance(expr, ast.MatchExpression):
            # Build match expression
            return self._build_match_expression(expr)

        else:
            raise HIRError(f"Unknown expression type: {type(expr).__name__}")

    # =========================================================================
    # Helpers
    # =========================================================================

    def _build_match_expression(self, expr: ast.MatchExpression) -> hir.HIRMatchExpression:
        """Build HIR match expression from AST."""
        # Build scrutinee
        scrutinee = self._build_expression(expr.scrutinee)

        # Build each arm
        arms = []
        for ast_arm in expr.arms:
            # Enter new scope for pattern bindings
            scope_id = self.symbol_table.enter_scope(ScopeKind.BLOCK)

            # Build pattern (may create bindings in scope)
            pattern = self._build_pattern(ast_arm.pattern)

            # Build body expression
            body = self._build_expression(ast_arm.body)

            # Exit scope
            self.symbol_table.exit_scope()

            arms.append(hir.HIRMatchArm(pattern=pattern, body=body, scope_id=scope_id))

        return hir.HIRMatchExpression(scrutinee=scrutinee, arms=arms, source_loc=expr.source_loc)

    def _build_pattern(self, pattern: ast.Pattern) -> hir.HIRPattern:
        """Build HIR pattern from AST pattern."""
        if isinstance(pattern, ast.LiteralPattern):
            return hir.HIRLiteralPattern(value=pattern.value)

        elif isinstance(pattern, ast.EnumPattern):
            # Resolve enum variant value
            enum_symbol = self.symbol_table.lookup(pattern.enum_name)
            if not enum_symbol or enum_symbol.kind != SymbolKind.ENUM:
                raise HIRError(f"Undefined enum: {pattern.enum_name}")

            qualified_name = f"{pattern.enum_name}::{pattern.variant_name}"
            variant_symbol = self.symbol_table.lookup(qualified_name)
            if not variant_symbol or variant_symbol.kind != SymbolKind.ENUM_VARIANT:
                raise HIRError(f"Undefined enum variant: {qualified_name}")

            variant_value = variant_symbol.const_value
            return hir.HIREnumPattern(
                enum_name=pattern.enum_name,
                variant_name=pattern.variant_name,
                variant_value=variant_value
            )

        elif isinstance(pattern, ast.WildcardPattern):
            return hir.HIRWildcardPattern()

        elif isinstance(pattern, ast.IdentifierPattern):
            # Create a new binding in current scope
            # Determine type from scrutinee during type checking
            symbol = self.symbol_table.define(
                name=pattern.name,
                kind=SymbolKind.LOCAL_VAR,
                var_type=None  # Will be set during type checking
            )
            return hir.HIRIdentifierPattern(name=pattern.name, symbol=symbol)

        elif isinstance(pattern, ast.OrPattern):
            patterns = [self._build_pattern(p) for p in pattern.patterns]
            return hir.HIROrPattern(patterns=patterns)

        else:
            raise HIRError(f"Unknown pattern type: {type(pattern).__name__}")

    def _build_struct_literal(self, expr: ast.StructLiteralExpr) -> hir.HIRStructLiteralExpr:
        """Build HIR struct literal from AST."""
        # Lookup struct definition
        struct_symbol = self.symbol_table.lookup(expr.struct_name)
        if not struct_symbol:
            raise HIRError(f"Undefined struct: {expr.struct_name}")
        if struct_symbol.kind != SymbolKind.STRUCT:
            raise HIRError(f"{expr.struct_name} is not a struct")

        # Get the struct declaration to access field information
        struct_decl = struct_symbol.definition

        # Build field offsets mapping
        field_offsets = {}
        field_types = {}
        if isinstance(struct_decl, hir.HIRStructDecl):
            # Already built
            for field in struct_decl.fields:
                field_offsets[field.name] = field.offset
                field_types[field.name] = field.field_type
        elif isinstance(struct_decl, ast.StructDecl):
            # Need to calculate offsets
            current_offset = 0
            for field in struct_decl.fields:
                field_type = self.type_resolver.resolve_type(field.field_type)
                field_offsets[field.name] = current_offset
                field_types[field.name] = field_type
                current_offset += self._get_type_size(field_type)

        # Build field initializers
        hir_fields = []
        for field_init in expr.fields:
            if field_init.name not in field_offsets:
                raise HIRError(f"Unknown field: {expr.struct_name}.{field_init.name}")

            value_expr = self._build_expression(field_init.value)
            hir_field = hir.HIRStructFieldInit(
                name=field_init.name,
                value=value_expr,
                field_offset=field_offsets[field_init.name]
            )
            hir_fields.append(hir_field)

        # Create HIR struct literal
        # Note: struct_decl may be AST node (pass 1 not complete) or HIR node
        # Type checker will resolve this properly
        return hir.HIRStructLiteralExpr(
            struct_name=expr.struct_name,
            struct_decl=struct_decl if isinstance(struct_decl, hir.HIRStructDecl) else None,
            fields=hir_fields
        )

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
            'is_entry': False
        }

        for attr in processed_attrs:
            if isinstance(attr, ModeAttribute):
                result['mode'] = attr
            elif isinstance(attr, PreservesAttribute):
                result['preserves'] = attr
            elif isinstance(attr, InterruptAttribute):
                result['interrupt'] = attr
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

        # Add implicit return A (unless return type is !)
        if return_type is None or not isinstance(return_type, ast.NeverType):
            a_symbol = self.symbol_table.lookup('A')
            hir_body.statements.append(
                hir.HIRReturnStmt(values=[hir.HIRRegister(name='A', symbol=a_symbol)])
            )

    def _get_type_size(self, type_info) -> int:
        """Get size of a type in bytes."""
        from .types import BasicTypeInfo, ArrayTypeInfo, PointerTypeInfo, FunctionTypeInfo

        if isinstance(type_info, BasicTypeInfo):
            if type_info.name in ['u8', 'i8', 'bool']:
                return 1
            elif type_info.name in ['u16', 'i16']:
                return 2
            else:
                raise HIRError(f"Unknown basic type: {type_info.name}")

        elif isinstance(type_info, ArrayTypeInfo):
            elem_size = self._get_type_size(type_info.element_type)
            return elem_size * type_info.size

        elif isinstance(type_info, PointerTypeInfo):
            return 3 if type_info.is_far else 2

        elif isinstance(type_info, FunctionTypeInfo):
            # Function pointers: 2 bytes for near fn(), 3 bytes for far fn()
            return 3 if type_info.is_far else 2

        elif isinstance(type_info, StructTypeInfo):
            # Look up struct definition and sum field sizes
            struct_symbol = self.symbol_table.lookup(type_info.name)
            if struct_symbol and struct_symbol.definition:
                struct_def = struct_symbol.definition
                if isinstance(struct_def, hir.HIRStructDecl):
                    # Sum up field sizes
                    return sum(self._get_type_size(f.field_type) for f in struct_def.fields)
                elif isinstance(struct_def, ast.StructDecl):
                    # Calculate from AST definition
                    size = 0
                    for field in struct_def.fields:
                        field_type = self.type_resolver.resolve_type(field.field_type)
                        size += self._get_type_size(field_type)
                    return size
            raise HIRError(f"Cannot determine size of struct: {type_info.name}")

        elif isinstance(type_info, EnumTypeInfo):
            # Enums are sized based on their underlying type
            # Default to u8 (1 byte) if not specified
            return 1

        else:
            raise HIRError(f"Cannot determine size of type: {type(type_info).__name__}")

    def _validate_include_bytes_path(self, path: str, source_loc: Optional[SourceLocation]) -> tuple[str, int]:
        """
        Validate that the file path for include_bytes! exists and return its info.

        Args:
            path: The file path from the include_bytes! expression
            source_loc: Source location for error reporting

        Returns:
            Tuple of (resolved_path, file_size)

        Raises:
            HIRError: If the file does not exist
        """
        # Resolve path relative to source file directory
        resolved_path = self.source_dir / path

        if not resolved_path.exists():
            raise HIRError(
                f"include_bytes!: file not found: '{path}' "
                f"(resolved to '{resolved_path}')",
                source_loc=source_loc
            )

        if not resolved_path.is_file():
            raise HIRError(
                f"include_bytes!: path is not a file: '{path}'",
                source_loc=source_loc
            )

        # Get file size
        file_size = resolved_path.stat().st_size
        return str(resolved_path), file_size
