"""
HIR Builder - transforms AST to HIR with name resolution and desugaring.

Uses a two-pass algorithm:
1. First pass: Declare all top-level symbols
2. Second pass: Build HIR nodes with resolved references
"""

from typing import List, Union, Optional
from r65.compiler.frontend import ast

from r65.compiler.hir import nodes as hir
from r65.compiler.hir.symbol_table import *
from r65.compiler.hir.types import *
from r65.compiler.hir.attributes import *
from r65.compiler.hir.const_eval import *
from r65.compiler.hir.errors import *


class HIRBuilder:
    """Builds HIR from AST with name resolution and desugaring."""

    def __init__(self):
        self.symbol_table = SymbolTable()
        self.const_evaluator = ConstEvaluator(self.symbol_table)
        self.type_resolver = TypeResolver(self.symbol_table, self.const_evaluator)
        self.attr_processor = AttributeProcessor()

    def build_program(self, ast_program: ast.Program) -> hir.HIRProgram:
        """
        Build HIR from AST program.

        Args:
            ast_program: AST Program node

        Returns:
            HIRProgram with symbol table
        """
        # Pass 1: Declare all top-level symbols
        for decl in ast_program.items:
            self._declare_toplevel(decl)

        # Pass 2: Build HIR nodes with resolved references
        hir_decls = []
        for decl in ast_program.items:
            hir_decl = self._build_declaration(decl)
            hir_decls.append(hir_decl)

        return hir.HIRProgram(
            declarations=hir_decls,
            symbol_table=self.symbol_table
        )

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
        else:
            raise HIRError(f"Unknown declaration type: {type(decl).__name__}")

    def _build_function(self, func: ast.FunctionDecl) -> hir.HIRFunctionDecl:
        """Build HIR function from AST."""
        # Process attributes
        processed_attrs = self.attr_processor.process_attributes(
            func.attributes,
            context='function'
        )

        # Extract specific attributes
        mode_attr = None
        preserves_attr = None
        bank_attr = None
        interrupt_attr = None
        is_entry = False

        for attr in processed_attrs:
            if isinstance(attr, ModeAttribute):
                mode_attr = attr
            elif isinstance(attr, PreservesAttribute):
                preserves_attr = attr
            elif isinstance(attr, BankAttribute):
                bank_attr = attr
            elif isinstance(attr, InterruptAttribute):
                interrupt_attr = attr
            elif isinstance(attr, EntryAttribute):
                is_entry = True

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
            symbol=func_symbol
        )

    def _build_parameter(self, param: ast.Parameter) -> hir.HIRParameter:
        """Build HIR parameter from AST."""
        # Resolve parameter type
        param_type = self.type_resolver.resolve_type(param.param_type)

        # Process binding
        binding = None
        if param.binding:
            if isinstance(param.binding, ast.Register):
                binding = hir.RegisterBinding(register_name=param.binding.name)
            elif isinstance(param.binding, str):
                # Variable binding - resolve variable
                var_symbol = self.symbol_table.lookup(param.binding)
                if not var_symbol:
                    raise HIRError(f"Undefined variable: {param.binding}")
                if var_symbol.kind != SymbolKind.STATIC_VAR:
                    raise HIRError(f"Parameter binding must be static variable, got {var_symbol.kind.value}")
                binding = hir.VariableBinding(
                    variable_name=param.binding,
                    variable_symbol=var_symbol
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

    def _build_static(self, static: ast.StaticDecl) -> hir.HIRStaticDecl:
        """Build HIR static declaration from AST."""
        # Process attributes
        processed_attrs = self.attr_processor.process_attributes(
            static.attributes,
            context='static'
        )

        storage_attr = None
        for attr in processed_attrs:
            if isinstance(attr, StorageAttribute):
                storage_attr = attr

        # Resolve type
        var_type = self.type_resolver.resolve_type(static.var_type)

        # Build initializer if present
        initializer = None
        if static.initializer:
            initializer = self._build_expression(static.initializer)

        # Get static symbol
        static_symbol = self.symbol_table.lookup(static.name)

        return hir.HIRStaticDecl(
            name=static.name,
            is_mutable=static.is_mut,
            var_type=var_type,
            initializer=initializer,
            storage_attr=storage_attr,
            symbol=static_symbol
        )

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

        return hir.HIRStructDecl(
            name=struct.name,
            fields=hir_fields,
            symbol=struct_symbol
        )

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

        return hir.HIREnumDecl(
            name=enum.name,
            variants=hir_variants,
            underlying_type=underlying_type,
            symbol=enum_symbol
        )

    def _build_type_alias(self, alias: ast.TypeAlias) -> hir.HIRTypeAlias:
        """Build HIR type alias from AST."""
        # Resolve aliased type
        aliased_type = self.type_resolver.resolve_type(alias.aliased_type)

        # Get alias symbol
        alias_symbol = self.symbol_table.lookup(alias.name)

        return hir.HIRTypeAlias(
            name=alias.name,
            aliased_type=aliased_type,
            symbol=alias_symbol
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
            scope_id=block_scope_id
        )

    def _build_statement(self, stmt: ast.Statement) -> hir.HIRStatement:
        """Build HIR statement from AST."""
        if isinstance(stmt, ast.Block):
            return self._build_block(stmt)

        elif isinstance(stmt, ast.LetStmt):
            return self._build_let(stmt)

        elif isinstance(stmt, ast.ExprStmt):
            return hir.HIRExprStmt(expr=self._build_expression(stmt.expr))

        elif isinstance(stmt, ast.ReturnStmt):
            values = [self._build_expression(v) for v in stmt.values]
            return hir.HIRReturnStmt(values=values)

        elif isinstance(stmt, ast.BreakStmt):
            return hir.HIRBreakStmt()

        elif isinstance(stmt, ast.ContinueStmt):
            return hir.HIRContinueStmt()

        elif isinstance(stmt, ast.IfStmt):
            return self._build_if(stmt)

        elif isinstance(stmt, ast.WhileStmt):
            return self._build_while(stmt)

        elif isinstance(stmt, ast.LoopStmt):
            return self._build_loop(stmt)

        elif isinstance(stmt, ast.AsmStmt):
            return hir.HIRAsmStmt(instructions=stmt.instructions)

        else:
            raise HIRError(f"Unknown statement type: {type(stmt).__name__}")

    def _build_let(self, let: ast.LetStmt) -> hir.HIRLetStmt:
        """Build HIR let statement from AST."""
        # Build initializer
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
            symbol=local_symbol
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
            else_block=else_block
        )

    def _build_while(self, while_stmt: ast.WhileStmt) -> hir.HIRWhileStmt:
        """Build HIR while statement from AST."""
        condition = self._build_expression(while_stmt.condition)
        body = self._build_block(while_stmt.body)

        return hir.HIRWhileStmt(
            condition=condition,
            body=body,
            is_infinite=False
        )

    def _build_loop(self, loop: ast.LoopStmt) -> hir.HIRWhileStmt:
        """Desugar loop to while true."""
        body = self._build_block(loop.body)

        return hir.HIRWhileStmt(
            condition=hir.HIRBooleanLiteral(value=True),
            body=body,
            is_infinite=True
        )

    # =========================================================================
    # Build Expressions
    # =========================================================================

    def _build_expression(self, expr: ast.Expression) -> hir.HIRExpression:
        """Build HIR expression from AST."""
        if isinstance(expr, ast.IntegerLiteral):
            return hir.HIRIntegerLiteral(value=expr.value)

        elif isinstance(expr, ast.BooleanLiteral):
            return hir.HIRBooleanLiteral(value=expr.value)

        elif isinstance(expr, ast.Identifier):
            # Resolve identifier
            symbol = self.symbol_table.lookup(expr.name)
            if not symbol:
                raise HIRError(f"Undefined identifier: {expr.name}")
            return hir.HIRIdentifier(name=expr.name, symbol=symbol)

        elif isinstance(expr, ast.Register):
            # Resolve register
            symbol = self.symbol_table.lookup(expr.name)
            return hir.HIRRegister(name=expr.name, symbol=symbol)

        elif isinstance(expr, ast.BinaryOp):
            left = self._build_expression(expr.left)
            right = self._build_expression(expr.right)
            return hir.HIRBinaryOp(op=expr.op, left=left, right=right)

        elif isinstance(expr, ast.UnaryOp):
            operand = self._build_expression(expr.operand)
            return hir.HIRUnaryOp(op=expr.op, operand=operand)

        elif isinstance(expr, ast.TypeCast):
            inner = self._build_expression(expr.expr)
            target = self.type_resolver.resolve_type(expr.target_type)
            return hir.HIRTypeCast(expr=inner, target_type=target)

        elif isinstance(expr, ast.FunctionCall):
            func = self._build_expression(expr.func)
            args = [self._build_expression(a) for a in expr.args]
            return hir.HIRFunctionCall(func=func, args=args)

        elif isinstance(expr, ast.ArrayIndex):
            array = self._build_expression(expr.array)
            index = self._build_expression(expr.index)
            return hir.HIRArrayIndex(array=array, index=index, original_ast=expr)

        elif isinstance(expr, ast.FieldAccess):
            base = self._build_expression(expr.base)
            # Field resolution happens in type checker
            return hir.HIRFieldAccess(base=base, field_name=expr.field)

        elif isinstance(expr, ast.Dereference):
            pointer = self._build_expression(expr.pointer)
            return hir.HIRDereference(pointer=pointer)

        elif isinstance(expr, ast.Assignment):
            target = self._build_expression(expr.target)
            value = self._build_expression(expr.value)
            return hir.HIRAssignment(target=target, value=value)

        else:
            raise HIRError(f"Unknown expression type: {type(expr).__name__}")

    # =========================================================================
    # Helpers
    # =========================================================================

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
        from .types import BasicTypeInfo, ArrayTypeInfo, PointerTypeInfo

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

        else:
            # For struct/enum, would need to look up definition
            # Simplified for now
            return 1
