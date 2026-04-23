# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
HIR (High-level Intermediate Representation) for R65 compiler.

The HIR layer performs:
- Name resolution and symbol table construction
- Attribute validation and processing
- Type annotation preparation
- Desugaring of complex constructs
"""

from r65.compiler.hir.errors import HIRError
from r65.compiler.hir.nodes import (
    # Base classes
    HIRNode,
    HIRExpression,
    HIRStatement,
    HIRDeclaration,
    SourceLocation,

    # Program
    HIRProgram,

    # Declarations
    HIRFunctionDecl,
    HIRParameter,
    HIRStaticDecl,
    HIRConstDecl,
    HIRStructDecl,
    HIRStructField,
    HIREnumDecl,
    HIREnumVariant,
    HIRTypeAlias,
    HIRImplDecl,

    # Statements
    HIRBlock,
    HIRLetStmt,
    HIRMultiLetStmt,
    HIRTupleLetStmt,  # backward-compat alias for HIRMultiLetStmt
    HIRExprStmt,
    HIRReturnStmt,
    HIRBreakStmt,
    HIRContinueStmt,
    HIRIfStmt,
    HIRWhileStmt,
    HIRAsmStmt,

    # Expressions
    HIRIntegerLiteral,
    HIRBooleanLiteral,
    HIREnumVariantExpr,
    HIRIdentifier,
    HIRRegister,
    HIRStatusFlagAccess,
    HIRIncludeBytesExpr,
    HIRArrayFillExpr,
    HIRArrayLiteralExpr,
    HIRStringLiteral,
    HIRStructFieldInit,
    HIRStructLiteralExpr,
    HIRBinaryOp,
    HIRUnaryOp,
    HIRTypeCast,
    HIRFunctionCall,
    HIRMethodCall,
    HIRArrayIndex,
    HIRFieldAccess,
    HIRDereference,
    HIRAddressOf,
    HIRAssignment,
    HIRMultiAssignment,
    HIRFunctionAddress,

    # Pattern Matching
    HIRPattern,
    HIRLiteralPattern,
    HIREnumPattern,
    HIRWildcardPattern,
    HIRIdentifierPattern,
    HIRRangePattern,
    HIROrPattern,
    HIRMatchArm,
    HIRMatchExpression,
    HIRBlockExpression,
    HIRIfExpression,
    HIRLoopExpression,

    # Bindings
    ParameterBinding,
    RegisterBinding,
    VariableBinding,
    LetBinding,
    RegisterLetBinding,
    VariableLetBinding,
)
from r65.compiler.hir.symbol_table import *
from r65.compiler.hir.types import *
from r65.compiler.hir.attributes import *
from r65.compiler.hir.ast_const_eval import ConstEvaluator
from r65.compiler.hir.builder import HIRBuilder

__all__ = [
    # Errors
    'HIRError',

    # Base classes
    'HIRNode',
    'HIRExpression',
    'HIRStatement',
    'HIRDeclaration',
    'SourceLocation',

    # Program
    'HIRProgram',

    # Declarations
    'HIRFunctionDecl',
    'HIRParameter',
    'HIRStaticDecl',
    'HIRConstDecl',
    'HIRStructDecl',
    'HIRStructField',
    'HIREnumDecl',
    'HIREnumVariant',
    'HIRTypeAlias',
    'HIRImplDecl',

    # Statements
    'HIRBlock',
    'HIRLetStmt',
    'HIRMultiLetStmt',
    'HIRTupleLetStmt',
    'HIRExprStmt',
    'HIRReturnStmt',
    'HIRBreakStmt',
    'HIRContinueStmt',
    'HIRIfStmt',
    'HIRWhileStmt',
    'HIRAsmStmt',

    # Expressions
    'HIRIntegerLiteral',
    'HIRBooleanLiteral',
    'HIREnumVariantExpr',
    'HIRIdentifier',
    'HIRRegister',
    'HIRStatusFlagAccess',
    'HIRIncludeBytesExpr',
    'HIRArrayFillExpr',
    'HIRArrayLiteralExpr',
    'HIRStringLiteral',
    'HIRStructFieldInit',
    'HIRStructLiteralExpr',
    'HIRBinaryOp',
    'HIRUnaryOp',
    'HIRTypeCast',
    'HIRFunctionCall',
    'HIRMethodCall',
    'HIRArrayIndex',
    'HIRFieldAccess',
    'HIRDereference',
    'HIRAddressOf',
    'HIRAssignment',
    'HIRMultiAssignment',
    'HIRFunctionAddress',

    # Pattern Matching
    'HIRPattern',
    'HIRLiteralPattern',
    'HIREnumPattern',
    'HIRWildcardPattern',
    'HIRIdentifierPattern',
    'HIRRangePattern',
    'HIROrPattern',
    'HIRMatchArm',
    'HIRMatchExpression',
    'HIRBlockExpression',
    'HIRIfExpression',
    'HIRLoopExpression',

    # Bindings
    'ParameterBinding',
    'RegisterBinding',
    'VariableBinding',
    'LetBinding',
    'RegisterLetBinding',
    'VariableLetBinding',

# Symbol table
    'Symbol',
    'SymbolKind',
    'ScopeKind',
    'Scope',
    'SymbolTable',

    # Types
    'TypeInfo',
    'BasicTypeInfo',
    'ArrayTypeInfo',
    'PointerTypeInfo',
    'FunctionTypeInfo',
    'StructTypeInfo',
    'EnumTypeInfo',
    'NeverTypeInfo',
    'RegisterTypeInfo',
    'TypeResolver',

    # Attributes
    'ProcessedAttribute',
    'ModeAttribute',
    'PreservesAttribute',
    'StorageAttribute',
    'StorageKind',
    'BankAttribute',
    'DataBankMode',
    'InterruptAttribute',
    'InterruptVector',
    'EntryAttribute',
    'AttributeProcessor',

    # Const evaluation
    'ConstEvaluator',

    # Builder
    'HIRBuilder',
]
