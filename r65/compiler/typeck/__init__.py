# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Type checking module for R65 compiler.

Performs:
- Processor mode tracking through control flow
- Type inference (limited)
- Type checking of expressions and statements
- Operator restriction validation
- Register preservation validation
"""

from r65.compiler.typeck.errors import TypeCheckError, TypeCheckWarning
from r65.compiler.typeck.processor_mode import ProcessorMode, ModeState, XModeState
from r65.compiler.typeck.type_checker import TypeChecker
from r65.compiler.typeck.type_utils import TypeUtils
from r65.compiler.typeck.operator_validator import OperatorValidator
from r65.compiler.typeck.mode_tracker import ModeTracker
from r65.compiler.typeck.cfg_builder import CFGBuilder, CFG, BasicBlock
from r65.compiler.typeck.preservation_checker import PreservationChecker
from r65.compiler.typeck.type_inference import TypeInference

__all__ = [
    # Errors
    'TypeCheckError',
    'TypeCheckWarning',

    # Main type checker
    'TypeChecker',

    # Mode tracking
    'ProcessorMode',
    'ModeState',
    'XModeState',
    'ModeTracker',

    # CFG
    'CFGBuilder',
    'CFG',
    'BasicBlock',

    # Utilities
    'TypeUtils',
    'OperatorValidator',
    'PreservationChecker',
    'TypeInference',
]
