"""
Tests for conditional compilation (cfg) functionality.
"""

import pytest
from r65.compiler.frontend import parse, ast
from r65.compiler.hir.cfg import CfgEvaluator, CfgIdentifier, CfgComparison, CfgNot, CfgAny, CfgAll


def test_cfg_simple_identifier():
    """Test parsing simple cfg identifier."""
    source = """
    #[cfg(snes)]
    fn snes_function() {
        A = 42;
    }
    """
    
    program = parse(source)
    func = program.items[0]
    
    assert len(func.attributes) == 1
    cfg_attr = func.attributes[0]
    assert cfg_attr.name == 'cfg'
    
    # Get the condition from the attribute argument
    condition = cfg_attr.args[0].value
    assert isinstance(condition, CfgIdentifier)
    assert condition.name == 'snes'
    
    print("✓ Simple cfg identifier test passed")


def test_cfg_key_value_comparison():
    """Test parsing cfg with key-value comparison."""
    source = """
    #[cfg(target = "snes")]
    fn snes_specific() {
        A = 42;
    }
    """
    
    program = parse(source)
    func = program.items[0]
    
    cfg_attr = func.attributes[0]
    condition = cfg_attr.args[0].value
    assert isinstance(condition, CfgComparison)
    assert condition.key == 'target'
    assert condition.operator == '='
    assert condition.value == 'snes'
    
    print("✓ CFG key-value comparison test passed")


def test_cfg_not_condition():
    """Test parsing cfg with not condition."""
    source = """
    #[cfg(not(debug))]
    fn release_function() {
        A = 42;
    }
    """
    
    program = parse(source)
    func = program.items[0]
    
    cfg_attr = func.attributes[0]
    condition = cfg_attr.args[0].value
    assert isinstance(condition, CfgNot)
    assert isinstance(condition.condition, CfgIdentifier)
    assert condition.condition.name == 'debug'
    
    print("✓ CFG not condition test passed")


def test_cfg_any_condition():
    """Test parsing cfg with any (||) condition."""
    source = """
    #[cfg(snes || genesis)]
    fn retro_function() {
        A = 42;
    }
    """
    
    program = parse(source)
    func = program.items[0]
    
    cfg_attr = func.attributes[0]
    condition = cfg_attr.args[0].value
    assert isinstance(condition, CfgAny)
    assert len(condition.conditions) == 2
    
    assert isinstance(condition.conditions[0], CfgIdentifier)
    assert condition.conditions[0].name == 'snes'
    assert isinstance(condition.conditions[1], CfgIdentifier)
    assert condition.conditions[1].name == 'genesis'
    
    print("✓ CFG any condition test passed")


def test_cfg_all_condition():
    """Test parsing cfg with all (&&) condition."""
    source = """
    #[cfg(snes && debug)]
    fn debug_snes_function() {
        A = 42;
    }
    """
    
    program = parse(source)
    func = program.items[0]
    
    cfg_attr = func.attributes[0]
    condition = cfg_attr.args[0].value
    assert isinstance(condition, CfgAll)
    assert len(condition.conditions) == 2
    
    assert isinstance(condition.conditions[0], CfgIdentifier)
    assert condition.conditions[0].name == 'snes'
    assert isinstance(condition.conditions[1], CfgIdentifier)
    assert condition.conditions[1].name == 'debug'
    
    print("✓ CFG all condition test passed")


def test_cfg_complex_condition():
    """Test parsing complex nested cfg condition."""
    source = """
    #[cfg((snes || genesis) && debug)]
    fn complex_condition_function() {
        A = 42;
    }
    """
    
    program = parse(source)
    func = program.items[0]
    
    cfg_attr = func.attributes[0]
    condition = cfg_attr.args[0].value
    assert isinstance(condition, CfgAll)
    assert len(condition.conditions) == 2
    
    # First condition: (snes || genesis)
    assert isinstance(condition.conditions[0], CfgAny)
    any_condition = condition.conditions[0]
    assert len(any_condition.conditions) == 2
    assert isinstance(any_condition.conditions[0], CfgIdentifier)
    assert any_condition.conditions[0].name == 'snes'
    assert isinstance(any_condition.conditions[1], CfgIdentifier)
    assert any_condition.conditions[1].name == 'genesis'
    
    # Second condition: debug
    assert isinstance(condition.conditions[1], CfgIdentifier)
    assert condition.conditions[1].name == 'debug'
    
    print("✓ Complex cfg condition test passed")


class TestCfgEvaluator:
    """Test cfg condition evaluation."""
    
    def test_evaluator_simple_features(self):
        """Test evaluator with simple feature flags."""
        evaluator = CfgEvaluator.from_string_list(['snes', 'debug'])
        
        assert evaluator.evaluate(CfgIdentifier(name='snes')) == True
        assert evaluator.evaluate(CfgIdentifier(name='debug')) == True
        assert evaluator.evaluate(CfgIdentifier(name='genesis')) == False
    
    def test_evaluator_key_values(self):
        """Test evaluator with key-value pairs."""
        evaluator = CfgEvaluator.from_string_list(['target=snes', 'platform=console'])
        
        assert evaluator.evaluate(CfgComparison(key='target', operator='=', value='snes')) == True
        assert evaluator.evaluate(CfgComparison(key='target', operator='=', value='genesis')) == False
        assert evaluator.evaluate(CfgComparison(key='target', operator='!=', value='genesis')) == True
        assert evaluator.evaluate(CfgComparison(key='target', operator='!=', value='snes')) == False
    
    def test_evaluator_not(self):
        """Test evaluator with not condition."""
        evaluator = CfgEvaluator.from_string_list(['debug'])
        
        # not(debug) should be false
        not_debug = CfgNot(condition=CfgIdentifier(name='debug'))
        assert evaluator.evaluate(not_debug) == False
        
        # not(release) should be true (release not in features)
        not_release = CfgNot(condition=CfgIdentifier(name='release'))
        assert evaluator.evaluate(not_release) == True
    
    def test_evaluator_any(self):
        """Test evaluator with any (||) condition."""
        evaluator = CfgEvaluator.from_string_list(['snes'])
        
        # snes || genesis should be true (snes is enabled)
        any_condition = CfgAny(conditions=[
            CfgIdentifier(name='snes'),
            CfgIdentifier(name='genesis')
        ])
        assert evaluator.evaluate(any_condition) == True
        
        # genesis || n64 should be false (neither enabled)
        any_condition = CfgAny(conditions=[
            CfgIdentifier(name='genesis'),
            CfgIdentifier(name='n64')
        ])
        assert evaluator.evaluate(any_condition) == False
    
    def test_evaluator_all(self):
        """Test evaluator with all (&&) condition."""
        evaluator = CfgEvaluator.from_string_list(['snes', 'debug'])
        
        # snes && debug should be true (both enabled)
        all_condition = CfgAll(conditions=[
            CfgIdentifier(name='snes'),
            CfgIdentifier(name='debug')
        ])
        assert evaluator.evaluate(all_condition) == True
        
        # snes && genesis should be false (genesis not enabled)
        all_condition = CfgAll(conditions=[
            CfgIdentifier(name='snes'),
            CfgIdentifier(name='genesis')
        ])
        assert evaluator.evaluate(all_condition) == False
    
    def test_evaluator_complex(self):
        """Test evaluator with complex nested conditions."""
        evaluator = CfgEvaluator.from_string_list(['snes', 'debug'])
        
        # (snes || genesis) && debug should be true
        complex_condition = CfgAll(conditions=[
            CfgAny(conditions=[
                CfgIdentifier(name='snes'),
                CfgIdentifier(name='genesis')
            ]),
            CfgIdentifier(name='debug')
        ])
        assert evaluator.evaluate(complex_condition) == True
        
        # (n64 || genesis) && debug should be false (no platforms enabled)
        complex_condition = CfgAll(conditions=[
            CfgAny(conditions=[
                CfgIdentifier(name='n64'),
                CfgIdentifier(name='genesis')
            ]),
            CfgIdentifier(name='debug')
        ])
        assert evaluator.evaluate(complex_condition) == False


if __name__ == '__main__':
    # Run basic parsing tests
    test_cfg_simple_identifier()
    test_cfg_key_value_comparison()
    test_cfg_not_condition()
    test_cfg_any_condition()
    test_cfg_all_condition()
    test_cfg_complex_condition()
    
    # Run evaluator tests
    test_evaluator = TestCfgEvaluator()
    test_evaluator.test_evaluator_simple_features()
    test_evaluator.test_evaluator_key_values()
    test_evaluator.test_evaluator_not()
    test_evaluator.test_evaluator_any()
    test_evaluator.test_evaluator_all()
    test_evaluator.test_evaluator_complex()
    
    print("✓ All cfg tests passed!")