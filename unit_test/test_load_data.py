"""
Unit tests for DataLoader class.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_model.load_data import DataLoader
from data_model.data_models import PromptManager


def test_load_data_separated_true():
    """Test loading data when separated=True."""
    csv_path = os.path.join(os.path.dirname(__file__), "test_separated.csv")
    
    loader = DataLoader(separated=True, n=None)
    pm = loader.load_data(csv_path)
    
    # Assertions
    assert isinstance(pm, PromptManager)
    assert len(pm.composite_prompts) == 3
    
    # Check first composite prompt
    cp1 = pm.composite_prompts[0]
    assert cp1.primary.text == 'Portrait of Elon Musk'
    assert len(cp1.secondaries) == 3
    assert cp1.secondaries[0].text == 'digital art'
    assert cp1.secondaries[1].text == 'high quality'
    assert cp1.secondaries[2].text == 'trending'
    
    print("✓ test_load_data_separated_true passed")


def test_load_data_separated_false():
    """Test loading data when separated=False (uses LLM decomposition)."""
    csv_path = os.path.join(os.path.dirname(__file__), "test_non_separated.csv")
    
    loader = DataLoader(separated=False, n=None)
    pm = loader.load_data(csv_path, batch_size=10)
    
    # Assertions
    assert isinstance(pm, PromptManager)
    assert len(pm.composite_prompts) == 2
    
    # Check that prompts were decomposed
    cp1 = pm.composite_prompts[0]
    assert cp1.primary.text is not None
    assert len(cp1.secondaries) > 0
    
    print("✓ test_load_data_separated_false passed")


if __name__ == '__main__':
    test_load_data_separated_true()
    test_load_data_separated_false()
    print("\nAll tests passed!")
