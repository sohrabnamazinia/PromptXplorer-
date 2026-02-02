"""
Unit tests for Clustering class.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_model.load_data import DataLoader
from data_model.data_models import PromptManager
from clustering.clusterer import Clustering


def test_clustering():
    """Test clustering with kmeans algorithm."""
    # Load data first
    csv_path = os.path.join(os.path.dirname(__file__), "test_clustering.csv")
    loader = DataLoader(separated=True, n=None)
    pm = loader.load_data(csv_path)
    
    print(f"Loaded {len(pm.composite_prompts)} composite prompts")
    
    # Before clustering
    primary_prompts = pm.get_all_primary_prompts()
    all_secondaries = []
    for cp in pm.composite_prompts:
        all_secondaries.extend([sec.text for sec in cp.secondaries])
    
    print("\n=== BEFORE CLUSTERING ===")
    print(f"Number of primary prompts: {len(primary_prompts)}")
    print(f"Number of secondary prompts: {len(all_secondaries)}")
    
    # Cluster
    clusterer = Clustering(pm, algorithm='kmeans')
    algorithm_params = {
        'primary': {'n_clusters': 5},
        'secondary': {'n_clusters': 8}
    }
    
    pm = clusterer.cluster(algorithm_params)
    
    # After clustering - Primary classes
    print("\n=== AFTER CLUSTERING - PRIMARY CLASSES ===")
    primary_prompts = pm.get_all_primary_prompts()
    primary_classes = {}
    for prompt in primary_prompts:
        if prompt.class_obj:
            class_idx = prompt.class_obj.index
            if class_idx not in primary_classes:
                primary_classes[class_idx] = {
                    'description': prompt.class_obj.description,
                    'prompts': []
                }
            primary_classes[class_idx]['prompts'].append(prompt.text)
    
    for class_idx in sorted(primary_classes.keys()):
        cls_info = primary_classes[class_idx]
        print(f"\nClass {class_idx}: {cls_info['description']}")
        print(f"  Prompts ({len(cls_info['prompts'])}):")
        for prompt_text in cls_info['prompts']:
            print(f"    - {prompt_text}")
    
    # After clustering - Secondary classes
    print("\n=== AFTER CLUSTERING - SECONDARY CLASSES ===")
    secondary_classes = {}
    for cp in pm.composite_prompts:
        for sec in cp.secondaries:
            if sec.class_obj:
                class_idx = sec.class_obj.index
                if class_idx not in secondary_classes:
                    secondary_classes[class_idx] = {
                        'description': sec.class_obj.description,
                        'prompts': []
                    }
                secondary_classes[class_idx]['prompts'].append(sec.text)
    
    for class_idx in sorted(secondary_classes.keys()):
        cls_info = secondary_classes[class_idx]
        print(f"\nClass {class_idx}: {cls_info['description']}")
        print(f"  Prompts ({len(cls_info['prompts'])}):")
        for prompt_text in cls_info['prompts']:
            print(f"    - {prompt_text}")
    
    # Support matrices
    print("\n=== PRIMARY TO SECONDARY SUPPORT ===")
    if pm.primary_to_secondary_support:
        sorted_support = sorted(pm.primary_to_secondary_support.items(), key=lambda x: x[1], reverse=True)
        for (primary_class, secondary_class), support in sorted_support:
            print(f"Primary Class {primary_class} -> Secondary Class {secondary_class}: support = {support}")
    else:
        print("Primary to secondary support matrix is empty")
    
    print("\n=== SECONDARY TO SECONDARY SUPPORT ===")
    if pm.secondary_to_secondary_support:
        sorted_support = sorted(pm.secondary_to_secondary_support.items(), key=lambda x: x[1], reverse=True)
        for (secondary_class1, secondary_class2), support in sorted_support:
            print(f"Secondary Class {secondary_class1} -> Secondary Class {secondary_class2}: support = {support}")
    else:
        print("Secondary to secondary support matrix is empty")
    
    # Assertions
    assert isinstance(pm, PromptManager)
    assert pm.primary_to_secondary_support is not None
    assert pm.secondary_to_secondary_support is not None
    assert len(pm.primary_to_secondary_support) > 0 or len(pm.secondary_to_secondary_support) > 0
    
    pm.get_composite_classes()
    assert pm.composite_classes is not None
    
    print("\n✓ test_clustering passed")


if __name__ == '__main__':
    test_clustering()
    print("\nAll tests passed!")
