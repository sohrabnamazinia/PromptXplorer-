"""
Sequence construction algorithms for PromptXplorer.
"""

import random
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_model.data_models import PromptManager
from llm.llm_interface import LLMInterface


class RandomWalk:
    """Random walk algorithm for generating composite class sequences."""
    
    def __init__(self, prompt_manager: PromptManager):
        """
        Args:
            prompt_manager: PromptManager object with clustering results and support matrices
        """
        self.prompt_manager = prompt_manager
        self.llm_interface = LLMInterface()
    
    def walk(self, user_input: str, phi: int):
        """
        Main method to generate a composite class sequence.
        
        Args:
            user_input: User's input primary prompt
            phi: Number of secondary classes to generate
        
        Returns:
            List of class indices: [primary_class, secondary1_class, ..., secondary_phi_class]
        """
        # Step 1: Choose primary class
        primary_class = self.walk_init_primary(user_input)
        
        # Step 2: Choose first secondary class from primary
        secondary_classes = []
        current_primary = primary_class
        
        # Step 3: Choose first secondary
        first_secondary = self.walk_primary_secondary(current_primary)
        secondary_classes.append(first_secondary)
        
        # Step 4: Choose remaining secondaries (phi - 1 more)
        current_secondary = first_secondary
        for _ in range(phi - 1):
            next_secondary = self.walk_secondary_secondary(current_secondary)
            secondary_classes.append(next_secondary)
            current_secondary = next_secondary
        
        return [primary_class] + secondary_classes
    
    def random_walk_iter(self, user_input: str, phi: int, large_k: int):
        """
        Runs random walk multiple times to generate multiple composite class sequences.
        
        Args:
            user_input: User's input primary prompt
            phi: Number of secondary classes to generate per sequence
            large_k: Number of times to run random walk
        
        Returns:
            List of composite class sequences, each is [primary_class, secondary1_class, ..., secondary_phi_class]
        """
        sequences = []
        for _ in range(large_k):
            sequence = self.walk(user_input, phi)
            sequences.append(sequence)
        return sequences
    
    def walk_init_primary(self, user_input: str):
        """
        Uses LLM to choose primary class from user input.
        
        Args:
            user_input: User's input primary prompt
        
        Returns:
            Primary class index
        """
        # Get all primary classes with their descriptions and sample prompts
        primary_prompts = self.prompt_manager.get_all_primary_prompts()
        primary_classes_info = {}
        
        for prompt in primary_prompts:
            if prompt.class_obj:
                class_idx = prompt.class_obj.index
                if class_idx not in primary_classes_info:
                    primary_classes_info[class_idx] = {
                        'description': prompt.class_obj.description,
                        'samples': []
                    }
                primary_classes_info[class_idx]['samples'].append(prompt.text)
        
        # Prepare context for LLM
        classes_text = []
        for class_idx, info in sorted(primary_classes_info.items()):
            samples = info['samples'][:3]  # Limit to 3 samples per class
            samples_text = "\n".join([f"  - {s}" for s in samples])
            classes_text.append(f"Class {class_idx}: {info['description']}\nSamples:\n{samples_text}")
        
        classes_context = "\n\n".join(classes_text)
        
        # Call LLM to choose primary class
        class_idx = self.llm_interface.select_primary_class(user_input, classes_context)
        
        # Validate the selected class
        if class_idx is not None and class_idx in primary_classes_info:
            return class_idx
        
        # Fallback: return first available class
        if primary_classes_info:
            return min(primary_classes_info.keys())
        return None
    
    def walk_primary_secondary(self, primary_class: int):
        """
        Chooses first secondary class from primary class using support values.
        Considers ALL secondary classes, using 0.1 for missing edges.
        
        Args:
            primary_class: Primary class index
        
        Returns:
            Secondary class index
        """
        # Get all secondary classes
        all_secondary_classes = set()
        for cp in self.prompt_manager.composite_prompts:
            for sec in cp.secondaries:
                if sec.class_obj:
                    all_secondary_classes.add(sec.class_obj.index)
        
        if not all_secondary_classes:
            return None
        
        # Build support values for all secondary classes
        # If edge exists, use support value; if not, use 0.1
        candidates = {}
        for sec_class in all_secondary_classes:
            if self.prompt_manager.primary_to_secondary_support:
                support = self.prompt_manager.primary_to_secondary_support.get((primary_class, sec_class), 0.1)
            else:
                support = 0.1
            candidates[sec_class] = support
        
        # Non-uniform sampling based on support values
        secondary_classes = list(candidates.keys())
        weights = [candidates[c] for c in secondary_classes]
        total_weight = sum(weights)
        probabilities = [w / total_weight for w in weights]
        
        chosen = np.random.choice(secondary_classes, p=probabilities)
        return chosen
    
    def walk_secondary_secondary(self, secondary_class: int):
        """
        Chooses next secondary class from current secondary class using support values.
        Considers ALL secondary classes, using 0.1 for missing edges.
        
        Args:
            secondary_class: Current secondary class index
        
        Returns:
            Next secondary class index
        """
        # Get all secondary classes
        all_secondary_classes = set()
        for cp in self.prompt_manager.composite_prompts:
            for sec in cp.secondaries:
                if sec.class_obj:
                    all_secondary_classes.add(sec.class_obj.index)
        
        if not all_secondary_classes:
            return None
        
        # Build support values for all secondary classes
        # If edge exists, use support value; if not, use 0.1
        candidates = {}
        for sec_class in all_secondary_classes:
            if self.prompt_manager.secondary_to_secondary_support:
                support = self.prompt_manager.secondary_to_secondary_support.get((secondary_class, sec_class), 0.1)
            else:
                support = 0.1
            candidates[sec_class] = support
        
        # Non-uniform sampling based on support values
        next_secondary_classes = list(candidates.keys())
        weights = [candidates[c] for c in next_secondary_classes]
        total_weight = sum(weights)
        probabilities = [w / total_weight for w in weights]
        
        chosen = np.random.choice(next_secondary_classes, p=probabilities)
        return chosen
