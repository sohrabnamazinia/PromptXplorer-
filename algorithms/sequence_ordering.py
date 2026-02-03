"""
Sequence Ordering algorithm for ordering final composite prompts.
"""

import sys
import os
import numpy as np
from openai import OpenAI

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_model.data_models import PromptManager


class OrderSequence:
    """Orders final composite prompts by diversity."""
    
    def __init__(self, prompt_manager: PromptManager):
        """
        Args:
            prompt_manager: PromptManager object with final_composite_prompts
        """
        self.prompt_manager = prompt_manager
        self.api_key = os.getenv('OPENAI_API_KEY')
        self.client = OpenAI(api_key=self.api_key)
        self.model = "text-embedding-3-small"
    
    def _embed_prompts(self, prompts):
        """Embed all prompts."""
        print(f"Embedding {len(prompts)} prompts...")
        response = self.client.embeddings.create(
            model=self.model,
            input=prompts
        )
        embeddings = [np.array(item.embedding) for item in response.data]
        return embeddings
    
    def _compute_distance(self, emb1, emb2):
        """Compute Euclidean distance between two embeddings."""
        return np.linalg.norm(emb1 - emb2)
    
    def _compute_avg_distance(self, emb, emb_list):
        """Compute average distance from emb to all embeddings in emb_list."""
        if not emb_list:
            return 0
        distances = [self._compute_distance(emb, e) for e in emb_list]
        return np.mean(distances)
    
    def order_sequences(self):
        """
        Order the final composite prompts by diversity.
        
        Returns:
            List of reordered prompt strings
        """
        if not self.prompt_manager.final_composite_prompts:
            return []
        
        prompts = self.prompt_manager.final_composite_prompts
        k = len(prompts)
        
        # Embed all prompts
        embeddings = self._embed_prompts(prompts)
        
        # Greedy ordering: start with first prompt
        ordered_prompts = [prompts[0]]
        ordered_embeddings = [embeddings[0]]
        remaining_indices = list(range(1, k))
        
        # Iteratively select the prompt furthest on average from already selected ones
        for _ in range(k - 1):
            if not remaining_indices:
                break
            
            max_avg_distance = -1
            best_idx = None
            
            for idx in remaining_indices:
                avg_distance = self._compute_avg_distance(embeddings[idx], ordered_embeddings)
                if avg_distance > max_avg_distance:
                    max_avg_distance = avg_distance
                    best_idx = idx
            
            if best_idx is not None:
                ordered_prompts.append(prompts[best_idx])
                ordered_embeddings.append(embeddings[best_idx])
                remaining_indices.remove(best_idx)
        
        # Store ordered prompts in PromptManager
        self.prompt_manager.final_composite_prompts = ordered_prompts
        
        return ordered_prompts
