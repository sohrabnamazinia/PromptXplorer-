"""
Prompt Selector algorithm for selecting actual prompt instances.
"""

import sys
import os
import numpy as np
from openai import OpenAI

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_model.data_models import PromptManager
from llm.rag import RAG
from llm.llm_interface import LLMInterface


class IndividualPromptSelector:
    """Selects actual prompt instances for composite class sequences."""
    
    def __init__(self, prompt_manager: PromptManager, rag: RAG):
        """
        Args:
            prompt_manager: PromptManager object with k_class_sequences
            rag: RAG object for selecting secondary prompts
        """
        self.prompt_manager = prompt_manager
        self.rag = rag
        self.llm_interface = LLMInterface()
        self.completed_prompts = []  # Store previously completed prompts
    
    def _select_secondary_prompt_with_context(self, current_prompt: str, secondary_class_index: int):
        """Select secondary prompt considering previously completed prompts."""
        # Get candidates from RAG (top-L similar prompts)
        # But we need to filter by class - RAG doesn't do this, so we'll need to modify
        
        # For now, get candidates from embeddings that match the class
        candidates = []
        if self.rag.embeddings_db:
            for item in self.rag.embeddings_db:
                if item['class_label'] == secondary_class_index:
                    candidates.append(item['text'])
        
        if not candidates:
            return None
        
        # Get top-L candidates by similarity to current prompt
        if len(candidates) > self.rag.top_l:
            api_key = os.getenv('OPENAI_API_KEY')
            client = OpenAI(api_key=api_key)
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=[current_prompt]
            )
            current_embedding = np.array(response.data[0].embedding)
            
            similarities = []
            for candidate in candidates:
                # Find embedding for this candidate
                for item in self.rag.embeddings_db:
                    if item['text'] == candidate:
                        similarity = np.dot(current_embedding, item['embedding']) / (
                            np.linalg.norm(current_embedding) * np.linalg.norm(item['embedding'])
                        )
                        similarities.append((similarity, candidate))
                        break
            
            similarities.sort(key=lambda x: x[0], reverse=True)
            candidates = [cand for _, cand in similarities[:self.rag.top_l]]
        
        # Call LLM with context of previously completed prompts
        result = self.llm_interface.select_next_prompt_rag(
            current_prompt, 
            candidates, 
            self.completed_prompts
        )
        
        return result
    
    def select_prompts(self, user_input: str, phi: int):
        """
        Select actual prompt instances for k sequences.
        
        Args:
            user_input: Initial user input prompt
            phi: Number of secondary prompts to add
        
        Returns:
            List of k complete composite prompts (strings)
        """
        if not self.prompt_manager.k_class_sequences:
            return []
        
        k_prompts = []
        self.completed_prompts = []  # Reset for each call
        
        for seq_idx, class_sequence in enumerate(self.prompt_manager.k_class_sequences):
            # class_sequence is [primary_class, secondary_class_1, ..., secondary_class_phi]
            secondary_classes = class_sequence[1:phi+1] if len(class_sequence) > 1 else []
            
            # Start with user input
            current_prompt = user_input
            
            # Iteratively add secondary prompts phi times
            for secondary_class in secondary_classes[:phi]:
                result = self._select_secondary_prompt_with_context(current_prompt, secondary_class)
                if result and 'updated_prompt' in result:
                    current_prompt = result['updated_prompt']
                else:
                    # Fallback: just append a candidate if available
                    candidates = [item['text'] for item in self.rag.embeddings_db 
                                if item['class_label'] == secondary_class]
                    if candidates:
                        current_prompt = f"{current_prompt}, {candidates[0]}"
            
            k_prompts.append(current_prompt)
            self.completed_prompts.append(current_prompt)
        
        # Store in PromptManager
        self.prompt_manager.final_composite_prompts = k_prompts
        
        return k_prompts
