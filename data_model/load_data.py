"""
Data loader for PromptXplorer framework.
"""

import csv
from .data_models import PromptManager, CompositePrompt, PrimaryPrompt, SecondaryPrompt


class DataLoader:
    """Loads and parses prompt data from CSV files."""
    
    def __init__(self, separated: bool = True, n: int = None):
        """
        Args:
            separated: If True, CSV has primary and secondaries already separated.
                      If False, CSV has single column with full prompts (needs LLM decomposition).
            n: Number of rows to consider from CSV (None = all rows)
        """
        self.separated = separated
        self.n = n
    
    def load_data(self, csv_path: str, batch_size: int = None):
        """
        Loads data from CSV file and returns PromptManager.
        
        Args:
            csv_path: Path to CSV file
            batch_size: Batch size for LLM processing (if separated=False)
        
        Returns:
            PromptManager object
        """
        pm = PromptManager()
        
        if self.separated:
            pm = self._load_separated(csv_path)
        else:
            import sys
            import os
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from llm.llm_interface import LLMInterface
            llm_interface = LLMInterface()
            pm = self._load_with_decomposition(csv_path, llm_interface, batch_size)
        
        return pm
    
    def _load_separated(self, csv_path: str):
        """Loads CSV where primary and secondaries are already separated."""
        pm = PromptManager()
        
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            
            # Skip header if it exists
            first_row = next(reader, None)
            if first_row and first_row[0].strip().lower() == 'prompt':
                pass  # Header was skipped
            else:
                # First row is data, process it
                if first_row:
                    self._process_row(first_row, pm)
            
            # Process remaining rows
            count = 0
            for row in reader:
                if self.n and count >= self.n:
                    break
                self._process_row(row, pm)
                count += 1
        
        return pm
    
    def _process_row(self, row: list, pm: PromptManager):
        """Processes a single CSV row into a CompositePrompt."""
        if not row or not row[0].strip():
            return
        
        # First column is primary, rest are secondaries
        primary_text = row[0].strip()
        primary = PrimaryPrompt(primary_text)
        
        secondaries = []
        for i in range(1, len(row)):
            sec_text = row[i].strip()
            if sec_text:
                secondaries.append(SecondaryPrompt(sec_text))
        
        cp = CompositePrompt(primary, secondaries)
        pm.composite_prompts.append(cp)
    
    def _load_with_decomposition(self, csv_path: str, llm_interface, batch_size: int):
        """Loads CSV with single column and uses LLM to decompose prompts."""
        pm = PromptManager()
        
        # Read all prompts - each line is one complete prompt
        prompts = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
            # Skip header if it's "prompt"
            start_idx = 0
            if lines and lines[0].strip().lower() == 'prompt':
                start_idx = 1
            
            count = 0
            for line in lines[start_idx:]:
                if self.n and count >= self.n:
                    break
                prompt = line.strip()
                if prompt:  # Skip empty lines
                    prompts.append(prompt)
                count += 1
        
        # Decompose using LLM
        decomposed = llm_interface.decompose_prompts(prompts, batch_size)
        
        # Create CompositePrompts from decomposed data
        for item in decomposed:
            primary = PrimaryPrompt(item['primary'])
            secondaries = [SecondaryPrompt(sec) for sec in item['secondaries']]
            cp = CompositePrompt(primary, secondaries)
            pm.composite_prompts.append(cp)
        
        return pm
