"""
RAG (Retrieval-Augmented Generation) for prompt selection.
"""

import os
import csv
import numpy as np
from openai import OpenAI

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llm.llm_interface import LLMInterface
from preprocessing.embedding import Embedding


class RAG:
    """RAG class for selecting next individual prompts."""
    
    def __init__(self, embedding: Embedding, llm_interface: LLMInterface, top_l: int = 5, csv_path: str = None):
        """
        Args:
            embedding: Embedding object
            llm_interface: LLMInterface object
            top_l: Number of top candidates to retrieve
            csv_path: Path to embeddings CSV file (if None, uses embedding.csv_path or finds most recent)
        """
        self.embedding = embedding
        self.llm_interface = llm_interface
        self.top_l = top_l
        self.embeddings_db = None
        self.csv_path = csv_path
        self._load_embeddings()
    
    def _load_embeddings(self):
        """Load embeddings from CSV file."""
        # Determine CSV path: use provided, or from embedding object, or find most recent
        if self.csv_path:
            csv_path = self.csv_path
        elif self.embedding.csv_path:
            csv_path = self.embedding.csv_path
        else:
            # Find most recent embedding file
            import glob
            embedding_files = glob.glob("embeddings_db/secondary_embeddings_*.csv")
            if embedding_files:
                csv_path = max(embedding_files, key=os.path.getctime)
            else:
                csv_path = None
        
        self.embeddings_db = []
        
        if csv_path and os.path.exists(csv_path):
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    embedding_str = row['secondary_prompt_vector_embedded']
                    embedding_vector = np.array([float(x) for x in embedding_str.split(',')])
                    self.embeddings_db.append({
                        'class_label': int(row['secondary_class_label']),
                        'text': row['secondary_prompt_text'],
                        'embedding': embedding_vector
                    })
            print(f"Loaded embeddings from {csv_path}")
    
    def select_next_individual_prompt(self, current_prompt: str):
        """
        Selects the next secondary prompt to add to current prompt.
        
        Args:
            current_prompt: Current prompt (user input + any already added prompts)
        
        Returns:
            Dictionary with 'selected_prompt' and 'updated_prompt' keys, or None
        """
        if not self.embeddings_db:
            return None
        
        # Embed current prompt
        api_key = os.getenv('OPENAI_API_KEY')
        client = OpenAI(api_key=api_key)
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=[current_prompt]
        )
        current_embedding = np.array(response.data[0].embedding)
        
        # Compute similarities and get top-L
        similarities = []
        for item in self.embeddings_db:
            similarity = np.dot(current_embedding, item['embedding']) / (
                np.linalg.norm(current_embedding) * np.linalg.norm(item['embedding'])
            )
            similarities.append((similarity, item))
        
        # Sort by similarity and get top-L
        similarities.sort(key=lambda x: x[0], reverse=True)
        top_candidates = [item for _, item in similarities[:self.top_l]]
        candidate_texts = [item['text'] for item in top_candidates]
        
        # Print top-L candidates
        print(f"\nTop-{self.top_l} candidate prompts:")
        for i, (sim, item) in enumerate(similarities[:self.top_l], 1):
            print(f"  {i}. [Class {item['class_label']}] Similarity: {sim:.4f} - {item['text']}")
        
        # Call LLM to select one
        result = self.llm_interface.select_next_prompt_rag(current_prompt, candidate_texts)
        return result
