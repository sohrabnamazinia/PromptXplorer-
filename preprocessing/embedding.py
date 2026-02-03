"""
Embedding module for computing and storing prompt embeddings.
"""

import os
import csv
import numpy as np
from openai import OpenAI

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_model.data_models import PromptManager


class Embedding:
    """Computes and stores embeddings for secondary prompts."""
    
    def __init__(self, prompt_manager: PromptManager, api_key: str = None):
        """
        Args:
            prompt_manager: PromptManager object
            api_key: OpenAI API key (if None, uses OPENAI_API_KEY environment variable)
        """
        self.prompt_manager = prompt_manager
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.client = OpenAI(api_key=self.api_key)
        self.model = "text-embedding-3-small"
        self.csv_path = None  # Will be set after calling embed()
    
    def embed(self):
        """
        Embeds all secondary prompts and saves to embeddings_db/secondary_embeddings_<timestamp>.csv.
        
        CSV format: secondary_class_label, secondary_prompt_text, secondary_prompt_vector_embedded
        
        Returns:
            Path to the saved CSV file
        """
        from datetime import datetime
        
        os.makedirs("embeddings_db", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = f"embeddings_db/secondary_embeddings_{timestamp}.csv"
        csv_path = self.csv_path
        
        # Collect all secondary prompts with their classes
        secondary_data = []
        for cp in self.prompt_manager.composite_prompts:
            for sec in cp.secondaries:
                if sec.class_obj:
                    secondary_data.append({
                        'class_label': sec.class_obj.index,
                        'text': sec.text
                    })
        
        # Remove duplicates (same text and class)
        seen = set()
        unique_secondaries = []
        for item in secondary_data:
            key = (item['class_label'], item['text'])
            if key not in seen:
                seen.add(key)
                unique_secondaries.append(item)
        
        # Compute embeddings
        print(f"Computing embeddings for {len(unique_secondaries)} secondary prompts...")
        embeddings = []
        texts = [item['text'] for item in unique_secondaries]
        
        # Batch embeddings (OpenAI supports up to 2048 texts per request)
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = self.client.embeddings.create(
                model=self.model,
                input=batch
            )
            batch_embeddings = [item.embedding for item in response.data]
            embeddings.extend(batch_embeddings)
        
        # Save to CSV
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['secondary_class_label', 'secondary_prompt_text', 'secondary_prompt_vector_embedded'])
            
            for item, embedding in zip(unique_secondaries, embeddings):
                # Convert embedding vector to string representation
                embedding_str = ','.join(map(str, embedding))
                writer.writerow([item['class_label'], item['text'], embedding_str])
        
        print(f"Saved embeddings to {csv_path}")
        return csv_path
