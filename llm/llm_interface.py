"""
LLM interface for PromptXplorer framework.
"""

import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


class LLMInterface:
    """Interface for LLM operations using OpenAI."""
    
    def __init__(self, api_key: str = None, model: str = "gpt-4o"):
        """
        Args:
            api_key: API key for OpenAI (if None, uses OPENAI_API_KEY environment variable)
            model: Model name to use
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.model = model
        self.llm = ChatOpenAI(
            api_key=self.api_key,
            model=model,
            temperature=0
        )
    
    def decompose_prompts(self, prompts: list, batch_size: int = None):
        """
        Decomposes full prompts into primary + satellite prompts using LLM.
        
        Args:
            prompts: List of full prompt strings
            batch_size: Batch size for processing (if None, processes all at once)
        
        Returns:
            List of dictionaries with 'primary' and 'secondaries' keys
        """
        if batch_size is None:
            batch_size = len(prompts)
        
        results = []
        
        # Process in batches
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            batch_results = self._decompose_batch(batch)
            results.extend(batch_results)
        
        return results
    
    def _decompose_batch(self, prompts: list):
        """Decomposes a batch of prompts."""
        import json
        
        results = []
        
        for prompt in prompts:
            # Create prompt template
            template = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful assistant that decomposes prompts into a primary prompt and secondary prompts."),
                ("user", """Given the following prompt, decompose it into:
1. A primary prompt (the core objective)
2. Secondary prompts (auxiliary details that enrich it)

Return a JSON object with two keys: "primary" and "secondaries".
- "primary": string (the primary prompt)
- "secondaries": array of strings (the secondary prompts)

Example JSON:
{{"primary": "Create a portrait of Elon Musk", "secondaries": ["digital art", "high quality", "trending"]}}

Prompt: {prompt}

Return only the JSON object:""")
            ])
            
            # Invoke LLM
            chain = template | self.llm
            response = chain.invoke({"prompt": prompt})
            
            # Parse JSON response
            try:
                response_text = response.content.strip()
                # Remove markdown code blocks if present
                if response_text.startswith("```"):
                    response_text = response_text.split("```")[1]
                    if response_text.startswith("json"):
                        response_text = response_text[4:]
                response_text = response_text.strip()
                
                decomposed = json.loads(response_text)
                results.append({
                    'primary': decomposed['primary'],
                    'secondaries': decomposed['secondaries']
                })
            except (json.JSONDecodeError, KeyError):
                # Fallback: use original prompt as primary
                results.append({
                    'primary': prompt,
                    'secondaries': []
                })
        
        return results
    
    def generate_class_description(self, prompts: list, max_tokens: int = None):
        """
        Generates a short description for a class given its prompts.
        
        Args:
            prompts: List of prompt strings belonging to the class
            max_tokens: Maximum tokens to use (if None, uses default)
        
        Returns:
            Description string (few words)
        """
        if not prompts:
            return "unknown"
        
        # Limit prompts to avoid exceeding token limits
        max_prompts = min(10, len(prompts))
        sample_prompts = prompts[:max_prompts]
        
        # Create prompt template
        prompts_text = "\n".join([f"- {p}" for p in sample_prompts])
        
        template = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant that generates concise class descriptions."),
            ("user", """Given the following prompts that belong to the same class, generate a short description (2-4 words) that captures the common theme or category.

Prompts:
{prompts}

Provide a concise description (2-4 words only):""")
        ])
        
        # Invoke LLM
        chain = template | self.llm
        response = chain.invoke({"prompts": prompts_text})
        
        description = response.content.strip()
        
        # Clean up description (remove quotes, extra words)
        description = description.strip('"\'')
        if len(description.split()) > 5:
            # If too long, take first few words
            description = " ".join(description.split()[:4])
        
        return description.lower()
    
    def select_primary_class(self, user_input: str, classes_context: str):
        """
        Selects the most appropriate primary class for a user input prompt.
        
        Args:
            user_input: User's input primary prompt
            classes_context: Formatted string with available primary classes, descriptions, and samples
        
        Returns:
            Selected primary class index (int) or None if parsing fails
        """
        template = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant that selects the most appropriate class for a user prompt."),
            ("user", """Given the user input prompt, choose the most appropriate primary class.

User input: {user_input}

Available primary classes:
{classes_context}

Return only the class index (just the number) that best matches the user input:""")
        ])
        
        chain = template | self.llm
        response = chain.invoke({"user_input": user_input, "classes_context": classes_context})
        response_text = response.content.strip()
        
        # Extract class index from response
        try:
            import re
            numbers = re.findall(r'\d+', response_text)
            if numbers:
                return int(numbers[0])
        except:
            pass
        
        return None
