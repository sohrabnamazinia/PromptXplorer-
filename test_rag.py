"""
Test RAG implementation.
"""

from data_model.load_data import DataLoader
from preprocessing.clusterer import Clustering
from preprocessing.embedding import Embedding
from llm.rag import RAG
from llm.llm_interface import LLMInterface

# Phase 1: Load data
print("Phase 1: Loading data...")
loader = DataLoader(separated=True, n=None)
pm = loader.load_data("data/test_prompts.csv")
print(f"Loaded {len(pm.composite_prompts)} composite prompts")

# Phase 2: Cluster
algorithm = 'kmeans'
print("\nPhase 2: Clustering...")
clusterer = Clustering(pm, algorithm=algorithm)
algorithm_params = {
    'primary': {'n_clusters': 5},
    'secondary': {'n_clusters': 10}
}
pm = clusterer.cluster(algorithm_params)
print("Clustering completed")

# Embedding
print("\nEmbedding secondary prompts...")
embedding = Embedding(pm)
embedding.embed()

# Test RAG
print("\nTesting RAG...")
llm_interface = LLMInterface()
rag = RAG(embedding, llm_interface, top_l=5)
current_prompt = "Create an imaginary portrait"
result = rag.select_next_individual_prompt(current_prompt)

if result:
    print(f"Selected prompt: {result['selected_prompt']}")
    print(f"Updated prompt: {result['updated_prompt']}")
else:
    print("RAG selection failed")

print("\n✓ RAG test completed!")
