"""
End-to-end test for Phase 1, Phase 2, and Phase 3.
"""

from data_model.load_data import DataLoader
from preprocessing.clusterer import Clustering
from preprocessing.embedding import Embedding
from algorithms.sequence_construction import RandomWalk
from algorithms.k_set_coverage import KSetCoverage
from algorithms.prompt_selector import IndividualPromptSelector
from algorithms.sequence_ordering import OrderSequence
from llm.rag import RAG
from llm.llm_interface import LLMInterface

# Phase 1: Load data
print("Phase 1: Loading data...")
loader = DataLoader(separated=True, n=None)
pm = loader.load_data("data/test_prompts.csv")
print(f"Loaded {len(pm.composite_prompts)} composite prompts")

# Phase 2: Cluster
algorithm='kmeans'
print("\nPhase 2: Clustering...")
clusterer = Clustering(pm, algorithm=algorithm)
algorithm_params = {
    'primary': {'n_clusters': 5},
    'secondary': {'n_clusters': 10}
}
pm = clusterer.cluster(algorithm_params)
print("Clustering completed")

# Save PromptManager
print("\nSaving PromptManager...")
csv_file = "data/test_prompts.csv"
output_name = "hi"
folder_name = pm.save(output_name, algorithm=algorithm, csv_filename=csv_file)
print(f"Saved to prompt_manager_objects/{folder_name}/")

# Embedding (needed for RAG)
print("\nEmbedding secondary prompts...")
embedding = Embedding(pm)
embedding.embed()

# Phase 3.1: Random Walk
print("\nPhase 3.1: Random Walk...")
random_walk = RandomWalk(pm)
user_input = "Create a portrait of a famous person"
phi = 4  # Number of secondary classes
large_k = 20  # Generate 20 sequences
composite_class_sequences = random_walk.random_walk_iter(user_input, phi, large_k)
print(f"Generated {len(composite_class_sequences)} composite class sequences:")
for i, sequence in enumerate(composite_class_sequences, 1):
    print(f"  Sequence {i}: Primary={sequence[0]}, Secondaries={sequence[1:]}")


# Phase 3.2: K-Set Coverage
print("\nPhase 3.2: K-Set Coverage...")
k_set_coverage = KSetCoverage(pm, composite_class_sequences)
k = 5  # Select 5 sequences
selected_sequences = k_set_coverage.run_greedy_coverage(k)
print(f"Selected {len(selected_sequences)} sequences:")
for i, sequence in enumerate(selected_sequences, 1):
    print(f"  Sequence {i}: Primary={sequence[0]}, Secondaries={sequence[1:]}")


# Phase 3.3: Prompt Selector
print("\nPhase 3.3: Prompt Selector...")
llm_interface = LLMInterface()
rag = RAG(embedding, llm_interface, top_l=5)
prompt_selector = IndividualPromptSelector(pm, rag)
prompt_selector.select_prompts(user_input, phi)

# Read final prompts from PromptManager
final_prompts = pm.final_composite_prompts
print(f"\n✓ Generated {len(final_prompts)} composite prompts (before ordering):")
for i, prompt in enumerate(final_prompts, 1):
    print(f"\n  Prompt {i}:")
    print(f"  {prompt}")

# Phase 3.4: Sequence Ordering
print("\nPhase 3.4: Sequence Ordering...")
order_sequence = OrderSequence(pm)
ordered_prompts = order_sequence.order_sequences()

print(f"\n✓ Final ordered {len(ordered_prompts)} composite prompts (after ordering):")
for i, prompt in enumerate(ordered_prompts, 1):
    print(f"\n  Prompt {i}:")
    print(f"  {prompt}")

print("\n✓ Phase 1, 2 & 3 completed successfully!")
