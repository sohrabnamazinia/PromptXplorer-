"""
End-to-end test for Phase 1, Phase 2, and Phase 3.
"""

from data_model.load_data import DataLoader
from clustering.clusterer import Clustering
from algorithms.sequence_construction import RandomWalk

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

# Phase 3: Random Walk
print("\nPhase 3: Random Walk...")
random_walk = RandomWalk(pm)
user_input = "Create a portrait of a famous person"
phi = 3  # Number of secondary classes
composite_class_sequence = random_walk.walk(user_input, phi)
print(f"Generated composite class sequence: {composite_class_sequence}")
print(f"  Primary class: {composite_class_sequence[0]}")
print(f"  Secondary classes: {composite_class_sequence[1:]}")
print("\n✓ Phase 1, 2 & 3 completed successfully!")
