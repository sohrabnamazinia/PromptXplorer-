# PromptXplorer Implementation Plan

## Overview

This repository implements the PromptXplorer framework, which constructs ordered sequences of composite prompts that maximize relevance, coverage, and diversity. The framework processes prompts as **sequences** (order matters), not sets.

## Stepwise Implementation Plan

### Phase 1: Data Loading & Preprocessing

**1.1. Data Models (`data_model/data_models.py`)**
- `PromptClass` class:
  - Attributes: `index` (int), `description` (str)
- `PrimaryPrompt` class:
  - Attributes: `text` (str), `class` (PromptClass, initially None)
- `SecondaryPrompt` class:
  - Attributes: `text` (str), `class` (PromptClass, initially None)
- `CompositePrompt` class:
  - Attributes: `primary` (PrimaryPrompt), `secondaries` (ordered list of SecondaryPrompt objects)
  - Method: `get_composite_class()` → returns ordered list [primary.class, secondary1.class, secondary2.class, ...]
- `Database` class:
  - Attributes: 
    - `composite_prompts` (list of CompositePrompt objects)
    - `composite_classes` (2D matrix, initially None) - same dimensions as loaded data, each cell contains class instead of text
    - `support_matrix` (2D matrix, initially None) - support values for composite classes
  - Methods:
    - `get_all_primary_prompts()` → returns list of all PrimaryPrompt objects
    - `get_all_composite_prompts()` → returns list of all CompositePrompt objects
    - `get_composite_classes()` → computes and sets `composite_classes` matrix by calling `get_composite_class()` on each CompositePrompt, returns the matrix

**1.2. Data Loader (`data_model/load_data.py`)**
- `DataLoader` class with parameters:
  - `separated` (bool): whether data is already separated
  - `n` (int): number of rows to consider from CSV
  - Method: `load_data(csv_path: str, batch_size: int = None) → Database`
    - **If `separated=True`**:
      - Reads CSV file (first `n` rows) where each row is one prompt
      - First column is primary prompt, remaining columns (comma-separated) are satellite/secondary prompts
      - Parses into primary + ordered secondary sequences (order matters)
      - Handles edge cases (empty secondaries, malformed data)
    - **If `separated=False`**:
      - Reads CSV file (first `n` rows) with single column containing full prompts
      - Uses LLM decomposition to split each prompt into primary + satellite prompts
      - Processes data in batches (max possible length) to LLM
      - Returns decomposed data in same format as separated=True
    - Returns Database object containing all composite prompts

**1.3. LLM Interface (`llm/llm_client.py`)**
- `LLMClient` class (will be extended with more functions later)
- Method: `decompose_prompts(prompts: List[str], batch_size: int) → List[Dict]`
  - Takes list of full prompts
  - Processes in batches (respecting max token length)
  - Calls LLM to decompose each prompt into primary + satellite prompts
  - Returns list of dictionaries with 'primary' and 'secondaries' keys
  - Handles batching logic and LLM API calls

---

### Phase 2: Clustering Module

**2.1. Clustering Class (`clustering/clusterer.py`)**
- `Clustering` class with parameters:
  - `database` (Database object): the database to cluster
  - `algorithm` (str): clustering algorithm to use (e.g., 'kmeans', 'dbscan', etc.)
- Clustering algorithm methods (each as a separate method):
  - `kmeans_clustering(texts, n_clusters)` → returns cluster labels and descriptions
  - `dbscan_clustering(texts, eps, min_samples)` → returns cluster labels and descriptions
  - `hdbscan_clustering(texts, min_cluster_size, min_samples)` → returns cluster labels and descriptions
  - Additional algorithms can be added as methods
- Main workflow method: `cluster(algorithm_params: dict) → Database`
  - Vectorize prompts (TF-IDF or embeddings)
  - Cluster primary prompts separately using selected algorithm
  - Cluster secondary prompts separately using selected algorithm
  - Call `store_cluster_results()` to assign classes to prompts
  - Call `compute_support()` and `build_support_matrix()` to populate support matrix
  - Returns updated Database object
- `store_cluster_results(primary_labels, primary_descriptions, secondary_labels, secondary_descriptions)`
  - Assigns cluster labels and descriptions to prompts
  - Creates `PromptClass` objects with index and description
  - Sets `class` attribute for all PrimaryPrompt and SecondaryPrompt objects
  - Saves cluster results to `cluster_results/` folder with meaningful unique filenames (e.g., timestamp + algorithm name)
  - Files can be used directly for algorithms/analysis
- `compute_support() → dict`
  - Computes support for each composite class
  - Returns dictionary mapping composite class sequences to support counts
- `build_support_matrix()`
  - Builds support matrix from computed support values
  - Sets `support_matrix` attribute of Database object

---

### Phase 3: Algorithms (`algorithms/` folder)

**3.1. Sequence Construction (`algorithms/sequence_construction.py`)**
- Simple interface/base class for sequence construction algorithms
- `IPF` class (inherits from interface):
  - Build pairwise probability matrices from frequent itemsets
  - Implement IPF algorithm (with reference to Dr. Das paper optimizations)
  - Generate top-k composite class sequences of specific length
  - Pruning bounds for efficiency
- `RandomWalk` class (inherits from interface):
  - Build directed graph from association rules (confidence as edge weights)
  - Implement random walk from primary class
  - Gelman-Rubin convergence checking
  - **First LLM Integration**: Check confidence threshold, use LLM if below threshold

**3.2. Prompt Selector (`algorithms/prompt_selector.py`)**
- `IndividualPromptSelector` class:
  - Embedding function (OpenAI embeddings or configurable)
  - Relevance scoring function (Euclidean distance or configurable)
  - Compute prefix embeddings for sequences
  - For each composite class sequence, select actual prompt instances
  - Use RAG to retrieve relevant complementary prompts from repository
  - **Second LLM Integration**: Select best instances based on relevance scores
  - Generate final composite prompt sequences (primary + ordered complementary instances)

**3.3. Representative Selection (`algorithms/representative_selection.py`)**
- `KSetCoverage` class:
  - Compute coverage (distinct complementary classes) for a set of sequences
  - Track which classes are covered by which sequences
  - Greedy algorithm to select k sequences maximizing coverage
  - Sampling-based variant: at each iteration, sample k candidate sequences and select best

**3.4. Sequence Ordering (`algorithms/sequence_ordering.py`)**
- `OrderSequence` class:
  - Compute order of priority (weights) for complementary classes given primary class
  - Compute pairwise diversity vectors between sequences
  - Compute weighted Hamming distance between sequences
  - Greedy ordering algorithm: start with most diverse sequence, iteratively select most diverse from previously selected
  - Maximize overall diversity vector

---

### Phase 4: Main Pipeline (`promptxplorer.py`)

**4.1. PromptXplorer Class**
- Main end-to-end framework class
- Input parameters:
  - `database` (Database object): the prompt repository
  - `primary_prompt` (str): user's new input primary prompt
  - `length` (int): desired composite prompt length
  - `k` (int): number of composite prompts to return
  - Additional configuration parameters (algorithm choices, thresholds, etc.)
- Method: `run() → List[CompositePrompt]`
  - Orchestrates all algorithm phases:
    1. Sequence construction (IPF or RandomWalk)
    2. Prompt selection (IndividualPromptSelector)
    3. Representative selection (KSetCoverage)
    4. Sequence ordering (OrderSequence)
  - Returns ordered sequence of k composite prompts

---

## Proposed File Structure

```
PromptXplorer-/
├── README.md
├── requirements.txt
├── config.py                    # Configuration parameters
├── data_model/
│   ├── data_models.py          # Phase 1.1: Data structures (PrimaryPrompt, SecondaryPrompt, CompositePrompt, Database)
│   └── load_data.py            # Phase 1.2-1.4: Data loading (DataLoader class + load_data function)
├── clustering/
│   └── clusterer.py            # Phase 2: Clustering (Clustering class with multiple algorithms, support computation)
├── cluster_results/            # Saved cluster results (created by store_cluster_results)
├── algorithms/
│   ├── sequence_construction.py  # Phase 3.1: Interface + IPF and RandomWalk classes
│   ├── prompt_selector.py        # Phase 3.2: IndividualPromptSelector class (RAG + LLM)
│   ├── representative_selection.py # Phase 3.3: KSetCoverage class
│   └── sequence_ordering.py      # Phase 3.4: OrderSequence class
├── llm/
│   ├── llm_client.py           # LLM integration (OpenAI/other) - Phase 1.3: decompose_prompts() + future functions
│   └── prompts.py              # LLM prompt templates
├── promptxplorer.py            # Phase 4: PromptXplorer main class (end-to-end framework)
├── config.py                   # Configuration parameters
└── utils/
    ├── logger.py
    └── metrics.py
```

## Key Design Principles

1. **Modularity**: Each phase is a separate module
2. **Configurability**: All parameters in `config.py`
3. **Sequence-aware**: All data structures preserve order
4. **LLM Integration**: Two clear integration points
5. **Extensibility**: Easy to swap algorithms (e.g., different clustering, embeddings)

## Algorithmic Pipeline Summary

1. **Cluster satellite prompts** (complementary prompts)
2. **Create sequences of clusters (classes)**:
   - IPF: Ordered pairwise probabilities → estimate top chains with highest joint probability
   - Random Walk: Assign class to primary → initiate random walk → get sequences
     - Gelman-Rubin for convergence
     - **First LLM use**: If confidence below threshold
3. **Convert cluster sequences → individual prompt sequences** (each sequence independently)
   - RAG-based approach
   - **Second LLM use**
4. **k-set coverage** (e.g., 100 sequences → 10):
   - Maximize coverage of distinct classes
   - **Improved**: Sampling combined k-set coverage
5. **Order sequences**:
   - Weighted Hamming distance between prompts
   - Greedy: most diverse from all → most diverse from first → most diverse from previous → ...
