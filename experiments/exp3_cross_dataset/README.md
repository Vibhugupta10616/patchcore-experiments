# Experiment 3: FAISS-Based Memory Compression & ANN Search

## Overview

This experiment evaluates **FAISS (Facebook AI Similarity Search)** with **IVF-PQ (Inverted File with Product Quantization)** as a fast, memory-efficient alternative to exact k-NN search in PatchCore.

**Goal:** Enable faster inference and reduced memory usage while preserving anomaly detection accuracy.

---

## Methodology

### Baseline
- **Exact k-NN:** Uses sklearn's NearestNeighbors with kd-tree algorithm
- Full precision memory bank stored in RAM
- Serves as the accuracy and performance reference

### FAISS IVF-PQ Indexing
Replaces exact k-NN with FAISS's approximate nearest neighbor search:

1. **IVF (Inverted File):** Clusters memory bank into `n_clusters` centroids
2. **PQ (Product Quantization):** Compresses feature vectors using subquantizers
   - Splits each feature into `m` subvectors
   - Quantizes each subvector to `nbits` bits
3. **nprobe:** Number of clusters searched during query (speed/accuracy trade-off)

### Experiments
Tests various configurations:
- **PQ bits:** `[4, 8, 16]` - Compression level
- **nprobe:** `[1, 4, 8, 16]` - Search thoroughness
- **Fixed parameters:** `n_clusters=100`, `m=8` (subquantizers)

---

## Metrics

1. **Image AUROC:** Anomaly detection accuracy (ROC-AUC score)
2. **Memory (MB):** RAM usage for feature storage
3. **Search Time (ms):** Average query time per test sample
4. **Speedup Factor:** `baseline_time / faiss_time`
5. **Memory Reduction:** `baseline_memory / faiss_memory`
6. **Accuracy Loss (%):** `(baseline_auroc - faiss_auroc) × 100`

---

---

## Expected Results

### Trade-offs

| Configuration | AUROC | Speedup | Memory Reduction | Accuracy Loss |
|--------------|-------|---------|------------------|---------------|
| Exact k-NN   | 0.950 | 1.0×    | 1.0×             | 0.0%          |
| 16-bit, nprobe=16 | 0.948 | 3.5× | 2.0× | 0.2% |
| 8-bit, nprobe=8 | 0.942 | 8.2× | 4.0× | 0.8% |
| 4-bit, nprobe=4 | 0.930 | 18.5× | 8.0× | 2.1% |

*Values are illustrative and depend on feature dimensionality and memory bank size.*

### Key Insights
1. **4-bit quantization:** Maximum speedup (~15-20×) with minimal accuracy loss (<2%)
2. **Higher nprobe:** Improves accuracy but reduces speedup
3. **8-bit quantization:** Best balance between speed, memory, and accuracy
4. **IVF-PQ scales well:** Performance gains increase with larger memory banks (>10K samples)

---

## Visualizations

The experiment generates `exp3_faiss_analysis.png` containing:

1. **Accuracy vs Speedup:** Main trade-off scatter plot
2. **Memory vs Accuracy:** Memory efficiency analysis
3. **AUROC vs Nprobe:** Effect of search breadth on accuracy
4. **Search Speed Comparison:** Bar chart across configurations
5. **Memory Usage Comparison:** RAM consumption by method
6. **Accuracy Loss Heatmap:** Loss across PQ bits × nprobe grid
7. **Speedup Heatmap:** Performance gains across configurations

---

## Implementation Details

### Key Functions (`exp3_utils.py`)

1. **`build_exact_knn_index(features, k=5)`**
   - Builds sklearn NearestNeighbors baseline index
   - Returns dictionary with index metadata

2. **`build_faiss_ivfpq_index(features, m, nbits, nprobe, n_clusters)`**
   - Trains FAISS IVF-PQ index on memory bank features
   - Configures quantization and search parameters
   - Returns dictionary with FAISS index

3. **`evaluate_index_performance(index, test_features, train_features, index_type)`**
   - Computes AUROC and memory usage
   - Generates synthetic anomaly labels for evaluation
   - Returns `(auroc, memory_mb)` tuple

4. **`benchmark_search_speed(index, test_features, n_runs=5)`**
   - Measures average query time across multiple runs
   - Returns per-query search time in seconds

---

## Configuration Parameters

### FAISS IVF-PQ Settings

- **`m` (subquantizers):** Number of feature splits (must divide feature dimension)
  - Higher `m` → Better compression but slower training
  - Typical values: 8, 16, 32

- **`nbits` (PQ bits):** Bits per subquantizer
  - 4 bits → High compression, low accuracy
  - 8 bits → Balanced (recommended)
  - 16 bits → Low compression, high accuracy

- **`n_clusters` (IVF):** Number of cluster centroids
  - Rule of thumb: `sqrt(N)` where N = memory bank size
  - More clusters → Better approximation, higher memory

- **`nprobe`:** Clusters searched during query
  - 1 → Fastest, lowest accuracy
  - 16+ → Slower, near-exact results
  - Default: 4-8 for good balance

---

