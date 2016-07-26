
This is a python implementation of an algorithm that efficiently finds the K nearest neighbors (from a specific subset) of every vertex in a graph.

# Interface

The main function here is geodesic_knn(W, labeled_mask, k)

**Input**:
* **W:** n x n matrix of edge weights. Must be of type scipy.sparse.csr_matrix.
* **labeled_mask:** boolean array of length n indicating which vertices are labeled
* **k:** how many nearest neighbors to return for each vertex.

**Output:**
* **knn:** this is an array of size n such that knn[i] is a list of up to k pairs of (dist, seed)

