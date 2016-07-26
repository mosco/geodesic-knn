import numpy as np
import scipy.sparse

import heapdict
import dijkstra

heap_pop_count = 0

def check_sparse_edge_weights_matrix(W):
    assert type(W) == scipy.sparse.csr.csr_matrix
    (n, n_other) = W.shape
    assert n == n_other
    assert (W.data >= 0).all()
    #assert (W.transpose() != W).nnz == 0
    return n

def geodesic_knn(W, labeled_mask, k):
    '''
    Input:
        W: n by n scipy.sparse.csr_matrix
            Edge *symmetric* weight matrix. We use the scipy.sparse.csgraph convention that non-edges are denoted by non-entries.

        labeled_mask: boolean array of length n indicating which vertices are labeled

    Output:
        knn: knn[i] is a list of up to k pairs of (dist, seed)
    '''
    global heap_pop_count

    n = check_sparse_edge_weights_matrix(W)
    assert labeled_mask.dtype == np.bool
    assert labeled_mask.shape == (n,)

    labeled_vertex_indices = labeled_mask.nonzero()[0]

    visited = set()
    knn = [[] for i in xrange(n)]
    heap = heapdict.heapdict()
    for s in labeled_vertex_indices:
        heap[(s, s)] = 0.0

    W_indptr = W.indptr
    W_indices = W.indices
    W_data = W.data
    while len(heap) > 0:
        ((seed, i), dist_seed_i) = heap.popitem()
        heap_pop_count += 1
        visited.add((seed, i))

        if len(knn[i]) < k:
            knn[i].append((dist_seed_i, seed))

            for pos in xrange(W_indptr[i], W_indptr[i+1]):
                j = W_indices[pos]
                if (seed, j) not in visited:
                    alt_dist = dist_seed_i + W_data[pos]
                    if (seed, j) not in heap or alt_dist < heap[(seed, j)]:
                        heap[(seed, j)] = alt_dist

    return knn

def geodesic_knn_dijkstra(W, labeled_mask, k):
    n = check_sparse_edge_weights_matrix(W)
    assert labeled_mask.dtype == np.bool
    assert labeled_mask.shape == (n,)
    labeled_vertex_indices = labeled_mask.nonzero()[0]
    assert k <= len(labeled_vertex_indices)

    dijkstra.heap_pop_count = 0
    distances_from_labeled = np.vstack(dijkstra.dijkstra(W, i) for i in labeled_vertex_indices)
    global heap_pop_count
    heap_pop_count += dijkstra.heap_pop_count

    assert distances_from_labeled.shape == (len(labeled_vertex_indices), n)

    knn = []
    for i in xrange(n):
        k_closest_labeled_to_i = np.argpartition(distances_from_labeled[:,i], k-1)[:k]
        knn.append(zip(distances_from_labeled[k_closest_labeled_to_i, i], labeled_vertex_indices[k_closest_labeled_to_i]))

    return knn
