import numpy as np
import scipy.sparse.csr
import heapdict

heap_pop_count = 0

def dijkstra(W, seed_index):
    '''
    Input:
        W: n by n scipy.sparse.csr_matrix
            Edge *symmetric* weight matrix. We use the scipy.sparse.csgraph convention that non-edges are denoted by non-entries.

        seed_index: the index of the starting vertex

    Output:
        distances: numpy array of size n holding the distance from seed_index to every vertex in the graph
    '''
    assert type(W) == scipy.sparse.csr.csr_matrix
    (n, n_other) = W.shape
    assert n == n_other
    assert (W.data >= 0).all()
    assert 0 <= seed_index < n

    distances = np.empty(n)
    distances.fill(np.inf)
    visited = np.zeros(n, np.bool)

    heap = heapdict.heapdict()
    heap[seed_index] = 0.0
    W_indptr = W.indptr
    W_indices = W.indices
    W_data = W.data
    global heap_pop_count
    while len(heap) > 0:
        (i, dist_i) = heap.popitem()
        heap_pop_count += 1
        distances[i] = dist_i

        for pos in xrange(W_indptr[i], W_indptr[i+1]):
            j = W_indices[pos]
            if distances[j] == np.inf:
                if j not in heap:
                    heap[j] = dist_i + W_data[pos]
                else:
                    alt_dist = dist_i + W_data[pos]
                    if alt_dist < heap[j]:
                        heap[j] = alt_dist

    return distances

