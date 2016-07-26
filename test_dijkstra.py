import numpy as np
import scipy.sparse.csgraph

import dijkstra

def test_dijkstra_against_scipy():
    W = np.random.random((100, 100))
    W += W.transpose()
    W[W<1.0] = np.inf
    W = scipy.sparse.csr_matrix(W)

    for seed in [0,1,2]:
        result0 = dijkstra.dijkstra(W, seed)
        result1 = scipy.sparse.csgraph.dijkstra(W, indices = [seed])[0]
        assert all(result0 == result1)
    

