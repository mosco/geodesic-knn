import numpy as np
import scipy.sparse

import geodesic_knn

def build_sparse_undirected_nonnegative_edge_matrix(n):
    W = np.random.random((n,n))
    W = W + W.transpose()
    W[W < 1.5] = np.inf
    return scipy.sparse.csr_matrix(W)

def test_geodesic_knn():
    N = 100
    p = 0.2 
    k = 5
    
    W = build_sparse_undirected_nonnegative_edge_matrix(N)
    labeled_mask = np.random.random(N) < p
    print 'labeled vertices:'
    print labeled_mask.nonzero()[0]

    result0 = geodesic_knn.geodesic_knn(W, labeled_mask, k)
    result1 = geodesic_knn.geodesic_knn_dijkstra(W, labeled_mask, k)

    for i in xrange(len(result0)):
        print 'result0[%d]:' % i
        print result0[i]

        print 'result1[%d]:' % i
        print sorted(result1[i])

        assert result0[i] == sorted(result1[i])
