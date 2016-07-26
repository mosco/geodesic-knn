import numpy as np
import scipy.sparse
import mnist
import edge_weights
from save_load_data import load_data

import pyximport
pyximport.install(setup_args={"include_dirs":np.get_include()})

import geodesic_knn as gk
import geodesic_knn_cython as gkc

class Timer(object):
    def __init__(self, text = None):
        if text != None:
            print ('%s:' % text), 
        sys.stdout.flush()
        self.start_clock = time.clock()
        self.start_time = time.time()
        self.end_time = None
        self.end_clock = None

    def elapsed(self):
        return time.time() - self.start_time

    def stop(self):
        self.end_time = time.time()
        self.end_clock = time.clock()

    def print_elapsed(self):
        print 'Wall time: %.3f seconds.  CPU time: %.3f seconds.' % (self.end_time - self.start_time, self.end_clock - self.start_clock)

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        self.stop()
        self.print_elapsed()

def build_sparse_undirected_nonnegative_edge_matrix(n):
    W = np.random.random((n,n))
    W = W + W.transpose()
    W[W < 1.5] = np.inf
    return scipy.sparse.csr_matrix(W)

def test_geodesic_knn_cython():
    N = 10000
    p = 0.1 
    
    #W = build_sparse_undirected_nonnegative_edge_matrix(N)
    labeled_mask = np.random.random(N) < p
    #print 'labeled vertices:'
    #print labeled_mask.nonzero()[0]

    with Timer('Loading edges'):
        data = load_data('edges_matrix_16x16_knn8_distance_%d' % N)

    with Timer('geodesic_knn'):
        result0 = gk.geodesic_knn(data.E, labeled_mask, 1)
    with Timer('geodesic_knn_cython'):
        result1 = gkc.geodesic_nn(data.E, labeled_mask)

    for i in xrange(len(result0)):
        #print 'result0[%d]:' % i
        #print result0[i]

        #print 'result1[%d]:' % i
        #print sorted(result1[i])

        assert result0[i][0][1] == result1[i]
