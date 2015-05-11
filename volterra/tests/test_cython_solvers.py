import numpy as np
from volterra import cython_solvers

def test_block1():
    N = 100
    dt = 0.1
    def k(t,s):
        return 1. + t + s

    def g(t):
        return 2*t*np.arctanh(np.sqrt(t/(t+1.))) / np.sqrt(t+1) + 2.0*np.sqrt(t)

    t = np.arange(0., dt*N, dt)
    F = cython_solvers.cblock(k, g, dt, N, F0=1.0)
    assert np.allclose(F, 1.0/(1.0 + t), atol=1e-2, rtol=0.0)

def time_cblock(N):
    def k(t,s):
        return 1./np.sqrt(2*np.pi)

    def g(x):
        return np.exp(-0.5/x)/np.sqrt(2*np.pi*x)

    t = np.arange(0., 0.01*N, 0.01)

    F = cython_solvers.cblock(k, g, 0.01, N, F0=0.0) 

    return t, F

if __name__ == '__main__':
    import time
    import volterra
    def k(t,s):
        return 1./np.sqrt(2*np.pi)

    def g(x):
        return np.exp(-0.5/x)/np.sqrt(2*np.pi*x)

    N = 100
    t = np.arange(0., 0.01*N, 0.01)

    start_time = time.time()
    F = cython_solvers.cblock(k, g, 0.01, N, F0=0.0) 
    end_time = time.time()
    t[0] = 1.0 # since t[0] = 0 it cause a division error, so ignore that data point
    if np.allclose(F[1:], (np.exp(-0.5/t)/np.sqrt(2*np.pi*t**3))[1:], atol=3e-3, rtol=0.0):
        print "Passed."
    else:
        print "Failed."
    print "cythonized code: ", end_time-start_time, "seconds"
    start_time = time.time()
    F = volterra.block(k, g, 0.01, N, F0=0.0) 
    end_time = time.time()
    print "pure block_test2: ", end_time-start_time, "seconds"
    
