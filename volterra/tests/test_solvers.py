import numpy as np
from volterra import block, midpt

# def midpt_test1():
#     def k(t,s):
#         return 1. + s
#     def g(t):
#         return (16./15.)*t**(5./2.) + (4./3.)*t**(3./2.)
#     Q,t = midpt(k, g, 1., 0.1)
#     return Q,t

def test_midpt2():
    """
    Step size of 0.5 result in an error of ~0.1
    >>> F,t = midpt_test2()
    >>> np.allclose(F, 1.0/(1.0 + t), atol=1e-1, rtol=0.0)
    True
    """
    def k(t,s):
        return 1. + t + s

    def g(t):
        return 2*t * np.arctanh(np.sqrt(t/(t+1.))) / np.sqrt(t+1) + 2.0*np.sqrt(t)

    dt = 0.5
    t = 10.0
    f,t = midpt(k, g, t, dt)

    assert np.allclose(f, 1.0/(1.0 + t), atol=1e-1, rtol=0.0)

def test_block1():

    def k(t,s):
        return 1. + t + s

    def g(t):
        return 2*t*np.arctanh(np.sqrt(t/(t+1.))) / np.sqrt(t+1) + 2.0*np.sqrt(t)
    N = 100
    dt = 0.5
    t = np.arange(0., dt*N, dt)
    F = block(k, g, 0.5, N)

    assert np.allclose(F, 1.0/(1.0 + t), atol=1e-2, rtol=0.0)

def test_block2():

    def k(t,s):
        return 1./np.sqrt(2*np.pi)

    def g(x):
        return np.exp(-0.5/x)/np.sqrt(2*np.pi*x)

    N = 100
    dt = 0.01
    t = np.arange(dt, dt*N, dt)
    F = block(k, g, dt, N)

    assert np.allclose(F[1:], (np.exp(-0.5/t)/np.sqrt(2*np.pi*t**3)), atol=3e-3, rtol=0.0)
