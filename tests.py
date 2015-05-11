import numpy as np
from volterra import block, midpt

# def midpt_test1():
#     def k(t,s):
#         return 1. + s
#     def g(t):
#         return (16./15.)*t**(5./2.) + (4./3.)*t**(3./2.)
#     Q,t = midpt(k, g, 1., 0.1)
#     return Q,t

def midpt_test2():
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

    f,t = midpt(k, g, 10.0, 0.5)
    return f,t

def block_test1(N):
    """
    >>> t, F = block_test1(10)
    >>> np.allclose(F, 1.0/(1.0 + t), atol=1e-2, rtol=0.0)
    True
    """
    def k(t,s):
        return 1. + t + s

    def g(t):
        return 2*t*np.arctanh(np.sqrt(t/(t+1.))) / np.sqrt(t+1) + 2.0*np.sqrt(t)

    t = np.arange(0., 0.5*N, 0.5)

    return t, block(k, g, 0.5, N, F0=1.0)

def block_test2(N):
    """
    >>> t, F = block_test2(100)
    >>> np.allclose(F[1:], (np.exp(-0.5/t)/np.sqrt(2*np.pi*t**3))[1:], atol=3e-3, rtol=0.0)
    True
    """
    def k(t,s):
        return 1./np.sqrt(2*np.pi)

    def g(x):
        return np.exp(-0.5/x)/np.sqrt(2*np.pi*x)

    t = np.arange(0., 0.01*N, 0.01)

    return t, block(k, g, 0.01, N, F0=0.0)

if __name__ == '__main__':
    import doctest
    doctest.testmod()
