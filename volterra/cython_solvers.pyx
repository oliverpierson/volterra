import numpy as np

def cblock(K, g, h, N, mu=0.5, F0=None):
    """
    Integral equation solver for Volterra equations of the first kind, i.e.
    equations of the form:
        g(t) = \int_{0}^{t} (t-s)^{-\mu}K(t,s)f(s) ds

    This routine uses the block method.

    Parameters
    ----------
    K : known function K(t,s)
    g : known function g(t)
    h : float, time step
    mu : float
    F0 : float, value of f(0) (this will be optional eventually)

    Returns
    -------
    F : array
        Value of function f(t) at points t=0, h, 2*h, ..., N*H

    References
    ----------
    ..  [1] P. Lenz, "Analytical and Numerical Methods for Volterra Equations"
        Ch. 10, SIAM, 1985
    """
    assert(F0 != None)
    F = np.zeros(N, dtype=float)
    F[0] = F0

    def delta_func(int d, mu=0.5):
        # TODO line below is for case mu=0.5 only
        return (2./15.)*((9 - 4*d)*(d - 1)**1.5 + np.sqrt(d)*(15 + d*(4*d - 15)))

    def eps_func(int d, mu=0.5):
        # TODO line below is for case mu=0.5 only
        return -(2./15.)*(4*d**1.5*(2*d - 5) + np.sqrt(d-1)*(7 + 16*d - 8*d**2))

    def zeta_func(int d, mu=0.5):
        # TODO line below is for case mu=0.5 only
        return (2./15.)*(d**1.5*(4*d - 5) + np.sqrt(d-1)*(1 + 3*d - 4*d**2))

    delta = [ delta_func(n-1) for n in np.arange(1, N+1) ]
    eps = [ eps_func(n-1) for n in np.arange(1, N+1) ]
    zeta = [ zeta_func(n-1) for n in np.arange(1, N+1) ]

    # TODO values below are for case mu=0.5 only
    alpha = -0.133333
    beta = 0.933333
    gamma = 1.2
    
    _A = np.zeros((2,2), dtype=float)
    _S = np.zeros(2, dtype=float)

    def A(int n):
        _A[1,0] = (beta + eps[2])*K(n*h, (n-1)*h)
        if(n > 2):
            _A[1,0] += zeta[3]*K(n*h, (n-1)*h)
        _A[1,1] = (gamma + zeta[2])*K(n*h, n*h)
        _A[0,0] = gamma*K((n-1)*h, (n-1)*h) + 0.75*beta*K((n-1)*h, (n-1.5)*h)
        if(n > 2):
            _A[0,0] += zeta[2]*K((n-1)*h, (n-1)*h)
        _A[0,1] = -0.125*beta*K((n-1)*h, (n-1.5)*h)
        return _A

    def S(int n):

        def I(int m, int i):
            ''' approx to integral \int_{t_i}^{t_{i+1}} f(s) K(t,s) (t_m - s) ds '''
            val = delta[m-i]*F[i]*K(m*h, i*h) + eps[m-i]*F[i+1]*K(m*h, (i+1)*h) \
                  + zeta[m-i]*F[i+2]*K(m*h, (i+2)*h)
            return h**(1-mu)*val

        _S[0] = g((n-1)*h) - np.sum([I(n-1, i) for i in np.arange(n-4+1)]) \
                - h**(1-mu)*((alpha*K((n-1)*h, (n-2)*h) + 0.375*beta*K((n-1)*h, (n-1.5)*h))*F[n-2])
        if(n > 2):
            _S[0] -= h**(1-mu)*(delta[2]*F[n-3]*K((n-1)*h, (n-3)*h) \
                     + eps[2]*F[n-2]*K((n-1)*h, (n-2)*h))

        _S[1] = g(n*h) - np.sum([I(n, i) for i in np.arange(n-4+1)]) \
                - h**(1-mu)*((alpha + delta[2])*F[n-2]*K(n*h, (n-2)*h))
        if(n > 2):
            _S[1] -= h**(1-mu)*(delta[3]*F[n-3]*K(n*h, (n-3)*h) + eps[3]*F[n-2]*K(n*h, (n-2)*h))

        return _S

    for m in np.arange(1,N-1):
        F[m:(m+2)] = np.linalg.solve(h**(1-mu)*A(m+1), S(m+1))
    return F
