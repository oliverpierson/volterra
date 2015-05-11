# Volterra Equation Solver
[volterra.py](./volterra.py) implements two methods for solving Volterra integral equations of the first kind,

![volterra equation](./eqn.png)

These are integral equations for the function `f` where `g` and `K` are known functions.  Note the  `t` in the integration bounds.  The factor `(t-s)^{-\mu}` accounts for any singularities in the kernel `K`.  In other words, if you have an integral equation such that `K(t,t)` is unbounded (singular) and `K(t,s)` diverges like `(t-s)^{-\mu}` as `t->s`, rewrite it in the form above (so that `K` is well-behaved part of the origial kernel).

# Example
The integral equation

![example eqn](./example.png)

has the exact solution

![example soln](./example_soln.png)

```python
>>> def g(t):
...   return 2*t*np.arctanh(np.sqrt(t/(t+1))) / np.sqrt(t+1) + 2*np.sqrt(t)
... 
>>> def K(t,s):
...     return 1 + t + s
...
>>> F = volterra.block_step(K, g, 0.1, 100, mu=0.5, F0=1.0)
>>> np.allclose(F, 1/(1+t), atol=1e-2, rtol=0.0)
True
```
