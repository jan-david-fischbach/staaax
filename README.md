# Staaax v0.2.4

![Logo](./docs/assets/logo2.svg)

`staaax` is a convenience wrapper for `sax`, that allows evaluating scattering matrices (S-matrices) for stacks of layered media (also called stratified media).

```{code-cell} ipython3
import staaax
import numpy as np

ns = [1, 2, 3+0.001j, 1]
ds = [1, 0.5]

wl = 1.5
theta_0=np.deg2rad(20)
k0 = 2*np.pi/wl
kx = k0*np.sin(theta_0)

stack, info = staaax.stratified(
  ds, ns, k0, kx, pol="s"
)

S = stack()
S
```