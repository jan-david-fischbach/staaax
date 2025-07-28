---
jupytext:
  cell_metadata_filter: -all
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.2
kernelspec:
  display_name: staaax (3.13.1)
  language: python
  name: python3
---

## Usage

Start by importing `staaax` (and other packages you want to use):

```{code-cell} ipython3
import numpy as np
import sax
import staaax
import matplotlib.pyplot as plt
```

Define a Stack of Layers

```{code-cell} ipython3
ns = [1, 2, 3+0.001j, 1]
ds = [1, 0.5]
```

Further define the (angular) wavenumber and parallel part of the wavevector (determining the angle of incidence)

```{code-cell} ipython3
wl = 1.5
theta_0=np.deg2rad(20)
k0 = 2*np.pi/wl
kx = k0*np.sin(theta_0)
```

You can then create a `sax` model for the stack by

```{code-cell} ipython3
stack, info = staaax.stratified(
  ds, ns, k0, kx, pol="s"
)
```

It can straightforwardly be evaluated by

```{code-cell} ipython3
S = stack()
S
```

Because it is a valid sax model, we can use the typical sax functionality. Including `vmap`s and differentiation. The typical wavelength batching also works:

```{code-cell} ipython3
k0s = np.linspace(2, 6, 101)
```

Because of JIT compilation repeated evaluation of stack for different parameters is fast.

```{code-cell} ipython3
%timeit stack(k0=k0s)
```

```{code-cell} ipython3
S = stack(k0=k0s)
plt.plot(k0s, np.abs(S['in', 'in'])**2)
plt.xlabel(r'$k_0$')
plt.ylabel(r'$R$')
plt.title('Reflectance for different $k_0$')
```

In principle arbitrary parameters can be used as batch settings. In this case let's sweep the refractive index of the incident half space:

```{code-cell} ipython3
settings = sax.get_settings(stack)
ni = np.linspace(1, 2, 21)
batch_settings = sax.update_settings(settings, 'if_0', ni=ni)
```

```{code-cell} ipython3
S = stack(**batch_settings)
plt.plot(ni, np.abs(S['in', 'in'])**2)
plt.xlabel(r'$n_i$')
plt.ylabel(r'$R$')
plt.title('Reflectance for $n_i$')
```

```{code-cell} ipython3

```
