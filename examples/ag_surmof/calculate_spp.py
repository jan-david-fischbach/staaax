#%%
import sys
sys.path.append(".")
sys.path.append("..")
from examples import material
import jax.numpy as jnp
import numpy as onp
from ag_surmof import f_to_k, k_to_f, k_to_wfreq, pcolormesh_k
import matplotlib.pyplot as plt
import angled_stratified

to_Hz = 300e12/(2*jnp.pi)
f_domain=jnp.array([7.5-0.6j, 10.5+0.01j]) *to_Hz
#f_domain=jnp.array([4-10j, 10.5+0.01j]) *to_Hz
k_domain=f_to_k(f_domain)

k_r = jnp.linspace(k_domain[0].real, k_domain[1].real, 300)
k_i = jnp.linspace(k_domain[0].imag, k_domain[1].imag, 301)
K_r, K_i = jnp.meshgrid(k_r, k_i)

F_r, F_i = k_to_f(K_r), k_to_f(K_i)

k0_mesh = K_r + 1j*K_i
wfreq = k_to_wfreq(k0_mesh)
eps = material.eps_ag(wfreq)

kx = k0_mesh * jnp.sqrt(eps/(eps+1))

# plt.pcolormesh(F_r, F_i, kx.real/k_domain[1].real)
# plt.colorbar()
# plt.title("kx.real")

# plt.figure()
# pcolormesh_k(K_r, K_i, kx.imag/k_domain[1].real)
# plt.colorbar()
# plt.contour(F_r/1e12, F_i/1e12, kx.imag/k_domain[1].real, levels=[0])
# plt.title("kx.imag")

ns = [1, jnp.sqrt(eps), 1]
ds = [30e-9]

ns = [1, jnp.sqrt(eps)]
ds = []

ns = [1, jnp.sqrt(eps), 3]
ds = [30e-9]
pol="p"
    
kx = kx[*jnp.array(kx.shape)//2].real
stack, info = angled_stratified.stack_smat_kx(ds, ns, k0_mesh, kx, pol=pol)
trans      = stack()[('in', 'out')]
refl       = stack()[('in', 'in')]
# %%
pcolormesh_k(K_r, K_i, jnp.abs(refl), norm="log")
# %%
kx
# %%
