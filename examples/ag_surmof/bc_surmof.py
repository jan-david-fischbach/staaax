from staaax.slab_bc_exploration import plot_batched, batch_with_k0

import matplotlib.pyplot as plt
import jax.numpy as jnp
import material
from staaax.propagation import angled_sqrt
from scipy.constants import c as c0

def k_to_wfreq(k0):
    return k0 * c0 / 300e12

def k_to_f(k0):
    return 1/(2*jnp.pi) * k0 * c0

def f_to_k(f):
    return 2*jnp.pi * f / c0


if __name__ == "__main__":
    to_Hz = 300e12/(2*jnp.pi)
    f_domain_roi=jnp.array([7.5-0.2j, 10.5+0.06j]) *to_Hz
    f_domain=jnp.array([1-0.6j, 10.5-0.01j]) *to_Hz

    k_domain_roi=f_to_k(f_domain_roi)
    k_domain=f_to_k(f_domain)

    res = 300
    k_r = jnp.concat([
        jnp.linspace(min(k_domain.real), min(k_domain_roi.real)*0.995, res//4),
        jnp.linspace(min(k_domain_roi.real), max(k_domain_roi.real), res)
        ])
    k_i = jnp.linspace(min(k_domain.imag), max(k_domain.imag), res)
    # -jnp.logspace(
    #     jnp.log10(-min(k_domain.imag)), 
    #     jnp.log10(-max(k_domain.imag)), 
    #     res
    # )

    k_i = jnp.sort(jnp.concat([k_i, -k_i]))
    #jnp.linspace(min(k_domain.imag), max(k_domain.imag), res)

    K_r, K_i = jnp.meshgrid(k_r, k_i)
    k0_mesh = K_r+1j*K_i

    pol = "p"
    kx = jnp.linspace(min(k_domain_roi.real)*1.1, min(k_domain_roi.real)*1.4, 17)
    kx = jnp.insert(kx, 0, 0)

    wfreq = k_to_wfreq(k0_mesh)
    
    #silver = angled_sqrt(material.eps_ag(wfreq), bc_angle=-jnp.pi)
    silver = jnp.sqrt(material.eps_ag(wfreq))                        [None, :]
    surmof = angled_sqrt(material.eps_cav(wfreq, 1), bc_angle=jnp.pi/2) [None, :]

    one = jnp.ones(1)[:, None, None]

    ns = [one, silver, surmof, silver, one]
    ds = [0.02e-6, 0.05e-6, 0.03e-6]

    kx, *ds = batch_with_k0(kx, *ds)
    k0_mesh = k0_mesh[None, ...]

    plot_batched(kx, ds, ns, pol, k0_mesh, 15, bc_width=0.08, plot_tilde=False, figsize=(20, 6))
    plt.xlim(min(k_domain_roi.real), max(k_domain_roi.real))
    plt.ylim(min(k_domain_roi.imag), max(k_domain_roi.imag))
    plt.savefig("tmp/surmof.png")