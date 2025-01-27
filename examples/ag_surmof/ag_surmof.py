import sys
sys.path.append(".")
import matplotlib.pyplot as plt
import jax.numpy as jnp
import staaax.angled_stratified as angled_stratified
from diffaaable import aaa
import material
from staaax.propagation import angled_sqrt
from scipy.constants import c as c0
import sax

def k_to_wfreq(k0):
    return k0 * c0 / 300e12

def k_to_f(k0):
    return 1/(2*jnp.pi) * k0 * c0

def f_to_k(f):
    return 2*jnp.pi * f / c0

def pcolormesh_k(K_r, K_i, val, **kwargs):
    plt.pcolormesh(k_to_f(K_r)/1e12, k_to_f(K_i)/1e12, val, **kwargs)
    plt.xlabel("$\Re\{f\}$ [THz]")
    plt.ylabel("$\Im\{f\}$ [THz]")


if __name__ == "__main__":
    to_Hz = 300e12/(2*jnp.pi)
    #f_domain=jnp.array([7.5-30j, 10.5+0.2j])*to_Hz
    #f_domain=jnp.array([7.5-0.3j, 10.5+0.03j])*to_Hz
    #f_domain=jnp.array([7.5-0.1j, 10.5+0.01j]) *to_Hz
    f_domain=jnp.array([5-40j, 18+0.01j]) *to_Hz
    k_domain=f_to_k(f_domain)

    pol = "p"
    pol_idx = 1 if pol=="p" else 0

    k_r = jnp.linspace(k_domain[0].real, k_domain[1].real, 300)
    k_i = jnp.linspace(k_domain[0].imag, k_domain[1].imag, 301)
    K_r, K_i = jnp.meshgrid(k_r, k_i)

    # t_mirror = jnp.logspace(jnp.log10(0.02e-6), jnp.log10(0.2e-6), 9)
    # kx = jnp.ones(9)*9.24e6 #

    n_shots = 15
    t_mirror = jnp.ones(n_shots)*0.02e-6
    kx = jnp.linspace(min(k_domain.real), max(k_domain.real)*1.5, n_shots)

    k0_mesh = K_r+ 1j*K_i

    k0_mesh = k0_mesh[None, :]
    kx = kx[:, None, None]
    t_mirror = t_mirror[:, None, None]

    wfreq = k_to_wfreq(k0_mesh)
    
    #silver = angled_sqrt(material.eps_ag(wfreq), bc_angle=-jnp.pi)
    silver = jnp.sqrt(material.eps_ag(wfreq))
    surmof = angled_sqrt(material.eps_cav(wfreq), bc_angle=jnp.pi/2)

    ns = [1, silver, surmof, silver, 1.45]
    ds = [t_mirror, 0.2e-6, 0.03e-6]

    # pcolormesh_k(K_r, K_i, jnp.squeeze(jnp.abs(ns[0])))
    # plt.colorbar()
    # plt.show()

    bc_width=3e-3
    stack, info = angled_stratified.stack_smat_kx(ds, ns, k0_mesh, kx, pol=pol)
    settings = sax.get_settings(stack)
    settings = sax.update_settings(
        settings, "if_0", bc_width_i=bc_width
    )
    settings = sax.update_settings(
        settings, "if_3", bc_width_j=bc_width
    )

    smat_mesh = stack(
        **settings
    )

    num = len(t_mirror)
    fig, axs = plt.subplots(3, int((num-0.1)//3+1))
    for idx, ax in enumerate(axs.flatten()):
        if idx == num:
            break
        plt.sca(ax)

        downsample = 20
        # z_j, f_j, w_j, z_n = aaa(
        #     k0_mesh[0, ::downsample, ::downsample], 
        #     smat_mesh[('in', 'out')][idx][::downsample, ::downsample], 
        #     tol=1e-7
        # )
        trans = jnp.abs(smat_mesh[('in', 'out')][idx])
        refl = jnp.abs(smat_mesh[('in', 'in')][idx])
        pcolormesh_k(K_r, K_i, trans, norm="log")#, vmin=1e-3, vmax=jnp.nanquantile(refl, 0.99))
        plt.colorbar()
        plt.xlim(plt.xlim())
        plt.ylim(plt.ylim())
        # z_n = k_to_f(z_n)/1e12
        # plt.scatter(z_n.real, z_n.imag)
    plt.show()