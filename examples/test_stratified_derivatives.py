import jax.numpy as jnp
import staaax.angled_stratified
import staaax.angled_stratified_treams
import sax
import jax
import pytest

def test_differentiation():
    ns = jnp.array([1, 2, 1.889+0.0035j, 1.802+0.0562j, 2.321+0.291j, 1.519+9j, 1])
    ds = jnp.array([1, 0.15, 0.03, 0.3, 0.2])

    wl = jnp.linspace(1.1, 1.8, 31)[:, None]
    theta_0=jnp.linspace(0.1*jnp.pi, 0.9*jnp.pi)[None, :]

    wl = 1.5
    theta_0 = 1

    k0 = 2*jnp.pi/wl
    kx = k0*jnp.sin(theta_0)

    pol = "p"

    def f(ds, ns, k0, kx):
        stack, info = staaax.angled_stratified.stack_smat_kx(
            ds, ns, k0, kx, pol=pol
        )

        smat_kx_direct = jax.jit(stack)()
        return jnp.abs(smat_kx_direct[('in', 'in')])
    
    g = jax.grad(f, argnums=(0,1,2,3))
    print(g(ds, ns, k0, kx))
