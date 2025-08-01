import jax.numpy as jnp
import staaax.angled_stratified_treams
import sax

def test_integration():
    ns = [1, 2, 1.889+0.0035j, 1.802+0.0562j, 2.321+0.291j, 1.519+9j, 1]
    ds = [1, 0.15, 0.03, 0.3, 0.2]

    wl = 1.5
    theta_0=0.7
    k0 = 2*jnp.pi/wl
    kx = k0*jnp.sin(theta_0)

    pol = "p"
    pol_idx = 1 if pol=="p" else 0

    smat_treams = staaax.angled_stratified_treams.stratified_treams(
        ds, ns, k0=k0, kx=kx, poltype="parity"
    )

    stack, info = staaax.stratified(
        ds, ns, k0, kx, pol=pol
    )
    smat_kx_direct = stack()

    S_kx_direct, portmap = sax.sdense(smat_kx_direct)
    S_treams = jnp.array(smat_treams)[:,:,pol_idx, pol_idx][::-1]

    assert jnp.allclose(S_treams, S_kx_direct)