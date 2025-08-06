import jax.numpy as jnp
import numpy.testing as npt
import pytest
import warnings
import staaax.angled_stratified_treams
import sax
import jax

n_freqs = 101
def example_stack():
    ns = [1, 2, 1.889+0.0035j, 1.802+0.0562j, 2.321+0.291j, 1.519+9j, 1]
    ds = [1, 0.15, 0.03, 0.3, 0.2]

    wl = jnp.linspace(1.4, 1.6, n_freqs)
    theta_0=0.7
    k0 = 2*jnp.pi/wl
    kx = k0*jnp.sin(theta_0)

    pol = "p"

    stack, info = staaax.stratified(
        ds, ns, k0, kx, pol=pol
    )

    return stack, ns, ds, k0, kx, pol

def test_integration():
    stack, ns, ds, k0, kx, pol = example_stack()
    pol_idx = 1 if pol=="p" else 0
    smat_kx_direct = stack()
    smat_treams = staaax.angled_stratified_treams.stratified_treams(
        ds, ns, k0=k0[0], kx=kx[0], poltype="parity"
    )

    S_kx_direct, portmap = sax.sdense(smat_kx_direct)
    S_treams = jnp.array(smat_treams)[:,:,pol_idx, pol_idx][::-1]

    assert jnp.allclose(S_treams, S_kx_direct[0])

def test_update_setting():
    stack, ns, ds, k0, kx, pol = example_stack()
    
    ns[2:5] = [2,3,4]
    settings = sax.get_settings(stack)
    new_settings = staaax.angled_stratified.update_settings(
        settings, ds, ns, k0=k0, kx=kx, pol=pol
    )

    import pprint
    pp = pprint.PrettyPrinter(depth=4)
    pp.pprint(settings)
    
    npt.assert_equal(float(new_settings['prop_1']['ni']), 2.0)

    S = stack(**new_settings)

def test_jit_update_setting():
    stack, ns, ds, k0, kx, pol = example_stack()
    
    @jax.jit
    def eval(n):
        ns[2:5] = [2,3-1e-4*i,4]
        settings = sax.get_settings(stack)
        new_settings = staaax.angled_stratified.update_settings(
            settings, ds, ns, k0=k0, kx=kx, pol=pol
        )

        S = stack(**new_settings)
        return S
    

    import time
    start = time.time()
    n_loops=1_000
    for i in range(n_loops):
        eval(3-1e-4*i)
    
    end = time.time()

    assert end - start < 3, f"Repeated eval took too long: {end - start:.2f} seconds"

@pytest.mark.timeout(15) 
@pytest.mark.parametrize("jit", [
    pytest.param("no-jit", marks=pytest.mark.xfail), 
    pytest.param("jit-before", marks=pytest.mark.xfail), 
    "jit-after"
])
def test_grad_update_setting(jit):
    stack, ns, ds, k0, kx, pol = example_stack()
    
    def eval(n):
        ns[2:5] = [2,3-1e-4*i,4]
        settings = sax.get_settings(stack)
        new_settings = staaax.angled_stratified.update_settings(
            settings, ds, ns, k0=k0, kx=kx, pol=pol
        )

        S = stack(**new_settings)
        return jnp.sum(jnp.abs(S['in', 'in'])**2)
    
    if jit == "jit-before":
        eval = jax.jit(eval)

    import time
    start = time.time()
    n_loops=1_000

    grad = jax.grad(eval)

    if jit == "jit-after":
        grad = jax.jit(grad)

    for i in range(n_loops):
        grad(3-1e-4*i)
    
    end = time.time()
    diff = end - start
    message = f"""
    {jit}: Gradient Evaluation took {diff:.2f} seconds. 
    That is {diff/n_loops:.4f} seconds per gradient eval
    (with a spectrum of {n_freqs} frequencies).
    """
    warnings.warn(message)
    if end - start > 3:
        assert False, message