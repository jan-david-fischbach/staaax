import jax.numpy as jnp
import staaax.angled_stratified
import staaax.angled_stratified_treams
import jax

def eval(k0, single=True):
    ns = [1, 2, 1]
    ds = [1]
    kx = jnp.linspace(0.5, 1.25, 3) 
    if not single:
        k0 = k0[None, :]
        kx = kx[:, None]
    kx *= k0
    pol = "p"

    stack, info = staaax.angled_stratified.stack_smat_kx(
        ds, ns, k0, kx, pol=pol
    )
    return stack()

def test_batching_manual():
    eval(jnp.linspace(0.1, 0.3, 2), single=False)

def test_batching_vmap():
    out = jax.vmap(eval)(jnp.linspace(0.1, 0.3, 2))
    print(out)


if __name__ == "__main__":
    test_batching_manual()
    test_batching_vmap()