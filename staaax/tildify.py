from staaax.propagation import angled_sqrt
import jax.numpy as jnp

def tildify(k, Cs, bcs, nan_tolerance=0):
    return 1/len(Cs)*jnp.sum(jnp.array([
        angled_sqrt(
            (k**2 - C**2), 
            bc_angle=bc, 
            nan_tolerance=nan_tolerance) for C, bc in zip(Cs, bcs)
    ]), axis=0)

def inverse_tildify(k_tilde, branchpoints):
    if len(branchpoints) == 1:
        k = jnp.sqrt(k_tilde**2 + branchpoints[0]**2)
        return jnp.concat([k, -k])
    if len(branchpoints) != 2:
        raise NotImplementedError("Only 1 or 2 branchpoints are supported")
    C1 = branchpoints[0]
    C2 = branchpoints[1]
    k_tilde *=2
    p1 = ((C1**2+C2**2+k_tilde**2)/2)**2
    k = jnp.sqrt(p1-C1**2*C2**2)/k_tilde
    return jnp.concat([k, -k])

if __name__ == "__main__":
    import numpy as np
    n = 3
    k = (np.random.random(n)-0.5)*10 + 1j*(np.random.random(n)-0.5)*10
    Cs = (1,2)
    bcs = (np.pi/2, np.pi/2)

    k_tilde = tildify(k, Cs, bcs)

    k_prime = inverse_tildify(k_tilde, Cs)

    print(k, k_prime)