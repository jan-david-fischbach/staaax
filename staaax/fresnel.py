import jax.numpy as jnp
from staaax.propagation import angled_sqrt

def fresnel_kx_direct(
    ni=1, nj=1, k0=1, kx=0, pol="p", 
    bc_angle_i=jnp.pi/2, bc_angle_j=jnp.pi/2,
    bc_width_i=0, bc_width_j=0,
    **kwargs
):

    kiz = -angled_sqrt((k0*ni)**2 - kx**2 + 0j, 
                       bc_angle_i, nan_tolerance=bc_width_i)
    kjz =  angled_sqrt((k0*nj)**2 - kx**2 + 0j, bc_angle_j, 
                       nan_tolerance=bc_width_j)

    # print(f"{kiz=}; {kjz=}")
    if pol in ["s", "TE"]:
        eta = 1
    elif pol in ["p", "TM"]:
        eta = nj/ni
    else:
        raise ValueError(f"polarization should be either 's'/'TM' or 'p'/'TE'")
    
    r_ij = (eta*kiz+kjz/eta) / (eta*kiz-kjz/eta)
    t_ij = 2*kiz / (eta*kiz-kjz/eta)
    t_ji = (1-r_ij**2)/t_ij

    r_ji = -r_ij

    sdict = {
        ("left", "left"): r_ij,
        ("left", "right"): t_ij,
        ("right", "left"): t_ji,
        ("right", "right"): r_ji,
    }

    return sdict