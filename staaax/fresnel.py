import jax.numpy as jnp
from staaax.propagation import angled_sqrt

def fresnel_kx_direct(
    ni=1, nj=1, k0=1, kx=0, pol="p", 
    bc_angle_i=jnp.pi/2, bc_angle_j=jnp.pi/2,
    bc_width_i=0, bc_width_j=0,
    **kwargs
):
    """ Fresnel coefficients for planar optical interfaces under angled illumination.
    
    .. svgbob::
       :align: center

                ni ┊ nj
       -> kz       ┊
     |         inc ┊
     V kx        \ ┊
                  \┊
                   +
                  /┊\ 
                 / ┊ \ 
              r /  ┊  \ t

    Args:
        ni (complex, optional): Refractive index of medium i. Defaults to 1.
        nj (complex, optional): Refractive index of medium j. Defaults to 1.
        k0 (float, optional): Vacuum wavenumber (angular). Defaults to 1.
        kx (float, optional): Wavenumber component parallel to the interface. Defaults to 0.
        pol (str, optional): Polarization, either "s" or "p". Defaults to "p".
        bc_angle_i (float, optional): Branchcut angle for calculating kz in medium i. Defaults to jnp.pi/2.
        bc_angle_j (float, optional): Branchcut angle for calculating kz in medium j. Defaults to jnp.pi/2.
        bc_width_i (float, optional): Cutout around the branchcut yielding nans. Usefull for visualizing the BC. Defaults to 0.
        bc_width_j (float, optional): See `bc_width_i`. Defaults to 0.

    Raises:
        ValueError: Polarization should be either 's'/'TM' or 'p'/'TE'.

    Returns:
        sax.SDict: The scattering matrix of the interface.
    """

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