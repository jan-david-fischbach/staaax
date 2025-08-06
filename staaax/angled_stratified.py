from staaax.fresnel import fresnel_kx_direct
from staaax.propagation import propagation_kx
import sax

def stratified(ds, ns, k0, kx, pol="s"):
    r"""Sax Model Factory for stratified media.

    .. svgbob::
       :align: center

                n0 ┊ n1 ┊           ┊ n_N+1
       -> kz       ┊    ┊           ┊
     |         inc ┊    ┊           ┊
     V kx        \ ┊    ┊           ┊
                  \┊    ┊           ┊
                   +    ┊           ┊
                  /┊\   ┊           ┊
                 / ┊ \  ┊           ┊
                /  ┊  \ ┊           ┊  
               /   ┊   \┊           ┊ 
              /    ┊    +         \ ┊
             /     ┊   /┊\   ooo   \┊
            /      ┊  / ┊ \         +
           /       ┊ /  ┊          /┊\ 
          /        ┊/   ┊         / ┊ \ 
              

    Args:
        ds (List[float]): Thicknesses of the layers (`len(ds)=N`).
        ns (List[complex]): Refractive indices of the layers 
            (N+2 entries including halfspaces).
        k0 (float): Vacuum (angular) Wavenumber
        kx (float): Parallel component of the wavevector.
        pol (str, optional): Polarization, either "s" or "p". Defaults to "p".

    Returns:
        tuple[SDictModel, CircuitInfo]: The sax model
    """
    instances = {}
    connections = {}
    models = {
        "if": fresnel_kx_direct,
        "prop": propagation_kx,
    }
    
    i=-1
    for i in range(len(ds)):
        settings = dict(ni=ns[i], nj=ns[i+1], k0=k0, 
                        kx=kx, di=ds[i], pol=pol)
        
        instances[f'if_{i}'] = {'component': 'if',     'settings': settings}
        settings = sax.update_settings(settings, ni=ns[i+1])
        instances[f'prop_{i}'] = {'component': 'prop', 'settings': settings}
        connections[f'if_{i},right'] = f'prop_{i},left'
        connections[f'prop_{i},right'] = f'if_{i+1},left'

    settings = dict(ni=ns[-2], nj=ns[-1], k0=k0, kx=kx, pol=pol)
    instances[f'if_{i+1}'] = {'component': "if", 'settings': settings}
    ports = {"in": "if_0,left", "out": f"if_{len(ds)},right"}

    netlist = {
        "instances": instances,
        "connections": connections,
        "ports": ports
    }

    return sax.circuit(
        netlist = netlist,
        models = models,
    )

def update_settings(settings, ds, ns, **kwargs):
    """Convenience function to update the settings of a stratified model.
    The number of layers cannot change.
    Args:
        settings (dict): Sax settings of the stack
        ds: New thicknesses of the layers.
        ns: New refractive indices of the layers.
        **kwargs: Additional keyword arguments to pass to sax.update_settings.
    """

    N_layers = len([1 for key in settings.keys() if key.startswith('prop_')])

    for i in range(N_layers):
        settings[f'if_{i}']['ni'] = ns[i]
        settings[f'if_{i}']['nj'] = ns[i+1]
        settings[f'prop_{i}']['ni'] = ns[i+1]
        settings[f'prop_{i}']['di'] = ds[i]

    i+=1
    settings[f'if_{i}']['ni'] = ns[i]
    settings[f'if_{i}']['nj'] = ns[i+1]

    return sax.update_settings(settings, **kwargs)