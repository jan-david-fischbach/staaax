import treams
def stratified_treams(ds, ns, k0, kx, poltype=None):
    """`treams` implementation of stratified media for reference.

    Args:
        ds (_type_): _description_
        ns (_type_): _description_
        k0 (_type_): _description_
        kx (_type_): _description_
        poltype (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """

    pwb = treams.PlaneWaveBasisByComp([[kx, 0, 0],[kx, 0, 1]])
    stack = []

    for i in range(len(ds)):
        inter = treams.SMatrices.interface(pwb, k0, [ns[i]**2, ns[i+1]**2], poltype=poltype)
        prop  = treams.SMatrices.propagation([0,0,ds[i]], pwb, k0, [ns[i+1]**2], poltype=poltype)
        stack.append(inter)
        stack.append(prop)
        
    inter = treams.SMatrices.interface(pwb, k0, [ns[-2]**2, ns[-1]**2], poltype=poltype)
    stack.append(inter)

    return treams.SMatrices.stack(stack)