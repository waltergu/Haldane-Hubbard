from HamiltonianPy import *
import numpy as np

__all__=['name','nnb','parametermap','idfmap','t1','t2','U','afm','H2','H6']

# The configs of the model
name='HH'
nnb=2

# parametermap
parametermap=None

# idfmap
idfmap=lambda pid: Fock(atom=pid.site%2,norbital=1,nspin=2,nnambu=1)

# haldane hopping
def haldane(bond):
    assert bond.spoint.pid.site%2==bond.epoint.pid.site%2
    theta,site=azimuthd(bond.rcoord),bond.epoint.pid.site%2
    if np.allclose(theta,60) or np.allclose(theta,180) or np.allclose(theta,300):
        result=1.0
    else:
        result=-1.0
    if site==1: result=-result
    return result

# terms
t1=lambda **parameters: Hopping('t1',parameters['t1'],neighbour=1)
t2=lambda **parameters: Hopping('t2',parameters['t2']*1.0j,neighbour=2,amplitude=haldane)
U=lambda **parameters: Hubbard('U',parameters['U'],modulate=True)
afm=lambda **parameters: Onsite('afm',parameters['afm'],indexpacks=sigmaz('sp')*sigmaz('sl'),modulate=True)

# cluster
H2=Hexagon('H2')
H6=Hexagon('H6')
