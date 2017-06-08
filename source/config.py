from HamiltonianPy import *

__all__=['name','nneighbour','idfmap','qnsmap','t1','t2','U','Um','H2','H4','H6']

# The configs of the model
name='HH'
nneighbour=2

# idfmap
idfmap=lambda pid: Fermi(atom=pid.site%2,norbital=1,nspin=2,nnambu=1)

# qnsmap
qnsmap=lambda index: SzPQNS(index.spin-0.5)

# haldane hopping
def haldane_hopping(bond):
    theta=azimuthd(bond.rcoord)
    if abs(theta)<RZERO or abs(theta-120)<RZERO or abs(theta-240)<RZERO: 
        result=1
    else:
        result=-1
    if bond.spoint.pid.site%2==1:
        result=-result
    return result

# terms
t1=lambda value: Hopping('t1',value)
t2=lambda value: Hopping('t2',value*1.0j,neighbour=2,amplitude=haldane_hopping)
U=lambda value: Hubbard('U',value)
Um=lambda value: Hubbard('U',value,modulate=True)

# cluster
H2,H4,H6=Hexagon('H2'),Hexagon('H4'),Hexagon('H6')