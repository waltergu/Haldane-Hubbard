import numpy as np
import HamiltonianPy.ED as ED
import HamiltonianPy.VCA as VCA
from HamiltonianPy import *
from config import *

__all__=['vcaconstruct']

def vcaconstruct(parameters,basis,cell,lattice,terms,weiss,mask=['nambu'],**karg):
    assert len(parameters)==len(terms)+len(weiss)
    config=IDFConfig(priority=DEFAULT_FERMIONIC_PRIORITY,pids=lattice.pids,map=idfmap)
    cgf=ED.FGF(operators=fspoperators(config.table(),lattice),nstep=150,prepare=ED.EDGFP,run=ED.EDGF)
    vca=VCA.VCA(
        dout=       'result/vca',
        din=        'data/vca',
        cgf=        cgf,
        name=       '%s_%s_%s'%(name,lattice.name,basis.rep),
        basis=      basis,
        cell=       cell,
        lattice=    lattice,
        config=     config,
        terms=      [term(parameter) for term,parameter in zip(terms,parameters[:len(terms)])],
        weiss=      [term(parameter) for term,parameter in zip(weiss,parameters[len(terms):])],
        mask=       mask,
        dtype=      np.complex128
        )
    # edit tasks
    eb=VCA.EB(name='EB',path=hexagon_gkm(nk=100),mu=parameters[2]/2,emin=-5.0,emax=5.0,eta=0.05,ne=401,run=VCA.VCAEB)
    vca.register(eb)
    vca.summary()
