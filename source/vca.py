import numpy as np
import HamiltonianPy.ED as ED
import HamiltonianPy.VCA as VCA
from HamiltonianPy import *
from config import *

__all__=['vcaconstruct']

def vcaconstruct(parameters,basis,cell,lattice,terms,weiss,mask=['nambu'],**karg):
    config=IDFConfig(priority=DEFAULT_FERMIONIC_PRIORITY,pids=lattice.pids,map=idfmap)
    # edit the value of nstep if needed
    cgf=ED.FGF(operators=fspoperators(config.table(),lattice),nstep=150,prepare=ED.EDGFP,run=ED.EDGF)
    vca=VCA.VCA(
        dlog=       'log/vca',
        din=        'data/vca',
        dout=       'result/vca',
        log=        '%s_%s_%s_%s_VCA.log'%(name,lattice.name,basis.rep,parameters),
        cgf=        cgf,
        name=       '%s_%s_%s'%(name,lattice.name,basis.rep),
        basis=      basis,
        cell=       cell,
        lattice=    lattice,
        config=     config,
        terms=      [term(*parameters) for term in terms],
        weiss=      [term(*parameters) for term in weiss],
        mask=       mask,
        dtype=      np.complex128
        )
    return vca
