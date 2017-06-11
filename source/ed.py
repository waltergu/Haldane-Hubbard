import numpy as np
import HamiltonianPy.ED as ED
from HamiltonianPy import *
from config import *

__all__=['edconstruct']

def edconstruct(parameters,basis,lattice,terms,**karg):
    config=IDFConfig(priority=DEFAULT_FERMIONIC_PRIORITY,pids=lattice.pids,map=idfmap)
    ed=ED.FED(
        dlog=       'log/ed',
        din=        'data/ed',
        dout=       'result/ed',
        log=        '%s_%s_%s_%s_ED.log'%(name,lattice.name,basis.rep,parameters),
        name=       '%s_%s_%s'%(name,lattice.name,basis.rep),
        basis=      basis,
        lattice=    lattice,
        config=     config,
        terms=      [term(*parameters) for term in terms],
        dtype=      np.complex128
        )
    return ed
