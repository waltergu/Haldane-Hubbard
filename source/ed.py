import numpy as np
import HamiltonianPy.ED as ED
from HamiltonianPy import *
from .config import *

__all__=['edconstruct']

def edconstruct(name,parameters,sectors,lattice,terms,boundary=None,**karg):
    config=IDFConfig(priority=DEFAULT_FOCK_PRIORITY,pids=lattice.pids,map=idfmap)
    ed=ED.FED(
        dlog=       'log',
        din=        'data',
        dout=       'result/ed',
        name=       '%s_%s_%s'%(name,lattice.name,'_'.join(repr(sector) for sector in sectors)),
        parameters= parameters,
        map=        parametermap,
        sectors=    sectors,
        lattice=    lattice,
        config=     config,
        terms=      [term(**parameters) for term in terms],
        boundary=   boundary,
        dtype=      np.complex128
        )
    return ed
