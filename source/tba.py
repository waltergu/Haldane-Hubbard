import numpy as np
import HamiltonianPy.FreeSystem as TBA
from HamiltonianPy import *
from config import *

__all__=['tbaconstruct']

def tbaconstruct(parameters,lattice,terms,**karg):
    config=IDFConfig(priority=DEFAULT_FERMIONIC_PRIORITY,pids=lattice.pids,map=idfmap)
    tba=TBA.TBA(
        dlog=       'log/tba',
        din=        'data/tba',
        dout=       'result/tba',
        log=        '%s_%s_%s_TBA.log'%(name,lattice.name,parameters),
        name=       '%s_%s'%(name,lattice.name),
        lattice=    lattice,
        config=     config,
        terms=      [term(*parameters) for term in terms],
        dtype=      np.complex128
        )
    return tba
