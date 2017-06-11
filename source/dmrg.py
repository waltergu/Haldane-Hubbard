import numpy as np
import HamiltonianPy.DMRG as DMRG
from HamiltonianPy import *
from HamiltonianPy.TensorNetwork import *
from config import *

__all__=['dmrgconstruct']

def dmrgconstruct(parameters,lattice,terms,targets,core='idmrg',**karg):
    priority,layers,mask=DEGFRE_FERMIONIC_PRIORITY,DEGFRE_FERMIONIC_LAYERS,['nambu']
    dmrg=DMRG.DMRG(
        dlog=       'log/dmrg',
        din=        'data/dmrg',
        dout=       'result/dmrg',
        log=        '%s_%s_%s_%s_DMRG.log'%(name,lattice.name.replace('+',str(2*len(targets))),parameters,repr(targets[-1])),
        name=       '%s_%s'%(name,lattice.name),
        mps=        MPS(mode='NB' if targets[-1] is None else 'QN'),
        lattice=    lattice,
        config=     IDFConfig(priority=priority,map=idfmap),
        degfres=    DegFreTree(mode='NB' if targets[-1] is None else 'QN',priority=priority,layers=layers,map=qnsmap),
        terms=      [term(*parameters) for term in terms],
        mask=       mask,
        dtype=      np.complex128
        )
    # edit the value of nmax and nmaxs if needed
    if core=='idmrg':
        tsg=DMRG.TSG(name='GROWTH',targets=targets,nmax=100,run=DMRG.DMRGTSG)
        dmrg.register(tsg)
    elif core=='fdmrg':
        tsg=DMRG.TSG(name='GROWTH',targets=targets,nmax=100,plot=False,run=DMRG.DMRGTSG)
        tss=DMRG.TSS(name='SWEEP',target=targets[-1],nsite=dmrg.nspb*len(targets)*2,nmaxs=[100,100],dependences=[tsg],run=DMRG.DMRGTSS)
        dmrg.register(tss)
    else:
        raise ValueError('dmrgconstruct error: not supported core %s.'%core)
    dmrg.summary()
    return dmrg
