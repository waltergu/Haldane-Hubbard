import numpy as np
import HamiltonianPy.DMRG as DMRG
from HamiltonianPy import *
from HamiltonianPy.TensorNetwork import *
from config import *

__all__=['dmrgconstruct']

def dmrgconstruct(parameters,lattice,terms,targets,core='idmrg',**karg):
    assert len(parameters)==len(terms)
    priority,layers,mask=DEGFRE_FERMIONIC_PRIORITY,DEGFRE_FERMIONIC_LAYERS,['nambu']
    dmrg=DMRG.DMRG(
        log=        Log('log/%s-%s-%s-%s.log'%(name,lattice.name.replace('+',str(2*len(targets))),parameters,repr(targets[-1]))),
        din=        'data/dmrg',
        dout=       'result/dmrg',
        name=       '%s_%s'%(name,lattice.name),
        mps=        MPS(mode='NB' if targets[-1] is None else 'QN'),
        lattice=    lattice,
        config=     IDFConfig(priority=priority,map=idfmap),
        degfres=    DegFreTree(mode='NB' if targets[-1] is None else 'QN',priority=priority,layers=layers,map=qnsmap),
        terms=      [term(parameter) for term,parameter in zip(terms,parameters)],
        mask=       mask,
        dtype=      np.complex128
        )
    if core=='idmrg':
        # edit 'nspb' and 'nmax'
        tsg=DMRG.TSG(name='GROWTH',targets=targets,nspb=len(lattice.block),nmax=100,run=DMRG.DMRGTSG)
        dmrg.register(tsg)
    elif core=='fdmrg':
        # edit 'nspb', 'nmax', 'nsite' and 'nmaxs'
        tsg=DMRG.TSG(name='GROWTH',targets=targets,nspb=len(lattice.block),nmax=100,plot=False,run=DMRG.DMRGTSG)
        tss=DMRG.TSS(name='SWEEP',target=targets[-1],nsite=len(lattice.block)*len(targets)*2,nmaxs=[100,100],dependences=[tsg],run=DMRGTSS)
        dmrg.register(tss)
    else:
        raise ValueError('dmrgconstruct error: not supported core %s.'%core)
    dmrg.summary()
