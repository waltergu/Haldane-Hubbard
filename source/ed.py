import numpy as np
import HamiltonianPy.ED as ED
from HamiltonianPy import *
from config import *

__all__=['edconstruct']

def edconstruct(parameters,basis,lattice,terms,**karg):
    assert len(parameters)==len(terms)
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
        terms=      [term(parameter) for term,parameter in zip(terms,parameters)],
        dtype=      np.complex128
        )
    # edit tasks
    if karg.get('job',None)=='APP':
        el=ED.EL(name='EL',path=BaseSpace(['U',np.linspace(0,30.0,301)]),ns=1,nder=2,run=ED.EDEL)
        ed.register(el)
        ed.summary()
    elif karg.get('job',None)=='GSE':
        GSE=ed.eig(k=1)[0]
        ed.log.open()
        ed.log<<Info.from_ordereddict({'Total':GSE,'Site':GSE/len(lattice)})<<'\n'
        ed.log.close()
    return ed
