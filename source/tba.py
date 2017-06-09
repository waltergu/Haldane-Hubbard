import numpy as np
import HamiltonianPy.FreeSystem as TBA
from HamiltonianPy import *
from config import *

__all__=['tbaconstruct']

def tbaconstruct(parameters,lattice,terms,**karg):
    assert len(parameters)==len(terms)
    config=IDFConfig(priority=DEFAULT_FERMIONIC_PRIORITY,pids=lattice.pids,map=idfmap)
    tba=TBA.TBA(
        dlog=       'log/tba',
        din=        'data/tba',
        dout=       'result/tba',
        log=        '%s_%s_%s_TBA.log'%(name,lattice.name,parameters),
        name=       '%s_%s'%(name,lattice.name),
        lattice=    lattice,
        config=     config,
        terms=      [term(parameter) for term,parameter in zip(terms,parameters)],
        dtype=      np.complex128
        )
    # edit tasks
    if karg.get('job',None)=='APP':
        apps=[]
        if lattice.name.count('P')==2:
            apps.append(EB(name='EB',path=hexagon_gkm(nk=100) if karg.get('kspace',True) else None,run=TBA.TBAEB))
        elif lattice.name.count('P')==1:
            apps.append(EB(name='EB',path=KSpace(reciprocals=lattice.reciprocals,segments=[(-0.5,0.5)],end=True,nk=401) if karg.get('kspace',True) else None,run=TBA.TBAEB))
        else:
            apps.append(EB(name='EB',run=TBA.TBAEB))
        for app in apps:
            tba.register(app)
        tba.summary()
    elif karg.get('job',None)=='GSE':
        kspace=KSpace(reciprocals=lattice.reciprocals,nk=200) if len(lattice.vectors)>0 and karg.get('kspace',True) else None
        GSE=tba.gse(filling=0.5,kspace=kspace)
        tba.log.open()
        tba.log<<Info.from_ordereddict({'Total':GSE,'Site':GSE/len(lattice)/(1 if kspace is None else kspace.rank('k'))})<<'\n'
        tba.log.close()
    return tba
