from HamiltonianPy import *
from source import *
from collections import OrderedDict
import numpy as np
import mkl

# tbatasks

def tbatasks(name,parameters,lattice,terms,jobs=()):
    import HamiltonianPy.FreeSystem as TBA
    tba=tbaconstruct(name,parameters,lattice,terms)
    if 'EB' in jobs:
        if len(lattice.vectors)==2:
            tba.register(EB(name='EB',path=hexagon_gkm(nk=100),run=TBA.TBAEB))
        elif len(lattice.vectors)==1:
            tba.register(EB(name='EB',path=KSpace(reciprocals=lattice.reciprocals,segments=[(-0.5,0.5)],end=True,nk=401),run=TBA.TBAEB))
        elif len(lattice.vectors)==0:
            tba.register(EB(name='EB',run=TBA.TBAEB))
    if 'GSE' in jobs:
        tba.register(TBA.GSE(name='GSE',filling=0.5,kspace=KSpace(reciprocals=lattice.reciprocals,nk=200) if len(lattice.vectors)>0 else None,run=TBA.TBAGSE))
    tba.summary()

# edtasks
def edtasks(name,parameters,basis,lattice,terms,jobs=()):
    import HamiltonianPy.ED as ED
    ed=edconstruct(name,parameters,[basis],lattice,terms)
    if 'EL' in jobs: ed.register(ED.EL(name='EL',path=BaseSpace(['U',np.linspace(0,30.0,301)]),ns=1,nder=2,run=ED.EDEL))
    if 'GSE' in jobs: ed.register(ED.EIGS(name='GSE',ne=1,run=ED.EDEIGS))
    ed.summary()

# vcatasks
def vcatasks(name,parameters,basis,cell,lattice,terms,weiss,baths=(),jobs=()):
    import HamiltonianPy.VCA as VCA
    vca=vcaconstruct(name,parameters,[basis],cell,lattice,terms,weiss)
    if 'EB' in jobs:
        vca.register(VCA.EB(name='EB',path=hexagon_gkm(reciprocals=cell.reciprocals,nk=100),mu=parameters['U']/2,emin=-5.0,emax=5.0,eta=0.05,ne=401,run=VCA.VCAEB))
    if 'TEB' in jobs:
        vca.register(VCA.EB(name='TEB',path=hexagon_gkm(reciprocals=cell.reciprocals,nk=100),mu=parameters['U']/2,run=VCA.VCATEB))
    if 'CN' in jobs:
        vca.register(BC(name='CN',BZ=KSpace(reciprocals=cell.reciprocals,nk=100),mu=parameters['U']/2,savedata=False,plot=False,run=VCA.VCABC))
    if 'GPM' in jobs:
        vca.add(GP(name='GP',mu=parameters['U']/2,BZ=KSpace(reciprocals=lattice.reciprocals,nk=100),run=VCA.VCAGP))
        vca.register(VCA.GPM(name='afm',BS=BaseSpace(('afm',np.linspace(0.0,0.1,11))),dependences=['GP'],run=VCA.VCAGPM))
    vca.summary()

if __name__=='__main__':
    mkl.set_num_threads(1)
    Engine.DEBUG=True

    # parameters
    parameters=OrderedDict()
    parameters['t1']=1.0
    parameters['t2']=0.2

    # tba
    #tbatasks(name,parameters,H2('1P-1P',nnb),[t1,t2],jobs=['EB'])
    #tbatasks(name,parameters,H2('1P-1P',nnb),[t1,t2],jobs=['GSE'])

    parameters['U']=5.00

    # ed
    #edtasks(name,parameters,FBasis(12,6,0.0),H6('1P-1P',nnb),[t1,t2,U],jobs=['EL'])
    #edtasks(name,parameters,FBasis(12,6,0.0),H6('1P-1P',nnb),[t1,t2,U],jobs=['GSE'])

    parameters['afm']=0.0

    # vca
    #vcatasks(name,parameters,FBasis(12,6,0.0),H2('1P-1P',nnb),H6('1P-1P',nnb),[t1,t2,U],[afm],jobs=['EB'])
    #vcatasks(name,parameters,FBasis(12,6,0.0),H2('1P-1P',nnb),H6('1P-1P',nnb),[t1,t2,U],[afm],jobs=['TEB'])
    #vcatasks(name,parameters,FBasis(12,6,0.0),H2('1P-1P',nnb),H6('1P-1P',nnb),[t1,t2,U],[afm],jobs=['CN'])
    #vcatasks(name,parameters,FBasis(12,6,0.0),H2('1P-1P',nnb),H6('1P-1P',nnb),[t1,t2,U],[afm],jobs=['GPM'])

    #vcatasks(name,parameters,FBasis(4,2,0.0),H2('1P-1P',nnb),H2('1P-1P',nnb),[t1,t2,U],[afm],jobs=['EB'])
    #vcatasks(name,parameters,FBasis(4,2,0.0),H2('1P-1P',nnb),H2('1P-1P',nnb),[t1,t2,U],[afm],jobs=['TEB'])
    #vcatasks(name,parameters,FBasis(4,2,0.0),H2('1P-1P',nnb),H2('1P-1P',nnb),[t1,t2,U],[afm],jobs=['CN'])
    #vcatasks(name,parameters,FBasis(4,2,0.0),H2('1P-1P',nnb),H2('1P-1P',nnb),[t1,t2,U],[afm],jobs=['GPM'])

    #vcatasks(name,parameters,FBasis(12,6,0.0),H6('1P-1P',nnb),H6('1P-1P',nnb),[t1,t2,U],[afm],jobs=['EB'])
    #vcatasks(name,parameters,FBasis(12,6,0.0),H6('1P-1P',nnb),H6('1P-1P',nnb),[t1,t2,U],[afm],jobs=['TEB'])
    #vcatasks(name,parameters,FBasis(12,6,0.0),H6('1P-1P',nnb),H6('1P-1P',nnb),[t1,t2,U],[afm],jobs=['CN'])
    #vcatasks(name,parameters,FBasis(12,6,0.0),H6('1P-1P',nnb),H6('1P-1P',nnb),[t1,t2,U],[afm],jobs=['GPM'])