from HamiltonianPy import *
from HamiltonianPy.Misc import mpirun
from source import *
import numpy as np
import mkl

# tbatasks
def tbatasks(parameters,lattice,job='EB',kspace=True):
    import HamiltonianPy.FreeSystem as TBA
    tba=tbaconstruct(parameters,lattice,[t1,t2])
    if job=='EB':
        if len(lattice.vectors)==2:
            tba.register(EB(name='EB',path=hexagon_gkm(nk=100) if kspace else None,run=TBA.TBAEB))
        elif len(lattice.vectors)==1:
            tba.register(EB(name='EB',path=KSpace(reciprocals=lattice.reciprocals,segments=[(-0.5,0.5)],end=True,nk=401) if kspace else None,run=TBA.TBAEB))
        elif len(lattice.vectors)==0:
            tba.register(EB(name='EB',run=TBA.TBAEB))
    if job=='GSE':
        tba.register(TBA.GSE(name='GSE',filling=0.5,kspace=KSpace(reciprocals=lattice.reciprocals,nk=200) if len(lattice.vectors)>0 and kspace else None,run=TBA.TBAGSE))
    tba.summary()

# edtasks
def edtasks(parameters,basis,lattice,job='EL'):
    import HamiltonianPy.ED as ED
    ed=edconstruct(parameters,basis,lattice,[t1,t2,Um])
    if job=='EL': ed.register(ED.EL(name='EL',path=BaseSpace(['U',np.linspace(0,30.0,301)]),ns=1,nder=2,run=ED.EDEL))
    if job=='GSE': ed.register(GSE(name='GSE',run=ED.EDGSE))
    ed.summary()

# vcatasks
def vcatasks(parameters,basis,cell,lattice,job='EB'):
    import HamiltonianPy.VCA as VCA
    vca=vcaconstruct(parameters,basis,cell,lattice,[t1,t2,U],[])
    if job=='EB': vca.register(VCA.EB(name='EB',path=hexagon_gkm(nk=100),mu=parameters[2]/2,emin=-5.0,emax=5.0,eta=0.05,ne=401,run=VCA.VCAEB))
    vca.summary()

if __name__=='__main__':
    mkl.set_num_threads(1)

    # When using log files, set it to be False
    Engine.DEBUG=True

    # Run the engines. Replace 'f' with the correct function
    #mpirn(f,parameters,bcast=True)

    # parameters
    m,n=2,1
    parameters=[-1.0,0.2,0.0]

    #tba tasks
    #tbatasks(parameters,H2('1P-1P',nneighbour),job='EB',kspace=True)
    #for m in [2,4,6]:
    #    tbatasks(parameters,H4('%sO-1P'%m,nneighbour),job='EB',kspace=True)
    #    tbatasks(parameters,H6('%sO-1P'%m,nneighbour),job='EB',kspace=True)
    #    tbatasks(parameters,H4('%sO-%sP'%(m,int(m*1.5)),nneighbour),job='EB',kspace=False)
    #    tbatasks(parameters,H6('%sO-%sP'%(m,int(m*1.5)),nneighbour),job='EB',kspace=False)
    #for m in [2,4,6]:
    #    tbatasks(parameters,H4('1P-%sO'%m,nneighbour),job='EB',kspace=True)
    #    tbatasks(parameters,H4('%sP-%sO'%(int(m*1.5),m),nneighbour),job='EB',kspace=False)
    #tbatasks(parameters,H4('%sO-%sP'%(m,n),nneighbour),job='GSE',kspace=False)

    # ed tasks
    #edtasks(parameters,FBasis((12,6)),H6('1P-1P',nneighbour),job='EL')
    #edtasks(parameters,FBasis((8*m*n,4*m*n)),H4('%sO-%sP'%(m,n),nneighbour),job='GSE')

    #vca tasks
    #vcatasks(parameters,FBasis((12,6)),H2('1P-1P',nneighbour),H6('1P-1P',nneighbour),job='EB')

    # dmrg
    #dmrgconstruct(parameters,H4.cylinder(0,'1O-%sP'%n,nneighbour),[t1,t2,U],[SPQN((8*n*(i+1),0.0)) for i in xrange(m/2)],core='idmrg')
