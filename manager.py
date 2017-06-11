from HamiltonianPy import *
from HamiltonianPy.Misc import mpirun
from source import *
import numpy as np
import mkl

# tbatasks
def tbatasks(parameters,lattice,job='APP',kspace=True):
    import HamiltonianPy.FreeSystem as TBA
    tba=tbaconstruct(parameters,lattice,[t1,t2])
    if job=='APP':
        if lattice.name.count('P')==2:
            eb=EB(name='EB',path=hexagon_gkm(nk=100) if kspace else None,run=TBA.TBAEB)
        elif lattice.name.count('P')==1:
            eb=EB(name='EB',path=KSpace(reciprocals=lattice.reciprocals,segments=[(-0.5,0.5)],end=True,nk=401) if kspace else None,run=TBA.TBAEB)
        else:
            eb=EB(name='EB',run=TBA.TBAEB)
        tba.register(eb)
        tba.summary()
    elif job=='GSE':
        kspace=KSpace(reciprocals=lattice.reciprocals,nk=200) if len(lattice.vectors)>0 and kspace else None
        GSE=tba.gse(filling=0.5,kspace=kspace)
        tba.log.open()
        tba.log<<Info.from_ordereddict({'Total':GSE,'Site':GSE/len(lattice)/(1 if kspace is None else kspace.rank('k'))})<<'\n'
        tba.log.close()

# edtasks
def edtasks(parameters,basis,lattice,job='APP'):
    import HamiltonianPy.ED as ED
    ed=edconstruct(parameters,basis,lattice,[t1,t2,Um])
    if job=='APP':
        el=ED.EL(name='EL',path=BaseSpace(['U',np.linspace(0,30.0,301)]),ns=1,nder=2,run=ED.EDEL)
        ed.register(el)
        ed.summary()
    elif job=='GSE':
        GSE=ed.eig(k=1)[0]
        ed.log.open()
        ed.log<<Info.from_ordereddict({'Total':GSE,'Site':GSE/len(lattice)})<<'\n'
        ed.log.close()

# vcatasks
def vcatasks(parameters,basis,cell,lattice):
    import HamiltonianPy.VCA as VCA
    vca=vcaconstruct(parameters,basis,cell,lattice,[t1,t2,U],[])
    eb=VCA.EB(name='EB',path=hexagon_gkm(nk=100),mu=parameters[2]/2,emin=-5.0,emax=5.0,eta=0.05,ne=401,run=VCA.VCAEB)
    vca.register(eb)
    vca.summary()

if __name__=='__main__':
    mkl.set_num_threads(1)

    # When using log files, set it to be False
    Engine.DEBUG=True

    # Run the engines. Replace 'f' with the correct function
    #mpirn(f,parameters,bcast=True)

    # parameters
    m,n=2,2
    parameters=[-1.0,0.2,0.0]

    #tba tasks
    #tbatasks(parameters,H2('1P-1P',nneighbour),job='APP',kspace=True)
    #for m in [2,4,6]:
    #    tbatasks(parameters,H4('%sO-1P'%m,nneighbour),job='APP',kspace=True)
    #    tbatasks(parameters,H6('%sO-1P'%m,nneighbour),job='APP',kspace=True)
    #    tbatasks(parameters,H4('%sO-%sP'%(m,int(m*1.5)),nneighbour),job='APP',kspace=False)
    #    tbatasks(parameters,H6('%sO-%sP'%(m,int(m*1.5)),nneighbour),job='APP',kspace=False)
    #for m in [2,4,6]:
    #    tbatasks(parameters,H4('1P-%sO'%m,nneighbour),job='APP',kspace=True)
    #    tbatasks(parameters,H4('%sP-%sO'%(int(m*1.5),m),nneighbour),job='APP',kspace=False)
    #tbatasks(parameters,H4('%sO-%sP'%(m,n),nneighbour),job='GSE',kspace=False)

    # ed tasks
    #edtasks(parameters,FBasis((12,6)),H6('1P-1P',nneighbour),job='APP')
    #edtasks(parameters,FBasis((16,8)),H4('2O-1P',nneighbour),job='GSE')

    #vca tasks
    #vcatasks(parameters,FBasis((12,6)),H2('1P-1P',nneighbour),H6('1P-1P',nneighbour))

    # dmrg
    #dmrgconstruct(parameters,H4.cylinder(0,'1O-%sP'%n,nneighbour),[t1,t2,U],[SPQN((8*n*(i+1),0.0)) for i in xrange(m/2)],core='idmrg')
