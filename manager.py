from HamiltonianPy import *
from HamiltonianPy.Misc import mpirun
from source import *
import numpy as np
import mkl

mkl.set_num_threads(1)

# When using log files, set it to be False
Engine.DEBUG=False

# Run the engines. Replace 'f' with the correct function
#mpirn(f,parameters,bcast=True)

# tba
parameters,terms=[-1.0,0.2],[t1,t2]
# APP
# bulk
#tbaconstruct(parameters,H2('1P-1P',nneighbour),terms,job='APP',kspace=True)
# armchair
#for m in [2,4,6]:
#    tbaconstruct(parameters,H4('%sO-1P'%m,nneighbour),terms,job='APP',kspace=True)
#    tbaconstruct(parameters,H6('%sO-1P'%m,nneighbour),terms,job='APP',kspace=True)
#    tbaconstruct(parameters,H4('%sO-%sP'%(m,int(m*1.5)),nneighbour),terms,job='APP',kspace=False)
#    tbaconstruct(parameters,H6('%sO-%sP'%(m,int(m*1.5)),nneighbour),terms,job='APP',kspace=False)
# zigzag
#for m in [2,4,6]:
#    tbaconstruct(parameters,H4('1P-%sO'%m,nneighbour),terms,job='APP',kspace=True)
#    tbaconstruct(parameters,H4('%sP-%sO'%(int(m*1.5),m),nneighbour),terms,job='APP',kspace=False)
# GSE
#tbaconstruct(parameters,H4('2O-2P',nneighbour),terms,job='GSE',kspace=False)

# ed
parameters,terms=[-1.0,0.2,1.0],[t1,t2,Um]
#edconstruct(parameters,FBasis((12,6)),H6('1P-1P',nneighbour),terms,job='APP')
#edconstruct(parameters,FBasis((16,8)),H4('2O-1P',nneighbour),terms,job='GSE')

# vca
parameters,terms,weiss=[-1.0,0.2,0.0],[t1,t2,U],[]
#vcaconstruct(parameters,FBasis((12,6)),H2('1P-1P',nneighbour),H6('1P-1P',nneighbour),terms,weiss)

# dmrg
m,n=2,1
parameters,terms=[-1.0,0.2,0.0],[t1,t2,U]
#dmrgconstruct(parameters,H4.cylinder(0,'1O-%sP'%n,nneighbour),terms,[SPQN((8*n*(i+1),0.0)) for i in xrange(m/2)],core='idmrg')
