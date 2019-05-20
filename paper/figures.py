import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as itp
import pdb

def lattice():
    from HamiltonianPy import PID,Point,tiling,translation,rotation,azimuthd,Lattice,SuperLattice
    from HamiltonianPy import Hexagon
    import itertools as it

    plt.ion()
    plt.figure(figsize=(6,6))
    gs=plt.GridSpec(2,2)
    gs.update(top=0.95,bottom=0.0,left=0.0,right=0.98,wspace=0,hspace=0)

    ax=plt.subplot(gs[0,0])
    ax.axis('off')
    ax.axis('equal')
    ax.set_ylim(-1.6,2.1)

    # Lattice
    h6=Hexagon('H6')
    H18=SuperLattice(
            name=           'H18',
            sublattices=    [
                            Lattice(name='0',rcoords=translation(h6.rcoords,vector=np.array([0,0]))),
                            Lattice(name='1',rcoords=translation(h6.rcoords,vector=h6.vectors[0])),
                            Lattice(name='2',rcoords=translation(h6.rcoords,vector=h6.vectors[1])),
                            ],
            neighbours      =1
            )
    for bond in H18.bonds:
        if bond.neighbour==0:
            x,y=bond.spoint.rcoord[0],bond.spoint.rcoord[1]
            ax.scatter(x,y,s=np.pi*5**2,color='brown' if bond.spoint.pid.site%2==0 else 'blue',zorder=3)
        else:
            x1,y1=bond.spoint.rcoord[0],bond.spoint.rcoord[1]
            x2,y2=bond.epoint.rcoord[0],bond.epoint.rcoord[1]
            ax.plot([x1,x2],[y1,y2],color='black',linewidth=2,zorder=1)

    # Tilling of lattice
    a1=np.array([1.5,np.sqrt(3)/2])
    a2=np.array([1.5,-np.sqrt(3)/2])
    ps=[None]*13
    ps[0]=Point(pid=PID(site=0),rcoord=[1.5,np.sqrt(3)/6])
    ps[1]=Point(pid=PID(site=1),rcoord=[1.0,2*np.sqrt(3)/3])
    ps[2]=Point(pid=PID(site=2),rcoord=ps[0].rcoord-a2)
    ps[3]=Point(pid=PID(site=3),rcoord=ps[1].rcoord-a1)
    ps[4]=Point(pid=PID(site=4),rcoord=ps[0].rcoord-a1)
    ps[5]=Point(pid=PID(site=5),rcoord=ps[1].rcoord-a1+a2)
    ps[6]=Point(pid=PID(site=6),rcoord=ps[0].rcoord-a2+a1)
    ps[7]=Point(pid=PID(site=7),rcoord=ps[1].rcoord+a1)
    ps[8]=Point(pid=PID(site=8),rcoord=ps[0].rcoord+a1)
    ps[9]=Point(pid=PID(site=9),rcoord=ps[1].rcoord+a2)
    ps[10]=Point(pid=PID(site=10),rcoord=ps[0].rcoord+a2)
    ps[11]=Point(pid=PID(site=11),rcoord=ps[5].rcoord+a2)
    ps[12]=Point(pid=PID(site=12),rcoord=ps[0].rcoord+a2-a1)
    SuperH14=Lattice.compose(name='SuperH14',points=ps)
    for bond in SuperH14.bonds:
        if bond.neighbour==1:
            x1,y1=bond.spoint.rcoord[0],bond.spoint.rcoord[1]
            x2,y2=bond.epoint.rcoord[0],bond.epoint.rcoord[1]
            ax.plot([x1,x2],[y1,y2],'g--',lw=2)
    ax.text(0.38,1.2,'H6',fontsize=16,color='g')

    # A, B site
    coord=H18.rcoord(PID('0',5))
    ax.text(coord[0]-0.1,coord[1]+0.13,'A',fontsize=16)
    coord=H18.rcoord(PID('1',0))
    ax.text(coord[0]-0.30,coord[1],'B',fontsize=16)

    # t term
    ax.annotate(s='',xy=H18.rcoord(PID('2',1)),xytext=H18.rcoord(PID('2',2)),arrowprops={'arrowstyle':'<->','color':'black','linewidth':2.0,'zorder':3})
    coord=(H18.rcoord(PID('2',1))+H18.rcoord(PID('2',2)))/2
    ax.text(coord[0],coord[1]-0.3,"$t$",fontweight='bold',color='black',fontsize=20)

    # t' term
    b1=np.array([1.0,0.0])
    b2=np.array([0.5,np.sqrt(3)/2])
    def farrow(ax,coord,inc):
        x,y=coord,coord+inc
        ax.plot([x[0],y[0]],[x[1],y[1]],color='red',linewidth=2)
        center,disp=(x+y)/2,inc/5
        ax.annotate(s='',xy=center-disp/2,xytext=center+disp/2,arrowprops={'color':'red','linewidth':2.5,'arrowstyle':'->','zorder':3})

    farrow(ax,H18.rcoord(PID('0',0)),b2)
    farrow(ax,H18.rcoord(PID('0',2)),(b1-b2))
    farrow(ax,H18.rcoord(PID('0',4)),-b1)
    ax.text(0.5-0.17,np.sqrt(3)/6-0.15,"$it'$",fontweight='bold',color='black',fontsize=20)

    farrow(ax,H18.rcoord(PID('1',5)),-b2)
    farrow(ax,H18.rcoord(PID('1',3)),-(b1-b2))
    farrow(ax,H18.rcoord(PID('1',1)),b1)
    ax.text(2.0-0.17,2*np.sqrt(3)/3-0.15,"$it'$",fontweight='bold',color='black',fontsize=20)

    # Hubbard term
    coord,disp,inc,delta=H18.rcoord(PID('2',3)),np.array([0.05,0.0]),np.array([0.0,0.25]),np.array([0.04,0.0])
    ax.annotate(s='',xy=coord-disp-inc-delta-0.015,xytext=coord-disp+inc+delta-0.015,arrowprops={'arrowstyle':'->','linewidth':2,'color':'purple','zorder':4})
    ax.annotate(s='',xy=coord+disp+inc+delta+0.015,xytext=coord+disp-inc-delta+0.015,arrowprops={'arrowstyle':'->','linewidth':2,'color':'purple','zorder':4})

    ax.text(coord[0]-0.1,coord[1]+0.25,"$U$",fontweight='bold',color='black',fontsize=20)

    # Encircle the AB site as a unit cell
    p0=Point(pid=PID('uc',0),rcoord=(h6.rcoords[1]+h6.rcoords[2])/2)
    p1=Point(pid=PID('uc',1),rcoord=(h6.rcoords[0]+h6.rcoords[3])/2)
    p2=Point(pid=PID('uc',2),rcoord=[-p1.rcoord[0],p1.rcoord[1]])
    p3=Point(pid=PID('uc',3),rcoord=[-p0.rcoord[0],p0.rcoord[1]])
    unitcell=Lattice.compose(name='uc',points=[p0,p1,p2,p3],neighbours=2)
    for bond in unitcell.bonds:
        if bond.neighbour>0:
            x1,y1=bond.spoint.rcoord
            x2,y2=bond.epoint.rcoord
            ax.plot([x1,x2],[y1,y2],ls='--',color='purple',lw=2)
    ax.text(-0.30,0.8,'H2',va='center',ha='right',fontsize=16,color='purple')

    # numbering of subplot
    ax.text(-0.4,1.85,"(a)",fontsize=16,color='black')


    # The Brillouin Zone
    ax=plt.subplot(gs[0,1])
    ax.axis('equal')
    ax.axis('off')
    ax.set_ylim(-1.0,1.0)
    ax.set_xlim(-0.7,0.7)

    # FBZ of H2
    center=(h6.rcoords[0]+h6.rcoords[5])/2
    bzrcoords=translation(rotation(cluster=h6.rcoords,angle=np.pi/6,center=center),-center)
    BZ=Lattice(name="BZ",rcoords=bzrcoords)
    for bond in BZ.bonds:
        if bond.neighbour==1:
            x1,y1=bond.spoint.rcoord[0],bond.spoint.rcoord[1]
            x2,y2=bond.epoint.rcoord[0],bond.epoint.rcoord[1]
            ax.plot([x1,x2],[y1,y2],linewidth=2,color='black')

    # FBZ of H6
    origin=np.array([0.0,0.0])
    BZ6Ps=[None]*6
    BZ6Ps[0]=Point(pid=PID('BZ6',0),rcoord=(bzrcoords[0]+bzrcoords[1]+origin)/3)
    BZ6Ps[1]=Point(pid=PID('BZ6',1),rcoord=(bzrcoords[1]+bzrcoords[2]+origin)/3)
    BZ6Ps[2]=Point(pid=PID('BZ6',2),rcoord=(bzrcoords[2]+bzrcoords[5]+origin)/3)
    BZ6Ps[3]=Point(pid=PID('BZ6',3),rcoord=(bzrcoords[5]+bzrcoords[4]+origin)/3)
    BZ6Ps[4]=Point(pid=PID('BZ6',4),rcoord=(bzrcoords[4]+bzrcoords[3]+origin)/3)
    BZ6Ps[5]=Point(pid=PID('BZ6',5),rcoord=(bzrcoords[3]+bzrcoords[0]+origin)/3)
    BZ6=Lattice.compose(name="BZ6",points=BZ6Ps)
    for bond in BZ6.bonds:
        if bond.neighbour==1:
            x1,y1=bond.spoint.rcoord[0],bond.spoint.rcoord[1]
            x2,y2=bond.epoint.rcoord[0],bond.epoint.rcoord[1]
            ax.plot([x1,x2],[y1,y2],linewidth=2,color='black',ls='-.')

    # axis
    p=(BZ.rcoord(PID('BZ',2))+BZ.rcoord(PID('BZ',5)))/2
    ax.plot([0.0,p[0]],[0.0,p[1]],ls='--',lw=2,color='k')
    p=BZ.rcoord(PID('BZ',2))
    ax.plot([0.0,p[0]],[0.0,p[1]],ls='--',lw=2,color='k')

    for disp in [np.array([0.60,0.0]),np.array([0.0,0.55])]:
        ax.arrow(-disp[0],-disp[1],2*disp[0],2*disp[1],head_width=0.03,head_length=0.05,fc='k',ec='k')
    ax.text(0.62,-0.03,"$\mathrm{k_x}$",va='top',ha='center',fontsize=14,color='black')
    ax.text(-0.02,0.56,"$\mathrm{k_y}$",va='center',ha='right',fontsize=14,color='black')

    # high-symmetric points
    ax.text(-0.05,0.01,"$\Gamma$",ha='center',va='bottom',fontsize=16,color='black')
    ax.text(0.6,0.05,"$K$",ha='center',fontsize=16,color='black')
    ax.text(0.3,0.5,"$K'$",fontsize=16,color='black')
    ax.text(0.475,0.275,"$M$",ha='center',va='center',fontsize=16,color='black')
    ax.text(0.31,0.125,"$K_R$",ha='left',va='center',fontsize=16,color='black')
    ax.text(0.31,-0.01,"$M_R$",ha='left',va='top',fontsize=16,color='black')
    ax.text(0.31,-0.165,"$K'_R$",ha='left',va='center',fontsize=16,color='black')

    # numbering of subplot
    ax.text(-0.65,0.6,"(b)",ha='left',va='top',fontsize=16,color='black')


    # edge geometry
    ax=plt.subplot(gs[1,:])
    ax.axis('equal')
    ax.axis('off')
    ax.set_ylim(-0.8,2.0)

    vectors=rotation(cluster=h6.vectors,angle=-np.pi/6)
    Edge=Lattice(name='edge',rcoords=tiling(BZ.rcoords,vectors=vectors,translations=it.product(range(2),range(4))))
    for bond in Edge.bonds:
        if bond.neighbour==0:
            x,y=bond.spoint.rcoord[0],bond.spoint.rcoord[1]
            ax.scatter(x,y,s=np.pi*5**2,color='blue' if bond.spoint.pid.site%2==0 else 'brown',zorder=3)
        else:
            x1,y1=bond.spoint.rcoord[0],bond.spoint.rcoord[1]
            x2,y2=bond.epoint.rcoord[0],bond.epoint.rcoord[1]
            ax.plot([x1,x2],[y1,y2],color='black',linewidth=2,zorder=2)

    # edge lattice tilling
    p=np.array([-3*np.sqrt(3)/4,-0.75])
    rect=Lattice(name='rect',rcoords=tiling([p],vectors=[vectors[0]*2,vectors[1]],translations=it.product(range(2),range(5))),neighbours=3)
    for bond in rect.bonds:
        if bond.neighbour in (1,3):
            theta=azimuthd(bond.rcoord)
            if not np.allclose(theta,120):
                x1,y1=bond.spoint.rcoord[0],bond.spoint.rcoord[1]
                x2,y2=bond.epoint.rcoord[0],bond.epoint.rcoord[1]
                ax.plot([x1,x2],[y1,y2],color='green',linewidth=1.5,zorder=1)

    coord1=p+vectors[0]
    coord2=coord1+vectors[1]*4
    ax.plot([coord1[0],coord2[0]],[coord1[1],coord2[1]],color='green',ls='--',linewidth=1.5,zorder=1)

    # x,y direction
    coord=p+vectors[1]*4+np.array([0.3,0.0])
    ax.arrow(coord[0],coord[1],0.75,0.0,head_width=0.05,head_length=0.1,fc='k',ec='k')
    ax.arrow(coord[0],coord[1],0.75/2,0.75*np.sqrt(3)/2,head_width=0.05,head_length=0.1,fc='k',ec='k')
    ax.text(coord[0]+0.75,coord[1]+0.1,'x',fontsize=16,color='black')
    ax.text(coord[0]+0.5,coord[1]+0.75,'y',fontsize=16,color='black')

    # numbering of subplot
    ax.text(-1.0,2,"(c)",fontsize=16,color='black')

    pdb.set_trace()
    plt.savefig('lattice.pdf')
    plt.close()

def phase():
    plt.ion()

    fig,axes=plt.subplots(nrows=1,ncols=3)
    fig.subplots_adjust(left=0.060,right=0.945,top=0.960,bottom=0.165,wspace=0.275)

    # phase diagram
    ax=axes[0]
    t=np.array([0.00,0.05,0.10,0.15,0.20,0.25,0.30])
    u_gap=np.array([0.00,2.80,3.60,4.10,4.60,5.00,5.40])
    u_afm=np.array([3.84,4.02,4.45,4.98,5.62,6.49,8.00])
    X=np.linspace(t.min(),t.max(),201)
    Y_gap=itp.splev(X,itp.splrep(t,u_gap,k=2),der=0)
    Y_afm=itp.splev(X,itp.splrep(t,u_afm,k=2),der=0)
    ax.plot(X,Y_gap,'blue',lw=2.0,label='by $\Delta$')
    ax.plot(X,Y_afm,'red',lw=2.0,label='by $M$')
    ax.legend(loc='upper left',fontsize=16,framealpha=0.75,fancybox=True)
    ax.vlines(0.20,0.0,10.0,colors='grey',linestyles='dotted',alpha=0.5)
    ax.scatter(0.20,4.0,color='k',s=10,zorder=4,facecolors='none')
    ax.scatter(0.20,4.6,color='k',s=10,zorder=4,facecolors='none')
    ax.scatter(0.20,5.0,color='k',s=10,zorder=4,facecolors='none')
    ax.scatter(0.20,5.75,color='k',s=10,zorder=4,facecolors='none')
    ax.scatter(0.20,5.87,color='k',s=10,zorder=4,facecolors='none')
    ax.scatter(0.20,6.0,color='k',s=10,zorder=4,facecolors='none')
    ax.text(0.14,2.00,'CI',va='center',ha='left',color='k',fontsize=22)
    ax.text(0.23,5.58,'NMI',va='center',ha='left',color='k',fontsize=22)
    ax.text(0.175,8.00,"AFM",va='center',ha='left',color='k',fontsize=22)
    ax.set_xlim(0.0,0.3)
    ax.set_ylim(0.0,10.0)
    ax.minorticks_on()
    ax.set_xticks(np.linspace(0.0,0.3,4))
    ax.set_yticks(np.linspace(0.0,10.0,6))
    for tick in ax.get_xticklabels():
        tick.set_fontsize(14)
    for tick in ax.get_yticklabels():
        tick.set_fontsize(14)
    ax.set_xlabel("$t'/t$",fontdict={'fontsize':18})
    ax.set_ylabel("$U/t$",fontdict={'fontsize':18})
    ax.text(-0.07,9.8,'(a)',fontsize=18,ha='left',va='center',color='black')


    # Potthoff potential
    ax=axes[1]
    Us=['6.0','5.87','5.75','5.0','4.6','4.0']
    for U in Us:
        data=np.loadtxt("../result/vca/HH_H6_0.5_-1.0_0.2j_%s_VCA_GPS.dat"%U)
        X=np.linspace(data[:,0].min(),data[:,0].max(),100)
        gp=itp.splev(X,itp.splrep(data[:,0],data[:,1]-data[0,1],k=3),der=0)
        ax.plot(X,gp,lw=3)
    ax.axhline(y=0.0,xmin=0.0,xmax=1.0,ls='--',linewidth=1,color='black')
    ax.legend(["$U=6.00$","$U=5.87$",'$U=5.75$',"$U=5.00$","$U=4.60$","$U=4.00$"],loc='upper right',fontsize=16,framealpha=0.75,fancybox=True)
    ax.annotate(s='',xy=[0.0228,-0.00004],xytext=[0.0228,-0.00024],arrowprops={'arrowstyle':'->','color':'red','linewidth':2.0,'zorder':3})
    ax.annotate(s='',xy=[0.0306,-0.00017],xytext=[0.0306,-0.00037],arrowprops={'arrowstyle':'->','color':'green','linewidth':2.0,'zorder':3})
    ax.annotate(s='',xy=[0.0362,-0.00035],xytext=[0.0362,-0.00055],arrowprops={'arrowstyle':'->','color':'blue','linewidth':2.0,'zorder':3})
    ax.text(-0.015,0.0014,'(b)',fontsize=18,color='black')
    ax.set_xlim(0.0,0.06)
    ax.set_ylim(-0.00060,0.0015)
    ax.set_xticks(np.linspace(0.0,0.06,5))
    ax.set_yticks(np.linspace(-0.0005,0.0015,5))
    ax.ticklabel_format(style='sci',scilimits=(-0.0006,0.0015),axis='y')
    for tick in ax.get_xticklabels():
        tick.set_fontsize(14)
    for tick in ax.get_yticklabels():
        tick.set_fontsize(14)
    ax.set_xlabel('$h_{AF}/t$',fontsize=18)
    ax.set_ylabel("$\Delta\Omega/t$",fontsize=18)
    ax.minorticks_on()


    # Delta, M and D
    # single particle gap
    ax=axes[2]
    data=np.loadtxt('../result/vca/HH_H6_0.5_-1.0_0.2j_VCA_GAP.dat')
    X=np.linspace(data[:,0].min(),data[:,0].max(),200)
    gap=itp.splev(X,itp.splrep(data[:,0],data[:,1],k=1),der=0)
    l1=ax.plot(X,gap,lw=3,color='purple',label="$\Delta$")
    ax.set_ylim(0,2.0)
    ax.set_xlim(2.50,6.50)
    ax.minorticks_on()
    ax.set_xticks(np.linspace(2.5,6.5,5))
    ax.set_yticks(np.linspace(0.0,2.0,5))
    for tick in ax.get_xticklabels():
        tick.set_fontsize(14)
    for tick in ax.get_yticklabels():
        tick.set_color('purple')
        tick.set_fontsize(14)
    ax.set_xlabel('$U/t$',fontsize=18)
    ax.set_ylabel('$\Delta/t$',color='purple',fontsize=18)

    # antiferromagnetic moment
    ax=ax.twinx()
    ax.minorticks_on()
    ax.set_ylim(-0.0001,0.90)
    data=np.loadtxt("../result/vca/HH_H6_0.5_-1.0_0.2j_VCA_OP.dat")
    X=np.linspace(data[:,0].min(),data[:,0].max(),200)
    af=itp.splev(X,itp.splrep(data[:,0],data[:,1],k=1),der=0)
    l2=ax.plot(X,af,lw=3,color='blue',label="$M$")
    ax.plot([2.5,5.621],[0.0,0.0],color='blue',lw=3)
    ax.set_yticks(np.linspace(0.0,0.9,4))
    for tick in ax.get_yticklabels():
        tick.set_color('blue')
        tick.set_fontsize(14)
    ax.set_ylabel('$M$',color='blue',fontsize=18)

    # double occupancy
    ax=axes[2]
    data=np.loadtxt('../result/vca/HH_H6_0.5_-1.0_0.2j_VCA_DO.dat')
    scaling=1.95/(0.22-0.07)
    l3=ax.plot(data[:,0],(data[:,1]-0.07)*scaling,ls='--',lw=3,color='green',label="$D$")

    ls=l1+l2+l3
    ax.legend(ls,[l.get_label() for l in ls],loc='lower left',fontsize=14,framealpha=0.75,fancybox=True)
    ax.axvline(x=4.600,ymin=0,ymax=1,ls='--',linewidth=1,color='black',zorder=2)
    ax.axvline(x=5.620,ymin=0,ymax=1,ls='--',linewidth=1,color='black',zorder=2)
    ax.text(4.59,1.8,'$U_M$',ha='right',fontsize=18,color='black')
    ax.text(5.59,1.8,'$U_{AF}$',ha='right',fontsize=18,color='black')
    ax.text(1.50,1.90,'(c)',fontsize=18,color='black')


    pdb.set_trace()
    plt.savefig('phase.pdf')
    plt.close()

def h2chernnumber():
    plt.ion()

    fig,axes=plt.subplots(nrows=1,ncols=2)
    fig.subplots_adjust(left=0.090,right=0.980,top=0.960,bottom=0.145,wspace=0.250)

    # h2 phase diagram
    ax=axes[0]
    t=np.array([0.00,0.05,0.10,0.15,0.20,0.25,0.30])
    u_gap=np.array([0.00,2.80,3.60,4.10,4.60,5.00,5.40])
    u_afm=np.array([3.84,4.02,4.45,4.98,5.62,6.49,8.00])
    u_cn=np.array([3.84,4.12,4.6,5.20,5.87,6.8,8.33])
    X=np.linspace(t.min(),t.max(),201)
    Y_gap=itp.splev(X,itp.splrep(t,u_gap,k=2),der=0)
    Y_afm=itp.splev(X,itp.splrep(t,u_afm,k=2),der=0)
    Y_cn=itp.splev(X,itp.splrep(t,u_cn,k=2),der=0)
    ax.plot(X,Y_gap,'blue',lw=2.0)
    ax.plot(X,Y_afm,'red',lw=2.0)
    ax.fill_between(X,y1=Y_gap,y2=0.0,color=(0.8,0.5,1.0))
    ax.fill_between(X,y1=Y_cn,y2=Y_gap,color=(0.8,1.0,0.5))
    ax.fill_between(X,y1=10.0,y2=Y_cn,color=(1.0,0.7,0.5))
    ax.vlines(0.20,0.0,10.0,colors='grey',linestyles='dotted',alpha=0.5)
    ax.text(0.12,2.00,'CI (C=2)',va='center',ha='left',color='k',fontsize=13)
    ax.text(0.21,5.49,'NMI (C=-2)',va='center',ha='left',color='k',fontsize=13)
    ax.text(0.11,8.50,"AFM (C=0)",va='center',ha='left',color='k',fontsize=13)
    ax.annotate('AFM (C=-2)',color='k',fontsize=13,xy=(0.19,5.45),xytext=(0.18,6.5),ha='center',arrowprops={'color':'k','arrowstyle':'->'})
    ax.scatter(0.20,4.0,color='k',s=10,zorder=4,facecolors='none')
    ax.scatter(0.20,4.6,color='k',s=10,zorder=4,facecolors='none')
    ax.scatter(0.20,5.0,color='k',s=10,zorder=4,facecolors='none')
    ax.scatter(0.20,5.75,color='k',s=10,zorder=4,facecolors='none')
    ax.scatter(0.20,5.87,color='k',s=10,zorder=4,facecolors='none')
    ax.scatter(0.20,6.0,color='k',s=10,zorder=4,facecolors='none')
    ax.set_xlim(0.0,0.3)
    ax.set_ylim(0.0,10.0)
    ax.minorticks_on()
    ax.set_xticks(np.linspace(0.0,0.3,4))
    ax.set_yticks(np.linspace(0.0,10.0,6))
    for tick in ax.get_xticklabels():
        tick.set_fontsize(14)
    for tick in ax.get_yticklabels():
        tick.set_fontsize(14)
    ax.set_xlabel("$t'/t$",fontdict={'fontsize':18})
    ax.set_ylabel("$U/t$",fontdict={'fontsize':18})
    ax.text(-0.068,9.8,'(a)',fontsize=18,ha='left',va='center',color='black')


    # h2 Chern number
    ax=axes[1]
    ax.plot([3.50,4.60,4.60,5.87,5.87,6.50],[+2.0,+2.0,-2.0,-2.0,0.0,0.0],lw=2,color='green')
    ax.axvline(x=4.600,ymin=0,ymax=1,ls='--',linewidth=1,color='black',zorder=2)
    ax.axvline(x=5.620,ymin=0,ymax=1,ls='--',linewidth=1,color='black',zorder=2)
    ax.text(4.59,1.6,'$U_M$',ha='right',fontsize=18,color='black')
    ax.text(5.59,1.6,'$U_{AF}$',ha='right',fontsize=18,color='black')
    ax.annotate(s='',xy=[3.50,0.5],xytext=[4.60,0.5],arrowprops={'arrowstyle':'<->','color':'black','linewidth':2.0,'zorder':3})
    ax.annotate(s='',xy=[4.60,0.5],xytext=[5.62,0.5],arrowprops={'arrowstyle':'<->','color':'black','linewidth':2.0,'zorder':3})
    ax.annotate(s='',xy=[5.62,0.5],xytext=[6.50,0.5],arrowprops={'arrowstyle':'<->','color':'black','linewidth':2.0,'zorder':3})
    ax.text((3.50+4.60)/2,0.51,'CI',fontsize=14,color='black',ha='center',va='bottom')
    ax.text((4.60+5.62)/2,0.51,'NMI',fontsize=14,color='black',ha='center',va='bottom')
    ax.text((5.62+6.50)/2,0.51,'AFM',fontsize=14,color='black',ha='center',va='bottom')
    ax.set_xlim(3.50,6.50)
    ax.set_ylim(-2.1,+2.1)
    ax.minorticks_on()
    ax.set_xticks(np.linspace(3.50,6.50,4))
    ax.set_yticks(np.linspace(-2.0,+2.0,3))
    for tick in ax.get_xticklabels():
        tick.set_fontsize(14)
    for tick in ax.get_yticklabels():
        tick.set_fontsize(14)
    ax.set_xlabel("$U/t$",fontdict={'fontsize':18})
    ax.set_ylabel("$C$",fontdict={'fontsize':18})
    ax.text(2.8,1.95,'(b)',fontsize=18,ha='left',va='center',color='black')

    pdb.set_trace()
    plt.savefig('h2chernnumber.pdf')
    plt.close()

def h2spectra():
    plt.ion()

    fig,axes=plt.subplots(nrows=2,ncols=6,sharex='col')
    fig.subplots_adjust(left=0.050,right=0.990,top=0.98,bottom=0.080,hspace=0.1,wspace=0.43)

    nk,ne=300,401
    Us=['4.0','4.6','5.0','5.75','5.87','6.0']
    afms=['0.0','0.0','0.0','0.0228','0.0306','0.0362']
    tags=['a','b','c','d','e','f']
    for i,(tag,U,afm) in enumerate(zip(tags,Us,afms)):
        data=np.loadtxt('../result/vca/HH_H6_H2_0.5_-1.0_0.2j_%s_%s_VCA_EB.dat'%(U,afm))
        x=data[:,0].reshape((ne,nk))
        y=data[:,1].reshape((ne,nk))
        z=data[:,2].reshape((ne,nk))
        axes[0][i].pcolormesh(x,y,z,cmap='gnuplot',rasterized=True)
        axes[0][i].axvline(x=99,ymin=0,ymax=1,ls='dotted',linewidth=1,color='red',alpha=0.5,zorder=2)
        axes[0][i].axvline(x=199,ymin=0,ymax=1,ls='dotted',linewidth=1,color='red',alpha=0.5,zorder=2)
        axes[0][i].text(210,2.8,'(%s$_1$)'%tag,fontsize=13,color='white')
        axes[0][i].set_xlim(0,299)
        axes[0][i].minorticks_on()
        axes[0][i].set_yticks([-4,-2,0,2,4])
        for tick in axes[0][i].get_yticklabels():
            tick.set_fontsize(14)
    axes[0][0].set_ylabel('Energy/t',fontsize=14)
    for i,(tag,U,afm) in enumerate(zip(tags,Us,afms)):
        data=np.loadtxt('../result/vca/HH_H6_H2_0.5_-1.0_0.2j_%s_%s_VCA_TEB.dat'%(U,afm))
        axes[1][i].plot(data[:,0],data[:,1:],color='green',linewidth=3)
        axes[1][i].axvline(x=99,ymin=0,ymax=1,ls='dotted',linewidth=1,color='red',alpha=0.5,zorder=2)
        axes[1][i].axvline(x=199,ymin=0,ymax=1,ls='dotted',linewidth=1,color='red',alpha=0.5,zorder=2)
        axes[1][i].text(210,data[:,1:].max()*0.85,'(%s$_2$)'%tag,va='center',fontsize=13,color='black')
        axes[1][i].set_xlim(0,299)
        axes[1][i].minorticks_on()
        axes[1][i].set_xticks([0,99,199,299])
        axes[1][i].set_yticks([-4,-2,0,2,4] if i in (0,1,2) else [-10,-5,0,5,10] if i==3 else [-10000,-5000,0,5000,10000] if i==4 else [-20,-10,0,10,20])
        for tick in axes[1][i].get_yticklabels():
            tick.set_fontsize(10 if i==4 else 14)
        axes[1][i].set_xticklabels(["$\Gamma$","$K$","$M$","$\Gamma$"],fontdict={'fontsize':15})
    axes[1][0].set_ylabel('Energy/t',fontsize=14)

    pdb.set_trace()
    plt.savefig('h2spectra.pdf')
    plt.close()

def edge():
    plt.ion()

    fig,axes=plt.subplots(nrows=2,ncols=6,sharex='col')
    fig.subplots_adjust(left=0.045,right=0.990,top=0.98,bottom=0.060,hspace=0.1,wspace=0.20)

    nk,ne=200,400
    Us=['4.0','4.6','5.0','5.75','5.87','6.0']
    afms=['0.0','0.0','0.0','0.0228','0.0306','0.0362']
    tags=['a','b','c','d','e','f']
    for i,(tag,U,afm) in enumerate(zip(tags,Us,afms)):
        data=np.loadtxt('../result/vca/HH_H6^1P15O_0.5_-1.0_0.2j_%s_%s_VCACCT_EB.dat'%(U,afm))
        x=data[:,0].reshape((ne,nk))
        y=data[:,1].reshape((ne,nk))
        z=data[:,2].reshape((ne,nk))
        axes[0][i].pcolormesh(x,y,z,cmap='gnuplot',rasterized=True)
        axes[0][i].text(145,2.8,'(%s$_1$)'%tag,fontsize=15,color='white')
        axes[0][i].set_xlim(0,199)
        axes[0][i].set_ylim(-4.0,4.0)
        axes[0][i].minorticks_on()
        axes[0][i].set_yticks([-4,-2,0,2,4])
        for tick in axes[0][i].get_yticklabels():
            tick.set_fontsize(14)
    axes[0][0].set_ylabel('Energy/t',fontsize=14)
    for i,(tag,U,afm) in enumerate(zip(tags,Us,afms)):
        data=np.loadtxt('../result/vca/HH_H6^1P15O_0.5_-1.0_0.2j_%s_%s_VCACCT_TEB.dat'%(U,afm))
        axes[1][i].plot(data[:,0],data[:,1:],color='green',linewidth=2)
        axes[1][i].text(135,5*0.7,'(%s$_2$)'%tag,fontsize=14,color='black')
        axes[0][i].set_xlim(0,199)
        axes[1][i].set_ylim(-5,5)
        axes[1][i].minorticks_on()
        axes[1][i].set_yticks([-5,-3,-1,1,3,5])
        for tick in axes[1][i].get_yticklabels():
            tick.set_fontsize(14)
        axes[1][i].set_xticks([0,99,199])
        axes[1][i].set_xticklabels(["$-\pi$","$0$","$\pi$"],fontsize=14)
    axes[1][0].set_ylabel('Energy/t',fontsize=14)

    pdb.set_trace()
    plt.savefig('edge.pdf')
    plt.close()

def h6chernnumber():
    plt.ion()

    fig,axes=plt.subplots(nrows=1,ncols=2)
    fig.subplots_adjust(left=0.090,right=0.980,top=0.960,bottom=0.145,wspace=0.250)

    # h6 phase diagram
    ax=axes[0]
    t=np.array([0.00,0.05,0.10,0.15,0.20,0.25,0.30])
    u_gap=np.array([0.00,2.80,3.60,4.10,4.60,5.00,5.40])
    u_afm=np.array([3.84,4.02,4.45,4.98,5.62,6.49,8.00])
    X=np.linspace(t.min(),t.max(),201)
    Y_gap=itp.splev(X,itp.splrep(t,u_gap,k=2),der=0)
    Y_afm=itp.splev(X,itp.splrep(t,u_afm,k=2),der=0)
    ax.plot(X,Y_gap,'blue',lw=2.0)
    ax.plot(X,Y_afm,'red',lw=2.0)
    ax.fill_between(X,y1=Y_gap,y2=0.0,color=(0.8,0.5,1.0))
    ax.fill_between(X,y1=Y_gap,y2=10.0,color=(1.0,0.7,0.5))
    ax.vlines(0.20,0.0,10.0,colors='grey',linestyles='dotted',alpha=0.5)
    ax.text(0.12,2.00,'CI (C=2)',va='center',ha='left',color='k',fontsize=13)
    ax.text(0.21,5.49,'NMI (C=0)',va='center',ha='left',color='k',fontsize=13)
    ax.text(0.11,8.50,"AFM (C=0)",va='center',ha='left',color='k',fontsize=13)
    ax.scatter(0.20,4.0,color='k',s=10,zorder=4,facecolors='none')
    ax.scatter(0.20,4.6,color='k',s=10,zorder=4,facecolors='none')
    ax.scatter(0.20,5.0,color='k',s=10,zorder=4,facecolors='none')
    ax.scatter(0.20,5.75,color='k',s=10,zorder=4,facecolors='none')
    ax.scatter(0.20,5.87,color='k',s=10,zorder=4,facecolors='none')
    ax.scatter(0.20,6.0,color='k',s=10,zorder=4,facecolors='none')
    ax.set_xlim(0.0,0.3)
    ax.set_ylim(0.0,10.0)
    ax.minorticks_on()
    ax.set_xticks(np.linspace(0.0,0.3,4))
    ax.set_yticks(np.linspace(0.0,10.0,6))
    for tick in ax.get_xticklabels():
        tick.set_fontsize(14)
    for tick in ax.get_yticklabels():
        tick.set_fontsize(14)
    ax.set_xlabel("$t'/t$",fontdict={'fontsize':18})
    ax.set_ylabel("$U/t$",fontdict={'fontsize':18})
    ax.text(-0.068,9.8,'(a)',fontsize=18,ha='left',va='center',color='black')


    # h6 Chern number
    ax=axes[1]
    ax.plot([3.50,4.60,4.60,6.50],[+2.0,+2.0,0.0,0.0],lw=2,color='green')
    ax.axvline(x=4.600,ymin=0,ymax=1,ls='--',linewidth=1,color='black',zorder=2)
    ax.axvline(x=5.620,ymin=0,ymax=1,ls='--',linewidth=1,color='black',zorder=2)
    ax.text(4.59,1.6,'$U_M$',ha='right',fontsize=18,color='black')
    ax.text(5.59,1.6,'$U_{AF}$',ha='right',fontsize=18,color='black')
    ax.annotate(s='',xy=[3.50,-1.0],xytext=[4.60,-1.0],arrowprops={'arrowstyle':'<->','color':'black','linewidth':2.0,'zorder':3})
    ax.annotate(s='',xy=[4.60,-1.0],xytext=[5.62,-1.0],arrowprops={'arrowstyle':'<->','color':'black','linewidth':2.0,'zorder':3})
    ax.annotate(s='',xy=[5.62,-1.0],xytext=[6.50,-1.0],arrowprops={'arrowstyle':'<->','color':'black','linewidth':2.0,'zorder':3})
    ax.text((3.50+4.60)/2,-0.99,'CI',fontsize=14,color='black',ha='center',va='bottom')
    ax.text((4.60+5.62)/2,-0.99,'NMI',fontsize=14,color='black',ha='center',va='bottom')
    ax.text((5.62+6.50)/2,-0.99,'AFM',fontsize=14,color='black',ha='center',va='bottom')
    ax.set_xlim(3.50,6.50)
    ax.set_ylim(-2.1,+2.1)
    ax.minorticks_on()
    ax.set_xticks(np.linspace(3.50,6.50,4))
    ax.set_yticks(np.linspace(-2.0,+2.0,3))
    for tick in ax.get_xticklabels():
        tick.set_fontsize(14)
    for tick in ax.get_yticklabels():
        tick.set_fontsize(14)
    ax.set_xlabel("$U/t$",fontdict={'fontsize':18})
    ax.set_ylabel("$C$",fontdict={'fontsize':18})
    ax.text(2.8,1.95,'(b)',fontsize=18,ha='left',va='center',color='black')

    pdb.set_trace()
    plt.savefig('h6chernnumber.pdf')
    plt.close()

def h6spectra():
    plt.ion()

    fig,axes=plt.subplots(nrows=2,ncols=6,sharex='col')
    fig.subplots_adjust(left=0.050,right=0.990,top=0.98,bottom=0.080,hspace=0.1,wspace=0.230)

    nk,ne=300,401
    Us=['4.0','4.6','5.0','5.75','5.87','6.0']
    afms=['0.0','0.0','0.0','0.0228','0.0306','0.0362']
    tags=['a','b','c','d','e','f']
    for i,(tag,U,afm) in enumerate(zip(tags,Us,afms)):
        data=np.loadtxt('../result/vca/HH_H6_H6_0.5_-1.0_0.2j_%s_%s_VCA_EB.dat'%(U,afm))
        x=data[:,0].reshape((ne,nk))
        y=data[:,1].reshape((ne,nk))
        z=data[:,2].reshape((ne,nk))
        axes[0][i].pcolormesh(x,y,z,cmap='gnuplot',rasterized=True)
        axes[0][i].axvline(x=99,ymin=0,ymax=1,ls='dotted',linewidth=1,color='red',alpha=0.5,zorder=2)
        axes[0][i].axvline(x=199,ymin=0,ymax=1,ls='dotted',linewidth=1,color='red',alpha=0.5,zorder=2)
        axes[0][i].text(210,2.8,'(%s$_1$)'%tag,fontsize=13,color='white')
        axes[0][i].set_xlim(0,299)
        axes[0][i].minorticks_on()
        axes[0][i].set_yticks([-4,-2,0,2,4])
        for tick in axes[0][i].get_yticklabels():
            tick.set_fontsize(14)
    axes[0][0].set_ylabel('Energy/t',fontsize=14)
    for i,(tag,U,afm) in enumerate(zip(tags,Us,afms)):
        data=np.loadtxt('../result/vca/HH_H6_H6_0.5_-1.0_0.2j_%s_%s_VCA_TEB.dat'%(U,afm))
        axes[1][i].plot(data[:,0],data[:,1:],color='green',linewidth=3)
        axes[1][i].axvline(x=99,ymin=0,ymax=1,ls='dotted',linewidth=1,color='red',alpha=0.5,zorder=2)
        axes[1][i].axvline(x=199,ymin=0,ymax=1,ls='dotted',linewidth=1,color='red',alpha=0.5,zorder=2)
        axes[1][i].text(210,data[:,1:].max()*0.85,'(%s$_2$)'%tag,va='center',fontsize=13,color='black')
        axes[1][i].set_xlim(0,299)
        axes[1][i].minorticks_on()
        axes[1][i].set_xticks([0,99,199,299])
        axes[1][i].set_yticks([-4,-2,0,2,4] if i in (0,1,2) else [-6,-3,0,3,6])
        for tick in axes[1][i].get_yticklabels():
            tick.set_fontsize(14)
        axes[1][i].set_xticklabels(["$\Gamma$","$K_R$","$M_R$","$\Gamma$"],fontdict={'fontsize':15})
    axes[1][0].set_ylabel('Energy/t',fontsize=14)

    pdb.set_trace()
    plt.savefig('h6spectra.pdf')
    plt.close()

if __name__=='__main__':
    import sys
    for arg in sys.argv:
        if arg in ('1','all'): lattice()
        if arg in ('2','all'): phase()
        if arg in ('3','all'): h2chernnumber()
        if arg in ('4','all'): h2spectra()
        if arg in ('5','all'): edge()
        if arg in ('6','all'): h6chernnumber()
        if arg in ('7','all'): h6spectra()
