from __future__ import division
import numpy as np
import scipy.constants as sconst
import matplotlib.pyplot as plt
import pickle
import math
from scipy import optimize
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm



def rot_matrix(u=np.array([1, 0, 0]), angle = 0):
    ux = u[0]
    uy = u[1]
    uz = u[2]
    co = np.cos(angle)
    si = np.sin(angle)
    rm = np.array([[co+ux*ux*(1-co), ux*uy*(1-co)-uz*si, ux*uz*(1-co)+uy*si],[uy*ux*(1-co)+uz*si, co+uy*uy*(1-co), uy*uz*(1-co)-ux*si],[uz*ux*(1-co)-uy*si, uz*uy*(1-co)+ux*si, co+uz*uz*(1-co)]])
    return rm


#Constants, in MHz

gNV = -2.0023
muNV = 9.2740154e-24
betaNV = 2.8
zfsNV = 2.8703e3
zfsNVes = 1.42e3
zfs14N=-4.941
beta14N=0.00076
A_NVN=-2.172
A_T = -2.630
mu0 = 4*np.pi*1e-7
r3d=0.02
r2e=1           #r2e=50*r3d, otherwise arbitrary
#hyperfine=False

#Spin-Operators

Sx = 1/np.sqrt(2)*np.array([[0, 1+0.j, 0], [1, 0, 1], [0, 1, 0]])
Sy = 1j/np.sqrt(2)*np.array([[0, -1+0.j, 0],[1, 0, -1],[0, 1, 0]])
Sz = np.array([[1+0.j, 0, 0],[0, 0, 0],[0, 0, -1]])

#Transformation of operators into the used Hilbert-space


def add_I(rho):
    """
    extend operator for nuclear spin subspace (N14)
    """
    rho_new=np.kron(rho, np.eye(3))
    return rho_new

def add_S(rho):
    """
    extend operator for electron spin subspace
    """
    rho_new=np.kron(np.eye(3), rho)
    return rho_new

def delete_I(rho):
    """
    trace out nuclear spin
    """
    new_rho=np.zeros((3, 3), dtype=complex)
    for ix in range(3):
        for iy in range(3):
            new_rho[ix, iy]=np.trace(rho[ix*3:ix*3+3, iy*3:iy*3+3])
    return new_rho
            

# partial Hamiltonians



def total(x, y, z):
    """
    Help function
    """
    tot=np.sqrt(x**2+y**2+z**2)
    return tot

def transform_field(bx, by, bz):
    """
    transforms magnetic field from NV1 system in system of NV 2
    """
    angle = 109.5*np.pi/180
    bx2 = bx*np.cos(angle)-bz*np.sin(angle)
    by2 = by
    bz2 = bz*np.cos(angle)+bx*np.sin(angle)
    return bx2, by2, bz2


lorentz=lambda x, x0, w, a: 2*a/np.pi*w/(4*(x-x0)**2+w**2)



#*******************************
#***   Start of the Program  ***
#*******************************

class System():
    def __init__(self, hyperfine = False):
        """
        b_tot=0, e_tot=0
        """
        self.hyperfine=hyperfine
        self.bx=0.
        self.by=0.
        self.bz=0.
        self.ex=0.
        self.ey=0.
        self.ez=0.
        self.d=zfsNV
        self.d_es=zfsNVes
        self.b_tot=total(self.bx, self.by, self.bz)
        self.e_tot=total(self.ex, self.ey, self.ez)
        self.hamiltonians=[self.H_zee_NV, self.H_zfs_NV, self.H_stark, self.H_zfs_N14, self.H_zee_N14, self.H_hfs] #, self.H_T]
        self.hamiltonians_es=[self.H_zfs_es, self.H_zee_NV]
        self.H=self.H_total()
        self.H_es = self.H_excited()
        self.get_energy_es()
        self.get_energy()
        self.n_states=len(self.energy)
        self.show_es = False

    def H_zee_NV(self):
        """
        NV Zeeman Hamiltonian
        """
        h_zee=-betaNV*(self.bx*Sx+self.by*Sy+self.bz*Sz)
        if self.hyperfine:
            return add_I(h_zee)
        else:
            return h_zee

    def H_zfs_NV(self):
        """
        NV ZFS GS 
        """
        h_zfs=self.d*np.dot(Sz, Sz)
        if self.hyperfine:
            return add_I(h_zfs)
        else:
            return h_zfs

    def H_zfs_es(self):
        """
        NV ZFS ES
        """
        h_zfs = self.d_es*np.dot(Sz, Sz)
        if self.hyperfine:
            return add_I(h_zfs)
        else:
            return h_zfs

    def H_stark(self):
        """
        NV GS Stark-Shift
        """
        h_stark=r3d*self.ez*(np.dot(Sz, Sz)-2/3*np.eye(3))-r2e*(self.ey*(np.dot(Sx, Sy)+np.dot(Sy, Sx))+self.ex*(np.dot(Sx, Sx)-np.dot(Sy, Sy)))
        if self.hyperfine:
            return add_I(h_stark)
        else:
            return h_stark

    def H_zfs_N14(self):
        """
        N14 ZFS
        """
        h_zfs=zfs14N*np.dot(Sz, Sz)
        if self.hyperfine:
            return add_S(h_zfs)
        else:
            return 0

    def H_zee_N14(self):
        """
        N14 Zeeman shift
        """
        h_zee=-beta14N*(self.bx*Sx+self.by*Sy+self.bz*Sz)
        if self.hyperfine:
            return add_S(h_zee)
        else:
            return 0

    def H_hfs(self):
        """
        N14-NV contact interaction Hfs
        """
        h_hfs=A_NVN*(np.dot(add_I(Sz), add_S(Sz)))
        if self.hyperfine:
            return h_hfs
        else:
            return 0

    def H_T(self):
        h_hfs=A_T*(np.dot(add_I(Sx), add_S(Sx))+np.dot(add_I(Sy), add_S(Sy)))
        if self.hyperfine:
            return h_hfs
        else:
            return 0

    def H_total(self):
        """
        NV ZFS, NV Zee, NV stark, N14 ZFS, N14 Zee, NV-N14 HFS
        """
        h_tot={True:np.zeros((9, 9), dtype='complex128'), False:np.zeros((3, 3), dtype='complex128')}[self.hyperfine]
        for ham in self.hamiltonians:
            h_tot+=ham()
        return h_tot

    def H_excited(self):
        """
        NV ES Hamiltonian, only Zfs + Zee
        """
        h_tot={True:np.zeros((9, 9), dtype='complex128'), False:np.zeros((3, 3), dtype='complex128')}[self.hyperfine]
        for ham in self.hamiltonians_es:
            h_tot+=ham()
        return h_tot

    #---------------------
    #--- set Parameter ---
    #---------------------

    def delete_h(self, ham, es=False):
        if es:
            try:
                self.hamiltonians_es.remove(ham)
            except:
                print 'not in'
            finally:
                self.H_es=self.H_excited()
        else:
            try:
                self.hamiltonians.remove(ham)
            except:
                print 'not in'
            finally:
                self.H=self.H_total()
        self.get_energy()

    def set_bx(self, new_bx):
        """
        calculates new energy/states for new B_x
        """
        self.bx=new_bx
        self.b_tot=total(self.bx, self.by, self.bz)
        self.H=self.H_total()
        self.H_es = self.H_excited()
        self.get_energy()

    def set_by(self, new_by):
        """
        calculates new energy/states for new B_y
        """
        self.by=new_by
        self.b_tot=total(self.bx, self.by, self.bz)
        self.H=self.H_total()
        self.H_es = self.H_excited()
        self.get_energy()
        self.get_energy_es()

    def set_bz(self, new_bz):
        """
        calculates new energy/states for new B_z
        """
        self.bz=new_bz
        self.b_tot=total(self.bx, self.by, self.bz)
        self.H=self.H_total()
        self.H_es = self.H_excited()
        self.get_energy()
        self.get_energy_es()

    def set_ex(self, new_ex):
        """
        calculates new energy/states for new E_x
        """
        self.ex=new_ex
        self.e_tot=total(self.ex, self.ey, self.ez)
        self.H=self.H_total()
        self.get_energy()
        
    def set_ey(self, new_ey):
        """
        calculates new energy/states for new E_y
        """
        self.ey=new_ey
        self.e_tot=total(self.ex, self.ey, self.ez)
        self.H=self.H_total()
        self.get_energy()

    def set_ez(self, new_ez):
        """
        calculates new energy/states for new E_z
        """
        self.ez=new_ez
        self.e_tot=total(self.ex, self.ey, self.ez)
        self.H=self.H_total()
        self.get_energy()

    def set_ephi(self, new_er, new_ephi):
        """
        sets E_x and E_y in polar coordinates
        new_ephi in degree
        """
        ephi=new_ephi*np.pi/180
        self.set_ex(new_er*np.cos(ephi))
        self.set_ey(new_er*np.sin(ephi))

    def set_bphi(self, new_br, new_bphi):
        """
        sets B_x and B_y in polar coordinates
        new_bphi in degree
        leaves B_z untouched
        """
        bphi=new_bphi*np.pi/180
        self.set_bx(new_br*np.cos(bphi))
        self.set_by(new_br*np.sin(bphi))

    def set_theta(self, new_btot, new_theta):
        """
        changes Bx, By, Bz but leaves B_phi same
        (CAUTION: angle not yet really kept)
        new_theta in degree, starting from pole
        """
        btheta = new_theta*np.pi/180
        bphi = np.arccos(self.bx/self.b_tot)
        self.set_bz(new_btot*np.sin(btheta))
        br = new_btot*np.cos(btheta)
        self.set_bx(br*np.cos(br))
        self.set_by(br*np.sin(br))

    def set_all_zero(self):
        self.ex=0
        self.ey=0
        self.ez=0
        self.bx=0
        self.by=0
        self.bz=0
        self.b_tot=total(self.bx, self.by, self.bz)
        self.e_tot=0.
        self.H=self.H_total()
        self.H_es = self.H_excited()
        self.get_energy()
        self.get_energy_es()

    def set_b_all(self, new_bx, new_by, new_bz):
        self.bx=new_bx
        self.by=new_by
        self.bz=new_bz
        self.b_tot=total(self.bx, self.by, self.bz)
        self.H=self.H_total()
        self.H_es = self.H_excited()
        self.get_energy()
        self.get_energy_es()

    

    #------------------------
    #--- Hamiltonian eval ---
    #------------------------

    def get_pairs(self):
        """
        creates pair_dic, dic where for every combination of states
        (gs) the transition matrix elements and energy difference
        is given
        pair_dic[(i, j)]=[<i|Sx|j>, <i|Sy|j>, sqrt(<Sx>^2+<Sy>^2), E_j-E_i]
        """
        self.pair_dic={}
        pair_list=[]
        for i in range(int(np.ceil(self.n_states/2))):
            for k in range(i+1, self.n_states):
                pair_list.append([i, k])
        Sx_trans=self.get_Sx_trans()
        Sy_trans=self.get_Sy_trans()
        for pair in pair_list:
            Sx_t=Sx_trans[pair[0], pair[1]]
            Sy_t=Sy_trans[pair[0], pair[1]]
            self.pair_dic[tuple(pair)]=[Sx_t, Sy_t, np.sqrt(Sx_t**2+Sy_t**2), self.energy[pair[1]]-self.energy[pair[0]]]


    def get_energy(self):
        """
        gets eigenvalues of Hamiltonian
        puts them in list self.energy
        number of states: self.n_states
        self.state_i = Eigenstates still complex
        self.state_a = Eigenstates with absolute value of elements - gives basis
        """
        energy, self.state=np.linalg.eigh(self.H)
        el=list(energy)
        self.energy=[float(el) for el in el]
        self.state_i=self.state.conj().T
        self.state_a=np.absolute(self.state_i)
        self.n_states=len(self.energy)
        self.get_Sx_trans()
        self.get_Sy_trans()
        self.get_Sr_trans()
        self.get_frequency()

    def get_energy_es(self):
        """
        gives energy, eigenstates of excited state,
        self.energy_es is sorted starting from the lowest
        -> eigenstates don't necessarily match energies 
        """
        energy, self.state_es = np.linalg.eigh(self.H_es)
        el = list(energy)
        self.energy_es = sorted([float(el) for el in el])
        
    def get_frequency(self):
        """
        creates matrix where every element is the
        absolute value of the energy difference
        between the energies of the resp. states
        same with excited state
        """
        freq=np.zeros((self.n_states, self.n_states), dtype=float)
        for i in range(self.n_states):
            for k in range(self.n_states):
                freq[i, k]=np.absolute(self.energy[i]-self.energy[k])
        self.frequency=freq
        freq_es = np.zeros((3, 3), dtype=float)
        for i in range(3):
            for k in range(3):
                freq_es[i, k]=np.absolute(self.energy_es[i]-self.energy[k])
        self.frequency_es = freq_es

    def get_Sx_trans(self):
        """
        calculates Sx in new Basis
        """
        sd=Sx
        if self.hyperfine: sd=add_I(Sx)
        s=np.dot(np.dot(self.state_i, sd), self.state)
        #s=np.absolute(s)
        self.Sx_trans=s

    def get_Sy_trans(self):
        """
        calculates Sy in new Basis
        """
        sd=Sy
        if self.hyperfine: sd=add_I(Sy)
        s=np.dot(np.dot(self.state_i, sd), self.state)
        #s=np.absolute(s)
        self.Sy_trans=s

    def get_Sr_trans(self):
        """
        calculates Sr in new Basis
        with Sr = sqrt(Sx^2+Sy^2)
        """
        s=np.zeros((self.n_states, self.n_states), dtype=complex)
        for i in range(self.n_states):
            for k in range(self.n_states):
                s[i, k]=np.sqrt(self.Sx_trans[i, k]**2+self.Sy_trans[i, k]**2)
        self.Sr_trans=s

    def get_S_phi(self, phi):
        """
        calculates S_phi in new Basis
        with S_phi given by get_intensity()
        """
        s=np.zeros((self.n_states, self.n_states), dtype=float)
        for i in range(self.n_states):
            for k in range(self.n_states):
                s[i, k]=self.get_intensity(phi, i, k)
        return s

    def get_transitions(self, n=6, phi=0):
        """
        gets the first n transition frequencies sorted by the coupling to MW of the resp. states
        f_(ik) sorted by (<i|Sx|k>)^2+(<i|Sy|k>)^2
        phi: angle of MW polarization in xy-plane, if 0: see upper eq.
        """
        if type(n)==int:
            pair_list=[]
            for i in range(self.n_states-1):
                for k in range(i+1, self.n_states):
                    pair_list.append([i, k])
        else:
            pair_list=n
            n=len(pair_list)
        if phi==0:
            intensity=self.Sr_trans
        else:
            intensity=self.get_S_phi(phi)
        pair_list=sorted(pair_list, key=lambda pair: -intensity[pair[0], pair[1]])
        pair_list=pair_list[:n]
        transitions=[self.frequency[pair[1], pair[0]] for pair in pair_list]
        transitions.sort()
        return transitions

    def get_transitions_es(self):
        """
        at the moment only works when b_tot<500
        """
        if self.b_tot<500:
            t1 = self.energy_es[1]-self.energy_es[0]
            t2 = self.energy_es[2]-self.energy_es[0]
        else:
            t1 = 0
            t2 = 0
        return [t1, t2]

    def get_transition_pairs(self, n=6):
        """
        gets state pairs with biggest coupling on MW
        sorted after their transition frequency
        """
        pair_list=[]
        for i in range(self.n_states-1):
            for k in range(i+1, self.n_states):
                pair_list.append([i, k])
        pair_list=sorted(pair_list, key=lambda pair: -self.Sr_trans[pair[0], pair[1]])
        pair_list=pair_list[:n]
        pair_list=sorted(pair_list, key=lambda pair: -self.frequency[pair[0], pair[1]])
        return pair_list
    
    def get_transitions_old(self, threshold=0.01):
        """
        no idea
        """
        sx=add_I(Sx)
        si1=np.dot(np.dot(self.state_i, sx), self.state)
        sil=si1.copy()
        sil=list(sil)
        trans=[]
        for i in range(len(sil)):
            sli=sil[i]
            for p in range(i+1, len(sil)):
                if sli[p]>threshold:
                    trans.append((np.absolute(self.energy[p]-self.energy[i])))
        sd=add_I(Sy)
        si1=np.dot(np.dot(self.state_i, sd), self.state)
        sil=si1.copy()
        sil=list(sil)
        for i in range(len(sil)):
            sli=sil[i]
            for p in range(i+1, len(sil)):
                if sli[p]>threshold:
                    trans.append((np.absolute(self.energy[p]-self.energy[i])))
        trans.sort()
        return trans

    def sort_t(self, pairs):
        """
        sort pairs for their transition frequency
        """
        new=sorted(pairs, key=lambda pair: np.absolute(self.energy[pair[0]]-self.energy[pair[1]]))
        return new

   
    def get_intensity(self, phi=0, state1=0, state2=1):
        """
        gives value of transition matrix element for rotation
        around phi in the xy-plane between states state1 and state2
        returns <i|Sr|k><k|Sr|i>
        """
        sd=add_I(np.cos(phi)*Sx+np.sin(phi)*Sy)
        si1=np.dot(np.dot(self.state_i, sd), self.state)
        si2=np.dot(np.dot(self.state, sd), self.state_i)
        trans=si1[state1, state2]*si1[state2, state1]
        return abs(trans)

    #-------------
    #--- Plots ---
    #-------------

    def plot_energylevels(self):
        """
        plots level scheme
        """
        for i in range(len(self.energy)):
            plt.plot([0, 1], [self.energy[i], self.energy[i]])
        if self.show_es:
            for i in range(3):
                plt.plot([0, 1], [self.energy_es[i], self.energy_es[i]])
        plt.show()

    def show_state(self):
        """
        Matrix plot how new Eigenbasis translates to old basis
        """
        fig=plt.figure()
        ax=fig.add_subplot(111)
        ax.imshow(self.state_a, interpolation="none")
        ax.set_yticks(range(9))
        ax.set_xticks(range(9))
        ax.set_yticklabels([str(i) for i in range(9)])
        ax.set_xticklabels(["11", "10", "1-1", "01", "00", "0-1", "-11", "-10", "-1-1"])
        ax.set_ylabel("New Eigenbasis")
        ax.set_xlabel("Old states (|m_s, m_i>)")
        plt.show()
        
    def get_info(self):
        """
        all applied fields
        """
        print "Ex: ", self.ex, ", Ey: ", self.ey, ", Ez: ", self.ez, ", Bx: ", self.bx, ", By: ", self.by, ", Bz: ", self.bz, ", B_tot: ", self.b_tot, ", E_tot: ", self.e_tot


    def plot_total(self, x_axis="bz", start=0, stop=1, step=0.01, pol=False, diff1=False, diff2=False, what = 1, give = False, plot = True, n=2, phi = 0, rot_phi=0, rot_theta=0):
        """
        start, stop, step: in units of the resp. fields, for rotation in numbers of 2pi 
        what: 1 = energy, 2 = transition, diff1: difference to starting energy/transition
        diff2: difference between upper/lower transition(?)
        give: returns x and y
        plot: if it should be plotted
        n: number of transitions shown
        phi: microwave polarization, 0=S_r
        rot_phi, rot_theta: angle for rot. axis (only for brot) in degree
        possible axes:  bx, by, bz, ex, ey, ez
                        ephi, bphi, bephi: rotation of resp. field(s) in xy plane
                        bcontrae: rotation of B-field in xy plane and simultaneous rot.
                                    of E-Field in xy plane in contrary direction
                        btheta: rotation of B-Field around 
                        brot: rotation of B-Field around axis given by rot_phi, rot_theta
        """
        x=np.arange(start, stop, step)
        y=[]
        y_es = []
        if x_axis in ["ephi", "bphi", "bephi", "bcontrae", "btheta", "brot"]:
            x = 2*x*np.pi
        initial_values = [self.bx, self.by, self.bz, self.ex, self.ey, self.ez]
        bx_list = self.bx*np.ones((len(x)), dtype = float)
        by_list = self.by*np.ones((len(x)), dtype = float)
        bz_list = self.bz*np.ones((len(x)), dtype = float)
        ex_list = self.ex*np.ones((len(x)), dtype = float)
        ey_list = self.ey*np.ones((len(x)), dtype = float)
        ez_list = self.ez*np.ones((len(x)), dtype = float)
        e_tot = self.e_tot
        b_tot = self.b_tot
        b_r = np.sqrt(self.bx**2+self.by**2)
        if b_r!=0:
            btheta = np.arctan(self.bz / b_r)
        else:
            btheta = 0
        if b_tot != 0:
            bphi = np.arccos(self.bx/self.b_tot)
        else:
            bphi = 0
        if x_axis=="bx":
            bx_list = x
        elif x_axis=="by":
            by_list = x
        elif x_axis=="bz":
            bz_list = x
        elif x_axis=="ex":
            ex_list = x
        elif x_axis=="ey":
            ey_list = x
        elif x_axis=="ez":
            ez_list = x
        elif x_axis=="ephi":
            ex_list = [np.cos(i)*e_tot for i in x]
            ey_list = [np.sin(i)*e_tot for i in x]
        elif x_axis=="bphi":
            bx_list = [np.cos(i)*b_tot for i in x]
            by_list = [np.sin(i)*b_tot for i in x]
        elif x_axis=="bephi":
            bx_list = [np.cos(i)*b_tot for i in x]
            by_list = [np.sin(i)*b_tot for i in x]
            ex_list = [np.cos(i)*e_tot for i in x]
            ey_list = [np.sin(i)*e_tot for i in x]
        elif x_axis=="bcontrae":
            bx_list = [np.cos(i)*b_tot for i in x]
            by_list = [np.sin(i)*b_tot for i in x]
            ex_list = [np.cos(-i)*e_tot for i in x]
            ey_list = [np.sin(-i)*e_tot for i in x]
        elif x_axis == 'btheta':
            bz_list = [i*np.cos(btheta) for i in x]
            bx_list = [np.cos(bphi)*i*np.sin(btheta) for i in x]
            by_list = [np.sin(bphi)*i*np.sin(btheta) for i in x]
        elif x_axis == 'brot':
            phi0 = rot_phi*np.pi / 180
            theta0 = rot_theta*np.pi / 180
            u = np.array([np.cos(phi0)*np.sin(theta0), np.sin(phi0)*np.sin(theta0), np.cos(theta0)])
            if theta0 == 0:
                v_start=np.array([1, 0, 0])
            else:
                v_start = np.array([u[1], -u[0], 0])/np.sqrt(u[1]**2+u[0]**2)
            b_new = np.array([b_tot*np.dot(rot_matrix(u, i), v_start) for i in x])
            bx_list = b_new[:,0]
            by_list = b_new[:,1]
            bz_list = b_new[:,2]

        for i in range(len(x)):
            self.set_bx(bx_list[i])
            self.set_by(by_list[i])
            self.set_bz(bz_list[i])
            energy1 = self.energy
            self.set_ex(ex_list[i])
            self.set_ey(ey_list[i])
            self.set_ez(ez_list[i])
            if what == 1:
                if diff1:
                    y.append(list(np.abs(np.array(self.energy)-np.array(energy1))))
                else:
                    y.append(self.energy)
                y_es.append(self.energy_es)
            elif what == 2:
                trans = self.get_transitions(n, phi)
                y.append(trans)
                trans_es = self.get_transitions_es()
                y_es.append(trans_es)
        self.set_bx(initial_values[0])
        self.set_by(initial_values[1])
        self.set_bz(initial_values[2])
        self.set_ex(initial_values[3])
        self.set_ey(initial_values[4])
        self.set_ez(initial_values[5])
        if plot:
            fig=plt.figure()
            ax=fig.add_subplot(111, polar=pol)
            y=np.array(y)
            y_es = np.array(y_es)
            if pol:
                if diff2:
                    y=y[:,1]-y[:,2]
                    y=y-y.min()
                    ax.plot(x, y)
                else:
                    for i in [1, 2]:
                        ax.plot(x, y[:, i])
            else:
                for i in range(len(self.energy)):
                    ax.plot(x, y)
                if len(y_es)!=0 and self.show_es:
                    ax.plot(x, y_es)
            if what==1:
                ax.set_ylabel("Energy in MHz")
            elif what==2:
                ax.set_ylabel('Frequency in MHz')
            ax.set_xlabel(x_axis)
            plt.show()
        if give:
            y = np.array(y)
            if len(y_es)!=0 and self.show_es:
                y_es = np.array(y_es)
                return x, y, y_es
            else:
                return x, y

    def show_intensities(self):
        fig=plt.figure()
        ax=fig.add_subplot(111)
        ax.imshow(self.Sr_trans, interpolation="none")
        ax.set_yticks(range(9))
        ax.set_xticks(range(9))
        ax.set_yticklabels(self.energy)
        ax.set_xticklabels(range(9))
        ax.set_title("<i|S_r|j>")
        ax.set_ylabel("Energy of state in MHz")
        ax.set_xlabel("state")
        plt.show()

    def get_intensity_list(self, state1, state2):
        phi=np.arange(0, 2*np.pi+0.005, 0.01)
        intensity=[]
        for p in phi:
            trans=self.get_intensity(p, state1, state2)
            intensity.append(trans)
        return phi, intensity

    def plot_intensity(self, state1, state2):
        phi, intensity=self.get_intensity_list(state1, state2)
        plt.plot(phi, intensity)

    def give_intensity(self, state1, state2):
        phi, intensity=self.get_intensity_list(state1, state2)
        return phi, intensity
    
    def plot_all_intensities(self, pairs=6):
        if type(pairs)==int:
            pairs=self.get_transition_pairs(n=pairs)
        fig=plt.figure()
        ax=fig.add_subplot(111)
        for pair in pairs:
            p, intensity=self.get_intensity_list(pair[0], pair[1])
            ax.plot(p, intensity)
        ax.set_xticks(np.arange(0, 2*np.pi+0.1, np.pi/4))
        ax.set_xticklabels(["0", "1/4", "1/2", "3/4", "1", "5/4", "3/2", "7/4", "2"])
        ax.legend(tuple([str(pair) for pair in pairs]), "lower right")
        ax.set_xlabel("Microwave polarization (in pi)")
        ax.set_ylabel("Transition probability")
        plt.show()

    def get_transition_intens(self, pol='x'):
        transition={'x':self.Sx_trans, 'y':self.Sy_trans}[pol]
        freq=self.frequency.flatten()
        intens=transition.flatten()
        return freq, intens

    def plot_transition_intens(self, pol='x', c='k', factor=1.):
        freq, intens=self.get_transition_intens(pol)
        for i, fr in enumerate(freq):
            if fr>1e3:
                plt.plot([fr, fr], [0, intens[i]*factor], c)
                

    def odmr(self, lw=0.1, phi=0., freq1=None, freq2=None, steps=1000):
        num_t=sum(range(len(self.energy)+1))
        freq=self.frequency.flatten()
        intens_x=self.Sx_trans.flatten()
        intens_x=intens_x*abs(np.cos(phi*np.pi/180))
        intens_y=self.Sy_trans.flatten()
        intens_y=intens_y*abs(np.sin(phi*np.pi/180))
        freli=np.arange(2860, 2890., 0.01)
        signal=np.zeros_like(freli)
        for i, fr in enumerate(freq):
            yx=lorentz(freli, fr, lw, -np.sqrt(intens_x[i]))
            yy=lorentz(freli, fr, lw, -np.sqrt(intens_y[i]))
            signal=signal+yx+yy
        return signal

       
class pairs():
    def __init__(self, hyperfine=False):
        self.hyperfine = hyperfine
        self.s1 = System(self.hyperfine)
        self.s2 = System(self.hyperfine)
        self.n_sys = {True:9, False:3}[self.hyperfine]
        self.posx=10e-9
        self.posy=0.
        self.posz=0.
        self.bx=0.
        self.by=0.
        self.bz=0.
        self.b_tot=0.
        self.sx1={True:np.kron(add_I(Sx), np.eye(9)), False:np.kron(Sx, np.eye(3))}[self.hyperfine]
        self.sy1={True:np.kron(add_I(Sy), np.eye(9)), False:np.kron(Sy, np.eye(3))}[self.hyperfine]
        self.sz1={True:np.kron(add_I(Sz), np.eye(9)), False:np.kron(Sz, np.eye(3))}[self.hyperfine]
        self.sx2={True:np.kron(np.eye(9), add_I(Sx)), False:np.kron(np.eye(3), Sx)}[self.hyperfine]
        self.sy2={True:np.kron(np.eye(9), add_I(Sy)), False:np.kron(np.eye(3), Sy)}[self.hyperfine]
        self.sz2={True:np.kron(np.eye(9), add_I(Sz)), False:np.kron(np.eye(3), Sz)}[self.hyperfine]
        self.coupling=True
        self.H = self.get_H()
        self.get_energy()
        self.n_states=len(self.energy)

    def get_H(self):
        ham = np.kron(self.s1.H, np.eye(self.n_sys))+np.kron(np.eye(self.n_sys), self.s2.H)
        if self.coupling:
            ham += self.H_dip()
        return ham         

    def H_dip(self):
        """
        Dipole-Hamiltonian of the two NVs, 2nd is rotated 109.5 deg around the y-axis
        """
        angle = 109.5*np.pi/180
        Sx1 = self.sx1
        Sy1 = self.sy1
        Sz1 = self.sz1
        Sx2 = np.cos(angle)*self.sx2+np.sin(angle)*self.sz2 #+-?
        Sy2 = self.sy2
        Sz2 = np.cos(angle)*self.sz2-np.sin(angle)*self.sx2 #+-?
        x=self.posx
        y=self.posy
        z=self.posz
        r = np.linalg.norm((x, y, z))
        h_dip = gNV*gNV*muNV*muNV*mu0*1e-6/sconst.h/np.pi/4/r**5*(r**2*(np.dot(Sx1, Sx2)+np.dot(Sy1, Sy2)+np.dot(Sz1, Sz2))-3*np.dot(x*Sx1+y*Sy1+z*Sz1, x*Sx2+y*Sy2+z*Sz2))
        return h_dip

    def get_energy(self):
        """
        gets eigenvalues of Hamiltonian
        puts them in list self.energy
        number of states: self.n_states
        self.state_i = Eigenstates still complex
        self.state_a = Eigenstates with absolute value of elements - gives basis
        """
        energy, self.state=np.linalg.eigh(self.H)
        el=list(energy)
        self.energy=[float(el) for el in el]
        self.state_i=self.state.conj().T
        self.state_a=np.absolute(self.state_i)
        self.n_states=len(self.energy)
        self.get_Sx_trans1()
        self.get_Sx_trans2()
        self.get_Sy_trans1()
        self.get_Sy_trans2()
        self.get_Sr_trans1()
        self.get_Sr_trans2()
        self.get_frequency()

    def get_frequency(self):
        """
        creates matrix where every element is the
        absolute value of the energy difference
        between the energies of the resp. states
        same with excited state
        """
        freq=np.zeros((self.n_states, self.n_states), dtype=float)
        for i in range(self.n_states):
            for k in range(self.n_states):
                freq[i, k]=np.absolute(self.energy[i]-self.energy[k])
        self.frequency=freq
        
    def get_Sx_trans1(self):
        """
        calculates Sx in new Basis
        """
        sd=self.sx1
        s=np.dot(np.dot(self.state_i, sd), self.state)
        #s=np.absolute(s)
        self.Sx_trans1=s

    def get_Sx_trans2(self):
        """
        calculates Sx in new Basis
        """
        sd=self.sx2
        s=np.dot(np.dot(self.state_i, sd), self.state)
        #s=np.absolute(s)
        self.Sx_trans2=s

    def get_Sy_trans1(self):
        """
        calculates Sx in new Basis
        """
        sd=self.sy1
        s=np.dot(np.dot(self.state_i, sd), self.state)
        #s=np.absolute(s)
        self.Sy_trans1=s

    def get_Sy_trans2(self):
        """
        calculates Sx in new Basis
        """
        sd=self.sy2
        s=np.dot(np.dot(self.state_i, sd), self.state)
        #s=np.absolute(s)
        self.Sy_trans2=s

    def get_Sr_trans1(self):
        """
        calculates Sr in new Basis
        with Sr = sqrt(Sx^2+Sy^2)
        """
        s=np.zeros((self.n_states, self.n_states), dtype=complex)
        for i in range(self.n_states):
            for k in range(self.n_states):
                s[i, k]=np.sqrt(self.Sx_trans1[i, k]**2+self.Sy_trans1[i, k]**2)
        self.Sr_trans1=s

    def get_Sr_trans2(self):
        """
        calculates Sr in new Basis
        with Sr = sqrt(Sx^2+Sy^2)
        """
        s=np.zeros((self.n_states, self.n_states), dtype=complex)
        for i in range(self.n_states):
            for k in range(self.n_states):
                s[i, k]=np.sqrt(self.Sx_trans2[i, k]**2+self.Sy_trans2[i, k]**2)
        self.Sr_trans2=s

    def get_transitions1(self, n=6):
        """
        gets the first n transition frequencies sorted by the coupling to MW of the resp. states
        f_(ik) sorted by (<i|Sx|k>)^2+(<i|Sy|k>)^2
        phi: angle of MW polarization in xy-plane, if 0: see upper eq.
        """
        if type(n)==int:
            pair_list=[]
            for i in range(self.n_states-1):
                for k in range(i+1, self.n_states):
                    pair_list.append([i, k])
        else:
            pair_list=n
            n=len(pair_list)
        intensity=self.Sr_trans1
        pair_list=sorted(pair_list, key=lambda pair: -intensity[pair[0], pair[1]])
        pair_list=pair_list[:n]
        transitions=[self.frequency[pair[1], pair[0]] for pair in pair_list]
        transitions.sort()
        return transitions

    def get_transitions2(self, n=6):
        """
        gets the first n transition frequencies sorted by the coupling to MW of the resp. states
        f_(ik) sorted by (<i|Sx|k>)^2+(<i|Sy|k>)^2
        phi: angle of MW polarization in xy-plane, if 0: see upper eq.
        """
        if type(n)==int:
            pair_list=[]
            for i in range(self.n_states-1):
                for k in range(i+1, self.n_states):
                    pair_list.append([i, k])
        else:
            pair_list=n
            n=len(pair_list)
        intensity=self.Sr_trans2
        pair_list=sorted(pair_list, key=lambda pair: -intensity[pair[0], pair[1]])
        pair_list=pair_list[:n]
        transitions=[self.frequency[pair[1], pair[0]] for pair in pair_list]
        transitions.sort()
        return transitions

    def get_transition_pairs1(self, n=6):
        """
        gets state pairs with biggest coupling on MW
        sorted after their transition frequency
        """
        pair_list=[]
        for i in range(self.n_states-1):
            for k in range(i+1, self.n_states):
                pair_list.append([i, k])
        pair_list=sorted(pair_list, key=lambda pair: -self.Sr_trans1[pair[0], pair[1]])
        pair_list=pair_list[:n]
        pair_list=sorted(pair_list, key=lambda pair: -self.frequency[pair[0], pair[1]])
        return pair_list

    def get_transition_pairs2(self, n=6):
        """
        gets state pairs with biggest coupling on MW
        sorted after their transition frequency
        """
        pair_list=[]
        for i in range(self.n_states-1):
            for k in range(i+1, self.n_states):
                pair_list.append([i, k])
        pair_list=sorted(pair_list, key=lambda pair: -self.Sr_trans2[pair[0], pair[1]])
        pair_list=pair_list[:n]
        pair_list=sorted(pair_list, key=lambda pair: -self.frequency[pair[0], pair[1]])
        return pair_list

    def set_bx(self, new_bx):
        self.bx=new_bx
        self.b_tot=total(self.bx, self.by, self.bz)
        bx2, by2, bz2 = transform_field(self.bx, self.by, self.bz)
        self.s1.set_bx(new_bx)
        self.s2.set_b_all(bx2, by2, bz2)
        self.H = self.get_H()
        self.get_energy()

    def set_by(self, new_by):
        self.by=new_by
        self.b_tot=total(self.bx, self.by, self.bz)
        self.s1.set_by(new_by)
        self.s2.set_by(new_by)
        self.H = self.get_H()
        self.get_energy()

    def set_bz(self, new_bz):
        self.bz=new_bz
        self.b_tot=total(self.bx, self.by, self.bz)
        bx2, by2, bz2 = transform_field(self.bx, self.by, self.bz)
        self.s1.set_bz(new_bz)
        self.s2.set_b_all(bx2, by2, bz2)
        self.H = self.get_H()
        self.get_energy()

    def set_b_all(self, new_bx, new_by, new_bz):
        self.bx=new_bx
        self.by=new_by
        self.bz=new_bz
        self.b_tot=total(self.bx, self.by, self.bz)
        bx2, by2, bz2 = transform_field(self.bx, self.by, self.bz)
        self.s1.set_b_all(new_bx, new_by, new_bz)
        self.s2.set_b_all(bx2, by2, bz2)
        self.H = self.get_H()
        self.get_energy()

    def set_bnv2(self, new_bz2):
        alpha=109.5*np.pi/180.
        self.set_b_all(np.sin(alpha)*new_bz2, 0, np.cos(alpha)*new_bz2)
        
    def set_posx(self, new_posx):
        self.posx=new_posx
        self.H = self.get_H()
        self.get_energy()

    def set_posy(self, new_posy):
        self.posy=new_posy
        self.H = self.get_H()
        self.get_energy()

    def set_posz(self, new_posz):
        self.posz = new_posz
        self.H = self.get_H()
        self.get_energy()

    def set_pos_all(self, new_posx, new_posy, new_posz):
        self.posx = new_posx
        self.posy = new_posy
        self.posz = new_posz
        self.H = self.get_H()
        self.get_energy()
        
    def plot_energylevels(self):
        """
        plots level scheme
        """
        for i in range(len(self.energy)):
            plt.plot([0, 1], [self.energy[i], self.energy[i]])
        plt.show()
        
    def get_info(self):
        """
        all applied fields
        """
        print "Bx: ", self.bx, ", By: ", self.by, ", Bz: ", self.bz, ", B_tot: ", self.b_tot, ", Bx_2: ", self.s2.bx, ", By_2: ", self.s2.by, ", Bz_2: ", self.s2.bz, ", PosX: ", self.posx, ", PosY: ", self.posy, ", PosZ: ", self.posz
        
    def plot_total(self, x_axis="bz", start=0, stop=1, step=0.01, what = 1, give = False, plot = True, n=2, which=1, diff1=False):
        """
        start, stop, step: in units of the resp. fields, for rotation in numbers of 2pi 
        what: 1 = energy, 2 = transition, diff1: difference to starting energy/transition
        diff2: difference between upper/lower transition(?)
        give: returns x and y
        plot: if it should be plotted
        n: number of transitions shown
        phi: microwave polarization, 0=S_r
        rot_phi, rot_theta: angle for rot. axis (only for brot) in degree
        possible axes:  bx, by, bz, ex, ey, ez
                        ephi, bphi, bephi: rotation of resp. field(s) in xy plane
                        bcontrae: rotation of B-field in xy plane and simultaneous rot.
                                    of E-Field in xy plane in contrary direction
                        btheta: rotation of B-Field around 
                        brot: rotation of B-Field around axis given by rot_phi, rot_theta
        """
        x=np.arange(start, stop, step)
        y=[]
        y2=[]
        initial_values = [self.bx, self.by, self.bz, self.posx, self.posy, self.posz]
        bx_list = self.bx*np.ones((len(x)), dtype = float)
        by_list = self.by*np.ones((len(x)), dtype = float)
        bz_list = self.bz*np.ones((len(x)), dtype = float)
        posx_list = self.posx*np.ones((len(x)), dtype = float)
        posy_list = self.posy*np.ones((len(x)), dtype = float)
        posz_list = self.posz*np.ones((len(x)), dtype = float)
        if x_axis=="bx":
            bx_list = x
        elif x_axis=="by":
            by_list = x
        elif x_axis=="bz":
            bz_list = x
        elif x_axis=="posx":
            posx_list = x
        elif x_axis=="posy":
            posy_list = x
        elif x_axis=="posz":
            posz_list = x
        elif x_axis=="bphi":
            bx_list = [np.cos(i)*b_tot for i in x]
            by_list = [np.sin(i)*b_tot for i in x]
        elif x_axis=="posphi":
            posx_list = [np.cos(i)*b_tot for i in x]
            posy_list = [np.sin(i)*b_tot for i in x]
        elif x_axis=="nv2":
            alpha=109.5*np.pi/180.
            bx_list = [i*np.tan(alpha)/np.sqrt(1+np.tan(alpha)**2) for i in x]
            bz_list = [i/np.sqrt(1+np.tan(alpha)**2) for i in x]
            by_list = np.zeros_like(x)
        for i in range(len(x)):
            self.set_posx(posx_list[i])
            self.set_posy(posy_list[i])
            self.set_posz(posz_list[i])
            energy1 = self.energy
            self.set_bx(bx_list[i])
            self.set_by(by_list[i])
            self.set_bz(bz_list[i])
            if what == 1:
                if diff1:
                    y.append(list(np.abs(np.array(self.energy)-np.array(energy1))))
                else:
                    y.append(self.energy)
                #y_es.append(self.energy_es)
            elif what == 2:
                trans1 = self.get_transitions1(n)
                trans2 = self.get_transitions2(n)
                y.append(trans1)
                y2.append(trans2)
            self.set_bx(initial_values[0])
            self.set_by(initial_values[1])
            self.set_bz(initial_values[2])
            self.set_posx(initial_values[3])
            self.set_posy(initial_values[4])
            self.set_posz(initial_values[5])
        if plot:
            fig=plt.figure()
            ax=fig.add_subplot(111)
            y=np.array(y)
            y2 = np.array(y2)
            if which==1:
                ax.plot(x, y)
            elif which==2:
                ax.plot(x, y2)
            elif which==3:
                ax.plot(x, y, 'b')
                ax.plot(x, y2, 'g')
            if what==1:
                ax.set_ylabel("Energy in MHz")
            elif what==2:
                ax.set_ylabel('Frequency in MHz')
            ax.set_xlabel(x_axis)
            plt.show()
        if give:
            y = np.array(y)
            y2 = np.array(y2)
            if which==1:
                return x, y
            elif which==2:
                return x, y2
            elif which == 3:
                return x, y, y2
           
    def coupling_strength(self, transition_B=-1, transition_A=-1, n=6):
        #tB={-1:True, 1:False}[transition_B]
        #tA={-1:True, 1:False}[transition_A]
        trans1 = self.get_transitions1(n)
        #trans2 = self.get_transitions2(n)
        d1 = trans1[1]-trans1[0]
        d2 = trans1[2]-trans1[1]
        return d1, d2

    def new_base_mask(self):
        """
        gives the old base in the new base with the state of the highest overlap
        """
        mask = np.zeros_like(self.state_a)
        for i in range(9):
            for k in range(9):
                if self.state_a[i, k]>0.5:
                    mask[i, k]=1
        return mask

    def new_base_dic(self):
        """
        gives the states where the 'old' states are most in the new base
        """
        mask = np.zeros_like(self.state_a)
        dic={}
        for i in range(9):
            for k in range(9):
                if self.state_a[i, k]>0.5:
                    dic[k]=i    
        return dic

    def NV1_lower_freqs(self):
        """
        Frequencies for NV1 in the lower energy state
        t0: transition freq for NV1 from 0 to lower with NV2 in 0
        tm1: freq for NV1 from 0 to lower with NV2 in lower
        tp1: freq for NV1 from 0 to lower with NV2 in higher
        """
        dic = self.new_base_dic()
        u1 = self.frequency[dic[4], dic[7]] #NV1 00 -> -10
        u2 = self.frequency[dic[4], dic[1]] #NV1 00 -> +10
        a1 = self.frequency[dic[4], dic[5]] #NV2 00 -> 0-1
        a2 = self.frequency[dic[4], dic[3]] #NV2 00 -> 0+1
        if u2>u1:
            t0=u1
            if a1 < a2:
                tm1=self.frequency[dic[5], dic[8]] #0-1 -> -1-1
                tp1=self.frequency[dic[3], dic[6]] #0+1 -> -1+1
            else:
                tp1=self.frequency[dic[5], dic[8]] #0-1 -> -1-1
                tm1=self.frequency[dic[3], dic[6]] #0+1 -> -1+1
        else:
            t0=u2
            if a1 < a2:
                tm1=self.frequency[dic[5], dic[2]] #0-1 -> +1-1
                tp1=self.frequency[dic[3], dic[0]] #0+1 -> +1+1
            else:
                tp1=self.frequency[dic[5], dic[2]] #0-1 -> +1-1
                tm1=self.frequency[dic[3], dic[0]] #0+1 -> +1+1
        return [t0, tm1, tp1]

    def NV2_lower_freqs(self):
        """
        Frequencies for NV2 in the lower energy state
        t0: transition freq for NV2 from 0 to lower with NV1 in 0
        tm1: freq for NV2 from 0 to lower with NV1 in lower
        tp1: freq for NV2 from 0 to lower with NV1 in higher
        """
        dic = self.new_base_dic()
        u1 = self.frequency[dic[4], dic[5]] #00 -> 0-1
        u2 = self.frequency[dic[4], dic[3]] #00 -> 0+1
        a1 = self.frequency[dic[4], dic[7]] #NV1 00 -> -10
        a2 = self.frequency[dic[4], dic[1]] #NV1 00 -> +10
        if u2>u1:
            t0=u1
            if a1 < a2:
                tm1=self.frequency[dic[7], dic[8]] #-10 -> -1-1
                tp1=self.frequency[dic[1], dic[2]] #+10 -> +1-1
            else:
                tp1=self.frequency[dic[7], dic[8]] #-10 -> -1-1 
                tm1=self.frequency[dic[1], dic[2]] #+10 -> +1-1
        else:
            t0=u2
            if a1 < a2:
                tm1=self.frequency[dic[7], dic[6]] #-10 -> -1+1
                tp1=self.frequency[dic[1], dic[0]] #+10 -> +1+1
            else:
                tp1=self.frequency[dic[7], dic[6]] #-10 -> -1+1
                tm1=self.frequency[dic[1], dic[0]] #-10 -> +1+1
        return [t0, tm1, tp1]

    def all_coupling(self):
        """
        d1: change in NV1 lower freq for NV2 from 0 -> higher
        d2: change in NV1 lower freq for NV2 from 0 -> lower
        d3: change in NV2 lower freq for NV1 from 0 -> higher
        d4: change in NV2 lower freq for NV1 from 0 -> lower
        """
        freq1 = self.NV1_lower_freqs()
        freq2 = self.NV2_lower_freqs()
        d1 = abs(freq1[0]-freq1[2])
        d2 = abs(freq1[0]-freq1[1])
        d3 = abs(freq2[0]-freq2[2])
        d4 = abs(freq2[0]-freq2[1])
        return [d1, d2, d3, d4]

    def calc_shift(self):
        freqs = self.NV1_lower_freqs()
        d1 = abs(freqs[0]-freqs[1])
        d2 = abs(freqs[0]-freqs[2])
        return d1, d2
    
    def get_sphere(self, radius=1.e-8, n=100):
        phi = np.linspace(0, 2*np.pi, n)
        theta = np.linspace(0, np.pi, int(n/2.))
        val1 = np.zeros((n, int(n/2.)))
        val2 = np.zeros_like(val1)
        x=np.zeros_like(val1)
        y=np.zeros_like(val1)
        z=np.zeros_like(val1)
        for i in range(n):
            for k in range(int(n/2.)):
                x[i, k]=np.cos(phi[i])*np.sin(theta[k])*radius
                y[i, k]=np.sin(phi[i])*np.sin(theta[k])*radius
                z[i, k]=np.cos(theta[k])*radius
                self.set_posz(1)
                self.set_posx(x[i, k])
                self.set_posy(y[i, k])
                self.set_posz(z[i, k])
                val1[i, k], val2[i, k] = self.calc_shift()
        return val1, val2, x, y, z
                
    def plot_sphere(self, num=1, radius=1.e-8, n=100):
        val1, val2, x, y, z = self.get_sphere(radius, n)
        val={1:val1, 2:val2}[num]
        fig = plt.figure()
        ax=fig.gca(projection='3d')
        surf=ax.plot_surface(x, y, z, rstride=4, cstride=4, facecolors=cm.jet(val/val.max()), antialiased=True)
        plt.show()


def clean_up(vals, limit):
    for i in range(len(vals)):
        for k in range(len(vals[0])):
            if vals[i, k]>limit:
                vals[i, k]=limit
    return vals

def paint_sphere(vals, x, y, z):
    fig=plt.figure()
    ax=fig.gca(projection='3d')
    surf=ax.plot_surface(x, y, z, rstride=4, cstride=4, facecolors=cm.jet(vals/vals.max()), antialiased=True)
    #fig.colorbar()
    plt.show()
    
def lin_gradient(freqs, b, k):
    y11, y12, y21, y22 = sort_back_arb(freqs, b, k)
    data = [y11, y12, y21, y22]
    cl = np.zeros((4))
    ml = np.zeros((4))
    for i in range(4):
        dati = data[i]
        p0=[dati[:,1].mean(), (dati[-1,1]-dati[0,1])/(dati[-1,0]-dati[0,0])]
        p1, success = optimize.leastsq(err_lin, p0[:], args=(dati[:,0], dati[:,1]))
        cl[i]=p1[1]
        ml[i]=p1[0]
    return cl, ml

def transition_dic(data):
    t0=[]
    t1=[]
    t2=[]
    t3=[]
    for i in range(len(data)):
        if data[i,0]==0:
            t0.append([data[i,1], data[i,2]])
        elif data[i,0]==1:
            t1.append([data[i,1], data[i,2]])
        elif data[i,0]==2:
            t2.append([data[i,1], data[i,2]])
        elif data[i,0]==3:
            t3.append([data[i,1], data[i,2]])
    dic = {'t0':t0, 't1':t1, 't2':t2, 't3':t3}
    return dic
    

def position(freq_array, p0=[5.e-9, 5.e-9, 5.e-9]):
    x = np.arange(12)
    y = freq_array
    p1, success = optimize.leastsq(err_coupling, p0[:], args=(x, y))
    f1 = fit_coupling(x, p1)
    return p1, f1

def position2(freq_array, b, k, p0=[5.e-9, 5.e-9, 5.e-9]):
    x=np.arange(len(freq_array))
    y = freq_array
    p1, success = optimize.leastsq(err_coup, p0[:], args=(x,y, b, k))
    f1 = fit_coupling_chaos(x, p1, b, k)
    return p1, f1


err_coupling = lambda p, x, y: fit_coupling(x, p)-y

def fit_coupling(x, p):
    tges = []
    p1=pairs(False)
    p1.set_posz(1)
    p1.set_posx(p[0])
    p1.set_posy(p[1])
    p1.set_posz(p[2])
    p1.set_bz(25)
    freqs = p1.NV1_lower_freqs()
    tges.append(abs(freqs[0]-freqs[2]))
    tges.append(abs(freqs[0]-freqs[1]))
    p1.set_bz(50)
    freqs = p1.NV1_lower_freqs()
    tges.append(abs(freqs[0]-freqs[2]))
    tges.append(abs(freqs[0]-freqs[1]))
    p1.set_bz(75)
    freqs = p1.NV1_lower_freqs()
    tges.append(abs(freqs[0]-freqs[2]))
    tges.append(abs(freqs[0]-freqs[1]))
    p1.set_bnv2(25)
    freqs = p1.NV2_lower_freqs()
    tges.append(abs(freqs[0]-freqs[2]))
    tges.append(abs(freqs[0]-freqs[1]))
    p1.set_bnv2(50)
    freqs = p1.NV2_lower_freqs()
    tges.append(abs(freqs[0]-freqs[2]))
    tges.append(abs(freqs[0]-freqs[1]))
    p1.set_bnv2(75)
    freqs = p1.NV2_lower_freqs()
    tges.append(abs(freqs[0]-freqs[2]))
    tges.append(abs(freqs[0]-freqs[1]))
    treturn = [tges[i] for i in x]
    return treturn

def fit_coupling2(x, p):
    tges = []
    p1=pairs(False)
    p1.set_pos_all(p[0], p[1], p[2])
    p1.set_bz(25)
    freqs = p1.NV1_lower_freqs()
    tges.append(abs(freqs[0]-freqs[2]))
    tges.append(abs(freqs[0]-freqs[1]))
    p1.set_bz(50)
    freqs = p1.NV1_lower_freqs()
    tges.append(abs(freqs[0]-freqs[2]))
    tges.append(abs(freqs[0]-freqs[1]))
    p1.set_bz(75)
    freqs = p1.NV1_lower_freqs()
    tges.append(abs(freqs[0]-freqs[2]))
    tges.append(abs(freqs[0]-freqs[1]))
    p1.set_bnv2(25)
    freqs = p1.NV2_lower_freqs()
    tges.append(abs(freqs[0]-freqs[2]))
    tges.append(abs(freqs[0]-freqs[1]))
    p1.set_bnv2(50)
    freqs = p1.NV2_lower_freqs()
    tges.append(abs(freqs[0]-freqs[2]))
    tges.append(abs(freqs[0]-freqs[1]))
    p1.set_bnv2(75)
    freqs = p1.NV2_lower_freqs()
    tges.append(abs(freqs[0]-freqs[2]))
    tges.append(abs(freqs[0]-freqs[1]))
    treturn = [tges[i] for i in x]
    return treturn

dic_bf = {3:np.array([25, 50, 75]), 7:np.array([10, 20, 30, 40, 50, 60, 70]), 15:np.linspace(1.e-7, 200, 15), 6:np.array([16, 25, 33, 41, 49, 57])}

def get_new_meas():
    f=open('new_data_format.pys')
    dic = pickle.load(f)
    f.close()
    freq = np.array(dic['f'])*1.e-3
    bn = np.array(dic['b'])*10.
    ks = np.array(dic['k'])
    return freq, bn, ks

def fit_coupling3(x, p):
    tges = []
    nl = int(len(x)/4)
    b_list = dic_bf[nl]
    p1=pairs(False)
    p1.set_pos_all(p[0], p[1], p[2])
    for i in range(nl):
        p1.set_bz(b_list[i])
        freqs = p1.NV1_lower_freqs()
        tges.append(abs(freqs[0]-freqs[2]))
        tges.append(abs(freqs[0]-freqs[1]))
    for i in range(nl):
        p1.set_bnv2(b_list[i])
        freqs = p1.NV2_lower_freqs()
        tges.append(abs(freqs[0]-freqs[2]))
        tges.append(abs(freqs[0]-freqs[1]))
    treturn = [tges[i] for i in x]
    return treturn

def fit_coupling_chaos(x, p, b, k):
    tges = []
    p1=pairs(False)
    p1.set_pos_all(p[0], p[1], p[2])
    for i in range(len(x)):
        if k[i]<1.5:
            p1.set_bz(b[i])
            freqs = p1.NV1_lower_freqs()
            if k[i]==0:
                tges.append(abs(freqs[0]-freqs[2]))
            else:
                tges.append(abs(freqs[0]-freqs[1]))
        else:
            p1.set_bnv2(b[i])
            freqs = p1.NV2_lower_freqs()
            if k[i]==2:
                tges.append(abs(freqs[0]-freqs[2]))
            else:
                tges.append(abs(freqs[0]-freqs[1]))
    treturn = [tges[i] for i in x]
    return treturn


err_coup = lambda p, x, y, b, k: fit_coupling_chaos(x, p, b, k)-y

def plotwhole(data, axis=2):
    n1=data.shape[axis]
    n2 = data.shape[axis-1]
    if axis<2:
        n3 = data.shape[axis+1]
    else:
        n3 = data.shape[0]
    a=int(np.ceil(np.sqrt(n1)))
    b=int(np.ceil(n1/float(a)))
    res = np.ones((a*n2, b*n3))*data.mean()
    for i in range(a):
        for l in range(b):
            arr = {0:data[i*b+l,:,:], 1:data[:,i*b+l,:], 2:data[:,:,i*b+l]}[axis]
            res[a*n2:(a+1)*n2, b*n3:(b+1)*n3]=arr
    return res
    
def sort_data_back(p):
    x=np.array([25, 50, 75])
    y11 = np.array([p[0], p[2], p[4]])
    y12 = np.array([p[1], p[3], p[5]])
    y21 = np.array([p[6], p[8], p[10]])
    y22 = np.array([p[7], p[9], p[11]])
    return y11, y12, y21, y22

def sort_back_arb(fr, k, b):
    y11=[]
    y12=[]
    y21=[]
    y22=[]
    for i in range(len(fr)):
        if k[i]==0:
            y11.append([b[i], fr[i]])
        elif k[i]==1:
            y12.append([b[i], fr[i]])
        elif k[i]==2:
            y21.append([b[i], fr[i]])
        elif k[i]==3:
            y22.append([b[i], fr[i]])
    return np.array(y11), np.array(y12), np.array(y21), np.array(y22)

def sort_for_arb(y11, y12, y21, y22):
    nl = len(y11)*4
    p = np.zeros((nl))
    ind11 = range(0, int(nl/2), 2)
    ind12 = range(1, int(nl/2), 2)
    ind21 = range(int(nl/2), nl, 2)
    ind22 = range(int(nl/2+1), nl, 2) 
    p[ind12]=y12
    p[ind21]=y21
    p[ind22]=y22
    return p
    
def shift_new(bx, by, bz):
    p1 = pairs(False)
    p2 = pairs(False)
    p1.set_posx(9.18e-9)
    p1.set_posy(0.985e-9)
    p1.set_posz(5.8e-9)
    p2.set_posx(-3.25e-9)
    p2.set_posy(-6.13e-9)
    p2.set_posz(5.14e-9)
    p1.set_bx(bx)
    p1.set_by(by)
    p1.set_bz(bz)
    p2.set_bx(bx)
    p2.set_by(by)
    p2.set_bz(bz)
    return p1.all_coupling(), p2.all_coupling()

def get_kart(n):
    theta, phi = np.meshgrid(np.linspace(0, np.pi, int(n/2)), np.linspace(0, 2*np.pi, n))
    x = np.cos(phi)*np.sin(theta)
    y = np.sin(phi)*np.sin(theta)
    z = np.cos(theta)
    return x, y, z

def shift_diff(b_tot=20, n=100):
    x, y, z=get_kart(n)
    print x.shape
    bx, by, bz= x*b_tot, y*b_tot, z*b_tot
    print bx.shape
    n2=int(n/2)
    print n2
    diffs = np.zeros((4, n, n2))
    for i in range(n):
        for k in range(n2):
            p1a, p2a = shift_new(bx[i, k], by[i, k], bz[i, k])
            #print type(p1a), len(p1a)
            diffs[:,i,k] = abs(np.array(p1a)-np.array(p2a))
    return diffs

def shift_val(b_tot=20, n=100):
    x, y, z=get_kart(n)
    print x.shape
    bx, by, bz= x*b_tot, y*b_tot, z*b_tot
    print bx.shape
    n2=int(n/2)
    print n2
    vals = np.zeros((2, 4, n, n2))
    for i in range(n):
        for k in range(n2):
            p1a, p2a = shift_new(bx[i, k], by[i, k], bz[i, k])
            vals[0, :, i, k] = p1a
            vals[1, :, i, k] = p2a
            #print type(p1a), len(p1a)
            #diffs[:,i,k] = abs(np.array(p1a)-np.array(p2a))
    return diffs

freq_valentina = np.array([60.51, 56.58, 63.34, 53.88, 65.21, 51.55, 64.37, 55.34, 66.64, 47.16, 73.12, 44.24])*1.e-3
freq_susan = np.array([52.24, 54.13, 51.25, 54.87, 49.95, 55.07, 50.78, 54.68, 49.02, 55.62, 47.24, 57.56])*1.e-3

def freq_pic(freqs, fit):
    y11, y12, y21, y22 = sort_data_back(fit)
    m11, m12, m21, m22 = sort_data_back(freqs)
    x=[25, 50, 75]
    fig=plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x, y11, 'b-', lw=3, alpha=0.4)
    ax.plot(x, y12, 'b-', lw=3, alpha=0.4)
    ax.plot(x, y21, 'r-', lw=3, alpha=0.4)
    ax.plot(x, y22, 'r-', lw=3, alpha=0.4)
    ax.plot(x, m11, 'b+-')
    ax.plot(x, m12, 'b+-')
    ax.plot(x, m21, 'r+-')
    ax.plot(x, m22, 'r+-')

def freq_pic_chaos(freqs, b, k, cl, ml):
    cdic = {0:'b', 1:'b', 2:'r', 3:'r'}
    clist = [cdic[i] for i in k]
    frfit = np.linspace(b.min(), b.max(), 100)
    fit0 = lin_fit([ml[0], cl[0]], frfit)
    fit1 = lin_fit([ml[1], cl[1]], frfit)
    fit2 = lin_fit([ml[2], cl[2]], frfit)
    fit3 = lin_fit([ml[3], cl[3]], frfit)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(frfit, fit0, 'b-', lw=3, alpha=0.4)
    ax.plot(frfit, fit1, 'b-', lw=3, alpha=0.4)
    ax.plot(frfit, fit2, 'r-', lw=3, alpha=0.4)
    ax.plot(frfit, fit3, 'r-', lw=3, alpha=0.4)
    ax.scatter(b, freqs, clist)
    

def valentina_to_lab(x, y, z):
    alpha = 54.8*np.pi/180.
    beta = 19.5*np.pi/180.
    xn = z*np.cos(beta)-x*np.sin(beta)
    yn = y
    zn = x*np.cos(beta)+z*np.cos(alpha)
    return xn, yn, zn

def susan_to_lab(pos = np.array([0,0,0])):
    x=pos[0]
    y = pos[1]
    z=pos[2]
    xn = (3*x-np.sqrt(3)*y)/(2*np.sqrt(3))
    yn = (x+np.sqrt(3)*y+2*np.sqrt(2)*z)/(2*np.sqrt(3))
    zn = (-np.sqrt(2)*x-np.sqrt(6)*y+2*z)/(2*np.sqrt(3))
    return np.array([xn, yn, zn])

def susan_to_lab2(pos = np.array([0,0,0])):
    alpha = 109.5*np.pi/360.
    beta = np.pi/6.
    r1 = np.array([[1,0,0], [0,np.cos(-alpha), -np.sin(-alpha)], [0, np.sin(-alpha), np.cos(-alpha)]])
    r2 = np.array([[np.cos(beta), -np.sin(beta),0],[np.sin(beta), np.cos(beta), 0],[0,0,1]])
    posn = np.dot(np.dot(r1,r2),pos)
    return posn

import math, random

def fibonacci_sphere(samples=1,randomize=False):
    rnd = 1.
    if randomize:
        rnd = random.random() * samples

    points = []
    offset = 2./samples
    increment = math.pi * (3. - math.sqrt(5.));

    for i in range(samples):
        y = ((i * offset) - 1) + (offset / 2);
        r = math.sqrt(1 - pow(y,2))

        phi = ((i + rnd) % samples) * increment

        x = math.cos(phi) * r
        z = math.sin(phi) * r

        points.append([x,y,z])

    return points

fit_lin=lambda p, x: p[0]+p[1]*x
err_lin=lambda p, x, y: fit_lin(p, x)-y

