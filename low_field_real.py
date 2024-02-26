from __future__ import division
import numpy as np
import scipy.constants as sconst
import matplotlib.pyplot as plt
import pickle
import math
from scipy import optimize
from scipy.optimize import minimize

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
betaNV = -2.8
zfsNV = 2.8703e3
zfsNVes = 1.42e3
zfs14N=-4.941
beta14N=0.0003
A_NVN=-2.172
A_T = -2.630
r3d=0.02
r2e=1           #r2e=50*r3d, otherwise arbitrary
r_new = 1
bz_lac = -zfsNV/betaNV
#hyperfine=False

betaHe3 = 3.24341 #kHz/G
betaXe129 = 1.1777


#Spin-Operators

Sx = 1/np.sqrt(2)*np.array([[0, 1+0.j, 0], [1, 0, 1], [0, 1, 0]])
Sy = 1j/np.sqrt(2)*np.array([[0, -1, 0],[1, 0, -1],[0, 1, 0]])
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

lorentz=lambda x, x0, w, a: 2*a/np.pi*w/(4*(x-x0)**2+w**2)



#*******************************
#***   Start of the Program  ***
#*******************************

class System():
    def __init__(self, hyperfine = False, excited=True):
        """
        b_tot=0, e_tot=0
        """
        self.hyperfine=hyperfine
        self.excited=excited
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
        self.hamiltonians=[self.H_zee_NV, self.H_zfs_NV, self.H_stark, self.H_stark_xz, self.H_zfs_N14, self.H_zee_N14, self.H_hfs, self.H_T]
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
        h_stark=r3d*self.ez*(np.dot(Sz, Sz)-2/3*np.eye(3))+r2e*(self.ey*(np.dot(Sx, Sy)+np.dot(Sy, Sx))-self.ex*(np.dot(Sx, Sx)-np.dot(Sy, Sy)))
        if self.hyperfine:
            return add_I(h_stark)
        else:
            return h_stark

    def H_stark_xz(self):
        """
        NV GS Stark-Shift xz term
        """
        h_stark2 = r_new*(self.ex*(np.dot(Sx, Sz)+np.dot(Sz, Sx))+self.ey*(np.dot(Sy, Sz)+np.dot(Sz, Sy)))
        if self.hyperfine:
            return add_I(h_stark2)
        else:
            return h_stark2

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
        """
        N14-NV transversal HF
        """
        h_hfs=A_T*(np.dot(add_I(Sx), add_S(Sx))+np.dot(add_I(Sy), add_S(Sy)))
        if self.hyperfine:
            return h_hfs
        else:
            return 0

    def H_total(self):
        """
        NV ZFS, NV Zee, NV stark, N14 ZFS, N14 Zee, NV-N14 HFS
        """
        h_tot={True:np.zeros((9, 9), dtype=complex), False:np.zeros((3, 3), dtype=complex)}[self.hyperfine]
        for ham in self.hamiltonians:
            h_tot+=ham()
            #print ham
        return h_tot

    def H_excited(self):
        """
        NV ES Hamiltonian, only Zfs + Zee
        """
        h_tot={True:np.zeros((9, 9), dtype=complex), False:np.zeros((3, 3), dtype=complex)}[self.hyperfine]
        for ham in self.hamiltonians_es:
            h_tot+=ham()
            #print ham
        return h_tot

    """
    def H_sz(self):
        h_sz = self.ex*(np.dot(Sx, Sz)+np.dot(Sz, Sx))+self.ey*(np.dot(Sy, Sz)+np.dot(Sz, Sy))
        return h_sz
    """
    
    #---------------------
    #--- set Parameter ---
    #---------------------

    def delete_h(self, ham, es=False):
        """
        delete specific hamiltonian ham from list of hamiltonians
        """
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

    def add_h(self, ham, es=False):
        if es:
            try:
                self.hamiltonians_es.append(ham)
            except:
                print 'not in'
            finally:
                self.H_es=self.H_excited()
        else:
            try:
                self.hamiltonians.append(ham)
            except:
                print 'not in'
            finally:
                self.H=self.H_total()
        self.get_energy()

    def set_d(self, new_d):
        self.d=new_d
        self.H=self.H_total()
        #self.H_es = self.H_excited()
        self.get_energy()
        #self.get_energy_es()

    def set_bx(self, new_bx):
        """
        calculates new energy/states for new B_x
        """
        self.bx=new_bx
        self.b_tot=total(self.bx, self.by, self.bz)
        self.H=self.H_total()
        self.get_energy()
        if self.excited:
            self.H_es = self.H_excited()
            self.get_energy_es()

    def set_by(self, new_by):
        """
        calculates new energy/states for new B_y
        """
        self.by=new_by
        self.b_tot=total(self.bx, self.by, self.bz)
        self.H=self.H_total()
        self.get_energy()
        if self.excited:
            self.H_es = self.H_excited()
            self.get_energy_es()

    def set_bz(self, new_bz):
        """
        calculates new energy/states for new B_z
        """
        self.bz=new_bz
        self.b_tot=total(self.bx, self.by, self.bz)
        self.H=self.H_total()
        self.get_energy()
        if self.excited:
            self.H_es = self.H_excited()
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
        self.set_b_all(new_br*np.cos(bphi), new_br*np.sin(bphi), self.bz)
        
    def set_theta(self, new_btot, new_theta):
        """
        changes Bx, By, Bz but leaves B_phi same
        (CAUTION: angle not yet really kept)
        new_theta in degree, starting from pole
        """
        btheta = new_theta*np.pi/180
        bphi = np.arccos(self.bx/self.b_tot)
        br = new_btot*np.cos(btheta)
        self.set_b_all(br*np.cos(bphi), br*np.sin(bphi), new_btot*np.sin(btheta))

    def set_all_zero(self):
        """
        calculates new energy/states for B-Field = 0
        """
        self.bx=0
        self.by=0
        self.bz=0
        self.b_tot=total(self.bx, self.by, self.bz)
        self.H=self.H_total()
        self.H_es = self.H_excited()
        self.get_energy()
        self.get_energy_es()
        
    def set_all_b(self, new_bx, new_by, new_bz):
        """
        sets all B-Field components before diagonalizing
        """
        self.bx=new_bx
        self.by=new_by
        self.bz=new_bz
        self.b_tot=total(self.bx, self.by, self.bz)
        self.H=self.H_total()
        self.H_es = self.H_excited()
        self.get_energy()
        self.get_energy_es()

    def set_b_d(self, new_bx, new_by, new_bz, new_d):
        """
        sets all B-Field components + D before diagonalizing
        """
        self.bx=new_bx
        self.by=new_by
        self.bz=new_bz
        self.d=new_d
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
            Sx_t=abs(Sx_trans[pair[0], pair[1]])
            Sy_t=abs(Sy_trans[pair[0], pair[1]])
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
        if self.hyperfine:
            self.ms_0 = abs(self.state_i[:, 3])**2+abs(self.state_i[:, 4])**2+abs(self.state_i[:, 5])**2
        else:
            self.ms_0 = self.state_i[:,1]**2
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
        #diff = np.zeros((self.n_states, self.n_states), dtype=float)
        fluorescence_diff=np.zeros((self.n_states, self.n_states))
        for i in range(self.n_states):
            for k in range(self.n_states):
                freq[i, k]=np.absolute(self.energy[i]-self.energy[k])
                fluorescence_diff[i, k]=np.absolute(self.ms_0[i]-self.ms_0[k])
        self.fluorescence = fluorescence_diff
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

    def get_Sz_trans(self):
        """
        calculates Sz in new Basis
        """
        sd=Sz
        if self.hyperfine: sd=add_I(Sz)
        s=np.dot(np.dot(self.state_i, sd), self.state)
        #s=np.absolute(s)
        self.Sz_trans=s

    def get_Sr_trans(self):
        """
        calculates Sr in new Basis
        with Sr = sqrt(Sx^2+Sy^2)
        """
        self.Sr_trans=np.zeros((self.n_states, self.n_states), dtype=complex)
        for i in range(self.n_states):
            for k in range(self.n_states):
                self.Sr_trans[i, k]=np.sqrt(abs(self.Sx_trans[i, k])**2+abs(self.Sy_trans[i, k])**2)
        

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
            intensity=abs(self.Sr_trans)
        else:
            intensity=abs(self.get_S_phi(phi))
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
        pair_list=sorted(pair_list, key=lambda pair: -abs(self.Sr_trans[pair[0], pair[1]]))
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
        if self.hyperfine:
            sd=add_I(np.cos(phi)*Sx+np.sin(phi)*Sy)
        else:
            sd = np.cos(phi)*Sx+np.sin(phi)*Sy
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
        initial_values = [self.bx, self.by, self.bz, self.ex, self.ey, self.ez, self.d]
        bx_list = self.bx*np.ones((len(x)), dtype = float)
        by_list = self.by*np.ones((len(x)), dtype = float)
        bz_list = self.bz*np.ones((len(x)), dtype = float)
        ex_list = self.ex*np.ones((len(x)), dtype = float)
        ey_list = self.ey*np.ones((len(x)), dtype = float)
        ez_list = self.ez*np.ones((len(x)), dtype = float)
        d_list = self.d*np.ones((len(x)), dtype=float)
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
        elif x_axis == 'd':
            d_list = x

        for i in range(len(x)):
            self.set_d(d_list[i])
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
        self.set_d(initial_values[6])
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

    def plot_barray(self, b_array=np.array([np.arange(3), np.arange(3), np.arange(3)]), what = 1, give = False, plot = True, n=2, phi = 0, xaxis=None):
        """
        what: 1 = energy, 2 = transition, diff1: difference to starting energy/transition
        give: returns x and y
        plot: if it should be plotted
        n: number of transitions shown
        phi: microwave polarization, 0=S_r
        """
        if xaxis==None:
            x=b_array[2] #np.sqrt(b_array[0]**2+b_array[1]**2+b_array[2]**2)  #np.arange(len(b_array[0]))
        else:
            x=xaxis
        y=[]
        initial_values = [self.bx, self.by, self.bz]
        bx_list = b_array[0] #self.bx*np.ones((len(x)), dtype = float)
        by_list = b_array[1] #self.by*np.ones((len(x)), dtype = float)
        bz_list = b_array[2] #self.bz*np.ones((len(x)), dtype = float)
        b_tot = self.b_tot
        b_r = np.sqrt(self.bx**2+self.by**2)
        
        for i in range(len(x)):
            self.set_bx(bx_list[i])
            self.set_by(by_list[i])
            self.set_bz(bz_list[i])
            energy1 = self.energy
            if what == 1:
                y.append(self.energy)
            elif what == 2:
                trans = self.get_transitions(n, phi)
                y.append(trans)
        self.set_bx(initial_values[0])
        self.set_by(initial_values[1])
        self.set_bz(initial_values[2])
        if plot:
            fig=plt.figure()
            ax=fig.add_subplot(111)
            y=np.array(y)
            for i in range(len(self.energy)):
                ax.plot(x, y)
            if what==1:
                ax.set_ylabel("Energy in MHz")
            elif what==2:
                ax.set_ylabel('Frequency in MHz')
            ax.set_xlabel('Magnetic field in G')
            plt.show()
        if give:
            y = np.array(y)
            return x, y


    def show_intensities(self):
        """
        plots the transition intensities
        """
        fig=plt.figure()
        ax=fig.add_subplot(111)
        ax.imshow(abs(self.Sr_trans)**2, interpolation="none")
        ax.set_yticks(range(9))
        ax.set_xticks(range(9))
        ax.set_yticklabels(self.energy)
        ax.set_xticklabels(range(9))
        ax.set_title("<i|S_r|j>")
        ax.set_ylabel("Energy of state in MHz")
        ax.set_xlabel("state")
        plt.show()

    def get_intensity_list(self, state1, state2):
        """
        gets the transition intensity between state1 and state2 for diff.
        MW angles phi
        """
        phi=np.arange(0, 2*np.pi+0.005, 0.01)
        intensity=[]
        for p in phi:
            trans=self.get_intensity(p, state1, state2)
            intensity.append(trans)
        return phi, intensity

    def plot_intensity(self, state1, state2):
        """
        plots the transition intensity between state1 and state2 for
        diff. MW angles phi
        """
        phi, intensity=self.get_intensity_list(state1, state2)
        plt.plot(phi, intensity)

    def plot_all_intensities(self, pairs=6):
        """
        plots the transition intensity between the states of 'pairs' number of
        state pairs with the highest intensity over phi
        """
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
        """
        gives all transition intensities and all transition frequencies for
        x- or y-polarized MW
        """
        transition={'x':self.Sx_trans, 'y':self.Sy_trans}[pol]
        freq=self.frequency.flatten()
        intens=transition.flatten()
        return freq, intens

    def plot_transition_intens(self, pol='x', c='k', factor=1.):
        """
        plots all transitions over 1 G with their intensity
        (pre-ODMR simulation)
        """
        freq, intens=self.get_transition_intens(pol)
        for i, fr in enumerate(freq):
            if fr>1e3:
                plt.plot([fr, fr], [0, intens[i]*factor], c)
                

    def odmr(self, lw=0.1, phi=0., freq1=None, freq2=None, steps=1000):
        """
        gives ODMR between freq1 and freq2. The linewidth is given bei lw
        and steps is the number of frequencies between freq1 and freq2.
        The contrast of the lines is given by sqrt(intens_x)+sqrt(intens_y)
        The calculation takes all transitions into account
        """
        num_t=sum(range(len(self.energy)+1))
        freq=self.frequency.flatten()
        intens_x=abs(self.Sx_trans.flatten())
        intens_x=intens_x*abs(np.cos(phi*np.pi/180))
        intens_y=abs(self.Sy_trans.flatten())
        intens_y=intens_y*abs(np.sin(phi*np.pi/180))
        if freq1 is not None and freq2 is not None:
            freli = np.linspace(freq1, freq2, steps)
        else:
            freli=np.arange(2860, 2890., 0.01)
        signal=np.zeros_like(freli)
        for i, fr in enumerate(freq):
            yx=lorentz(freli, fr, lw, -np.sqrt(intens_x[i]))
            yy=lorentz(freli, fr, lw, -np.sqrt(intens_y[i]))
            signal=signal+yx+yy
        return signal

    def odmr2(self, lw=0.1, phi=0., freq1=None, freq2=None, steps=1000, num=6.):
        """
        gives ODMR between freq1 and freq2. The linewidth is given bei lw
        and steps is the number of frequencies between freq1 and freq2.
        The contrast of the lines is given by intens_x+intens_y
        The calculation takes all transitions into account
        Difference to odmr() is, that the intensities are not sqrt()
        """
        num_t=sum(range(len(self.energy)+1))
        freq=self.frequency.flatten()
        intens_x=abs(self.Sx_trans.flatten())
        intens_x=intens_x*abs(np.cos(phi*np.pi/180))
        intens_y=abs(self.Sy_trans.flatten())
        intens_y=intens_y*abs(np.sin(phi*np.pi/180))
        if freq1 is not None and freq2 is not None:
            freli = np.linspace(freq1, freq2, steps)
        else:
            freli=np.arange(2860, 2890., 0.01)
        signal=np.zeros_like(freli)
        for i, fr in enumerate(freq):
            yx=lorentz(freli, fr, lw, -intens_x[i])
            yy=lorentz(freli, fr, lw, -intens_y[i])
            signal=signal+yx+yy
        return signal

    def hyperfine_odmr(self, iz=1, lw=0.1, freq1=None, freq2=None, steps=1000, a1=1., a2=1., a3=1.):
        """
        gives ODMR between freq1 and freq2. The linewidth is given bei lw
        and steps is the number of frequencies between freq1 and freq2.
        The contrast of the lines is given by sqrt(intens_x**2+intens_y**2)
        The calculation takes all transitions into account.
        The nuclear spin is polarized into the state iz.
        """
        rho = {1:np.diag([1+0.j, 0, 0]), 0:np.diag([0+0.j, 1, 0]), -1:np.diag([0+0.j, 0, 1])}[iz]
        rho = np.kron(np.eye(3, dtype=complex), rho)
        rho_t = np.dot(np.dot(self.state_i, rho), self.state)
        #print rho_t.dtype
        #print self.Sx_trans.dtype
        num_t = sum(range(4))
        freq = self.frequency.flatten()
        self.get_Sz_trans()
        Sx_pol = abs(np.dot(np.dot(rho_t.T, self.Sx_trans), rho_t))
        Sy_pol = abs(np.dot(np.dot(rho_t.T, self.Sy_trans), rho_t))
        Sz_pol = abs(np.dot(np.dot(rho_t.T, self.Sz_trans), rho_t))
        intens_x = a1*Sx_pol.flatten()
        intens_y = a2*Sy_pol.flatten()
        intens_z = a3*Sz_pol.flatten()
        if freq1 is not None and freq2 is not None:
            freli = np.linspace(freq1, freq2, steps)
        else:
            freli = np.arange(2860., 2890., 0.01)
        signal = np.zeros_like(freli)
        for i, fr in enumerate(freq):
            yx = lorentz(freli, fr, lw, -np.sqrt(abs(intens_x[i])**2+abs(intens_y[i])**2+abs(intens_z[i])**2))
            signal += yx
        return signal
                
    def hyperfine_odmr_plus(self, iz=1, lw=0.1, freq1=None, freq2=None, steps=1000, a1=1., a2=1., a3=1.):
        """
        gives ODMR between freq1 and freq2. The linewidth is given bei lw
        and steps is the number of frequencies between freq1 and freq2.
        The contrast of the lines is given by sqrt(intens_x**2+intens_y**2)
        The calculation takes all transitions into account.
        The nuclear spin is polarized into the state iz.
        Additionally, the difference in parts of ms=0 between two states is taken into account (linearly multiplied)
        The MW polarization can be given by a1, a2 and a3 for x, y and z
        """
        rho = {1:np.diag([1+0.j, 0, 0]), 0:np.diag([0+0.j, 1, 0]), -1:np.diag([0+0.j, 0, 1])}[iz]
        rho = np.kron(np.eye(3, dtype=complex), rho)
        rho_t = np.dot(np.dot(self.state_i, rho), self.state)
        num_t = sum(range(4))
        freq = self.frequency.flatten()
        fluor = self.fluorescence.flatten()
        self.get_Sz_trans()
        #ms0 = np.dot(np.dot(self.state_i, np.kron(np.diag([0, 1, 0]), np.eye(3))), self.state)
        Sx_pol = abs(np.dot(np.dot(rho_t.T, self.Sx_trans), rho_t))
        Sy_pol = abs(np.dot(np.dot(rho_t.T, self.Sy_trans), rho_t))
        Sz_pol = abs(np.dot(np.dot(rho_t.T, self.Sz_trans), rho_t))
        intens_x = a1*Sx_pol.flatten()
        intens_y = a2*Sy_pol.flatten()
        intens_z = a3*Sz_pol.flatten()
        if freq1 is not None and freq2 is not None:
            freli = np.linspace(freq1, freq2, steps)
        else:
            freli = np.arange(2860., 2890., 0.01)
        signal = np.zeros_like(freli)
        for i, fr in enumerate(freq):
            yx = lorentz(freli, fr, lw, -np.sqrt(abs(intens_x[i])**2+abs(intens_y[i])**2+abs(intens_z[i])**2)*fluor[i])
            signal += yx
        return signal

    def diffE(self, bz = 0.1, bx=10, ex=0.1):
        """
        gives energy difference between no E-Field and E-Field of ex
        """
        bzalt = self.bz
        bxalt = self.bx
        byalt = self.by
        exalt = self.ex
        eyalt = self.ey
        ezalt = self.ez
        self.set_bz(bz)
        self.set_bx(bx)
        self.set_by(0)
        self.set_ex(0)
        self.set_ez(0)
        self.set_ey(0)
        en1 = self.energy
        self.set_ex(ex)
        en2 = self.energy
        self.set_ex(exalt)
        self.set_ey(eyalt)
        self.set_ez(eyalt)
        self.set_bx(bxalt)
        self.set_by(byalt)
        self.set_bz(bzalt)
        return (en1[2]-en1[0])-(en2[2]-en2[0])

    def lac_pl(self, axis='bx', bstart=-4, bstop=4, n=1000):
        """
        old calculation of pl by giving the amound of pop in ms=0?
        """
        b0 = {'bx':self.bx, 'by':self.by, 'bz':self.bz}[axis]
        fun = {'bx':self.set_bx, 'by':self.set_by, 'bz':self.set_bz}[axis]
        x = np.linspace(bstart, bstop, n)
        rho_1 = {True:1/np.sqrt(3)*np.diag([0,0,0,1,1,1,0,0,0]), False:np.diag([0, 1, 0])}[self.hyperfine]
        #ms0 = {False:1, True:4}[self.hyperfine]
        if axis=='bz':
                x+=bz_lac
        y = np.zeros((n))
        for i in range(n):
                fun(x[i])
                z_base=np.dot(self.state, np.dot(np.diag(np.diag(np.dot(self.state_i, np.dot(rho_1, self.state)))), self.state_i))
                if self.hyperfine:
                    y[i]=abs(z_base[3,3])+abs(z_base[4,4])+abs(z_base[5,5])
                else:
                    y[i]=abs(z_base[1,1])
        fun(b0)
        return y

    def testcoil(self, db=0.01, axis='bx'):
        """
        old calculation of change in fluorescence with magn. field change db
        """
        b0 = {'bx':self.bx, 'by':self.by, 'bz':self.bz}[axis]
        fun = {'bx':self.set_bx, 'by':self.set_by, 'bz':self.set_bz}[axis]
        rho_1 = {True:1/np.sqrt(3)*np.diag([0,0,0,1,1,1,0,0,0]), False:np.diag([0, 1, 0])}[self.hyperfine]
        fun(b0-db)
        z_base=np.dot(self.state, np.dot(np.diag(np.diag(np.dot(self.state_i, np.dot(rho_1, self.state)))), s.state_i))
        if s.hyperfine:
                y0=abs(z_base[3,3])+abs(z_base[4,4])+abs(z_base[5,5])
        else:
                y0=abs(z_base[1,1])
        fun(b0+db)
        z_base=np.dot(self.state, np.dot(np.diag(np.diag(np.dot(self.state_i, np.dot(rho_1, self.state)))), self.state_i))
        if s.hyperfine:
                y1=abs(z_base[3,3])+abs(z_base[4,4])+abs(z_base[5,5])
        else:
                y1=abs(z_base[1,1])
        fun(b0)
        return y0-y1


    def fluor(self, keg=66., kes0=8., kes1=53., ksg0=0.975, ksg1=0.725, rge=100, states=False, t1n=0.001, t1e=0):
        """
        gives fluorescence of the NV with the appl. field
        """
        st = self.rate_rot()
        stt = st.T
        mat = {True:pmat_hf(keg, kes0, kes1, ksg0, ksg1, rge, t1n), False:pmat(keg, kes0, kes1, ksg0, ksg1, rge)}[self.hyperfine]
        mt = np.dot(np.dot(stt, mat), st)
        mt[0,2]+=t1e
        mt[2,0]+=t1e
        mt = mt-np.diag(mt.sum(0))
        x1={True:np.zeros(20), False:np.zeros(6)}[self.hyperfine]
        if self.hyperfine:
            x1[-1]=1./3.
            x1[-2]=1./3.
        sol = minimize(minprob, x1, method='SLSQP', bounds=[(0.,1) for x in xrange(len(x1))], args = mt)
        p = {False: np.zeros((7)), True:np.zeros((21))}[self.hyperfine]
        p[:-1]=sol['x']
        p[-1]=1-sol['x'].sum()
        if not self.hyperfine:
            if states:
                return p
            else:
                return p[3:6].sum()*keg
        else:
            if states:
                return p
            else:
                return p[9:18].sum()*keg



    def fluors(self, start=980, stop=1060, nbz=300, keg=66., kes0=8., kes1=53., ksg0=0.975, ksg1=0.725, rge=100, states=False, t1n=0.001, t1e=0.):
        """
        gives fluorescence of the NV with the appl. field ramping from Bz=start to Bz=stop
        in nbz steps
        with hyperfine: rate t1n as nuclear relax time so that the nuclear spin gets equalled.
        """
        bzlist = np.linspace(start, stop, nbz)
        res_f = np.zeros((nbz))
        x1={True:np.zeros(20), False:np.zeros(6)}[self.hyperfine]
        p0 = np.zeros((len(x1)+1))
        if self.hyperfine:
            p0[-1]=1./3.
            p0[-2]=1./3.
            #p0[-3]=1./3.
        res_s = {True:np.zeros((nbz, 21)), False:np.zeros((nbz, 7))}[self.hyperfine]
        for iz, bz in enumerate(bzlist):
            self.set_bz(bz)
            st = self.rate_rot()
            stt = st.T
            x1 = np.dot(stt, p0)[:-1]
            mat = {True:pmat_hf(keg, kes0, kes1, ksg0, ksg1, rge, t1n), False:pmat(keg, kes0, kes1, ksg0, ksg1, rge)}[self.hyperfine]
            mt = np.dot(np.dot(stt, mat), st)
            mt[0,2]+=t1e
            mt[2,0]+=t1e
            mt = mt-np.diag(mt.sum(0))
            sol = minimize(minprob, x1, method='SLSQP', args=mt, bounds=[(0.,1) for x in xrange(len(x1))])
            p = {False: np.zeros((7)), True:np.zeros((21))}[self.hyperfine]
            if sol['message']!='Optimization terminated successfully.':
                print sol['message'], bz, iz
            p[:-1]=sol['x']
            p[-1]=1-sol['x'].sum()
            x1 = p[:-1]
            res_f[iz]={False:p[3:6].sum()*keg, True:p[9:18].sum()*keg}[self.hyperfine]
            res_s[iz]=p
            p0=np.dot(stt, p)
        return {False:res_f, True:res_s}[states]

    def fluors2(self, start=980, stop=1060, nbz=300, keg=66., kes0=8., kes1=53., ksg0=0.975, ksg1=0.725, rge=100, states=False, t1n=0.001, t1e=0.):
        """
        gives fluorescence of the NV with the appl. field ramping from Bz=start to Bz=stop
        in nbz steps
        with hyperfine: rate t1n as nuclear relax time so that the nuclear spin gets equalled.
        """
        bzlist = np.linspace(start, stop, nbz)
        res_f = np.zeros((nbz))
        x1={True:np.zeros(20), False:np.zeros(6)}[self.hyperfine]
        p0 = np.zeros((len(x1)+1))
        if self.hyperfine:
            p0[-1]=1./3.
            p0[-2]=1./3.
        res_s = {True:np.zeros((nbz, 21)), False:np.zeros((nbz, 7))}[self.hyperfine]
        for iz, bz in enumerate(bzlist):
            self.set_bz(bz)
            st = self.rate_rot()
            stt = st.T
            x1 = np.dot(stt, p0)[:-1]
            mat = {True:pmat_hf(keg, kes0, kes1, ksg0, ksg1, rge, t1n), False:pmat(keg, kes0, kes1, ksg0, ksg1, rge)}[self.hyperfine]
            mt = np.dot(np.dot(stt, mat), st)
            mt[0,2]+=t1e
            mt[2,0]+=t1e
            mt = mt-np.diag(mt.sum(0))
            sol = minimize(minprob, x1, method='SLSQP', args=mt, bounds=[(0.,1) for x in xrange(len(x1))], options={'maxiter':1000, 'ftol':1.e-7})
            p = {False: np.zeros((7)), True:np.zeros((21))}[self.hyperfine]
            if sol['message']!='Optimization terminated successfully.':
                print sol['message'], bz, iz
            p[:-1]=sol['x']
            p[-1]=1-sol['x'].sum()
            x1 = p[:-1]
            res_f[iz]={False:p[3:6].sum()*keg, True:p[9:18].sum()*keg}[self.hyperfine]
            res_s[iz]=p
            p0=np.dot(stt, p)
        return {False:res_f, True:res_s}[states]
            
    def fluor_t1(self, keg=66., kes0=8., kes1=53., ksg0=0.975, ksg1=0.725, rge=100, t1=0.005, states=False):
        """
        gives fluorescence of NV with T1, no hyperfine
        """
        if self.hyperfine:
            return np.array([])
        st = self.rate_rot()
        stt = st.T
        mat = pmat_t1(keg, kes0, kes1, ksg0, ksg1, rge, t1)
        mt = np.dot(np.dot(stt, mat), st)
        mt = mt-np.diag(mt.sum(0))
        sol = minimize(minprob, np.zeros((6)), method='SLSQP', bounds=[(0.,1) for x in xrange(6)], args = mt)
        p = np.zeros((7))
        p[:-1]=sol['x']
        p[-1]=1-sol['x'].sum()
        if states:
                return p
        else:
                return p[3:6].sum()*keg

            
    def rate_rot(self):
        """
        rotation matrix for rate equation
        """
        matr = {False:np.zeros((7, 7)), True:np.zeros((21, 21))}[self.hyperfine]
        multi = {False:1, True:3}[self.hyperfine]
        for i in range(multi*3):
            for k in range(multi*3):
                    matr[i, k]=abs(self.state[i, k])**2
                    matr[i+3*multi, k+3*multi]=abs(self.state_es[i, k])**2
        matr[-1,-1]=1
        if self.hyperfine:
            matr[-2, -2]=1
            matr[-3, -3]=1
        return matr

import os

path = os.getcwd()


def pmat(keg, kes0, kes1, ksg0, ksg1, rge):
    """
    rate matrix w/o t1, no hyperfine
    """
    arr = np.array([[0, 0, 0, keg, 0, 0, ksg1], [0, 0, 0, 0, keg, 0, ksg0], [0, 0, 0, 0, 0, keg,ksg1], [rge, 0, 0, 0, 0, 0, 0], [0, rge, 0, 0, 0, 0, 0], [0, 0, rge, 0, 0, 0, 0], [0, 0, 0, kes1, kes0, kes1, 0]], dtype=float)
    return arr	

def pmat_hf(keg, kes0, kes1, ksg0, ksg1, rge, t1n):
    """
    rate matrix with t1 (MW), hyperfine 
    """
    arrs = pmat(keg, kes0, kes1, ksg0, ksg1, rge)
    arrn = np.array([[0, t1n, t1n], [t1n, 0, t1n], [t1n, t1n, 0]])
    return np.kron(arrs, np.eye(3))#+np.kron(np.eye(7), arrn)

def pmat_t1(keg, kes0, kes1, ksg0, ksg1, rge, t1):
    """
    rate matrix with t1, no hyperfine
    """
    arr = np.array([[0, t1, 0, keg, 0, 0, ksg1], [t1, 0, t1, 0, keg, 0, ksg0], [0, t1, 0, 0, 0, keg,ksg1], [rge, 0, 0, 0, 0, 0, 0], [0, rge, 0, 0, 0, 0, 0], [0, 0, rge, 0, 0, 0, 0], [0, 0, 0, kes1, kes0, kes1, 0]])
    return arr

def expand_1(x):
    """
    expand vector x by one part which is 1-x.sum()
    """
    if len(x)==6:
        x1 = np.zeros((len(x)+1))
        x1[:len(x)]=x
        x1[-1]=1-x.sum()
    else:
        x1 = np.zeros((len(x)+1))
        x1[:len(x)]=x
        x1[-1]=1.-x.sum()
    return x1

def minprob(x, mat):
    """
    minimum finder for rate equation
    """
    res = np.linalg.norm(np.dot(mat, expand_1(x)))
    return res

def give_zeeman_shift(num, bx, bz):
    """
    gives Zeeman-Shift of the num strongest transitions for
    bx and bz
    """
    s=System()
    s.set_bx(bx)
    s.set_bz(bz)
    trans=s.get_transitions()
    t1=[trans[n] for n in num]
    return t1
    
fit_b=lambda p, x: give_zeeman_shift(x, p[0], p[1])
err_b=lambda p, x, y: fit_b(p, x)-y

def get_b(t1, t2):
    """
    gives transversal and axial magnetic field for two transition frequencies (single NV)
    """
    y=np.array([t1, t2])
    x=np.array([1,2])
    p0=[10., (t2-t1)/5.6]
    p1, success=optimize.leastsq(err_b, p0[:], args=(x, y), maxfev=5000)
    f1=fit_b(p1, x)
    b_tot=np.sqrt(p1[0]**2+p1[1]**2)
    return p1, f1, b_tot

def get_3_orientations(bx, by, bz):
    """
    gives magnetic fields felt by the three non-111 directions
    for the 
    """
    bz1 = 0.5*(np.sqrt(3)*bx-by)
    bx1 = 0.5*(bx+np.sqrt(3)*by)
    bz2 = -0.5*(np.sqrt(3)*bx+by)
    bx2 = 0.5*(bx-np.sqrt(3)*by)
    bz3 = by
    bx3 = -bx

    cos = np.cos(19.5*np.pi/180.)
    sin = np.sin(19.5*np.pi/180.)
    bz1a = cos*bz1-sin*bz
    by1a = sin*bz1+cos*bz
    bz2a = cos*bz2-sin*bz
    by2a = sin*bz2+cos*bz
    bz3a = cos*bz3-sin*bz
    by3a = sin*bz3+cos*bz

    return [bx1, by1a, bz1a, bx2, by2a, bz2a, bx3, by3a, bz3a]

def get_3_orientations_xyrev(bx, by, bz):
    """
    gives magnetic fields felt by the three non-111 directions
    for the 
    """
    cos = np.cos(19.5*np.pi/180.)
    sin = np.sin(19.5*np.pi/180.)
    
    bx1 = bx*sin+bz*cos
    by1 = -by
    bz1 = bx*cos-sin*bz
    
    bx2 = 0.5*(np.sqrt(3)*bx-by)
    by2 = bz*cos-sin*0.5*(np.sqrt(3)*by+bx)
    bz2 = -bz*sin-0.5*(np.sqrt(3)*by+bx)*cos
    
    bx3 = 0.5*(np.sqrt(3)*bx+by)
    by3 = bz*cos+0.5*(np.sqrt(3)*by-bx)*sin
    bz3 = -bz*sin+0.5*(np.sqrt(3)*by-bx)*cos
    
    return [bx1, by1, bz1, bx2, by2, bz2, bx3, by3, bz3]

def get_3_orientations_old(bx, by, bz):
    bz1 = 0.5*(np.sqrt(3)*bx-by)
    bx1 = 0.5*(bx+np.sqrt(3)*by)
    bz2 = -0.5*(np.sqrt(3)*bx+by)
    bx2 = 0.5*(bx-np.sqrt(3)*by)
    bz3 = by
    bx3 = -bx

    bz1a = 0.5*(np.sqrt(3)*bz1-bz)
    by1a = 0.5*(bz1+np.sqrt(3)*bz)
    bz2a = 0.5*(np.sqrt(3)*bz2-bz)
    by2a = 0.5*(bz2+np.sqrt(3)*bz)
    bz3a = 0.5*(np.sqrt(3)*bz3-bz)
    by3a = 0.5*(bz3+np.sqrt(3)*bz)

    return [bx1, by1a, bz1a, bx2, by2a, bz2a, bx3, by3a, bz3a]

def set_3_orientations(num, bx, by, bz, d=2870.5):
    """
    sets a magnetic field which is purely transversal (bz=0)
    """
    bz=0
    s1=System(False)
    s2=System(False)
    s3=System(False)
    bf = get_3_orientations(bx, by, bz)
    s1.set_all_b(bf[0], bf[1], bf[2])
    s1.set_bx(bf[0])
    s1.set_by(bf[1])
    s1.set_bz(bf[2])
    s1.set_d(d)
    s2.set_bx(bf[3])
    s2.set_by(bf[4])
    s2.set_bz(bf[5])
    s2.set_d(d)
    s3.set_bx(bf[6])
    s3.set_by(bf[7])
    s3.set_bz(bf[8])
    s3.set_d(d)
    t1 = sorted(s1.get_transitions(n=3))[::-1][:-1]
    t2 = sorted(s2.get_transitions(n=3))[::-1][:-1]
    t3 = sorted(s3.get_transitions(n=3))[::-1][:-1]
    tges = [t1[0], t1[1], t2[0], t2[1], t3[0], t3[1]]
    treturn = [tges[i] for i in num]
    #print bx, by, bz
    return treturn

def set_3_orientations2(num, bx, by, bz, d=2870.5):
    #bz=0
    s1=System(False)
    s2=System(False)
    s3=System(False)
    s4=System(True)
    s4.delete_h(s4.H_T)
    bf = get_3_orientations(bx, by, bz)
    s1.set_bx(bf[0])
    s1.set_by(bf[1])
    s1.set_bz(bf[2])
    s1.set_d(d)
    s2.set_bx(bf[3])
    s2.set_by(bf[4])
    s2.set_bz(bf[5])
    s2.set_d(d)
    s3.set_bx(bf[6])
    s3.set_by(bf[7])
    s3.set_bz(bf[8])
    s3.set_d(d)
    s4.set_bx(bx)
    s4.set_by(by)
    s4.set_bz(bz)
    s4.set_d(d)
    t1 = s1.get_transitions(n=2)
    t2 = s2.get_transitions(n=2)
    t3 = s3.get_transitions(n=2)
    t4 = s4.get_transitions(n=6)
    tges = [t1[0], t1[1], t2[0], t2[1], t3[0], t3[1], t4[0], t4[1], t4[4], t4[5]]
    treturn = [tges[i] for i in num]
    #print bx, by, bz
    return treturn

def set_3_orientations3(num, bx, by, bz, d=2870.5):
    #bz=0
    s1=System(False)
    s2=System(False)
    s3=System(False)
    s4=System(False)
    #s4.delete_h(s4.H_T)
    bf = get_3_orientations(bx, by, bz)
    s1.set_bx(bf[0])
    s1.set_by(bf[1])
    s1.set_bz(bf[2])
    s1.set_d(d)
    s2.set_bx(bf[3])
    s2.set_by(bf[4])
    s2.set_bz(bf[5])
    s2.set_d(d)
    s3.set_bx(bf[6])
    s3.set_by(bf[7])
    s3.set_bz(bf[8])
    s3.set_d(d)
    s4.set_bx(bx)
    s4.set_by(by)
    s4.set_bz(bz)
    s4.set_d(d)
    t1 = s1.get_transitions(n=2)
    t2 = s2.get_transitions(n=2)
    t3 = s3.get_transitions(n=2)
    t4 = s4.get_transitions(n=2)
    tges = [t1[0], t1[1], t2[0], t2[1], t3[0], t3[1], t4[0], t4[1]]
    treturn = [tges[i] for i in num]
    #print bx, by, bz
    return treturn

def set_system_system(bx, by, bz, d, hyperfine):
    s1=System(hyperfine)
    s2=System(hyperfine)
    s3=System(hyperfine)
    s4=System(hyperfine)
    bf = get_3_orientations(bx, by, bz)
    s1.set_bx(bf[0])
    s1.set_by(bf[1])
    s1.set_bz(bf[2])
    s1.set_d(d)
    s2.set_bx(bf[3])
    s2.set_by(bf[4])
    s2.set_bz(bf[5])
    s2.set_d(d)
    s3.set_bx(bf[6])
    s3.set_by(bf[7])
    s3.set_bz(bf[8])
    s3.set_d(d)
    s4.set_bx(bx)
    s4.set_by(by)
    s4.set_bz(bz)
    s4.set_d(d)
    #treturn = [tges[i] for i in num]
    #print bx, by, bz
    return s1, s2, s3, s4

def create_mean(l):
    """
    Calculate the mean of the array if some entries are nan
    """
    clean_l=[]
    for i in l:
        if not np.isnan(i):
            clean_l.append(i)
    mean = np.array(clean_l).mean()
    return mean

err_3orient=lambda p, x, y: set_3_orientations(x, p[0], p[1], p[2], p[3])-y
err_3orient2=lambda p, x, y: set_3_orientations2(x, p[0], p[1], p[2], p[3])-y
err_3orient3=lambda p, x, y: set_3_orientations3(x, p[0], p[1], p[2], p[3])-y
err_3orient4=lambda p, x, y: set_3_orientations3(x, p[0], p[1], 0, p[3])-y


#def err_3orient(p, x, y):
#    print x, p, y
#    set_3_orientations(x, p[0], p[1], p[2], p[3])-y

def odmr_all(freq1=2800., freq2=2940., n=1000, lw=0.1, hf=True, bx=0, by=0, bz=0, c1=1, c2=1, c3=1, c4=1):
    s1 = System(hf)
    s2 = System(hf)
    s3 = System(hf)
    s4 = System(hf)
    bf = get_3_orientations(bx, by, bz)
    s1.set_all_b(bf[0], bf[1], bf[2])
    s2.set_all_b(bf[3], bf[4], bf[5])
    s3.set_all_b(bf[6], bf[7], bf[8])
    s4.set_all_b(bx, by, bz)
    od1 = c1*s1.odmr2(freq1=freq1, freq2=freq2, steps=n)
    od1 += c2*s2.odmr2(freq1=freq1, freq2=freq2, steps=n)
    od1 += c3*s3.odmr2(freq1=freq1, freq2=freq2, steps=n)
    od1 += c4*s4.odmr2(freq1=freq1, freq2=freq2, steps=n)
    freq = np.linspace(freq1, freq2, n)
    return od1, freq

def odmr_plot_separate(freq1=2800., freq2=2940., n=1000, lw=0.1, hf=True, bx=0, by=0, bz=0, c1=1, c2=1, c3=1, c4=1):
    s1 = System(hf)
    s2 = System(hf)
    s3 = System(hf)
    s4 = System(hf)
    bf = get_3_orientations(bx, by, bz)
    s1.set_all_b(bf[0], bf[1], bf[2])
    s2.set_all_b(bf[3], bf[4], bf[5])
    s3.set_all_b(bf[6], bf[7], bf[8])
    s4.set_all_b(bx, by, bz)
    od1 = c1*s1.odmr2(freq1=freq1, freq2=freq2, steps=n)
    od2 = c2*s2.odmr2(freq1=freq1, freq2=freq2, steps=n)
    od3 = c3*s3.odmr2(freq1=freq1, freq2=freq2, steps=n)
    od4 = c4*s4.odmr2(freq1=freq1, freq2=freq2, steps=n)
    freq = np.linspace(freq1, freq2, n)
    plt.plot(freq, od1, 'g')
    plt.plot(freq, od2, 'c')
    plt.plot(freq, od3, 'y')
    plt.plot(freq, od4, 'b')
    
    
    
def plot_all(pt=100, n=2, bx1=0, bx2=0, by1=0, by2=0, bz1=0, bz2=0, hf=True, xaxis=None):
    bx = np.linspace(bx1, bx2, pt)
    by = np.linspace(by1, by2, pt)
    bz = np.linspace(bz1, bz2, pt)
    s1 = System(hf)
    s2 = System(hf)
    s3 = System(hf)
    s4 = System(hf)
    bf1 = get_3_orientations(bx1, by1, bz1)
    bf2 = get_3_orientations(bx2, by2, bz2)
    arr1 = np.array([np.linspace(bf1[0], bf2[0], pt), np.linspace(bf1[1], bf2[1], pt), np.linspace(bf1[2],bf2[2], pt)])
    arr2 = np.array([np.linspace(bf1[3], bf2[3], pt), np.linspace(bf1[4], bf2[4], pt), np.linspace(bf1[5],bf2[5], pt)])
    arr3 = np.array([np.linspace(bf1[6], bf2[6], pt), np.linspace(bf1[7], bf2[7], pt), np.linspace(bf1[8],bf2[8], pt)])
    arr4 = np.array([bx, by, bz])
    if xaxis==None:
        xaxis=bz
    x, y1 = s1.plot_barray(arr1, what=2, n=n, plot=False, give=True)
    x, y2 = s2.plot_barray(arr2, what=2, n=n, plot=False, give=True)
    x, y3 = s3.plot_barray(arr3, what=2, n=n, plot=False, give=True)
    x, y4 = s2.plot_barray(arr4, what=2, n=n, plot=False, give=True)
    cl=['b', 'g', 'r', 'c']
    ylist = [y1, y2, y3, y4]
    for i in range(4):
        for k in range(len(y1[0])):
            plt.plot(xaxis, ylist[i][:,k], cl[i])

def create_b_list(bx1=0, bx2=0, by1=0, by2=0, bz1=0, bz2=0, pt=100):
    barray = np.array([np.linspace(bx1, bx2, pt), np.linspace(by1, by2, pt), np.linspace(bz1, bz2, pt)])
    return barray
        
def plot_all_save(pt=100, n=2, bx1=0, bx2=0, by1=0, by2=0, bz1=0, bz2=0, hf=True, xaxis=None, filename='plottest'):
    bx = np.linspace(bx1, bx2, pt)
    by = np.linspace(by1, by2, pt)
    bz = np.linspace(bz1, bz2, pt)
    s1 = System(hf)
    s2 = System(hf)
    s3 = System(hf)
    s4 = System(hf)
    bf1 = get_3_orientations(bx1, by1, bz1)
    bf2 = get_3_orientations(bx2, by2, bz2)
    arr1 = np.array([np.linspace(bf1[0], bf2[0], pt), np.linspace(bf1[1], bf2[1], pt), np.linspace(bf1[2],bf2[2], pt)])
    arr2 = np.array([np.linspace(bf1[3], bf2[3], pt), np.linspace(bf1[4], bf2[4], pt), np.linspace(bf1[5],bf2[5], pt)])
    arr3 = np.array([np.linspace(bf1[6], bf2[6], pt), np.linspace(bf1[7], bf2[7], pt), np.linspace(bf1[8],bf2[8], pt)])
    arr4 = np.array([bx, by, bz])
    if xaxis==None:
        xaxis=bz
    x, y1 = s1.plot_barray(arr1, what=2, n=n, plot=False, give=True)
    x, y2 = s2.plot_barray(arr2, what=2, n=n, plot=False, give=True)
    x, y3 = s3.plot_barray(arr3, what=2, n=n, plot=False, give=True)
    x, y4 = s2.plot_barray(arr4, what=2, n=n, plot=False, give=True)
    ylist = [y1, y2, y3, y4]
    cl=['b', 'g', 'r', 'c']
    for i in range(4):
        for k in range(len(y1[0])):
            plt.plot(xaxis, ylist[i][:,k], cl[i])
    plt.savefig(filename+'.png')

def fit_b(t1, t2, t3, t4, t5, t6, p0=[20., 5., 0., 2870.0]):
    """
    fits magnetic field for transition freqs t1,...,t6
    fourth transition is set to transversal
    p0 = [Bx, By, Bz, D]
    """
    y=np.array([t1, t2, t3, t4, t5, t6])
    x=np.array(range(6))
    
    p1, success=optimize.leastsq(err_3orient, p0[:], args=(x, y), maxfev=5000)
    f1=set_3_orientations(x, p1[0], p1[1], p1[2], p1[3])
    b_tot=np.sqrt(p1[0]**2+p1[1]**2+p1[2]**2)
    return p1, f1, b_tot

def fit_b2(t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, p0=[20., 5., 0., 2870.0]):
    y=np.array([t1, t2, t3, t4, t5, t6, t7, t8, t9, t10])
    x=np.array(range(10))
    
    p1, success=optimize.leastsq(err_3orient2, p0[:], args=(x, y), maxfev=5000)
    f1=set_3_orientations2(x, p1[0], p1[1], p1[2], p1[3])
    b_tot=np.sqrt(p1[0]**2+p1[1]**2+p1[2]**2)
    return p1, f1, b_tot

def fit_b3(t1a, t1b, t2a, t2b, t3a, t3b, t4a, t4b, p0=[20., 5., 0., 2870.0]):
    y=np.array([t1a, t1b, t2a, t2b, t3a, t3b, t4a, t4b])
    x=np.array(range(8))
    
    p1, success=optimize.leastsq(err_3orient3, p0[:], args=(x, y), maxfev=5000)
    f1=set_3_orientations3(x, p1[0], p1[1], p1[2], p1[3])
    b_tot=np.sqrt(p1[0]**2+p1[1]**2+p1[2]**2)
    return p1, f1, b_tot

def fit_b4(t1a, t1b, t2a, t2b, t3a, t3b, p0=[20., 5., 0., 2870.0]):
    y=np.array([t1a, t1b, t2a, t2b, t3a, t3b])
    x1=np.array(range(6))
    x2 = np.array(range(8))
    
    p1, success=optimize.leastsq(err_3orient3, p0[:], args=(x1, y), maxfev=5000)
    f1=set_3_orientations3(x2, p1[0], p1[1], p1[2], p1[3])
    b_tot=np.sqrt(p1[0]**2+p1[1]**2+p1[2]**2)
    return p1, f1, b_tot

def fit_b5(t1a, t1b, t2a, t2b, t3a, t3b, p0=[20., 5., 0., 2870.0]):
    y=np.array([t1a, t1b, t2a, t2b, t3a, t3b])
    x1=np.array(range(6))
    x2 = np.array(range(8))
    
    p1, success=optimize.leastsq(err_3orient4, p0[:], args=(x1, y), maxfev=5000)
    f1=set_3_orientations3(x2, p1[0], p1[1], p1[2], p1[3])
    b_tot=np.sqrt(p1[0]**2+p1[1]**2+p1[2]**2)
    return p1, f1, b_tot

def fit_measurement(dic):
    #f=open(filename)
    #dic=pickle.load(f)
    #f.close()
    y=dic['y']*1e-6
    n_meas = len(y[0])
    #print n_meas
    times = dic['time_diff']
    if len(times)>n_meas:
        times=times[:n_meas]
    bx_values=np.zeros((n_meas))
    by_values = np.zeros((n_meas))
    d_values = np.zeros((n_meas))
    p0=[10., 0., 0., 2869]
    for i in range(n_meas):
        y1=y.T[i]
        try:
            p1, f1, b_tot = fit_b(y1[5], y1[20], y1[8], y1[17], y1[11], y1[14], p0 )
        except:
            p1=[None, None, None, None]
        bx_values[i]=p1[0]
        by_values[i]=p1[1]
        d_values[i]=p1[3]
        if i==0:
            print p1
    return {'y':y, 'times':times, 'bx_values':bx_values, 'by_values':by_values, 'd_values':d_values}

def give_df(dic):
    y=dic['y']
    bx_values=dic['bx_values']
    by_values=dic['by_values']
    d_values=dic['d_values']
    n_meas=len(bx_values)
    s4=System(False)
    df1=np.zeros((n_meas))
    df2=np.zeros((n_meas))
    for i in range(n_meas):
        try:
            s4.set_d(d_values[i])
            s4.set_bx(bx_values[i])
            s4.set_by(by_values[i])
            f=s4.get_transitions(n=2)
            y1=y.T[i]
            df1[i]=f[0]-y1[1]
            df2[i]=f[1]-y1[2]
        except:
            df1[i]=None
            df2[i]=None
    dic['df1']=df1
    dic['df2']=df2
    return dic

def get_fit_error(t_list1, t_list2):
    err=0
    for i, ti in enumerate(sorted(t_list1)):
       err += abs(ti-sorted(t_list2)[i])
    return abs(err)

def get_b_field(t_list, phi_guess=10., strength_guess=30., nv_order=[0, 1, 2]):
    phi_guess=phi_guess*np.pi/180.
    x_guess = strength_guess*np.cos(phi_guess)
    y_guess = strength_guess*np.sin(phi_guess)
    tl = [t_list[nv_order[0]], t_list[5-nv_order[0]], t_list[nv_order[1]], t_list[5-nv_order[1]], t_list[nv_order[2]], t_list[5-nv_order[2]]]
    p1, f1, b_tot=fit_b(tl[0], tl[1], tl[2], tl[3], tl[4], tl[5], p0=[x_guess, y_guess, 0, 2869.])
    err = get_fit_error(tl, f1)
    fit_list = [[x_guess, -y_guess], [-x_guess, y_guess], [-x_guess, -y_guess], [y_guess, x_guess], [y_guess, -x_guess], [-y_guess, x_guess], [-y_guess, x_guess], [20., 5], [20, -5], [-20, 5], [5, 20], [5, -20], [-5, 20]]
    i_guess = 0
    while err>1.:
        p1, f1, b_tot=fit_b(tl[0], tl[1], tl[2], tl[3], tl[4], tl[5], p0=[fit_list[i_guess][0], fit_list[i_guess][1], 0, 2869.])
        err = get_fit_error(tl, f1)
        i_guess+=1
        if i_guess>=len(fit_list):
            print 'did not work'
            break
    return p1, f1, b_tot
    
def arctan(y, x):
	if y>0 and x>0:
		return np.arctan(y/x)
	if x<0:
		return np.arctan(y/x)+np.pi
	if x>0 and y<0:
		return np.arctan(y/x)+np.pi*2.

def clean_up(l, cut=0.01):
    mean = create_mean(l)
    for i_n, fr in enumerate(l):
        if abs(mean-fr)>cut:
            l[i_n]=None
    return l

def plot_shift(dic, num=1):
    fig = plt.figure()
    ax=fig.add_subplot(111)
    ax.set_xlabel('Time [h]')
    ax.set_ylabel('Shift [kHz]')
    shift = {1:dic['df1'], 2:dic['df2']}[num]
    ax.plot(np.array(dic['times'])/3600., shift*1e3)
    plt.show()


def odmr_plot_bz(freq1=0.3, freq2=30, n1=400, n2=1000, mwx=1., mwy=1., mwz=1., izm1=0.1, iz0=0.1, izp1=1., bz1=-4, bz2=4):
    bzlist = np.linspace(bz1, bz2, n1)


def b_zylinder(z, Br=10000, D=0.03, R=0.035):
	bfield = Br*(((D+z)/(np.sqrt(R**2+(D+z)**2))-(z/(np.sqrt(R**2+z**2)))))/2.
	return bfield

    
def read_pic(filename, cmap='jet'):
    from PIL import Image
    img = Image.open(filename).convert('RGBA')
    im2 = np.array(img)
    from matplotlib import cm
    vals = np.zeros((255, 3))
    colormap = {'jet':cm.jet, 'viridis':cm.viridis}[cmap]
    for i in range(255):
        vals[i]=colormap(i)[:3]
    def distance(a, b):
        return np.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2+(a[2]-b[2])**2)
    c1, c2, c3 = im2.shape
    imn = np.zeros_like(im2)
    for i in range(c1):
        for k in range(c2):
            a = im2[i,k]
            distl = [np.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2+(a[2]-b[2])**2) for b in vals]
            imn[i,k]=distl.index(min(distl))
    return imn

