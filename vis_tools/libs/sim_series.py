import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import functools
try:
    import numba
except:
    print("[sim.py] numba not found! Currently ok.")
import scipy.ndimage as ndimage
import os
from load_c import *
from datetime import datetime
from IPython.display import clear_output
from IPython.utils import io
import glob
    
class Snapshot: 
    def __init__(self, parent, dump_no):
        '''
        Parameters
        dump_no                 with dump file to load (integer)
        '''
        
        self.parent  = parent
        self.dump_no = dump_no
        self._readParamData()
        
    '''
    A little tedious, but for all properties that are defined in Simulation rather than Snapshot, the following will allow @properties will be allowed for them to be called with eg the short cut self.x rather than self.parent.x. 
    For consistency, new Simulation properties that are desired to be ran inside Snapshot should be given the same treatment. 
    '''
    @property
    def sim_dir(self):
        return self.parent.sim_dir
    @property
    def DISK_THICKNESS(self):
        return self.parent.DISK_THICKNESS
    @property
    def lowres1(self):
        return self.parent.lowres1
    @property
    def lowres2(self):
        return self.parent.lowres2
    @property
    def lowres3(self):
        return self.parent.lowres3
    @property
    def axisym(self):
        return self.parent.axisym
    @property
    def flatten(self):
        return self.parent.flatten
    @property
    def check_files(self):
        return self.parent.check_files
    @property
    def export_raytracing(self):
        return self.parent.export_raytracing
    @property
    def interpolate_var(self):
        return self.parent.interpolate_var
    @property
    def do_box(self):
        return self.parent.do_box
    @property
    def AMR_ACTIVE(self):
        return self.parent.AMR_ACTIVE
    @property
    def AMR_LEVEL(self):
        return self.parent.AMR_LEVEL
    @property
    def AMR_REFINED(self):
        return self.parent.AMR_REFINED
    @property
    def AMR_COORD1(self):
        return self.parent.AMR_COORD1
    @property
    def AMR_COORD2(self):
        return self.parent.AMR_COORD2
    @property
    def AMR_COORD3(self):
        return self.parent.AMR_COORD3
    @property
    def AMR_PARENT(self):
        return self.parent.AMR_PARENT
    @property
    def AMR_CHILD1(self):
        return self.parent.AMR_CHILD1
    @property
    def AMR_CHILD2(self):
        return self.parent.AMR_CHILD2
    @property
    def AMR_CHILD3(self):
        return self.parent.AMR_CHILD3
    @property
    def AMR_CHILD4(self):
        return self.parent.AMR_CHILD4
    @property
    def AMR_CHILD5(self):
        return self.parent.AMR_CHILD5
    @property
    def AMR_CHILD6(self):
        return self.parent.AMR_CHILD6
    @property
    def AMR_CHILD7(self):
        return self.parent.AMR_CHILD7
    @property
    def AMR_CHILD8(self):
        return self.parent.AMR_CHILD8
    @property
    def AMR_NBR1(self):
        return self.parent.AMR_NBR1
    @property
    def AMR_NBR2(self):
        return self.parent.AMR_NBR2
    @property
    def AMR_NBR3(self):
        return self.parent.AMR_NBR3
    @property
    def AMR_BR4(self):
        return self.parent.AMR_NBR4
    @property
    def AMR_NBR5(self):
        return self.parent.AMR_NBR5
    @property
    def AMR_NBR6(self):
        return self.parent.AMR_NBR6
    @property
    def AMR_NODE(self):
        return self.parent.AMR_NODE
    @property
    def AMR_POLE(self):
        return self.parent.AMR_POLE
    @property
    def AMR_GROUP(self):
        return self.parent.AMR_GROUP
    @property
    def AMR_CORN1(self):
        return self.parent.AMR_CORN1
    @property
    def AMR_CORN2(self):
        return self.parent.AMR_CORN2
    @property
    def AMR_CORN3(self):
        return self.parent.AMR_CORN3
    @property
    def AMR_CORN4(self):
        return self.parent.AMR_CORN4
    @property
    def AMR_CORN5(self):
        return self.parent.AMR_CORN5
    @property
    def AMR_CORN6(self):
        return self.parent.AMR_CORN6
    @property
    def AMR_CORN7(self):
        return self.parent.AMR_CORN7
    @property
    def AMR_CORN8(self):
        return self.parent.AMR_CORN8
    @property
    def AMR_CORN9(self):
        return self.parent.AMR_CORN9
    @property
    def AMR_CORN10(self):
        return self.parent.AMR_CORN10
    @property
    def AMR_CORN11(self):
        return self.parent.AMR_CORN11
    @property
    def AMR_CORN12(self):
        return self.parent.AMR_CORN12
    @property
    def AMR_LEVEL1(self):
        return self.parent.AMR_LEVEL1
    @property
    def AMR_LEVEL2(self):
        return self.parent.AMR_LEVEL2
    @property
    def AMR_LEVEL3(self):
        return self.parent.AMR_LEVEL3
    @property
    def block(self):
        return self.parent.block
    @property
    def n_ord(self):
        return self.parent.n_ord
    @property
    def AMR_LEVEL1_c(self):
        return self.parent.AMR_LEVEL1_c
    @property
    def AMR_LEVEL2_c(self):
        return self.parent.AMR_LEVEL2_c
    @property
    def AMR_LEVEL3_c(self):
        return self.parent.AMR_LEVEL3_c
    @property
    def AMR_COORD1_c(self):
        return self.parent.AMR_COORD1_c
    @property
    def AMR_COORD2_c(self):
        return self.parent.AMR_COORD2_c
    @property
    def AMR_COORD3_c(self):
        return self.parent.AMR_COORD3_c
    @property
    def bs1new(self):
        return self.parent.bs1new
    @property
    def bs2new(self):
        return self.parent.bs2new
    @property
    def bs3new(self):
        return self.parent.bs3new
    @property
    def nx(self):
        return self.parent.nx
    @property
    def ny(self):
        return self.parent.ny
    @property
    def nz(self):
        return self.parent.nz
    @property
    def x1(self):
        return self.parent.x1
    @property
    def x2(self):
        return self.parent.x2
    @property
    def x3(self):
        return self.parent.x3
    @property
    def r(self):
        return self.parent.r
    @property
    def h(self):
        return self.parent.h
    @property
    def ph(self):
        return self.parent.ph
    @property
    def gcov(self):
        return self.parent.gcov
    @property
    def gcon(self):
        return self.parent.gcon
    @property
    def gdet(self):
        return self.parent.gdet
    @property
    def dxdxp(self):
        return self.parent.dxdxp
    @property
    def _dx1(self):
        return self.parent._dx1
    @property
    def _dx2(self):
        return self.parent._dx2
    @property
    def _dx3(self):
        return self.parent._dx3
    @property
    def i_min(self):
        return self.parent.i_min
    @property
    def i_max(self):
        return self.parent.i_max
    @property
    def j_min(self):
        return self.parent.j_min
    @property
    def j_max(self):
        return self.parent.j_max
    @property
    def k_min(self):
        return self.parent.k_min
    @property
    def k_max(self):
        return self.parent.k_max
    @property
    def lookup(self):
         return self.parent.lookup
    @property
    def dxdxp_inv(self):
        return self.parent.dxdxp_inv
    @property
    def dxBLdx(self):
        return self.parent.dxBLdx
    @property
    def dxdxBL(self):
        return self.parent.dxdxBL
    @property
    def dxdxe(self):
        return self.parent.dxdxe
    @property
    def dxedx(self):
        return self.parent.dxedx
    @property
    def gcov_KS(self):
        return self.parent.gcov_KS
    @property
    def gcov_BL(self):
        return self.parent.gcov_BL
    @property
    def gcov_LC(self):
        return self.parent.gcov_LC
        
    def _readParamData(self):
        '''
        old analogue: "rpar_new"
        
        Reads in parameters file for this simulation. 
        '''        
        path_to_params = self.sim_dir + "/dumps%d/parameters" % self.dump_no
        try: 
            fin = open(path_to_params, "rb")
        except OSError as err_msg:
            print("[_readParamData] %s" % err_msg)


            
        ##
        self.t              = np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]
        self.n_active       = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
        self.n_active_total = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
        self.nstep          = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
        self.Dtd            = np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]
        self.Dtl            = np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]
        self.Dtr            = np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]
        self.dump_cnt       = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
        self.rdump_cnt      = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
        self.dt             = np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]
        self.failed         = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
        
        self.bs1  = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
        self.bs2  = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
        self.bs3  = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
        self.nmax = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
        self.nb1  = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
        self.nb2  = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
        self.nb3  = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]

        self.startx1        = np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]
        self.startx2        = np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]
        self.startx3        = np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]
        
        ## These are already set in Simulation
        _dx1           = np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]#*r1 # coarsest level of grid
        _dx2           = np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]#*r2 # add _dx1new, etc...
        _dx3           = np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]#*r3 # M: how does r1/2/3 differ from lowres?

        self.tf             = np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]
        self.a              = np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]
        self.gam            = np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]
        self.cour           = np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]
        self.Rin            = np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]
        self.Rout           = np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]
        self.R0             = np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]
        self.fractheta      = np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]        
        
        for n in range(0,13):
            trash = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
        trash = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
        if(trash >= 10000000):
            DO_NS = 1
            trash=trash-10000000
        else:
            DO_NS = 0
        if(trash >= 1000):
            P_NUM=1
            trash=trash-1000
        else:
            P_NUM=0
        if (trash >= 100):
            TWO_T = 1
            trash = trash - 100
        else:
            TWO_T = 0
        if (trash >= 10):
            RESISTIVE = 1
            trash = trash - 10
        else:
            RESISTIVE = 0
        if (trash >= 1):
            RAD_M1 = 1
            trash = trash - 1
        else:
            RAD_M1 = 0
        trash = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]

        nb = self.n_active_total
        rhor = 1 + (1 - self.a ** 2) ** 0.5

        NODE=np.copy(self.n_ord)
        TIMELEVEL=np.copy(self.n_ord)

        REF_1=1
        REF_2=1
        REF_3=1
        flag_restore = 0
        size = os.path.getsize(path_to_params)
        if(size>=66*4+3*self.n_active_total*4):
            n=0
            while n<self.n_active_total:
                self.n_ord[n]=np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
                TIMELEVEL[n] = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
                NODE[n] = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
                n=n+1
        elif(size >= 66 * 4 + 2 * self.n_active_total * 4):
            n = 0
            flag_restore=1
            while n < self.n_active_total:
                self.n_ord[n] = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
                TIMELEVEL[n] = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
                n = n + 1
                
        
        if(self.export_raytracing==1 and (self.bs1%self.lowres1!=0 or self.bs2%self.lowres2!=0 or self.bs3%self.lowres3!=0 or ((self.lowres1 & (self.lowres1-1) == 0) and self.lowres1 != 0)!=1 or ((self.lowres2 & (self.lowres2-1) == 0) and self.lowres2 != 0)!=1 or ((self.lowres3 & (self.lowres3-1) == 0) and self.lowres3 != 0)!=1)):
            print("[Simulation._readParamData] You set export_raytracing=1; For raytracing block size needs to be divisable by lowres!")
        if(self.export_raytracing==1 and self.interpolate_var==0):
            print("[Simulation._readParamData] You set export_raytracing=1 and interpolate_var=0; Warning: Variable interpolation is highly recommended for raytracing!")
        
        fin.close()
        
        # What we will need later:     
        self.DO_NS        = DO_NS   
        self.RAD_M1       = RAD_M1
        self.trash        = trash
        self.nb           = nb
        self.rhor         = rhor
        self.NODE         = NODE
        self.REF_1        = REF_1
        self.REF_2        = REF_2
        self.REF_3        = REF_3
        self.flag_restore = flag_restore
        self.TWO_T        = TWO_T
        self.RESISTIVE    = RESISTIVE
        self.P_NUM        = P_NUM

    def _read_dump(self,mytype=np.float32):
        '''
        old analogue: rdump_new
        
        Description: 
            Reads in the dump file for this simulation. 
        '''

        # Initialize array
        self.rho        = np.zeros((self.nb, self.bs1new, self.bs2new, self.bs3new), dtype=mytype, order='C')
        self.ug         = np.zeros((self.nb, self.bs1new, self.bs2new, self.bs3new), dtype=mytype, order='C')
        self.uu         = np.zeros((4, self.nb, self.bs1new, self.bs2new, self.bs3new), dtype=mytype, order='C')
        self.B          = np.zeros((4, self.nb, self.bs1new, self.bs2new, self.bs3new), dtype=mytype, order='C')
        if(self.RAD_M1):
            self.E_rad  = np.zeros((self.nb, self.bs1new, self.bs2new, self.bs3new), dtype=mytype, order='C')
            self.uu_rad = np.zeros((4, self.nb, self.bs1new, self.bs2new, self.bs3new), dtype=mytype, order='C')
        else:
            self.E_rad  = self.ug
            self.uu_rad = self.uu
        if (self.RESISTIVE):
            self.E = np.zeros((4,self.nb, self.bs1new, self.bs2new, self.bs3new), dtype=mytype, order='C')
        else:
            self.E = self.B

        if(self.TWO_T):
            self.TE = np.zeros((self.nb, self.bs1new, self.bs2new, self.bs3new), dtype=mytype, order='C')
            self.TI = np.zeros((self.nb, self.bs1new, self.bs2new, self.bs3new), dtype=mytype, order='C')
        else:
            self.TE=self.rho
            self.TI=self.rho

        if(self.P_NUM):
            self.photon_number=np.zeros((self.nb, self.bs1new, self.bs2new, self.bs3new), dtype=mytype, order='C')
        else:
            self.photon_number=self.rho
        

        self.lookup['rho']    = "Gas density array"
        self.lookup['ug']     = "Internal energy array"
        self.lookup['uu']     = "Contravariant four velocity array"
        self.lookup['B']      = "Magnetic field (contravariant? not sure) array"
        self.lookup['E_rad']  = "Radiation energy density?"
        self.lookup['uu_rad'] = "?"
        self.lookup['E']      = "Electric field array (if RESISTIVE=1; else, = B)"
        self.lookup['TE']     = "Electron temperature (if TWO_T=1; else, = rho)"
        self.lookup['TI']     = "Ion temperature (if TWO_T=1; else, = rho)"
        self.lookup["photon_number"] = "Photon number (if P_NUM=1; else, = rho)"
        

        if(os.path.isfile("dumps%d/new_dump" %self.dump_no)):
            flag=1
        else:
            flag=0

            
        kernel_read_dump(flag, self.RAD_M1, self.RESISTIVE, self.TWO_T, self.P_NUM, c_char_p(self.sim_dir.encode('utf-8')), self.dump_no,int(self.n_active_total),self.lowres1,self.lowres2,self.lowres3,int(self.nb),int(self.bs1),int(self.bs2),int(self.bs3),self.rho,self.ug,self.uu,self.B,self.E,self.E_rad, self.uu_rad, self.TE, self.TI, self.photon_number,self.gcov,self.gcon,self.axisym)
        
        print("Finishing _read_dump().")
          
        
    def _read_dump_flatten(self,mytype=np.float32):
        '''
        old analogue: rdump_griddata
        
        Description: 
            Reads in the dump file for this simulation and flattens blocks to a single uniform grid. 
        '''
        
        ACTIVE1 = np.max(self.block[self.n_ord, self.AMR_LEVEL1])
        ACTIVE2 = np.max(self.block[self.n_ord, self.AMR_LEVEL2])
        ACTIVE3 = np.max(self.block[self.n_ord, self.AMR_LEVEL3])
        
        if self.do_box:
            gridsizex1 = self.i_max-self.i_min
            gridsizex2 = self.j_max-self.j_min
            gridsizex3 = self.k_max-self.k_min
        else: 
            gridsizex1 = int(self.nb1 * (1 + self.REF_1) ** ACTIVE1 * self.bs1/self.lowres1)
            gridsizex2 = int(self.nb2 * (1 + self.REF_2) ** ACTIVE2 * self.bs2/self.lowres2)
            gridsizex3 = int(self.nb3 * (1 + self.REF_3) ** ACTIVE3 * self.bs3/self.lowres3)
        
        
        if ((int(self.nb1 * (1 + self.REF_1) ** ACTIVE1 * self.bs1) % self.lowres1) != 0 or (int(self.nb2 * (1 + self.REF_2) ** ACTIVE2 * self.bs2) % self.lowres2) != 0 or (int(self.nb3 * (1 + self.REF_3) ** ACTIVE3 * self.bs3) % self.lowres3) != 0):
            print("[SimulationOld._read_dump] Incompatible low res settings!")

        # Initialize array
        self.rho = np.zeros((1, gridsizex1, gridsizex2, gridsizex3), dtype=mytype, order='C')
        self.ug = np.zeros((1, gridsizex1, gridsizex2, gridsizex3), dtype=mytype, order='C')
        self.uu = np.zeros((4, 1, gridsizex1, gridsizex2, gridsizex3), dtype=mytype, order='C')
        self.B = np.zeros((4, 1, gridsizex1, gridsizex2, gridsizex3), dtype=mytype, order='C')
        self._bsq = np.zeros((1, gridsizex1, gridsizex2, gridsizex3), dtype=mytype, order='C') # _bsq and not bsq for compatibility with flatten=0; the bsq @property checks if _bsq is set already. If flatten=0, it's set in the @property. Else, kernel_read_dump_griddata sets _bsq. see bsq() for reference.  
        
        if(self.RAD_M1):
            self.E_rad = np.zeros((1, gridsizex1, gridsizex2, gridsizex3), dtype=mytype, order='C')
            self.uu_rad = np.zeros((4, 1, gridsizex1, gridsizex2, gridsizex3), dtype=mytype, order='C')
        else:
            self.E_rad  = self.ug
            self.uu_rad = self.uu
            
        if (self.RESISTIVE):
            self.E = np.zeros((4, 1, gridsizex1, gridsizex2, gridsizex3), dtype=mytype, order='C')
        else:
            self.E = self.B
            
        if(self.TWO_T):
            self.TE = np.zeros((1, gridsizex1, gridsizex2, gridsizex3), dtype=mytype, order='C')
            self.TI = np.zeros((1, gridsizex1, gridsizex2, gridsizex3), dtype=mytype, order='C')
        else:
            self.TE=self.rho
            self.TI=self.rho

        if(self.P_NUM):
            self.photon_number = np.zeros((1, gridsizex1, gridsizex2, gridsizex3), dtype=mytype, order='C')
        else:
            self.photon_number=self.rho
            
        if(self.export_raytracing):
            self.Rdot = np.zeros((1, gridsizex1, gridsizex2, gridsizex3), dtype=mytype, order='C')
        else:
            self.Rdot = np.zeros((1, 1, 1, 1), dtype=mytype, order='C')


        self.lookup['rho']    = "Gas density array"
        self.lookup['ug']     = "Internal energy array"
        self.lookup['uu']     = "Contravariant four velocity array"
        self.lookup['B']      = "Magnetic field (contravariant? not sure) array"
        self.lookup['E_rad']  = "Radiation energy density?"
        self.lookup['uu_rad'] = "?"


        if(os.path.isfile("dumps%d/new_dump" %self.dump_no)):
            flag=1
        else:
            flag=0
        
        kernel_read_dump_griddata(flag, self.interpolate_var, self.RAD_M1, self.RESISTIVE, self.TWO_T, self.P_NUM, c_char_p(self.sim_dir.encode('utf-8')), self.dump_no,int(self.n_active_total),self.lowres1,self.lowres2,self.lowres3,int(self.nb),int(self.bs1),int(self.bs2),int(self.bs3),self.rho,self.ug,self.uu,self.B, self.E, self.E_rad, self.uu_rad, self.TE, self.TI, self.photon_number, self.gcov, self.gcon, self.axisym, self.n_ord, ACTIVE1, ACTIVE2, ACTIVE3, self.AMR_LEVEL1_c, self.AMR_LEVEL2_c, self.AMR_LEVEL3_c, self.AMR_COORD1_c, self.AMR_COORD2_c, self.AMR_COORD3_c, int(self.nb1), int(self.nb2), int(self.nb3), int(self.REF_1), int(self.REF_2), int(self.REF_3), self.export_raytracing, self.DISK_THICKNESS, self.a, self.gam, self.Rdot, self._bsq, self.r, self.startx1, self.startx2, self.startx3, self._dx1, self._dx2, self._dx3, self.x1, self.x2, self.x3, self.i_min, self.i_max, self.j_min, self.j_max, self.k_min, self.k_max)        
        
        print("Finishing _read_dump_flatten().")
    
    def _calc_warp(self, tilt = 0):
        '''
        Description: 
            This is a modified approach as that in https://arxiv.org/pdf/astro-ph/0403356.pdf

            Main difference: kernel_calc_warp defines everything in Cartesian coordinates - essentially neglecting the metric. This could potentially cause issues close to the BH. Rewrite later?

            Also, average quantities are computed as well (i.e., average tilt across disk)

        Parameters:
            tilt    tilt of metric. standardly 0. 
            
        TO DO:
        - add a self.metric_tilt to sim attributes and then set tilt = self.metric_tilt
        - try rewriting kernel to be in code coordinates instead of spherical
        '''

        
        # J_car = Angular momentum in the disk in Cartesian coordinates, as function of radius
        # S_r   = This is the Cartesian version of the quantity S^{alpha}S_alpha in Eq 17 of https://arxiv.org/pdf/astro-ph/0403356.pdf
        # J_BH_cross_D = J_BH x J_car
        # J_BH = Angular momentum of BH. If metric isn't tilted, should only be in z direction
        J_car = np.zeros((4, self.nb, self.bs1new), dtype=np.float32, order='C')
        S_r = np.zeros((self.nb, self.bs1new), dtype=np.float32, order='C')
        JBH_cross_D = np.zeros((4, self.nb, self.bs1new), dtype=np.float32, order='C')
        J_BH = np.zeros((4, self.nb, self.bs1new), dtype=np.float32, order='C')

        ## Avg 
        J_car_avg = np.zeros((4, self.nb, 1), dtype=np.float32, order='C')
        S_r_avg = np.zeros((self.nb, 1), dtype=np.float32, order='C')
        JBH_cross_D_avg = np.zeros((4, self.nb, 1), dtype=np.float32, order='C')
        J_BH_avg = np.zeros((4, self.nb, 1), dtype=np.float32, order='C')


        #Su is defined under Eq 17 of https://arxiv.org/pdf/astro-ph/0403356.pdf, L is Eq 17
        #the "_disk" quantities are as is shown in the paper, the "_corona" quantities are calculated for densities <2.5e-9 (this is hard-coded into the kernel and could be revisited in the future). 

        Su_disk = np.zeros((4, self.nb, self.bs1new), dtype=np.float32, order='C')
        L_disk = np.zeros((4, 4, self.nb, self.bs1new), dtype=np.float32, order='C')

        Su_corona = np.zeros((4, self.nb, self.bs1new), dtype=np.float32, order='C')
        L_corona = np.zeros((4, 4, self.nb, self.bs1new), dtype=np.float32, order='C')

        Su_disk_avg = np.zeros((4, self.nb, 1), dtype=np.float32, order='C')
        L_disk_avg = np.zeros((4, 4, self.nb, 1), dtype=np.float32, order='C')

        Su_corona_avg = np.zeros((4, self.nb, 1), dtype=np.float32, order='C')
        L_corona_avg = np.zeros((4, 4, self.nb, 1), dtype=np.float32, order='C')

        print("[Simulation._calc_warp] Running kernel_calc_warp...")
        kernel_calc_warp(self.bs1new, self.bs2new, self.bs3new, self.nb, self.axisym, 1, self.r, self.h, self.ph, self.rho, self.ug, self.uu, self.B, self.gam, self.gcov, self.gcon, self.gdet, self.dxdxp, Su_disk, L_disk, Su_corona, L_corona, Su_disk_avg, L_disk_avg, Su_corona_avg, L_corona_avg)
        print("[Simulation._calc_warp] Finished kernel_calc_warp...")
    
    
        #############################################
        ## Calculate tilt and precession angles of disk
        #############################################
        
        ## S_r = S^2
        S_r[0] = Su_disk[0, 0] * Su_disk[0, 0] + Su_disk[1, 0] * Su_disk[1, 0] + Su_disk[2, 0] * Su_disk[2, 0] + Su_disk[3, 0] * Su_disk[3, 0]
        ## Calculate Cartesian components of disk angular momentum via Eq 17 of https://arxiv.org/pdf/astro-ph/0403356.pdf
        J_car[3] = (L_disk[1, 2] * Su_disk[0] - L_disk[2, 1] * Su_disk[0] + L_disk[2, 0] * Su_disk[1] - L_disk[0, 2] * Su_disk[1] - L_disk[1, 0] * Su_disk[2] + L_disk[0, 1] * Su_disk[2]) / (2 * np.sqrt(np.abs(-S_r)))
        J_car[2] = -(L_disk[1, 3] * Su_disk[0] - L_disk[3, 1] * Su_disk[0] - L_disk[1, 0] * Su_disk[3] + L_disk[3, 0] * Su_disk[1] - L_disk[0, 3] * Su_disk[1] + L_disk[0, 1] * Su_disk[3]) / (2 * np.sqrt(np.abs(-S_r)))
        J_car[1] = (L_disk[2, 3] * Su_disk[0] - L_disk[3, 2] * Su_disk[0] - L_disk[2, 0] * Su_disk[3] + L_disk[0, 2] * Su_disk[3] + L_disk[3, 0] * Su_disk[2] - L_disk[0, 3] * Su_disk[ 2]) / (2 * np.sqrt(np.abs(-S_r)))
        J_length = np.sqrt(J_car[1] * J_car[1] + J_car[2] * J_car[2] + J_car[3] * J_car[3])

        ## Black hole angular momentum
        ## tilt = tilt of metric, usually = 0
        J_BH[1] = -np.sin(tilt) * J_car[1] / J_car[1]
        J_BH[2] = 0 * J_car[1] / J_car[1]
        J_BH[3] = np.cos(tilt) * J_car[1] / J_car[1]
        J_BH_length = np.sqrt(J_BH[1] * J_BH[1] + J_BH[2] * J_BH[2] + J_BH[3] * J_BH[3])

        ## Cross product of BH and disk angular momentum
        JBH_cross_D[1] = J_BH[2] * J_car[3] - J_BH[3] * J_car[2]
        JBH_cross_D[2] = J_BH[3] * J_car[1] - J_BH[1] * J_car[3]
        JBH_cross_D[3] = J_BH[1] * J_car[2] - J_BH[2] * J_car[1]
        JBH_cross_D_length = np.sqrt(JBH_cross_D[1] * JBH_cross_D[1] + JBH_cross_D[2] * JBH_cross_D[2] + JBH_cross_D[3] * JBH_cross_D[3])

        ## Disk tilt and precession angles as a function of radius, in degrees
        self._tilt_disk = np.arccos(np.abs(J_car[1] * J_BH[1] + J_car[2] * J_BH[2] + J_car[3] * J_BH[3]) / (J_BH_length * J_length)) * 180.0 / np.pi
        self._prec_disk = -np.arctan2(JBH_cross_D[1], JBH_cross_D[2]) * 180.0 / np.pi

        #############################################
        ## Calculate average tilt and precession angles of disk
        #############################################
        
        ## S_r_avg = S_avg^2
        S_r_avg[0] = -Su_disk_avg[0, 0] * Su_disk_avg[0, 0] + Su_disk_avg[1, 0] * Su_disk_avg[1, 0] + Su_disk_avg[2, 0] * Su_disk_avg[2, 0] + Su_disk_avg[3, 0] * Su_disk_avg[3, 0]
        
        ## Calculate average Cartesian components of disk angular momentum via Eq 17 of https://arxiv.org/pdf/astro-ph/0403356.pdf
        J_car_avg[3] = (L_disk_avg[1, 2] * Su_disk_avg[0] - L_disk_avg[2, 1] * Su_disk_avg[0] + L_disk_avg[2, 0] * Su_disk_avg[1] - L_disk_avg[0, 2] * Su_disk_avg[1] - L_disk_avg[1, 0] * Su_disk_avg[2] + L_disk_avg[0, 1] * Su_disk_avg[2]) / (2 * np.sqrt(np.abs(-S_r_avg)))
        J_car_avg[2] = -(L_disk_avg[1, 3] * Su_disk_avg[0] - L_disk_avg[3, 1] * Su_disk_avg[0] - L_disk_avg[1, 0] * Su_disk_avg[3] + L_disk_avg[3, 0] * Su_disk_avg[1] - L_disk_avg[0, 3] * Su_disk_avg[1] + L_disk_avg[0, 1] * Su_disk_avg[3]) / (2 * np.sqrt(np.abs(-S_r_avg)))
        J_car_avg[1] = (L_disk_avg[2, 3] * Su_disk_avg[0] - L_disk_avg[3, 2] * Su_disk_avg[0] - L_disk_avg[2, 0] * Su_disk_avg[3] + L_disk_avg[0, 2] * Su_disk_avg[3] + L_disk_avg[3, 0] * Su_disk_avg[2] - L_disk_avg[0, 3] * Su_disk_avg[ 2]) / (2 * np.sqrt(np.abs(-S_r_avg)))
        ## Alternative, try comparing later: 
        #J_car_avg[3] = (L_disk_avg[1, 2])
        #J_car_avg[2] = -(L_disk_avg[1, 3])
        #J_car_avg[1] = (L_disk_avg[2, 3])
        J_length_avg = np.sqrt(J_car_avg[1] * J_car_avg[1] + J_car_avg[2] * J_car_avg[2] + J_car_avg[3] * J_car_avg[3])

        ## Black hole angular momentum
        ## tilt = tilt of metric, usually = 0
        J_BH_avg[1] = -np.sin(tilt) * J_car_avg[1] / J_car_avg[1]
        J_BH_avg[2] = 0 * J_car_avg[1] / J_car_avg[1]
        J_BH_avg[3] = np.cos(tilt) * J_car_avg[1] / J_car_avg[1]
        J_BH_length_avg = np.sqrt(J_BH_avg[1] * J_BH_avg[1] + J_BH_avg[2] * J_BH_avg[2] + J_BH_avg[3] * J_BH_avg[3])
        
        ## Cross product of BH and average disk angular momentum
        JBH_cross_D_avg[1] = J_BH_avg[2] * J_car_avg[3] - J_BH_avg[3] * J_car_avg[2]
        JBH_cross_D_avg[2] = J_BH_avg[3] * J_car_avg[1] - J_BH_avg[1] * J_car_avg[3]
        JBH_cross_D_avg[3] = J_BH_avg[1] * J_car_avg[2] - J_BH_avg[2] * J_car_avg[1]
        JBH_cross_D_length_avg = np.sqrt(JBH_cross_D_avg[1] * JBH_cross_D_avg[1] + JBH_cross_D_avg[2] * JBH_cross_D_avg[2] + JBH_cross_D_avg[3] * JBH_cross_D_avg[3])

        ## Average disk tilt and precession angles as a function of radius, in degrees
        self._tilt_disk_avg = np.arccos(np.abs(J_car_avg[1] * J_BH_avg[1] + J_car_avg[2] * J_BH_avg[2] + J_car_avg[3] * J_BH_avg[3]) / (J_BH_length_avg * J_length_avg)) * 180.0 / np.pi
        self._prec_disk_avg = -np.arctan2(JBH_cross_D_avg[1], JBH_cross_D_avg[2]) * 180.0 / np.pi

        #############################################
        ## Calculate tilt and precession angles of corona
        #############################################
        
        ## S_r = S_corona^2
        S_r[0] = Su_corona[0, 0] * Su_corona[0, 0] + Su_corona[1, 0] * Su_corona[1, 0] + Su_corona[2, 0] * Su_corona[2, 0] + Su_corona[3, 0] * Su_corona[3, 0]
        
        ## Calculate Cartesian components of corona angular momentum via Eq 17 of https://arxiv.org/pdf/astro-ph/0403356.pdf
        J_car[3] = (L_corona[1, 2] * Su_corona[0] - L_corona[2, 1] * Su_corona[0] + L_corona[2, 0] * Su_corona[1] - L_corona[0, 2] * Su_corona[1] - L_corona[1, 0] * Su_corona[2] + L_corona[0, 1] * Su_corona[2]) / (2 * np.sqrt(np.abs(-S_r)))
        J_car[2] = -(L_corona[1, 3] * Su_corona[0] - L_corona[3, 1] * Su_corona[0] - L_corona[1, 0] * Su_corona[3] + L_corona[3, 0] * Su_corona[1] - L_corona[0, 3] * Su_corona[1] + L_corona[0, 1] * Su_corona[3]) / (2 * np.sqrt(np.abs(-S_r)))
        J_car[1] = (L_corona[2, 3] * Su_corona[0] - L_corona[3, 2] * Su_corona[0] - L_corona[2, 0] * Su_corona[3] + L_corona[0, 2] * Su_corona[3] + L_corona[3, 0] * Su_corona[2] - L_corona[0, 3] * Su_corona[ 2]) / (2 * np.sqrt(np.abs(-S_r)))
        J_length = np.sqrt(J_car[1] * J_car[1] + J_car[2] * J_car[2] + J_car[3] * J_car[3])

        ## Black hole angular momentum
        ## tilt = tilt of metric, usually = 0
        J_BH[1] = -np.sin(tilt) * J_car[1] / J_car[1]
        J_BH[2] = 0 * J_car[1] / J_car[1]
        J_BH[3] = np.cos(tilt) * J_car[1] / J_car[1]
        J_BH_length = np.sqrt(J_BH[1] * J_BH[1] + J_BH[2] * J_BH[2] + J_BH[3] * J_BH[3])

        ## Cross product of BH and corona angular momentum
        JBH_cross_D[1] = J_BH[2] * J_car[3] - J_BH[3] * J_car[2]
        JBH_cross_D[2] = J_BH[3] * J_car[1] - J_BH[1] * J_car[3]
        JBH_cross_D[3] = J_BH[1] * J_car[2] - J_BH[2] * J_car[1]
        JBH_cross_D_length = np.sqrt(JBH_cross_D[1] * JBH_cross_D[1] + JBH_cross_D[2] * JBH_cross_D[2] + JBH_cross_D[3] * JBH_cross_D[3])

        ## Corona tilt and precession angles as a function of radius, in degrees
        self._tilt_corona = np.arccos(np.abs(J_car[1] * J_BH[1] + J_car[2] * J_BH[2] + J_car[3] * J_BH[3]) / (J_BH_length * J_length)) * 180.0 / np.pi
        self._prec_corona = -np.arctan2(JBH_cross_D[1], JBH_cross_D[2]) * 180.0 / np.pi

        #############################################
        ## Calculate average tilt and precession angles of corona
        #############################################

        ## S_r_avg = S_avg^2
        S_r_avg[0] = Su_corona_avg[0, 0] * Su_corona_avg[0, 0] + Su_corona_avg[1, 0] * Su_corona_avg[1, 0] + Su_corona_avg[2, 0] * Su_corona_avg[2, 0] + Su_corona_avg[3, 0] * Su_corona_avg[3, 0]
        
        ## Calculate average Cartesian components of corona angular momentum via Eq 17 of https://arxiv.org/pdf/astro-ph/0403356.pdf
        J_car_avg[3] = (L_corona_avg[1, 2] * Su_corona_avg[0] - L_corona_avg[2, 1] * Su_corona_avg[0] + L_corona_avg[2, 0] * Su_corona_avg[1] - L_corona_avg[0, 2] * Su_corona_avg[1] - L_corona_avg[1, 0] * Su_corona_avg[2] + L_corona_avg[0, 1] * Su_corona_avg[2]) / (2 * np.sqrt(np.abs(-S_r_avg)))
        J_car_avg[2] = -(L_corona_avg[1, 3] * Su_corona_avg[0] - L_corona_avg[3, 1] * Su_corona_avg[0] - L_corona_avg[1, 0] * Su_corona_avg[3] + L_corona_avg[3, 0] * Su_corona_avg[1] - L_corona_avg[0, 3] * Su_corona_avg[1] + L_corona_avg[0, 1] * Su_corona_avg[3]) / (2 * np.sqrt(np.abs(-S_r_avg)))
        J_car_avg[1] = (L_corona_avg[2, 3] * Su_corona_avg[0] - L_corona_avg[3, 2] * Su_corona_avg[0] - L_corona_avg[2, 0] * Su_corona_avg[3] + L_corona_avg[0, 2] * Su_corona_avg[3] + L_corona_avg[3, 0] * Su_corona_avg[2] - L_corona_avg[0, 3] * Su_corona_avg[ 2]) / (2 * np.sqrt(np.abs(-S_r_avg)))
        J_length_avg = np.sqrt(J_car_avg[1] * J_car_avg[1] + J_car_avg[2] * J_car_avg[2] + J_car_avg[3] * J_car_avg[3])

        ## Black hole angular momentum
        ## tilt = tilt of metric, usually = 0
        J_BH_avg[1] = -np.sin(tilt) * J_car_avg[1] / J_car_avg[1]
        J_BH_avg[2] = 0 * J_car_avg[1] / J_car_avg[1]
        J_BH_avg[3] = np.cos(tilt) * J_car_avg[1] / J_car_avg[1]
        J_BH_length_avg = np.sqrt(J_BH_avg[1] * J_BH_avg[1] + J_BH_avg[2] * J_BH_avg[2] + J_BH_avg[3] * J_BH_avg[3])

        ## Cross product of BH and average corona angular momentum
        JBH_cross_D_avg[1] = J_BH_avg[2] * J_car_avg[3] - J_BH_avg[3] * J_car_avg[2]
        JBH_cross_D_avg[2] = J_BH_avg[3] * J_car_avg[1] - J_BH_avg[1] * J_car_avg[3]
        JBH_cross_D_avg[3] = J_BH_avg[1] * J_car_avg[2] - J_BH_avg[2] * J_car_avg[1]
        JBH_cross_D_length_avg = np.sqrt(JBH_cross_D_avg[1] * JBH_cross_D_avg[1] + JBH_cross_D_avg[2] * JBH_cross_D_avg[2] + JBH_cross_D_avg[3] * JBH_cross_D_avg[3])

        ## Average corona tilt and precession angles as a function of radius, in degrees
        self._tilt_corona_avg = np.arccos(np.abs(J_car_avg[1] * J_BH_avg[1] + J_car_avg[2] * J_BH_avg[2] + J_car_avg[3] * J_BH_avg[3]) / (J_BH_length_avg * J_length_avg)) * 180.0 / np.pi
        self._prec_corona_avg = -np.arctan2(JBH_cross_D_avg[1], JBH_cross_D_avg[2]) * 180.0 / np.pi

    '''
    The following are for looking up definitions of variables
    '''
        
    def _print_definitions(self,print_unentered=0):
        """
        This reads out all of the saved attributes in this instance of simulation, given that they have an entry
        in the self.lookup dictionary. 
        
        Parameters
        
            print_unentered : 0 or 1, if 1, also print attributes without a self.lookup entry.
        """
        
        for attr in self.__dict__.keys():
            if str(attr) in self.lookup:
                self._lookup(str(attr))
            else:
                if print_unentered: print("\n %s \n Type: %s \n Description: none entered \n" % (attr, type(attr)) )

                    
    def _lookup(self,attr):
        """
        This prints out an entry in the self.lookup dictionary, along with the type. Basically the same as
        printing self.lookup[attr] with some formatting. 

        Parameters
        
            attr : the attribute you want to lookup the definition of. Should be type str and have an entry inside self.lookup.
        """
        
        assert type(attr) == str, ('[Simulation._lookup] attr should be of type string!')
        if attr in self.lookup:
            attr_val  = self.__dict__[attr]
            attr_type = type(attr_val)            
            if attr_type == np.ndarray:
                print("\n %s \n Shape: %s \n Type: %s \n Description: %s \n" % (attr, np.shape(attr_val), attr_type, self.lookup[attr]) )                
            else:
                print("\n %s \n Value: %s \n Type: %s \n Description: %s \n" % (attr, str(attr_val), attr_type, self.lookup[attr]) )
        else: print("[Simulation._lookup] Attribute \"%s\" not found!" % attr)
           


    '''
    The following are derived quantities, using the @property decorator so they are only constructed when needed
    '''
    ## to do: add derived quantities to lookup[]
        
    @property
    def rho0(self):
        # rest mass density
        return self.rho * self.uu[0]
    
    @property
    def x(self):
        # cartesian coordinate x 
        return self.r*np.cos(self.ph)*np.sin(self.h)
    
    @property
    def y(self):
        # cartesian coordinate y
        return self.r*np.sin(self.ph)*np.sin(self.h)

    @property
    def z(self):
        # cartesian coordinate z
        return self.r*np.cos(self.h)
    
    @property
    def s(self):
        # cylindrical coordinate s
        return np.sqrt(self.x**2 + self.y**2)
    
    @property
    def vr(self):
        # radial velocity
        return np.sqrt(self.gcov[1,1])*self.uu[1]#np.sqrt(self.uu[1]*self.ud[1])#np.sum(self.dxdxp[1]*self.uu,axis=0)/self.uu[0]
    
    @property
    def vh(self):
        # theta velocity
        return np.sqrt(self.gcov[2,2])*self.uu[2]#np.sqrt(self.uu[2]*self.ud[2])#self.r*np.sum(self.dxdxp[2]*self.uu,axis=0)/self.uu[0]
    
    @property
    def vp(self):
        # phi velocity
        return np.sqrt(self.gcov[3,3])*self.uu[3]#np.sqrt(self.uu[3]*self.ud[3])#self.r*np.sin(self.h)*np.sum(self.dxdxp[3]*self.uu,axis=0)/self.uu[0]
    
    @property
    def vx(self):
        # x velocity
        return self.vr*np.sin(self.h)*np.cos(self.ph) - self.vp*np.sin(self.ph) + self.vh*np.cos(self.h)*np.cos(self.ph)

    @property
    def vy(self):
        # y velocity
        return self.vr*np.sin(self.h)*np.sin(self.ph) + self.vp*np.cos(self.ph) + self.vh*np.cos(self.h)*np.sin(self.ph)

    @property
    def vz(self):
        # z velocity
        return self.vr*np.cos(self.h) - self.vh*np.sin(self.h)
    
    @property
    def Br(self):
        # radial magnetic field
        return np.sqrt(self.gcov[1,1])*self.B[1]#np.sqrt(self.B[1]*self.Bd[1])#np.sum(self.dxdxp[1]*self.B,axis=0)
    
    @property
    def Bh(self):
        # theta magnetic field
        return np.sqrt(self.gcov[2,2])*self.B[2]#np.sqrt(self.B[2]*self.Bd[2])#self.r*np.sum(self.dxdxp[2]*self.B,axis=0)
    
    @property
    def Bp(self):
        # phi magnetic field
        return np.sqrt(self.gcov[3,3])*self.B[3]#np.sqrt(self.B[3]*self.Bd[3])#self.r*np.sin(self.h)*np.sum(self.dxdxp[3]*self.B,axis=0)
    
    @property
    def Bx(self):
        # x magnetic field
        return self.Br*np.sin(self.h)*np.cos(self.ph) - self.Bp*np.sin(self.ph) + self.Bh*np.cos(self.h)*np.cos(self.ph)

    @property
    def By(self):
        # y magnetic field
        return self.Br*np.sin(self.h)*np.sin(self.ph) + self.Bp*np.cos(self.ph) + self.Bh*np.cos(self.h)*np.sin(self.ph)

    @property
    def Bz(self):
        # z magnetic field
        return self.Br*np.cos(self.h) - self.Bh*np.sin(self.h)
    
    
    @property
    def v(self):
        '''
        Choose magnitude of velocity relative to a zero angular momentum observer
        
        ZAMO observer can be defined by assuming a four velocity,
        u_mu = alpha*(-1, 0, 0, 0)
        where alpha is the lapse = sqrt(-1/g^tt)
        
        '''
        return np.sqrt(1. - 1./((-self.gcon[0,0])**(-0.5)*self.uu[0])**2.)
        #return np.sqrt(self.vx*self.vx + self.vy*self.vy + self.vz*self.vz)
    
    @property
    def pgas(self):
        # gas pressure = (gam - 1)*internal energy
        return (self.gam-1)*self.ug
    
    @property
    def dpdr(self):
        # radial pressure gradient of pressure
        # axis=-3 corresponds to radius
        return np.gradient(self.pgas,axis=-3)/np.gradient(self.r,axis=-3)
    
    @property
    def ud(self):
        '''
        Construct covariant u
        '''
        return self._lower(self.uu)
    
    @property
    def Bd(self):
        '''
        Construct covariant u
        '''
        return self._lower(self.B)
    
    @property
    def bu(self):
        '''
        Construct contravariant b
        '''
        bu = np.zeros(np.shape(self.uu))
        for i in range(4):
            bu[0] += self.B[1]*self.uu[i] * self.gcov[1,i] + self.B[2] * self.uu[i] * self.gcov[2,i] + self.B[3] * self.uu[i] * self.gcov[3,i]
        for i in range(1,4): 
            bu[i] = (self.B[i] + bu[0]*self.uu[i])/self.uu[0]
        return bu
    
    @property
    def bd(self):
        '''
        Construct covariant b
        '''
        return self._lower(self.bu)

    
    @property
    def bsq(self): 
        '''
        Calculate magnetic field squared
        '''
        return np.sum(self.bu*self.bd,axis=0)
    
    @property
    def pb(self):
        '''
        Follows from magnetic pressure term in GRMHD stress-energy tensor, i.e., pressure term is:
        (pgas + b^2/2)g^mu^nu
        '''
        return self.bsq/2
    
    @property
    def beta(self):
        '''
        Ratio of gas pressure to magnetic pressure
        '''
        
        return self.pgas/(1e-20 + self.pb)
    
    @property
    def mass_flux(self):
        '''
        radial mass flux (mass / area / time)
        
        Note: calculated wrong? should use uu? don't use this yet
        '''
        
        return self.vr*self.rho
    
    @property
    def cs(self):
        '''
        Sound speed
        '''
        return np.sqrt(self.gam*self.pgas/self.rho)
    
    @property
    def mach(self):
        '''
        Mach number
        '''
        return np.abs(self.v/self.cs)
    
    @staticmethod
    def _calc_roots(p):
        '''
        Compute the positive, real roots for an array of polynomials.

        Args
            p: Polynomial coefficient array of shape (N x A), where N is the number of coefficients and A is the shape of the polynomial array.

        Returns
            real_roots: Polynomial root array of shape (A).
        '''
        # convert coefficient array to numpy array
        p = np.array(p)
        # get shape information
        N = p.shape[0]
        coeff_shape = p.shape[1:]
        coeff_dims = np.shape(coeff_shape)[0]
        # build companion matrices for each polynomial
        A0 = np.diag(np.ones((N-2,)), -1)
        A = np.zeros(np.concatenate((A0.shape, coeff_shape)))
        index1 = tuple([slice(None, None)] * (2 + coeff_dims))
        index2 = tuple([slice(None, None)] * 2 + [np.newaxis] * coeff_dims)
        A[index1] = A0[index2]
        A[0] = -p[1:] / p[0]
        # find companion matrix eigenvalues
        roots = np.linalg.eigvals(np.moveaxis(A, (0, 1), (-2, -1)))
        # ignore negative and imaginary roots
        try:
            real_roots = roots[np.where((roots.real > 0) * (np.abs(roots.imag) < 1e-5))].real.reshape(coeff_shape)
            return real_roots
        except ValueError:
            raise ValueError("Failed to find unique postive real root.")
        return real_roots
    
    def calc_temp(self):
        '''
        Compute the temperature for an EOS which implicitly includes radiation pressure.
        Solve the quartic equation u = u_gas + u_rad = nkT / (gamg - 1) + aT^4.
        '''
        rho_cgs = self.rho * self.mass_density_scale
        u_cgs = self.ug * self.energy_density_scale
        n = rho_cgs / (self.mh_cgs * self.mmw)
        coeff = np.array([np.full_like(rho_cgs, self.arad_cgs), np.zeros_like(rho_cgs), np.zeros_like(rho_cgs), n * self.boltz_cgs / (self.gamg - 1), -u_cgs])
        #if twoD: coeff = [arr[0, :, self.bs2new//2, :] for arr in coeff]
        self._temp = self._calc_roots(coeff)
        
    @property
    def temp(self):
        try:
            return self._temp
        except:
            self.calc_temp()
            return self._temp
    
    @property
    def gami(self):
        '''
        Effective adiabatic index
        
        Defined by p_tot = (gami - 1) u_tot.
        '''
        rho_cgs = self.rho * self.mass_density_scale
        n = rho_cgs / (self.mh_cgs * self.mmw)
        beta = 3 * n * self.boltz_cgs / (self.arad_cgs * self.temp**3 + 3 * n * self.boltz_cgs)
        gami = (beta * (3 * self.gamg - 4) - 4 * (self.gamg - 1)) / (beta * (3 * self.gamg - 4) - 3 * (self.gamg - 1))
        return gami
    
    @property
    def gama(self):
        '''
        Effective polytropic index
        
        Defined by k = p_tot rho^gama.
        See Section 13.2 of Stellar Structure and Evolution (https://link.springer.com/content/pdf/10.1007%2F978-3-642-30304-3.pdf).
        '''
        rho_cgs = self.rho * self.mass_density_scale
        n = rho_cgs / (self.mh_cgs * self.mmw)
        beta = 3 * n * self.boltz_cgs / (self.arad_cgs * self.temp**3 + 3 * n * self.boltz_cgs)
        gama = (beta**2 * (3 * self.gamg - 4) + 4 * (3 * beta - 4) * (self.gamg - 1)) / (beta * (12 * self.gamg - 13) - 12 * (self.gamg - 1))
        return gama
    
    @property
    def bernoulli(self):
        '''
        Bernoulli parameter
        
        bernoulli > 0 is unbound material, bernoulli < 0 is bound material
        '''
        return -(self.ud[0] * (self.rho + self.ug * self.gamg) / self.rho + 1)
    
    @property
    def S(self):
        '''
        Entropy
        '''
        return (self.gamg - 1) * self.ug / self.rho**self.gamg
    
    @property
    def Ledd(self):
        '''
        Eddington luminosity
        '''
        Ledd_cgs = 4*np.pi * self.G_cgs * self.M_BH_cgs * self.c_cgs * self.mh_cgs / self.thomson_cgs
        return Ledd_cgs / (self.mass_scale * self.length_scale**2 / self.time_scale**3)
    
    @property
    def Medd(self):
        '''
        Eddington accretion rate
        '''
        Medd_cgs = 4*np.pi * self.G_cgs * self.M_BH_cgs * self.mh_cgs / (self.epsilon * self.c_cgs * self.thomson_cgs)
        return Medd_cgs / (self.mass_scale / self.time_scale)
    
    @staticmethod
    def _rotate_coord(X, tilt):
        '''
        Rotate a set a Cartesian coordinates.
        '''
        X_copy = np.copy(X)
        X[1] = X_copy[1] * np.cos(tilt) + X_copy[3] * np.sin(tilt)
        X[3] = -X_copy[1] * np.sin(tilt) + X_copy[3] * np.cos(tilt)
        return X
    
    def set_tilted_arrays(self, tilt, prec):
        ph_old = self.ph[:, :, self.bs2new - 1:self.bs2new, :]
        tilt_tmp = np.zeros((self.nb, self.bs1new, 1, 1), dtype=np.float32)
        prec_tmp = np.zeros((self.nb, self.bs1new, 1, 1), dtype=np.int32)

        tilt_tmp[0, :, 0, 0] = tilt / 180.0 * np.pi
        prec_tmp[0, :, 0, 0] = prec / 360.0 * self.bs3new

        X = np.array([ph_old*0, np.cos(ph_old), np.sin(ph_old), ph_old*0])
        X = self._rotate_coord(X, tilt_tmp)
        h_new, ph_new = np.arccos(X[3]), np.arctan2(X[2], X[1])

        X2 = np.array([ph_old*0, -np.sin(ph_old), np.cos(ph_old), ph_old*0])
        X2 = self._rotate_coord(X2, tilt_tmp)
        self._theta_to_phi = X2[1] * np.cos(h_new) * np.cos(ph_new) + X2[2] * np.cos(h_new) * np.sin(ph_new) - X2[3] * np.sin(h_new)
        self._phi_to_phi = -X2[1] * np.sin(ph_new) + X2[2] * np.cos(ph_new)

        X3 = np.array([ph_old*0, ph_old*0, ph_old*0, ph_old*0 - 1])
        X3 = self._rotate_coord(X3, tilt_tmp)
        self._theta_to_theta = X3[1] * np.cos(h_new) * np.cos(ph_new) + X3[2] * np.cos(h_new) * np.sin(ph_new) - X3[3] * np.sin(h_new)
        self._phi_to_theta = -X3[1] * np.sin(ph_new) + X3[2] * np.cos(ph_new)
        
        for i in range(0, self.bs1new):
            self._phi_to_phi[0, i] = np.roll(self._phi_to_phi[0, i], prec_tmp[0, i, 0, 0], axis=1)
            self._phi_to_theta[0, i] = np.roll(self._phi_to_theta[0, i], prec_tmp[0, i, 0, 0], axis=1)
            self._theta_to_theta[0, i] = np.roll(self._theta_to_theta[0, i], prec_tmp[0, i, 0, 0], axis=1)
            self._theta_to_phi[0, i] = np.roll(self._theta_to_phi[0, i], prec_tmp[0, i, 0, 0], axis=1)
            
    @property
    def phi_to_phi(self):
        try:
            return self._phi_to_phi
        except:
            self.set_tilted_arrays(0, 0)
            return self._phi_to_phi
        
    @property
    def phi_to_theta(self):
        try:
            return self._phi_to_theta
        except:
            self.set_tilted_arrays(0, 0)
            return self._phi_to_theta
        
    @property
    def theta_to_theta(self):
        try:
            return self._theta_to_theta
        except:
            self.set_tilted_arrays(0, 0)
            return self._theta_to_theta
        
    @property
    def theta_to_phi(self):
        try:
            return self._theta_to_phi
        except:
            self.set_tilted_arrays(0, 0)
            return self._theta_to_phi
    
    def project_vector(self, B):
        B_proj = np.copy(B)
        B_proj[1] = B[1] * np.sqrt(self.gcov[1, 1])
        B_proj[2] = B[2] * np.sqrt(self.gcov[2, 2]) * self.theta_to_theta + B[3] * np.sqrt(self.gcov[3, 3]) * self.phi_to_theta
        B_proj[3] = B[2] * np.sqrt(self.gcov[2, 2]) * self.theta_to_phi + B[3] * np.sqrt(self.gcov[3, 3]) * self.phi_to_phi
        return B_proj
    
    def calc_Q(self):
        '''
        Compute the magnetic field convergence metrics.
        See Hawley et al. 2011 (https://arxiv.org/pdf/1103.5987.pdf) for more information.
        '''
        self._Q = np.zeros((4, 1, self.bs1new, self.bs2new, self.bs3new), dtype=np.float32, order='C')
        dx = np.zeros((4, self.nb, self.bs1new, self.bs2new, 1), dtype=np.float32, order='C')
        dx[1] = self._dx1 / self.lowres1 * np.sqrt(self.gcov[1, 1, :, :, :, :])
        dx[2] = self._dx2 / self.lowres2 * np.sqrt(self.gcov[2, 2, :, :, :, :])
        dx[3] = self._dx3 / self.lowres3 * np.sqrt(self.gcov[3, 3, :, :, :, :])
        
        bu_proj = self.project_vector(self.bu)

        for i in range(1, 4):
            alf_speed = np.sqrt(np.abs(bu_proj[i]**2) / (self.rho + self.bsq + self.gam * self.ug))
            vrot = np.sqrt((self.uu[3]**2 * self.gcov[3][3] + self.uu[2]**2 * self.gcov[2][2] + self.uu[1]**2 * self.gcov[1][1])) / self.uu[0]
            wavelength = 2 * np.pi * alf_speed * self.r / vrot
            if (i == 1): self._Q[i] = wavelength / dx[i]
            elif (i == 2): self._Q[i] = wavelength / (dx[2] * self.theta_to_theta + dx[3] * np.abs(self.phi_to_theta))
            elif (i == 3): self._Q[i] = wavelength / (dx[2] * np.abs(self.theta_to_phi) + dx[3] * self.phi_to_phi)
            self._Q[i] = np.nan_to_num(self.Q[i])
            
    @property
    def Q(self):
        try:
            return self._Q
        except:
            self.calc_Q()
            return self._Q
        
    @property
    def Tuu(self):
        '''
        This is the contravariant electromagnetic stress tensor in code coordinates
        
        T^{munu}
        
        Equation 12 of Noble et al 2005
        '''
        
        return (self.rho + self.pgas + self.ug + self.bsq)*self.uu[:,None]*self.uu[None,:] + (self.pgas + self.pb)*self.gcon - self.bu[:,None]*self.bu[None,:]
    
    @property
    def Tdd(self):
        '''
        This is the covariant electromagnetic stress tensor in code coordinates
        
        T_{munu}
        '''
        
        return (self.rho + self.pgas + self.ug + self.bsq)*self.ud[:,None]*self.ud[None,:] + (self.pgas + self.pb)*self.gcov - self.bd[:,None]*self.bd[None,:]
        
    @property
    def Tud(self):
        '''
        This is the mixed electromagnetic stress tensor in code coordinates
        
        T^{mu}_{nu}
        
        
        To do: verify on-diagonal terms are correct (off diagonal terms appear to be right, but on diagonals depend on metric)
        '''
        
        ## Kronicker delta 
        I = np.eye(4)
        I = I[:,:,None,None,None,None] * np.ones((4,4,self.nb,self.bs1new,self.bs2new,self.bs3new))

        
        return (self.rho + self.pgas + self.ug + self.bsq)*self.uu[:,None]*self.ud[None,:] + (self.pgas + self.pb)*I - self.bu[:,None]*self.bd[None,:]
    
    
    ## the eight quantities tilt/prec_disk/corona(_avg) are all calculated in the self._calc_warp function, so are handled
    ## differently than the other @property attributes, and are executed with a try except loop in case self_calc_warp() hasn't been 
    ## called yet. (no need for @lru_cache() since after self._calc_warp() is called once, tilt_disk will just retrieve _tilt_disk, etc 
    
    @property
    def tilt_disk(self):
        try: 
            return self._tilt_disk
        except: 
            self._calc_warp()
            return self._tilt_disk

    @property
    def tilt_corona(self):
        try: 
            return self._tilt_corona
        except: 
            self._calc_warp()
            return self._tilt_corona
    
    @property
    def prec_disk(self):
        try: 
            return self._prec_disk
        except: 
            self._calc_warp()
            return self._prec_disk

    @property
    def prec_corona(self):
        try: 
            return self._prec_corona
        except: 
            self._calc_warp()
            return self._prec_corona
    
    @property
    def tilt_disk_avg(self):
        try: 
            return self._tilt_disk_avg
        except: 
            self._calc_warp()
            return self._tilt_disk_avg

    @property
    def tilt_corona_avg(self):
        try: 
            return self._tilt_corona_avg
        except: 
            self._calc_warp()
            return self._tilt_corona_avg
    
    @property
    def prec_disk_avg(self):
        try: 
            return self._prec_disk_avg
        except: 
            self._calc_warp()
            return self._prec_disk_avg

    @property
    def prec_corona_avg(self):
        try: 
            return self._prec_corona_avg
        except: 
            self._calc_warp()
            return self._prec_corona_avg
        
    @property
    def r1d(self):
        return self.r[0, :, self.bs2new//2, 0]
    
    @property
    def h1d(self):
        return self.h[0, self.bs1new//2, :, 0]
    
    @property
    def ph1d(self):
        return self.ph[0, self.bs1new//2, self.bs2new//2, :]

    # Some integrated quantities
    @property
    def mdot(self):
        return (-self.gdet * self.rho * self.uu[1] * self._dx2 * self._dx3).sum(-1).sum(-1)

    @property
    def edot_EM(self):
        T_rt = -self.bu[1]*self.bd[0]*(self.bsq/self.rho>2)
        return  (self.gdet*T_rt * self._dx2*self._dx3).sum(-1).sum(-1)
    
    @property
    def edot(self):
        w = self.rho + self.pgas + self.ug
        ud = np.sum(self.gcov*self.uu[:,None],axis=0)
        T_rt = (w+self.bsq)*self.uu[1]*ud[0] - self.bu[1]*self.bd[0]
        return  (self.gdet*T_rt * self._dx2*self._dx3).sum(-1).sum(-1)

    @property
    def Phi(self):
        return 0.5*(self.gdet * np.abs(self.B[1])*self._dx2*self._dx3).sum(-1).sum(-1)
    
    '''
    "Spherical Kerr-Schild" four-velocity
    Converted from "modified" "Spherical Kerr-Schild" coordinates
    '''
    def calc_uu_KS(self):
        self._uu_KS = np.einsum('ij...,j...->i...',self.dxdxp,self.uu) 
    @property
    def uu_KS(self):
        try:
            return self._uu_KS
        except:
            self.calc_uu_KS()
            return self._uu_KS
        
    '''
    Boyer-Lindquist four-velocity
    converted from "Spherical Kerr-Schild" coordinates
    '''
    def calc_uu_BL(self):
        self._uu_BL = np.einsum('ij...,j...->i...',self.dxBLdx,self.uu_KS) 
    @property
    def uu_BL(self):
        try:
            return self._uu_BL
        except:
            self.calc_uu_BL()
            return self._uu_BL
        
    '''
    Convert four-velocity to local Minkowski coordinates in the orthonormal tetrad of the equatorial circular orbit at the given r
    note: does not work for theta
    '''
    def calc_uu_LC(self):
        self._uu_LC  = np.einsum('ij...,j...->i...',self.dxedx,self.uu_BL)
    @property
    def uu_LC(self):
        try:
            return self._uu_LC
        except:
            self.calc_uu_LC()
            return self._uu_LC
    '''
    Lower uu_LC to ud_LC
    '''
    def calc_ud_LC(self):
        self._ud_LC = np.einsum('ij...,j...->i...',self.gcov_LC,self.uu_LC)
    @property
    def ud_LC(self):
        try:
            return self._ud_LC
        except:
            self.calc_ud_LC()
            return self._ud_LC
            
    def dump_visit(self, r_max, **kwargs):
        '''
        Save a vtk file for 3D visualizations in Visit.
        
        Args
            r_max: Maximum radius of saved data.
            kwargs: Variables names and arrays to save.
        '''
        from tvtk.api import tvtk, write_data
        from tvtk.tvtk_access import tvtk
    
        r_max_idx = np.argmin(np.abs(self.r1d - r_max))

        points = np.empty((idx + 1, bs2new, bs3new, 3,), dtype=float)
        points[..., 0] = self.x[0, :idx]
        points[..., 1] = self.y[0, :idx]
        points[..., 2] = self.z[0, :idx]

        points = points.transpose(2, 1, 0, 3).copy()
        points.shape = points.size // 3, 3

        for var_name, var_arr in kwargs.items():
            sg = tvtk.StructuredGrid(dimensions=(idx + 1, self.bs2new, self.bs3new), points=points)
            sg.point_data.scalars = np.copy(var_arr).T.ravel()
            sg.point_data.scalars.name = name
            write_data(sg, os.path.join(os.getcwd(), "visit/%s%d.vtk" % (var_name, self.D)))
    
    '''
    The following are helper functions for making calculations
    '''
    def avg(self, var, cond=None, coords=[1,2,3], weight=None):
        '''
        Computes average of quantity over radius and/or theta and/or phi
        
        Parameters
        var           quantity to average
        cond          condition on the region of integration
        coords        list of coordinates over which to average (e.g. [1,2,3] averages over r, theta, and phi)
        weight        weight in the case of a weighed average     
        '''
        if cond is None: cond = 1
        if weight is None: weight = 1
        coords = sorted(coords)
        _dx_list = [self._dx1, self._dx2, self._dx3]
        _dV = np.prod([_dx_list[coord - 1] for coord in coords])
        numerator = (var * cond * self.gdet * weight * _dV).sum(0)
        denomenator = (cond * self.gdet * weight * _dV).sum(0)
        for idx, coord in enumerate(coords):
            numerator = numerator.sum(coord - 1 - idx)
            denomenator = denomenator.sum(coord - 1 - idx)
        average = numerator / denomenator
        return average
    
    def sum(self, var, cond=1, coords=[1,2,3], weight=None):
        '''
        Computes sum of quantity over radius and/or theta and/or phi
        
        Parameters
        var           quantity to sum
        cond          condition on the region of integration
        coords        list of coordinates over which to sum (e.g. [1,2,3] averages over r, theta, and phi)
        weight        weight in the case of a weighed sum
        '''
        if cond is None: cond = 1
        if weight is None: weight = 1
        coords = sorted(coords)
        _dx_list = [self._dx1, self._dx2, self._dx3]
        _dV = np.prod([_dx_list[coord - 1] for coord in coords])
        tot = (var * cond * self.gdet * weight * _dV).sum(0)
        for idx, coord in enumerate(coords):
            tot = tot.sum(coord - 1 - idx)
        return tot
    
    def dxi(self, Q, i):
        '''
        Numerical derivative with respect to internal coordinate xi.
        '''
        assert i in [1,2,3], "i must be 1, 2, or 3"
        coord1d = [
            self.x1[0, :, self.bs2new//2, 0], 
            self.x2[0, self.bs1new//2, :, 0], 
            self.x3[0, self.bs1new//2, self.bs2new//2, :]
        ][i - 1]
        return np.gradient(Q, coord1d, axis=i)

    
    def curl(self, A):
        '''
        Curl in internal coordinates.
        Eq. 20 of https://www.jfoadi.me.uk/documents/lecture_mathphys2_05.pdf
        '''
        assert A.shape == (4, 1, self.bs1new, self.bs2new, self.bs3new), "Input must have shape %s" % str((4, 1, self.bs1new, self.bs2new, self.bs3new))
        curlA = np.zeros(A.shape)
        for i in range(1, 4):
            j = i % 3 + 1
            k = j % 3 + 1
            curlA[i] = (-1)**j * np.sqrt(self.gcov[i, i]) * ((self.dxi(np.sqrt(self.gcov[k, k]) * A[k], j) - self.dxi(np.sqrt(self.gcov[j, j]) * A[j], k)))
        curlA /= np.sqrt(self.gcov[1, 1] * self.gcov[2, 2] * self.gcov[3, 3])
        return curlA

    def div(self, A):
        '''
        Divergence in internal coordinates.
        Eq. 15 of https://www.jfoadi.me.uk/documents/lecture_mathphys2_05.pdf
        '''
        assert A.shape == (4, 1, self.bs1new, self.bs2new, self.bs3new), "Input must have shape %s" % str((4, 1, self.bs1new, self.bs2new, self.bs3new))
        divA = np.zeros(A.shape[1:])
        for i in range(1, 4):
            j = i % 3 + 1
            k = j % 3 + 1
            divA += self.dxi(A[i] * np.sqrt(self.gcov[j, j]) * np.sqrt(self.gcov[k, k]), i)
        divA /= np.sqrt(self.gcov[1, 1] * self.gcov[2, 2] * self.gcov[3, 3])
        return divA

    def grad(self, Q):
        '''
        Gradient in internal coordinates.
        Eq. 10 of https://www.jfoadi.me.uk/documents/lecture_mathphys2_05.pdf
        '''
        assert Q.shape == (1, self.bs1new, self.bs2new, self.bs3new), "Input must have shape %s" % str((1, self.bs1new, self.bs2new, self.bs3new))
        gradQ = np.zeros((4,) + tuple(Q.shape))
        gradQ[1:4] = np.array([self.dxi(Q, i) / np.sqrt(self.gcov[i, i]) for i in range(1, 4)])
        return gradQ

    def laplace(self, Q):
        '''
        Laplaxian in internal coordinates.
        '''
        assert Q.shape == (1, self.bs1new, self.bs2new, self.bs3new), "Input must have shape %s" % str((1, self.bs1new, self.bs2new, self.bs3new))
        return self.div(self.grad(Q))
    
    def _lower(self,upper):
        '''
        Turns contravariant 4-vector into covariant 4-vector
        
        Parameters
        upper       contravariant 4-vector (first axis should be size 4)
        
        Returns
        lower       covariant 4-vector (same shape as upper)
        '''
        return np.sum(self.gcov*upper[:,None],axis=0)

    
    def _interpolate_2D(self, field, mode, width, resolution=51):
        '''
        Interpolates a field to a two-dimensional cartesian grid. 
        
        To do: 
            add support for mode='phi' with alternative phi values. when mode='phi' this function implicitly only supports phi=0 i.e. the x-z plane.
            add support for non-structured grids. 
            add interpolation options other than 'nearest', order=1

        Parameters
        field         to interpolate to cartesian grid
        mode          interpolate onto a cartesian grid with fixed phi (x-z plot) or fixed theta plot (x-y plot) ['phi' or 'theta']
        width         width of cartesian grid in code units
        resolution    resolution of cartesian grid that velocity fields are interpolated onto.       
        
        Returns
        x_ax            x values of 2D cartesian grid
        y_ax (or z_ax)  y values of 2D cartesian grid (does not necessarily refer to 'y', depending on mode)
        f_ax            field values of 2D cartesian grid
        '''

        ## Check if mode is correct
        if mode != 'phi' and mode != 'theta':
            raise ValueError('[Simulation._interpolate_2D] You passed mode = \'%s\'. Valid modes are \'phi\' or \'theta\'' % mode)

        ## If in x-y plane (theta-hat is normal vector to plane)
        if mode == 'theta':
            ## Produce 1-D vectors for grid cells
            x_ax = np.linspace(-width/2.0,width/2.0,resolution)
            y_ax = np.linspace(-width/2.0,width/2.0,resolution)

            ## Produce 2-D grid with x/y positions and flatten into 1-D arrays
            x_ax,y_ax=np.meshgrid(x_ax,y_ax,indexing="ij")
            x_ax = x_ax.flatten()
            y_ax = y_ax.flatten()

            ## Get corresponding spherical vector positions on cartesian grid
            r_ax = np.sqrt(x_ax**2.0 + y_ax**2.0)
            p_ax = np.arctan2(y_ax,x_ax)
            p_ax[p_ax<0.0] += 2.0*np.pi


            ## Assuming a uniform grid in x1/x2/x3, create 1-D grid arrays of interpolated indices
            ## since mode=='theta', the 'j' index (corresponding to the theta value of the grid) is fixed
            i_ax = (self.bs1new - 1)/(np.max(np.log(self.r))-np.min(np.log(self.r)))*(np.log(r_ax + 1e-3) - np.min(np.log(self.r)))
            j_ax = self.bs2new//2
            k_ax = (self.bs3new - 1)/(np.max(self.ph)-np.min(self.ph))*(p_ax - np.min(self.ph))

            ## Reshape interpolated i and k indices back into 2-D grid
            i_ax = np.reshape(i_ax,(resolution,resolution))
            k_ax = np.reshape(k_ax,(resolution,resolution))

            ## Reshape flattened x and y grid arrays back into 2-D grid
            x_ax = np.reshape(x_ax,(resolution,resolution))
            y_ax = np.reshape(y_ax,(resolution,resolution))

            ## Map the field, at fixed theta coordinates, onto interpolated i and k grid
            f_ax = ndimage.map_coordinates(field[0,:,j_ax,:],np.array([i_ax,k_ax]),order=1,mode="nearest")

            return x_ax,y_ax,f_ax     
        ## If in x-z (or y-z) plane (phi-hat is normal vector to plane)
        if mode == 'phi':
            ## Ensure requested resolution is even. This is necessary for how the arrays are constructed to the left and right of the polar axis. 
            resolution -= resolution % 2

            ## Produce 1-D vectors for grid cells. Since a fixed phi value refers to one side of the polar axis, we need to split the arrays into left (vectors labeled '_L') and right regions (vectors labeled '_R'). 
            x_ax_L = np.linspace(-width/2,0,resolution//2,endpoint=False)
            x_ax_R = np.linspace(0,width/2,resolution//2+1)
            x_ax   = np.linspace(-width/2,width/2,resolution+1)
            z_ax   = np.linspace(-width/2.0,width/2.0,resolution+1)

            ## Produce 2-D grid with x/y positions and flatten into 1-D arrays
            x_ax_L,z_ax_L = np.meshgrid(x_ax_L,z_ax,indexing="ij")
            x_ax_R,z_ax_R = np.meshgrid(x_ax_R,z_ax,indexing="ij")
            x_ax,z_ax     = np.meshgrid(x_ax,z_ax,indexing="ij")
            x_ax_L        = x_ax_L.flatten()
            x_ax_R        = x_ax_R.flatten()
            z_ax_L        = z_ax_L.flatten()
            z_ax_R        = z_ax_R.flatten()


            ## Get corresponding spherical vector positions on cartesian grid
            r_ax_L = np.sqrt(x_ax_L**2.0 + z_ax_L**2.0)
            r_ax_R = np.sqrt(x_ax_R**2.0 + z_ax_R**2.0)
            h_ax_L = np.arccos(z_ax_L/r_ax_L)
            h_ax_R = np.arccos(z_ax_R/r_ax_R)

            ## Assuming a uniform grid in x1/x2/x3, create 1-D grid arrays of interpolated indices
            ## since mode=='phi', the 'k' index (corresponding to the phi value of the grid) is fixed. However, there are two values for the left and right regions, and one k value is rotated by ~180 degrees from the other.
            i_ax_L = (self.bs1new - 1)/(np.max(np.log(self.r))-np.min(np.log(self.r)))*(np.log(r_ax_L + 1e-3) - np.min(np.log(self.r)))
            j_ax_L = (self.bs2new - 1)/(np.max(self.h)-np.min(self.h))*(h_ax_L - np.min(self.h))
            i_ax_R = (self.bs1new - 1)/(np.max(np.log(self.r))-np.min(np.log(self.r)))*(np.log(r_ax_R + 1e-3) - np.min(np.log(self.r)))
            j_ax_R = (self.bs2new - 1)/(np.max(self.h)-np.min(self.h))*(h_ax_R - np.min(self.h))
            k_ax_L = self.bs3new//2
            k_ax_R = 0
            
            ## Reshape interpolated i and j indices back into 2-D grid
            i_ax_L = np.reshape(i_ax_L,(resolution//2,resolution+1))
            i_ax_R = np.reshape(i_ax_R,(resolution//2+1,resolution+1))
            j_ax_L = np.reshape(j_ax_L,(resolution//2,resolution+1))
            j_ax_R = np.reshape(j_ax_R,(resolution//2+1,resolution+1))

            ## Reshape flattened x and z grid arrays back into 2-D grid
            x_ax_L = np.reshape(x_ax_L,(resolution//2,resolution+1))
            x_ax_R = np.reshape(x_ax_R,(resolution//2+1,resolution+1))
            z_ax_L = np.reshape(z_ax_L,(resolution//2,resolution+1))
            z_ax_R = np.reshape(z_ax_R,(resolution//2+1,resolution+1))

            ## Map the field, at fixed theta coordinates, onto interpolated i and j grid in left and right regions
            f_ax_L = ndimage.map_coordinates(field[0,:,:,k_ax_L],np.array([i_ax_L,j_ax_L]),order=1,mode="nearest")
            f_ax_R = ndimage.map_coordinates(field[0,:,:,k_ax_R],np.array([i_ax_R,j_ax_R]),order=1,mode="nearest")
            ## Combine left and right regions into a single grid
            f_ax   = np.vstack((f_ax_L,f_ax_R))
            
            return x_ax,z_ax,f_ax

    
    def _interpolate_3D(self, field, width, resolution=51):
        '''
        to do: UPDATE DESCRIPTION
                ADD COMMENTS
        assumes uniform grid in log(r)/theta/phi
        
        extrapolates to field to x/y/z grid and returns x, y, and z
        
        to do: should be generalized to (i) non uniform grids using griddata(), (ii) off-center grids, (iii), other modes/orders to
        pass to map_coordinates, (iv) only 2D grid with rotations (that cane be a diff function)
        '''

        x_ax = np.linspace(-width/2.0,width/2.0,resolution)
        y_ax = np.linspace(-width/2.0,width/2.0,resolution)
        z_ax = np.linspace(-width/2.0,width/2.0,resolution)

        x_ax,y_ax,z_ax=np.meshgrid(x_ax,y_ax,z_ax,indexing="ij")
        x_ax = x_ax.flatten()
        y_ax = y_ax.flatten()
        z_ax = z_ax.flatten()

        r_ax = np.sqrt(x_ax**2.0 + y_ax**2.0 + z_ax**2.0)
        h_ax = np.arccos(z_ax/r_ax)
        p_ax = np.arctan2(y_ax,x_ax)
        p_ax[p_ax<0.0] += 2.0*np.pi

        i_data = np.arange(self.bs1new)
        j_data = np.arange(self.bs2new)
        k_data = np.arange(self.bs3new)

        i_ax = (self.bs1new - 1)/(np.max(np.log(self.r))-np.min(np.log(self.r)))*(np.log(r_ax + 1e-3) - np.min(np.log(self.r)))
        j_ax = (self.bs2new - 1)/(np.max(self.h)-np.min(self.h))*(h_ax - np.min(self.h))
        k_ax = (self.bs3new - 1)/(np.max(self.ph)-np.min(self.ph))*(p_ax - np.min(self.ph))


        i_ax = np.reshape(i_ax,(resolution,resolution,resolution))
        j_ax = np.reshape(j_ax,(resolution,resolution,resolution))
        k_ax = np.reshape(k_ax,(resolution,resolution,resolution))

        x_ax = np.reshape(x_ax,(resolution,resolution,resolution))
        y_ax = np.reshape(y_ax,(resolution,resolution,resolution))
        z_ax = np.reshape(z_ax,(resolution,resolution,resolution))

        f_ax = ndimage.map_coordinates(field[0],np.array([i_ax,j_ax,k_ax]),order=1,mode="nearest")

        return x_ax,y_ax,z_ax,f_ax

        
        
    '''
    The following are helper functions for producing plots.
    '''
    
    def _add_grid_to_ax(self,ax,mode='phi',phi_offset=0):
        '''
        Helper matplotlib function to plot the block boundaries on an axis.
        
        Note: Only works if unflattened coordinates arrays were saved on initialization with flatten=0 or flatten=2.
        
        Parameters
        ax            matplotlib axis to draw plot onto
        mode          produce a plot with fixed phi (x-z plot) or fixed theta plot (x-y plot) ['phi' or 'theta']
        phi_offset    if mode='theta', this sets the phi slice to plot [degrees]
        '''
        if self.flatten == 1:
            raise ValueError("[Simulation._add_grid_to_ax] Unflattened coordiantes arrays not available. Initialize the Simulation object with keyword argument flatten=0 or flatten=2 to save unflattened coordinate arrays.")
        
        ## Check if mode is correct
        if mode != 'phi' and mode != 'theta':
            raise ValueError('[Simulation._add_grid_to_ax] You passed mode = \'%s\'. Valid modes are \'phi\' or \'theta\'' % mode)

        ## Fixed value of phi (i.e., x-z/y-z plots)
        if mode == 'phi':

            phi_offset = phi_offset*np.pi/180

            # get indices of blocks in the vertical plane
            vertical_blocks = np.where(np.min(np.abs(self.phflat[:, :, -1, -1] - phi_offset), axis=1) <= 10*np.min(np.abs(self.phflat[:, :, -1, -1] - phi_offset)))
            
            for block in vertical_blocks[0]:
                r_min, r_max = np.min(self.rflat[block]), np.max(rflat[block])
                h_min, h_max = np.min(self.hflat[block]), np.max(self.hflat[block])
                r_space = np.linspace(r_min, r_max, 100)
                h_space = np.linspace(h_min, h_max, 100)
                ax.plot(r_space * np.sin(h_min), r_space * np.cos(h_min), color="black")
                ax.plot(r_space * np.sin(h_max), r_space * np.cos(h_max), color="black")
                ax.plot(r_min * np.sin(h_space), r_min * np.cos(h_space), color="black")
                ax.plot(r_max * np.sin(h_space), r_max * np.cos(h_space), color="black")
                
        ## only x-y plane is available
        if mode == 'theta':
            
            # get indices of blocks in the equatorial plane
            equatorial_blocks = np.where(np.min(np.abs(self.hflat[:, -1, :, -1] - np.pi/2), axis=1) <= 10*np.min(np.abs(self.hflat[:, -1, :, -1] - np.pi/2)))
            
            for block in equatorial_blocks[0]:
                r_min, r_max = np.min(self.rflat[block]), np.max(self.rflat[block])
                ph_min, ph_max = np.min(self.phflat[block]), np.max(self.phflat[block])
                r_space = np.linspace(r_min, r_max, 100)
                ph_space = np.linspace(ph_min, ph_max, 100)
                ax.plot(r_space * np.cos(ph_min), r_space * np.sin(ph_min), color="black")
                ax.plot(r_space * np.cos(ph_max), r_space * np.sin(ph_max), color="black")
                ax.plot(r_min * np.cos(ph_space), r_min * np.sin(ph_space), color="black")
                ax.plot(r_max * np.cos(ph_space), r_max * np.sin(ph_space), color="black")
            
    
    def _contourf(self,field,fmin,fmax,width,mode='phi',figsize=(10,8),phi_offset=0,fill_cells=1,dolog=1,cmap='jet',internal_grid=0,stream=None,bh=True,twoD=False,axes_markers=False,aspect=True,cbar=None):
        '''
        Helper matplotlib function to produce a single contourf plot. 

        Parameters
        field         field to visualize [str, must be attribute of class, i.e. 'rho' will plot self.rho]
        fmin          minimum value of field in linear space
        fmax          maximum value of field in linear space
        width         sets xlim, ylim to +-width/2. If left as None, xlim/ylim aren't set. 
        mode          produce a plot with fixed phi (x-z plot) or fixed theta plot (x-y plot) ['phi' or 'theta']
        figsize       size of figure [tuple of x and y size, i.e., (10, 8)]
        phi_offset    if mode='theta', this sets the phi slice to plot [degrees]
        fill_cells    whether or not to 'squeeze' final cells in theta/phi to remove whitespace from plots [0 or 1]
        dolog         whether or not to plot the log10 of the field [0 or 1]
        internal_grid whether or not to plot in internal coordinates
        stream        if a dictionary is passed, create streamlines according to dictionary values. See Simulation._add_streamplot_to_ax for details. 
        bh            whether or not to plot the BH and ergosphere
        twoD          whether or not field is 2D
        axes_markers  whether or not to show axes with dotted lines
        cbar          if dictionary is passed, add a colorbar to the figure

        Returns
        fig           matplotlib figure
        ax            matplotlib axis with newly drawn plot
        c             instance of contourf (useful if you want to make modifications to the plot after calling the function)
        aspect:       whether to use equal aspect ratio
        '''

        #### Begin constructing plot

        fig = plt.figure(figsize=figsize)
        ax  = fig.add_subplot(111)

        ax, c = self._add_contourf_to_ax(ax, field,fmin,fmax,width,mode=mode,phi_offset=phi_offset,fill_cells=fill_cells,dolog=dolog,cmap=cmap,internal_grid=internal_grid,stream=stream,bh=bh,twoD=twoD,axes_markers=axes_markers,aspect=aspect)
        
        if mode == "phi":
            ax.set_xlabel(r'$s\ [r_{\rm g}]$')
            ax.set_ylabel(r'$z\ [r_{\rm g}]$')
        if mode == "theta":
            ax.set_xlabel(r'$x\ [r_{\rm g}]$')
            ax.set_ylabel(r'$y\ [r_{\rm g}]$')
            
        if cbar is not None:
            label = cbar.pop("label", "")
            cb = fig.colorbar(c, ax=ax, **cbar)
            cb.set_label(label)

        ## return figure, axis, and instance of contourf
        return fig, ax, c

    def _contourf_grid(self,field,fmin,fmax,width,mode='phi',figsize=(10,8),phi_offset=0,fill_cells=1,dolog=1,cmap='jet',rows=1,cols=1,internal_grid=0,stream=None,bh=True,twoD=False, axes_markers=False,aspect=True,cbar=None):
        '''
        Helper matplotlib function to produce a grid of contourf plot. 

        Note: each parameter can be passed with its intended type, or a list of its intended type. If you pass a list, the length of the list
        needs to be the same as the total number of plots (= rows * cols). 
        
        to do:
              add list support for internal grid

        Parameters
        field         field to visualize [str, must be attribute of class, i.e. 'rho' will plot self.rho]
        fmin          minimum value of field in linear space
        fmax          maximum value of field in linear space
        width         sets xlim, ylim to +-width/2
        mode          produce a plot with fixed phi (x-z plot) or fixed theta plot (x-y plot) ['phi' or 'theta']
        figsize       size of figure [tuple of x and y size, i.e., (10, 8)]
        phi_offset    if mode='theta', this sets the phi slice to plot [degrees]
        fill_cells    whether or not to 'squeeze' final cells in theta/phi to remove whitespace from plots [0 or 1]
        dolog         whether or not to plot the log10 of the field [0 or 1]
        rows          number of rows of plots
        cols          number of cols of plots
        internal_grid whether or not to plot in internal coordinates (CURRENTLY DOES NOT SUPPORT LIST OPTIONS FOR internal grid = 0 or 1 FOR DIFFERENT AXES)
        stream        if a dictionary is passed, createstreamlines according to dictionary values. See Simulation._add_streamplot_to_ax for details. 
        bh            whether or not to plot the BH and ergosphere
        twoD          whether or not field is 2D
        axes_markers  whether or not to show axes with dotted lines
        aspect:       whether to use equal aspect ratio
        cbar:         if dictionary is passed, add a colorbar to the figure

        Returns
        fig           matplotlib figure
        axs           flattened list of matplotlib axes with newly drawn plots
        cs            list of contourf instances for each axis (useful if you want to make modifications to the plot after calling the function)
        '''

        ## total number of plots
        nplots = rows*cols

        assert nplots > 1, '[Simulation._contourf_grid] nplots = 1 (you set rows=cols=1). If you want a single plot, use _contourf, not contourf_grid.'

        fig,axs = plt.subplots(rows,cols,figsize=figsize)
        print(type(axs),np.shape(axs))
        axs     = axs.flatten()
        cs      = [] # instances of contourf, to be filled


        ## Each plot argument can be single-valued (applied to each axis) or can be an array of values
        ## Here we create functions to check this

        # for string arguments
        def assert_str(arg, i):
            if type(arg) == str:
                return arg
            else:
                assert len(arg) == nplots, '[Simulation._contourf_grid] One of your arguments was passed as a list with length %d, but needs to be single-valued or with length equal to the number of plots, %d' % (len(arg),nplots)
                return arg[i]

        # for integer or float arguments
        def assert_num(arg, i):
            try: 
                tmp_arg = arg[i]
                if len(arg) == nplots: 
                    error_flag = 1
                else:
                    error_flag = 0    
            except:
                tmp_arg = np.copy(arg)
                error_flag = 1
            assert error_flag, '[Simulation._contourf_grid] One of your arguments was passed as a list with length %d, but needs to be single-valued or with length equal to the number of plots, %d' % (len(arg),nplots)
            return tmp_arg
        
        # for dictionary arguments
        def assert_dict(arg, i):
            if type(arg) == dict:
                return arg
            else:
                assert len(arg) == nplots, '[Simulation._contourf_grid] One of your arguments was passed as a list with length %d, but needs to be single-valued or with length equal to the number of plots, %d' % (len(arg),nplots)
                return arg[i]

        for i in range(nplots):
            # get each argument for this axis
            tmp_field      = assert_str(field,i)
            tmp_mode       = assert_str(mode,i)
            tmp_cmap       = assert_str(cmap,i)
            tmp_fmin       = assert_num(fmin,i)
            tmp_fmax       = assert_num(fmax,i)
            tmp_phi_offset = assert_num(phi_offset,i)
            tmp_fill_cells = assert_num(fill_cells,i)
            tmp_dolog      = assert_num(dolog,i)
            tmp_width      = assert_num(width,i)
            if stream is None: 
                tmp_stream = None
            else:
                tmp_stream = assert_dict(stream,i)
            if cbar is None or type(cbar) == dict:
                tmp_cbar = None
            else:
                tmp_cbar = assert_dict(cbar,i)

            axs[i], c = self._add_contourf_to_ax(axs[i],tmp_field,tmp_fmin,tmp_fmax,tmp_width,mode=tmp_mode,phi_offset=tmp_phi_offset,
                                                 fill_cells=tmp_fill_cells,dolog=tmp_dolog,cmap=tmp_cmap, internal_grid=internal_grid, stream=tmp_stream, bh=bh, twoD=twoD, axes_markers=axes_markers,aspect=aspect)
            cs.append(c)

            if tmp_mode == "phi":
                axs[i].set_xlabel(r'$s\ [r_{\rm g}]$')
                axs[i].set_ylabel(r'$z\ [r_{\rm g}]$')
            if tmp_mode == "theta":
                axs[i].set_xlabel(r'$x\ [r_{\rm g}]$')
                axs[i].set_ylabel(r'$y\ [r_{\rm g}]$')
                
            if tmp_cbar is not None:
                label = tmp_cbar.pop("label", "")
                cb = fig.colorbar(cs[i], ax=axs[i], **tmp_cbar)
                cb.set_label(label)
                
        if type(cbar) == dict:
            label = cbar.pop("label", "")
            cb = fig.colorbar(cs[-1], ax=axs, **cbar)
            cb.set_label(label)
        
        return fig, axs, cs

    def _add_contourf_to_ax(self,ax,field,fmin,fmax,width,mode='phi',phi_offset=0,fill_cells=1,dolog=1,cmap='jet', internal_grid=0, stream=None, bh=True, twoD=False, axes_markers=False,aspect=True):
        '''
        Helper matplotlib function to produce a contourf plot. This is should be called by another function that produces a figure and
        axes, and this draws the contourf on the axis.

        Parameters
        ax            matplotlib axis to draw plot onto
        field         field to visualize [str, must be attribute of class, i.e. 'rho' will plot self.rho]
        fmin          minimum value of field in linear space
        fmax          maximum value of field in linear space
        width         sets xlim, ylim to +-width/2 
        mode          produce a plot with fixed phi (x-z plot) or fixed theta plot (x-y plot) ['phi' or 'theta']
        phi_offset    if mode='theta', this sets the phi slice to plot [degrees]
        fill_cells    whether or not to 'squeeze' final cells in theta/phi to remove whitespace from plots [0 or 1]
        dolog         whether or not to plot the log10 of the field [0 or 1]
        stream        if a dictionary is passed, create streamlines according to dictionary values. See Simulation._add_streamplot_to_ax for details. 
        bh            whether or not to plot the BH and ergosphere
        twoD          whether or not field is 2D
        axes_markers  whether or not to show axes with dotted lines
        aspect:       whether to use equal aspect ratio

        Returns
        ax            matplotlib axis with newly drawn plot
        c             instance of contourf (useful if you want to make modifications to the plot after calling the function)
        '''

        #### Some checks to make sure function will work as intended: 

        ## Check if blocks are flattened
        assert self.parent.nb == 1, '[Simulation._contour_plot] Number of blocks is > 1. Currently, Simulation._contour_plot() only works for a flattened grid. You can do this by calling Simulation._flatten_blocks().'

        ## Check if field is a possible choice
        ## a @property named "v" will not show up in the attribute dictionary, so this is evaluated in the 'except' clause. 
        if type(field)==str:
            if hasattr(self,field):
                try: 
                    f = self.__dict__[field]
                except:
                    f = getattr(self.__class__, field).__get__(self)
            else:
                raise ValueError('[Simulation._contour_plot] You passed field = \'%s\'. The Simulation class does not have this attribute. Possible fields are \'rho\', \'bsq\', etc' % field)
        else:
            f = field


        ## Check if mode is correct
        if mode != 'phi' and mode != 'theta':
            raise ValueError('[Simulation._contour_plot] You passed mode = \'%s\'. Valid modes are \'phi\' or \'theta\'' % mode)

        
        ## Dual mode allows plotting of quantities with positive and negative values on a log scale
        dual = False
        
        if dolog:
            if np.min(f) < 0:
                dual = True
                fp, fm    = np.log10(f), np.log10(-f)
            else:
                f    = np.log10(f)
            fmin = np.log10(fmin)
            fmax = np.log10(fmax)

        ## create range of colormap from fmin, fmax
        levels = np.arange(fmin, fmax, (fmax-fmin)/300.0)

        ## Fixed value of phi (i.e., x-z/y-z plots)
        if mode == 'phi':
            ## convert phi offset from degrees to radians
            phi_offset = phi_offset*np.pi/180

            phis    = np.copy(self.ph[0,0,0,:])
            ph_ind  = np.argmin(np.abs(phis-phi_offset))
            ph_ind2 = (ph_ind + self.bs3new//2) % self.bs3new

            plot_h = np.copy(self.h)


            ## Normally, the theta direction doesn't extend exactly from 0 to pi
            ## This step stretches the first/last cells so that the theta direction does extend from 0 to pi
            ## this removes white-space at the poles
            if fill_cells:
                hmax, hmin = plot_h.max(), plot_h.min()
                dhmax = np.pi - hmax
                dhmin = 0       - hmin
                plot_h[:,:,0,:]  += dhmin
                plot_h[:,:,-1,:] += dhmax

            ## create arrays to plot contourf on
            plot_x = self.r*np.sin(plot_h)
            plot_z = self.r*np.cos(plot_h)

            if twoD:
                f2d = f
                f = np.zeros_like(self.rho)
                f[:, :, :, :] = f2d[None, :, :, None]
            
            ## create plot
            if internal_grid:
                if dual:
                    c = ax.contourf(self.x1[0, :, :, ph_ind2], self.x2[0, :, :, ph_ind2],fp[0,:,:,ph_ind2],levels=levels,cmap="Reds",extend='both')
                    ax.contourf(self.x1[0, :, :, ph_ind2], self.x2[0, :, :, ph_ind2],fm[0,:,:,ph_ind2],levels=levels,cmap="Blues",extend='both')
                else:
                    c = ax.contourf(self.x1[0, :, :, ph_ind2], self.x2[0, :, :, ph_ind2],f[0,:,:,ph_ind2],levels=levels,cmap=cmap,extend='both')
            else:
                c = ax.contourf(plot_x[0, :, :,  ph_ind],plot_z[0, :, :, ph_ind],f[0,:,:,ph_ind],levels=levels,cmap=cmap,extend='both')
                for cline in c.collections:
                    cline.set_edgecolor('none')
                    cline.set_linewidth(0)
                c = ax.contourf(-plot_x[0, :, :, ph_ind2], plot_z[0, :, :, ph_ind2],f[0,:,:,ph_ind2],levels=levels,cmap=cmap,extend='both')
            for cline in c.collections:
                cline.set_edgecolor('none')
                cline.set_linewidth(0)
        if mode == 'theta':
            ## only x-y plane is available

            plot_ph = np.copy(self.ph)

            ## Normally, the phi direction doesn't extend exactly from 0 to 2pi
            ## This step stretches the first/last cells so that the phi direction does extend from 0 to 2pi
            ## this removes white-space at the borders
            if fill_cells:
                phmax, phmin = plot_ph.max(), plot_ph.min()
                dphmax = 2*np.pi - phmax
                dphmin = 0       - phmin
                plot_ph[:,:,:,0]  += dphmin
                plot_ph[:,:,:,-1] += dphmax

            ## create arrays to plot contourf on
            plot_x = self.r*np.cos(plot_ph)
            plot_y = self.r*np.sin(plot_ph)


            if twoD:
                f2d = f
                f = np.zeros_like(self.rho)
                f[:, :, :, :] = f2d[None, :, None, :]
            
            ## create plot
            if internal_grid:
                if dual:
                    c = ax.contourf(self.x1[0, :, self.bs2new//2, :], self.x3[0, :, self.bs2new//2, :],fp[0,:, self.bs2new//2,:],levels=levels,cmap="Reds",extend='both')
                    ax.contourf(self.x1[0, :, self.bs2new//2, :], self.x3[0, :, self.bs2new//2, :],fm[0,:, self.bs2new//2,:],levels=levels,cmap="Blues",extend='both')
                else:
                    c = ax.contourf(self.x1[0, :, self.bs2new//2, :], self.x3[0, :, self.bs2new//2, :],f[0,:, self.bs2new//2,:],levels=levels,cmap=cmap,extend='both')
            else:
                c = ax.contourf(plot_x[0, :, self.bs2new//2, :], plot_y[0, :, self.bs2new//2, :],f[0,:, self.bs2new//2,:],levels=levels,cmap=cmap,extend='both')
            for cline in c.collections:
                cline.set_edgecolor('none')
                cline.set_linewidth(0)

        ax.set_xlim(-width/2,width/2)
        ax.set_ylim(-width/2,width/2)

        if aspect: ax.set_aspect('equal')
        
        ## make contour borders invisible
        for contour in c.collections:
            contour.set_edgecolor("face")
        
        ## add optional streamlines to axis
        if stream is not None: self._add_streamplot_to_ax(ax, mode, width, stream=stream)
        
        ## add optional BH to axis
        if bh: self._add_bh_to_ax(ax, mode)
        
        ## add optional axes markers
        if axes_markers:
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
            
        return ax,c
    
    def _add_bh_to_ax(self, ax, mode):
        '''
        Helper matplotlib function to add BH and ergosphere to a contourf plot. This will draw the event horizon and ergosphere ontop of the countour plot.
        
        Parameters
        ax            matplotlib axis to draw the BH onto
        mode          produce a plot with fixed phi (x-z plot) or fixed theta plot (x-y plot) ['phi' or 'theta']
        
        Returns
        None
        '''
        
        ## Check if mode is correct
        if mode != 'phi' and mode != 'theta':
            raise ValueError('[Simulation._add_bh_to_ax] You passed mode = \'%s\'. Valid modes are \'phi\' or \'theta\'' % mode)
        
        ## If in x-y plane (theta-hat is normal vector to plane)
        if mode == 'theta':
        
            r_ergo = 2

            hor = plt.Circle((0, 0), self.rhor, color='black')
            ergo = plt.Circle((0, 0), r_ergo, color='black', alpha=0.2)
            ax.add_patch(hor)
            ax.add_patch(ergo)
            
            return
        
        ## If in x-z (or y-z) plane (phi-hat is normal vector to plane)
        if mode == 'phi':
            
            hor = plt.Circle((0, 0), self.rhor, color='black')

            h_ergo = np.linspace(0, 2 * np.pi, 100)
            r_ergo = 1 + np.sqrt(1 - self.a**2 * np.cos(h_ergo)**2)
            s_ergo = r_ergo * np.sin(h_ergo)
            z_ergo = r_ergo * np.cos(h_ergo)
        
            ax.add_patch(hor)
            ax.fill(s_ergo, z_ergo, "black", alpha=0.2)
    
    def _add_streamplot_to_ax(self, ax, mode, width, stream = {}, start_points=None):
        '''
        Helper matplotlib function to add streamlines to a contourf plot. This will draw streamlines of the desired type ontop of the contourf plot. 
        
        To do: 
            add support for magnetic field streamlines
            add support for mode='phi' with alternative phi values. when mode='phi' this function implicitly only supports phi=0 i.e. the x-z plane.

        Parameters
        ax            matplotlib axis to draw streamlines onto
        mode          produce a plot with fixed phi (x-z plot) or fixed theta plot (x-y plot) ['phi' or 'theta']
        width         width of plot in code units
        stream        dictionary that holds information for creating streamlines. All values are optional and have defaults (see below). Options are as follows. 
                      Descriptions for plt.streamplot values taken from https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.streamplot.html#matplotlib.pyplot.streamplot. 
                        'type'        Vector field to construct streamlines from. [only 'velocity' supported currently]
                        'color'       The streamline color. If given an array, its values are converted to colors using cmap and norm. The array must have the same shape as velocity fields. 
                        'density'     Controls the closeness of streamlines. When density = 1, the domain is divided into a 30x30 grid. density linearly scales this grid. Each cell in the grid can have, at most, one traversing streamline. For different densities in each direction, use a tuple (density_x, density_y). 
                        'linewidth'   The width of the stream lines. With a 2D array the line width can be varied across the grid. The array must have the same shape as velocity fields.
                        'maxlength'   Maximum length of streamline in axes coordinates.
                        'resolution'  Resolution of cartesian grid that velocity fields are interpolated onto. 
        start_points    array of start points for the streamlines
                        
        Returns
        strm            matplotlib StreamPlotSet object
        '''
    
        ## Check if mode is correct
        if mode != 'phi' and mode != 'theta':
            raise ValueError('[Simulation._add_streamplot_to_ax] You passed mode = \'%s\'. Valid modes are \'phi\' or \'theta\'' % mode)
    
        ## Set default values for stream dictionary
        stream.setdefault('type','velocity')
        stream.setdefault('color','black')
        stream.setdefault('density',1)
        stream.setdefault('linewidth',1)
        stream.setdefault('maxlength',4)
        stream.setdefault('resolution',200)
        
        
        ## Ensure specified type of streamlines is supported. 
        if stream['type'] != 'velocity' and stream['type'] != 'magnetic_field':
            raise ValueError('[Simulation._add_streamplot_to_ax] You passed stream[\'type\'] = \'%s\'. Valid streamline types are \'velocity\' and \'magnetic_field\'' % stream['type'])
        
        ## If in x-y plane (theta-hat is normal vector to plane)
        if mode == 'theta':
            ## construct velocity streamlines
            if stream['type'] == 'velocity':
                ## Interpolate streamline vectors to 2D cartesian grid
                x, y, sx = self._interpolate_2D(self.vx, mode, width,resolution=stream['resolution'])
                x, y, sy = self._interpolate_2D(self.vy, mode, width,resolution=stream['resolution'])
            ## construct magnetic field streamlines
            if stream['type'] == 'magnetic_field':
                ## Interpolate streamline vectors to 2D cartesian grid
                x, y, sx = self._interpolate_2D(self.Bx, mode, width,resolution=stream['resolution'])
                x, y, sy = self._interpolate_2D(self.By, mode, width,resolution=stream['resolution'])
    
            ## Add streamlines to axis
            kwargs = dict(color=stream['color'], density=stream['density'], maxlength=stream['maxlength'], linewidth=stream['linewidth'])
            if start_points is not None: kwargs['start_points'] = start_points
            strm = ax.streamplot(x.T, y.T, sx.T, sy.T, **kwargs)

            return strm
        ## If in x-z (or y-z) plane (phi-hat is normal vector to plane)
        if mode == 'phi':
            ## construct velocity streamlines
            if stream['type'] == 'velocity':
                ## Interpolate streamline vectors to 2D cartesian grid
                x, y, sx = self._interpolate_2D(self.vx, mode, width,resolution=stream['resolution'])
                x, y, sz = self._interpolate_2D(self.vz, mode, width,resolution=stream['resolution'])
            ## construct magnetic field streamlines
            if stream['type'] == 'magnetic_field':
                ## Interpolate streamline vectors to 2D cartesian grid
                x, y, sx = self._interpolate_2D(self.Bx, mode, width,resolution=stream['resolution'])
                x, y, sz = self._interpolate_2D(self.Bz, mode, width,resolution=stream['resolution'])

            ## Add streamlines to axis
            kwargs = dict(color=stream['color'], density=stream['density'], maxlength=stream['maxlength'], linewidth=stream['linewidth'])
            if start_points is not None: kwargs['start_points'] = start_points
            strm = ax.streamplot(x.T, y.T, sx.T, sz.T, **kwargs)

            return strm

    
    def _get_along_streamline(self, fields, ax, mode, width, start_point, stream = {}):
        '''
        Helper matplotlib function to get the interpolate fields onto a streamline. This will also draw the relevant streamline.
        
        Parameters
        fields                list of fields to interpolate along streamline
        ax                    matplotlib axis to draw streamline onto
        mode                  produce a plot with fixed phi (x-z plot) or fixed theta plot (x-y plot) ['phi' or 'theta']
        width                 width of plot in code units
        stream                if a dictionary is passed, create streamlines according to dictionary values. See Simulation._add_streamplot_to_ax for details.
        start_point           start points for the streamline
        
        Returns
        fields_along_strm     list of arrays of variables along the streamline
        '''
        from scipy.interpolate import interp2d
        
        if type(fields) != list: fields = [fields]
        for i, field in enumerate(fields): 
            if type(field)==str:
                if hasattr(self, field):
                    try: 
                        fields[i] = self.__dict__[field]
                    except:
                        fields[i] = getattr(self.__class__, field).__get__(self)
                else:
                    raise ValueError('[Simulation._get_along_streamline] You passed field = \'%s\'. The Simulation class does not have this attribute. Possible fields are \'rho\', \'bsq\', etc' % field)
        
        start_points = np.array([start_point])
        strm = self._add_streamplot_to_ax(ax, mode, width, stream=stream, start_points=start_points)
        ax.plot(start_point[0], start_point[1], 'o', color='black')

        num_pts = len(strm.lines.get_segments())
        streamline = np.full((num_pts, 2), np.nan)
        for i in range(num_pts):
            streamline[i, :] = strm.lines.get_segments()[i][0, :]

        fields_along_strm = []
        
        if mode == 'phi': print('Phi mode not yet supported.')
        
        if mode == 'theta':
            
            x2d = self.x[0, :, self.bs2new//2, :]
            y2d = self.y[0, :, self.bs2new//2, :]
            
            for field in fields:
            
                field2d = field[0, :, self.bs2new//2, :]
                interp_field = interp2d(x2d, y2d, field2d)
                fields_along_strm.append(np.array([interp_field(*point) for point in streamline]).flatten())
        
        if len(fields_along_strm) == 1: fields_along_strm = fields_along_strm[0]
        
        return fields_along_strm

class Simulation:
    def __init__(self, sim_dir, manual_load=0, lowres1=1,lowres2=1,lowres3=1,axisym=1,flatten=1,check_files=0,export_raytracing=0,DISK_THICKNESS=0.1,interpolate_var=0,
                 do_box=0,r_min=None,r_max=None,theta_min=None,theta_max=None,phi_min=None,phi_max=None, snapshot_class=Snapshot):
        '''
        Parameters
        sim_dir                 directory of dump folders (string)
        manual_load             whether to load dump file automatically (boolean)
        lowres1/2/3             factor at which to lower resolution in radial/polar/azimuthal directions (integer)
                               must be integer multiple of bs1/2/3 respectively. if manual_load=1.  
        axisym 
                               should almost always be =1.
        flatten                 whether or not to flatten blocks while reading in gdumps. This allows lores to be divisible by
                               total number of cells rather than the block size. Incompatible with manual_load=1. If flatten=2,
                               unflattened coordinate arrays will be stored but blocks will be flattened. (boolean)
        
        check_files             Whether or not to check if each gdumps file is present. Incompatible with flatten=0. (boolean)
        export_raytracing       NEEDS DESCRIPTION. this is 'export_raytracing_RAZIEH' in original macros. (boolean)
        DISK_THICKNESS          H/R of disk. Currently only used if export_raytracing = 1. (float)
        interpolate_var         NEEDS DESCRIPTION. (boolean)
        do_box                  Whether or not to only load in only a segment of the grid. (boolean)
                                IF do_box=1, then r_/theta_/phi_ min/max should each be set to appropriate values. (floats)
        '''
        
        ## misc
        self.sim_dir     = sim_dir
        self.DISK_THICKNESS = DISK_THICKNESS
        
        ## init flags
        self.lowres1           = lowres1
        self.lowres2           = lowres2
        self.lowres3           = lowres3
        self.axisym            = axisym
        self.flatten           = flatten
        self.check_files       = check_files
        self.export_raytracing = export_raytracing
        self.interpolate_var   = interpolate_var
        self.do_box      = do_box
        
        ## if a box is to be read in
        self.r_min       = r_min
        self.r_max       = r_max
        self.theta_min   = theta_min
        self.theta_max   = theta_max
        self.phi_min     = phi_min
        self.phi_max     = phi_max
        
        ## The type of snapshot class to use (by default, just Snapshot)
        self.snapshot_class = snapshot_class
        
        self.lookup = {}
        
        ## Get list of dumps
        dumps = np.array(glob.glob(sim_dir + "/dumps*"))
        str_ln = len(sim_dir)+6
        dumps = np.array([dumps[i][str_ln:] for i in range(len(dumps))])#.astype(np.int32)
        dumps_new = []
        for i in range(len(dumps)):
            #print(dumps[i],len(dumps[i]))
            try:
                dumps_new.append(int(dumps[i]))
            except:
                continue
        dumps_new = np.sort(dumps_new)
        self.dump_list = dumps_new.astype(np.int32)
        self.dump_id   = np.arange(len(self.dump_list)).astype(np.int32)
        
        if manual_load==0:
            
            print("Running self._readBlockData()")
            self._readBlockData()
            print("Running self._readParamDataInit()")
            self._readParamDataInit()
            if flatten == 0 or flatten == 2:
                print("Running self._read_gdumps()")
                self._read_gdumps()
                self.rflat, self.hflat, self.phflat = self.r, self.h, self.ph
            if flatten == 1 or flatten == 2:
                print("Running self._read_gdumps_flatten()")
                self._read_gdumps_flatten()
            print("Finished loading data!")
            
        self._setPlotParams()
        self._setScale()
        
    def load_dump(self,D):
        print("[Simulation.load_dump] Loading dump file D = %d..." % D)
        s = self.snapshot_class(self,D)
        
        flatten = self.flatten
        if flatten == 0 or flatten == 2:
            print("[Simulation.load_dump] Running self._read_dump()")
            s._read_dump()
            s.rflat, s.hflat, s.phflat = self.r, self.h, self.ph
        if flatten == 1 or flatten == 2:
            print("[Simulation.load_dump] Running self._read_dump_flatten()")
            s._read_dump_flatten()
        print("[Simulation.load_dump] Finished loading dump.")
        
        return s
        
    def preload_dumps(self):
        dumps = []
        
        for i in self.dump_id:
            dumps.append(self.snapshot_class(self, self.dump_list[i]))
            
        self.dumps = dumps
        self.read_dump_params = lambda i: self.dumps[np.where((np.int32(self.dump_list)==i))[0][0]]
        
        
    @staticmethod
    def _setPlotParams():
        
        mpl.rcParams['axes.unicode_minus']=False
        mpl.rcParams['font.family'] = 'serif'
        mpl.rcParams['font.serif'] = 'cmr10'
        mpl.rcParams['font.sans-serif'] = 'cmr10'
        mpl.rcParams['mathtext.fontset'] = 'cm'
        mpl.rc('text', usetex=False)
        fontsize = 15
        mpl.rc('xtick', labelsize=fontsize) 
        mpl.rc('ytick', labelsize=fontsize) 
        legend = {'fontsize': fontsize}
        mpl.rc('legend',**legend)
        axes = {'labelsize': fontsize}
        mpl.rc('axes', **axes)
        
    def _setScale(self):
        
        self.M_sol_cgs = 1.98847e33 # solar mass [g]
        self.G_cgs = 6.67259e-8 # gravitational constant [cm^3/gs^2]
        self.c_cgs = 2.99792458e10 # speed of light [cm/s]
        self.arad_cgs = 7.5646e-15 # radiation constant [erg/(cm^3 K^4)]
        self.boltz_cgs = 1.3806504e-16 # boltzman constant [erg/K]
        self.mh_cgs = 1.6733e-24 # hydrogen mass [g]
        self.thomson_cgs = 6.652e-25 # thomson cross section [cm^2]
        self.epsilon = 0.1 # fraction of rest mass energy radiated away during accretion
        self.mmw = 1.69 # mass fraction

        self.M_IC = 1455700.6 # IC mass
        self.M_ratio = 1e6 # star-to-BH mass ratio

        self.M_BH_cgs = self.M_ratio * self.M_sol_cgs # black hole mass [g]
        self.R_g_cgs = self.G_cgs * self.M_BH_cgs / self.c_cgs**2 # gravitational radius [cm]

        # cgs unit per simulation unit
        self.mass_scale = self.M_sol_cgs / self.M_IC
        self.length_scale = self.R_g_cgs
        self.time_scale = self.length_scale / self.c_cgs

        self.mass_density_scale = self.mass_scale / self.length_scale**3
        self.energy_density_scale = self.mass_density_scale * self.c_cgs**2
        self.magnetic_density_scale = np.sqrt(self.mass_density_scale) * self.c_cgs
        
        self.gamg = 5/3
        
    def _readParamDataInit(self):
        '''
        old analogue: "rpar_new"
        
        Reads in parameters file for this simulation. 
        '''        
        path_to_params = self.sim_dir + "/dumps%d/parameters" % self.dump_list[0]
        try: 
            fin = open(path_to_params, "rb")
        except OSError as err_msg:
            print("[_readParamDataInit] %s" % err_msg)


            
        ##
        self.t              = np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]
        self.n_active       = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
        self.n_active_total = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
        self.nstep          = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
        self.Dtd            = np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]
        self.Dtl            = np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]
        self.Dtr            = np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]
        self.dump_cnt       = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
        self.rdump_cnt      = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
        self.dt             = np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]
        self.failed         = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
        
        self.bs1  = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
        self.bs2  = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
        self.bs3  = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
        self.nmax = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
        self.nb1  = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
        self.nb2  = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
        self.nb3  = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]

        self.startx1        = np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]
        self.startx2        = np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]
        self.startx3        = np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]
        self._dx1           = np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]#*r1 # coarsest level of grid
        self._dx2           = np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]#*r2 # add _dx1new, etc...
        self._dx3           = np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]#*r3 # M: how does r1/2/3 differ from lowres?

        self.tf             = np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]
        self.a              = np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]
        self.gam            = np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]
        self.cour           = np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]
        self.Rin            = np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]
        self.Rout           = np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]
        self.R0             = np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]
        self.fractheta      = np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]        
        
        for n in range(0,13):
            trash = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
        trash = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
        if(trash >= 10000000):
            DO_NS = 1
            trash=trash-10000000
        else:
            DO_NS = 0
        if(trash >= 1000):
            P_NUM=1
            trash=trash-1000
        else:
            P_NUM=0
        if (trash >= 100):
            TWO_T = 1
            trash = trash - 100
        else:
            TWO_T = 0
        if (trash >= 10):
            RESISTIVE = 1
            trash = trash - 10
        else:
            RESISTIVE = 0
        if (trash >= 1):
            RAD_M1 = 1
            trash = trash - 1
        else:
            RAD_M1 = 0
        trash = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
        
        # Set grid spacing. M: why does this have to be done manually now?
        self._dx1=(np.log(self.Rout)-np.log(self.Rin))/(self.bs1*self.nb1)
        self._dx2=2.0/(self.bs2*self.nb2)
        self._dx3=2.0*np.pi/(self.bs3*self.nb3)

        nb = self.n_active_total
        rhor = 1 + (1 - self.a ** 2) ** 0.5

        NODE=np.copy(self.n_ord)
        TIMELEVEL=np.copy(self.n_ord)

        REF_1=1
        REF_2=1
        REF_3=1
        flag_restore = 0
        size = os.path.getsize(path_to_params)
        if(size>=66*4+3*self.n_active_total*4):
            n=0
            while n<self.n_active_total:
                self.n_ord[n]=np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
                TIMELEVEL[n] = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
                NODE[n] = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
                n=n+1
        elif(size >= 66 * 4 + 2 * self.n_active_total * 4):
            n = 0
            flag_restore=1
            while n < self.n_active_total:
                self.n_ord[n] = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
                TIMELEVEL[n] = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
                n = n + 1
                
        
        if(self.export_raytracing==1 and (self.bs1%self.lowres1!=0 or self.bs2%self.lowres2!=0 or self.bs3%self.lowres3!=0 or ((self.lowres1 & (self.lowres1-1) == 0) and self.lowres1 != 0)!=1 or ((self.lowres2 & (self.lowres2-1) == 0) and self.lowres2 != 0)!=1 or ((self.lowres3 & (self.lowres3-1) == 0) and self.lowres3 != 0)!=1)):
            print("[Simulation._readParamDataInit] You set export_raytracing=1; For raytracing block size needs to be divisable by lowres!")
        if(self.export_raytracing==1 and self.interpolate_var==0):
            print("[Simulation._readParamDataInit] You set export_raytracing=1 and interpolate_var=0; Warning: Variable interpolation is highly recommended for raytracing!")
        
        fin.close()
        
        # What we will need later:        
        self.RAD_M1       = RAD_M1
        self.trash        = trash
        self.nb           = nb
        self.rhor         = rhor
        self.NODE         = NODE
        self.REF_1        = REF_1
        self.REF_2        = REF_2
        self.REF_3        = REF_3
        self.flag_restore = flag_restore
        self.TWO_T        = TWO_T
        self.RESISTIVE    = RESISTIVE
        self.P_NUM        = P_NUM
        self.DO_NS        = DO_NS
        
        # Save to dictionary
        self.lookup["t"]              = "Current simulation time [r_g/c]"
        self.lookup["n_active"]       = "Number of active blocks?"
        self.lookup["n_active_total"] = "Number of maximum active blocks?"
        self.lookup["nstep"]          = "Current simulation iteration number"
        self.lookup["Dtd"]            = "Dump frequency [c/r_g]"
        self.lookup["Dtl"]            = "Log file [c/r_g]" ## come back
        self.lookup["Dtr"]            = "Restart file [c/r_g]"
        self.lookup["dump_cnt"]       = "Current dump output number"
        self.lookup["rdump_cnt"]      = "Most recent restart dump output number"
        self.lookup["dt"]             = "Current timestep [r_g/c]"
        self.lookup["failed"]         = "[deprecated] 1 if output upon code failure (due to inversion error or whatever)" # disabled
        self.lookup["bs1"]            = "Number of cells per block in radial (\"1\") direction" # M: relate to AMR?
        self.lookup["bs2"]            = "Number of cells per block in polar (\"2\") direction"
        self.lookup["bs3"]            = "Number of cells per block in azimuthal (\"3\") direction"
        self.lookup["nmax"]           = ""
        self.lookup["nb1"]            = "Number of blocks in radial (\"1\") direction"
        self.lookup["nb2"]            = "Number of blocks in polar (\"2\") direction"
        self.lookup["nb3"]            = "Number of blocks in azimuthal (\"3\") direction"
        self.lookup["startx1"]        = "" # minimum value for each code coordinate
        self.lookup["startx2"]        = ""
        self.lookup["startx3"]        = ""
        self.lookup["_dx1"]           = "Coordinate cell width in radial (\"1\") direction" # M: in AMR, is this minimum cell width? Me: correct for lowres?
        self.lookup["_dx2"]           = "Coordinate cell width in polar (\"2\") direction" 
        self.lookup["_dx3"]           = "Coordinate cell width in azimuthal (\"3\") direction"
        self.lookup["tf"]             = "Simulation end time [r_g/c]"
        self.lookup["a"]              = "Dimensionless black hole spin (-1 < a < 1)"
        self.lookup["gam"]            = "Adiabatic index of flow (1 < gam < 5/3)"
        self.lookup["cour"]           = "Courant number (cour < 1 for stability)"
        self.lookup["Rin"]            = "Location of inner radial boundary [r_g/c]"
        self.lookup["Rout"]           = "Location of outer radial boundary [r_g/c]"
        self.lookup["R0"]             = "[deprecated]" # used without Local adaptive timestepping, people would log --> super-log, scale set with R0
        self.lookup["fractheta"]      = "[deprecated]" # was used for which fraction of grid to use
        self.lookup["RAD_M1"]         = "Whether radiation is enabled [0 or 1]"
        self.lookup["DO_NS"]          = "Whether NS is enabled [0 or 1]"
        self.lookup["trash"]          = ""
        self.lookup["nb"]             = "Total number of blocks on grid. (check this = nb1*nb2*nb3)" # check this
        self.lookup["rhor"]           = "" # radius of event horizion (check code)
        self.lookup["NODE"]           = "" # (stupid) 
        self.lookup["REF_1"]          = "[deprecated] Whether to refine radial (\"1\") direction for AMR. Deprecated, should always be= 1."
        self.lookup["REF_2"]          = "[deprecated] Whether to refine polar (\"2\") direction for AMR. Deprecated, should always be = 1."
        self.lookup["REF_3"]          = "[deprectaed] Whether to refine azimuthal (\"3\") direction for AMR. Deprecated, should be 0 if bs3 = 1 and 1 otherwise."
        self.lookup["TWO_T"]          = "Whether two temperature plasma is enabled [0 or 1]"
        self.lookup["RESISTIVE"]      = "Whether resistive MHD (?) is enabled [0 or 1]"
        self.lookup["P_NUM"]          = "? [0 or 1]"
    
    def _readBlockData(self):
        '''
        _readBlockData
        
        old analogue: "rblock_new"
        '''
        
        # declare mnemonics for AMR data
        self.AMR_ACTIVE = 0
        self.AMR_LEVEL = 1
        self.AMR_REFINED = 2
        self.AMR_COORD1 = 3
        self.AMR_COORD2 = 4
        self.AMR_COORD3 = 5
        self.AMR_PARENT = 6
        self.AMR_CHILD1 = 7
        self.AMR_CHILD2 = 8
        self.AMR_CHILD3 = 9
        self.AMR_CHILD4 = 10
        self.AMR_CHILD5 = 11
        self.AMR_CHILD6 = 12
        self.AMR_CHILD7 = 13
        self.AMR_CHILD8 = 14
        self.AMR_NBR1 = 15
        self.AMR_NBR2 = 16
        self.AMR_NBR3 = 17
        self.AMR_NBR4 = 18
        self.AMR_NBR5 = 19
        self.AMR_NBR6 = 20   
        self.AMR_NODE = 21
        self.AMR_POLE = 22
        self.AMR_GROUP = 23
        self.AMR_CORN1 = 24
        self.AMR_CORN2 = 25
        self.AMR_CORN3 = 26
        self.AMR_CORN4 = 27
        self.AMR_CORN5 = 28
        self.AMR_CORN6 = 29
        self.AMR_CORN7 = 30
        self.AMR_CORN8 = 31
        self.AMR_CORN9 = 32
        self.AMR_CORN10 = 33
        self.AMR_CORN11 = 34
        self.AMR_CORN12 = 35
        self.AMR_LEVEL1=  110
        self.AMR_LEVEL2 = 111
        self.AMR_LEVEL3 = 112
        
        path_to_gdump = self.sim_dir + "/gdumps/grid"
        
        # Read in data for every block
        
        # nmax is maxi number of blocks that can be active
        # // 4 because float
        if ( os.path.isfile(path_to_gdump)):
            fin = open(path_to_gdump, "rb")
            size = os.path.getsize(path_to_gdump)
            # M: 
            nmax = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
            NV   = (size - 1) // nmax // 4
        else:
            path_to_dump  = self.sim_dir + "/dumps%d/grid" % self.dump_no
            if ( os.path.isfile(path_to_dump)):
                fin = open(path_to_dump, "rb")
                size = os.path.getsize(path_to_dump)
                nmax = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
                NV   = 36
            else:
                print("[Simulation_readBlockData] Cannot find grid file!")


        ## M: where is '200' coming from?
        ## A: 200 may not work in future
        ## if NV > 200 print statement

        # Allocate memory
        block = np.zeros((nmax, 200), dtype=np.int32, order='C')
        n_ord = np.zeros((nmax), dtype=np.int32, order='C')

        gd = np.fromfile(fin, dtype=np.int32, count=NV*nmax, sep='')
        gd = gd.reshape((NV, nmax), order='F').T
        block[:,0:NV] = gd
        if(NV<170): # previous version of HAMR could refine in 1 or 3 
            block[:, self.AMR_LEVEL1] = gd[:, self.AMR_LEVEL]
            block[:, self.AMR_LEVEL2] = gd[:, self.AMR_LEVEL]
            block[:, self.AMR_LEVEL3] = gd[:, self.AMR_LEVEL]

        i = 0
        for n in range(0, nmax):
            if block[n, self.AMR_ACTIVE] == 1:
                n_ord[i] = n
                i += 1

        fin.close()

        # What we will need later:
        self.block = block
        self.nmax  = nmax
        self.n_ord = n_ord
                
        ## Used for _flatten_blocks
        self.AMR_LEVEL1_c = np.ascontiguousarray(block[:,self.AMR_LEVEL1], dtype=np.int32)
        self.AMR_LEVEL2_c = np.ascontiguousarray(block[:,self.AMR_LEVEL2], dtype=np.int32)
        self.AMR_LEVEL3_c = np.ascontiguousarray(block[:,self.AMR_LEVEL3], dtype=np.int32)
        self.AMR_COORD1_c = np.ascontiguousarray(block[:,self.AMR_COORD1], dtype=np.int32)
        self.AMR_COORD2_c = np.ascontiguousarray(block[:,self.AMR_COORD2], dtype=np.int32)
        self.AMR_COORD3_c = np.ascontiguousarray(block[:,self.AMR_COORD3], dtype=np.int32)
        
        return
    
    def _read_gdumps(self,mytype=np.float32):
        ### to do: add comments, add to look up table
        '''
        old analogue: rgdump_new
        
        Description: 
            Reads in the gdumps file for this simulation. 
            
        TO DO: add lookup tables
        '''
        
        if((self.bs1%self.lowres1)!=0 or (self.bs2%self.lowres2)!=0 or (self.bs3%self.lowres3)!=0):
            print("[Simulation._read_gdump] Incompatible low res settings!")   
            if ((self.bs1%self.lowres1)!=0): print("[Simulation._read_gdump] self.bs1 % self.lowres1 = ", int(self.bs1%self.lowres1))
            if ((self.bs2%self.lowres2)!=0): print("[Simulation._read_gdump] self.bs2 % self.lowres2 = ", int(self.bs2%self.lowres2))
            if ((self.bs3%self.lowres3)!=0): print("[Simulation._read_gdump] self.bs3 % self.lowres3 = ", int(self.bs3%self.lowres3))
            print("\n")

        self.bs1new = int(self.bs1 / self.lowres1)
        self.bs2new = int(self.bs2 / self.lowres2)
        self.bs3new = int(self.bs3 / self.lowres3)
        
        # M: why x,y,z??
        self.nx = self.bs1new * self.nb1
        self.ny = self.bs2new * self.nb2
        self.nz = self.bs3new * self.nb3

        # Allocate memory
        self.x1 = np.zeros((self.nb, self.bs1new, self.bs2new, self.bs3new), dtype=mytype, order='C')
        self.x2 = np.zeros((self.nb, self.bs1new, self.bs2new, self.bs3new), dtype=mytype, order='C')
        self.x3 = np.zeros((self.nb, self.bs1new, self.bs2new, self.bs3new), dtype=mytype, order='C')
        self.r  = np.zeros((self.nb, self.bs1new, self.bs2new, self.bs3new), dtype=mytype, order='C')
        self.h  = np.zeros((self.nb, self.bs1new, self.bs2new, self.bs3new), dtype=mytype, order='C')
        self.ph = np.zeros((self.nb, self.bs1new, self.bs2new, self.bs3new), dtype=mytype, order='C')

        if self.axisym:
            self.gcov  = np.zeros((4, 4, self.nb, self.bs1new, self.bs2new, 1), dtype=mytype, order='C')
            self.gcon  = np.zeros((4, 4, self.nb, self.bs1new, self.bs2new, 1), dtype=mytype, order='C')
            self.gdet  = np.zeros((self.nb, self.bs1new, self.bs2new, 1), dtype=mytype, order='C')
            self.dxdxp = np.zeros((4, 4, self.nb, self.bs1new, self.bs2new, 1), dtype=mytype, order='C')
        else:
            self.gcov  = np.zeros((4, 4, self.nb, self.bs1new, self.bs2new, self.bs3new), dtype=mytype, order='C')
            self.gcon  = np.zeros((4, 4, self.nb, self.bs1new, self.bs2new, self.bs3new), dtype=mytype, order='C')
            self.gdet  = np.zeros((self.nb, self.bs1new, self.bs2new, self.bs3new), dtype=mytype, order='C')
            self.dxdxp = np.zeros((4, 4, self.nb, self.bs1new, self.bs2new, self.bs3new), dtype=mytype, order='C')
            
        size = os.path.getsize(self.sim_dir + '/gdumps/gdump%d' % self.n_ord[0])
        if(size==58*self.bs3*self.bs2*self.bs1*8 and self.bs3!=1):
            flag=1
        else:
            flag=0

        kernel_read_gdumps(flag, c_char_p(self.sim_dir.encode('utf-8')), self.axisym, self.n_ord,
                        self.lowres1,self.lowres2,self.lowres3,
                        self.nb,self.bs1,self.bs2,self.bs3, self.x1,self.x2, self.x3, 
                        self.r,self.h, self.ph,self.gcov, self.gcon,self.dxdxp,self.gdet)
        print("Finishing _read_gdumps().")
        
    def _read_gdumps_flatten(self,mytype=np.float32):
        '''
        old analogue: rgdump_griddata
        
        Description: 
            Reads in the gdumps file for this simulation and flattens blocks to a single uniform grid. 
            
        TO DO: add lookup tables
        '''
        
        ACTIVE1 = np.max(self.block[self.n_ord, self.AMR_LEVEL1])
        ACTIVE2 = np.max(self.block[self.n_ord, self.AMR_LEVEL2])
        ACTIVE3 = np.max(self.block[self.n_ord, self.AMR_LEVEL3])
        
        if ((int(self.nb1 * (1 + self.REF_1) ** ACTIVE1 * self.bs1) % self.lowres1) != 0 or (int(self.nb2 * (1 + self.REF_2) ** ACTIVE2 * self.bs2) % self.lowres2) != 0 or (int(self.nb3 * (1 + self.REF_3) ** ACTIVE3 * self.bs3) % self.lowres3) != 0):
            raise ValueError("[SimulationOld._read_gdumps] Incompatible low res settings!")
            
            
        gridsizex1 = int(self.nb1 * (1 + self.REF_1) ** ACTIVE1 * self.bs1/self.lowres1)
        gridsizex2 = int(self.nb2 * (1 + self.REF_2) ** ACTIVE2 * self.bs2/self.lowres2)
        gridsizex3 = int(self.nb3 * (1 + self.REF_3) ** ACTIVE3 * self.bs3/self.lowres3)
        
        self._dx1 = self._dx1 * self.lowres1 * (1.0 / (1.0 + self.REF_1) ** ACTIVE1)
        self._dx2 = self._dx2 * self.lowres2 * (1.0 / (1.0 + self.REF_2) ** ACTIVE2)
        self._dx3 = self._dx3 * self.lowres3 * (1.0 / (1.0 + self.REF_3) ** ACTIVE3)

        ### Only load in data within specified box
        if self.do_box: 
            self.i_min = max(np.int32((np.log(self.r_min)-(self.startx1+0.5*self._dx1)) / self._dx1) + 1, 0)
            self.i_max = min(np.int32((np.log(self.r_max)-(self.startx1+0.5*self._dx1)) / self._dx1) + 1, gridsizex1)
            self.j_min = max(np.int32(((2.0/np.pi*(self.theta_min)-1.0)-(self.startx2+0.5*self._dx2))/self._dx2) + 1,0)
            self.j_max = min(np.int32(((2.0/np.pi*(self.theta_max)-1.0)-(self.startx2+0.5*self._dx2))/self._dx2) + 1,gridsizex2)
            self.k_min = max(np.int32((self.phi_min-(self.startx3+0.5*self._dx3))/self._dx3) + 1,0)
            self.k_max = min(np.int32((self.phi_max-(self.startx3+0.5*self._dx3))/self._dx3) + 1,gridsizex3)
            
            if((self.j_max<self.j_min or self.i_max<self.i_min or self.k_max<self.k_min)):
                raise ValueError("[_read_gdumps_flatten] Bad box selection;")
            
            gridsizex1 = self.i_max-self.i_min
            gridsizex2 = self.j_max-self.j_min
            gridsizex3 = self.k_max-self.k_min
            
        else:
            self.i_min=0
            self.i_max=gridsizex1
            self.j_min=0
            self.j_max=gridsizex2
            self.k_min=0
            self.k_max=gridsizex3

            self.nx = gridsizex1
            self.ny = gridsizex2
            self.nz = gridsizex3
            
        self.bs1new = gridsizex1
        self.bs2new = gridsizex2
        self.bs3new = gridsizex3
        

        # Allocate memory
        self.x1 = np.zeros((1, gridsizex1, gridsizex2, gridsizex3), dtype=mytype, order='C')
        self.x2 = np.zeros((1, gridsizex1, gridsizex2, gridsizex3), dtype=mytype, order='C')
        self.x3 = np.zeros((1, gridsizex1, gridsizex2, gridsizex3), dtype=mytype, order='C')
        self.r = np.zeros((1, gridsizex1, gridsizex2, gridsizex3), dtype=mytype, order='C')
        self.h = np.zeros((1, gridsizex1, gridsizex2, gridsizex3), dtype=mytype, order='C')
        self.ph = np.zeros((1, gridsizex1, gridsizex2, gridsizex3), dtype=mytype, order='C')

        if self.axisym:
            self.gcov = np.zeros((4, 4, 1, gridsizex1, gridsizex2, 1), dtype=mytype, order='C')
            self.gcon = np.zeros((4, 4, 1, gridsizex1, gridsizex2, 1), dtype=mytype, order='C')
            self.gdet = np.zeros((1, gridsizex1, gridsizex2, 1), dtype=mytype, order='C')
            self.dxdxp = np.zeros((4, 4, 1, gridsizex1, gridsizex2, 1), dtype=mytype, order='C')
        else:
            self.gcov = np.zeros((4, 4, 1, gridsizex1, gridsizex2, gridsizex3), dtype=mytype, order='C')
            self.gcon = np.zeros((4, 4, 1, gridsizex1, gridsizex2, gridsizex3), dtype=mytype, order='C')
            self.gdet = np.zeros((1, gridsizex1, gridsizex2, gridsizex3), dtype=mytype, order='C')
            self.dxdxp = np.zeros((4, 4, 1, gridsizex1, gridsizex2, gridsizex3), dtype=mytype, order='C')
            
            
        ## Ensure that each gdumps file is present
        if self.check_files==1:
            for n in range(0,self.n_active_total):
                if(os.path.isfile(self.sim_dir + '/gdumps/gdump%d' %self.n_ord[n])==0 or os.path.getsize(self.sim_dir + '/gdumps/gdump%d' %self.n_ord[n])!=(9*self.bs1*self.bs2*self.bs3+(self.bs1*self.bs2*49)*(self.axisym)+(self.bs1*self.bs2*self.bs3*49)*(self.axisym==0))*8):
                    print("Gdump file %d doesn't exist" % self.n_ord[n])

        size = os.path.getsize(self.sim_dir + '/gdumps/gdump%d' % self.n_ord[0])
        if(size==58*self.bs3*self.bs2*self.bs1*8):
            flag=1
        else:
            flag=0

        kernel_read_gdumps_griddata(flag, int(self.interpolate_var), c_char_p(self.sim_dir.encode('utf-8')), int(self.axisym), self.n_ord, int(self.lowres1), int(self.lowres2), int(self.lowres3), int(self.nb), int(self.bs1), int(self.bs2), int(self.bs3), self.x1, self.x2, self.x3, self.r, self.h, self.ph, self.gcov, self.gcon, self.dxdxp, self.gdet, ACTIVE1, ACTIVE2, ACTIVE3, self.AMR_LEVEL1_c, self.AMR_LEVEL2_c, self.AMR_LEVEL3_c, self.AMR_COORD1_c, self.AMR_COORD2_c, self.AMR_COORD3_c, int(self.nb1), int(self.nb2), int(self.nb3), int(self.REF_1), int(self.REF_2), int(self.REF_3), self.startx1, self.startx2, self.startx3, self._dx1, self._dx2, self._dx3, int(self.export_raytracing), int(self.i_min), int(self.i_max), int(self.j_min), int(self.j_max), int(self.k_min), int(self.k_max))
        
        self.nb = 1
        self.nb1 = 1
        self.nb2 = 1
        self.nb3 = 1

        
        print("Finishing _read_gdumps_flatten().")
        
    def calc_dxdxp_inv(self):
        dxdxp_inv = np.zeros((4, 4, self.nb, self.bs1new, self.bs2new, 1), dtype=np.float32, order='C')

        kernel_invert_4x4(self.dxdxp,dxdxp_inv,self.nb, self.bs1new, self.bs2new, 1)

        self._dxdxp_inv = dxdxp_inv
    
    @property
    def dxdxp_inv(self):
        try:
            return self._dxdxp_inv
        except:
            self.calc_dxdxp_inv()
            return self._dxdxp_inv

    def calc_dxBLdx(self):
        # https://arxiv.org/pdf/astro-ph/0404512.pdf
        r  = self.r[0,:,0,0]
        a  = self.a
        delta = r*r - 2*r + a*a

        dxdxBL = np.zeros((4,4,1,self.bs1new,1,1),dtype=np.float32)
        dxdxBL[0,1]=(2*r/delta)[None,:,None,None] # dtdr
        dxdxBL[1,3]=(a/delta)[None,:,None,None]   # drdphi
        dxdxBL[0,0]=dxdxBL[1,1]=dxdxBL[2,2]=dxdxBL[3,3]=1     # diagonals are unity

        dxBLdx = np.zeros_like(dxdxBL)
        kernel_invert_4x4(dxdxBL,dxBLdx,1, self.bs1new, 1, 1)

        self._dxBLdx = dxBLdx
        self._dxdxBL = dxdxBL
    
    @property
    def dxBLdx(self):
        try:
            return self._dxBLdx
        except:
            self.calc_dxBLdx() 
            return self._dxBLdx
    @property
    def dxdxBL(self):
        try:
            return self._dxdxBL
        except:
            self.calc_dxBLdx() 
            return self._dxdxBL

    @staticmethod
    def eq_tetrad(r,a):

        # orthonormal tetrad for circular, equatorial orbits
        # transformations from BL to local frame

        # https://www.its.caltech.edu/~kip/index.html/PubScans/II-48.pdf
        # Novikov thorne

        # see also 

        # Eqs 5.4.1a-5.4.1g
        A = 1 + (a**2)/r**2 + 2*(a**2)/r**3
        B = 1 + a/r**(3/2)
        C = 1 - 3/r + 2*a/r**(3/2)
        D = 1 - 2/r + (a**2)/r**2
        E = 1 + 4*(a**2)/r**2 - 4*(a**2)/r**3 + 3*(a**4)/r**4
        F = 1 - 2*a/r**(3/2) + (a**2)/r**2
        G = 1 - 2/r + a/r**(3/2)

        # Eq 5.4.2.b
        w = 2*a/r**3/A

        if isinstance(r, (list, tuple, np.ndarray)):
            shape = tuple(np.concatenate(((4,4),np.shape(r))))
        else:
            shape = (4,4)
        e = np.zeros(shape,dtype=np.float32)

        # Equation 5.4.5a
        # e^mu_a (transforms covariant stuff)
        e[0,0] = B/C**(1/2)
        e[0,3] = F/C**(1/2)/D**(1/2)/r**(1/2)
        e[1,1] = D**(1/2)
        e[2,2] = -1/r # is this correct?
        e[3,0] = B/C**(1/2) * r**(-3/2)/B
        e[3,3] = B*D**(1/2)/A/C**(1/2)/r + (F/C**(1/2)/D**(1/2)/r**(1/2))*w

        e1f = np.zeros_like(e)
        # e^a_mu (transforms contravariant stuff)
        e1f[0,0] = G/C**(1/2)
        e1f[0,3] = -F/C**(1/2)*r**(1/2)
        e1f[1,1] = D**(-1/2)
        e1f[2,2] = -r
        e1f[3,3] = B*D**(1/2)/C**(1/2)*r
        e1f[3,0] = -D**(1/2)/C**(1/2)*r**(-1/2)
        
        return e,e1f

        
    def calc_eq_tetrad(self):
        e,e1f = self.eq_tetrad(self.r,self.a)
        self._dxdxe = e
        self._dxedx = e1f
        
    @property
    def dxdxe(self):
        try:
            return self._dxdxe
        except:
            self.calc_eq_tetrad() 
            return self._dxdxe
    @property
    def dxedx(self):
        try:
            return self._dxedx
        except:
            self.calc_eq_tetrad() 
            return self._dxedx
        
    '''
    "Spherical Kerr-Schild" covariant metric tensor
    Converted from "modified" "Spherical Kerr-Schild" coordinates
    '''
    def calc_gcov_KS(self):
        self._gcov_KS = np.einsum('ij...,ik...,jl...->kl...',self.gcov,self.dxdxp_inv,self.dxdxp_inv) 
    @property
    def gcov_KS(self):
        try:
            return self._gcov_KS
        except:
            self.calc_gcov_KS()
            return self._gcov_KS
        
    '''
    Boyer-Lindquist covariant metric tensor
    converted from "Spherical Kerr-Schild" coordinates
    '''
    def calc_gcov_BL(self):
        self._gcov_BL = np.einsum('ij...,ik...,jl...->kl...',self.gcov_KS,self.dxdxBL,self.dxdxBL) 
    @property
    def gcov_BL(self):
        try:
            return self._gcov_BL
        except:
            self.calc_gcov_BL()
            return self._gcov_BL
        
    '''
    Local frame covariant metric tensor
    converted from Boyer-Lindquist coordinates
    should be approximately Minkowski about reference orbit! 
    '''
    def calc_gcov_LC(self):
        self._gcov_LC = np.einsum('ij...,ik...,jl...->kl...',self.gcov_BL,self.dxdxe,self.dxdxe) 
    @property
    def gcov_LC(self):
        try:
            return self._gcov_LC
        except:
            self.calc_gcov_LC()
            return self._gcov_LC
         