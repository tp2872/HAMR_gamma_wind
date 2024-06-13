import numpy as np
from load_c import kernel_invert_4x4
from scipy import ndimage

def _calc_gcov_kerr(self):
    # Set covariant Kerr metric in double
    gcov_kerr = np.zeros((4, 4, self.nb, self.bs1new, self.bs2new, 1), dtype=np.float32)

    r = self.r
    h = self.h
    ph = self.ph
    a = self.a

    cth = np.cos(h)
    sth = np.sin(h)
    s2 = sth * sth
    rho2 = r * r + a * a * cth * cth

    gcov_kerr[0, 0, 0, :, :, 0] = (-1. + 2. * r / rho2)[0, :, :, 0]
    gcov_kerr[0, 1, 0, :, :, 0] = (2. * r / rho2)[0, :, :, 0]
    gcov_kerr[0, 3, 0, :, :, 0] = (-2. * a * r * s2 / rho2)[0, :, :, 0]
    gcov_kerr[1, 0, 0, :, :, 0] = gcov_kerr[0, 1, 0, :, :, 0]
    gcov_kerr[1, 1, 0, :, :, 0] =  (1. + 2. * r / rho2)[0, :, :, 0]
    gcov_kerr[1, 3, 0, :, :, 0] = (-a * s2 * (1. + 2. * r / rho2))[0, :, :, 0]
    gcov_kerr[2, 2, 0, :, :, 0] = rho2[0, :, :, 0]
    gcov_kerr[3, 0, 0, :, :, 0] = gcov_kerr[0, 3, 0, :, :, 0]
    gcov_kerr[3, 1, 0, :, :, 0] = gcov_kerr[1, 3, 0, :, :, 0]
    gcov_kerr[3, 3, 0, :, :, 0] = (s2 * (rho2 + a * a * s2 * (1. + 2. * r / rho2)))[0, :, :, 0]
    return gcov_kerr
### sets indices for shifting

def preset_transform_scalar(self):
    X = np.zeros((4, self.nb, self.bs1new, self.bs2new, self.bs3new), dtype=np.float32)
    tilt_tmp = np.zeros((self.nb, self.bs1new, 1, 1), dtype=np.float32)
    prec_tmp = np.zeros((self.nb, self.bs1new, 1, 1), dtype=np.float32)
    t1 = np.zeros((self.nb, self.bs1new, 1, 1), dtype=np.float32)
    t2 = np.zeros((self.nb, 1, self.bs2new, 1), dtype=np.float32)
    t3 = np.zeros((self.nb, 1, 1, self.bs3new), dtype=np.float32)
    ti = np.zeros((self.nb, self.bs1new, self.bs2new, self.bs3new), dtype=np.float32)
    tj = np.zeros((self.nb, self.bs1new, self.bs2new, self.bs3new), dtype=np.float32)
    tk = np.zeros((self.nb, self.bs1new, self.bs2new, self.bs3new), dtype=np.float32)
    
    
    t1[0, :, 0, 0] = np.arange(self.bs1new)
    t2[0, 0, :, 0] = np.arange(self.bs2new)
    t3[0, 0, 0, :] = np.arange(self.bs3new)

    ti[:, :, :, :] = t1
    tj[:, :, :, :] = t2
    tk[:, :, :, :] = t3

    tilt_tmp[0, :, 0, 0] = self.tilt_disk[0] / 180.0 * np.pi
    prec_tmp[0, :, 0, 0] = (self.prec_disk[0] / 180.0 * np.pi)
    prec_tmp[prec_tmp<0] += 2*np.pi
    
    x_new = self.x*np.cos(tilt_tmp) + self.z*np.sin(tilt_tmp)
    y_new = np.copy(self.y)
    z_new = -self.x*np.sin(tilt_tmp) + self.z*np.cos(tilt_tmp)
    
    h_new = np.arccos(z_new/np.sqrt(x_new*x_new + y_new*y_new + z_new*z_new))
    ph_new = np.arctan2(y_new, x_new)
    ph_new = (ph_new + prec_tmp)

    tj = (((h_new[0] - self.h[0]) / (self._dx2)*2.0/np.pi + tj)) % self.bs2new
    tk = (((ph_new[0] - self.ph[0]) / (self._dx3) + tk)) % self.bs3new
    return ti,tj,tk

### transforms scalar given index matrices from preset_transform_scalar
def transform_scalar(self, ti, tj, tk, scalar):
    output = np.zeros((self.nb, self.bs1new, self.bs2new, self.bs3new), dtype=np.float32)

    output[0] = ndimage.map_coordinates(scalar[0], [[ti], [tj], [tk]], order=1, mode='nearest')
    return output

## Calculate spherical Kerr Schild to Cartesian
def _dxdr(self):
    '''
    Transformation matrix from Spherical Kerr Schild to Cartesian Kerr Schild (is this Cartesian Kerr Schild)
    '''
        
    dxdr = np.zeros((4, 4, self.nb, self.bs1new, self.bs2new, self.bs3new), dtype=np.float32, order='C')
    dxdr[0, 0] = 1
    dxdr[0, 1] = 0
    dxdr[0, 2] = 0
    dxdr[0, 3] = 0
    dxdr[1, 0] = 0
    dxdr[1, 1] = (np.sin(self.h) * np.cos(self.ph))
    dxdr[1, 2] = (self.r * np.cos(self.h) * np.cos(self.ph))
    dxdr[1, 3] = (-self.r * np.sin(self.h) * np.sin(self.ph))
    dxdr[2, 0] = 0
    dxdr[2, 1] = (np.sin(self.h) * np.sin(self.ph))
    dxdr[2, 2] = (self.r * np.cos(self.h) * np.sin(self.ph))
    dxdr[2, 3] = (self.r * np.sin(self.h) * np.cos(self.ph))
    dxdr[3, 0] = 0
    dxdr[3, 1] = (np.cos(self.h))
    dxdr[3, 2] = (-self.r * np.sin(self.h))
    dxdr[3, 3] = 0
    return dxdr


### get transformation matrices for tilted coordinate system
def _drtdr_both(self):
    dxdr = _dxdr(self)
    drtdr = np.zeros((4, 4, self.nb, self.bs1new, self.bs2new, self.bs3new), dtype=np.float32, order='C')
    drtdr_inv = np.zeros((4, 4, self.nb, self.bs1new, self.bs2new, self.bs3new), dtype=np.float32, order='C')
    # Set coordinates
    x0 = (self.r * np.sin(self.h) * np.cos(self.ph))
    y0 = (self.r * np.sin(self.h) * np.sin(self.ph))
    z0 = (self.r * np.cos(self.h))

    tilt = self.tilt_disk[:,:,None,None] * np.pi/180
    prec = -self.prec_disk[:,:,None,None] * np.pi/180
    xt = ((x0 * np.cos(prec) - y0 * np.sin(prec)) * np.cos(tilt) - z0 * np.sin(tilt))
    yt = (y0 * np.cos(prec) + x0 * np.sin(prec))
    zt = ((x0 * np.cos(prec) - y0 * np.sin(prec)) * np.sin(tilt) + z0 * np.cos(tilt))

    rt = np.sqrt(xt * xt + yt * yt + zt * zt)
    ht = np.arccos(zt / (rt))
    pht = np.arctan2(yt, xt)

    for i in range(0, self.bs1new):
        # Alloccate temporary arrays
        dxtdx = np.zeros((4, 4, self.nb, self.bs2new, self.bs3new), dtype=np.float32, order='C')
        dxtdr = np.zeros((4, 4, self.nb, self.bs2new, self.bs3new), dtype=np.float32, order='C')
        # NK: Added extra dim of 1 between nb and bs2new. Doesn't change anything, was just convenient
        # for inverting the matrix the same way as the other ndim=6 arrays.
        dxtdrt = np.zeros((4, 4, self.nb, 1, self.bs2new, self.bs3new), dtype=np.float32, order='C')
        dxtdrt_inv = np.zeros((4, 4, self.nb, 1, self.bs2new, self.bs3new), dtype=np.float32, order='C')
    
        if 0:
            # Transformation matrix to to tilted Cartesian Kerr Schild from Cartesian Kerr Schild
            dxtdx[0, 0] = 1
            dxtdx[0, 1] = 0
            dxtdx[0, 2] = 0
            dxtdx[0, 3] = 0
            dxtdx[1, 0] = 0
            dxtdx[1, 1] = (np.cos(prec[:, i]) * np.cos(tilt[:, i]))
            dxtdx[1, 2] = (-np.sin(prec[:, i]))
            dxtdx[1, 3] = (np.cos(prec[:, i])*np.sin(tilt[:, i]))
            dxtdx[2, 0] = 0
            dxtdx[2, 1] = (np.cos(tilt[:, i])*np.sin(prec[:, i]))
            dxtdx[2, 2] = (np.cos(prec[:, i]))
            dxtdx[2, 3] = (np.sin(tilt[:, i])*np.sin(prec[:, i]))
            dxtdx[3, 0] = 0
            dxtdx[3, 1] = -np.sin(tilt[:, i])
            dxtdx[3, 2] = 0
            dxtdx[3, 3] = (np.cos(tilt[:, i]))
        else: ## ""corrected"" version
            dxtdx[0, 0] = 1
            dxtdx[0, 1] = 0
            dxtdx[0, 2] = 0
            dxtdx[0, 3] = 0
            dxtdx[1, 0] = 0
            dxtdx[1, 1] = (np.cos(tilt[:, i]) * np.cos(prec[:, i]))
            dxtdx[1, 2] = (-np.cos(tilt[:, i]) * np.sin(prec[:, i]))
            dxtdx[1, 3] = (-np.sin(tilt[:, i]))
            dxtdx[2, 0] = 0
            dxtdx[2, 1] = (np.sin(prec[:, i]))
            dxtdx[2, 2] = (np.cos(prec[:, i]))
            dxtdx[2, 3] = 0
            dxtdx[3, 0] = 0
            dxtdx[3, 1] = (np.sin(tilt[:, i]) * np.cos(prec[:, i]))
            dxtdx[3, 2] = (-np.sin(tilt[:, i]) * np.sin(prec[:, i]))
            dxtdx[3, 3] = (np.cos(tilt[:, i]))

        # Calculate transformation matrix from tilted Cartesian to tilted Kerr-Schild
        dxtdrt[0, 0, 0] = 1
        dxtdrt[0, 1, 0] = 0
        dxtdrt[0, 2, 0] = 0
        dxtdrt[0, 3, 0] = 0
        dxtdrt[1, 0, 0] = 0
        dxtdrt[1, 1, 0] = (np.sin(ht[:, i]) * np.cos(pht[:, i]))
        dxtdrt[1, 2, 0] = (rt[:, i] * np.cos(ht[:, i]) * np.cos(pht[:, i]))
        dxtdrt[1, 3, 0] = (-rt[:, i] * np.sin(ht[:, i]) * np.sin(pht[:, i]))
        dxtdrt[2, 0, 0] = 0
        dxtdrt[2, 1, 0] = (np.sin(ht[:, i]) * np.sin(pht[:, i]))
        dxtdrt[2, 2, 0] = (rt[:, i] * np.cos(ht[:, i]) * np.sin(pht[:, i]))
        dxtdrt[2, 3, 0] = (rt[:, i] * np.sin(ht[:, i]) * np.cos(pht[:, i]))
        dxtdrt[3, 0, 0] = 0
        dxtdrt[3, 1, 0] = (np.cos(ht[:, i]))
        dxtdrt[3, 2, 0] = (-rt[:, i] * np.sin(ht[:, i]))
        dxtdrt[3, 3, 0] = 0

        temp = xt[:, i] ** 2.0 + yt[:, i] ** 2.0
        temp2 = np.sqrt(temp) * rt[:, i] ** 2.0
        dxtdrt_inv[0, 0] = 1
        dxtdrt_inv[0, 1] = 0
        dxtdrt_inv[0, 2] = 0
        dxtdrt_inv[0, 3] = 0
        dxtdrt_inv[1, 0] = 0
        dxtdrt_inv[1, 1] = xt[:, i] / rt[:, i]
        dxtdrt_inv[1, 2] = yt[:, i] / rt[:, i]
        dxtdrt_inv[1, 3] = zt[:, i] / rt[:, i]
        dxtdrt_inv[2, 0] = 0
        dxtdrt_inv[2, 1] = xt[:, i] * zt[:, i] / temp2
        dxtdrt_inv[2, 2] = yt[:, i] * zt[:, i] / temp2
        dxtdrt_inv[2, 3] = (-xt[:, i] ** 2 - yt[:, i] ** 2) / temp2
        dxtdrt_inv[3, 0] = 0
        dxtdrt_inv[3, 1] = -yt[:, i] / temp
        dxtdrt_inv[3, 2] = xt[:, i] / temp
        dxtdrt_inv[3, 3] = 0
        
        for i1 in range(0, 4):
            for j1 in range(0, 4):
                for k in range(0, 4):
                    dxtdr[i1, j1] = dxtdr[i1, j1] + dxtdx[i1, k] * dxdr[k, j1, :, i]
        for i1 in range(0, 4):
            for j1 in range(0, 4):
                for k in range(0, 4):
                    drtdr[i1, j1, :, i] = drtdr[i1, j1, :, i] + dxtdrt_inv[i1, k, 0] * dxtdr[k, j1]

    kernel_invert_4x4(drtdr,drtdr_inv,1, self.bs1new, self.bs2new, self.bs3new)
    
    return drtdr, drtdr_inv

def _warp(self):
    ### Assumes nb=1
    
    if self.nb != 1:
        raise ValueError("[Simulation.warp] Error! self.nb != 1! This function only works for a flattened grid with self.nb=1.")
        
    T = self.tilt_disk[:,:,None,None]*np.pi/180.
    P = self.prec_disk[:,:,None,None]*np.pi/180.
    #dr = np.gradient(self.r[:,:,0,0],axis=1)[:,:,None,None]
    
    #print("dr shape = ", dr.shape)

    ### 
    l = np.zeros((4, self.nb, self.bs1new, 1, 1), dtype=np.float32, order='C')
    m = np.zeros((4, self.nb, self.bs1new, 1, 1), dtype=np.float32, order='C')
    n = np.zeros((4, self.nb, self.bs1new, 1, 1), dtype=np.float32, order='C')

    ## l (angular momentum unit vector)
    # x direction
    l[1] = np.sin(T)*np.cos(P)
    # y direction
    l[2] = np.sin(T)*np.sin(P)
    # z direction
    l[3] = np.cos(T)
    
    ## m (= dldlnr )
    # x direction
    m[1] = np.gradient(l[1],axis=1)/self._dx1
    # y direction
    m[2] = np.gradient(l[2],axis=1)/self._dx1
    # z direction
    m[3] = np.gradient(l[3],axis=1)/self._dx1
    # normalize
    norm = np.sqrt(m[1]**2 + m[2]**2 + m[3]**2 + 1e-10)
    m[1] = m[1]/norm
    m[2] = m[2]/norm
    m[3] = m[3]/norm
    norm = np.sqrt(m[1]**2 + m[2]**2 + m[3]**2)
    
    ## n (= l x dldr = l x m)
    n[1:] = np.cross(l[1:],m[1:],axis=0)
    norm = np.sqrt(n[1]**2 + n[2]**2 + n[3]**2 + 1e-10)

    return l,m,n

def _psi(self, l):
    dl = np.gradient(l[:,0,:,0,0],axis=1)
    psi = np.sqrt(dl[0]**2 + dl[1]**2 + dl[2]**2)/self._dx1
    return psi