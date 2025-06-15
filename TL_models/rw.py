'''
@author: Ziad (zi.hatab@gmail.com)

This is an implementation of a rectangular waveguide model based on the references [1-2] (see the class RW).
In addition to the general equations, I included the option to calculate Jacobian with respect to the inputs.

[1] K. Lomakin, G. Gold and K. Helmreich, 
"Transmission line model for rectangular waveguides accurately incorporating loss effects," 
2017 IEEE 21st Workshop on Signal and Power Integrity (SPI), Lake Maggiore, Italy, 2017, 
pp. 1-4, doi: 10.1109/SaPIW.2017.7944024.
https://ieeexplore.ieee.org/document/7944024

[2] K. Lomakin, G. Gold and K. Helmreich, 
"Analytical Waveguide Model Precisely Predicting Loss and Delay Including Surface Roughness," 
in IEEE Transactions on Microwave Theory and Techniques, 
vol. 66, no. 6, pp. 2649-2662, June 2018, doi: 10.1109/TMTT.2018.2827383.
https://ieeexplore.ieee.org/document/8356729
'''

# pip install numpy scipy -U
import numpy as np
import scipy.optimize as so  # for derivative computation

def R1(w, h, f, mur=1, sr=1):
    '''
    Parameters
    ----------
    w : number
        width of the waveguide.
    h : number
        hight of the waveguide (h < w).
    f : number or array
        frequency in Hz.
    mur : number, optional
        relative permeability to vacuum permeability (1.25663706212e-6). The default is 1.
    sr : number, optional
        relative conductivity to copper conductivity (5.98e7). The default is 1.

    Returns
    -------
    series resistance per unit length.
    '''
    s0  = 5.98e7 # (copper conductivity)
    mu0 = 1.25663706212e-6
    mu = mur*mu0
    sigma = sr*s0
    delta = np.sqrt(1/np.pi/mu/sigma/f)
    return 2/sigma/delta/h

def R2(w, h, f, mur=1, sr=1):
    '''
    Parameters
    ----------
    w : number
        width of the waveguide.
    h : number
        hight of the waveguide (h < w).
    f : number or array
        frequency in Hz.
    mur : number, optional
        relative permeability to vacuum permeability (1.25663706212e-6). The default is 1.
    sr : number, optional
        relative conductivity to copper conductivity (5.98e7). The default is 1.

    Returns
    -------
    shunt resistance per unit length
    '''
    s0  = 5.98e7
    mu0 = 1.25663706212e-6
    mu = mur*mu0
    sigma = sr*s0
    delta = np.sqrt(1/np.pi/mu/sigma/f)
    return 2*w*(w + 2*h)/h/np.pi**2/sigma/delta

def L1(w, h, f, mur=1, sr=1):
    '''
    Parameters
    ----------
    w : number
        width of the waveguide.
    h : number
        hight of the waveguide (h < w).
    f : number or array
        frequency in Hz.
    mur : number, optional
        relative permeability to vacuum permeability (1.25663706212e-6). The default is 1.
    sr : number, optional
        relative conductivity to copper conductivity (5.98e7). The default is 1.

    Returns
    -------
    series inductance per unit length
    '''
    mu0 = 1.25663706212e-6
    mu = mur*mu0
    omega = 2*np.pi*f
    L1_o = mu
    L1_i = R1(w,h,f,mur,sr)/omega
    return L1_o + L1_i

def L2(w, h, f, mur=1, sr=1):
    '''
    Parameters
    ----------
    w : number
        width of the waveguide.
    h : number
        hight of the waveguide (h < w).
    f : number or array
        frequency in Hz.
    mur : number, optional
        relative permeability to vacuum permeability (1.25663706212e-6). The default is 1.
    sr : number, optional
        relative conductivity to copper conductivity (5.98e7). The default is 1.

    Returns
    -------
    shunt inductance per unit length
    '''
    mu0 = 1.25663706212e-6
    omega = 2*np.pi*f
    L2_o = mu0*w**2/np.pi**2
    L2_i = R2(w,h,f,mur,sr)/omega
    return L2_o + L2_i

def C(er=1):
    '''
    Parameters
    ----------
    er : number, optional
        relative permittivity to vacuum (8.8541878128e-12). The default is 1.
        
    Returns
    -------
    shunt capacitance per unit length
    '''
    ep0 = 8.8541878128e-12
    return er*ep0

def G(f,er=1,tand=0):
    '''
    Parameters
    ----------
    f : number
        frequency in Hz.
    er : number, optional
        relative permittivity to vacuum (8.8541878128e-12). The default is 1.
    tand : number, optional
        loss tangent. The default is 0.

    Returns
    -------
    shunt conductance per unit length
    '''
    omega = 2*np.pi*f
    return omega*C(er)*tand

def get_all_paras(x, rw):
    '''
    This function is used to compute the Jacobian of the parameters
    '''
    # x = [w, h, mur, sr, er, tand]
    rw.w    = x[0]
    rw.h    = x[1]
    rw.mur  = x[2]
    rw.sr   = x[3]
    rw.er   = x[4]
    rw.tand = x[5]
    rw.update()
    
    # convert complex-valued array to real-valued equivalent
    hh = lambda x: np.kron(x.real, [1,0]) + np.kron(x.imag, [0,1])
    
    # RW line parameters
    gamma = rw.gamma
    h_gamma = hh(gamma)
    ereff = rw.ereff
    h_ereff = hh(ereff)
    Z0 = rw.Z0
    h_Z0 = hh(Z0)
    return np.hstack((h_gamma, h_ereff, h_Z0))

class RW:
    """
    Analytical model of rectangular waveguide (RW) based on [1-2] (see comments at top of this file).

    Parameters
    ----------
    w : number
        width in meters.
    h : number
        height in meters.
    f : number or 1d-array
        frequency in Hz.
    mur : float number
        relative permeability
    sr : float number
        relative conductivity to copper (5.98e7 S/m)
    er : float number
        real-part relative permittivity.
    tand : float number
        loss tangent.
    """
    def __init__(self, w, h, f, mur=1, sr=1, er=1, tand=0):
        self.w     = w
        self.h     = h
        self.f     = np.atleast_1d(f)
        self.mur   = mur
        self.sr    = sr
        self.er    = er
        self.tand  = tand
        self.update()  # run the code
    
    def update(self):
        mu0 = 1.25663706212e-6
        ep0 = 8.8541878128e-12
        c0  = 1/np.sqrt(mu0*ep0) # 299792458 # speed of light in vacuum (m/s)
        
        w    = self.w
        h    = self.h
        f    = self.f
        mur  = self.mur
        sr   = self.sr
        er   = self.er
        tand = self.tand
        
        omega = 2*np.pi*f
        
        # per unit length parameters
        Cp  = np.array([C(er) for ff in f])
        Gp  = np.array([G(ff,er,tand) for ff in f])
        Rp  = np.array([R1(w,h,ff,mur,sr) for ff in f])
        Rpp = np.array([R2(w,h,ff,mur,sr) for ff in f])
        Lp  = np.array([L1(w,h,ff,mur,sr) for ff in f])
        Lpp = np.array([L2(w,h,ff,mur,sr) for ff in f])
        
        Z = Rp + 1j*omega*Lp
        Y = Gp + 1j*omega*Cp + 1/(Rpp + 1j*omega*Lpp)

        # Final results
        self.gamma = np.sqrt(Z*Y)
        self.ereff = -(c0/2/np.pi/f*self.gamma)**2
        self.Z0    = np.sqrt(Z/Y)
        
        # set Jacobians to None if newly evaluated (you need to run update_jac() to compute them)
        self.jac_gamma = None
        self.jac_ereff = None
        self.jac_Z0    = None
        self.jac_Gamma = None
        
    def update_jac(self):
        '''
        Updates the jacobian of the parameters with respect to the input parameters.
        '''
        # these are the input parameters to which the Jacobian is computed
        w    = self.w
        h    = self.h
        mur  = self.mur
        sr   = self.sr
        er   = self.er
        tand = self.tand
        
        N = len(self.f)
        M = 2*N
        eps = np.sqrt(np.finfo(float).eps)
        x   = [w, h, mur, sr, er, tand]
        big_J = so.approx_fprime(x, get_all_paras, [eps]*len(x), self)
        
        # split the jacobian in correct order
        self.jac_gamma = big_J[:M].reshape((N,2,-1))
        self.jac_ereff = big_J[M:2*M].reshape((N,2,-1))
        self.jac_Z0    = big_J[2*M:3*M].reshape((N,2,-1))
        self.jac_Gamma = np.array([ np.array([[(1/2/z).real, (1j/2/z).real],[(1/2/z).imag, (1j/2/z).imag]])@Jz for z,Jz in zip(self.Z0, self.jac_Z0) ])
                
        # undo the changes done by the function so.approx_fprime().
        self.w    = w
        self.h    = h
        self.mur  = mur
        self.sr   = sr
        self.er   = er
        self.tand = tand
        
if __name__ == '__main__':
    import matplotlib.pyplot as plt # for plotting
    import metas_unclib as munc # metas package to propagate general uncertainties through functions
    munc.use_linprop()

    # reference material parameters
    mu0 = 1.25663706212e-6
    ep0 = 8.8541878128e-12
    c0  = 1/np.sqrt(mu0*ep0) # 299792458 # speed of light in vacuum (m/s)
    s0  = 5.98e7 # copper conductivity
    
    # useful functions
    mag2db = lambda x: 20*np.log10(abs(x))
    db2mag = lambda x: 10**(x/20)
    gamma2ereff = lambda x,f: -(c0/2/np.pi/f*x)**2
    ereff2gamma = lambda x,f: 2*np.pi*f/c0*np.sqrt(-(x-1j*np.finfo(float).eps))  # eps to ensure positive square-root
    gamma2dbmm  = lambda x: mag2db(np.exp(x.real*1e-3))  # loss dB/mm
    
    # WR12 example
    f = np.linspace(60, 90, 250)*1e9
    w = 3.0988e-3
    h = 1.5494e-3
    sr = 0.28  # relative conductivity of Brass to copper
    rw = RW(w,h,f, sr=sr)
    
    # assume uncertainties, the order is as follows [w, h, mur, sr, er, tand]
    max_dim_deviation = 0.5e-3
    dw = max_dim_deviation/3
    dh = max_dim_deviation/3
    dmur  = 0
    dsr   = 0
    der   = 0
    dtand = 0
    U = np.diag([dw, dh, dmur, dsr, der, dtand])**2
    rw.update_jac()
    
    # effective relative permittivity
    Jereff = rw.jac_ereff
    Uereff = np.array([J.dot(U).dot(J.T) for J in Jereff])
    ereff = np.array([munc.ucomplex(x, covariance=u) for x,u in zip(rw.ereff, Uereff)])
    
    # propagation constant
    Jgamma = rw.jac_gamma
    Ugamma = np.array([J.dot(U).dot(J.T) for J in Jgamma])
    gamma  = np.array([munc.ucomplex(x, covariance=u) for x,u in zip(rw.gamma, Ugamma)])
    
    # characteristic impedance
    JZ0 = rw.jac_Z0
    UZ0 = np.array([J.dot(U).dot(J.T) for J in JZ0])
    Z0  = np.array([munc.ucomplex(x, covariance=u) for x,u in zip(rw.Z0, UZ0)])
    
    plt.figure()
    mu  = munc.get_value(ereff).real
    std = munc.get_stdunc(ereff).real
    k = 2
    plt.plot(f*1e-9, mu, lw=2)
    plt.fill_between(f*1e-9, mu+k*std, mu-k*std, alpha=0.3)
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('Relative effective permittivity')
    plt.title('95% uncertainty coverage of a Gaussian distribution')
    
    plt.figure()
    mu  = munc.get_value(Z0).real
    std = munc.get_stdunc(Z0).real
    k = 2
    plt.plot(f*1e-9, mu, lw=2)
    plt.fill_between(f*1e-9, mu+k*std, mu-k*std, alpha=0.3)
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('Characteristic impedance (ohm)')
    plt.title('95% uncertainty coverage of a Gaussian distribution')
    
    plt.figure()
    loss_dbcm = gamma2dbmm(munc.umath.real(gamma))*10
    mu  = munc.get_value(loss_dbcm)
    std = munc.get_stdunc(loss_dbcm)
    k = 2
    plt.plot(f*1e-9, mu, lw=2)
    plt.fill_between(f*1e-9, mu+k*std, mu-k*std, alpha=0.3)
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('Loss (dB/cm)')
    plt.title('95% uncertainty coverage of a Gaussian distribution')
    
    plt.show()

# EOF