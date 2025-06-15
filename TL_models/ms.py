'''
@author: Ziad (zi.hatab@gmail.com)

This is an implementation of a microstrip line model based on the reference [1].
I also included the option to calculate the Jacobian with respect to the inputs.

[1] F. Schneider and W. Heinrich, "Model of thin-film microstrip line for circuit design," 
in IEEE Transactions on Microwave Theory and Techniques, 
vol. 49, no. 1, pp. 104-110, Jan. 2001, doi: 10.1109/22.899967
'''

# pip install numpy scipy -U
import numpy as np
from scipy.integrate import quad
import scipy.optimize as so  # for derivative computation

def ZL0(w,h):
    # intermediat function from [1]
    ep0 = 8.8541878128e-12 # permittivity of free space (F/m)
    mu0 = 1.25663706212e-6 # permeability of free space (H/m)
    eta0 = np.sqrt(mu0/ep0)
    F1 = 6 + (2*np.pi - 6)*np.exp(-(30.666*h/w)**0.7528)
    return eta0/2/np.pi*np.log(F1*h/w + np.sqrt(1 + (2*h/w)**2))

def ereff0(w,h,er):
    # intermediat function from [1]
    a = 1 + 1/49*np.log(((w/h)**4 + (w/h/52)**2)/((w/h)**4 + 0.432)) + 1/18.7*np.log(1 + (w/18.1/h)**3)
    b = 0.564*((er - 0.9)/(er + 3))**0.053
    return (er + 1)/2 + (er - 1)/2*(1 + 10*h/w)**(-a*b)

def Fer(w,h,f,er):
    # intermediat function from [1]
    # high freqency correction factor for effective relative permittivity
    GHzcm = 1e9 * 1e-2  # GHz*cm to Hz*m
    P1 = 0.27488 + (w/h)*(0.6315 + 0.525/(1 + 0.157*f*h/GHzcm)**20) - 0.065683*np.exp(-8.7513*w/h)
    P2 = 0.33622*(1 - np.exp(-0.03442*er))
    P3 = 0.0363*np.exp(-4.6*w/h)*(1 - np.exp(-((f*h)/(3.87*GHzcm))**4.97))
    P4 = 1 + 2.751*(1 - np.exp(-(er/15.916)**8))
    P = P1*P2*((0.1844 + P3*P4)*10*(f*h/GHzcm))**1.5763
    
    ereff0_val = ereff0(w, h, er)
    return (er/ereff0_val) - (er/ereff0_val - 1)/(1 + P)

def FZ(w,h,f,er):
    # intermediat function from [1]
    # high frequency correction factor for characteristic impedance
    ereff0_val = ereff0(w,h,er)
    Fer_val = Fer(w,h,f,er)
    return (ereff0_val*Fer_val - 1)/(ereff0_val - 1)/np.sqrt(Fer_val)

def C(w,h,t,f,er):
    """
    Calculates the capacitance per-unit-length.

    Parameters:
    - w (float): Width of the signal line (m).
    - h (float): Height of the substrate (m).
    - t (float): Thickness of the conductor (m).
    - f (float): Frequency (Hz).
    - er (float): Relative permittivity of the substrate.

    Returns:
    - C (float): Capacitance of the transmission line (F/m).
    """
    c0 = 299792458         # speed of light in vacuum (m/s)
    coth = lambda x: 1/np.tanh(x)
    weq0 = w + t/np.pi*np.log( 1 + 4*np.exp(1)/(t/h)/coth(np.sqrt(6.517*w/h))**2 )
    weqZ = w + (weq0 - w)/2*( 1 + 1/np.cosh(np.sqrt(er - 1)) )
    ereff = ereff0(weqZ,h,er)*( ZL0(weq0,h)/ZL0(weqZ,h) )**2
    Ca = 1/c0/ZL0(weq0,h)
    FC = np.sqrt(Fer(w,h,f,er))/FZ(w,h,f,er) # high frequency correction factor
    return ereff*Ca*FC, ereff, Ca

def G(w,h,t,f,er,tand):
    """
    Calculates the conductance per-unit-length.

    Parameters:
    - w (float): Width of the signal line (m).
    - h (float): Height of the substrate (m).
    - t (float): Thickness of the conductor (m).
    - f (float): Frequency (Hz).
    - er (float): Relative permittivity of the substrate.
    - tand (float): Loss tangent.

    Returns:
    - G (float): Conductance of the transmission line (S/m).
    """
    _, ereff, Ca = C(w,h,t,f,er)
    Cep = (ereff - 1)/(er - 1)*er*Ca
    omega = 2*np.pi*f
    return omega*Cep*tand

def R(w,h,t,f,mur,sr,er,wg):
    """
    Calculates the resistance per-unit-length.

    Parameters:
    - w (float): Width of the signal line (m).
    - h (float): Height of the substrate (m).
    - t (float): Thickness of the conductor (m).
    - f (float): Frequency (Hz).
    - mur (float): Relative permeability (for both conductor and substrate).
    - sr (float): Relative conductivity to copper.
    - er (float): Relative permittivity of the substrate.
    - wg (float): Width of the ground plane (m).

    Returns:
    - R (float): Resistance of the transmission line (Ohm/m).
    - Rse (function): skin-effect resistance per-unit-length (Ohm/m).
    """
    s0  = 58e6             # conductivity of copper (S/m)
    mu0 = 1.25663706212e-6 # permeability of free space (H/m)
    mu = mu0*mur
    s  = s0*sr
    coth = lambda x: 1/np.tanh(x)
    weq0 = w + t/np.pi*np.log( 1 + 4*np.exp(1)/(t/h)/coth(np.sqrt(6.517*w/h))**2 )
    weqZ = w + (weq0 - w)/2*( 1 + 1/np.cosh(np.sqrt(er - 1)) )
    
    ereff = ereff0(weqZ,h,er)*( ZL0(weq0,h)/ZL0(weqZ,h) )**2
    ZL    = ZL0(weqZ,h)/np.sqrt(ereff0(weqZ,h,er))
    
    A = 1 + h/weq0 * (1 + 1.25/np.pi * np.log(2 * h/t))
    Rs = lambda x: np.sqrt(np.pi*x*mu/s)
    if w/h <= 1:
        alpha_c = lambda x: 0.1589 * A * (Rs(x) / (h * ZL)) * ((32 - (weq0 / h)**2) / (32 + (weq0 / h)**2))
    else:
        alpha_c = lambda x: 7.0229e-6 * A * (Rs(x) * ZL * ereff) / h * (weq0 / h + 0.667 * (weq0 / h) / (weq0 / h + 1.444))
    Rse = lambda x: 2*ZL*alpha_c(x)
    Rdcw = 1/s/w/t
    Rdcg = 1/s/wg/t
    Rdc = Rdcw + Rdcg
    f0 = (2 * Rdcw * Rdcg) / (mu * (Rdcw + Rdcg))
    fse = (1.6 + (10 * (t / w)) / (1 + (w / h))) / (np.pi * mu * s * t**2)
    R = Rdc + (Rse(fse) * ((np.sqrt(f/fse) + np.sqrt(1 + (f/fse)**2)) / (1 + np.sqrt(f/fse))) - (Rse(fse) - Rdc) / np.sqrt(1 + (f/f0)**2) - Rdc) / (1 + 0.2 / (1 + w/h) * np.log(1 + fse/f))
    return R, Rse

def L(w,h,t,f,mur,sr,er,wg):
    """
    Calculates the inductance per-unit-length.

    Parameters:
    - w (float): Width of the signal line (m).
    - h (float): Height of the substrate (m).
    - t (float): Thickness of the conductor (m).
    - f (float): Frequency (Hz).
    - mur (float): Relative permeability (for both conductor and substrate).
    - sr (float): Relative conductivity to copper.
    - er (float): Relative permittivity of the substrate.
    - wg (float): Width of the ground plane (m).

    Returns:
    - L (float): Inductance of the transmission line (H/m).
    """
    c0 = 299792458         # speed of light in vacuum (m/s)
    ep0 = 8.8541878128e-12 # permittivity of free space (F/m)
    s0  = 58e6             # conductivity of copper (S/m)
    mu0 = 1.25663706212e-6 # permeability of free space (H/m)
    mu = mu0*mur
    s  = s0*sr
    _, ereff, Ca = C(w,h,t,f,er)

    La = 1/(c0**2 * Ca)
    _, Rse = R(w,h,t,f,mur,sr,er,wg)
    Li = lambda x: Rse(x)/(2*np.pi*x)
    
    K4 = lambda z: (z**4 / 24) * (np.log(z) - 25/12)
    def Km(a, b, c, d, h):
        integ_eval = K4(-a/2 - 1j*b/2 - c/2 - 1j*d/2 - 1j*h) - K4(-a/2 - 1j*b/2 - c/2 + 1j*d/2 - 1j*h) - K4(-a/2 - 1j*b/2 + c/2 - 1j*d/2 - 1j*h) + K4(-a/2 - 1j*b/2 + c/2 + 1j*d/2 - 1j*h) - K4(-a/2 + 1j*b/2 - c/2 - 1j*d/2 - 1j*h) + K4(-a/2 + 1j*b/2 - c/2 + 1j*d/2 - 1j*h) + K4(-a/2 + 1j*b/2 + c/2 - 1j*d/2 - 1j*h) - K4(-a/2 + 1j*b/2 + c/2 + 1j*d/2 - 1j*h) - K4(a/2 - 1j*b/2 - c/2 - 1j*d/2 - 1j*h) + K4(a/2 - 1j*b/2 - c/2 + 1j*d/2 - 1j*h) + K4(a/2 - 1j*b/2 + c/2 - 1j*d/2 - 1j*h) - K4(a/2 - 1j*b/2 + c/2 + 1j*d/2 - 1j*h) + K4(a/2 + 1j*b/2 - c/2 - 1j*d/2 - 1j*h) - K4(a/2 + 1j*b/2 - c/2 + 1j*d/2 - 1j*h) - K4(a/2 + 1j*b/2 + c/2 - 1j*d/2 - 1j*h) + K4(a/2 + 1j*b/2 + c/2 + 1j*d/2 - 1j*h)
        return np.real(-integ_eval)
    Ks = lambda a, b: np.real(4 * (K4(a) + K4(1j*b)) - 2 * (K4(a + 1j*b) + K4(a - 1j*b))) + (1/3) * np.pi * a * b**3

    Ldc = (-mu/(2*np.pi*t**2))*((1/w**2)*Ks(w, t) - (2/(w*wg)) * Km(w, t, wg, t, h + t) + (1/wg**2) * Ks(wg, t))

    Rdcw = 1/s/w/t
    Rdcg = 1/s/wg/t
    f0 = (2 * Rdcw * Rdcg) / (mu * (Rdcw + Rdcg))
    fse = (1.6 + (10 * (t / w)) / (1 + (w / h))) / (np.pi * mu * s * t**2)
    
    L = La + Li(fse) / (1 + np.sqrt(f/fse)) + (Ldc - La - Li(fse)) / np.sqrt(1 + (f/f0)**2)
    FL = np.sqrt(Fer(w,h,f,er))*FZ(w,h,f,er) # high frequency correction factor  
    return L*FL

def get_all_paras(x, ms):
    '''
    This function is used to help with the computation of the Jacobian of the parameters
    '''
    # x = [w, h, t, wg, mur, sr, er, tand]
    ms.w    = x[0]
    ms.h    = x[1]
    ms.t    = x[2]
    ms.wg   = x[3]
    ms.mur  = x[4]
    ms.sr   = x[5]
    ms.er   = x[6]
    ms.tand = x[7]
    ms.update()
    
    # convert complex-valued array to real-valued equivalent
    hh = lambda x: np.kron(x.real, [1,0]) + np.kron(x.imag, [0,1])
    
    # microstrip line parameters
    gamma = ms.gamma
    h_gamma = hh(gamma)
    ereff = ms.ereff
    h_ereff = hh(ereff)
    Z0 = ms.Z0
    h_Z0 = hh(Z0)
    return np.hstack((h_gamma, h_ereff, h_Z0))

class MS:
    """
    Analytical model of microstrip line based on [1] (see comments at top of this file).

    Parameters
    ----------
    w : number
        width of signal line in meters.
    h : number
        height of dielectric substrate in meters.
    t : number
        Thickness of conductor in meters.
    wg : number
        Width of GND plane in meters.
    f : number or 1d-array
        frequency in Hz.
    mur : float number
        relative permeability
    sr : float number
        relative conductivity to copper (5.8e7 S/m)
    er : float number
        real-part relative permittivity.
    tand : float number
        loss tangent.
    """
    def __init__(self, w, h, t, wg, f, mur=1, sr=1, er=1.00001, tand=0):
        self.w    = w
        self.h    = h
        self.t    = t
        self.wg   = wg
        self.f    = np.atleast_1d(f)
        self.mur  = mur
        self.sr   = sr
        self.er   = er
        self.tand = tand
        self.update()  # run the code
    
    def update(self):
        mu0 = 1.25663706212e-6
        ep0 = 8.8541878128e-12
        c0  = 1/np.sqrt(mu0*ep0) # 299792458 # speed of light in vacuum (m/s)
        
        w    = self.w
        h    = self.h
        t    = self.t
        wg   = self.wg
        f    = self.f
        mur  = self.mur
        sr   = self.sr
        er   = self.er
        tand = self.tand
        
        # per unit length parameters
        Cp  = np.array([C(w, h, t, ff, er)[0] for ff in f])
        Gp  = np.array([G(w, h, t, ff, er, tand) for ff in f])
        Lp  = np.array([L(w, h, t, ff, mur, sr, er, wg) for ff in f])
        Rp  = np.array([R(w, h, t, ff, mur, sr, er, wg)[0] for ff in f])
                
        omega = 2*np.pi*f
        Z = Rp + 1j*omega*Lp
        Y = Gp + 1j*omega*Cp
        
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
        t    = self.t
        wg   = self.wg
        mur  = self.mur
        sr   = self.sr
        er   = self.er
        tand = self.tand
        
        N = len(self.f)
        M = 2*N
        eps = np.sqrt(np.finfo(float).eps)
        x   = [w, h, t, wg, mur, sr, er, tand]
        big_J = so.approx_fprime(x, get_all_paras, [eps]*len(x), self)
        
        # split the jacobian in correct order
        self.jac_gamma = big_J[:M].reshape((N,2,-1))
        self.jac_ereff = big_J[M:2*M].reshape((N,2,-1))
        self.jac_Z0    = big_J[2*M:3*M].reshape((N,2,-1))
        self.jac_Gamma = np.array([ np.array([[(1/2/z).real, (1j/2/z).real],[(1/2/z).imag, (1j/2/z).imag]])@Jz for z,Jz in zip(self.Z0, self.jac_Z0) ])
                
        # undo the changes done by the function so.approx_fprime().
        self.w    = w
        self.h    = h
        self.t    = t
        self.wg   = wg
        self.mur  = mur
        self.sr   = sr
        self.er   = er
        self.tand = tand

# Example
if __name__ == '__main__':
    import matplotlib.pyplot as plt # for plotting
    import metas_unclib as munc     # metas package to propagate general uncertainties through functions. python -m pip install metas_unclib
    munc.use_linprop()

    # reference material parameters
    mu0 = 1.25663706212e-6
    ep0 = 8.8541878128e-12
    c0  = 1/np.sqrt(mu0*ep0) # 299792458 # speed of light in vacuum (m/s)
    s0  = 58e6 # copper conductivity
    
    # useful functions
    mag2db = lambda x: 20*np.log10(abs(x))
    db2mag = lambda x: 10**(x/20)
    gamma2ereff = lambda x,f: -(c0/2/np.pi/f*x)**2
    ereff2gamma = lambda x,f: 2*np.pi*f/c0*np.sqrt(-(x-1j*np.finfo(float).eps))  # eps to ensure positive square-root
    gamma2dbmm  = lambda x: mag2db(np.exp(x.real*1e-3))  # loss dB/mm
    
    # microstrip line example from [1]: F. Schnieder and W. Heinrich, "Model of thin-film microstrip line for circuit design," doi: 10.1109/22.899967
    #f = np.linspace(.01, 1000, 1000)*1e9
    f = np.logspace(0, 3, 512)*1e9
    w = 8e-6
    h = 1.7e-6
    wg = 88e-6
    t  = 0.8e-6
    sr = 2.5/5.8  # relative conductivity to copper
    er = 2.7
    tand = 0.015
    ms = MS(w, h, t, wg, f, sr=sr, er=er, tand=tand)
    
    # uncertainties 10%, the order is as follows [w, h, t, wg, mur, sr, er, tand]
    dw = 0.1*w
    dh = 0.1*h
    dt  = 0.1*t
    dwg = 0.1*wg
    dmur  = 0
    dsr   = 0.1*sr
    der   = 0.1*er
    dtand = 0.1*tand
    U = np.diag([dw, dh, dt, dwg, dmur, dsr, der, dtand])**2
    ms.update_jac()
    
    # effective relative permittivity
    Jereff = ms.jac_ereff
    Uereff = np.array([J.dot(U).dot(J.T) for J in Jereff])
    ereff = np.array([munc.ucomplex(x, covariance=u) for x,u in zip(ms.ereff, Uereff)])
    
    # propagation constant
    Jgamma = ms.jac_gamma
    Ugamma = np.array([J.dot(U).dot(J.T) for J in Jgamma])
    gamma  = np.array([munc.ucomplex(x, covariance=u) for x,u in zip(ms.gamma, Ugamma)])
    
    # characteristic impedance
    JZ0 = ms.jac_Z0
    UZ0 = np.array([J.dot(U).dot(J.T) for J in JZ0])
    Z0  = np.array([munc.ucomplex(x, covariance=u) for x,u in zip(ms.Z0, UZ0)])
    
    plt.figure()
    val = munc.umath.real(ereff)
    mu  = munc.get_value(val)
    std = munc.get_stdunc(val)
    k = 2
    plt.semilogx(f*1e-9, mu, lw=2)
    plt.fill_between(f*1e-9, mu+k*std, mu-k*std, alpha=0.3)
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('Relative effective permittivity')
    plt.title('95% uncertainty coverage of a Gaussian distribution')
    
    plt.figure()
    val = munc.umath.real(Z0)
    mu  = munc.get_value(val)
    std = munc.get_stdunc(val)
    k = 2
    plt.semilogx(f*1e-9, mu, lw=2)
    plt.fill_between(f*1e-9, mu+k*std, mu-k*std, alpha=0.3)
    ax = plt.gca()
    ax2 = ax.twinx()
    val = munc.umath.imag(Z0)
    mu  = munc.get_value(val)
    std = munc.get_stdunc(val)
    ax2.semilogx(f*1e-9, mu, lw=2, color='red')
    ax2.fill_between(f*1e-9, mu+k*std, mu-k*std, alpha=0.3, color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    plt.xlabel('Frequency (GHz)')
    ax.set_ylabel('real(Z0) (ohm)')
    ax2.set_ylabel('imag(Z0) (ohm)')
    plt.title('95% uncertainty coverage of a Gaussian distribution')
    
    plt.figure()
    val = gamma2dbmm(munc.umath.real(gamma))*10
    mu  = munc.get_value(val)
    std = munc.get_stdunc(val)
    k = 2
    plt.semilogx(f*1e-9, mu, lw=2)
    plt.fill_between(f*1e-9, mu+k*std, mu-k*std, alpha=0.3)
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('Loss (dB/cm)')
    plt.title('95% uncertainty coverage of a Gaussian distribution')
    
    
    plt.show()

# EOF