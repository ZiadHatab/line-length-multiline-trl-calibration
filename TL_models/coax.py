'''
@author: Ziad (zi.hatab@gmail.com)

This is an implementation of a coaxial line model based on the reference [1].
Also, I included the option to calculate Jacobian with respect to the inputs.

[1] F. M. Tesche, "A Simple Model for the Line Parameters of a Lossy Coaxial 
Cable Filled With a Nondispersive Dielectric," 
in IEEE Transactions on Electromagnetic Compatibility, 
vol. 49, no. 1, pp. 12-17, Feb. 2007, 
doi: https://doi.org/10.1109/TEMC.2006.888185
'''

# pip install numpy scipy -U
import numpy as np
import scipy.optimize as so  # for derivative computation

def Za(a, f, mur=1, sr=1):
    '''
    Parameters
    ----------
    a : number
        inner radius.
    f : number
        frequency.
    mur : number, optional
        relative permeability. The default is 1.
    sr : number, optional
        relative conductivity to copper conductivity (58 MS/m). The default is 1.

    Returns
    -------
    number
        complex impedance per unit length of inner conductor.
    '''
    
    s0  = 58e6 # (S/m)
    mu0 = 1.25663706212e-6
    mu = mur*mu0
    sigma = sr*s0
    omega = 2*np.pi*f
    
    Rdc  = 1/np.pi/a**2/sigma
    Lint = mu/8/np.pi
    Zhf  = (1+1j)/2/np.pi/a*np.sqrt(omega*mu/2/sigma) 
    Rhf  = Zhf.real
    Lhf  = Zhf.imag/omega
    
    return Rdc + 1j*omega*Lint*( Rhf + 1j*omega*Lhf )/( Rhf + 1j*omega*(Lhf + Lint) )


def Zb(b, t, f, mur=1, sr=1):
    '''
    Parameters
    ----------
    b : number
        outer radius (from inside edge).
    t : number
        thickness of outer conductor.
    f : number
        frequency.
    mur : number, optional
        relative permeability. The default is 1.
    sr : number, optional
        relative conductivity to copper conductivity (58 MS/m). The default is 1.

    Returns
    -------
    number
        complex impedance per unit length of outer conductor.
    '''
    
    s0  = 58e6 # (S/m)
    mu0 = 1.25663706212e-6
    mu = mur*mu0
    sigma = sr*s0
    omega = 2*np.pi*f
        
    c = b + t
    
    Rdc  = 1/2/np.pi/b/t/sigma
    Lint = mu/2/np.pi*( c**4*np.log(c/b)/(c**2-b**2)**2 + (b**2-3*c**2)/4/(c**2-b**2))
    Zhf  = (1+1j)/2/np.pi/b*np.sqrt(omega*mu/2/sigma) 
    Rhf  = Zhf.real
    Lhf  = Zhf.imag/omega
    
    return Rdc + 1j*omega*Lint*( Rhf + 1j*omega*Lhf )/( Rhf + 1j*omega*(Lhf + Lint) )

def L(a,b,mur=1):
    '''
    Parameters
    ----------
    a   : radius of the inner conductor in meters
    b   : radius of the outer conductor in meters
    mur : number, optional
        relative permeability to vacuum permeability (1.25663706212e-6). The default is 1.
        
    Returns
    -------
    high frequency inductance per unit length
    '''
    mu0 = 1.25663706212e-6
    return mu0*mur/2/np.pi*np.log(b/a)  # correction of [1] b/a not a/b
    
def C(a,b,er=1):
    '''
    Parameters
    ----------
    a  : radius of the inner conductor in meters
    b  : radius of the outer conductor in meters
    er : number, optional
        relative permittivity to vacuum (8.8541878128e-12). The default is 1.
        
    Returns
    -------
    shunt capacitance per unit length
    '''
    ep0 = 8.8541878128e-12
    return 2*np.pi*er*ep0/np.log(b/a)  # correction of [1] b/a not a/b

def G(a,b,f,er=1,tand=0):
    '''
    Parameters
    ----------
    a  : radius of the inner conductor in meters
    b  : radius of the outer conductor in meters
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
    return omega*C(a,b,er)*tand


def get_all_paras(x, coax):
    '''
    This function is used to compute the Jacobian of the parameters
    '''
    # x = [a, b, t, mur, sr, er, tand]
    coax.a    = x[0]
    coax.b    = x[1]
    coax.t    = x[2]
    coax.mur  = x[3]
    coax.sr   = x[4]
    coax.er   = x[5]
    coax.tand = x[6]
    coax.update()
    
    # convert complex-valued array to real-valued equivalent
    hh = lambda x: np.kron(x.real, [1,0]) + np.kron(x.imag, [0,1])
    
    # coax line parameters
    gamma = coax.gamma
    h_gamma = hh(gamma)
    ereff = coax.ereff
    h_ereff = hh(ereff)
    Z0 = coax.Z0
    h_Z0 = hh(Z0)
    return np.hstack((h_gamma, h_ereff, h_Z0))

class COAX:
    """
    Analytical model of coaxial cable based on [1] (see comments at top of this file).

        Parameters
    ----------
    a : number
        radius of inner conductor (signal) in meters.
    b : number
        radius of outer conductor (GND) from inside in meters.
    t : number
        Thickness of GND in meters.
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
    def __init__(self, a, b, t, f, mur=1, sr=1, er=1, tand=0):
        self.a     = a
        self.b     = b
        self.t     = t
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
        
        a    = self.a
        b    = self.b
        t    = self.t
        f    = self.f
        mur  = self.mur
        sr   = self.sr
        er   = self.er
        tand = self.tand
        
        omega = 2*np.pi*f
        
        # per unit length parameters
        Cp  = np.array([C(a,b,er) for ff in f])
        Gp  = np.array([G(a,b,ff,er,tand) for ff in f])
        Lp  = np.array([L(a,b,mur) for ff in f])
        Zap = np.array([Za(a,ff,mur,sr) for ff in f])
        Zbp = np.array([Zb(b,t,ff,mur,sr) for ff in f])
        
        Z = 1j*omega*Lp + Zap + Zbp
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
        a    = self.a
        b    = self.b
        t    = self.t
        mur  = self.mur
        sr   = self.sr
        er   = self.er
        tand = self.tand
        
        N = len(self.f)
        M = 2*N
        eps = np.sqrt(np.finfo(float).eps)
        x   = [a, b, t, mur, sr, er, tand]
        big_J = so.approx_fprime(x, get_all_paras, [eps]*len(x), self)
        
        # split the jacobian in correct order
        self.jac_gamma = big_J[:M].reshape((N,2,-1))
        self.jac_ereff = big_J[M:2*M].reshape((N,2,-1))
        self.jac_Z0    = big_J[2*M:3*M].reshape((N,2,-1))
        self.jac_Gamma = np.array([ np.array([[(1/2/z).real, (1j/2/z).real],[(1/2/z).imag, (1j/2/z).imag]])@Jz for z,Jz in zip(self.Z0, self.jac_Z0) ])
                
        # undo the changes done by the function so.approx_fprime().
        self.a    = a
        self.b    = b
        self.t    = t
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
    s0  = 58e6 # copper conductivity
    
    # useful functions
    mag2db = lambda x: 20*np.log10(abs(x))
    db2mag = lambda x: 10**(x/20)
    gamma2ereff = lambda x,f: -(c0/2/np.pi/f*x)**2
    ereff2gamma = lambda x,f: 2*np.pi*f/c0*np.sqrt(-(x-1j*np.finfo(float).eps))  # eps to ensure positive square-root
    gamma2dbmm  = lambda x: mag2db(np.exp(x.real*1e-3))  # loss dB/mm
    
    # k-band connector example
    f = np.linspace(0.1, 40, 250)*1e9
    a = 1.270e-3/2
    b = 2.92e-3/2
    t = 6.555e-3/2 - b
    sr = 0.01  # relative conductivity to copper
    coax = COAX(a, b, t, f, sr=sr)
    
    
    # assume uncertainties, the order is as follows [w, h, mur, sr, er, tand]
    da = 0.0025e-3/2
    db = 0.005e-3/2
    dt = 0
    dmur  = 0
    dsr   = 0
    der   = 0.02
    dtand = 0
    U = np.diag([da, db, dt, dmur, dsr, der, dtand])**2
    coax.update_jac()
    
    # effective relative permittivity
    Jereff = coax.jac_ereff
    Uereff = np.array([J.dot(U).dot(J.T) for J in Jereff])
    ereff = np.array([munc.ucomplex(x, covariance=u) for x,u in zip(coax.ereff, Uereff)])
    
    # propagation constant
    Jgamma = coax.jac_gamma
    Ugamma = np.array([J.dot(U).dot(J.T) for J in Jgamma])
    gamma  = np.array([munc.ucomplex(x, covariance=u) for x,u in zip(coax.gamma, Ugamma)])
    
    # characteristic impedance
    JZ0 = coax.jac_Z0
    UZ0 = np.array([J.dot(U).dot(J.T) for J in JZ0])
    Z0  = np.array([munc.ucomplex(x, covariance=u) for x,u in zip(coax.Z0, UZ0)])
    
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