"""
@author: Ziad (zi.hatab@gmail.com, https://github.com/ZiadHatab)

This example demonstrates error term sensitivity analysis using Monte Carlo simulation.
It compares a commercial ISS lengths against optimal computed ones from my script.
"""
import copy
import skrf as rf
import numpy as np
import matplotlib.pyplot as plt

# my code
from lengthcalc import LineLengthCalculator
from TL_models.cpw import CPW

class PlotSettings:
    # to make plots look better for publication
    # https://matplotlib.org/stable/tutorials/introductory/customizing.html
    def __init__(self, font_size=10, latex=False): 
        self.font_size = font_size 
        self.latex = latex
    def __enter__(self):
        plt.style.use('seaborn-v0_8-paper')
        # make svg output text and not curves
        plt.rcParams['svg.fonttype'] = 'none'
        # fontsize of the axes title
        plt.rc('axes', titlesize=self.font_size*1.2)
        # fontsize of the x and y labels
        plt.rc('axes', labelsize=self.font_size)
        # fontsize of the tick labels
        plt.rc('xtick', labelsize=self.font_size)
        plt.rc('ytick', labelsize=self.font_size)
        # legend fontsize
        plt.rc('legend', fontsize=self.font_size*1)
        # fontsize of the figure title
        plt.rc('figure', titlesize=self.font_size)
        # controls default text sizes
        plt.rc('text', usetex=self.latex)
        #plt.rc('font', size=self.font_size, family='serif', serif='Times New Roman')
        plt.rc('lines', linewidth=1.5)
    def __exit__(self, exception_type, exception_value, traceback):
        plt.style.use('default')

def s2t(S, pseudo=False):
    T = S.copy()
    T[0,0] = -(S[0,0]*S[1,1]-S[0,1]*S[1,0])
    T[0,1] = S[0,0]
    T[1,0] = -S[1,1]
    T[1,1] = 1
    return [T,S[1,0]] if pseudo else T/S[1,0]

def t2s(T, pseudo=False):
    S = T.copy()
    S[0,0] = T[0,1]
    S[0,1] = T[0,0]*T[1,1]-T[0,1]*T[1,0]
    S[1,0] = 1
    S[1,1] = -T[1,0]
    return [S,T[1,1]] if pseudo else S/T[1,1]

def Qnm(Zn, Zm):
    # Impedance transformer in T-parameters from on Eqs. (86) and (87) in
    # R. Marks and D. Williams, "A general waveguide circuit theory," 
    # Journal of Research (NIST JRES), National Institute of Standards and Technology, 1992.
    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4914227/
    Gnm = (Zm-Zn)/(Zm+Zn)
    return np.sqrt(Zn.real/Zm.real*(Zm/Zn).conjugate())/np.sqrt(1-Gnm**2)*np.array([[1, Gnm],[Gnm, 1]])
    
def TL(l, cpw, Z01=None, Z02=None):
    # create skrf network from a general transmission line model from an cpw object (file: cpw.py)
    N = len(cpw.Z0)  # number of frequency points
    Z01 = cpw.Z0 if Z01 is None else np.atleast_1d(Z01)*np.ones(N)
    Z02 = Z01 if Z02 is None else np.atleast_1d(Z02)*np.ones(N)
    S = []
    for g,zc,z01,z02 in zip(cpw.gamma, cpw.Z0, Z01, Z02):
        T = Qnm(z01,zc)@np.diag([np.exp(-l*g), np.exp(l*g)])@Qnm(zc,z02)
        S.append(t2s(T))
    freq = rf.Frequency.from_f(cpw.f, unit='Hz')
    freq.unit = 'GHz'
    return rf.Network(s=np.array(S), frequency=freq, name=f'l={l*1e3:.2f}mm')

def offset_open(l,cpw, l2=None):
    # create a 2-port offset open network from cpw object (file: cpw.py)
    if l2 is None:
        l2 = l
    freq = rf.Frequency.from_f(cpw.f, unit='Hz')
    freq.unit = 'GHz'
    single_port_1 = rf.Network(s=np.array([np.exp(-2*l*g) for g in cpw.gamma]), frequency=freq, name='open')
    single_port_2 = rf.Network(s=np.array([np.exp(-2*l2*g) for g in cpw.gamma]), frequency=freq, name='open')
    return rf.two_port_reflect(single_port_1,single_port_2)  # make it 2-port (S11=S22)

def add_white_noise(NW, covs):
    # add white noise to a network's S-parameters
    NW_new = NW.copy()
    for inx,(s,cov) in enumerate(zip(NW_new.s,covs)):
        h = np.kron(s.flatten('F').real,[1,0]) + np.kron(s.flatten('F').imag,[0,1])
        noise = np.random.multivariate_normal(np.zeros(h.size), cov)
        E = np.kron(np.eye(len(s)*2), [1,1j])
        NW_new.s[inx] = s + E.dot(noise).reshape((2,2),order='F')
    return NW_new

def get_norm_coef_MAE(coef_MC, coefs_ideal, name):
    '''
    compute mean absolute error of normalized error terms.
    '''
    EDF_mc = np.array([x['forward directivity'] for x in coef_MC])
    ESF_mc = np.array([x['forward source match'] for x in coef_MC])
    ERF_mc = np.array([x['forward reflection tracking'] for x in coef_MC])
    EDR_mc = np.array([x['reverse directivity'] for x in coef_MC])
    ESR_mc = np.array([x['reverse source match'] for x in coef_MC]) 
    ERR_mc = np.array([x['reverse reflection tracking'] for x in coef_MC])

    EDF_ideal = coefs_ideal['forward directivity']
    ESF_ideal = coefs_ideal['forward source match']
    ERF_ideal = coefs_ideal['forward reflection tracking']
    EDR_ideal = coefs_ideal['reverse directivity']
    ESR_ideal = coefs_ideal['reverse source match']
    ERR_ideal = coefs_ideal['reverse reflection tracking']

    if name == 'a21_a11':
        a21_a11_mc = -ESF_mc/(ERF_mc - EDF_mc*ESF_mc)
        a21_a11_ideal = -ESF_ideal/(ERF_ideal - EDF_ideal*ESF_ideal)
        return np.array([ abs(x-a21_a11_ideal) for x in a21_a11_mc ]).mean(axis=0)
    elif name == 'a12':
        a12_mc = EDF_mc
        a12_ideal = EDF_ideal
        return np.array([ abs(x-a12_ideal) for x in a12_mc ]).mean(axis=0)
    elif name == 'b12_b11':
        b12_b11_mc = ESR_mc/(ERR_mc - EDR_mc*ESR_mc) 
        b12_b11_ideal = ESR_ideal/(ERR_ideal - EDR_ideal*ESR_ideal)
        return np.array([ abs(x-b12_b11_ideal) for x in b12_b11_mc ]).mean(axis=0)
    elif name == 'b21':
        b21_mc = -EDR_mc
        b21_ideal = -EDR_ideal
        return np.array([ abs(x-b21_ideal) for x in b21_mc ]).mean(axis=0)

if __name__=='__main__':
    # useful functions
    c0 = 299792458   # speed of light in vacuum (m/s)
    mag2db = lambda x: 20*np.log10(abs(x))
    db2mag = lambda x: 10**(x/20)
    gamma2ereff = lambda x,f: -(c0/2/np.pi/f*x)**2
    ereff2gamma = lambda x,f: 2*np.pi*f/c0*np.sqrt(-(x-1j*np.finfo(float).eps))  # eps to ensure positive square-root
    gamma2dbmm  = lambda x: mag2db(np.exp(x.real*1e-3))  # losses dB/mm
    
    freq = rf.Frequency.from_f(np.linspace(1, 150, 150)*1e9) # frequency axis
    f = freq.f

    # CPW model parameters 
    w, s, wg, t = 49.1e-6, 25.5e-6, 273.3e-6, 4.9e-6
    Dk = 9.9
    Df = 0.0
    sig = 4.11e7  # conductivity of Gold
    cpw = CPW(w,s,wg,t,f,Dk*(1-1j*Df),sig)
    line_lengths = np.array([200e-6, 450e-6, 900e-6, 1800e-6, 3500e-6, 5250e-6])
    line_lengths = line_lengths - line_lengths[0]

    L1 = TL(line_lengths[0], cpw, cpw.Z0)
    L2 = TL(line_lengths[1], cpw, cpw.Z0)
    L3 = TL(line_lengths[2], cpw, cpw.Z0)
    L4 = TL(line_lengths[3], cpw, cpw.Z0)
    L5 = TL(line_lengths[4], cpw, cpw.Z0)
    L6 = TL(line_lengths[5], cpw, cpw.Z0)
    OPEN = offset_open(0, cpw)

    # mTRL definition
    lines = [L1, L2, L3, L4, L5, L6]
    reflect = OPEN
    reflect_est = 1
    reflect_offset = 0
    ereff_est = 5.45-0.0001j

    # reference mTRL calibration
    cal = rf.TUGMultilineTRL(line_meas=lines, line_lengths=line_lengths, er_est=ereff_est,
                             reflect_meas=reflect, reflect_est=reflect_est, reflect_offset=reflect_offset)
    # noise uncertainties
    sigma = 0.1
    L_cov = np.array([np.eye(8) for _ in range(len(f))])*sigma**2
    uSlines   = np.array([L_cov]*len(line_lengths)) # measured lines
    uSreflect = L_cov  # measured reflect 
    
    # length uncertainties
    l_std = 40e-6  # for the line
    ulengths  = l_std**2  
    l_open_std = 40e-6 # uncertainty in length used for the reflect
    
    # cross-section uncertainties
    w_std   = 2.55e-6
    s_std   = 2.55e-6
    wg_std  = 2.55e-6
    t_std   = 0.49e-6
    Dk_std  = 0.2
    Df_std  = 0
    sig_std = sig*0.1

    # original lengths
    line_lengths_orig = line_lengths.copy()
    print("Commercial lengths (mm):", line_lengths_orig*1e3)
    
    line_lengths_m101 = np.array([0, 0.32, 0.80, 2.02, 5.09])*1e-3
    print("Microwave101 lengths (mm):", line_lengths_m101*1e3)

    # new optimized lengths
    fmin  = 1e9
    fmax  = 150e9
    freq  = [fmin, fmax]
    ereff = 5.1
    phi   = 30
    # constrain the solution
    lmax = line_lengths_orig[-1]
    length_std = 40e-6   # expected standard deviation in lengths (for the optimizer method)
    lmin = 50e-6         # minimum length spacing
    force_integer_multiple = True  # quantize the lengths to multiple of lmin > 0
    N = 6                # force the number of lines (for optimizer solution)
    polish = True        # run an additional local minimizer after the global one
    opt_max_iter = 1000  # max iteration in the global optimization

    # Instantiate the calculator. 
    calc = LineLengthCalculator(freq, ereff, phi, lmax=lmax, lmin=lmin,
                                length_std=length_std, N=N,
                                force_integer_multiple=force_integer_multiple, 
                                polish=polish, opt_max_iter=opt_max_iter)
    # Calculate lengths
    # the optimizer doesn't always give same answer back because of randomness,
    # so don't expect exact same answer, but should be still 'optimal' in defined frequency range
    line_lengths_opt = calc.calc_lengths_optimize()
    print("Optimized lengths (mm):", line_lengths_opt*1e3)
    #line_lengths_opt = np.array([0., 0.4, 0.8, 2.5, 3.75, 5.05])*1e-3

    ## plot normalized eigenvalue vs frequency
    kap  = calc.kappa(f, line_lengths_orig, ereff=ereff)    
    kap2 = calc.kappa(f, line_lengths_opt, ereff=ereff)
    kap3 = calc.kappa(f, line_lengths_m101, ereff=ereff)    
    with PlotSettings(14):
        # Define highlight regions
        highlight_regions = [(35, 50), (70, 90)]
        highlight_color = 'red'
        highlight_alpha = 0.1
        fig = plt.figure(figsize=(10,4))
        fig.set_dpi(600)
        for start, end in highlight_regions:
            plt.axvspan(start, end, color=highlight_color, alpha=highlight_alpha)
        plt.plot(f/1e9, kap, label="Commercial ISS", lw=2, marker='>', markevery=25, markersize=10)
        plt.plot(f/1e9, kap2, label="Optimized", lw=2, marker='<', markevery=25, markersize=10) 
        plt.plot(f/1e9, kap3, label="Microwave101", lw=2, marker='^', markevery=25, markersize=10)
        plt.xlabel('Frequency (GHz)')
        plt.ylabel('Normalized eigenvalue')
        plt.ylim(0, 2)
        plt.xlim(0, f[-1]/1e9)
        #plt.grid(True)
        plt.legend()
    
    
    # Run MC for commercial, optimized, and microwave101 lengths
    # Monte Carlo simulation 
    M = 10 # number of MC runs
    cpw_MC = copy.deepcopy(cpw)    
    cal_MC_orig = []
    cal_MC_opt = []
    cal_MC_m101 = []
    for m in range(M):
        print(f'MC index {m+1} out of {M}')
        lines_model_MC_orig = []
        lines_model_MC_opt = []
        lines_model_MC_m101 = []
        
        # Generate perturbed CPW parameters once per iteration
        cpw_MC.w  = w + np.random.randn()*w_std
        cpw_MC.s  = s + np.random.randn()*s_std
        cpw_MC.wg = wg + np.random.randn()*wg_std
        cpw_MC.t  = t + np.random.randn()*t_std
        cpw_MC.er = (Dk + np.random.randn()*Dk_std)*(1-1j*(Df + np.random.randn()*Df_std))
        cpw_MC.sigma = sig + np.random.randn()*sig_std
        cpw_MC.update()
        
        # Process all sets of lengths
        for i, (l_orig, l_opt, cov) in enumerate(zip(line_lengths_orig, line_lengths_opt, uSlines)):
            l_noise = np.random.randn()*l_std
            # Original lengths
            length_orig = l_orig + l_noise
            embbed_line_orig = cal.embed(TL(length_orig, cpw_MC, cpw.Z0))
            lines_model_MC_orig.append(add_white_noise(embbed_line_orig, cov))
            
            # Optimized lengths
            length_opt = l_opt + l_noise
            embbed_line_opt = cal.embed(TL(length_opt, cpw_MC, cpw.Z0))
            lines_model_MC_opt.append(add_white_noise(embbed_line_opt, cov))

            # Microwave101 lengths (skip 6th line since m101 only has 5 lines)
            if i < len(line_lengths_m101):
                length_m101 = line_lengths_m101[i] + l_noise
                embbed_line_m101 = cal.embed(TL(length_m101, cpw_MC, cpw.Z0))
                lines_model_MC_m101.append(add_white_noise(embbed_line_m101, cov))
        
        # Process reflect for all cases
        open_model_MC = offset_open(0 + np.random.randn()*l_open_std, cpw_MC, l2 = 0 + np.random.randn()*l_open_std)
        reflect_model_MC = add_white_noise(cal.embed(open_model_MC), uSreflect)
        
        # Calculate calibration for all sets
        cal_MC_orig.append(rf.TUGMultilineTRL(line_meas=lines_model_MC_orig, line_lengths=line_lengths_orig, 
                          reflect_meas=reflect_model_MC, reflect_est=reflect_est, 
                          reflect_offset=reflect_offset, er_est=ereff_est))
        
        cal_MC_opt.append(rf.TUGMultilineTRL(line_meas=lines_model_MC_opt, line_lengths=line_lengths_opt, 
                         reflect_meas=reflect_model_MC, reflect_est=reflect_est, 
                         reflect_offset=reflect_offset, er_est=ereff_est))

        cal_MC_m101.append(rf.TUGMultilineTRL(line_meas=lines_model_MC_m101, line_lengths=line_lengths_m101, 
                         reflect_meas=reflect_model_MC, reflect_est=reflect_est, 
                         reflect_offset=reflect_offset, er_est=ereff_est))
    
    coefs_ideal = cal.coefs
    coefs_orig = [x.coefs for x in cal_MC_orig]
    coefs_opt = [x.coefs for x in cal_MC_opt]
    coefs_m101 = [x.coefs for x in cal_MC_m101]
        
    a21_a11_orig = get_norm_coef_MAE(coefs_orig, coefs_ideal, 'a21_a11')
    a12_orig = get_norm_coef_MAE(coefs_orig, coefs_ideal, 'a12')
    b12_b11_orig = get_norm_coef_MAE(coefs_orig, coefs_ideal, 'b12_b11')
    b21_orig = get_norm_coef_MAE(coefs_orig, coefs_ideal, 'b21')

    a21_a11_opt = get_norm_coef_MAE(coefs_opt, coefs_ideal, 'a21_a11')
    a12_opt = get_norm_coef_MAE(coefs_opt, coefs_ideal, 'a12')
    b12_b11_opt = get_norm_coef_MAE(coefs_opt, coefs_ideal, 'b12_b11')
    b21_opt = get_norm_coef_MAE(coefs_opt, coefs_ideal, 'b21')

    a21_a11_m101 = get_norm_coef_MAE(coefs_m101, coefs_ideal, 'a21_a11')
    a12_m101 = get_norm_coef_MAE(coefs_m101, coefs_ideal, 'a12')
    b12_b11_m101 = get_norm_coef_MAE(coefs_m101, coefs_ideal, 'b12_b11')
    b21_m101 = get_norm_coef_MAE(coefs_m101, coefs_ideal, 'b21')

    with PlotSettings(14):
        fig, axs = plt.subplots(2,2, figsize=(10,6))
        fig.set_dpi(600)
        fig.tight_layout(h_pad=2, w_pad=2)
        
        # Define highlight regions
        highlight_regions = [(35, 50), (70, 90)]
        highlight_color = 'red'
        highlight_alpha = 0.1

        ax = axs[0,0]
        for start, end in highlight_regions:
            ax.axvspan(start, end, color=highlight_color, alpha=highlight_alpha)
        ax.plot(f*1e-9, mag2db(a21_a11_orig), lw=2, label='Commercial ISS', marker='>', markevery=25, markersize=10)
        ax.plot(f*1e-9, mag2db(a21_a11_opt), lw=2, label='Optimized', marker='<', markevery=25, markersize=10)
        ax.plot(f*1e-9, mag2db(a21_a11_m101), lw=2, label='Microwave101', marker='^', markevery=25, markersize=10)
        ax.set_xlabel('Frequency (GHz)')
        ax.set_ylabel('MAE(a21/a11) (dB)')
        ax.set_xlim(0,150)
        ax.set_ylim(-30,-10)
        ax.set_yticks(np.arange(-30,-9,5))
        ax.set_xticks(np.arange(0,151,30))

        ax = axs[0,1]
        for start, end in highlight_regions:
            ax.axvspan(start, end, color=highlight_color, alpha=highlight_alpha)
        ax.plot(f*1e-9, mag2db(b12_b11_orig), lw=2, label='Commercial ISS', marker='>', markevery=25, markersize=10)
        ax.plot(f*1e-9, mag2db(b12_b11_opt), lw=2, label='Optimized', marker='<', markevery=25, markersize=10)
        ax.plot(f*1e-9, mag2db(b12_b11_m101), lw=2, label='Microwave101', marker='^', markevery=25, markersize=10)
        ax.set_xlabel('Frequency (GHz)')
        ax.set_ylabel('MAE(b12/b11) (dB)')
        ax.set_xlim(0,150)
        ax.set_ylim(-30,-10)
        ax.set_yticks(np.arange(-30,-9,5))
        ax.set_xticks(np.arange(0,151,30))
        
        ax = axs[1,0]
        for start, end in highlight_regions:
            ax.axvspan(start, end, color=highlight_color, alpha=highlight_alpha)
        ax.plot(f*1e-9, mag2db(a12_orig), lw=2, label='Commercial ISS', marker='>', markevery=25, markersize=10)
        ax.plot(f*1e-9, mag2db(a12_opt), lw=2, label='Optimized', marker='<', markevery=25, markersize=10)
        ax.plot(f*1e-9, mag2db(a12_m101), lw=2, label='Microwave101', marker='^', markevery=25, markersize=10)
        ax.set_xlabel('Frequency (GHz)')
        ax.set_ylabel('MAE(a12) (dB)')
        ax.set_xlim(0,150)
        ax.set_ylim(-30,-10)
        ax.set_yticks(np.arange(-30,-9,5))
        ax.set_xticks(np.arange(0,151,30))
        
        ax = axs[1,1]
        for start, end in highlight_regions:
            ax.axvspan(start, end, color=highlight_color, alpha=highlight_alpha)
        ax.plot(f*1e-9, mag2db(b21_orig), lw=2, label=f'Commercial ISS: [{", ".join([f"{l*1e3:.2f}" for l in line_lengths_orig])}]mm', marker='>', markevery=25, markersize=10)
        ax.plot(f*1e-9, mag2db(b21_opt), lw=2, label=f'Optimized: [{", ".join([f"{l*1e3:.2f}" for l in line_lengths_opt])}]mm', marker='<', markevery=25, markersize=10)
        ax.plot(f*1e-9, mag2db(b21_m101), lw=2, label=f'Microwave101: [{", ".join([f"{l*1e3:.2f}" for l in line_lengths_m101])}]mm', marker='^', markevery=25, markersize=10)
        ax.set_xlabel('Frequency (GHz)')
        ax.set_ylabel('MAE(b21) (dB)')
        ax.set_xlim(0,150)
        ax.set_ylim(-30,-10)
        ax.set_yticks(np.arange(-30,-9,5))
        ax.set_xticks(np.arange(0,151,30))

        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, bbox_to_anchor=(0.5, 0.98), 
                   loc='lower center', ncol=1, borderaxespad=0)
        plt.suptitle("Mean Absolute Error (MAE) of normalized error terms", 
                     verticalalignment='bottom', fontsize=18).set_y(1.15)
    
    plt.show()
# EOF