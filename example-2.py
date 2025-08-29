"""
@author: Ziad (zi.hatab@gmail.com, https://github.com/ZiadHatab)

Example demonstrating the calculation of lengths for dispersive, 
lossy and frequency-dependent transmission lines.
"""

import numpy as np
import matplotlib.pyplot as plt

# my code
from lengthcalc import LineLengthCalculator
from TL_models.rw import RW
from TL_models.ms import MS

if __name__ == "__main__":
    # example of WR-12 waveguide
    fmin  = 60e9  # in Hz
    fmax  = 90e9  # in Hz
    fpoints = 1001
    freq = np.linspace(fmin, fmax, fpoints)
    
    # cross-section to compute ereff (dispersive)
    w = 3.0988e-3
    h = 1.5494e-3
    sr = 0.28  # relative conductivity of Brass to copper
    rw = RW(w, h, freq, sr=sr)
    ereff = rw.ereff

    # minium phase margin
    # from https://eprintspublications.npl.co.uk/4346/1/TQE5.pdf
    phi = 57

    # Instantiate the calculator. With current settings, we will get standard TRL answer (2 lines) 
    calc = LineLengthCalculator(freq, ereff, phi)

    # Calculate lengths (all should overlap)
    lengths_wichmann = calc.calc_length_wichmann()
    lengths_golomb   = calc.calc_length_golomb()
    lengths_optimized = calc.calc_lengths_optimize()
    
    print('==== Standard TRL Solution ====')
    print("Optimized lengths (mm):", lengths_optimized*1e3)
    print("Wichmann lengths (mm):", lengths_wichmann*1e3)
    print("Golomb lengths (mm):", lengths_golomb*1e3)
    print('')
    
    # plot normalized eigenvalue vs frequency
    f    = freq
    kap  = calc.kappa(f, lengths_optimized, ereff=ereff)    
    kap2 = calc.kappa(f, lengths_wichmann, ereff=ereff)
    kap3 = calc.kappa(f, lengths_golomb, ereff=ereff)
    
    plt.figure()
    plt.plot(f/1e9, kap, label="Optimized", lw=2)
    plt.plot(f/1e9, kap2, label="Wichmann", lw=2)
    plt.plot(f/1e9, kap3, label="Golomb", lw=2)
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('Normalized eigenvalue')
    plt.title('Waveguide WR-12 example. Minimum parameters provided')
    plt.ylim(0, 2)
    plt.xlim(f[0]/1e9, f[-1]/1e9)
    plt.grid(True)
    plt.legend()
    plt.show()
    
    # Same waveguide WR-12 example, but now enforce using a maximum length
    # This forces using more than two lines
    fmin  = 60e9  # in Hz
    fmax  = 90e9  # in Hz
    fpoints = 1001
    freq = np.linspace(fmin, fmax, fpoints)
    
    # cross-section to compute ereff (dispersive)
    w = 3.0988e-3
    h = 1.5494e-3
    sr = 0.28  # relative conductivity of Brass to copper
    rw = RW(w, h, freq, sr=sr)
    ereff = rw.ereff

    # minium phase margin
    # from https://eprintspublications.npl.co.uk/4346/1/TQE5.pdf
    phi = 57
    
    # optional parameters to further constrain the solution
    lmax = 5e-3          # bound max length
    length_std = 20e-6   # expected standard deviation in lengths (for the optimizer method)
    lmin = 100e-6        # minimum length spacing
    force_integer_multiple = True  # quantize the lengths to multiple of lmin > 0
    polish = True        # run an additional local minimizer after the global one
    opt_max_iter = 1000  # max iteration in the global optimization

    # Instantiate the calculator. 
    calc = LineLengthCalculator(freq, ereff, phi, lmax=lmax, lmin=lmin,
                                length_std=length_std, 
                                force_integer_multiple=force_integer_multiple, 
                                polish=polish, opt_max_iter=opt_max_iter)
    # Calculate lengths
    lengths_wichmann_constrained  = calc.calc_length_wichmann()
    lengths_golomb_constrained    = calc.calc_length_golomb()
    lengths_optimized_constrained = calc.calc_lengths_optimize()

    print('==== SOLUTION WITH ADDITIONAL CONSTRAINTS ====')
    print("Optimized lengths (mm):", lengths_optimized_constrained*1e3)
    print("Wichmann lengths (mm):", lengths_wichmann_constrained*1e3)
    print("Golomb lengths (mm):", lengths_golomb_constrained*1e3)
    print('')
    
    # plot normalized eigenvalue vs frequency
    f    = freq
    kap  = calc.kappa(f, lengths_optimized_constrained, ereff=ereff)    
    kap2 = calc.kappa(f, lengths_wichmann_constrained, ereff=ereff)
    kap3 = calc.kappa(f, lengths_golomb_constrained, ereff=ereff)
    plt.figure()
    plt.plot(f/1e9, kap, label="Optimized", lw=2)
    plt.plot(f/1e9, kap2, label="Wichmann", lw=2)
    plt.plot(f/1e9, kap3, label="Golomb", lw=2)
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('Normalized eigenvalue')
    plt.title('Waveguide WR-12 example. Constrained maximum length.')
    plt.ylim(0, 2)
    plt.xlim(f[0]/1e9, f[-1]/1e9)
    plt.grid(True)
    plt.legend()
    plt.show()
    
    ## Lossy Microstrip line example from 
    # F. Schnieder and W. Heinrich, "Model of thin-film microstrip line for circuit design," doi: 10.1109/22.899967
    freq = np.logspace(0, np.log10(500), 512)*1e9
    w = 8e-6
    h = 1.7e-6
    wg = 88e-6
    t  = 0.8e-6
    sr = 2.5/5.8  # relative conductivity to copper
    er = 2.7
    tand = 0.001
    ms = MS(w, h, t, wg, freq, sr=sr, er=er, tand=tand)

    ereff = ms.ereff
    phi   = 30
    
    # optional parameters to further constrain the solution
    lmax = None          # bound max length (overwrite calculated value from fmin)
    length_std = 5e-6   # expected standard deviation in lengths (for the optimizer method)
    lmin = 100e-6        # minimum length spacing
    force_integer_multiple = True  # quantize the lengths to multiple of lmin > 0
    polish = True        # run an additional local minimizer after the global one
    opt_max_iter = 1000  # max iteration in the global optimization

    # Instantiate the calculator. 
    calc = LineLengthCalculator(freq, ereff, phi, lmax=lmax, lmin=lmin,
                                length_std=length_std, 
                                force_integer_multiple=force_integer_multiple, 
                                polish=polish, opt_max_iter=opt_max_iter)
    # Calculate lengths
    lengths_wichmann_ms  = calc.calc_length_wichmann()
    lengths_golomb_ms    = calc.calc_length_golomb()
    lengths_optimized_ms = calc.calc_lengths_optimize()

    print('==== MS SOLUTION WITH CONSTRAINTS ====')
    print("Optimized lengths (mm):", lengths_optimized_ms*1e3)
    print("Wichmann lengths (mm):", lengths_wichmann_ms*1e3)
    print("Golomb lengths (mm):", lengths_golomb_ms*1e3)
    print('')
    
    # plot normalized eigenvalue vs frequency
    f    = freq
    kap  = calc.kappa(f, lengths_optimized_ms, ereff=ereff)    
    kap2 = calc.kappa(f, lengths_wichmann_ms, ereff=ereff)
    kap3 = calc.kappa(f, lengths_golomb_ms, ereff=ereff)
    
    plt.figure()
    plt.semilogy(f/1e9, kap, label="Optimized", lw=2)
    plt.semilogy(f/1e9, kap2, label="Wichmann", lw=2)
    plt.semilogy(f/1e9, kap3, label="Golomb", lw=2)
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('Normalized eigenvalue')
    plt.title('Thin-film lossy MS example.')
    plt.ylim(0.1, 1000)
    plt.xlim(0, f[-1]/1e9)
    plt.grid(True)
    plt.legend()
    
    
    plt.show()
    # EOF