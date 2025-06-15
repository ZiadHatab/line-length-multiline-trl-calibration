"""
@author: Ziad (zi.hatab@gmail.com, https://github.com/ZiadHatab)

Example showing basic usage of the line length calculator.
If you want a quick answer, just use the Golomb method, which can support up to maximum 28 lines.
"""

import numpy as np
import matplotlib.pyplot as plt

# my code
from lengthcalc import LineLengthCalculator

if __name__ == "__main__":
    # specify frequency limits in Hz.
    # Remember length is inverse proportional to frequency,
    # i.e., going from 1GHz to 100MHz lead to x10 increase in length 
    # going further down to 10MHz lead to x100 increase from 1GHz.
    # Generally, fmin sets the max length and fmax sets the minimum length.
    fmin = 1e9
    fmax = 150e9
    freq = [fmin, fmax]

    # relative effective dielectric constant (can be complex value)
    ereff = 5.2

    # sets the phase margin (degrees) for the low frequency limit
    phi  = 30
    
    # Instantiate the calculator. 
    # freq, ereff, phi are the minimum requirement to run the code.
    calc = LineLengthCalculator(freq, ereff, phi)

    # Calculate lengths using spares rulers. 
    # This is quick. The Golomb will always give lower or equal line counts than Wichmann
    # https://en.wikipedia.org/wiki/Sparse_ruler
    # https://en.wikipedia.org/wiki/Golomb_ruler
    lengths_wichmann = calc.calc_length_wichmann()
    lengths_golomb   = calc.calc_length_golomb()

    # Calculate lengths via global optimization
    # because the seed for the optimizer is random and there are too many local minimum, 
    # you might not always get same answer if you searching for a lot of lines, 
    # but generally you will be very close to the answer.
    lengths_optimized = calc.calc_lengths_optimize()
    
    print('==== SOLUTION WITH MINIMUM PARAMETERS ====')
    print("Optimized lengths (mm):", lengths_optimized*1e3)
    print("Wichmann lengths (mm):", lengths_wichmann*1e3)
    print("Golomb lengths (mm):", lengths_golomb*1e3)
    print('')

    # plot normalized eigenvalue vs frequency
    f    = np.linspace(fmin, fmax, 1001)
    kap  = calc.kappa(f, lengths_optimized, ereff=ereff)    
    kap2 = calc.kappa(f, lengths_wichmann, ereff=ereff)
    kap3 = calc.kappa(f, lengths_golomb, ereff=ereff)
    
    plt.figure()
    plt.plot(f/1e9, kap, label="Optimized", lw=2)
    plt.plot(f/1e9, kap2, label="Wichmann", lw=2)
    plt.plot(f/1e9, kap3, label="Golomb", lw=2)
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('Normalized eigenvalue')
    plt.title('Minimum parameters provided')
    plt.ylim(0, 2)
    plt.xlim(0, f[-1]/1e9)
    plt.grid(True)
    plt.legend()
    plt.show()

    ## Below is the same example with further constraints
    # required parameters
    fmin  = 1e9
    fmax  = 150e9
    freq  = [fmin, fmax]
    ereff = 5.2
    phi   = 30
    
    # optional parameters to further constrain the solution
    lmax = 10e-3         # bound max length (overwrite calculated value from fmin)
    length_std = 50e-6   # expected standard deviation in lengths (for the optimizer method)
    lmin = 200e-6        # minimum length spacing
    force_integer_multiple = True  # quantize the lengths to multiple of lmin > 0
    # N = 6              # force the number of lines (for optimizer solution)
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
    f    = np.linspace(fmin, fmax, 1001)
    kap  = calc.kappa(f, lengths_optimized_constrained, ereff=ereff)    
    kap2 = calc.kappa(f, lengths_wichmann_constrained, ereff=ereff)
    kap3 = calc.kappa(f, lengths_golomb_constrained, ereff=ereff)
    
    plt.figure()
    plt.plot(f/1e9, kap, label="Optimized", lw=2)
    plt.plot(f/1e9, kap2, label="Wichmann", lw=2)
    plt.plot(f/1e9, kap3, label="Golomb", lw=2)
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('Normalized eigenvalue')
    plt.title('Further constrained solution')
    plt.ylim(0, 2)
    plt.xlim(0, f[-1]/1e9)
    plt.grid(True)
    plt.legend()
    
    plt.show()
    
# EOF