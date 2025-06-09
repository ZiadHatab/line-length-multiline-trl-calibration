"""
@author: Ziad (zi.hatab@gmail.com, https://github.com/ZiadHatab)

A script to compute line lengths for multiline TRL calibration, specifically for the algorithm developed in [1,2]. 
Note, that the results you get here are strictly optimal for the algorithm in [1,2], and might not be optimal for other multiline TRL algorithms.

Features:
- Given maximum length and maximum frequency, compute minimum required lines and their lengths.
- Given minimum frequency, compute maximum required line length.
- Constrained lengths to have minimum spacing.
- Constrained lengths solution to minimize sensitivity to length errors provided as standard deviation.
- predefined solutions using Sparse rulers.

[1] Z. Hatab, M. Gadringer and W. Bösch, "Improving The Reliability of The Multiline TRL Calibration Algorithm," 
2022 98th ARFTG Microwave Measurement Conference (ARFTG), 2022, pp. 1-5, doi: 10.1109/ARFTG52954.2022.9844064.
[2] Z. Hatab, M. E. Gadringer, and W. Bösch, "Propagation of Linear Uncertainties through Multiline Thru-Reflect-Line Calibration," 
in IEEE Transactions on Instrumentation and Measurement, vol. 72, pp. 1-9, 2023, doi: 10.1109/TIM.2023.3296123.
"""

import numpy as np
import scipy

class LineLengthCalculator:
    """
    A calculator for determining optimal line lengths for multiline TRL calibration.
    Provides methods for optimized, Wichmann, and Golomb ruler-based line length calculations.
    Parameters
    ----------
    freq : array-like
        List or array of frequency points (Hz) [start, stop] or array.
    ereff : float or array-like
        Effective relative permittivity (scalar or array matching freq).
    phi : float, optional
        Minimum phase difference between lines in degrees (default: 20).
    lmax : float, optional
        Maximum allowed line length (meters).
    lmin : float, optional
        Minimum allowed line length or spacing (meters, default: 0).
    length_std : float, optional
        Standard deviation of line length error (meters, default: 0).
    N : int, optional
        Number of lines (if None, computed automatically).
    obj_type : str, optional
        Objective type for optimization: 'minmax' or 'ls' (default: 'minmax').
    fit_max_iter : int, optional
        Maximum number of optimization iterations (default: 1000).
    force_integer_multiple : bool, optional
        If True, restricts lengths to integer multiples of lmin (default: False).
    polish : bool, optional
        If True, performs local search after global optimization (default: False).
    """
    def __init__(self, freq, ereff, phi=20, lmax=None, lmin=0, length_std=0, N=None, obj_type='minmax',
                 fit_max_iter=1000, force_integer_multiple=False, polish=False):
        if force_integer_multiple and lmin == 0:
            raise ValueError("lmin must be nonzero when force_integer_multiple is True")
        self.c0 = 299792458
        self.freq = np.atleast_1d(freq)
        self.ereff = np.atleast_1d(ereff)
        self.phi = phi
        self.lmax = lmax
        self.lmin = lmin
        self.length_std = length_std
        self.N = N
        self.obj_type = obj_type
        self.fit_max_iter = fit_max_iter
        self.force_integer_multiple = force_integer_multiple
        self.polish = polish
        self.optimization_result = None
        self.lengths_opt = None
        self.lengths_wichmann = None
        self.lengths_golomb = None
        # Internal parameters
        self.fmin_opt = None
        self.fmax_opt = None
        self.nmax = None
        self.r = None
        self.s = None
        self.golomb_N = None

    @staticmethod
    def wichmann_ruler(r: int, s: int) -> list:
        """
        Generate a Wichmann ruler sequence. https://en.wikipedia.org/wiki/Sparse_ruler

        Args:
            r (int): Number of repeated segments in the Wichmann construction.
            s (int): Additional segment parameter in the Wichmann construction.

        Returns:
            list: List of ruler marks (positions) for the Wichmann sparse ruler.
        """
        segments = []
        segments.extend([1]*r)
        segments.append(r + 1)
        segments.extend([2*r + 1]*r)
        segments.extend([4*r + 3]*s)
        segments.extend([2*r + 2]*(r + 1))
        segments.extend([1]*r)
        marks = [0]
        current = 0
        for seg in segments:
            current += seg
            marks.append(current)
        return marks

    @staticmethod
    def kappa(f, lengths, ereff=1-0j, apply_norm=True):
        """
        Calculate the kappa metric for a set of line lengths and frequencies.

        Args:
            f (array-like): Frequencies at which to evaluate kappa (Hz).
            lengths (array-like): Line lengths (meters).
            ereff (float or complex, optional): Effective relative permittivity. Defaults to 1-0j.
            apply_norm (bool, optional): If True, normalize kappa by the sum of |W|. Defaults to True.

        Returns:
            np.ndarray: Kappa values for each frequency.
        """
        f = np.atleast_1d(f)
        lengths = np.atleast_1d(lengths)
        c0 = 299792458
        gamma = 2*np.pi*f/c0*np.sqrt(-ereff*(1+0j))
        kap = []
        for g in gamma:
            y = np.exp(g*lengths)
            z = 1/y
            W  = (np.outer(y,z) - np.outer(z,y)).conj()
            lam = abs(W.conj()*W).sum()/2
            norm = abs(W).sum()/2 if apply_norm else 1
            kap.append(lam/norm)
        return np.array(kap)

    @staticmethod
    def kappa_jac(f, lengths, ereff=1-0j):
        """
        Compute the Jacobian (derivative) of the kappa metric with respect to line lengths.

        Args:
            f (array-like): Frequencies at which to evaluate the Jacobian (Hz).
            lengths (array-like): Line lengths (meters).
            ereff (float or complex, optional): Effective relative permittivity. Defaults to 1-0j.

        Returns:
            np.ndarray: Jacobian of kappa with respect to line lengths for each frequency.
        """
        f = np.atleast_1d(f)
        lengths = np.atleast_1d(lengths)
        c0 = 299792458
        gamma = 2*np.pi*f/c0*np.sqrt(-ereff*(1+0j))
        kap_jac = []
        for g in gamma:
            y = np.exp(g*lengths)
            z = 1/y
            Wn = (np.outer(y,z) - np.outer(z,y)).conj()
            Wn_sym = Wn.copy()
            i_lower = np.tril_indices_from(Wn, k=-1)
            Wn_sym[i_lower] = Wn.T[i_lower]
            Wp = (np.outer(y,z) + np.outer(z,y)).conj()
            fval = abs(Wn.conj()*Wn).sum()/2
            f_prime = 2*g*(Wn_sym*Wp).sum(axis=0)
            gval = abs(Wn).sum()/2
            Wn_divid_with = Wn.copy()
            np.fill_diagonal(Wn_divid_with, 1)
            g_prime = gval*(Wn_sym*Wp/abs(Wn_divid_with)).sum(axis=0)
            kap_jac.append((gval*f_prime - fval*g_prime)/gval**2)
        return np.array(kap_jac)

    @staticmethod
    def obj(x, *args):
        """
        Objective function for optimizing line lengths.

        Args:
            x (np.ndarray): Array of line lengths to be optimized.
            *args: Tuple containing:
                - f (np.ndarray): Array of frequencies.
                - ereff (float or np.ndarray): Effective relative permittivity.
                - length_std (float): Standard deviation of length error.
                - obj_type (str): Objective type, either 'minmax' or 'ls'.
                - force_integer_multiple (bool): If True, restricts lengths to integer multiples of lmin.
                - lmin (float): Minimum length increment.

        Returns:
            float: Value of the objective function based on the selected obj_type.
        """
        f, ereff, length_std, obj_type, force_integer_multiple, lmin = args
        if force_integer_multiple:
            lengths = np.ceil(x/lmin)*lmin
        else:
            lengths = x
        kap = LineLengthCalculator.kappa(f, lengths, ereff=ereff, apply_norm=True)
        J = LineLengthCalculator.kappa_jac(f, lengths, ereff=ereff)
        JJT = np.array([x.dot(x.conj()).real for x in J])
        if obj_type == 'minmax':
            return (-kap).max() + (length_std**2*JJT).mean()
        elif obj_type == 'ls':
            return (-kap**2).mean() + (length_std**2*JJT).mean()
        else:
            raise ValueError("obj_type must be 'minmax' or 'ls'")

    def calc_lengths_optimize(self):
        """
        Calculates the optimized line lengths using differential evolution.

        Returns:
            np.ndarray: Optimized line lengths (in meters).
        """
        freq = self.freq
        ereff = self.ereff
        phi = self.phi
        lmax = self.lmax
        lmin = self.lmin
        length_std = self.length_std
        N = self.N
        obj_type = self.obj_type
        fit_max_iter = self.fit_max_iter
        force_integer_multiple = self.force_integer_multiple
        polish = self.polish

        fmin, fmax = freq[0], freq[-1]
        if lmax is None:
            n = 0
            lmax = self.c0/2/fmin/np.sqrt(ereff[0].real)*(n+phi/180)
            self.lmax = lmax
        nmax = np.ceil(lmax*2*fmax*np.sqrt(ereff[-1].real)/self.c0 - 1 + phi/180).astype(int) + 1
        if N is None:
            N = np.ceil( 1/2 + np.sqrt(1 + 8*nmax)/2 ).astype(int)
            self.N = N
        fmin_opt = self.c0/2/np.sqrt(ereff[0].real)/lmax*(0+1/2)
        fmax_opt = self.c0/2/np.sqrt(ereff[-1].real)/lmax*((nmax-1)+1/2)
        f = np.linspace(fmin_opt, fmax_opt, nmax*10)
        if np.size(ereff) == 1:
            ereff_interp = ereff*np.ones_like(f)
        else:
            if len(ereff) != len(freq):
                raise ValueError("ereff must be a scalar or have the same length as freq")
            ereff_interp = np.interp(f, freq, ereff, left=ereff[0], right=ereff[-1])
        linear_constraint = scipy.optimize.LinearConstraint(np.eye(N+1,N, k=-1) - np.eye(N+1,N), 
                                             [0]+[-lmax]*(N-1)+[lmax], 
                                             [0]+[-lmin]*(N-1)+[lmax])
        bounds = [(0, 0)] + [(0, lmax)]*(N-2) + [(lmax, lmax)]
        if self.golomb_ruler(N) is not None:
            ruler = self.golomb_ruler(N)
            x0 = np.array(ruler)*lmax/ruler[-1]
        else:
            x0 = self.calc_length_chebyshev(N, lmax)
        save_sol = []
        save_iteration_results = lambda xk,convergence: save_sol.append(xk)
        xx = scipy.optimize.differential_evolution(
            self.obj, bounds, x0=x0, args=(f, ereff_interp, length_std, obj_type, force_integer_multiple, lmin),
            disp=True, polish=polish, maxiter=fit_max_iter, 
            constraints=(linear_constraint),
            strategy='randtobest1bin', init='sobol',
            popsize=10, mutation=(0.1,1.9), recombination=0.9, 
            tol=1e-6,
            updating='deferred', workers=-1, callback=save_iteration_results
        )
        if force_integer_multiple:
            lengths = np.ceil(xx.x/lmin)*lmin
        else:
            lengths = xx.x
        self.optimization_result = xx
        self.lengths_opt = lengths
        self.fmin_opt = fmin_opt
        self.fmax_opt = fmax_opt
        self.nmax = nmax
        return lengths

    def calc_length_wichmann(self):
        """
        Calculates the line lengths using the Wichmann sparse ruler construction.

        Returns:
            np.ndarray: Calculated line lengths (in meters) based on the Wichmann ruler.
        """
        fmin = self.freq[0]
        fmax = self.freq[-1]
        ereff = self.ereff[0] if np.size(self.ereff) == 1 else self.ereff
        phi = self.phi
        lmax = self.lmax
        force_integer_multiple = self.force_integer_multiple
        lmin = self.lmin
        if lmax is None:
            n = 0
            lmax = self.c0/2/fmin/np.sqrt(ereff.real)*(n+phi/180)
            self.lmax = lmax
        M = np.ceil(lmax*2*fmax*np.sqrt(ereff.real)/self.c0 - 1 + phi/180).astype(int) + 1
        N = np.ceil( 1/2 + np.sqrt(1 + 8*M)/2 ).astype(int)
        found = False
        r = s = None
        while not found:
            for r_try in range(0, N):
                s_try = N - 4*r_try - 3
                if s_try < 0:
                    continue
                M_test = 4*r_try*(r_try+s_try+2) + 3*(s_try+1)
                if M_test >= M:
                    found = True
                    r = r_try
                    s = s_try
                    M = M_test
                    break
            if not found:
                N += 1
        ruler = self.wichmann_ruler(r, s)
        lengths = np.array(ruler)*lmax/ruler[-1]
        if force_integer_multiple:
            lengths = np.ceil(lengths/lmin)*lmin
        self.lengths_wichmann = lengths
        return lengths

    def calc_length_golomb(self):
        """
        Calculates the line lengths using the Golomb ruler construction.

        Returns:
            np.ndarray: Calculated line lengths (in meters) based on the Golomb ruler.
        """
        fmin = self.freq[0]
        fmax = self.freq[-1]
        ereff = self.ereff[0] if np.size(self.ereff) == 1 else self.ereff
        phi = self.phi
        lmax = self.lmax
        force_integer_multiple = self.force_integer_multiple
        lmin = self.lmin
        if lmax is None:
            n = 0
            lmax = self.c0/2/fmin/np.sqrt(ereff.real)*(n+phi/180)
            self.lmax = lmax
        M = np.ceil(lmax*2*fmax*np.sqrt(ereff.real)/self.c0 - 1 + phi/180).astype(int) + 1
        N = np.ceil( 1/2 + np.sqrt(1 + 8*M)/2 ).astype(int)
        while True:
            if N > 28:
                N = 28
                print("Warning: Maximum defined Golomb ruler order is 28. Using order 28.")
            ruler = self.golomb_ruler(N)
            if ruler is None:
                raise ValueError(f"No Golomb ruler found for order {N}")
            if ruler[-1] >= M:
                break
            N += 1
        
        lengths = np.array(ruler)*lmax/ruler[-1]
        if force_integer_multiple:
            lengths = np.ceil(lengths/lmin)*lmin
        self.lengths_golomb = lengths
        return lengths

    @staticmethod
    def calc_length_chebyshev(N, lmax):
        """
        Calculate Chebyshev-spaced line lengths between 0 and lmax.
        https://en.wikipedia.org/wiki/Chebyshev_nodes

        Args:
            N (int): Number of line lengths.
            lmax (float): Maximum line length (meters).

        Returns:
            np.ndarray: Array of Chebyshev-spaced line lengths (meters).
        """
        k = np.arange(N)
        x = 0.5 * (1 - np.cos(np.pi*k/(N-1)))
        lengths = x*lmax
        return lengths

    @staticmethod
    def golomb_ruler(order):
        """
        Returns the Golomb ruler for a given order if available.
        https://en.wikipedia.org/wiki/Golomb_ruler

        Args:
            order (int): The order (number of marks) of the Golomb ruler.

        Returns:
            list or None: List of ruler marks (positions) for the given order, or None if not available.
        """
        golomb_rulers = [
            {"order": 1, "length": 0, "ruler": [0]},
            {"order": 2, "length": 1, "ruler": [0, 1]},
            {"order": 3, "length": 3, "ruler": [0, 1, 3]},
            {"order": 4, "length": 6, "ruler": [0, 1, 4, 6]},
            {"order": 5, "length": 11, "ruler": [0, 1, 4, 9, 11]},
            {"order": 6, "length": 17, "ruler": [0, 1, 4, 10, 12, 17]},
            {"order": 7, "length": 25, "ruler": [0, 1, 4, 10, 18, 23, 25]},
            {"order": 8, "length": 34, "ruler": [0, 1, 4, 9, 15, 22, 32, 34]},
            {"order": 9, "length": 44, "ruler": [0, 1, 5, 12, 25, 27, 35, 41, 44]},
            {"order": 10, "length": 55, "ruler": [0, 1, 6, 10, 23, 26, 34, 41, 53, 55]},
            {"order": 11, "length": 72, "ruler": [0, 1, 4, 13, 28, 33, 47, 54, 64, 70, 72]},
            {"order": 12, "length": 85, "ruler": [0, 2, 6, 24, 29, 40, 43, 55, 68, 75, 76, 85]},
            {"order": 13, "length": 106, "ruler": [0, 2, 5, 25, 37, 43, 59, 70, 85, 89, 98, 99, 106]},
            {"order": 14, "length": 127, "ruler": [0, 4, 6, 20, 35, 52, 59, 77, 78, 86, 89, 99, 122, 127]},
            {"order": 15, "length": 151, "ruler": [0, 4, 20, 30, 57, 59, 62, 76, 100, 111, 123, 136, 144, 145, 151]},
            {"order": 16, "length": 177, "ruler": [0, 1, 4, 11, 26, 32, 56, 68, 76, 115, 117, 134, 150, 163, 168, 177]},
            {"order": 17, "length": 199, "ruler": [0, 5, 7, 17, 52, 56, 67, 80, 81, 100, 122, 138, 159, 165, 168, 191, 199]},
            {"order": 18, "length": 216, "ruler": [0, 2, 10, 22, 53, 56, 82, 83, 89, 98, 130, 148, 153, 167, 188, 192, 205, 216]},
            {"order": 19, "length": 246, "ruler": [0, 1, 6, 25, 32, 72, 100, 108, 120, 130, 153, 169, 187, 190, 204, 231, 233, 242, 246]},
            {"order": 20, "length": 283, "ruler": [0, 1, 8, 11, 68, 77, 94, 116, 121, 156, 158, 179, 194, 208, 212, 228, 240, 253, 259, 283]},
            {"order": 21, "length": 333, "ruler": [0, 2, 24, 56, 77, 82, 83, 95, 129, 144, 179, 186, 195, 255, 265, 285, 293, 296, 310, 329, 333]},
            {"order": 22, "length": 356, "ruler": [0, 1, 9, 14, 43, 70, 106, 122, 124, 128, 159, 179, 204, 223, 253, 263, 270, 291, 330, 341, 353, 356]},
            {"order": 23, "length": 372, "ruler": [0, 3, 7, 17, 61, 66, 91, 99, 114, 159, 171, 199, 200, 226, 235, 246, 277, 316, 329, 348, 350, 366, 372]},
            {"order": 24, "length": 425, "ruler": [0, 9, 33, 37, 38, 97, 122, 129, 140, 142, 152, 191, 205, 208, 252, 278, 286, 326, 332, 353, 368, 384, 403, 425]},
            {"order": 25, "length": 480, "ruler": [0, 12, 29, 39, 72, 91, 146, 157, 160, 161, 166, 191, 207, 214, 258, 290, 316, 354, 372, 394, 396, 431, 459, 467, 480]},
            {"order": 26, "length": 492, "ruler": [0, 1, 33, 83, 104, 110, 124, 163, 185, 200, 203, 249, 251, 258, 314, 318, 343, 356, 386, 430, 440, 456, 464, 475, 487, 492]},
            {"order": 27, "length": 553, "ruler": [0, 3, 15, 41, 66, 95, 97, 106, 142, 152, 220, 221, 225, 242, 295, 330, 338, 354, 382, 388, 402, 415, 486, 504, 523, 546, 553]},
            {"order": 28, "length": 585, "ruler": [0, 3, 15, 41, 66, 95, 97, 106, 142, 152, 220, 221, 225, 242, 295, 330, 338, 354, 382, 388, 402, 415, 486, 504, 523, 546, 553, 585]},
        ]
        return next((ruler["ruler"] for ruler in golomb_rulers if ruler["order"] == order), None)

if __name__ == "__main__":
    fmin = 2e9
    fmax = 150e9
    freq = [fmin, fmax]
    ereff = 5.2
    phi = 30
    lmax = 5050e-6

    # Instantiate the calculator
    calc = LineLengthCalculator(freq, ereff, phi, length_std=50e-6, 
                                lmin=50e-6, force_integer_multiple=True, 
                                polish=True, fit_max_iter=1000, lmax=lmax)
    # Calculate optimized lengths
    lengths_optimzed = calc.calc_lengths_optimize()
    lengths_wichmann = calc.calc_length_wichmann()
    lengths_golomb = calc.calc_length_golomb()
    lengths_industry = np.array([0, 250, 700, 1600, 3300, 5050])*1e-6
    N = calc.N

    print("Optimized lengths (mm):", lengths_optimzed*1e3)
    print("Wichmann lengths (mm):", lengths_wichmann*1e3)
    print("Golomb lengths (mm):", lengths_golomb*1e3)
    print("Industy lengths (mm): ", lengths_industry*1e3)

    f = np.linspace(fmin, fmax, 1001)
    kap = calc.kappa(f, lengths_optimzed, ereff=ereff)    
    kap2 = calc.kappa(f, lengths_wichmann, ereff=ereff)
    kap3 = calc.kappa(f, lengths_golomb, ereff=ereff)
    kap4 = calc.kappa(f, lengths_industry, ereff=ereff)

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(f/1e9, kap, label="Optimized")
    plt.plot(f/1e9, kap2, label="Wichmann")
    plt.plot(f/1e9, kap3, label="Golomb")
    plt.plot(f/1e9, kap4, label="Industy")
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('Kappa')
    plt.title('Kappa (normalized eigenvalue) vs Frequency')
    plt.ylim(0, 2)
    plt.xlim(0, f[-1]/1e9)
    plt.grid(True)
    plt.legend()

    plt.show()

# EOF