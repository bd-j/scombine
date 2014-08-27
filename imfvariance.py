import numpy as np
import matplotlib.pyplot as pl
import fsps
from scipy.integrate import romberg as qromb

sps = fsps.StellarPopulation()


def kroupa(mass, imf0=0.3, imf1=1.3, imf2=2.3, imf3=2.3,
           imf_lower_limit=0.08, imf_upper_limit=100):
    """
    Implements a Kroupa (2001) IMF, normalized to 1 M_sun
    """
    mass = np.atleast_1d(mass)
    alphas = np.array([imf0, imf1, imf2, imf3])
    lims = [imf_lower_limit, 0.08, 0.5, 1.0, imf_upper_limit]
    #lims = [0.08, 0.5, 1.0]
    #alpha_ind = np.digitize(masses, lims)
    #phi = masses**(-alphas[alpha_ind])

    phi = np.zeros_like(mass)
    norm = 0.0
    for i, alpha in enumerate(alphas):
        this = (mass > lims[i]) & (mass < lims[i+1])
        if i == 0:
            n = 1.0
        else:
            n *= lims[i]**(alpha - alphas[i-1])
        phi[this] = mass[this]**(-alpha) * n
        ex = (2.0-alpha)
        norm += (lims[i+1]**ex - lims[i]**ex) * n/ex
    return phi/norm

def salpeter(mass, imf=2.35,
             imf_lower_limit=0.08, imf_upper_limit=100):

    """
    Implements Salpeter IMF (or any pure power-law) as a special case
    of the broken power-law kroupa IMF.
    """
    return kroupa(mass, imf0=imf, imf1=imf, imf2=imf, imf3=imf,
                  imf_lower_limit=imf_lower_limit, imf_upper_limit=imf_upper_limit)

def effective_imf(masses, imf0=2.7, var=0.7, dalpha=0.05, function=kroupa):
    """
    Work out the effective (slope averaged) imf in the case of
    variable upper slopes.
    """
    phi = np.zeros_like(masses)
    imfs = np.arange(0.1, 5.4, dalpha)
    weights = 1./(np.sqrt(var*2.*np.pi)) * np.exp( -(imfs - imf0)**2/(2*var)) * dalpha
    for w, imf in zip(weights, imfs):
        phi += w * function(masses, imf3=imf)
    return phi

def sps_varimf(imf0=2.7, var=0.7, dalpha=0.05, verbose=False):
    """
    Work out the spectrum resulting from a weighted sum of different
    upper imf slopes, where the weights are given by a Gaussian in
    \alpha.  All slopes \alpha are given for the form N(m)\Delta m
    \propto m^{\alpha}, for which salpeter is \alpha=2.35.

    :param imf0: (default 2.7)
        The mean of the imf slope distribution.
        
    :param var: (default 0.7)
        The variance (\sigma^2) of the Gaussian imf slope distribution
        
    :param dalpha: (default 0.05)
        The resolution in \alpha for the numeric integration.  smaller
        numbers are slower bu more accurate.

    returns blob:
        A list of numpy arrays containing data used to make the
        plot. The order is the total integrated spectrum, imf_slopes,
        weights, full spectral array, stellar mass (current, including
        remnants).
    """
    
    sps.params['sfh'] = 1 #five parameter sfh
    sps.params['const'] = 1 #everything in the constant SF component
    sps.params['imf_type'] = 2 # Kroupa 3-slope IMF

    imfs = np.arange(0.1, 5.4, dalpha)
    weights = 1./(np.sqrt(var*2.*np.pi)) * np.exp( -(imfs - imf0)**2/(2*var))
    
    spec = np.zeros([ len(sps.wavelengths), len(imfs) ])
    mass = np.zeros(len(imfs)) 
    for i, imf in enumerate(imfs):
        sps.params['imf3'] = imf
        wave, ispec = sps.get_spectrum(peraa=True, tage=1)
        mass[i] = sps.stellar_mass
        spec[:,i] = ispec
    norm = (weights*dalpha).sum()
    sumspec = (spec * (weights)[None, :]).sum(-1) * dalpha
    if verbose: print(norm)

    return [sumspec, imfs, weights, spec, mass]
        
def plot_varimf(sumspec, spec, imfs, df_imf_slopes = [2.7, 2.35]):
    """
    Plot the integrated spectrum from the variable imf divided by the
    integrated spectrum from \delta-function imf slopes.

    :param sumspec:
        The integrated spectrum for a variable imf, ndarray of shape
        (nwave,)
        
    :param spec:
        The spectrum array containing the spectra of 1 M_sun (formed
        stellar mass) with different delta function imfs.  ndarray of
        shape (nwave, nimf)
        
    :param imfs:
        The upper imf slopes corresponding to spec, ndarray of shape
        (nimf,)

    :param df_imf_slopes:
        The desired imfs for for which you want to plot spectral
        ratios.  Iterable.
        
    :returns fig:
        matplotlib figure object containing a plot of  both the integrated
        spectrum for a delta function at imf0 and the integrated
        spectrum for an imf with delta function alpha=2.3

    :returns ax:
        The matplotlib axis object corresponding to the subplot of the
        above figure.
    """

    wave = sps.wavelengths
    fig, ax = pl.subplots(1,1)
    for alpha in df_imf_slopes:
        ind = np.argmin(np.abs(imfs - alpha))
        label = r'$L_\lambda(\alpha=vimf)/L_\lambda(\alpha={0:4.2f}) \, \times \, C_1$'.format(imfs[ind])
        ax.plot(wave, sumspec/spec[:,ind], label = label)
    ax.set_xlim(1e2,1e4)
    ax.set_ylim(0.5, 2.5)
    ax.legend(loc=0)
    return fig, ax
    
if __name__ == '__main__':
    
    imf0, var = 2.7, 0.7
    #blob = sps_varimf(imf0=imf0, var=var)
    #fig, ax = plot_varimf(blob[0], blob[3], blob[1], df_imf_slopes=[imf0, 2.3])
    #fig.show()

    fig, ax = pl.subplots(1,1)
    masses = np.arange(0.08,100.,0.001)

    phik = kroupa(masses, imf3=2.3)
    ax.plot(masses, phik, label = r'Kroupa (2001)$\,$')

    phi_steep = kroupa(masses, imf3=2.7)
    ax.plot(masses, phi_steep, label = r'single IMF $\alpha_3=2.7$')
    
    phi_var = effective_imf(masses, imf0=imf0, var=var, function=kroupa)
    ax.plot(masses, phi_var, label = r'Variable IMF ($\mu={0}, \sigma^2={1}$)'.format(imf0,var))

    imf0 = 2.3
    phi_var = effective_imf(masses, imf0=imf0, var=var, function=kroupa)
    ax.plot(masses, phi_var, label = r'Variable IMF ($\mu={0}, \sigma^2={1}$)'.format(imf0,var))

    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('$m$')
    ax.set_ylabel(r'$N(m)dm$')
    ax.legend(loc=0)
    ax.set_title(r'$m_{{total}} = 1 M_\odot$, $0.08<m<100$')
    fig.show()
