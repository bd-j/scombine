import numpy as np
import matplotlib.pyplot as pl
import fsps

sps = fsps.StellarPopulation()


def effective_imf(masses, imf0=2.7, var=0.7, dalpha=0.05):
    """
    Work out the effective (slope averaged) imf in the case of
    variable slope.
    """
    imfs = np.arange(0.1, 5.4, dalpha)
    weights = 1./(np.sqrt(var*2.*np.pi)) * np.exp( -(imfs - imf0)**2/(2*var))
    phi = masses[:,None] ** imfs[None, :]
    

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

    :returns fig:
        matplotlib figure object containing a plot of the integrated
        spectrum from the variable imf divided by both the integrated
        spectrum for a delta function at imf0 and the integrated
        spectrum for an imf with delta function alpha=2.3

    :returns ax:
        The matplotlib axis object corresponding to the subplot of the above figure.

    returns blob:
        A list of numpy objects containing data used to make the plot.  
        
    """
    sps.params['sfh'] = 1 #five parameter sfh
    sps.params['const'] = 1 #everything in the constant
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
    
    ind = np.argmin(np.abs(imfs-imf0))
    ind_salp = np.argmin(np.abs(imfs-2.3))
    
    fig, ax = pl.subplots(1,1)
    ax.plot(wave, sumspec/spec[:,ind], label = r'$L_\lambda(\alpha=vimf)/L_\lambda(\alpha=2.7) \, \times \, C_1$')
    ax.plot(wave, sumspec/spec[:,ind_salp], label = r'$L_{{\lambda}}(\alpha=vimf)/L_\lambda(\alpha=2.3) \, \times \, C_2$')
    ax.set_xlim(1e2,1e4)
    ax.set_ylim(0.5, 2.5)
    ax.legend(loc=0)
    return fig, ax, [imfs,  weights, spec, mass, sumspec]

    
if __name__ == '__main__':
    fig, ax, blob = sps.varimf()
    fig.show()
