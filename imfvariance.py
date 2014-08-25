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

    return [imfs, weights, spec, mass, sumspec]
        
def plot_varimf(sumspec, spec, imfs, df_imf_slopes = [2.7, 2.35])
    """
    Plot the integrated spectrum from the variable imf divided by the
    integrated spectrum from \delta-function imf slopes.

    :param sumspec:
        The integrated spectrum from the 
    :returns fig:
        matplotlib figure object containing a plot of  both the integrated
        spectrum for a delta function at imf0 and the integrated
        spectrum for an imf with delta function alpha=2.3

    :returns ax:
        The matplotlib axis object corresponding to the subplot of the above figure.
    """

    wave = sps.wavelengths
    fig, ax = pl.subplots(1,1)
    for alpha in df_imf_slopes:
        ind = np.argmin(np.abs(imfs - alpha))
        label = r'$L_\lambda(\alpha=vimf)/L_\lambda(\alpha={0:4.2f}) \, \times \, C_1$'.format(alpha)
        ax.plot(wave, sumspec/spec[:,ind], label = label)
    ax.set_xlim(1e2,1e4)
    ax.set_ylim(0.5, 2.5)
    ax.legend(loc=0)
    return fig, ax

    
if __name__ == '__main__':
    imf0, var = 2.7, 0.7
    blob = sps_varimf(imf0=imf0, var=var)
    fig, ax = plot_varimf(blob[-1], blob[2], df_imf_slops=[imf0, 2.35])
    fig.show()
