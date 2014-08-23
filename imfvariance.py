import numpy as np
import matplotlib.pyplot as pl
import fsps

sps = fsps.StellarPopulation()

if __name__ == '__main__':

    imf0, var = 2.7, 0.7
    dalpha = 0.1
    sps.params['sfh'] = 1 #five parameter sfh
    sps.params['const'] = 1 #everything in the constant
    sps.params['imf_type'] = 2 # Kroupa 3-slope IMF

    imfs = np.arange(0.1, 5.4, dalpha)
    weights = 1./(np.sqrt(var*2.*np.pi)) * np.exp( -(imfs - imf0)**2/(2*var))
    
    spec = np.zeros([ len(sps.wavelengths), len(imfs) ])
    for i, imf in enumerate(imfs):
        sps.params['imf3'] = imf
        wave, ispec = sps.get_spectrum(peraa=True, tage=1)
        logmass = sps.stellar_mass
        spec[:,i] = ispec
    sumspec = (spec * weights[None, :]).sum(-1) * dalpha
    ind = np.argmin(np.abs(imfs-imf0))
    ind_salp = np.argmin(np.abs(imfs-2.3))
    
    fig, ax = pl.subplots(1,1)
    ax.plot(wave, sumspec/spec[:,ind], label = r'$L_\lambda(\alpha=vimf)/L_\lambda(\alpha=2.7) \, \times \, C_1$')
    ax.plot(wave, sumspec/spec[:,ind_salp], label = r'$L_{{\lambda}}(\alpha=vimf)/L_\lambda(\alpha=2.3) \, \times \, C_2$')
    ax.set_xlim(1e3,1e4)
    ax.set_ylim(0.5, 2.5)
    ax.legend(loc=0)
    fig.show()

    

