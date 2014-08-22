import numpy as np
import fsps

sps = fsps.StellarPopulation()


sps.params['sfh'] = 1 #five parameter sfh
sps.params['const'] = 1 #everything in the constant
sps.params['imf_type'] = 2 # Kroupa 3-slope IMF


imfs = np.arange(0, 6, 0.1)
weights = 1./(np.sqrt(var*2.*np.pi)) * np.exp( -(imfs - imf0)**2/(2*var))

spec = np.zeros(len(sps.wavelengths), len(imfs))
for i, imf in enumerate(imfs):
    sps.params['imf3'] = imf
    wave, ispec = sps.get_spectrum(peraa=True, tage=0.1)
    logmass = sps.stellar_mass
    spec[:,i] = ispec


    

