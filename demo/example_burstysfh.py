import os, glob
import numpy as np
import matplotlib.pyplot as pl

import fsps
import scombine.sfhutils as utils
import scombine.bursty_sfh as bsp
from sedpy import attenuation, observate

sps = fsps.StellarPopulation(zcontinuous=1)
sps.params['sfh'] = 0
sps.params['logzsol'] = 0.0 # solar
sps.params['imf_type'] = 0

filternamelist = ['galex_FUV','wfc3_uvis_f275w']
filterlist = observate.load_filters(filternamelist)                  
maggies_to_cgs = 10**(-0.4*(2.406 + 5*np.log10([f.wave_effective for f in filterlist])))
dm =24.47

t_lookback = [0.0, 1e8]
tl = ['present','100 Myr ago']

files = glob.glob('./sfhs/*sfh')
objname = [os.path.basename(f).split('.')[0] for f in files]

pl.figure()
for i, filen in enumerate(files):
    
    av, dav = 0.1, 0.1
    #read the binned sfh, put in linear units
    sfh = utils.load_angst_sfh(filen)
    sfh['t1'] = 10.**sfh['t1']
    sfh['t2'] = 10.**sfh['t2']
    sfh['sfr'][0] *=  1 - (sfh['t1'][0]/sfh['t2'][0])
    sfh[0]['t1'] = 0.
    mtot = ((sfh['t2'] - sfh['t1']) * sfh['sfr']).sum()

    #convert into a high resolution sfh, with *no* intrabin sfr variations
    lt, sfr, fb = bsp.burst_sfh(fwhm_burst=0.05, f_burst=0., contrast=1.,
                                sfh=sfh, bin_res=20.)
    
    # Get the attenuated spectra
    #  and IR luminosities
    wave, spec, mass, lir = bsp.bursty_sps(lt, sfr, sps, lookback_time=t_lookback,
                                           av=av, dav=dav, nsplit=30)
    for j, jt in enumerate(t_lookback):
        pl.plot(wave, spec[j,:] * wave * bsp.to_cgs,
                label = '{0} @ {1}'.format(objname[i], tl[j]))
        
    # Project onto filters to get
    #   absolute magnitudes
    mags = observate.getSED(wave, spec * bsp.to_cgs, filterlist = filterlist)
    
    # Get the intrinsic spectrum and project onto filters
    wave, spec, mass, _ = bsp.bursty_sps(lt, sfr, sps, lookback_times=t_lookback,
                                         av=None, dav=None)
    mags_int = observate.getSED(wave, spec * bsp.to_cgs, filterlist = filterlist)


pl.xlim(1e3, 1e4)
pl.ylim(1e-3,1e1)
pl.yscale('log')
pl.xlabel(r'wavelength ($\AA$)')
pl.ylabel(r'$\lambda$L$_\lambda$ (erg/s/cm$^2$ @ 10pc)')
pl.legend(loc = 'lower right')
pl.savefig('demo_bsp.png')
pl.show()
