import sys
import numpy as np
import matplotlib.pyplot as pl

import astropy.constants as constants
import fsps

from sfhutils import weights_1DLinear, load_angst_sfh

lsun = constants.L_sun.cgs.value
pc = constants.pc.cgs.value
lightspeed = 2.998e18 #AA/s
#value to go from L_sun/AA to erg/s/cm^2/AA at 10pc
to_cgs = lsun/(4.0 * np.pi * (pc*10)**2 )


def gauss(x, mu, A, sigma):
    mu, A, sigma = np.atleast_2d(mu), np.atleast_2d(A), np.atleast_2d(sigma)
    val = A/(sigma * np.sqrt(np.pi * 2)) * np.exp(-(x[:,None] - mu)**2/(2 * sigma**2))
    return val.sum(axis = -1)


def convert_burst_pars(fwhm_burst = 0.05, f_burst = 0.5, contrast = 5,
                       bin_width = 1.0, bin_sfr = 1e9):
    
    #print(bin_width, bin_sfr)
    width, mstar = bin_width, bin_width * bin_sfr
    if width < fwhm_burst * 2:
        f_burst = 0.0 #no bursts if bin is short - they are resolved
    #constant SF component
    a = mstar * (1 - f_burst) /width
    #determine burst_parameters
    sigma = fwhm_burst / 2.355
    maxsfr = contrast * a
    A = maxsfr * (sigma * np.sqrt(np.pi * 2))
    if A > 0:
        nburst = np.round(mstar * f_burst / A)
    else:
        nburst = 0
    #recalculate a to preserve total mass formed in the face of burst number stochasticity
    a = (mstar - A * nburst) / width
    tburst = np.random.uniform(0,width, nburst)
    #print(a, nburst, A, sigma)
    return [a, tburst, A, sigma]

def burst_sfh(fwhm_burst = 0.05, f_burst = 0.5, contrast = 5, sfh = None):
    
    a, tburst, A, sigma, f_burst_actual = [],[],[],[],[]
    for i,abin in enumerate(sfh):
     #   print('------\nbin #{0}'.format(i))
        res = convert_burst_pars(fwhm_burst = fwhm_burst, f_burst = f_burst, contrast = contrast,
                             bin_width = (abin['t2']-abin['t1']), bin_sfr = abin['sfr'])
        a += [res[0]]
        if len(res[1]) > 0:
            tburst += (res[1] + abin['t1']).tolist()
            A += len(res[1]) * [res[2]]
            sigma += len(res[1]) * [res[3]]
        #f_burst_actual += [res[4]]
    if len(sigma) == 0:
        dt = (sfh['t2'] - sfh['t1']).min()/10.
    else:
        dt = np.min(sigma)/5. #make sure you sample the bursts reasonably well
    times = np.arange(np.round(sfh['t2'].max()/dt)) * dt
    #print(dt, np.round(sfh['t2'].max()/dt))
    #sys.exit()
    #figure out which bin each time is in
    bins = [sfh[0]['t1']] + sfh['t2'].tolist()
    bin_num = np.digitize(times, bins) -1
    #calculate SFR from all components
    sfr = np.array(a)[bin_num] + gauss(times, tburst, A, sigma)
    
    return times, sfr, f_burst_actual

def bursty_sps(lookback_time, lt, sfr, sps):
    """Obtain the spectrum of a stellar poluation with arbitrary complex
    SFH at a given lookback time.  The SFH is provided in terms of SFR vs
    t_lookback. Note that this in in contrast to the normal specification
    in terms of time since the big bang. Interpolation of the available
    SSPs to the time sequence of the SFH is accomplished by linear interpolation
    in log t.  Highly oscillatory SFHs require dense sampling of the temporal
    axis to obtain accurate results.

    :param lookback_time: float
        The lookback time(s) at which to obtain the spectrum.
    :param lt: ndarray, shape (ntime)
        The lookback time sequence of the provided SFH.  Assumed to have
        have equal linear time intervals, i.e. to be a regular grid in logt
    :param sfr: ndarray, shape (ntime)
        The SFR corresponding to each element of lt
    :param sps: StellarPopulation
        The fsps stellar population (with metallicty and IMF parameters set)
        to use for the SSP spectra

    :returns wave: ndarray, shape (nwave)
        The wavelength array
    :returns int_spec: ndarray, shape(nwave)
        The integrated spectrum at lookback_time
    """
    dt = lt[1]-lt[0]
    sps.params['sfh'] = 0 #set to SSPs
    wave, spec = sps.get_spectrum(peraa = True, tage = 0)
    ssp_ages = 10**sps.log_age #in yrs
    target_lt = np.atleast_1d(lookback_time)
    int_spec = np.zeros( [ len(target_lt), len(wave) ] )
    aw = np.zeros( [ len(target_lt), len(ssp_ages) ] )
    for i,tl in enumerate(target_lt):
        valid = lt >= tl
        inds, weights = weights_1DLinear(np.log(ssp_ages), np.log(lt[valid] - tl))
        #aggregate the weights for each index, after accounting for SFR
        agg_weights = np.bincount( inds.flatten(),
                                   weights = (weights * sfr[valid,None]).flatten(),
                                   minlength = len(ssp_ages) ) * dt
        int_spec[i,:] = (spec * agg_weights[:,None]).sum(axis = 0)
        aw[i,:] = agg_weights
    return wave, int_spec, aw



def selftest(filename = '/Users/bjohnson/Projects/angst/sfhs/angst_sfhs/gr8.lowres.ben.v1.sfh',
             lookback_time = [1e9, 10e9]):
    """ A quick test and demonstration of the algorithms.
    """
    sps = fsps.StellarPopulation()

    f_burst, fwhm_burst, contrast = 0.5, 0.05 * 1e9, 5
    sfh = load_angst_sfh(filename)
    sfh['t1'] = 10.**sfh['t1']
    sfh['t2'] = 10.**sfh['t2']
    sfh['sfr'][0] *=  1 - (sfh['t1'][0]/sfh['t2'][0])
    sfh[0]['t1'] = 0.
    mtot = ((sfh['t2'] - sfh['t1']) * sfh['sfr']).sum()
    lt, sfr, fb = burst_sfh(fwhm_burst = fwhm_burst, f_burst = f_burst, contrast = contrast, sfh = sfh)

    wave, spec, aw = bursty_sps(lookback_time, lt, sfr, sps)

    pl.figure()
    for i,t in enumerate(lookback_time):
        pl.plot(wave, spec[i,:], label = r'$t_{{lookback}} = ${0:5.1f} Gyr'.format(t/1e9))
    pl.legend()
    pl.xlim(1e3,1e4)
    pl.xlabel('wave')
    pl.ylabel(r'$F_\lambda$')

    fig, ax = pl.subplots(2,1)
    for i,t in enumerate(lookback_time):
        ax[1].plot(10**sps.log_age, aw[i,:], label = r'$t_{{lookback}} = ${0:5.1f} Gyr'.format(t/1e9), marker ='o', markersize = 2)
        print(aw[i,:].sum(), mtot, aw[i,:].sum()/mtot)
    ax[1].set_xlabel('SSP age - lookback time')
    ax[1].set_ylabel('Mass')
    ax[1].legend(loc = 'upper left')

    ax[0].plot(lt, sfr, 'k')
    ax[0].set_xlabel('lookback time')
    ax[0].set_ylabel('SFR')
    ax[0].set_title(r'f$_{{burst}}={0:3.1f}$, fwhm$_{{burst}}=${1:3.0f}Myr, contrast ={2}'.format(f_burst, fwhm_burst/1e6, contrast))
    for t in lookback_time:
        ax[0].axvline(x = t, color = 'r', linestyle =':', linewidth = 5)
    pl.show()
