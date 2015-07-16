import numpy as np

def sexAmodel(davmax, ages=None, sfh=None):
    """Calculate the distribution of maximum Av as a function of age
    for the Dolphin 2002 differential attenuation model.

    :param avmax:
        The maximum A_v at t_lookback = 0

    :param ages:
        Ages at which to generate Av values.  Linear years
        
    :param sfh:
        sfh as numpy structured array, output of
        sfhutils.load_angst_sfh().  It should have a `t2` field,
        giving age in units of log10(years)

    :returns dav:
        A sequence of length the number of time bins or ages, giving
        dA_v max in each time bin
    """
    if sfh is not None:
        if ages is not None:
            print("Warning: SFH and ages both passed to dust.sexAmodel.  Using SFH")
        ages = 10**sfh['t2']
    dav = np.clip(((ages-4e7) *
                   (-davmax) / 0.6e8 + davmax), 0, davmax)
    return dav


def redden(wave, spec, av=None, dav=None, nsplit=9,
           dust_curve=None, wlo=1216., whi=2e4, **kwargs):
    """Redden the spectral components.  The attenuation of a given
    star is given by the model av + U(0,dav) where U is the uniform
    random function.  Extensive use of broadcasting.

    :param wave:  ndarray of shape (nwave)
        The wavelength vector.
    
    :param spec: ndarray of shape (nspec, nwave)
        The input spectra to redden. nspec is the number of spectra.
        
    :param av: scalar or ndarray of shape (nspec)
        The attenuation at V band, in magnitudes, that affects all
        stars equally.  Can be a scalar if its the same for all
        spectra or an ndarray to apply different reddenings to each
        spectrum.

    :param dav: scalar or ndarray of shape (nspec)
        The maximum differential attenuation, in V band magnitudes.
        Can be a scalar if it's the same for all spectra or an array
        to have a different value for each spectrum.  Stars are
        assumed to be affected by an random additional amount of
        attenuation uniformly distributed from 0 to dav.

    :param nsplit: (default 10.0)
        The number of pieces in which to split each spectrum when
        performing the integration over differntial attenuation.
        Higher nsplit ensures greater accuracy, especially for very
        large dav.  However, because of the broadcasting, large nsplit
        can result in memory constraint issues.

    :param dust_curve: function
        The function giving the attenuation curve normalized to the V
        band, \tau_lambda/\tau_v.  This function must accept a
        wavelength vector as its argument and return tau_lambda/tau_v
        at each wavelength.

    :returns spec: ndarray of shape (nwave, nspec)
        The attenuated spectra.

    :returns lir: ndarray of shape (nspec)
        The integrated difference between the unattenuated and
        attenuated spectra, for each spectrum. The integral is
        performed over the interval [wlo,whi].
    """
    if (av is None) and (dav is None):
        return spec, None
    if dust_curve is None:
        print('Warning:  no dust curve was given')
        return spec, None
    #only split if there's a nonzero dAv 
    nsplit = nsplit * np.any(dav > 0) + 1
    lisplit = spec/nsplit
    # Enable broadcasting if av and dav aren't vectors
    # and convert to an optical depth instead of an attenuation
    av = np.atleast_1d(av)/1.086
    dav = np.atleast_1d(dav)/1.086
    lisplit = np.atleast_2d(lisplit)
    #uniform distribution from Av to Av + dAv
    avdist = av[None, :] + dav[None,:] * ((np.arange(nsplit) + 0.5)/nsplit)[:,None]
    #apply it
    ee = (np.exp(-dust_curve(wave)[None,None,:] * avdist[:,:,None]))
    spec_red = (ee * lisplit[None,:,:]).sum(axis = 0)
    #get the integral of the attenuated light in the optical-
    # NIR region of the spectrum
    opt = (wave >= wlo) & (wave <= whi) 
    lir = np.trapz((spec - spec_red)[:,opt], wave[opt], axis = -1)
    return np.squeeze(spec_red), lir

#def redden_analytic(wave, spec, av = None, dav = None,
#                    dust_curve = None, wlo = 1216., whi = 2e4, **kwargs):
#    Incorrect.  
#    k = dust_curve(wave)
#    alambda = av / (np.log10(av+dav)) * ( 10**(-0.4 * k * (av+dav)) - 10**(-0.4 * k * av))
#    spec_red = spec * alambda
#    return spec_red, None
        
