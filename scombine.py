import os, glob
import subprocess
import numpy as np

try:
    import astropy.io.fits as pyfits
except (ImportError):
    import pyfits
import astropy.constants as constants

import observate, attenuation
import sfhutils as utils

lsun = constants.L_sun.cgs.value
pc = constants.pc.cgs.value
lightspeed = 2.998e18 #AA/s
#value to go from L_sun/AA to erg/s/cm^2/AA at 10pc
to_cgs = lsun/(4.0 * np.pi * (pc*10)**2 )
imfname = ['salp','chab','kroupa']

spsdir = os.getenv('SPS_HOME')
Zsun = 0.019
executable = spsdir + 'src/autosps.exe'
tabsfh = spsdir + 'data/sfh.dat'
fsps_outname = 'angstbin.out'

class Combiner(object):
    """Class for linearly combining spectra of different SF components,
    accounting for dust. Usage example (to obtain present day UV magnitudes
    for a given SFH assuming Salpeter IMF, solar metallicity, and the Calzetti
    attenuation curve):

    ::
        specfile = generate_basis(sfhfile, zmet = 1.0, imf_type = 0, outroot = 'L0_tl0', t_lookback = 0.0)
        sc = Combiner(specfile, dust_law = attenuate.calzetti)      #initialize the combiner
        fl = observate.load_filters(['galex_FUV', 'galex_NUV', 'wfc3_uvis_f275w'])  #list of filter objects
        davdist = sc.sexAmodel(1.0, sfhfile)     #age dependent attenuation up to Av = 1.0
        spec, mstar, phot = sc.combine(sfhfile, av = 0.1, dav = davdist, filterlist = fl)
        spec, mstar, phot = sc.combine(sfhfile2, av = 0.1, dav = davdist, filterlist = fl) #another SFH, with same binning and metallicity
        
    :param basis_file:
        string name (and path) of a fits file containing the flux (and mass
        and wavelngth) vectors for this set of time bins (and metallicity)
        
    :param dust_law:
        function that returns A_lambda/A_v given a wavelength vector (in AA).
        Defaults to CCM if attenuate module available
    """

    def __init__(self, basis_file, dust_law = attenuation.cardelli):
        #note:  flux units are erg/s/AA/cm^2 for d=10pc and SFR = 1
        #scale as necessary to get desired distance and SFRs
        ff = pyfits.open(basis_file)
        li0 = ff[0].data
        self.wave = li0[0,:] #pull out the wavelength vector 
        self.spec = li0[1:,:] #strip the wavelength vector from the fluxes
        self.mstar = li0[1:,-1] #pull out the mass vector
        self.nbin = self.mstar.shape[0]
        self.tau_curve = dust_law(self.wave)
        ff.close()

    def combine(self, sfhfile, av = 0, dav = 0, filterlist = None):
        """Generate a composite spectrum from an SFH and a spectral basis,
        including reddening.

        INPUTS
        ---------
        :param sfhfile:
            string giving the name (and path) to the file containing the sfh
            
        :param av:
            Scalar or sequence giving the 'foreground' reddening affecting all stars
            of a given time bin equally.  If a sequence, should be of same length as
            the number of sfh bins
            
        :param dav:
            scalar or sequence giving the maximum reddening of a linear distribution
            of reddening for a given time bin, from av to dav.  If a sequence, should
            be of same length as the number of sfh bins
            
        :param filterlist:
            sequence of 'sedpy' filter objects for which to return the broadband photometry
            
        :returns spectrum:
            The composite spectrum, reddened. Units are erg/s/cm^2/AA at a distance of 10pc
        :returns mstar:
            the current stellar mass, in M_sun
        :returns broadband:
            A numpy array of broadband AB absolute magnitudes
        """
        sfh = utils.load_angst_sfh(sfhfile)
        # Adjust most recent bin sfr to have same total mass for t = 0 to t2
        sfh['sfr'][0] *=  1 - (10**sfh['t1'][0]/10**sfh['t2'][0])
        dt = 10**sfh['t2'] - 10**sfh['t1']
        # Build spectrum
        if (av is 0) and (dav is 0):
            flux = self.spec.T * sfh['sfr'] #use broadcasting
            total_spec = flux.sum(axis = 1) #and sum over age bins
        else:
            nsplit = 9. * np.any(dav > 0) + 1
            total_spec = self.redden(sfh, np.asarray(av), np.asarray(dav), nsplit = nsplit)        
        mtot = (sfh['sfr'] * dt).sum()
        mstar = (sfh['sfr'] * self.mstar).sum()
        if filterlist is not None:
            broadband = observate.getSED(self.wave, total_spec, filterlist)
        else:
            broadband = None
            
        return total_spec, mstar, broadband

    def redden(self, sfh, av, dav, nsplit = 10.):
        """
        Redden the spectral bases in 'self.spec' and combine them to produce an
        integrated spectrum.  av and dav may be vectors with same length as the
        number of SFR bins (i.e. for age dependent attenuation)

        :param av:
            The foreground attenuation affecting all stars of all ages equally,
            scalar or sequence
            
        :param dav:
            The differential reddening, defined such that the distribution of
            reddening is uniform from av to dav, scalar or sequence.
            
        :param nsplit: (default: 10)
            The number of peices into which to split the spectrum when approximating
            the uniform distribution up to dav.  use larger numbers for higher accuracy
            (especially when dav is large)

        :returns total_spec_red:
            The reddened spectrum
        """
        lisplit = self.spec/nsplit
        #enable broadcasting if av and dav aren't vectors
        av = np.atleast_1d(av)
        dav = np.atleast_1d(dav) 
        #uniform distribution from Av to Av + dAv
        avdist = av[None, :] + dav[None,:] * ((np.arange(nsplit) + 0.5)/nsplit)[:,None]
        ee = (np.exp(-self.tau_curve[None,None,:] * avdist[:,:,None]))
        li0_red = (ee * lisplit[None,:,:]).sum(axis = 0)
        total_spec_red = (li0_red.T * sfh['sfr']).sum(axis = 1)
        return total_spec_red

        # Redden spectrum via analytic expressions for the distribution
        #of tau - why doesn't this work?
        #factor = 1/(tau_curve * dav / 1.086) * (np.exp(0 - tau_curve * av/1.086) -
        #   np.exp(0 - tau_curve * (av+dav)/1.086))
        #li0_red = factor[None,:] * li0.sum(axis = 0)
        #total_spec_red_v2 = (li0_red.T * sfh['sfr']).sum(axis = 1)

    def sexAmodel(self, avmax, sfile):
        """Calculate the distribution of max Av as a function of age for the Dolphin 2002
        differential attenuation model.

        :param avmax:
            The maximum A_v at t_lookback = 0
        :param sfile:
            String piving the name and path to the SFH file defining the time binning

        :returns dav:
            A sequence of length the number of time bins, giving dA_v max in each time bin
        """
        sfh = utils.load_angst_sfh(sfile)
        dav = np.clip(((10**sfh['t2']-4e7) * (-avmax) / 0.6e8 + avmax), 0, avmax)
        return dav


def generate_basis(sfh_template, zmet = 1.0, imf_type = 0, outroot = 'L0',  t_lookback = [0],
                   narr = 500, t0 = 0.0005 * 1e9, clobber = False):
    """Method to produce a spectral basis file for a given set of time bins.
    Uses subprocess to call autosps from FSPS after generating a top hat user
    SFH file with SFR = 1 Msun/yr, which is a pretty hacky workaround.  Requires
    FSPS to be installed and $SPS_HOME to point to the correct fsps directory.

    :param sfh_template:
        String giving the name and path to the SFH file defining the time binning
        
    :param zmet: (default: 1.0)
        Metallicity in units of solar (linear)
        
    :param imf_type: (default: 0)
        The IMF to use.  Defaults to Salpeter. see FSPS manual for details.
        
    :param narr:
        Number of linear time points to use for defining each bin in the sfh.dat
        user file.
        
    :param t0:
        Extra time padding to add to the beginning of each bin, to please FSPS.
        
    :param outroot:
        Output file name root.  Metallicity and IMF info will be automatically
        appended to this.
        
    :param t_lookback:
        lookback time (in yrs) at which to calculate the spectrum of each bin of SF.
        If it's before the beginning of a given bin then the spectrum and mass for
        that bin are set to zero.

    :param clobber: (deafult False)
        If False, then if the filename already exists do not recompute the basis and
        return the filename.  Otherwise, force recomputation of the basis.
        
    :returns outname:
        A string (array) giving the path and filenames of the produced basis file(s).  The basis file
        consists of  an array of shape [NBIN+1, NWAVE+1] where the extra indices are to
        store the wavelength vector and the stellar mass vector.  Otherwise the data consists
        of the spectrum of each bin (top-hat SFH) at time t_lookback, in units of
        erg/s/cm**2/AA per M_sun/yr at 10pc distance.
    """

    # Set up the output filename(s) and return if file(s) already exists and not clobber
    t_lookback = np.atleast_1d(t_lookback)
    outnames = ['{2}_{0}_z{1:.1f}_tl{3:2.4f}Gyr.fits'.format(imfname[imf_type], np.log10(zmet), outroot, tl*1e-9) for tl in t_lookback]
        
    if (False not in [os.path.exists(o) for o in outnames]) and (clobber is False):
        return outnames
    
    # Read the template SFH file to get time bin definitions
    sfh = utils.load_angst_sfh(sfh_template)
    nbin = len(sfh)
    sfh['t2'] = 10**sfh['t2']
    sfh['t1'] = 10**sfh['t1']
    sfh[0]['t1'] = 0

    biggest_bin = (sfh['t2'] - sfh['t1']).max()
    smallest_bin = (sfh['t2'] - sfh['t1']).min()
    dt = biggest_bin/narr
    time = np.arange(narr+10) * dt
    time += t0 #FSPS is not happy with SFHs that start at zero age, so pad
    nlead = 20.
    leadin = t0/nlead

    for i, abin in enumerate(sfh):

        # Build the SFH
        sfr = np.zeros_like(time)
        on = (time-time[0] < (abin['t2'] - abin['t1']))
        sfr[on] = 1.0 #SFR of one

        # Produce and read an FSPS spectral file
        a_gyr, logm, logl, logs, spec, wave, hdr = get_fsps_spectrum(time, sfr, zmet, imf_type, leadin)
        spec *= to_cgs * lightspeed/(wave[None,:]**2)
        # If necessary build the output
        if i is 0:
            aspec = np.zeros([ len(t_lookback), len(sfh), len(wave)])
            amass= np.zeros([ len(t_lookback), len(sfh)])
    
        # Interpolate the fsps spectra (and masses) to the present day
        #   (or to t_lookback), taking into account the non-zero begining of the SFH
        #   use broadcasting to do this for multiple lookback times
        tprime = abin['t2'] + t0 - t_lookback
        future_sf = tprime <= 0 #if tprime > 0:
        tprime[future_sf] = 1e9
        inds, weights = utils.weights_1DLinear(np.log(a_gyr),np.log(tprime / 1e9))
        inds[future_sf,:] = 0 
        weights[future_sf,:] = 0 #zero out the contributions from bins younger than the lookback time
        aspec[:,i,:] = (spec[inds,:] * weights[...,None]).sum(axis =1)
        amass[:,i] = (weights * 10**logm[inds]).sum(axis = -1)
    
    w0 = wave.copy()
    for it, tl in enumerate(t_lookback):
        # Dumb storage scheme, but subarrays in pyfits binary tables don't work so well
        # add a dummy wavelength for the stellar masses
        wave = np.hstack([w0, w0.max()+10])
        # append the stellar mass to the end of the spectrum
        spec = np.vstack([aspec[it,...].T, amass[it,:]]).T
        # append wavelength as the first 'spectrum'
        data = np.vstack([wave, spec]) 

        # Write the FITS file
        print('Writing basis file to {0}'.format(outnames[it]))
        hdu = pyfits.PrimaryHDU(data)
        hdu.header['BUNIT'] = 'erg/s/cm**2/AA per M_sun/yr at 10pc'
        hdu.header['comment'] = ('Each spectrum is for a top-hat' +
                                ' SFH with SFR of 1 M_sun/year.')
        hdu.header['comment'] = ('The first spectrum vector is ' +
                                'actually the wavelength scale')
        hdu.header['comment'] = ('The last wavelength vector is actually' +
                                ' the stellar mass, in units of M_sun')
        hdu.writeto(outnames[it], clobber =True)

    return outnames

def get_fsps_spectrum(time, sfr, zmet, imf_type, leadin):
    """ Use the autosps.exe program of FSPS to generate the spectral
    evolution for an arbitrary SFH, and return properties of the stellar
    population so generated.  This is a hacky interface to FSPS.

    :param time:
       time axis (in yrs) for the SFH definition. This should *not*
       start at zero. NDARRAY

    :param sfr:
        SFR (M_Sun/yr) corresponding to the time array.  NDARRAY
        
    :param zmet: (default: 1.0)
        Metallicity in units of solar (linear)
        
    :param imf_type: (default: 0)
        The IMF to use.  Defaults to Salpeter. see FSPS manual for details.

    :param leadin:
        time step to use for the padding of the time array.  4 extra time steps
        of size leadin and SFR = 0 will be prepended to the SFH

    :returns sps:
        see sfhutils.read_fsps() for a description of the output.

    """
    # Write an SFH file
    f = open(tabsfh, 'wb')
    # add a few SFR=zero timesteps before the main SFH
    for j in [3,2,1,0]:
        f.write('{0}  {1}  {2}\n'.format((time[0] - j* leadin)/1e9,
                                         0.0, zmet * Zsun))
    # the main SFH
    for j, t in enumerate(time):
        f.write('{0}  {1}  {2}\n'.format(t/1e9, sfr[j], zmet * Zsun))
    f.close()

    # Call autosps and feed it the relevant parameters
    #    should really just build the tophats from the fsps SSPs....
    p = subprocess.Popen(executable,stdin=subprocess.PIPE,
                         stdout = subprocess.PIPE, shell =True)
    p.stdin.write('{0}\n'.format(imf_type))
    p.stdin.write('2\n') #use tabulated SFH
    p.stdin.write('\n') #no dust
    p.stdin.write('{0}\n'.format(fsps_outname))
    _ = p.communicate()[0]

    # Read the spectral file produced by fsps
    return utils.read_fsps(spsdir + 'OUTPUTS/' + fsps_outname + '.spec')




#rp = {'specfile': 'L0_cluster_.fits',
#      'nrebin':5, 'nsplit':30.,
#      'outtext':'results/m31_broadband_region'}
#rp['filternamelist'] = ['galex_FUV','galex_NUV','sdss_g0', 'sdss_r0']
#rp['filterlist'] = observate.load_filters(rp['filternamelist'])

# Parameters for rebinning the spectra in different age bins
#ss = (rp['nbin']/rp['nrebin'])*np.arange(rp['nrebin'])
#se = (ss-1).tolist()[1:]+[rp['nbin']-1]
#rebin = {'start':ss.tolist(), 'end':se}
#rebin['color'] = ['m','b','y','g','r']

#plot stellar mass normalized spectra
#utils.plot_agespec(region, sfh, flux/mtot[ig], rp, rebin, ylim = [1e-7,1e-2]) 


