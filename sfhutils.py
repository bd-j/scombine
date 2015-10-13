import glob
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as pl

loglsol = 3.839e33
pc2cm = 3.086e18
magsphere = 4.*np.pi*100*pc2cm**2
skiprows = 0 #number of extra rows at the top of the SFH files

def load_angst_sfh(name, sfhdir = '', skiprows = 0, fix_youngest = False, bg_flag=False, skip_footer=2):
    """
    Read a `match`-produced, zcombined SFH file into a numpy
    structured array.

    :param name:
        String giving the name (and optionally the path) of the SFH
        file.
    :param skiprows:
        Number of header rows in the SFH file to skip.
    """
    #hack to calculate skiprows on the fly
    tmp = open(name, 'rb')
    while len(tmp.readline().split()) < 14:
        skiprows += 1
    tmp.close()
    ty = '<f8'
    dt = np.dtype([('t1', ty), ('t2',ty), ('dmod',ty), ('sfr',ty), ('met', ty), ('mformed',ty)])
    #fn = glob.glob("{0}*{1}*sfh".format(sfhdir,name))[0]
    fn = name
    if bg_flag:
        data = np.genfromtxt(fn, usecols=(0,1,2,3,6,12) , dtype=dt, skip_header=skiprows, skip_footer=skip_footer)
    else:
        data = np.loadtxt(fn, usecols = (0,1,2,3,6,12) ,dtype = dt, skiprows = skiprows)
    if fix_youngest:
        pass
    return data

def load_phat_sfh(name, zlegend):
    """
    Read a `match`-produced SFH file that was not zcombined into a
    numpy structured array.

    :param name:
        String giving the name (and optionally the path) of the SFH
        file.
        
    :param skiprows:
        Number of header rows in the SFH file to skip.
    """
    zedges = np.log10(zlegend/0.019)
    zedges = zedges[0:-1]+ np.diff(zedges)/2.
    #zedges = np.array([-np.inf] + zedges.tolist() + [np.inf])
    ty = '<f8'
    dt = np.dtype([('t1', ty), ('t2',ty), ('dmod',ty), ('sfr',ty), ('met', ty), ('mformed',ty)])

    alldat = []
    sfhs = []

    # Read in file, discarding header and footer
    #  and convert to a big array
    f = open(name,'r')
    for i, line in enumerate(f):
        dat = line.split()
        if len(dat) < 5:
            continue
        alldat += [[float(d) for d in dat]]
    alldat = np.array(alldat)
    zbin = np.digitize(alldat[:,4], zedges)

    for iz in range(len(zlegend)):
        zinds = (zbin == iz)
        thismet = alldat[zinds,:]
        t1s, t2s = np.unique(thismet[:,0]), np.unique(thismet[:,1])
        nt = len(t1s)
        sfh = np.zeros(nt, dtype = dt)
        sfh['t1'], sfh['t2'] = t1s, t2s
        sfh['met'], sfh['dmod'] =  np.log10(zlegend[iz]/0.019), alldat[0,-1]
        
        # For each time bin, accumulate the sfr
        #  of all entries in this metallicity bin 
        for it in range(nt):
            sfr  = thismet[thismet[:,0] == t1s[it], 2].sum()
            sfh['sfr'][it] = sfr
        sfhs += [sfh]
    return sfhs

def read_lfs(filename):
    """
    Read a Villaume/FSPS produced cumulative LF file, interpolate LFs
    at each age to a common magnitude grid, and return a dictionary
    containing the interpolated CLFs and ancillary information.

    :param filename:
        The filename (including path) of the Villaume CLF file

    :returns luminosity_func:
        A dictionary with the following key-value pairs:
        ssp_ages: Log of the age for each CLF, ndarray of shape (nage,)
        lf:       The interpolated CLFs, ndarray of shape (nage, nmag)
        bins:     Magnitude grid for the interpolated CLFs, ndarray of
                  shape (nmag,)
        orig:     2-element list contining the original magnitude grids
                  and CLFs as lists.
        
    """
    #luminosity_functions = {}
    age, bins, lfs = [], [], []
    f = open(filename, "r")
    for i,line in enumerate(f):
        dat = [ float(d) for d in line.split() ]
        if (i % 3) == 0:
            age += [ dat[0]]
        elif (i % 3) == 1:
            bins += [dat]
        elif (i % 3) == 2:
            lfs += [dat]
    f.close()
    
    age = np.array(age)
    minage, maxage = np.min(age)-0.05, np.max(age)+0.10
    minl, maxl = np.min(bins)[0], np.max(bins)[0]+0.01
    allages = np.arange(minage, maxage, 0.05)
    mags = np.arange(minl, maxl, 0.01)
    print(minl, maxl)
    
    lf = np.zeros([ len(allages), len(mags)])
    for i, t in enumerate(allages):
        inds = np.isclose(t,age)
        if inds.sum() == 0:
            continue
        ind = np.where(inds)[0][0]
        x = np.array(bins[ind] + [np.max(mags)])
        y = np.log10(lfs[ind] +[np.max(lfs[ind])])
        
        lf[i, :] = 10**interp1d(np.sort(x), np.sort(y), fill_value = -np.inf, bounds_error = False)(mags)

    
    luminosity_func ={}
    luminosity_func['ssp_ages'] = allages
    luminosity_func['lf'] = lf
    luminosity_func['bins'] = mags
    luminosity_func['orig'] = [bins, lfs]

    return luminosity_func

def read_fsps(filename):
    """
    Read a .spec file produced by FSPS and return a number of arrays
    giving quantities of the stellar population.  These are: [age (in
    Gyr), log(Mstar), log(Lbol), log(SFR), spectrum, wavelength,
    header]

    :param filename:
        The file name of the FSPS output .spec file, including path.
        
    :returns age:
        The age of the stellar population (in Gyr).  1D array of shape
        (Nage,)
        
    :returns mstar:
        The log of the stellar mass (in solar units) at `age'.  1D
        array.
        
    :returns lbol:
        The log of the bolometric luminosity at `age'.  1D array
        
    :returns sfr:
        The log of the SFR (M_sun/yr) at `age'.  1D array
        
    :returns spectrum:
        The spectrum of the stellar population at `age'.  (L_sun/Hz),
        2D array of shape (Nage, Nwave)
        
    :returns wavelngth:
        The wavelength vector (AA).  1D array.
        
    :returns header:
        String array of header information in the .spec file
    """
    age, logmass, loglbol, logsfr, spec, header =[],[],[],[],[], []
    with open(filename, "r") as f:
        line = f.readline()
        while line[0] is '#':
            header.append(line)
            line = f.readline()
        nt, nl = line.split()
        wave = [float(w) for w in f.readline().split()]
        for i in range(int(nt)):
            a, m, l, s = f.readline().split()
            age.append(10**float(a)/1e9)
            logmass.append(float(m))
            loglbol.append(float(l))
            logsfr.append(float(s))
            s = [float(l) for l in f.readline().split()]
            spec.append(s)
        f.close()
    return np.array(age), np.array(logmass), np.array(loglbol), np.array(logsfr), np.array(spec), np.array(wave), header

def weights_1DLinear(model_points, target_points,
                     extrapolate = False, left=0.0, right=0.0,
                     **extras):
    """The interpolation weights are determined from 1D linear
    interpolation.
    
    :param model_points: ndarray, shape(nmod)
        The parameter coordinate of the available models.  assumed to
        be sorted ascending
                
    :param target_points: ndarray, shape(ntarg)
        The coordinate to which you wish to interpolate
            
    :returns inds: ndarray, shape(ntarg,2)
         The model indices of the interpolates
             
    :returns weights: narray, shape (ntarg,2)
         The weights of each model given by ind in the interpolates.
    """
    #well this is ugly.
    mod_sorted = model_points
    
    x_new_indices = np.searchsorted(mod_sorted, target_points)
    x_new_indices = x_new_indices.clip(1, len(mod_sorted)-1).astype(int)
    lo = x_new_indices - 1
    hi = x_new_indices
    x_lo = mod_sorted[lo]
    x_hi = mod_sorted[hi]
    width = x_hi - x_lo    
    w_lo = (x_hi - target_points)/width
    w_hi = (target_points - x_lo)/width

    if extrapolate is False:
        #and of course I have these labels backwards
        above_scale = w_lo < 0 #fidn places where target is above or below the model range
        below_scale = w_hi < 0
        lo[above_scale] = hi[above_scale] #set the indices to be indentical in these cases
        hi[below_scale] = lo[below_scale]
        w_lo[above_scale] = 0 #make the combined weights sum to one
        w_hi[above_scale] = left
        w_hi[below_scale] = 0
        w_lo[below_scale] = right

    inds = np.vstack([lo,hi]).T
    weights = np.vstack([w_lo, w_hi]).T
    #inds = order[inds]
    return inds, weights

def plot_agespec(name, sfh, flux, rp, rebin, ylim = None):
    pl.figure(1)
    for i in xrange(rp['nrebin']):
        binned_flux = flux[:, rebin['start'][i]:rebin['end'][i]+1].sum(axis =1)
        label = '{0:3.2f} <log t<{1:4.2f}'.format(sfh['t1'][rebin['start'][i]], sfh['t2'][rebin['end'][i]])
        
        pl.plot(rp['wave'], binned_flux*(magsphere/loglsol), label = label, color = rebin['color'][i])

    pl.plot(rp['wave'], flux.sum(axis = 1)*(magsphere/loglsol), label = 'total', color = 'k')
    pl.title(name)
    pl.xlabel('wave')
    pl.ylabel(r'$L_{\lambda,bin}/M_{*,tot} (L_{\odot}/M_{\odot}/\AA)$')
    pl.yscale('log')
    pl.xscale('log')
    pl.xlim(2e3, 2e4)
    if ylim is not None: pl.ylim(ylim[0], ylim[1])
    pl.legend()
    pl.savefig('{0}_agespec.png'.format(name))
    pl.close()
