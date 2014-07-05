# Read the Harris and Zaritsky data files ( obtained from
#  http://djuma.as.arizona.edu/~dennis/mcsurvey/Data_Products.html )
#  into a dictionary of SFHs.  Each key of the returned dictionary is
#  a region name, and each value is a dictionary containing a list of
#  SFHs for that region (one for each metallicity), a list of
#  metallicities of each SFH, and a location string giving the RA, Dec
#  coordinates of the region.  There is also one key in the
#  dictionary, 'header', containing header information from the
#  original files.  Each of the SFHs is a structured ndarray that can be
#  input to scombine methods.

import numpy as np
lmcfile = '/Users/bjohnson/Projects/mcmcmc/sfhs/lmc_sfh.dat'
smcfile = '/Users/bjohnson/Projects/mcmcmc/sfhs/smc_sfh.dat'

def lmc_regions(filename = lmcfile):

    regions = {}
    k = 'header'
    regions[k] = []
    
    f = open(filename, 'rb')
    for line in f:
        line = line.strip()
        if line.find('Region') >=0:
            k = line.split()[-1]
            regions[k] = {'sfhs':[]}
        elif line.find('(') == 0:
            regions[k]['loc'] = line
        elif len(line.split()) > 0 and line[0] != '-':
            if k == 'header':
                regions[k] += [line]
            else:
                regions[k]['sfhs'] += [[float(c) for c in line.split()]]
    f.close()
    for k, v in regions.iteritems():
        if k == 'header':
            continue
        sfhs, zs = process_lmc_sfh( v['sfhs'])
        regions[k]['sfhs'] = sfhs
        regions[k]['zmet'] = zs
        
    return regions

def process_lmc_sfh(dat):
    """
    Take a list of lists, where each row is a time bin, and convert it
    into several SFHs at different metallicities.
    """
    all_sfhs = []
    zlegend = np.array([0.001, 0.0025, 0.004, 0.008])
    usecol = [10,7,4,1]
    
    s = np.array(dat)
    inds = np.argsort(s[:,0])
    s = s[inds, :]
    wt = np.diff(s[:,0]).tolist()
    wt = np.array([wt[0]]+wt)
    nt = s.shape[0]
    ty = '<f8'
    dt = np.dtype([('t1', ty), ('t2',ty), ('dmod',ty), ('sfr',ty), ('met', ty), ('mformed',ty)])
    for zindex, zmet in enumerate(zlegend):
        data = np.zeros(nt, dtype = dt)
        data['t1'] = s[:,0] - wt
        data['t2'] = s[:,0]
        data['met'] = np.log10(zmet/0.019)
        data['sfr'] = s[:, usecol[zindex]] * 1e-6
        data['dmod'] = 18.50
        all_sfhs += [data]

    return all_sfhs, zlegend
    
def smc_regions(filename = smcfile):
    #wow.  really?  could this *be* harder to parse?
    #why so differnt than lmc?
    regions ={'header':[]}
    f = open(filename, 'rb')
    for i, line in enumerate(f):
        line = line.strip()
        if i < 26:
            regions['header'] += line
        else:
            cols = line.split()
            reg = cols[0]
            if len(cols)  == 13:
                #padding.  because who does this?  unequal line lengths?
                cols = cols + ['0','0','0']
            
            regions[reg] = regions.get(reg, []) + [[float(c) for c in cols[1:]]]
    f.close()
    #reprocess to be like lmc
    for k,v in regions.iteritems():
        if k == 'header':
            continue
        sfhs, zs, loc = process_smc_sfh( v )
        regions[k] = {}
        regions[k]['sfhs'] = sfhs
        regions[k]['zmet'] = zs
        regions[k]['loc'] = loc
        
    return regions

def process_smc_sfh(dat):
    """
    Take a list of lists, where each row is a time bin, and convert it
    into several SFHs at different metallicities.
    """
    all_sfhs = []
    zlegend = np.array([0.001, 0.004, 0.008])
    usecol = [13,10,7]
    
    s = np.array(dat)
    inds = np.argsort(s[:,4])
    s = s[inds, :]
    nt = s.shape[0]
    ty = '<f8'
    dt = np.dtype([('t1', ty), ('t2',ty), ('dmod',ty), ('sfr',ty), ('met', ty), ('mformed',ty)])
    for zindex, zmet in enumerate(zlegend):
        data = np.zeros(nt, dtype = dt)
        data['t1'] = s[:,4]
        data['t2'] = s[:,5]
        data['met'] = np.log10(zmet/0.019)
        data['sfr'] = s[:, usecol[zindex]] * 1e-6
        data['dmod'] = 18.50
        all_sfhs += [data]
    loc = "( {0:02.0f}h {1:02.0f}m {2:02.0f}d {3:02.0f}m )".format(*dat[0][0:4])
    return all_sfhs, zlegend, loc
