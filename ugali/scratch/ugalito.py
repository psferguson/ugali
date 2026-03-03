#!/usr/bin/env python
"""
Light version of ultra-faint galaxy likelihood tools.
"""
__author__ = "Alex Drlica-Wagner"
import os
import warnings
warnings.filterwarnings('ignore')
import copy
from collections import OrderedDict as odict

import scipy.special
import scipy.stats

import pandas as pd
import numpy as np

import scipy.stats
from scipy.stats import poisson, norm

from astropy.io import fits


BOUNDS = odict([
     ('rich', [0,    5000]),   # [stars]
     ('lon' , [1000, 2500]),   # [pix]
     ('lat' , [250,  1900]),   # [pix]
     ('ext' , [50,   2500]),   # [pix]
     ('ell' , [0,    0.95]),   #
     ('pa'  , [0,     180]),   # [deg]
     ('sd'  , [0,    1000]),   # [stars/Mpix]
])
PARAMS = list(BOUNDS.keys())

NPIX_X,NPIX_Y = 3300,2250
PIX_SCALE = 2* 0.0807 # arcsec per pixel 

# Binned parameters
BINSIZE = 25
NBINS_X, NBINS_Y = NPIX_X//BINSIZE, NPIX_Y//BINSIZE
XMIN,XMAX = 0,NPIX_X
YMIN,YMAX = 0,NPIX_Y
# Bin edges
XEDGE = np.linspace(XMIN,XMAX,NBINS_X + 1)
YEDGE = np.linspace(YMIN,YMAX,NBINS_Y + 1)
# Bin centers
XCENT = (XEDGE[1:] + XEDGE[:-1])/2.
YCENT = (YEDGE[1:] + YEDGE[:-1])/2.
# Bin widths (shouldn't this just be BINSIZE?)
XDEL = XEDGE[1]-XEDGE[0]
YDEL = YEDGE[1]-YEDGE[0]
# Bin coordinates
XX,YY= np.meshgrid(XCENT,YCENT)
print(XX.shape)
print(YY.shape)

# For statistics
_alpha   = 0.32
_nbins   = 300
_npoints = 500


# Statistics and distributions
def vonmises_rvs(loc, scale, size):
    """ Modified Von Mises random variate in degrees. Returns values from [loc-90,loc+90].
    WARNING: Not extensively validated...

    For large values of kappa:
      kappa ~ 1/sigma**2

    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.vonmises.html
    https://en.wikipedia.org/wiki/Von_Mises_distribution

    Parameters
    ----------
    loc   : the central location (deg)
    scale : approximate width (analogous to Gaussian sigma) (deg)
    size  : number of samples
    
    Returns
    -------
    rvs   : random samples
    """
    kappa = 1/np.radians(2*scale)**2 # factor of 2 because sampled from -pi to pi
    samples = scipy.stats.vonmises.rvs(kappa, loc=0, size=size)
    samples = np.degrees(samples)/2. # map from -90 to 90
    samples += loc                   # move center to loc
    #samples += 180*(samples < 0) - 180*(samples > 180)   # wrap to 0 to 180
    return samples

RVS = odict([
    ('norm',       scipy.stats.norm.rvs),
    ('poisson',    scipy.stats.poisson.rvs),
    ('uniform',    np.random.uniform),
    ('loguniform', scipy.stats.loguniform.rvs),
    ('vonmises',   vonmises_rvs),
])


###########################
### General Utilities #####
###########################


def mkdir(path):
    # https://stackoverflow.com/a/600612/4075339
    import errno    
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise
    return path


def generate_random_params(params, size=1):
    """Generate random parameter values drawn from some random variate.
    Parameter values are constrained by the parameter bounds.
    """
    random_params = []

    for name in PARAMS:
        loc,scale = params[name][:2]
        rvs = RVS[params[name][2]]
        values = rvs(loc,scale,size=size)
        values = np.abs(values) # all parameters are positive
        values = np.clip(values, BOUNDS[name][0], BOUNDS[name][1]) # within bounds

        # Set values outside bounds to the value of loc
        #outside = (values < BOUNDS[name][0]) | (values > BOUNDS[name][1])
        #values[outside] = loc

        # Pad the bounds a bit
        #epsilon = 1e-2 * (BOUNDS[name][1] - BOUNDS[name][0])
        #values = np.clip(values, BOUNDS[name][0] + epsilon, BOUNDS[name][1] - epsilon)

        random_params.append(values)

    return np.array(random_params).T

def generate_walkers(values, nwalkers=100):
    """ Initiate walkers around nominal values (help quick convergence).

    Parameters
    ----------
    values   : initial parameter values
    scale    : width of normal distribution as a fraction of values
    nwalkers : number of walkers

    Returns
    -------
    pos     : Initial walker positions (nparams, nwalkers)
    """
    params = odict()
    params['rich'] = [values[0], 0.1*values[0], 'norm']
    params['lon']  = [values[1], 100, 'norm']
    params['lat']  = [values[2], 100, 'norm']
    params['ext']  = [values[3], 50, 'norm']
    params['ell']  = [values[4]-0.1, values[4]+0.1, 'uniform']
    #params['pa']   = [values[5], 25, 'vonmises']
    #params['pa']   = [0,        180, 'uniform']
    params['pa']   = [values[5]-90, values[5]+90, 'uniform']
    params['sd']   = [values[6], 0.1*values[6], 'norm']

    return generate_random_params(params, nwalkers)


def bin2d(image,binsize):
    """ Rebin the 2-dimensional image.
    Taken from https://stackoverflow.com/questions/61325586/fast-way-to-bin-a-2d-array-in-python
    """
    m_bins = image.shape[0]//binsize
    n_bins = image.shape[1]//binsize
    return image.reshape(m_bins, binsize, n_bins, binsize).sum(3).sum(1)

def interval(best,lo=np.nan,hi=np.nan):
    """
    Pythonized interval for easy output to yaml
    """
    return [float(best),[float(lo),float(hi)]]

def median_interval(data, alpha=0.32):
    """
    Median including Bayesian credible interval.

    Parameters
    ----------
    data  : posterior samples
    alpha : 1 - confidence interval

    Returns
    -------
    [med,[lo, hi]] : median, lower, and upper percentiles
    
    """
    q = [100*alpha/2., 50, 100*(1-alpha/2.)]
    lo,med,hi = np.percentile(data,q)
    return interval(med,lo,hi)

def histogram_peak(data, bins=100):
    """
    Bin the distribution and find the mode

    Parameters:
    -----------
    data  : The 1d data sample
    bins  : Number of bins

    Returns
    -------
    peak : peak of the kde
    """
    num,edges = np.histogram(data,bins=bins)
    centers = (edges[1:]+edges[:-1])/2.
    return centers[np.argmax(num)]


def kde_peak(data, npoints=_npoints, clip=5.0):
    """
    Identify peak using Gaussian kernel density estimator.

    Parameters:
    -----------
    data    : The 1d data sample
    npoints : The number of kde points to evaluate
    clip    : NMAD to clip

    Returns
    -------
    peak : peak of the kde
    """
    return kde(data,npoints,clip)[0]

def kde(data, npoints=_npoints, clip=5.0):
    """
    Identify peak using Gaussian kernel density estimator.
    
    Parameters:
    -----------
    data    : The 1d data sample
    npoints : The number of kde points to evaluate
    clip    : NMAD to clip

    Returns
    -------
    peak : peak of the kde
    """

    # Clipping of severe outliers to concentrate more KDE samples
    # in the parameter range of interest
    mad = np.median(np.fabs(np.median(data) - data))
    if clip > 0:
        cut  = (data > np.median(data) - clip * mad)
        cut &= (data < np.median(data) + clip * mad)
        x = data[cut]
    else:
        x = data
    kde = scipy.stats.gaussian_kde(x)
    # No penalty for using a finer sampling for KDE evaluation
    # except computation time
    values = np.linspace(np.min(x), np.max(x), npoints)
    kde_values = kde.evaluate(values)
    peak = values[np.argmax(kde_values)]
    return peak, kde.evaluate(peak)

def peak_interval(data, alpha=_alpha, npoints=_npoints):
    """Identify minimum interval containing the peak of the posterior as
    determined by a Gaussian kernel density estimator.

    Parameters
    ----------
    data   : the 1d data sample
    alpha  : the confidence interval
    npoints: number of kde points to evaluate

    Returns
    -------
    interval : the minimum interval containing the peak
    """
    peak = kde_peak(data,npoints)
    x = np.sort(data.flat); n = len(x)
    # The number of entries in the interval
    window = int(np.rint((1.0-alpha)*n))
    # The start, stop, and width of all possible intervals
    starts = x[:n-window]; ends = x[window:]
    widths = ends - starts
    # Just the intervals containing the peak
    select = (peak >= starts) & (peak <= ends)
    widths = widths[select]
    if len(widths) == 0:
        raise ValueError('Too few elements for interval calculation')
    min_idx = np.argmin(widths)
    lo = starts[select][min_idx]
    hi = ends[select][min_idx]
    return interval(peak,lo,hi)

def angle_shift_peak(data):
    # Transform so peak in the middle of the distribution
    peak = kde_peak(data)
    shift = -180.*(data > (peak+90)) + 180.*(data < (peak-90))
    # Shouldn't happen, but maybe...
    #if peak < -90: shift += 180.
    #if peak > 270: shift -= 180.
    return data + shift

def angle_peak_interval(data, alpha=_alpha):
    """ Estimate the position angle from the posterior dealing
    with the periodicity of the position angle.

    Parameters
    ----------
    data  : posterior samples
    alpha : 1 - confidence interval

    Returns
    -------
    [med, [lo, hi]] : median, lower, and upper percentiles
    """
    # Transform so peak in the middle of the distribution
    peak = kde_peak(data)
    shift = -180.*(data > (peak+90)) + 180.*(data < (peak-90))
    

    # Get the kde interval
    ret = peak_interval(data + shift,alpha)
    # Shift to be positive
    if ret[0] < 0: 
        ret[0] += 180.; ret[1][0] += 180.; ret[1][1] += 180.;
    return ret

def angle_median_interval(data, alpha=_alpha):
    """ Median and percentile interval suitable for angle wrapping from positional angle.

    Parameters
    ----------
    data  : posterior samples
    alpha : 1 - confidence interval

    Returns
    -------
    [med, [lo, hi]] : median, lower, and upper percentiles
    """
    # Transform so peak in the middle of the distribution
    peak = kde_peak(data)
    shift = -180.*(data > (peak+90)) + 180.*(data < (peak-90))
    # Get the kde interval
    ret = median_interval(data + shift,alpha)
    # Shift to be positive
    if ret[0] < 0: 
        ret[0] += 180.; ret[1][0] += 180.; ret[1][1] += 180.;
    return ret


###########################
### Data Manipulation #####
###########################

def load_image(filename, shape=(0,0), threshold = 1000):
    """ Load the image and extend to the desired shape.
    
    Parameters
    ----------
    filename : name of image file
    shape    : desired output shape
    
    Returns
    -------
    raw,data : raw and extended image data
    """
    hdul = fits.open(filename)
    try: data = hdul[1].data
    except: data = hdul[0].data
    extended_data = np.zeros(shape, dtype=float)
    extended_data[:data.shape[0],:data.shape[1]] = data

    full_res_mask = np.zeros_like(extended_data)
    full_res_mask[extended_data >= threshold] = 1 
    full_res_mask[extended_data  < threshold] = 0

    return data, extended_data, full_res_mask

def create_binned_mask(full_res_mask, binsize, zero_thresh):
    """Create a binned mask from the image data. Uses pixel value
    threshold to determine whether a pixel was observed or not.

    Parameters
    ---------
    image     : image data array (extended)
    threshold : threshold to determine whether a pixel was observed.

    Returns
    -------
    binned_mask : binned weight map

    """
    # Binsize currently fixed at import [edit: now changed]
    binsize= binsize

    ## 1000 is above the level of the non-science data but below bkg of science data
    

    binned_mask = bin2d(image=full_res_mask, binsize=binsize) / binsize**2
 
    ## mask superpixels covered by a small amount or less (avoids CRs counting as coverage...)
    binned_mask[binned_mask < zero_thresh] = 0
    return binned_mask

def data(x,y):
    """ Calculate the binned data counts. This only needs to be done
    once (not at each model evaluation), but this seemed easier to
    understand if it paralleled the model counts calculation.

    Parameters
    ----------
    x : the x coordinate of the data
    y : the y coordinate of the data

    Returns
    -------
    data_counts : the data counts in each bin
    """
    # Histogram uses opposite indexing convention...
    data_counts,_,_ = np.histogram2d(x,y,bins=[XEDGE,YEDGE])
    return data_counts.T

###########################
### Likelihood analysis ###
###########################
def plummer_kernel(x, y, lon=0, lat=0, ext=100, ell=0.3, pa=45):
    """ Evaluate the elliptical Plummer kernel at coordinates x,y. 
    Normalized to unity over all space...

    Parameters
    ----------
    x: x-coord for evaluating kernel [pix]
    y: y-coord for evaluating kernel [pix]
    lon: x-coord of kernel centroid [pix]
    lat: y-coord of kernel centroid [pix]
    ext: extension [pix]
    ell: ellipticity
    pa:  position angle [deg]

    Returns
    -------
    pdf : probability density (should integrate to unity over all space)
    """

    # Elliptical radius of each x,y coord
    costh = np.cos(np.radians(-pa))
    sinth = np.sin(np.radians(-pa))
    dx = x-lon
    dy = y-lat
    radius = np.sqrt(((dx*costh-dy*sinth)/(1-ell))**2 + (dx*sinth+dy*costh)**2)

    #PLUMMER SCALE RADIUS = HALF-LIGHT RADIUS 
    r_e = ext
    #Normalization (integrates to unity over all space?) [stars/pix^2]
    norm = r_e**2/(np.pi*(1-ell))

    # Plummer PDF
    pdf = norm / ((radius**2 + r_e**2)**2)

    return pdf


def model(theta):
    """ Calculate the binned model counts. This extends over the
    entire pixel range, but we will apply the mask later.

    Parameters
    ----------
    theta : the model parameters
    
    Returns
    -------
    model_counts : the model counts in each bin
    """
    #FIT ALL MODEL PARAMETERS
    richness = theta[0]
    kwargs = dict(lon=theta[1],lat=theta[2],ext=theta[3],ell=theta[4],pa=theta[5])
    surface_density = theta[6] # [stars/Mpix]
    
    #THIS CAN BE USED TO HOLD SOME OF THE PARAMETERS FIXED
    # Default values for the other parameters
    #kwargs.update(ext=ERI_EXT_PIX,ell=ERI_ELL)

    #CHANGE KERNEL CALLED HERE TO USE A DIFFERENT FUNCTIONAL FORM FOR THE SURFACE DENSITY OF THE GALAXY
    # The kernel in pixel coordinates
    pdf = plummer_kernel(XX,YY,**kwargs)

    # Calculate the model predicted counts in each pixel
    binarea = XDEL*YDEL
    model_counts = richness * pdf * binarea + (1e-6 * surface_density * binarea)
    #print('XX shape',XX.shape)
    return model_counts

def lnlike(theta, x, y, mask):
    """ Likelihood function
    Parameters
    ----------
    theta : model parameter array (richness,lon,lat,ext,ell,pa,sd)
    x: x-coordinate of data
    y: y-coordinate of data
    
    Returns
    -------
    lnlike: log-likelihood
    """
    # Calculate the data counts and model predicted counts in each pixel bin
    data_counts = data(x,y)
    model_counts = model(theta)
    #print('MODEL COUNTS SHAPE',model_counts.shape)
    #print(data_counts.shape)
    # Apply the mask to the data and model. This selects only pixels
    # in the image for calculating the likelihood.
    idx = np.where(mask > 0)
    data_counts_masked = data_counts[idx]
    model_counts_masked = (model_counts * mask)[idx]
    
    # Evaluate Equation C2 from Drlica-Wagner et al. 2020 (1912.03302; ignore k! term)
    lnlike = np.sum(-model_counts_masked + data_counts_masked * np.log(model_counts_masked))
    return lnlike

def lnprior(theta):
    """ The log-prior. Add whatever you want here... 
    
    Parameters
    ----------
    theta : model parameters

    Returns
    -------
    lnprior : log-prior
    """
    #PRIORS FOR EACH FITTED PARAMETER
    rich1,lon1,lat1,ext1,ell1,pa1,sd1 = theta[0],theta[1],theta[2],theta[3],theta[4],theta[5],theta[6]
    if not (BOUNDS['rich'][0] <= rich1 <= BOUNDS['rich'][1]):  return np.inf
    if not (BOUNDS['lon'][0]  <= lon1  <= BOUNDS['lon'][1]): return np.inf
    if not (BOUNDS['lat'][0]  <= lat1  <= BOUNDS['lat'][1]): return np.inf
    if not (BOUNDS['ext'][0]  <= ext1  <= BOUNDS['ext'][1]): return np.inf
    if not (BOUNDS['ell'][0]  <= ell1  <= BOUNDS['ell'][1]): return np.inf
    if not (BOUNDS['pa'][0]   <= pa1   <= BOUNDS['pa'][1]): return np.inf
    if not (BOUNDS['sd'][0]   <= sd1   <= BOUNDS['sd'][1]): return np.inf
    return 0


def lnprob(theta, x, y, mask):
    """ The log-probability = lnlike + lnprob 

    Parameters
    ----------
    theta : the model parameter vector
    x     : x-coord of the data
    y     : y-coord of the data
    
    Returns
    -------
    lnprob : log-probability
    """
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, x, y, mask)

###########################
###### Simulation #########
###########################

def model_simulate_dwarf(theta, mask):
    """Generate simulated data by drawing a Poisson sample of the model.

    Parameters
    ----------
    theta : model parameters
    mask  : weight mask

    Returns
    -------
    x,y : x,y [pix] coordinates of simulated stars
    """
    # Generate model predicted counts
    model_counts = model(theta)
    model_counts_masked = model_counts*mask

    # Draw a Poisson sample of those counts
    sim_counts = poisson.rvs(model_counts_masked)

    # Index of pixels that contain objects
    idx = np.where(sim_counts)

    # x,y coordinates of those objects
    x = np.repeat(XX[idx],sim_counts[idx])
    y = np.repeat(YY[idx],sim_counts[idx])

    return x, y
