import numpy as np
import exoplanet as xo
import pymc3 as pm
import pymc3_ext as pmx
import arviz as az
import theano.tensor as tt
import decimal

import scipy.ndimage as sn
from scipy.stats import median_abs_deviation
from astropy.stats import sigma_clip
from celerite2.theano import terms, GaussianProcess
from itertools import chain, combinations
from ldtk import TabulatedFilter, LDPSetCreator

from .io_utils import load_phot_data
from .plot_utils import trace_plot, corner_plot, plot_aperture_opt, \
    plot_quickfit, plot_covariates, plot_initial_map, tripleplot,\
    plot_outlier_rejection, plot_white_light_curves, tripleplot_joint,\
    plot_tess, doubleplot, plot_initial_map_tess, doubleplot_joint

def clean_up(x, ys, yerrs, compars, weight_guess, cutoff_frac,
    end_num, filter_width, sigma_cut, plot = False, plot_dir = None,
    verbose = True, tag = ''):
    """
    Cleans up photometry by sigma clipping and removing any data during large flux dips

    Parameters
    ----------
    x : numpy.ndarray
        Array of time values for the photometric flux measurements.
    ys : numpy.ndarrays
        Arrays of flux values for the photometry measurements of the target and comparison stars at each aperture size.
    yerrs : numpy.ndarrays
        Arrays of error values on the aperture photometry measurements.
    compars : numpy.ndarray
        Arrays of flux values for the photometry measurements on the comparison stars.
    weight_guess : numpy.ndarray
        Initial guesses for the weights of all the comparison stars to be used in the fit. Default is equal weighting for each comparsion star.
    cutoff_frac : float
        Minimum flux value below which the photometry is clipped.
    end_num : int
        The number of data points at the end of the data set to exclude.
    filter_width : int
        The number of data points to be used for the window length of the median filter.
    sigma_cut : int
        The sigma threshold to be used for clipping data with the median filter.
    plot : boolean, optional
        If set to true, a plot for the outlier rejection from median filtering will be generated. The default is False.
    plot_dir : str, optional
        Path to the directory in which the outlier rejection image will be stored. Needs to be included if plot =True. The default is None.

    Returns
    -------
    masked x
        Masked array of time values for the photometric flux measurements.
    masked ys
        Masked arrays of flux values for the photometry measurements of the target and comparison stars at each aperture size.
    masked yerrs
        Masked arrays of error values on the aperture photometry measurements.
    masked compars
        Masked arrays of flux values for the photometry measurements on the comparison stars.
    full_mask : numpy.ndarray
        Array of booleans representing which measurements to use in the final light curves.

    """

    if verbose:
        print(f"Using a sigma threshold of {sigma_cut}...")
    #n sigma outlier rejection against a median filter
    full_mask = np.ones(x.shape, dtype = 'bool')
    quick_detrend = ys[0]/weight_guess.dot(compars)
    median_filter = sn.median_filter(quick_detrend, filter_width, 
            mode = 'constant', cval = 1.)
    filt = quick_detrend / median_filter
    masked_arr = sigma_clip(filt, sigma = sigma_cut,
        stdfunc = median_abs_deviation)
    full_mask = ~masked_arr.mask
    if verbose:
        print(f"Clipped {sum(~full_mask)} points in initial sigma clipping")

    #flux cutoff for very rapidly varying light curve
    cutoff = cutoff_frac*max(ys[0])
    mask = ys[0] > cutoff
    full_mask &= mask
    if verbose:
        print(f"Clipped {sum(~mask)} points in flux cutoff")

    #cut nans
    mask = np.isnan(ys[0]) | np.isnan(yerrs[0])
    full_mask &= ~mask
    if verbose:
        print(f"Clipped {sum(mask)} points in nan clipping")

    #loosing the last few data
    mask = np.ones(x.shape, dtype = 'bool')
    if end_num > 0:
        mask[-end_num:] = False
    full_mask &= mask
    if verbose:
        print(f"Clipped {sum(~mask)} points in edge cutoff")

    n_reject = sum(~full_mask)
    clip_perc = n_reject/len(full_mask)*100.
    if verbose:
        print(f"Clipped {clip_perc}% of the data")
    if plot and plot_dir is not None:
        plot_outlier_rejection(plot_dir, x, quick_detrend,
            median_filter, full_mask, tag)
    return x[full_mask], ys[:,full_mask], yerrs[:,full_mask], \
        compars[:, full_mask], full_mask

def clean_after_map(x, ys, yerrs, compars, map_soln, sigma_cut, plot = True,
    plot_dir = None, extra_tag = '', verbose = True, n_model = ''):
    """
    Cleaning the light curves by clipping any data that significantly deviates from the initial MAP fit.

    Parameters
    ----------
    x : numpy.ndarray
        Array of time values for the photometric flux measurements.
    ys : numpy.ndarrays
        Arrays of flux values for the photometry measurements of the target and comparison stars at each aperture size.
    yerrs : numpy.ndarrays
        Arrays of error values on the aperture photometry measurements.
    compars : numpy.ndarray
        Arrays of flux values for the photometry measurements on the comparison stars.
    map_soln : dict
        Dictionary of arrays for all the fit parameters from the initial MAP solution using exoplanet.
    sigma_cut : int
        The sigma threshold to be used for clipping data with the median filter.
    plot : boolean, optional
        If set to true, a plot for the outlier rejection from median filtering will be generated. The default is False.
    plot_dir : str, optional
        Path to the directory in which the outlier rejection image will be stored. Needs to be included if plot =True. The default is None.

    Returns
    -------
    masked x
        Masked array of time values for the photometric flux measurements.
    masked ys
        Masked arrays of flux values for the photometry measurements of the target and comparison stars at each aperture size.
    masked yerrs
        Masked arrays of error values on the aperture photometry measurements.
    masked compars
        Masked arrays of flux values for the photometry measurements on the comparison stars.
    full_mask : numpy.ndarray
        Array of booleans representing which measurements to use in the final light curves.

    """
    if n_model != '':
        n_model = '_' + n_model

    resid = ys[0] / map_soln[f'full_model{n_model}']
    masked_arr = sigma_clip(resid, sigma = sigma_cut,
        stdfunc = median_abs_deviation)
    full_mask = ~masked_arr.mask

    n_reject = sum(~full_mask)
    clip_perc = n_reject/len(full_mask)*100.

    if plot and plot_dir is not None:
        plot_outlier_rejection(plot_dir, x, resid, np.ones(x.shape),
            full_mask, tag = f'_afterMAP_{extra_tag}{n_model}')

    if verbose:
        print(f"Clipped {n_reject} points out of {x.shape} after MAP fitting")
        print(f"corresponding to {clip_perc}% of the data")

    return x[full_mask], ys[:,full_mask], yerrs[:,full_mask], \
        compars[:, full_mask], full_mask

def quick_aperture_optimize(dump_dir, plot_dir, apertures,
    flux_cutoff = 0., end_num = 0, filter_width = 31,
    sigma_cut = 5):
    """
    Quickly detrend the light curves and calculate the final RMS deviations for all the apertures included in the "apertures" range to deterrmine the optimal aperture.

    Parameters
    ----------
	dump_dir : string
			Path to directory in which the saved covariates background values and pickled raw photometry values will be stored
    plot_dir : str
        Path to the directory in which the aperture optimization image will be stored.
    apertures : range
        Range of target apertures to test for aperture photometry measurements.
    flux_cutoff : float, optional
        Minimum flux value below which the photometry is clipped.. The default is 0.
    end_num : int
        The number of data points at the end of the data set to exclude.
    filter_width : int
        The number of data points to be used for the window length of the median filter.
    sigma_cut : int
        The sigma threshold to be used for clipping data with the median filter.

    Returns
    -------
    best_ap : int
        The target aperture that resulted in the lowest RMS deviation for the final light curve.

    """
    print("Running quick aperture optimization...")
    rmses = []
    for i in apertures:
        x, ys, yerrs, _, _, _, _, _ = \
            load_phot_data(dump_dir, i)
        compars= ys[1:]
        weight_guess = np.array([1./len(compars)]*len(compars))

        x, ys, yerrs, compars, _ = clean_up(
            x, ys, yerrs, compars, weight_guess, flux_cutoff,
            end_num, filter_width, sigma_cut, verbose = False)

        quick_detrend = ys[0]/weight_guess.dot(compars)
        median_filter = sn.median_filter(quick_detrend, filter_width, 
            mode = 'constant', cval = 1.)
        filt = quick_detrend / median_filter
        rmses.append(np.std(filt)/len(x))

    plot_aperture_opt(plot_dir, apertures, rmses)
    best_ap = apertures[np.argmin(rmses)]
    print(f"Complete! Optimal aperture is {best_ap} pixels.")

    return best_ap

def get_covariates(bkgs_init, centroid_x_init, centroid_y_init, airmass, widths,
    background_mode, mask):
    """
    Generates a dictionary of arrays where the data for all possible flux covariates is stored. 
    
    Parameters
    ----------
    bkgs_init : numpy.ndarray
        Array of scale factors for the background brightness relative to the initial sky background dither.
    centroid_x_init : numpy.ndarrays
        Arrays of the x pixel location of the target and comparsion stars in each image.
    centroid_y_init : numpy.ndarrays
        Arrays of the y pixel location of the target and comparsion stars in each image.
    airmass : numpy.ndarray
        Array of the airmass values in each image.
    widths : numpy.ndarrays
        Arrays of the PSF widths of the target and comparsion stars in each image.
    background_mode : float
        The method of sky background generation. Options are 'helium','global', and 'median'
    mask : numpy.ndarray
        Array of booleans to determine which images are to be included for the final photometry.

    Returns
    -------
    covariate_dict : dict
         A dictionary of arrays for all covariate data from each science image. Covariates include: 'psf_width','airmass','background','water_proxy','d_from_med','x_cent','y_cent'.

    """

    d_from_med_init = gen_d_from_med(centroid_x_init, centroid_y_init)

    if background_mode == 'helium':
        background = None
        water_proxy = gen_water_proxy(bkgs_init)[mask]
    else:
        water_proxy = None
        background = bkgs_init[mask]
    
    covariate_dict = {
            'x_cent': np.array(
                centroid_x_init[0][mask],dtype = float),
            'y_cent': np.array(
                centroid_y_init[0][mask],dtype = float),
            'd_from_med': np.array(
                d_from_med_init[mask],dtype=float),
            'water_proxy': np.array(water_proxy,dtype = float),
            'airmass' : np.array(airmass[mask],dtype = float),
            'psf_width' : np.array(widths[0][mask], dtype = float),
            'background' : np.array(background, dtype = float)}
    return covariate_dict

def crossmatch_covariates(covariates, covariate_dict):
    """
    Retain the specific desired coravariates from all the covariate data.

    Parameters
    ----------
    covariates : list
        A list of strings representing all the desired covariates to be retained and used in the final light curve fitting.
    covariate_dict : dict
        A dictionary of arrays for all covariate data from each science image. Covariates include: 'psf_width','airmass','background','water_proxy','d_from_med','x_cent','y_cent'

    Returns
    -------
    list
        The covariate_dict including only the data selected by the user in the covariates list.

    """
    return [covariate_dict[cov] for cov in covariates]

def fit_lightcurve(dump_dir, plot_dir, best_ap, background_mode,
    covariate_names, texp, r_star_prior, t0_prior, period_prior,
    a_rs_prior, b_prior, jitter_prior, ror_prior = None,
    fpfs_prior = None, tune = 1000, 
    draws = 1500, target_accept = 0.99, phase = 'primary',
    ldc_val = None, bin_time = 10., 
    flux_cutoff = 0., end_num = 0, filter_width = 31, refit = True,
    sigma_cut = 5, gp = False, sigma_prior = None, rho_prior = None,
    ror_nominal = None, return_bic = True, fixed_jitter = None,
    tess_x = None, tess_y = None, tess_yerr = None, tess_texp = None,
    tess_t0_prior = None, sigma_cut_init = 10):
    """
    The main function used to fit the target light curve and find the best-fit transit model parameters.

    Parameters
    ----------
	dump_dir : string
			Path to directory in which the saved covariates background values and pickled raw photometry values are stored.
    plot_dir : str
        Path to the directory in which the light curve fitting plots will be stored.
    best_ap : int
        The target aperture that resulted in the lowest RMS deviation for the final light curve.
    background_mode : float
        The method of sky background generation. Options are 'helium','global', and 'median'
    covariate_names : list
        list of floats for the desired covariates to be included in the fit.
    texp : float
        Science image exposure time in days.
    r_star_prior : tuple
        Prior on the stellar radius in solar radii.
    t0_prior : tuple
        Prior on the transit midtime in bjd.
    period_prior : tuple
        Prior on the planet's orbital period.
    a_rs_prior : tuple
        Prior on the planet semimajor axis to stellar radius ratio.
    b_prior : tuple
        Prior on the planet impact parameter.
    jitter_prior : tuple
        Prior on the parameter for the extra jitter in the light curve (a systematic noise parameter).
    ror_prior : tuple, optional
        Prior on the planet to star radius ratio. The default is None.
    fpfs_prior : tuple, optional
        Prior on the secondary eclipse depth. The default is None.
    tune : int
        The number of tuning steps for each chain in the sample. The default is 1000.
    draws : int
        The number of draw iterations for each chain in the final posterior sample. The default is 1500.
    target_accept : float
        The target acceptance probability for each chain. Must be < 1, the default is 0.99.
    phase : float, optional
        The phase of the event to be modeled in the light curve. The default is 'primary'.
    ldc_val : list, optional
        If the limb darkening paramaters are already known with high confidence, they can be set here and will not be included in the fit. The default is None.
    bin_time : float, optional
        The time in minutes to bin the final light curve down to for plotting. The default is 5.
    flux_cutoff : float, optional
        Minimum flux value below which the photometry is clipped.. The default is 0.
    end_num : int
        The number of data points at the end of the data set to exclude. The default is 0.
    filter_width : int
        The number of data points to be used for the window length of the median filter. The default is 31.
    sigma_cut : int
        The sigma threshold to be used for clipping data with the median filter. The default is 5.
    gp : bool, optional
        Whether or not to include a gp in the light curve fit. The default is False.
    sigma_prior : tuple, optional
        If gp = True, this is the prior on the sigma parameter for the gp. The default is None.
    rho_prior : tuple, optional
        If gp = True, this is the prior on the rho parameter for the gp. The default is None.

    Returns
    -------
    None.

    """
    
    fit_tess = (tess_x is not None) and (tess_y is not None) and \
        (tess_yerr is not None) and (tess_texp is not None)

    x_init, ys_init, yerrs_init, bkgs_init, centroid_x_init, \
        centroid_y_init, airmass, widths = \
        load_phot_data(dump_dir, best_ap)
    compars_init = ys_init[1:]
    weight_guess_init = np.array([1./len(compars_init)]*len(compars_init))

    x, ys, yerrs, compars, mask = clean_up(x_init, ys_init, yerrs_init,
            compars_init, weight_guess_init, flux_cutoff,
            end_num, filter_width, sigma_cut_init, plot = True,
            plot_dir = plot_dir)

    cov_dict = get_covariates(bkgs_init, centroid_x_init, centroid_y_init,
        airmass, widths, background_mode, mask)
    covs = crossmatch_covariates(covariate_names, cov_dict)
    plot_quickfit(plot_dir, x, ys, yerrs)
    plot_covariates(plot_dir, x, covariate_names, covs)

    weight_guess = np.array([1./compars.shape[0]]*compars.shape[0] + \
        [0]*(len(covs) + 1))
    compars = np.vstack((compars, *covs))
    compars = np.vstack((compars, x - np.mean(x)))

    mask = np.zeros(x.shape, dtype = bool)
    i = 0
    #refitting the MAP with iterative sigma clipping until no new outliers found
    while sum(~mask) > 0:
        i += 1
        if i == 1:
            print("Constructing model...")
        else:
            print("Refitting MAP...")
        model, map_soln = make_model(x, ys, yerrs, compars, weight_guess,
            texp, r_star_prior, t0_prior, period_prior,
            a_rs_prior, b_prior, jitter_prior, phase, ror_prior,
            fpfs_prior, ldc_val, gp, sigma_prior, rho_prior,
            ror_nominal, tess_x, tess_y, tess_yerr, tess_texp, 
            fixed_jitter = fixed_jitter, tess_t0_prior = tess_t0_prior)
        plot_initial_map(plot_dir, x, ys, yerrs, compars, map_soln, gp, 
            fixed_jitter)
        if fit_tess:
            plot_initial_map_tess(plot_dir, tess_x, tess_y, tess_yerr, map_soln,
                    tess_t0_prior)
        print("Initial MAP found!")
        if refit:
            x, ys, yerrs, compars, mask = clean_after_map(x, ys, yerrs,
                compars, map_soln, sigma_cut, plot = True,
                plot_dir = plot_dir, extra_tag = i)
        else:
            mask = np.ones(x.shape, dtype = bool)
    print("MAP found!")
    plot_white_light_curves(plot_dir, x, ys)

    print("Sampling posterior...")
    trace = sample_model(model, map_soln, tune, draws, target_accept)
    trace.posterior.to_netcdf(f'{dump_dir}posterior.nc', engine='scipy')
    print("Sampling complete!")
    new_map, ll = get_new_map(trace, get_log_like = True)
    bic = get_bic(x, ll, new_map[f'weights'])
    systematics = np.dot(np.array(new_map[f"weights"]), compars)
    vec = x - np.mean(x)
    if gp:
        baseline = np.array(new_map['gp_pred'])
    else:
        baseline = 0.
    detrended_data = (ys[0] - baseline)/systematics
    if fixed_jitter is None:
        true_err = np.sqrt(yerrs[0]**2 + float(new_map[f"jitter"])**2) 
    else:
        true_err = np.sqrt(yerrs[0]**2 + fixed_jitter**2) 
    plot_nominal = ror_nominal is not None
    summary, varnames = gen_summary(dump_dir, trace, phase, ldc_val, gp,
        plot_nominal, fit_tess = fit_tess, fixed_jitter = fixed_jitter,
        tess_t0_prior = tess_t0_prior)
    gen_latex_table(dump_dir, summary)
    gen_lightcurve_table(dump_dir, x, detrended_data, true_err)

    print("Making corner and trace plots...")
    trace_plot(plot_dir, trace, varnames)
    corner_plot(plot_dir, trace, varnames)
    print("Visualizing fit...")
    t0 = float(new_map["t0"]) \
        if t0_prior[0] != 'fixed' else t0_prior[1]
    p = float(new_map['period']) \
        if period_prior[0] != 'fixed' else period_prior[1]
    doubleplot(plot_dir, dump_dir, x, ys, yerrs, compars, detrended_data,
        new_map, trace, texp, t0, p, bin_time = bin_time,
        phase = phase, gp = gp, fixed_jitter = fixed_jitter, 
        plot_nominal = plot_nominal, baseline = baseline, fit_tess = fit_tess)
 
    tripleplot(plot_dir, dump_dir, x, ys, yerrs, compars, detrended_data,
        new_map, trace, texp, t0, p, bin_time = bin_time,
        phase = phase, gp = gp, fixed_jitter = fixed_jitter,
        plot_nominal = plot_nominal, baseline = baseline, fit_tess = fit_tess)
    if fit_tess:
        if tess_t0_prior is not None:
            t0 = float(new_map["t0_tess"])
        plot_tess(plot_dir, tess_x, tess_y, tess_yerr, new_map, trace,
            tess_texp, t0, p, tess_t0_prior, bin_time)
    print("Fitting complete!")
    if return_bic:
        jit = new_map[f"jitter"] if fixed_jitter is None else fixed_jitter
        return bic, len(x), float(jit), trace
    else:
        return None    

def gen_summary(dump_dir, trace, phase, ldc_val, gp = False,
    plot_nominal = False, joint = 1, fit_tess = False, 
    fixed_jitter = None, tess_t0_prior = None):
    """
    Generates a csv table with summary information for all the parameters of the fit.

    Parameters
    ----------
    dump_dir : str
        Path to directory in which the saved covariates, background values, pickled raw photometry arrays, and fit summary data arrer stored.
    trace : arviz.data.inference_data.InferenceData
        Information on the trace of all fitted parameters used in the model sampling, including the posterior values, log_likelihood, sample_stats, and observed_data.
    phase : float
        The phase of the event to be modeled in the light curve. Typically 'primary', but for secondary eclipse could be 'secondary'.
    ldc_val : list, optional
        If the limb darkening paramaters are already known with high confidence, they can be set here and will not be included in the fit.
    gp : bool, optional
        Whether or not to include a gp in the light curve fit. The default is False.

    Returns
    -------
    summary : pandas.core.frame.DataFrame
        pandas dataFrame containing summary information on the fit parameter distributions.
    varnames : list
        List of strings for all the variable names included in the fit.

    """
    if phase == 'primary':
        varnames = ['t0', 'period', 'a_rs', 'b', 'ror']
    else:
        varnames = ['t_second', 'period', 'a_rs', 'b', 'fpfs',
            'r_star', 'jitter', 'weights']
 
    if joint == 1:
        if fixed_jitter is None:
            varnames += ['jitter', 'weights']
        else:
            varnames += ['weights']
    else:
        varnames += [f'jitter_{i}' for i in range(joint)]
        varnames += [f'weights_{i}' for i in range(joint)]
    if ldc_val is None:
        varnames += ['u']
    if gp:
        varnames += ['sigma', 'rho']
    if plot_nominal:
        varnames += ['mid_transit_excess_depth']
    if fit_tess:
        varnames += ['u_tess', 'ror_tess', 'tess_error_scaling']
    if tess_t0_prior is not None:
        varnames += ['t0_tess']

    func_dict = {
        "16%": lambda x: np.percentile(x, 16),
        "50%": lambda x: np.percentile(x, 50),
        "84%": lambda x: np.percentile(x, 84),
        "95%": lambda x: np.percentile(x, 95)
    }

    summary = az.summary(trace, var_names = varnames,
        stat_funcs = func_dict, round_to = 16,
        kind = 'all')
    summary.to_csv(f'{dump_dir}fit_summary.csv')

    return summary, varnames

def get_bic(x, log_like, weights):
    k = len(weights)
    lnn = np.log(len(x))
    return (k+7)*lnn - 2*log_like

def calculate_all_bics(dump_dir, img_dir, best_ap, background_mode, 
    all_covar_names, texp, r_star_prior, t0_prior, period_prior, a_rs_prior,
    b_prior, jitter_prior, phase, ror_prior, tune, draws, target_accept,
    sigma_cut, filter_width, ror_nominal, ldc_val, gp = False,
    sigma_prior = None, rho_prior = None, sigma_cut_init = 4, end_num = 0):
    bics = []
    ndatas = []
    traces = {}
    covar_combos = [[]]
    s = list(all_covar_names)
    covar_combos = list(chain.from_iterable(
        combinations(s, r) for r in range(len(s)+1)))
    labels = []
    print("Measuring jitter term...")
    _,_,jitter,_=fit_lightcurve(dump_dir, img_dir, best_ap, background_mode, 
        [], texp, r_star_prior, t0_prior, period_prior,
        a_rs_prior, b_prior, jitter_prior, phase = phase, ror_prior = ror_prior,
        tune = tune, draws = draws, target_accept = target_accept,
        sigma_cut = sigma_cut, sigma_cut_init = sigma_cut_init,
        filter_width = filter_width, end_num = end_num,
        ror_nominal = ror_nominal, refit = False, ldc_val = ldc_val, gp = gp,
        sigma_prior = sigma_prior, rho_prior = rho_prior)
    print("Measured jitter: ", jitter*1e6, " ppm")
    print("Testing covariate combinations: ", covar_combos)
    for i, covariate_names in enumerate(covar_combos):
        bic, ndata, _, trace = fit_lightcurve(dump_dir, img_dir, best_ap,
            background_mode, covariate_names, texp, r_star_prior, t0_prior,
            period_prior, a_rs_prior, b_prior, jitter_prior, phase = phase, 
            ror_prior = ror_prior, tune = tune, draws = draws,
            target_accept = target_accept, sigma_cut = sigma_cut,
            sigma_cut_init = sigma_cut_init, end_num = end_num,
            filter_width = filter_width, ror_nominal = ror_nominal,
            refit = False, ldc_val = ldc_val, gp = gp, 
            sigma_prior = sigma_prior, rho_prior = rho_prior,
            fixed_jitter = jitter)
        bics.append(bic)
        ndatas.append(ndata)
        traces[i] = trace
        labels.append(i)
    if np.all(np.array(ndatas) == ndatas[0]):
        print("All datasets in the BIC comparison have the same length.")
    else:
        print("WARNING...not all datasets in the BIC comparison have the " + \
            "same length, so you're comparing apples to oranges with the " + \
            "likelihoods. Here are the dataset sizes: ", ndatas)

    df_comp = az.compare(traces, ic = 'waic')
    df_comp.to_csv(dump_dir + 'WAIC_comparison.csv')
    df_comp = az.compare(traces, ic = 'loo')
    df_comp.to_csv(dump_dir + 'LOO_comparison.csv')

    print("Completed testing covariate combinations!")
    print(f'Minimum deltaBIC is {min(bics - bics[0])} for' +  \
        f'{covar_combos[np.argmin(bics)]}')
    write_bics(dump_dir, covar_combos, bics)
    print("Refitting final...")
    fit_lightcurve(dump_dir, img_dir, best_ap, background_mode, 
        covar_combos[np.argmin(bics)], texp, r_star_prior, t0_prior,
        period_prior, a_rs_prior, b_prior, jitter_prior, phase = phase,
        ror_prior = ror_prior, tune = tune, draws = draws,
        target_accept = target_accept, sigma_cut = sigma_cut,
        filter_width = filter_width, ror_nominal = ror_nominal,
        ldc_val = ldc_val, gp = gp, sigma_prior = sigma_prior,
        rho_prior = rho_prior, end_num = end_num)
    return None

def write_bics(dump_dir, covar_combos, bics):
    f = open(f'{dump_dir}bics.txt', 'w')
    f.write(f'Covariate List;BIC\n')
    for covar, bic in zip(covar_combos, bics):
        f.write(f'{covar};{bic - bics[0]}\n')
    f.close()
    return None

def gen_latex_table(dump_dir, summary):
    """
    Generates a latex table for the information on all the fit parameter distributions.

    Parameters
    ----------
    dump_dir : str
        Path to directory in which the saved covariates, background values, pickled raw photometry arrays, and fit summary data are stored
    summary : pandas.core.frame.DataFrame
        pandas dataFrame containing summary information on the fit parameter distributions.

    Returns
    -------
    None.

    """
    f = open(f'{dump_dir}latex_format_table.txt', 'w')
    for index, row in summary.iterrows():
        med = row['50%']
        lo = row['50%'] - row['16%']
        hi = row['84%'] - row['50%']
        decs = ["{:0.2g}".format(x) for x in [med, lo, hi]]
        places = [-decimal.Decimal(x).as_tuple().exponent for x in decs]
        dec = max(places)

        fmtstr = "{:0."+str(dec)+"f}"
        medstr = fmtstr.format(med)
        lostr = fmtstr.format(lo) 
        histr = fmtstr.format(hi)
        printstr = f'${medstr}_{{-{lostr}}}^{{+{histr}}}$'

        print(index, '&', printstr, file = f)
    f.close()    
    return None

def gen_lightcurve_table(dump_dir, x, detrended, true_errs):
    """
    Saves the final detrended light curve data to a csv table.

    Parameters
    ----------
	dump_dir : string
			Path to directory in which the saved covariates background values and pickled raw photometry values are stored.
    x : numpy.ndarray
        Array of time values for the flux measurements.
    detrended : numpy.ndarray
        Final detrended array of flux values for the target.
    true_errs : numpy.ndarray
        Final array of errors on the flux values for the target.

    Returns
    -------
    None.

    """
    fname = f'{dump_dir}detrended_light_curve.csv'
    f = open(fname, 'w')
    print('BJD,detrended_flux,err', file = f)
    for xi, yi, yerri in zip(x, detrended, true_errs):
        dec = "{:0.2g}".format(yerri)
        places = -decimal.Decimal(dec).as_tuple().exponent
        fmtstr = "{:0."+str(places)+"f}"
        xistr = fmtstr.format(xi)
        yistr = fmtstr.format(yi)
        yerristr = fmtstr.format(yerri)
        print(f'{xi},{yi},{yerri}', file = f)
    f.close()
    return None

def get_new_map(trace, get_log_like = False):
    """
    Returns the new MAP model from exoplanet after sampling the posterior.

    Parameters
    ----------
    trace : arviz.data.inference_data.InferenceData
        Information on the trace of all fitted parameters used in the model sampling, including the posterior values, log_likelihood, sample_stats, and observed_data.

    Returns
    -------
    new_map : dict
        Dictionary of arrays for all the fit parameters from the final MAP solution after sampling the posterior with exoplanet and optimizing.

    """
    dat = np.array(trace.sample_stats.lp)
    ind = np.unravel_index(dat.argmax(), dat.shape)
    new_map = trace.posterior.isel(chain=ind[0], draw = ind[1])
    if get_log_like:
        ll = np.array(trace.log_likelihood.obs)
        return new_map, sum(ll[ind])
    return new_map

def sample_model(model, map_soln, tune, draws, target_accept):
    """
    Samples the posterior distribution for the exoplanet light curve model.

    Parameters
    ----------
    model : pymc3.model.Model
        The pymc3 model for the light curve constrructed with exoplanet using the initial guess parameters from the user.
    map_soln : dict
        Dictionary of arrays for all the fit parameters from the initial MAP solution using exoplanet.
    tune : int
        The number of tuning steps for each chain in the sample.
    draws : int
        The number of draw iterations for each chain in the final posterior sample.
    target_accept : float
        The target acceptance probability for each chain. Must be < 1, a good initial guess is 0.99 assuming you have a good number of tuning steps.

    Returns
    -------
    trace : arviz.data.inference_data.InferenceData
        Information on the trace of all fitted parameters used in the model sampling, including the posterior values, log_likelihood, sample_stats, and observed_data.

    """
    with model:
        trace = pmx.sample(
            tune=tune,
            draws=draws,
            start=map_soln,
            return_inferencedata = True,
            target_accept=target_accept
        )
        return trace


def unpack_prior(name, prior_tuple):
    """
    Unpacks prior tuples to return the prior function as specified in the tuple

    Parameters
    ----------
    name : float
        The variable name.
    prior_tuple : tuple
        The prior tuple including an initial float to specify whether it is a 'uniform' or 'normal' prior, and then two floats to specify the prior bounds.

    Returns
    -------
    Prior function
        The corresponding function to the prior tuple input.

    """
    if prior_tuple[0] == 'fixed':
        return prior_tuple[1]
    func_dict = {'normal': pm.Normal,
        'uniform': pm.Uniform}
    func, a, b = prior_tuple
    if func == 'uniform':
        testval = (a + b)/2
    else:
        testval = a
    return func_dict[func](name, a, b, testval = testval, shape = 1)

def make_model(x, ys, yerrs, compars, weight_guess, texp, r_star_prior,
    t0_prior, period_prior, a_rs_prior, b_prior,
    jitter_prior, phase = 'primary', ror_prior = None, fpfs_prior = None,
    ldc_val = None, gp = False,
    sigma_prior = None, rho_prior = None, 
    ror_nominal = None, tess_x = None, tess_y = None, tess_yerr = None,
    tess_texp = None, fixed_jitter = None, tess_t0_prior = None,
    nominal_bounds = (-0.2, 0.2)):
    """
    Generates the light curve model with exoplanet.

    Parameters
    ----------
    x : numpy.ndarray
        Array of time values for the photometric flux measurements.
    ys : numpy.ndarrays
        Arrays of flux values for the photometry measurements of the target and comparison stars at each aperture size.
    yerrs : numpy.ndarrays
        Arrays of error values on the aperture photometry measurements.
    compars : numpy.ndarray
        Arrays of flux values for the photometry measurements on the comparison stars.
    weight_guess : numpy.ndarray
        Initial guesses for the weights of all the comparison stars to be used in the fit. Default is equal weighting for each comparsion star.
    texp : float
        Science image exposure time in days.
    r_star_prior : tuple
        Prior on the stellar radius in solar radii.
    t0_prior : tuple
        Prior on the transit midtime in bjd.
    period_prior : tuple
        Prior on the planet's orbital period.
    a_rs_prior : tuple
        Prior on the planet semimajor axis to stellar radius ratio.
    b_prior : tuple
        Prior on the planet impact parameter.
    jitter_prior : tuple
        Prior on the parameter for the extra jitter in the light curve (a systematic noise parameter).
    phase : float, optional
        The phase of the event to be modeled in the light curve. The default is 'primary'.
    ror_prior : tuple, optional
        Prior on the planet to star radius ratio. The default is None.
    fpfs_prior : tuple, optional
        Prior on the secondary eclipse depth. The default is None.
    ldc_val : list, optional
        If the limb darkening paramaters are already known with high confidence, they can be set here and will not be included in the fit. The default is None.
    gp : bool, optional
        Whether or not to include a gp in the light curve fit. The default is False.
    sigma_prior : tuple, optional
        If gp = True, this is the prior on the sigma parameter for the gp. The default is None.
    rho_prior : tuple, optional
        If gp = True, this is the prior on the rho parameter for the gp. The default is None.

    Returns
    -------
    model : pymc3.model.Model
        The pymc3 model for the light curve constrructed with exoplanet.
    map_soln : dict
        Dictionary of arrays for all the best-fit parameters for the MAP solution from the maximized posterior.

    """
    ##currently doing circular orbits ONLY
    fit_tess = (tess_x is not None) and (tess_y is not None) and \
        (tess_yerr is not None) and (tess_texp is not None)

    with pm.Model() as model:
        if ldc_val:
            star = xo.LimbDarkLightCurve(ldc_val)
        else:
            u = xo.distributions.QuadLimbDark("u")
            star = xo.LimbDarkLightCurve(u)
        r_star = unpack_prior('r_star', r_star_prior)

        period = unpack_prior('period', period_prior)
        t0 = unpack_prior('t0', t0_prior)
        #if phase == 'primary':
        #t = t0
        #else:
        t = pm.Deterministic("t_second", t0)

        a_rs = unpack_prior('a_rs', a_rs_prior)
        b = unpack_prior('b', b_prior)
        if phase == 'primary':
            ror = unpack_prior('ror', ror_prior)
        else:
            fpfs = unpack_prior('fpfs', fpfs_prior)
            ror = np.sqrt(fpfs)

        orbit = xo.orbits.KeplerianOrbit(period = period,
            t0 = t, b = b, a = a_rs*r_star, r_star = r_star)
        #lightcurve
        lightcurve = pm.Deterministic("light_curve", pm.math.sum(
            star.get_light_curve(orbit=orbit, r = ror*r_star,
            t = x, texp = texp), axis = -1) + 1.)

        arr = np.linspace(nominal_bounds[0], nominal_bounds[1], 1000)
        dummy_t = pm.Deterministic("dummy_t", t + arr)
        dummy_lightcurve = pm.Deterministic("dummy_light_curve",
            pm.math.sum(star.get_light_curve(orbit = orbit,
            r = ror*r_star, t = dummy_t, texp = texp), axis = -1) + 1) 
        if ror_nominal is not None or fit_tess:
            if fit_tess:
                u_tess = xo.distributions.QuadLimbDark("u_tess")
                star_tess = xo.LimbDarkLightCurve(u_tess)
                ror_tess = pm.Uniform('ror_tess', 0, 0.3, testval = 0.1)
                if tess_t0_prior is not None:
                    t0_tess = unpack_prior('t0_tess', tess_t0_prior)
                    dummy_t_tess = pm.Deterministic("dummy_t_tess",
                            t0_tess + arr)
                    orbit_tess = xo.orbits.KeplerianOrbit(period = period,
                        t0 = t0_tess, b = b, a = a_rs*r_star,
                        r_star = r_star)
                else:
                    orbit_tess = orbit
                    dummy_t_tess = dummy_t
                lightcurve_tess = pm.Deterministic("light_curve_TESS", 
                    pm.math.sum(star_tess.get_light_curve(orbit=orbit_tess, 
                    r = ror_tess*r_star, t = tess_x, texp = tess_texp),
                    axis = -1) + 1.)
                nominal_lightcurve = pm.Deterministic("light_curve_nominal",
                    pm.math.sum(star.get_light_curve(orbit = orbit_tess,
                    r = ror_tess*r_star, t = dummy_t_tess, texp = tess_texp),
                    axis = -1) + 1)
                mid_transit = pm.Deterministic("mid_transit_excess_depth",
                    -(tt.min(dummy_lightcurve) - tt.min(nominal_lightcurve)))
            else:
                nominal_lightcurve = pm.Deterministic("light_curve_nominal",
                    pm.math.sum(star.get_light_curve(orbit = orbit,
                    r = ror_nominal*r_star, t = dummy_t, texp = texp),
                    axis = -1) + 1)
                mid_transit = pm.Deterministic("mid_transit_excess_depth",
                    -(tt.min(dummy_lightcurve) - tt.min(nominal_lightcurve)))       

        #systematics
        comp_weights = pm.Uniform("weights", -2., 2.,
            testval = weight_guess, shape = len(weight_guess))
        systematics = pm.math.dot(comp_weights, compars)

        if fixed_jitter is None:
            jitter = unpack_prior('jitter', jitter_prior)
        else:
            jitter = fixed_jitter
        full_variance = yerrs[0]**2 + jitter**2

        if gp:
            y_gp = ys[0] - systematics*lightcurve
            sigma = unpack_prior('sigma', sigma_prior)
            rho = unpack_prior('rho', rho_prior)
            kernel = terms.Matern32Term(sigma = sigma, rho = rho)
            gp = GaussianProcess(kernel, t = x,
                diag = full_variance, quiet = True)
            gp.marginal(f"obs", observed = y_gp)
            pm.Deterministic(f"gp_pred", gp.predict(y_gp))

        else:
            full_model = systematics*lightcurve
            pm.Deterministic("full_model", full_model)
            if fit_tess:
                err_scale = pm.Uniform('tess_error_scaling', 0.5, 1.5,
                    testval = 1.)
                pm.Normal("tess_obs", mu = lightcurve_tess,
                    sd = tess_yerr*err_scale, observed = tess_y)  
            pm.Normal(f"obs", mu=full_model,
                sd=np.sqrt(full_variance), observed=ys[0])
        map_soln = model.test_point
        map_soln = pmx.optimize(map_soln)

        return model, map_soln


def gen_water_proxy(bkgs):
    """
    Generate an array of values for the absorption proxy for each image in the case of helium photometry.

    Parameters
    ----------
    bkgs : numpy.ndarray
        Array of scale factors for the background brightnesses for each image relative to the initial sky background frame.

    Returns
    -------
    absorption_proxy : nump.ndarray
        Array of values for the absorption proxy for each science image.

    """
    oh_2 = np.mean(bkgs[:,72:89], axis = 1) 
    oh_3 = np.mean(bkgs[:,180:190], axis = 1)
    oh_4 = np.mean(bkgs[:,201:210], axis = 1)
    
    emission_proxy = (oh_3 +oh_4)/2
    absorption_proxy = oh_2/emission_proxy
    absorption_proxy /= np.median(absorption_proxy)
    return absorption_proxy

def gen_d_from_med(centroid_x_init, centroid_y_init):
    """
    Calculate the total distance of the star centroids from the initial position on the detector using the x and y pixel offsets for each image.

    Parameters
    ----------
    centroid_x_init : numpy.ndarrays
        Arrays of the x pixel location of the target and comparsion stars in each image.
    centroid_y_init : numpy.ndarrays
        Arrays of the y pixel location of the target and comparsion stars in each image.

    Returns
    -------
    d_from_med_init : numpy.ndarrays
        Arrays of the radial distances of the target and comparsion star centroids relative to their initial location for each image.

    """
    med_x = np.median(centroid_x_init[0])
    med_y = np.median(centroid_y_init[0])
    d_from_med_init = np.sqrt((centroid_x_init[0] - med_x)**2 \
        + (centroid_y_init[0] - med_y)**2)
    return d_from_med_init




def fit_lightcurve_joint(dump_dirs, plot_dir, best_aps, background_modes,
    covariate_names_list, texps, r_star_prior, t0_prior, period_prior,
    a_rs_prior, b_prior, jitter_prior, ror_prior = None,
    fpfs_prior = None, tune = 1500, 
    draws = 2500, target_accept = 0.99, phase = 'primary',
    ldc_val = None, bin_time = 10., 
    flux_cutoffs = [0.], end_nums = [0], filter_widths = [31],
    sigma_cuts = [5], sigma_cut_init = 10, 
    sigma_prior = None, rho_prior = None,
    ror_nominal = None):
    """
    The main function used to fit the target light curve and find the best-fit transit model parameters.

    Parameters
    ----------

    """
    x_list, ys_list, yerrs_list, compars_list, weight_guess_list = [],[],[],[],[]
    n_datasets = len(dump_dirs)
    for i, (dump_dir, best_ap, background_mode, covariate_names, texp, \
        flux_cutoff, end_num, filter_width, sigma_cut) in enumerate(zip(
        dump_dirs, best_aps, background_modes, covariate_names_list, texps,
        flux_cutoffs, end_nums, filter_widths, sigma_cuts)):
        tag = f'_{i}'
        x_init, ys_init, yerrs_init, bkgs_init, centroid_x_init, \
            centroid_y_init, airmass, widths = \
            load_phot_data(dump_dir, best_ap)
        compars_init = ys_init[1:]
        weight_guess_init = np.array([1./len(compars_init)]*len(compars_init))

        x, ys, yerrs, compars, mask = clean_up(x_init, ys_init, yerrs_init,
                compars_init, weight_guess_init, flux_cutoff,
                end_num, filter_width, sigma_cut_init, plot = True,
                plot_dir = plot_dir, tag = tag)

        cov_dict = get_covariates(bkgs_init, centroid_x_init, centroid_y_init,
            airmass, widths, background_mode, mask)
        covs = crossmatch_covariates(covariate_names, cov_dict)
        plot_quickfit(plot_dir, x, ys, yerrs, tag)
        plot_covariates(plot_dir, x, covariate_names, covs, tag)

        weight_guess = np.array([1./compars.shape[0]]*compars.shape[0] + \
            [0]*(len(covs) + 1)) 
        compars = np.vstack((compars, *covs))
        compars = np.vstack((compars, x - np.mean(x)))
        x_list.append(x)
        ys_list.append(ys)
        yerrs_list.append(yerrs)
        compars_list.append(compars)
        weight_guess_list.append(weight_guess)

    '''
    mask = np.zeros(x.shape, dtype = bool)
    i = 0
    #refitting the MAP with iterative sigma clipping until no new outliers found
    while sum(~mask) > 0:
        i += 1
        if i == 1:
            print("Constructing model...")
        else:
            print("Refitting MAP...")
        model, map_soln = make_model(x, ys, yerrs, compars, weight_guess,
            texp, r_star_prior, t0_prior, period_prior,
            a_rs_prior, b_prior, jitter_prior, phase, ror_prior,
            fpfs_prior, ldc_val, gp, sigma_prior, rho_prior,
            ror_nominal, tess_x, tess_y, tess_yerr, tess_texp,
            fixed_jitter = fixed_jitter, tess_t0_prior = tess_t0_prior)
        plot_initial_map(plot_dir, x, ys, yerrs, compars, map_soln, gp,
            fixed_jitter)
        if fit_tess:
            plot_initial_map_tess(plot_dir, tess_x, tess_y, tess_yerr, map_soln,
                    tess_t0_prior)
        print("Initial MAP found!")
        if refit:
            x, ys, yerrs, compars, mask = clean_after_map(x, ys, yerrs,
                compars, map_soln, sigma_cut, plot = True,
                plot_dir = plot_dir, extra_tag = i)
        else:
            mask = np.ones(x.shape, dtype = bool)
    print("MAP found!")
    plot_white_light_curves(plot_dir, x, ys)

    '''
    masks = [np.zeros(x.shape, dtype = bool) for x in x_list]
    j = 0
    mask_sums = [sum(~mask_i) for mask_i in masks]
    while sum(mask_sums) > 0:
        j += 1
        if j == 1:
            print("Constructing model...")
        else:
            print("Refitting MAP...")
        ##model in pymc3
        model, map_soln = make_model_joint(x_list, ys_list, yerrs_list,
            compars_list, weight_guess_list, texps, r_star_prior, t0_prior,
            period_prior, a_rs_prior, b_prior, jitter_prior, ror_prior,
            ldc_val, ror_nominal)
        for i, (x, ys, yerrs, compars) in enumerate(zip(x_list, \
            ys_list, yerrs_list, compars_list)):

            plot_initial_map(plot_dir, x, ys, yerrs, compars, 
                    map_soln, gp = False, joint = True, joint_num = i)
            x, ys, yerrs, compars, mask = clean_after_map(x, ys, yerrs,
                compars, map_soln, sigma_cut, plot = True,
                plot_dir = plot_dir, extra_tag = str(j),
                n_model = str(i))
            x_list[i] = x
            ys_list[i] = ys
            yerrs_list[i] = yerrs
            compars_list[i] = compars
            masks[i] = mask
        mask_sums = [sum(~mask_i) for mask_i in masks]

    print("MAP found!")
    for i, (x, ys) in enumerate(zip(x_list, ys_list)): 
        plot_white_light_curves(plot_dir, x, ys, tag = f'_{i}')

    print("Sampling posterior...")
    trace = sample_model(model, map_soln, tune, draws, target_accept)
    trace.posterior.to_netcdf(f'{plot_dir}posterior.nc', engine='scipy')
    print("Sampling complete!")

    print("Making corner and trace plots...") 
    plot_nominal = ror_nominal is not None
    summary, varnames = gen_summary(plot_dir, trace, phase, ldc_val,
        False, plot_nominal, joint = len(x_list))
    gen_latex_table(plot_dir, summary)
    trace_plot(plot_dir, trace, varnames)
    corner_plot(plot_dir, trace, varnames)

    print("Visualizing fit...")
    new_map = get_new_map(trace)
    t0 = float(new_map["t0"]) \
        if t0_prior[0] != 'fixed' else t0_prior[1]
    p = float(new_map['period']) \
        if period_prior[0] != 'fixed' else period_prior[1]
    x_final, y_final, yerr_final, resid_final, full_x, full_y, \
        full_yerr, full_resid = fully_fold(x_list,
        ys_list, yerrs_list, compars_list, new_map, t0, p)
    tripleplot_joint(plot_dir, x_final, y_final, yerr_final, yerrs_list, 
        resid_final, new_map, trace, texps, t0, p,
        full_x, full_y, full_yerr, full_resid,
        bin_time = bin_time, plot_nominal = plot_nominal)
    doubleplot_joint(plot_dir, x_final, y_final, yerr_final, yerrs_list, 
        resid_final, new_map, trace, texps, t0, p,
        full_x, full_y, full_yerr, full_resid,
        bin_time = bin_time, plot_nominal = plot_nominal)
    gen_lightcurve_table(plot_dir, full_x, full_y, full_yerr)
    print("Fitting complete!")
    return None    

def fully_fold(x_list, ys_list, yerrs_list, compars_list, 
    new_map, map_t0, map_p):
    full_x = np.array([])
    full_y = np.array([])
    full_yerr = np.array([])
    full_resid = np.array([])
    x_final = []
    y_final = []
    yerr_final = []
    resid_final = []
    for i in range(len(x_list)):
        systematics = np.dot(np.array(new_map[f"weights_{i}"]),
             compars_list[i])
        vec = x_list[i] - np.mean(x_list[i]) #+ base[1]
        #baseline = np.poly1d(np.array(new_map[f'baseline_{i}']))(vec)
        #detrended_data = (ys_list[i][0] - baseline)/systematics
        detrended_data = (ys_list[i][0])/systematics
        true_err = np.sqrt(yerrs_list[i][0]**2 + float(new_map[f"jitter_{i}"])**2)
        lc = np.array(new_map[f'light_curve_{i}'])
        x_fold = (x_list[i] - map_t0 + 0.5 * map_p) % map_p - 0.5 * map_p
        x_final.append(x_fold)
        y_final.append(detrended_data)
        yerr_final.append(true_err)
        resid_final.append(detrended_data-lc)
        full_x = np.append(full_x, x_fold)
        full_y = np.append(full_y, detrended_data)
        full_yerr = np.append(full_yerr, true_err)
        full_resid = np.append(full_resid, detrended_data - lc)
    sort = np.argsort(full_x)
    full_x = full_x[sort]
    full_y = full_y[sort]
    full_yerr = full_yerr[sort]
    full_resid = full_resid[sort]
    return x_final, y_final, yerr_final, resid_final, \
         full_x, full_y, full_yerr, full_resid

def make_model_joint(x_list, ys_list, yerrs_list, compars_list, 
    weight_guess_list, texps, r_star_prior, t0_prior, period_prior,
    a_rs_prior, b_prior, jitter_prior, ror_prior = None,
    ldc_val = None, ror_nominal = None, nominal_bounds = (-0.2, 0.2)):
    """
    Generates the light curve model with exoplanet.


    """
    ##currently doing circular orbits ONLY

    with pm.Model() as model:
        if ldc_val:
            star = xo.LimbDarkLightCurve(ldc_val)
        else:
            u = xo.distributions.QuadLimbDark("u")
            star = xo.LimbDarkLightCurve(u)
        r_star = unpack_prior('r_star', r_star_prior)

        period = unpack_prior('period', period_prior)
        t0 = unpack_prior('t0', t0_prior)
        t = t0

        a_rs = unpack_prior('a_rs', a_rs_prior)
        b = unpack_prior('b', b_prior)
        ror = unpack_prior('ror', ror_prior)

        orbit = xo.orbits.KeplerianOrbit(period = period,
            t0 = t, b = b, a = a_rs*r_star, r_star = r_star)

        #lightcurve
        arr = np.linspace(nominal_bounds[0], nominal_bounds[1], 1000)
        dummy_t = pm.Deterministic("dummy_t", t + arr)
        dummy_lightcurve = pm.Deterministic("dummy_light_curve",
            pm.math.sum(star.get_light_curve(orbit = orbit, 
            r = ror*r_star, t = dummy_t, texp = texps[0]), axis = -1) + 1)
        if ror_nominal is not None:
            nominal_lightcurve = pm.Deterministic("light_curve_nominal",
                pm.math.sum(star.get_light_curve(orbit = orbit, 
                r = ror_nominal*r_star, t = dummy_t, texp = texps[0]),
                axis = -1) + 1)
            mid_transit = pm.Deterministic("mid_transit_excess_depth", 
                -(tt.min(dummy_lightcurve) - tt.min(nominal_lightcurve)))

        for i in range(len(x_list)):
            lightcurve = pm.Deterministic(f"light_curve_{i}", pm.math.sum(
                star.get_light_curve(orbit=orbit, r = ror*r_star,
                t = x_list[i], texp = texps[i]), axis = -1) + 1.)

            #systematics
            comp_weights = pm.Uniform(f"weights_{i}", -2., 2.,
                testval = weight_guess_list[i],
                shape = len(weight_guess_list[i]))
            systematics = pm.math.dot(comp_weights, compars_list[i])
            jitter = unpack_prior(f'jitter_{i}', jitter_prior)
            full_variance = yerrs_list[i][0]**2 + jitter**2
            #baseline
            #vec = x_list[i] - np.mean(x_list[i])
            #base = pm.Uniform(f"baseline_{i}", -2, 2., shape = 2,
            #    testval = np.array([0.,0.]))
            #baseline = base[0]*vec + base[1]

            #full_model = baseline + systematics*lightcurve
            full_model = systematics*lightcurve
            pm.Deterministic(f"full_model_{i}", full_model)
            pm.Normal(f"obs_{i}", mu=full_model,
                sd=np.sqrt(full_variance), observed=ys_list[i][0])
  
        map_soln = model.test_point
        map_soln = pmx.optimize(map_soln)

        return model, map_soln

def calculate_ldcs(teff, logg, z):
    #reading filter transmission curve
    fname = '/Users/shrey/Desktop/My Papers/Helium Paper 1 (W69, W52)/'+\
        'Data Behind the Figure/filter_transmission_function.csv'
    data = np.genfromtxt(fname, delimiter=',', skip_header = 1)
    lab_filter_wavs = data[:,1]
    lab_filter_sigs = data[:,2]
    filt = TabulatedFilter("helium_filter", lab_filter_wavs, lab_filter_sigs)
    sc = LDPSetCreator(teff=teff, logg=logg, z=z, filters=[filt])
    ps = sc.create_profiles(nsamples=200000)
    ps.resample_linear_z(30000)
    qc,qe = ps.coeffs_qd()
    print("Limb darkening coefficients: ", qc[0])
    return qc[0]
