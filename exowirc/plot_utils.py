import numpy as np
import lightkurve as lk
import pymc3 as pm
import arviz as az
import matplotlib.pyplot as plt
import photutils
import corner
import matplotlib
from .io_utils import save_photon_noise

def plot_sources(img_dir, image, sources, fwhm, ann_rads):
    positions = [(x, y) for x, y in zip(
        sources['xcentroid'], sources['ycentroid'])]
    apertures = photutils.CircularAperture(positions, r = fwhm/2)
    annuli = photutils.CircularAnnulus(
        positions, r_in = ann_rads[0], r_out = ann_rads[1])
    plt.figure(figsize = (10, 8))
    plt.imshow(image, cmap = 'Greys', vmin = -5000, vmax = 5000,
        origin = 'lower')
    plt.colorbar()
    apertures.plot(color = 'blue', lw = 1.5, alpha = 0.5)
    annuli.plot(color = 'green', lw = 1.5, alpha = 0.5)
    plt.savefig(img_dir + 'source_plot.png')
    plt.close()
    return None

def plot_outlier_rejection(plot_dir, x, quick_detrend, filtered, mask,
    tag = ''):
    plt.plot(x, quick_detrend, 'k.')
    plt.plot(x, filtered, 'r-')
    plt.plot(x[~mask], quick_detrend[~mask], 'r.')
    plt.xlabel("Time [BJD]")
    if 'afterMAP' in tag:
        plt.ylabel("Data / MAP light curve")
    else:
        plt.ylabel("Relative Flux")
    plt.savefig(f'{plot_dir}outlier_rejection'+tag+'.png')
    plt.close()
    return None

def plot_initial_map(plot_dir, x, ys, yerrs, compars, map_soln, gp = False,
    fixed_jitter = None, joint = False, joint_num = 'None'):
    if joint:
        append = f'_{joint_num}'
    else:
        append = ''
    lc = map_soln["light_curve" + append]
    vec = x - np.mean(x)
    systematics = np.dot(map_soln["weights" + append], compars)
    if gp:
        baseline = map_soln["gp_pred" + append]
    else:
        baseline = 0.
    detrended_data = (ys[0])/systematics
    if fixed_jitter is None:
        true_err= np.sqrt(yerrs[0]**2 + map_soln["jitter" + append]**2)
    else:
        true_err= np.sqrt(yerrs[0]**2 + fixed_jitter**2)
    plt.errorbar(x, detrended_data, yerr = true_err,
        color = 'k', marker = '.', linestyle = 'None')
    plt.plot(x, lc, 'r-', zorder = 10)
    plt.savefig(f'{plot_dir}initial_map_soln{append}.png')
    plt.close()
    return None

def plot_quickfit(plot_dir, x, ys, yerrs, tag = ''):
    plt.errorbar(x, ys[0]/np.mean(ys[1:], axis = 0), yerr = yerrs[0],
        marker = '.', linestyle = 'None')
    plt.xlabel("Time [BJD]")
    plt.ylabel("Normalized Flux (Quick Detrend)")
    plt.savefig(f'{plot_dir}quickfit{tag}.png')
    plt.close()
    return None

def plot_covariates(plot_dir, x, covariate_names, covs, tag = ''):
    for name, cov in zip(covariate_names, covs):
        plt.plot(x, cov, 'C0-')
        plt.xlabel("Time [BJD]")
        plt.ylabel(f"{name}")
        plt.savefig(f'{plot_dir}{name}{tag}.png')
        plt.close()
    return None

def plot_white_light_curves(plot_dir, x, ys, tag = ''):
    for i, y in enumerate(ys):
        fmtstr = 'k.' if i == 0 else '.'
        lblstr = f'Comp {i}' if i > 0 else 'Target'
        plt.plot(x, y, fmtstr, label = lblstr)
    plt.legend(loc = 'best')
    plt.savefig(f'{plot_dir}white_light_curves{tag}.png')
    plt.legend(loc = 'best')
    plt.close()
    return None

def plot_aperture_opt(plot_dir, apertures, rmses):
    plt.plot(apertures, rmses)
    plt.xlabel("Aperture Size [px]")
    plt.ylabel("Per-Point rms")
    plt.savefig(f'{plot_dir}rms_vs_aperture.png')
    plt.close()
    return None

def gen_rms(lc, binsize):
    binned = lc.bin(time_bin_size = binsize)
    return(np.nanstd(binned.flux))

def gen_photon_noise(lc, photon_noise, binsize, texp):
    binned = lc.bin(time_bin_size = binsize)
    length_arr = len(binned.time)
    length_factor = np.sqrt(length_arr/(length_arr - 1))
    white_noise = photon_noise/np.sqrt(binsize/texp)*length_factor
    return white_noise

def gen_photon_noise_errors(lc, photon_noise, binsize):
    binned = lc.bin(time_bin_size = binsize)
    length_arr = len(binned.time)
    rms = np.nanstd(binned.flux)
    return rms/np.sqrt(2*length_arr)

def represent_noise_stats(dump_dir, new_map, resid, yerrs,
    fixed_jitter = None, tag = ''):

    filename = open(f'{dump_dir}noise_stats{tag}.txt', 'w')
    shot_noise = yerrs[0]
    if fixed_jitter is None:
        sigma_extra = float(new_map[f'jitter{tag}'])
    else:
        sigma_extra = fixed_jitter
    rms = np.nanstd(resid)
    npts = len(resid)

    print("Mean shot noise (data + bkg): ", np.mean(shot_noise)*1e6, "ppm",
        file = filename)
    if fixed_jitter is None:
        print("Extra noise: ", 
                sigma_extra*1e6, "ppm", file = filename)
    print("Per point rms: ", rms*1e6, "ppm", file = filename)
    print("Number of points: ", npts, file = filename)
    return None

def corner_plot(plot_dir, samples, varnames):
    corner.corner(samples, var_names = varnames,
        hist2d_kwargs = {'data_kwargs': {'rasterized': True}})
    plt.savefig(f'{plot_dir}cornerplt.png', dpi = 200,
        bbox_inches = 'tight')
    plt.close()
    return None

def trace_plot(plot_dir, data, varnames):
    az.plot_trace(data, var_names = varnames,
        trace_kwargs = {'rasterized': True})
    plt.savefig(f'{plot_dir}traceplt.pdf')
    plt.close()
    return None

def side_by_side(plot_dir, dump_dir, x, ys, yerrs, compars, detrended_data,
    new_map, trace, texp, map_t0, map_p, tess_x, tess_ys, tess_yerrs, tess_texp,
    phase = 'primary', bin_time = 10,
    gp = False, fixed_jitter = None, plot_nominal = False, baseline = None,
    fit_tess = False):
    #bin_time in mins

    matplotlib.rcParams['mathtext.fontset'] = 'cm'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'
    matplotlib.rcParams['font.size'] = 20
    fig, ax = plt.subplots(2, 2, figsize = (16, 8), sharex = 'col', sharey = 'row',
        gridspec_kw={'height_ratios': [3, 1]})

    #MAP lightcurve and reduction
    lc = np.array(new_map[f'light_curve'])
    if fixed_jitter is None:
        true_err = np.sqrt(yerrs[0]**2 + float(new_map[f"jitter"])**2)
    else:
        true_err = np.sqrt(yerrs[0]**2 + fixed_jitter**2)
    #if phase == 'primary':
    x_fold = (x - map_t0 + 0.5 * map_p) % map_p - 0.5 * map_p
    #else:
    #	x_fold = (x - map_t0) % map_p - 0.5 * map_p

    if gp:
        ax[0,0].plot(x_fold, 1+baseline, color = 'g', linestyle = '-', lw = 1)

	#setting the light curves for plotting and residuals
    plot_lc = lk.LightCurve(time = x_fold, flux = detrended_data,
        flux_err = true_err)
    bin_lc = plot_lc.bin(time_bin_size = bin_time / 1440.)
    plot_resid = lk.LightCurve(time = x_fold, flux = detrended_data - lc,
        flux_err = true_err)
    bin_resid = plot_resid.bin(time_bin_size = bin_time / 1440.)

    #plotting light curve and residuals, unbinned and binned
    ax[0,0].errorbar(x_fold, detrended_data, yerr = true_err,
        color = 'k', marker = '.', linestyle = 'None', alpha = 0.1)
    ax[0,0].errorbar(bin_lc.time.value, bin_lc.flux.value,
        yerr = bin_lc.flux_err.value,
        color = 'k', marker = '.', linestyle = 'None', ms = 5)
    ax[1,0].errorbar(plot_resid.time.value, plot_resid.flux.value,
        yerr = plot_resid.flux_err.value, color = 'k', marker = '.',
        linestyle = 'None', alpha = 0.1)
    ax[1,0].errorbar(bin_resid.time.value, bin_resid.flux.value,
        yerr = bin_resid.flux_err.value, color = 'k', marker = '.',
        linestyle = 'None', ms = 5)

    #MAP wirc light curve
    dummy_fold = new_map['dummy_t'] - map_t0
    dummy_lc = new_map['dummy_light_curve']
    ax[0,0].plot(dummy_fold, dummy_lc, f'r-', zorder = 10, lw = 2)
    #68 percentile on the confidence interval
    stacked = trace.posterior.stack(draws=("chain", "draw"))
    lcsamps = stacked.dummy_light_curve.values
    lower = np.percentile(lcsamps, 16, axis = 1)
    upper = np.percentile(lcsamps, 84, axis = 1)
    ax[0,0].fill_between(dummy_fold, lower, upper,
        alpha = 0.3, facecolor = f'r', lw = 1)


    minx = min(x_fold)
    maxx = max(x_fold)
    ymin, ymax = ax[0,0].get_ylim()
    scale = (ymax - ymin)/3
    ax[1,0].set_ylim(-scale/2, scale/2)
    ax[0,0].set_ylabel("Relative Flux")
    ax[1,0].set_ylabel("Residual")
    ax[1,0].set_xlabel("Time from Eclipse Center [d]")
    ax[1,1].set_xlabel("Time from Eclipse Center [d]")
    ax[0,0].set_xlim(minx, maxx)
    ax[1,0].set_xlim(minx, maxx)

    ####TESS LIGHTCURVE######

    #MAP lightcurve and reduction
    lc = np.array(new_map[f'light_curve_TESS'])
    true_err = tess_yerrs*float(new_map[f'tess_error_scaling'])
    x_fold = (tess_x - map_t0 + 0.5 * map_p) % map_p - 0.5 * map_p
    sort_inds = np.argsort(x_fold)
    #setting the light curves for plotting and residuals
    plot_lc = lk.LightCurve(time = x_fold[sort_inds], flux = tess_ys[sort_inds],
        flux_err = true_err[sort_inds])
    bin_lc = plot_lc.bin(time_bin_size = bin_time / 1440.)
    plot_resid = lk.LightCurve(time = x_fold[sort_inds], 
        flux = tess_ys[sort_inds] - lc[sort_inds], flux_err = true_err[sort_inds])
    bin_resid = plot_resid.bin(time_bin_size = bin_time / 1440.)

    #plotting light curve and residuals, unbinned and binned
    ax[0,1].errorbar(x_fold, tess_ys, yerr = true_err,
        color = 'k', marker = '.', linestyle = 'None', alpha = 0.01)
    ax[0,1].errorbar(bin_lc.time.value, bin_lc.flux.value,
        yerr = bin_lc.flux_err.value,
        color = 'k', marker = '.', linestyle = 'None', ms = 5, zorder = 5)
    ax[1,1].errorbar(plot_resid.time.value, plot_resid.flux.value,
        yerr = plot_resid.flux_err.value, color = 'k', marker = '.',
        linestyle = 'None', alpha = 0.01)
    ax[1,1].errorbar(bin_resid.time.value, bin_resid.flux.value,
        yerr = bin_resid.flux_err.value, color = 'k', marker = '.',
        linestyle = 'None', ms = 5, zorder = 5)

    #MAP tess light curve
    ax[0,1].plot(x_fold[sort_inds], lc[sort_inds], f'b-', zorder = 10, lw = 2)
    #68 percentile on the confidence interval
    stacked = trace.posterior.stack(draws=("chain", "draw"))
    lcsamps = stacked.light_curve_TESS.values
    lower = np.percentile(lcsamps, 16, axis = 1)
    upper = np.percentile(lcsamps, 84, axis = 1)
    ax[0,1].fill_between(x_fold[sort_inds], lower[sort_inds], upper[sort_inds],
        alpha = 0.3, facecolor = f'b', lw = 1)

    fig.align_ylabels(ax)
    fig.subplots_adjust(hspace=0)
    ax[0,1].set_xlim(-0.2, 0.2)
    ax[1,1].set_xlim(-0.2, 0.2)
    ax[0,0].set_title("WIRC", pad = 10)
    ax[0,1].set_title("TESS", pad = 10)

    plt.tight_layout()
    plt.savefig(f'{plot_dir}side_by_side.pdf', dpi = 200,
        bbox_inches = 'tight')
    plt.close()
    return None

def side_by_side_joint(plot_dir, x_final, y_final, yerrs_final, yerrs_list,
    resid_final, new_map, trace, texp_list, map_t0, map_p, full_x, full_y, 
    full_yerr, full_resid, tess_x, tess_ys, tess_yerrs, tess_texp,
    bin_time = 5, plot_nominal = False):

    matplotlib.rcParams['mathtext.fontset'] = 'cm'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'
    matplotlib.rcParams['font.size'] = 20
    fig, ax = plt.subplots(2, 2, figsize = (16, 8), sharex = 'col', sharey = 'row',
        gridspec_kw={'height_ratios': [3, 1]})
    minx = 0
    maxx = 0

    for i in range(len(x_final)):
        minx = min(min(x_final[i]), minx)
        maxx = max(max(x_final[i]), maxx)
        plot_lc = lk.LightCurve(time = x_final[i], flux = y_final[i],
            flux_err = yerrs_final[i])
        plot_resid = lk.LightCurve(time = x_final[i],
            flux = resid_final[i], flux_err = yerrs_final[i])

        if i == 0:
            m = '.'
        elif i == 1:
            m = '^'
        else:
            m = 's'
        ax[0,0].errorbar(plot_lc.time.value, plot_lc.flux.value, 
            yerr = plot_lc.flux_err.value, color = 'k', marker = m,
            linestyle = 'None', alpha = 0.1)
        ax[1,0].errorbar(plot_resid.time.value, plot_resid.flux.value,
            yerr = plot_resid.flux_err.value, color = 'k', 
            marker = m, linestyle = 'None', alpha = 0.1)

    plot_lc = lk.LightCurve(time = full_x, flux = full_y,
        flux_err = full_yerr)
    bin_lc = plot_lc.bin(time_bin_size = bin_time / 1440.)
    plot_resid = lk.LightCurve(time = full_x, 
        flux = full_resid, flux_err = full_yerr)
    bin_resid = plot_resid.bin(time_bin_size = bin_time / 1440.)
    ax[0,0].errorbar(bin_lc.time.value, bin_lc.flux.value,
        yerr = bin_lc.flux_err.value, 
        color = 'k', marker = '.', linestyle = 'None', ms = 5)
    ax[1,0].errorbar(bin_resid.time.value, bin_resid.flux.value,
        yerr = bin_resid.flux_err.value, 
        color = 'k', marker = '.', linestyle = 'None', ms = 5)

    #MAP wirc light curve
    dummy_fold = new_map['dummy_t'] - map_t0
    dummy_lc = new_map['dummy_light_curve']
    ax[0,0].plot(dummy_fold, dummy_lc, f'r-', zorder = 10, lw = 2)

    #68 percentile on the confidence interval
    stacked = trace.posterior.stack(draws=("chain", "draw"))
    lcsamps = stacked.dummy_light_curve.values
    lower = np.percentile(lcsamps, 16, axis = 1)
    upper = np.percentile(lcsamps, 84, axis = 1)
    ax[0,0].fill_between(dummy_fold, lower, upper,
        alpha = 0.3, facecolor = f'r', lw = 1)

    ymin, ymax = ax[0,0].get_ylim()
    scale = (ymax - ymin)/3
    ax[1,0].set_ylim(-scale/2, scale/2)
    ax[0,0].set_ylabel("Relative Flux")
    ax[1,0].set_ylabel("Residual")
    ax[1,0].set_xlabel("Time from Eclipse Center [d]")
    ax[1,1].set_xlabel("Time from Eclipse Center [d]")
    ax[0,0].set_xlim(minx, maxx)
    ax[1,0].set_xlim(minx, maxx)

    ####TESS LIGHTCURVE######

    #MAP lightcurve and reduction
    lc = np.array(new_map[f'light_curve_TESS'])
    true_err = tess_yerrs*float(new_map[f'tess_error_scaling'])
    x_fold = (tess_x - map_t0 + 0.5 * map_p) % map_p - 0.5 * map_p
    sort_inds = np.argsort(x_fold)
    #setting the light curves for plotting and residuals
    plot_lc = lk.LightCurve(time = x_fold[sort_inds], flux = tess_ys[sort_inds],
        flux_err = true_err[sort_inds])
    bin_lc = plot_lc.bin(time_bin_size = bin_time / 1440.)
    plot_resid = lk.LightCurve(time = x_fold[sort_inds], 
        flux = tess_ys[sort_inds] - lc[sort_inds], flux_err = true_err[sort_inds])
    bin_resid = plot_resid.bin(time_bin_size = bin_time / 1440.)

    #plotting light curve and residuals, unbinned and binned
    ax[0,1].errorbar(x_fold, tess_ys, yerr = true_err,
        color = 'k', marker = '.', linestyle = 'None', alpha = 0.01)
    ax[0,1].errorbar(bin_lc.time.value, bin_lc.flux.value,
        yerr = bin_lc.flux_err.value,
        color = 'k', marker = '.', linestyle = 'None', ms = 5, zorder = 5)
    ax[1,1].errorbar(plot_resid.time.value, plot_resid.flux.value,
        yerr = plot_resid.flux_err.value, color = 'k', marker = '.',
        linestyle = 'None', alpha = 0.01)
    ax[1,1].errorbar(bin_resid.time.value, bin_resid.flux.value,
        yerr = bin_resid.flux_err.value, color = 'k', marker = '.',
        linestyle = 'None', ms = 5, zorder = 5)

    #MAP tess light curve
    ax[0,1].plot(x_fold[sort_inds], lc[sort_inds], f'b-', zorder = 10, lw = 2)
    #68 percentile on the confidence interval
    stacked = trace.posterior.stack(draws=("chain", "draw"))
    lcsamps = stacked.light_curve_TESS.values
    lower = np.percentile(lcsamps, 16, axis = 1)
    upper = np.percentile(lcsamps, 84, axis = 1)
    ax[0,1].fill_between(x_fold[sort_inds], lower[sort_inds], upper[sort_inds],
        alpha = 0.3, facecolor = f'b', lw = 1)

    fig.align_ylabels(ax)
    fig.subplots_adjust(hspace=0)
    ax[0,1].set_xlim(-0.2, 0.2)
    ax[1,1].set_xlim(-0.2, 0.2)
    ax[0,0].set_title("WIRC", pad = 10)
    ax[0,1].set_title("TESS", pad = 10)

    plt.tight_layout()
    plt.savefig(f'{plot_dir}side_by_side_joint.pdf', dpi = 200,
        bbox_inches = 'tight')
    plt.close()

    return None

def doubleplot(plot_dir, dump_dir, x, ys, yerrs, compars, detrended_data,
    new_map, trace, texp, map_t0, map_p, phase = 'primary', bin_time = 10,
    gp = False, fixed_jitter = None, plot_nominal = False, baseline = None, 
    fit_tess = False):
    #bin_time in mins

    matplotlib.rcParams['mathtext.fontset'] = 'cm'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'
    matplotlib.rcParams['font.size'] = 20
    fig, ax = plt.subplots(2, 1, figsize = (8, 8), sharex = True,
        gridspec_kw={'height_ratios': [3, 1]})

    #MAP lightcurve and reduction
    lc = np.array(new_map[f'light_curve'])
    if fixed_jitter is None:
        true_err = np.sqrt(yerrs[0]**2 + float(new_map[f"jitter"])**2)
    else:
        true_err = np.sqrt(yerrs[0]**2 + fixed_jitter**2)
    #if phase == 'primary':
    x_fold = (x - map_t0 + 0.5 * map_p) % map_p - 0.5 * map_p
    #else:
    #	x_fold = (x - map_t0) % map_p - 0.5 * map_p

    if gp:
        ax[0].plot(x_fold, 1+baseline, color = 'g', linestyle = '-', lw = 1)

	#setting the light curves for plotting and residuals
    plot_lc = lk.LightCurve(time = x_fold, flux = detrended_data,
        flux_err = true_err)
    bin_lc = plot_lc.bin(time_bin_size = bin_time / 1440.)
    plot_resid = lk.LightCurve(time = x_fold, flux = detrended_data - lc,
        flux_err = true_err)
    bin_resid = plot_resid.bin(time_bin_size = bin_time / 1440.)

    #plotting light curve and residuals, unbinned and binned
    ax[0].errorbar(x_fold, detrended_data, yerr = true_err,
        color = 'k', marker = '.', linestyle = 'None', alpha = 0.1)
    ax[0].errorbar(bin_lc.time.value, bin_lc.flux.value,
        yerr = bin_lc.flux_err.value,
        color = 'k', marker = '.', linestyle = 'None', ms = 5)
    ax[1].errorbar(plot_resid.time.value, plot_resid.flux.value,
        yerr = plot_resid.flux_err.value, color = 'k', marker = '.',
        linestyle = 'None', alpha = 0.1)
    ax[1].errorbar(bin_resid.time.value, bin_resid.flux.value,
        yerr = bin_resid.flux_err.value, color = 'k', marker = '.',
        linestyle = 'None', ms = 5)

    #MAP wirc light curve
    dummy_fold = new_map['dummy_t'] - map_t0
    dummy_lc = new_map['dummy_light_curve']
    ax[0].plot(dummy_fold, dummy_lc, f'r-', zorder = 10, lw = 2)
    #68 percentile on the confidence interval
    stacked = trace.posterior.stack(draws=("chain", "draw"))
    lcsamps = stacked.dummy_light_curve.values
    lower = np.percentile(lcsamps, 16, axis = 1)
    upper = np.percentile(lcsamps, 84, axis = 1)
    ax[0].fill_between(dummy_fold, lower, upper,
        alpha = 0.3, facecolor = f'r', lw = 1)

    #MAP wirc light curve
    if plot_nominal == True:
        dummy_lc_opt = new_map['light_curve_nominal']
        ax[0].plot(dummy_fold, dummy_lc_opt, f'b-', zorder = 10, lw = 2)
        #lcsamps = stacked.light_curve_nominal.values
        #lower = np.percentile(lcsamps, 16, axis = 1)
        #upper = np.percentile(lcsamps, 84, axis = 1)
        #ax[0].fill_between(dummy_fold, lower, upper,
        #	alpha = 0.3, facecolor = f'b', lw = 1)
    if fit_tess:
        dummy_lc_opt = new_map['light_curve_nominal']
        #ax[0].plot(dummy_fold, dummy_lc_opt, f'b-', zorder = 10, lw = 2)

    minx = min(x_fold)
    maxx = max(x_fold)
    ymin, ymax = ax[0].get_ylim()
    scale = (ymax - ymin)/3
    ax[1].set_ylim(-scale/2, scale/2)
    ax[0].set_ylabel("Relative Flux")
    ax[1].set_ylabel("Residual")
    ax[1].set_xlabel("Time from Eclipse Center [d]")
    ax[0].set_xlim(minx, maxx)
    ax[1].set_xlim(minx, maxx)
    fig.align_ylabels(ax)
    fig.subplots_adjust(hspace=0)
    
    plt.tight_layout()
    plt.savefig(f'{plot_dir}double_plot.pdf', dpi = 200,
        bbox_inches = 'tight')
    plt.close()
    return None

def tripleplot(plot_dir, dump_dir, x, ys, yerrs, compars, detrended_data,
    new_map, trace, texp, map_t0, map_p, phase = 'primary', bin_time = 10,
    gp = False, fixed_jitter = None, plot_nominal = False, baseline = None, 
    fit_tess = False):
    #bin_time in mins

    matplotlib.rcParams['mathtext.fontset'] = 'cm'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'
    matplotlib.rcParams['font.size'] = 20
    fig, ax = plt.subplots(3, 1, figsize = (8, 12),
        gridspec_kw={'height_ratios': [3, 1, 3]})

    #MAP lightcurve and reduction
    lc = np.array(new_map[f'light_curve'])
    if fixed_jitter is None:
        true_err = np.sqrt(yerrs[0]**2 + float(new_map[f"jitter"])**2)
    else:
        true_err = np.sqrt(yerrs[0]**2 + fixed_jitter**2)
    #if phase == 'primary':
    x_fold = (x - map_t0 + 0.5 * map_p) % map_p - 0.5 * map_p
    #else:
    #	x_fold = (x - map_t0) % map_p - 0.5 * map_p

    if gp:
        ax[0].plot(x_fold, 1+baseline, color = 'g', linestyle = '-', lw = 1)

    #setting the light curves for plotting and residuals
    plot_lc = lk.LightCurve(time = x_fold, flux = detrended_data,
        flux_err = true_err)
    bin_lc = plot_lc.bin(time_bin_size = bin_time / 1440.)
    plot_resid = lk.LightCurve(time = x_fold, flux = detrended_data - lc,
        flux_err = true_err)
    bin_resid = plot_resid.bin(time_bin_size = bin_time / 1440.)
    represent_noise_stats(dump_dir, new_map, plot_resid.flux, yerrs, 
        fixed_jitter)

    #plotting light curve and residuals, unbinned and binned
    ax[0].errorbar(x_fold, detrended_data, yerr = true_err,
        color = 'k', marker = '.', linestyle = 'None', alpha = 0.1)
    ax[0].errorbar(bin_lc.time.value, bin_lc.flux.value,
        yerr = bin_lc.flux_err.value,
        color = 'k', marker = '.', linestyle = 'None', ms = 5)
    ax[1].errorbar(plot_resid.time.value, plot_resid.flux.value,
        yerr = plot_resid.flux_err.value, color = 'k', marker = '.',
        linestyle = 'None', alpha = 0.1)
    ax[1].errorbar(bin_resid.time.value, bin_resid.flux.value,
        yerr = bin_resid.flux_err.value, color = 'k', marker = '.',
        linestyle = 'None', ms = 5)

    #MAP wirc light curve
    dummy_fold = new_map['dummy_t'] - map_t0
    dummy_lc = new_map['dummy_light_curve']
    ax[0].plot(dummy_fold, dummy_lc, f'r-', zorder = 10, lw = 2)
    #68 percentile on the confidence interval
    stacked = trace.posterior.stack(draws=("chain", "draw"))
    lcsamps = stacked.dummy_light_curve.values
    lower = np.percentile(lcsamps, 16, axis = 1)
    upper = np.percentile(lcsamps, 84, axis = 1)
    ax[0].fill_between(dummy_fold, lower, upper,
        alpha = 0.3, facecolor = f'r', lw = 1)

    #MAP wirc light curve
    if plot_nominal == True:
        dummy_lc_opt = new_map['light_curve_nominal']
        ax[0].plot(dummy_fold, dummy_lc_opt, f'b-', zorder = 10, lw = 2)
        #lcsamps = stacked.light_curve_nominal.values
        #lower = np.percentile(lcsamps, 16, axis = 1)
        #upper = np.percentile(lcsamps, 84, axis = 1)
        #ax[0].fill_between(dummy_fold, lower, upper,
        #	alpha = 0.3, facecolor = f'b', lw = 1)
    if fit_tess:
        dummy_lc_opt = new_map['light_curve_nominal']
        ax[0].plot(dummy_fold, dummy_lc_opt, f'b-', zorder = 10, lw = 2)

    #rms vs binsize
    tsep = np.median(np.ediff1d(x))
    binsizes = np.arange(1, 30) * tsep
    rmses = [gen_rms(plot_resid, bs) for bs in binsizes]
    photon_noise = np.median(yerrs[0])
    photon_noises = [gen_photon_noise(
        plot_resid, photon_noise, bs, tsep) for bs in binsizes]
    photon_noise_errors = [gen_photon_noise_errors(
        plot_resid, photon_noise, bs) for bs in binsizes]

    save_photon_noise(dump_dir, binsizes, rmses, photon_noise_errors,
            photon_noises)

    ax[2].errorbar(binsizes*1440., rmses, yerr = photon_noise_errors,
        color = 'k')
    ax[2].plot(binsizes*1440., photon_noises, 'r-')
    ax[2].plot(binsizes*1440.,
        np.array(photon_noises)*rmses[0]/photon_noises[0], 'r--')

    minx = min(x_fold)
    maxx = max(x_fold)

    ymin, ymax = ax[0].get_ylim()
    scale = (ymax - ymin)/3
    ax[1].set_ylim(-scale/2, scale/2)
	
    ax[0].set_ylabel("Relative Flux")
    ax[1].set_ylabel("Residual")
    ax[1].set_xlabel("Time from Eclipse Center [d]")
    ax[0].set_xlim(minx, maxx)
    ax[1].set_xlim(minx, maxx)
    ax[2].set_ylabel("rms scatter")
    ax[2].set_xlabel("Binsize [min]")
    ax[2].set_xscale('log')
    ax[2].set_yscale('log')
    fig.align_ylabels(ax)
    plt.tight_layout()
    plt.savefig(f'{plot_dir}triple_plot.pdf', dpi = 200,
        bbox_inches = 'tight')
    plt.close()
    return None

def plot_tess(plot_dir, x, ys, yerrs, new_map, trace, texp, map_t0, map_p, 
    tess_t0_prior, bin_time = 10):
    #bin_time in mins

    matplotlib.rcParams['mathtext.fontset'] = 'cm'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'
    matplotlib.rcParams['font.size'] = 20
    fig, ax = plt.subplots(2, 1, figsize = (8, 8),
        gridspec_kw={'height_ratios': [3, 1]})

    #MAP lightcurve and reduction
    lc = np.array(new_map[f'light_curve_TESS'])
    true_err = yerrs*float(new_map[f'tess_error_scaling'])
    x_fold = (x - map_t0 + 0.5 * map_p) % map_p - 0.5 * map_p
    sort_inds = np.argsort(x_fold)
    #setting the light curves for plotting and residuals
    plot_lc = lk.LightCurve(time = x_fold[sort_inds], flux = ys[sort_inds],
        flux_err = true_err[sort_inds])
    bin_lc = plot_lc.bin(time_bin_size = bin_time / 1440.)
    plot_resid = lk.LightCurve(time = x_fold[sort_inds], 
        flux = ys[sort_inds] - lc[sort_inds], flux_err = true_err[sort_inds])
    bin_resid = plot_resid.bin(time_bin_size = bin_time / 1440.)

    #plotting light curve and residuals, unbinned and binned
    ax[0].errorbar(x_fold, ys, yerr = true_err,
        color = 'k', marker = '.', linestyle = 'None', alpha = 0.1)
    ax[0].errorbar(bin_lc.time.value, bin_lc.flux.value,
        yerr = bin_lc.flux_err.value,
        color = 'k', marker = '.', linestyle = 'None', ms = 5, zorder = 5)
    ax[1].errorbar(plot_resid.time.value, plot_resid.flux.value,
        yerr = plot_resid.flux_err.value, color = 'k', marker = '.',
        linestyle = 'None', alpha = 0.1)
    ax[1].errorbar(bin_resid.time.value, bin_resid.flux.value,
        yerr = bin_resid.flux_err.value, color = 'k', marker = '.',
        linestyle = 'None', ms = 5, zorder = 5)

    #MAP wirc light curve
    ax[0].plot(x_fold[sort_inds], lc[sort_inds], f'b-', zorder = 10, lw = 2)
    #68 percentile on the confidence interval
    stacked = trace.posterior.stack(draws=("chain", "draw"))
    lcsamps = stacked.light_curve_TESS.values
    lower = np.percentile(lcsamps, 16, axis = 1)
    upper = np.percentile(lcsamps, 84, axis = 1)
    ax[0].fill_between(x_fold[sort_inds], lower[sort_inds], upper[sort_inds],
        alpha = 0.3, facecolor = f'b', lw = 1)

    minx = -0.2
    maxx = 0.2

    ymin, ymax = ax[0].get_ylim()
    scale = (ymax - ymin)/3
    ax[1].set_ylim(-scale/2, scale/2)
	
    ax[0].set_ylabel("Relative Flux")
    ax[1].set_ylabel("Residual")
    ax[1].set_xlabel("Time from Eclipse Center [d]")
    ax[0].set_xlim(minx, maxx)
    ax[1].set_xlim(minx, maxx)
    fig.align_ylabels(ax)
    fig.subplots_adjust(hspace=0)

    plt.tight_layout()
    plt.savefig(f'{plot_dir}double_plot_tess.pdf', dpi = 200,
        bbox_inches = 'tight')
    plt.close()
    return None

def tripleplot_joint(plot_dir, x_final, y_final, yerrs_final, yerrs_list,
    resid_final, new_map, trace, texp_list, map_t0, map_p, full_x, full_y, 
    full_yerr, full_resid, bin_time = 5, plot_nominal = False):

    matplotlib.rcParams['mathtext.fontset'] = 'cm'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'
    matplotlib.rcParams['font.size'] = 20
    fig, ax = plt.subplots(3, 1, figsize = (8, 12),
        gridspec_kw={'height_ratios': [3, 1, 3]})
    minx = 0
    maxx = 0
    for i in range(len(x_final)):
        minx = min(min(x_final[i]), minx)
        maxx = max(max(x_final[i]), maxx)
        plot_lc = lk.LightCurve(time = x_final[i], flux = y_final[i],
            flux_err = yerrs_final[i])
        plot_resid = lk.LightCurve(time = x_final[i],
            flux = resid_final[i], flux_err = yerrs_final[i])

    #represent_noise_stats(dump_dir, new_map, plot_resid.flux, yerrs)
        if i == 0:
            m = '.'
        elif i == 1:
            m = '^'
        else:
            m = 's'
        ax[0].errorbar(plot_lc.time.value, plot_lc.flux.value, 
            yerr = plot_lc.flux_err.value, color = 'k', marker = m,
            linestyle = 'None', alpha = 0.1)
        ax[1].errorbar(plot_resid.time.value, plot_resid.flux.value,
            yerr = plot_resid.flux_err.value, color = 'k', 
            marker = m, linestyle = 'None', alpha = 0.1)

        represent_noise_stats(plot_dir, new_map, plot_resid.flux, 
                yerrs_list[i], fixed_jitter = None,
                tag = '_' + str(i))

        tsep = np.median(np.ediff1d(x_final[i]))
        binsizes = np.arange(1, 30) * tsep
        rmses = [gen_rms(plot_resid, bs) for bs in binsizes]
        photon_noise = np.median(yerrs_list[i][0])
        photon_noises = [gen_photon_noise(
            plot_resid, photon_noise, bs, tsep) for bs in binsizes]
        photon_noise_errors = [gen_photon_noise_errors(
            plot_resid, photon_noise, bs) for bs in binsizes]

        save_photon_noise(plot_dir, binsizes, rmses, photon_noise_errors,
                photon_noises, tag = str(i))

        ax[2].errorbar(binsizes*1440., rmses, marker = m,
            yerr = photon_noise_errors, color = 'k')
        ax[2].plot(binsizes*1440., photon_noises, marker = m,
            color = 'r', linestyle = '-')
        ax[2].plot(binsizes*1440.,
            np.array(photon_noises)*rmses[0]/photon_noises[0],
            marker = m, color = 'r', linestyle = '--')
    plot_lc = lk.LightCurve(time = full_x, flux = full_y,
        flux_err = full_yerr)
    bin_lc = plot_lc.bin(time_bin_size = bin_time / 1440.)
    plot_resid = lk.LightCurve(time = full_x, 
        flux = full_resid, flux_err = full_yerr)
    bin_resid = plot_resid.bin(time_bin_size = bin_time / 1440.)
    ax[0].errorbar(bin_lc.time.value, bin_lc.flux.value,
        yerr = bin_lc.flux_err.value, 
        color = 'k', marker = '.', linestyle = 'None', ms = 5)
    ax[1].errorbar(bin_resid.time.value, bin_resid.flux.value,
        yerr = bin_resid.flux_err.value, 
        color = 'k', marker = '.', linestyle = 'None', ms = 5)

    #MAP wirc light curve
    dummy_fold = new_map['dummy_t'] - map_t0
    dummy_lc = new_map['dummy_light_curve']
    dummy_lc_opt = new_map['light_curve_nominal']
    ax[0].plot(dummy_fold, dummy_lc, f'r-', zorder = 10, lw = 2)
    ax[0].plot(dummy_fold, dummy_lc_opt, f'b-', zorder = 10, lw = 2)
    #68 percentile on the confidence interval
    stacked = trace.posterior.stack(draws=("chain", "draw"))
    lcsamps = stacked.dummy_light_curve.values
    lower = np.percentile(lcsamps, 16, axis = 1)
    upper = np.percentile(lcsamps, 84, axis = 1)
    ax[0].fill_between(dummy_fold, lower, upper,
        alpha = 0.3, facecolor = f'r', lw = 1)

    #rms vs binsize
    ymin, ymax = ax[0].get_ylim()
    scale = (ymax - ymin)/3
    ax[1].set_ylim(-scale/2, scale/2)
	
    ax[0].set_ylabel("Relative Flux")
    ax[1].set_ylabel("Residual")
    ax[1].set_xlabel("Time from Eclipse Center [d]")
    ax[0].set_xlim(minx, maxx)
    ax[1].set_xlim(minx, maxx)
    ax[2].set_ylabel("rms scatter")
    ax[2].set_xlabel("Binsize [min]")
    ax[2].set_xscale('log')
    ax[2].set_yscale('log')
    fig.align_ylabels(ax)

    plt.tight_layout()
    plt.savefig(f'{plot_dir}triple_plot_joint.pdf', dpi = 200,
        bbox_inches = 'tight')
    plt.close()

    return full_x, full_y, full_yerr

def plot_initial_map_tess(plot_dir, x, ys, yerrs, map_soln, tess_t0_prior):
    lc = np.array(map_soln[f'light_curve_TESS'])
    true_err = yerrs*float(map_soln[f'tess_error_scaling'])
    if tess_t0_prior is not None:
        map_t0 = float(map_soln['t0_tess'])
    else:
        map_t0 = float(map_soln['t0'])
    map_p = float(map_soln['period'])
    x_fold = (x - map_t0 + 0.5 * map_p) % map_p - 0.5 * map_p
    sort_inds = np.argsort(x_fold)
    #setting the light curves for plotting and residuals
    plot_lc = lk.LightCurve(time = x_fold[sort_inds], flux = ys[sort_inds],
        flux_err = true_err[sort_inds])
    plt.errorbar(plot_lc.time.value, plot_lc.flux.value, yerr = plot_lc.flux_err.value,
        color = 'k', marker = '.', linestyle = 'None')

    if tess_t0_prior is not None:
        dummy_fold = np.array(map_soln['dummy_t_tess']) - map_t0
    else:
        dummy_fold = np.array(map_soln['dummy_t']) - map_t0

    dummy_lc = np.array(map_soln['light_curve_nominal'])
    plt.plot(dummy_fold, dummy_lc, f'b-', zorder = 10, lw = 2)

    plt.savefig(f'{plot_dir}initial_map_soln_tess.png')
    plt.close()
    return None

def doubleplot_joint(plot_dir, x_final, y_final, yerrs_final, yerrs_list,
    resid_final, new_map, trace, texp_list, map_t0, map_p, full_x, full_y, 
    full_yerr, full_resid, bin_time = 5, plot_nominal = False):

    matplotlib.rcParams['mathtext.fontset'] = 'cm'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'
    matplotlib.rcParams['font.size'] = 20
    fig, ax = plt.subplots(2, 1, figsize = (8, 8), sharex = True,
        gridspec_kw={'height_ratios': [3, 1]})
    minx = 0
    maxx = 0

    for i in range(len(x_final)):
        minx = min(min(x_final[i]), minx)
        maxx = max(max(x_final[i]), maxx)
        plot_lc = lk.LightCurve(time = x_final[i], flux = y_final[i],
            flux_err = yerrs_final[i])
        plot_resid = lk.LightCurve(time = x_final[i],
            flux = resid_final[i], flux_err = yerrs_final[i])

        if i == 0:
            m = '.'
        elif i == 1:
            m = '^'
        else:
            m = 's'
        ax[0].errorbar(plot_lc.time.value, plot_lc.flux.value, 
            yerr = plot_lc.flux_err.value, color = 'k', marker = m,
            linestyle = 'None', alpha = 0.1)
        ax[1].errorbar(plot_resid.time.value, plot_resid.flux.value,
            yerr = plot_resid.flux_err.value, color = 'k', 
            marker = m, linestyle = 'None', alpha = 0.1)

    plot_lc = lk.LightCurve(time = full_x, flux = full_y,
        flux_err = full_yerr)
    bin_lc = plot_lc.bin(time_bin_size = bin_time / 1440.)
    plot_resid = lk.LightCurve(time = full_x, 
        flux = full_resid, flux_err = full_yerr)
    bin_resid = plot_resid.bin(time_bin_size = bin_time / 1440.)
    ax[0].errorbar(bin_lc.time.value, bin_lc.flux.value,
        yerr = bin_lc.flux_err.value, 
        color = 'k', marker = '.', linestyle = 'None', ms = 5)
    ax[1].errorbar(bin_resid.time.value, bin_resid.flux.value,
        yerr = bin_resid.flux_err.value, 
        color = 'k', marker = '.', linestyle = 'None', ms = 5)

    #MAP wirc light curve
    dummy_fold = new_map['dummy_t'] - map_t0
    dummy_lc = new_map['dummy_light_curve']
    dummy_lc_opt = new_map['light_curve_nominal']
    ax[0].plot(dummy_fold, dummy_lc, f'r-', zorder = 10, lw = 2)
    ax[0].plot(dummy_fold, dummy_lc_opt, f'b-', zorder = 10, lw = 2)

    #68 percentile on the confidence interval
    stacked = trace.posterior.stack(draws=("chain", "draw"))
    lcsamps = stacked.dummy_light_curve.values
    lower = np.percentile(lcsamps, 16, axis = 1)
    upper = np.percentile(lcsamps, 84, axis = 1)
    ax[0].fill_between(dummy_fold, lower, upper,
        alpha = 0.3, facecolor = f'r', lw = 1)

    ymin, ymax = ax[0].get_ylim()
    scale = (ymax - ymin)/3
    ax[1].set_ylim(-scale/2, scale/2)
    ax[0].set_ylabel("Relative Flux")
    ax[1].set_ylabel("Residual")
    ax[1].set_xlabel("Time from Eclipse Center [d]")
    ax[0].set_xlim(minx, maxx)
    ax[1].set_xlim(minx, maxx)
    fig.align_ylabels(ax)
    fig.subplots_adjust(hspace=0)

    plt.tight_layout()
    plt.savefig(f'{plot_dir}double_plot_joint.pdf', dpi = 200,
        bbox_inches = 'tight')
    plt.close()

    return None


