import exowirc.calib_utils as cu
import exowirc.fit_utils as fu
import exowirc.photo_utils as pu
import exowirc.io_utils as iu
import numpy as np
import warnings

######## pipeline steps #################################
remake_darks_and_flats = True 
remake_bkg = True
calibrate_data = True
photometric_extraction = True
fit_for_eclipse = True
######## calibration params ###############
data_dir = '/Volumes/brassnose/20190824/'
output_dir = '../../data_products/'
test_name = 'Kepler289'
nonlinearity_fname = None
naming_style = 'image'
science_seqs = [(65, 456)] 
dark_seqs = [(458, 477)]
flat_seq = (22, 41)
dark_for_flat_seq = (2, 21)
bkg_seq = (60, 63)
bkg_sigma_lower = 5
bkg_sigma_upper = 1000
background_mode = 'median'
covariate_names = ['d_from_med', 'background', 'airmass']
####### extraction params ###############
source_coords = [1210, 671]
finding_fwhm = 15.
extraction_rads = range(5, 25)
ann_rads = (25, 50)
source_detection_sigma = 100.
max_num_compars = 10
######## planet params ################
phase = 'primary'
texp = (50./60.)/1440. #days
r_star_prior = ('normal', 1.01, 0.045) #Berger+18
period_prior = ('normal', 125.8518, 0.0076) #Schmitt+14
t0_prior = ('normal', 2455069.8518, 0.0077) #Schmitt+14
a_rs_prior = ('normal', 108.6, 1.1) #Schmitt+14
b_prior = ('normal', 0.394, 0.029) #Schmitt+14
ror_prior = ('uniform', 0., 0.25)
jitter_prior = ('uniform', 1e-6, 1e-2)
######## fitting params ###############
tune = 1000            #number of burn-in steps per chain
draws = 1500           #number of steps per chain
target_accept = 0.99   #basically step-size tuning, closer to 1 -> small steps
######### reducing the data ############################

if __name__ == '__main__':
	calib_dir, dump_dir, img_dir = iu.init_output_direcs(output_dir,
		test_name)	

	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		bkg = cu.make_calibrated_bkg_image(data_dir, calib_dir,	bkg_seq,
			dark_seqs, dark_for_flat_seq, flat_seq,
			naming_style = naming_style, 
			nonlinearity_fname = nonlinearity_fname,
			sigma_lower = bkg_sigma_lower, 
			sigma_upper = bkg_sigma_upper, 
			remake_darks_and_flats = remake_darks_and_flats,
			remake_bkg = remake_bkg)

	if calibrate_data:
		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			cu.calibrate_all(data_dir, calib_dir, dump_dir,
				science_seqs, dark_seqs, dark_for_flat_seq,
				flat_seq, style = naming_style, 
				background_mode = background_mode, 
				bkg_filename = bkg)

	if photometric_extraction:
		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			pu.perform_photometry(calib_dir, dump_dir, img_dir,
				science_seqs, source_coords,
				style = naming_style,
				finding_fwhm = finding_fwhm, 
				extraction_rads = extraction_rads,
				background_mode = background_mode,
				ann_rads = ann_rads,
				source_detection_sigma = source_detection_sigma,
				max_num_compars = max_num_compars,
				bkg_fname = bkg)

	if fit_for_eclipse:
		with warnings.catch_warnings():
			
			warnings.simplefilter("ignore")
			best_ap = fu.quick_aperture_optimize(dump_dir, img_dir, 
				extraction_rads)
			fu.fit_lightcurve(dump_dir, img_dir, best_ap,
				background_mode, covariate_names, texp,
				r_star_prior, t0_prior, period_prior,
				a_rs_prior, b_prior, jitter_prior,
				phase = phase, ror_prior = ror_prior,
				tune = tune, draws = draws, 
				target_accept = target_accept)
