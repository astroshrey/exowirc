import exowirc.calib_utils as cu
import exowirc.fit_utils as fu
import exowirc.photo_utils as pu
import exowirc.io_utils as iu
import numpy as np
import warnings

######## pipeline steps #################################
remake_darks_and_flats = False 
remake_bkg = False
calibrate_data = False 
photometric_extraction = False 
fit_for_eclipse = True
######## calibration params ###############
data_dir = '../../data/20210327/'
output_dir = '../../data_products/'
test_name = 'WASP107'
nonlinearity_fname = None
naming_style = 'image'
science_seqs = [(132, 335)] 
dark_seqs = [(402, 420)]
flat_seq = (56, 76)
dark_for_flat_seq = (14, 34)
bkg_seq = (126, 129)
bkg_sigma_lower = 5
bkg_sigma_upper = 1000
mask_channels = [1, 2]
destripe = True
background_mode = 'helium'
covariate_names = ['d_from_med', 'water_proxy', 'airmass']
####### extraction params ###############
source_coords = [530, 1430]
target_and_compars = [source_coords, [1012, 1106]] 
extraction_rads = range(5, 25)
ann_rads = (25, 50)
max_num_compars = 10
######## planet params ################
phase = 'primary'
texp = 1./1440.
r_star_prior = ('normal', 0.67, 0.02) #Piaulet+20
period_prior = ('normal', 5.721490, 0.000002) #Anderson+17
t0_prior = ('normal', 2456514.4106, 0.0001) #Anderson+17
a_rs_prior = ('normal', 18.2, 0.1) #Anderson+17
b_prior = ('normal', 0.090, 0.070) #Anderson+17
ror_prior = ('uniform', 0., 0.5)
jitter_prior = ('uniform', 1e-6, 5e-2)
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
				flat_seq, destripe = destripe,
				style = naming_style, 
				background_mode = background_mode, 
				bkg_filename = bkg,
				mask_channels = mask_channels)

	if photometric_extraction:
		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			pu.perform_photometry(calib_dir, dump_dir, img_dir,
				science_seqs, source_coords,
				style = naming_style,
				extraction_rads = extraction_rads,
				background_mode = background_mode,
				ann_rads = ann_rads,
				max_num_compars = max_num_compars,
				bkg_fname = bkg,
				target_and_compars = target_and_compars)

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
