import numpy as np
from scipy.ndimage import median_filter
from astropy.io import fits
from astropy import time as ap_time, coordinates as coord, units as u
from astropy.stats import median_absolute_deviation, sigma_clip, \
	sigma_clipped_stats
import cv2

from .io_utils import get_science_img_list, load_calib_files, \
	get_img_name, save_image, save_multicomponent_frame, save_covariates

def calibrate_all(raw_dir, calib_dir, dump_dir, science_ranges, dark_ranges,
	dark_for_flat_range, flat_range, destripe = True, style = 'wirc',
	background_mode = None, bkg_filename = None,
	correct_nonlinearity = False, remake_darks_and_flats = False,
	nonlinearity_fname = None, mask_channels = []):
	"""Creates a sequence of science images by calibrating raw images with the dark image data ("darks") and the flat image data ("flats"). The darks remove ambient thermal noises. The flats remove dead pixels or overlysensitive hot pixels in the raw images. The resulting calibrated science images are stored inside the calib folder, and metadata of the covariates, which are objects and conditions that affected the interstellar brightness when images were taken, are stored inside the dump directory.

	Parameters
	------------
	raw_dir : string
			Path to the directory in which all the raw data are stored
	calib_dir : string
			Path to the directory in which the [*calibrated data*] will be stored
	dump_dir : string
			Path to directory in which the saved covariates background values will be stored
	science_ranges : list of tuples
			List of (int1, int2) tuples, where each tuple defines a single linear sequence of raw science frames to be calibrated
	dark_ranges : list of tuples
			List of (int1, int2) tuples. List is usually of length 1 if all raw images were taken with the [*same exposure time*] in one setting. List length may be longer than 1 if the raw images were taken with different [exposure times] in multiple sequences and settings. Each tuple defines a single linear sequence of darks from which a combined dark will be constructed
	dark_for_flat_range : tuple of ints
			Tuple (int1, int2) that defines a linear sequence of darks from which a combined dark will be created to correct the flats
	flat_range : tuple of ints
			Tuple (int1, int2) that defines a linear sequence of flats from which a combined flat will be created
	destripe : boolean, optional
			Flag that indicates whether raw images should be 'destriped'. The four quadrants--up, right, bottom, left--of the telescope detector produce a stripping effect on the image output. Setting the flag subtracts bias levels from the image output
	style : string, optional
			Prefix convention used in naming the image. Usually 'image' or 'wirc' unless otherwise specified during observations
	background_mode : string or None, optional
			Either None, 'median', 'global', or 'helium' indicating the background subtraction procedure. 'median'-[*insert explanation*]. 'global'-[*insert explanation*]. 'helium'-[*insert explanation*]
	bkg_filename : string, optional
			Path to the [*reduced background frame*]
	correct_nonlinearity : boolean, optional
			Flag that indicates whether or not to do nonlinearity correction on [*the data*]
	remake_darks_and_flats : boolean, optional
			Flag that indicates whether or not to remake the darks and flats
	nonlinearity_fname : string or None, optional
			Path to the file with the [*nonlinearity 	correction coefficients*]
	
	Returns
	-------------
	calib_dir : string
			Path to the directory containining calibrated science images [*why is it returned? is this dir used anywhere else in the code?*]
	"""
	assert (len(science_ranges) == len(dark_ranges)) or \
		(len(dark_ranges) == 1)

	#making/loading darks and flats
	flat, darks, bp, hps = make_darks_and_flats(raw_dir, calib_dir, 
		dark_ranges, dark_for_flat_range, flat_range, style,
		remake_darks_and_flats)

	#making mcf
	mcf = None
	if background_mode == 'helium':
		mcf = construct_multicomponent_frame(calib_dir, dump_dir)	

	covariates = {'bkgs': [], 'bjd': [], 'AIRMASS': []}

	for i, science_range in enumerate(science_ranges):
		science_range = [science_range]
		to_calibrate = get_science_img_list(science_range)
		dark_index = 0
		if len(darks) > 1:
			dark_index = i
		covariates = calibrate_sequence(raw_dir, calib_dir,
			to_calibrate, flat, darks[dark_index], bp,
			hps[dark_index], bkg_filename, destripe, style,
			background_mode, correct_nonlinearity,
			nonlinearity_fname, mcf, covariates,
			mask_channels)

	save_covariates(dump_dir, covariates)

	print("CALIBRATION COMPLETE")

	return calib_dir

def calibrate_sequence(raw_dir, calib_dir, science_sequence, flat, dark, bp, hp,
	bkg, destripe, style, background_mode, correct_nonlinearity,
	nonlinearity_fname, mcf, covariates, mask_channels):
	"""Calibrates the raw science sequence images with darks and flats to construct an array of scalar values that represents the median of all covariates in the sky background in each image.
	
	Parameters
	------------
	raw_dir : string
			Path to the directory in which all the raw data are stored
	calib_dir : string
			Path to the directory into which the calibrated data will be stored
	science_sequence : tuple of ints
			A tuple (int1, int2) that defines a linear sequence of raw science images to be corrected
	flat : string
			Path to the combined flat that will be used to calibrate the science sequence
	dark : string
			Path to the combined dark that will be used to calibrate the science sequence
	bp : string
			path to the bad pixel file that will be used to calibrate the science sequence
	hp : string
			path to the hot pixel file that will be used to calibrate the science sequence
	bkg : string or None
			Path to the sky background file that will be used to calibrate the science sequence
	destripe : boolean, optional
			Flag indicating whether raw images should be 'destriped'. The four quadrants--up, right, bottom, left--of the telescope detector produce a stripping effect on the image output. Setting the flag subtracts bias levels from the image output
	median_sub : boolean
			Flag that indicates whether the sigma-clipped median of the image should be subtracted during calibration or not
	style : string, optional
			Prefix convention used in naming the image. Usually 'image' or 'wirc' unless otherwise specified during observations
	save_bkg : boolean
			Flag that indicates whether median values subtracted as background should be saved or not. If true, saves to dump_dir

	Returns
	--------------
	bkgs : array of floats
		An array of floats containing the saved median background value of each image. May be an empty array if save_bkg is set to false.
	"""
	flat, dark, bp, hp, nonlinearity_array, correct_nonlinearity = \
		load_calib_files(flat,dark,bp,hp,nonlinearity_fname)
	if bkg is not None:
		with fits.open(bkg) as hdul:
			background_frame = hdul[0].data
	else:
		background_frame = None
	
	for i in science_sequence:
		image = get_img_name(raw_dir, i, style = style) 
		print(f"Reducing {image}...")
		calib, covariates = calibrate_image(image, flat, dark, bp, hp,
			correct_nonlinearity = correct_nonlinearity,
			nonlinearity_array = nonlinearity_array,
			destripe = destripe, background_mode = background_mode,
			background_frame = background_frame,
			multicomponent_frame = mcf,
			covariate_dict = covariates,
			mask_channels = mask_channels)
		outname = get_img_name(calib_dir, i, style = style)
		save_image(calib, outname)

	return covariates 

###Checking saved versions###
def check_saved(dirname, dark_seqs, flat_seq, style):
	"""Checks that the dark and flat files have already been created and all related metadata stored in the image header file are saved.

	Parameters
	--------------------
	dirname : string
			Path to the directory in which all darks and flats are stored
	dark_seqs : list of tuples
			List of (int1, int2) tuples. List is usually of length 1 if all raw images were taken with the [*same exposure time*] in one setting. List length may be longer than 1 if the raw images were taken with different [exposure times] in multiple sequences and settings. Each tuple defines a single linear sequence of darks from which a combined dark will be constructed
	flat_seq : tuple of ints
			Tuple (int1, int2) that defines a linear sequence of flat from which a combined flat will be created
	style : string, optional
			Prefix convention used in naming the image. Usually 'image' or 'wirc' unless otherwise specified during observations
	Returns
	-----------------
	flat : string
			Path to the combined flat image
	darks : list of strings
			List of paths to the combined darks
	bpname : string
			Path to the bad pixel file from the combined flat
	hps : list of strings
			Paths to the hot pixel files from the combined darks
	"""
	darks = []
	hps = []
	for dark in dark_seqs:
		num = dark[-1]
		zeros = '0'*(4-len(str(num)))
		darkname = f'{dirname}{style}{zeros}{num}' + '_combined_dark.fits'
		hpname = f'{dirname}{style}{zeros}{num}' + '_combined_hp_map.fits'
		darks.append(darkname)
		hps.append(hpname)
		hdul = fits.open(darkname)
		hdul.close()
		hdul = fits.open(hpname)
		hdul.close()

	num = flat_seq[-1]
	zeros = '0'*(4 - len(str(num)))
	flatname = f'{dirname}{style}{zeros}{num}' + \
		'_combined_flat.fits'
	bpname = f'{dirname}{style}{zeros}{num}' + \
		'_combined_bp_map.fits'
	hdul = fits.open(flatname)
	hdul.close()
	hdul = fits.open(bpname)
	hdul.close()

	print("Loaded saved darks and flats...")
	return flatname, darks, bpname, hps

def check_saved_background(imname):
	"""Checks that the background image has already been created and saved.

	Parameters
	----------
	imname : string
			Path to the background image

	Returns
	-------
	string
			Path to the background image
	"""
	hdul = fits.open(imname)
	hdul.close()
	print("Loaded saved background file...")
	return imname

###Flats and Darks###

def make_darks_and_flats(dirname, calib_dir, dark_seqs, dark_for_flat_seq,
	flat_seq, style, remake_darks_and_flats = True):
	"""Creates combined dark, dark for flat, and combined flat for calibrating the raw science images. [*Output paths*] for flat, darks, bad pixels (bp), and hot pixels [*type of file or DS*]
	
	Parameters
	-------------
	dirname : string
			Path to the directory in which all darks and flats are stored
	dark_seqs : list of tuples
			List of (int1, int2) tuples. List is usually of length 1 if all raw images were taken with the [*same exposure time*] in one setting. List length may be longer than 1 if the raw images were taken with different [exposure times] in multiple sequences and settings. Each tuple defines a single linear sequence of darks from which a combined dark will be constructed
	dark_for_flat_seq : tuple of ints
			Tuple (int1, int2) that defines a linear sequence of darks from which a combined dark will be created to correct the flat
	flat_seq : tuple of ints
			Tuple (int1, int2) that defines a linear sequence of flat from which a combined flat will be created
	style : string, optional
			Prefix convention used in naming the image. Usually 'image' or 'wirc' unless otherwise specified during observations

	Returns
	--------------
	flat : string
			Path to the combined flat image
	darks : list of strings
			List of paths to the combined darks
	bp : string
			Path to the bad pixel file from the combined flat
	hps : list of strings
			Paths to the hot pixel files from the combined darks
	"""
	#check if saved versions exist
	if not remake_darks_and_flats:
		try:
			return check_saved(calib_dir, dark_seqs,
				flat_seq, style)
		except FileNotFoundError as e:
			print("Can't find saved darks/flats -- remaking...")

	temp,_ = make_combined_image(dirname, calib_dir, *dark_for_flat_seq, # * specify pointer value   dark_for_flat_seq [(int, int)]
		style = style, calibration = 'dark') # make_combined_image returns var name temp, an array
	print('DARK FOR FLAT CREATED')  # dark for flat : remove thermal noise over a period of time (dark - dome also covered for the same amount of time to calculate thermal sensitivity) and pixel sensitivity (flat). Dark for flat is thermal noise to be reomved from flat images because flat images also have thermal noise
	flat,bp = make_combined_image(dirname, calib_dir, *flat_seq, # use dark for flat to correct flat images to remove thermal noise in each of the flat to make the combined flat
		style = style, calibration = 'flat', dark_for_flat_name = temp)
	print('COMBINED FLAT CREATED')
	
	darks = []
	hps = []
	for seq in dark_seqs: # for each tuple in dark_seqs, spit out dark file and hot pixel map
		dark, hp = make_combined_image(dirname, calib_dir, *seq,
			style = style, calibration = 'dark')
		darks.append(dark)
		hps.append(hp)
		print('COMBINED DARK CREATED')
			
	return flat, darks, bp, hps   # flat and darks are 2D arrays of scalar value. # bp and hp are 2D boolean arrays (2048 * 2048)

def make_combined_image(dirname, calib_dir, seq_start, seq_end,
	calibration = 'dark', dark_for_flat_name = None, style = 'wirc'):
	"""Given a dark or flat sequence, constructs a combined frame.

	Parameters
	-------------
	dirname : string
		path to the directory in which raw frames are stored
	dirname : string
		path to the directory you want calibrated frames saved
	seq_start : int
		starting image number for the sequence
	seq_end : int
		ending image number for the sequence
	calibration : string
		either 'dark' or 'flat'
	dark_for_flat_name : string
		path to the dark used to correct flat-fielding images;
		must not be None if making a flat
	style : string, optional
		the prefix for the image number. usually 'image' or 'wirc'
		unless otherwise specified during observations. 

	Returns
	--------------
	savename : string
		path to the combined frame
	bpname : string
		path to the bad/hot pixel file from the combined frame
	"""
	zeros = '0'*(4 - len(str(seq_end)))  # seq_end is the last img num for the 21 images to make combined from. It's saying to take that num at the end of seq and then create a var zeros which is a string '0'*4-(whateve the last num is) to create a tale end of the file name to save the combined img
	# if dark seq ends on img 21, then it creates a new file name img for the combined dark
	image_list = [get_img_name(dirname, i, style = style) for i in range(seq_start, seq_end + 1)]  # get_img_name
	# print(len(image_list))
	stack = np.zeros([2048, 2048, len(image_list)])
	for i,name in enumerate(image_list):
		with fits.open(name) as hdul:
			temp = hdul[0].data  # hdul - hdu list is a astro term. fits img files are used commonly in astro. fits have unique structure. hdu is the "header data unit" of fits file, within the index 0 of which has a header file that has identiftying info about the file like temp and other meta data. data gives value in an indivudal dark or flat image
		print(f"Stacking image {name}...")
		if calibration == 'dark':
			stack[:,:,i] = temp  # stacking. stack arr is a 3d matrix of 2048 * 2048 * 21 images 
		else: # stacking flat, remove dark for flat noise
			with fits.open(dark_for_flat_name) as hdul:
				dark_for_flat = hdul[0].data
			dark_corr = temp - dark_for_flat# take temp arr, flat file, subtract dark 
			stack[:,:,i] = dark_corr/np.nanmedian( 
				dark_corr.flatten())  # .flatten() makes the 2D array into a 1D array
	combined = np.nanmedian(stack, axis = 2)
	# I think I can break this function down into two seperate functions, and hence the condition check in the for-loop above can be slightly improved
	if calibration == 'dark':  # scalar hp
		print("Generating hot pixel map...")
		hp = get_hot_px(stack) 
		bpname = f'{calib_dir}{style}{zeros}{seq_end}'
		bpname += f'_combined_hp_map.fits'
		save_image(hp, bpname)
	else:
		print("Generating bad pixel map...")
		bp = get_bad_px(combined) # boolean bp
		bpname = f'{calib_dir}{style}{zeros}{seq_end}'
		bpname += f'_combined_bp_map.fits'
		save_image(bp, bpname)
		bp = np.array(bp, dtype = 'bool')
		med = np.nanmedian(combined[~bp])  # the combined value is the stacked flat image file. take the median of all the bp arrays that are not bad pixel
		combined = combined/med  # 
	savename = f'{calib_dir}{style}{zeros}{seq_end}'
	savename += f'_combined_{calibration}.fits'
	save_image(combined, savename)
	return savename, bpname

def get_hot_px(dark_stack, sig_hot_pix = 5):  # sig_hot_pix is standard deviation 
	"""original implement by WIRC+Pol team
	This function identifies pixel values that are too many standard deviations away from the median (default to 5 std) and mask them (turn them into NAN with sigma clipping)

	Parameters
	----------
	dark_stack : [type]
			[description]
	sig_hot_pix : int, optional
			[description], by default 5

	Returns
	-------
	[type]
			[description]
	"""
	MAD = median_absolute_deviation(dark_stack, axis = 2)  # median_absolute_deviation FROM ASTROPY module measure std in images. this will tell us where are the hot pixels. 2 means it's a 2 D matrix - find the hot pix in the 2D array 
	hot_px = sigma_clip(MAD, sigma = sig_hot_pix)   #sigma_clip another AstroPy module goes through img and clips out any img above the treshhold
	return np.array(hot_px.mask, dtype = 'int') # hot_px.mask -> mask is a built-in python object returning the same shape as obj you're applying to. (boolean arr?) all entries are true . return 2D array where hot pixel found, ret true, where good pixel found, false

def stddevFilter(img, box_size):
	"""from stackoverflow 28931265, implemented by WIRC+Pol team
	computes standard deviation of image in a moving box of given size.

	Parameters
	----------
	img : [type]
			[description]
	box_size : [type]
			[description]

	Returns
	-------
	[type]
			[description]
	"""
	wmean, wsqrmean = (cv2.boxFilter(x, -1, (box_size, box_size), 
		borderType=cv2.BORDER_REFLECT) for x in (img, img*img))
	return np.sqrt(wsqrmean - wmean*wmean)

def get_bad_px(flat, local_sig_bad_pix = 3, global_sig_bad_pix = 9,
	local_box_size = 11):
	"original implement by WIRC+Pol team"
	median_flat = median_filter(flat, local_box_size)
	stddev_im = stddevFilter(flat, local_box_size)
	local_bad_pix = np.abs(median_flat - flat) > local_sig_bad_pix*stddev_im
	pix_to_pix = flat/median_flat
	global_bad_px = sigma_clip(pix_to_pix, sigma = global_sig_bad_pix).mask
	non_positive = flat <= 0

	bad_px = np.logical_or(global_bad_px, local_bad_pix)
	bad_px = np.logical_or(bad_px, non_positive)
	return np.array(bad_px, dtype = 'int')

def clean_bad_pix(image, bad_px_map, replacement_box = 5):
	"original implement by WIRC+Pol team"
	bad_px_map = np.logical_or(bad_px_map, image <= 0)
	bad_px_map = np.logical_or(bad_px_map, ~np.isfinite(image))
	med_fil = median_filter(image, size = replacement_box)
	cleaned = image*~bad_px_map + med_fil*bad_px_map
	return cleaned

###Background construction###

def make_calibrated_bkg_image(data_dir, calib_dir, bkg_seq, dark_ranges, 
	dark_for_flat_range, flat_range, naming_style = 'wirc', 
	nonlinearity_fname = None, sigma_lower = 5, 
	sigma_upper = 3, plot = False, remake_bkg = False,
	remake_darks_and_flats = False):
	"""[summary]
	Creates a calibrated sky background image used for removing the sky background noise from the raw science images. 

	Parameters
	--------------------
	data_dir : [type]
			[description]
	calib_dir : [type]
			[description]
	bkg_seq : [type]
			[description]
	dark_ranges : [type]
			[description]
	dark_for_flat_range : [type]
			[description]
	flat_range : [type]
			[description]
	naming_style : str, optional
			[description], by default 'wirc'
	nonlinearity_fname : [type], optional
			[description], by default None
	sigma_lower : int, optional
			[description], by default 5
	sigma_upper : int, optional
			[description], by default 3
	plot : bool, optional
			[description], by default False
	remake_bkg : bool, optional
			[description], by default False
	remake_darks_and_flats : bool, optional
			[description], by default False

	Returns
	--------------------
	[type]
			[description]
	"""

	image_list = [get_img_name(data_dir, i,
		style = naming_style) for \
		i in range(bkg_seq[0], bkg_seq[1] + 1)]
	imname = get_img_name(calib_dir, bkg_seq[-1],
		style = naming_style, img_type = 'calibrated_background')
	if not remake_bkg:
		try:
			return check_saved_background(imname)
		except FileNotFoundError as e:
			print("Can't find saved background -- remaking...")

	print("Creating background frame...")
	#create/load up flats and darks
	flat, darks, bp, hps = make_darks_and_flats(data_dir, calib_dir,
		dark_ranges, dark_for_flat_range, flat_range, naming_style,
		remake_darks_and_flats)
	dark = darks[0]
	hp = hps[0]
	flat, dark, bp, hp, nonlinearity_array, correct_nonlinearity = \
		load_calib_files(flat,dark,bp,hp,nonlinearity_fname)
	
	i = 0
	med_val = 0
	clipped_ims = np.ma.zeros([2048,2048, len(image_list)])
	for name in image_list:
		print(f"Stacking image {name}...")
		calib, _ = calibrate_image(name, flat, dark, bp, hp,
			correct_nonlinearity = correct_nonlinearity,
			nonlinearity_array = nonlinearity_array)
		clipped = sigma_clip(calib,
			sigma_lower = sigma_lower,
			sigma_upper = sigma_upper)
		if plot:
			plt.figure(figsize = (8,8))
			plt.imshow(clipped, origin = 'lower', vmin = 0, vmax = 70e3, cmap = 'Blues')
			plt.show()
		if i == 0:
			med_val = np.nanmedian(clipped.flatten())
		scale_factor = med_val / np.nanmedian(clipped.flatten())
		clipped *= scale_factor
		clipped_ims[:,:,i] = clipped
		i += 1

	background = np.ma.median(clipped_ims, axis = -1)
	save_image(background.filled(0.), imname)
	print("BACKGROUND FRAME CREATED")

	return imname

###Image calibration###

def get_bjd(header):
	date_in = header['UTSHUT']
	target_pos = coord.SkyCoord(header['RA'], header['DEC'],
		unit = (u.hourangle, u.deg), frame='icrs')
	palomar = coord.EarthLocation.of_site('Palomar')
	time=ap_time.Time(date_in, format='isot', scale='utc', location=palomar)
	half_exptime=0.5*header['EXPTIME']*header['COADDS']/(24*3600)
	ltt_bary = time.light_travel_time(target_pos)
	time = time.tdb+ltt_bary
	bjd_tdb = time.jd+half_exptime
	return bjd_tdb
	
def calibrate_image(im_name, flat, dark, bp, hp, correct_nonlinearity = False,
	nonlinearity_array = None, destripe = False, background_mode = None,
	background_frame = None, multicomponent_frame = None,
	covariate_dict = None, mask_channels = []):
	"""[summary]

	Parameters
	--------------------
	im_name : string
			name of the raw science image
	flat : string
			Path to the combined flat that will be used to calibrate the science sequence
	dark : string
			Path to the combined dark that will be used to calibrate the science sequence
	bp : string
			Path to the bad pixel file that will be used to calibrate the science sequence
	hp : string
			Path to the hot pixel file that will be used to calibrate the science sequence
	correct_nonlinearity : bool, optional
			Flag indicating whether nonlinearity array calibration is needed due to the presence of really bright objects/stars in the nightsky, by default False
	nonlinearity_array : array, optional
				[description], by default None
	destripe : bool, optional
				[description], by default False
	background_mode : [type], optional
				[description], by default None
	background_frame : [type], optional
				[description], by default None
	multicomponent_frame : [type], optional
				[description], by default None
	covariate_dict : [type], optional
				[description], by default None
	mask_channels : list, optional
				[description], by default []

	Returns
	--------------------
	cleaned :
			[description]
	covariate_dict : 
			[description]
	"""
	with fits.open(im_name) as hdul:
		hdu = hdul[0]
		header = hdu.header
		image = hdu.data
	if correct_nonlinearity:
		image = nonlinearity_correction(image, header,
			nonlinearity_array)
	dark_corr = image - dark
	corr = dark_corr/flat
	bad_px_map = np.logical_or(bp, hp)
	cleaned = clean_bad_pix(corr, bad_px_map)
	cleaned[~np.isfinite(cleaned)] = 0.
	retval = None
	if len(mask_channels) > 0:
		cleaned = mask_bad_channels(cleaned, mask_channels)

	if background_mode == 'median':
		#simple sigma-clipped median removal
		_, med, _ = sigma_clipped_stats(cleaned.flatten())
		cleaned -= med
		retval = med

	elif background_mode == 'global':
		#requires reduced background frame
		scale = np.nanmedian(cleaned) / np.nanmedian(background_frame)
		cleaned -= scale*background_frame
		retval = scale

	elif background_mode == 'helium':
		#requires multicomponent frame and reduced background frame
		cleaned, retval = helium_background_subtraction(cleaned,
			background_frame, multicomponent_frame)	
	
	if destripe:
		cleaned = destripe_image(cleaned)

	if covariate_dict is not None:
		for covariate in covariate_dict.keys():
			if covariate == 'bjd':
				covariate_dict[covariate].append(
					get_bjd(header))
			elif covariate == 'bkgs':
				covariate_dict[covariate].append(retval)
			else:
				covariate_dict[covariate].append(
					header[covariate])

	return cleaned, covariate_dict

def mask_bad_channels(cleaned, to_mask):
	"""[summary]
	channel labeling: bottom left quad, bottom to top = 0-7
	top right, bottom to top = 8-15
	top left, left to right = 16-23
	bottom right, left to right = 24-31

	ordering is basically all horizontal channels bottom to top,
	then all vertical channels left to right

	Parameters
	-----------------
	cleaned : [type]
			[description]
	to_mask : [type]
			[description]

	Returns
	-----------------
	[type]
			[description]
	"""
	for channel in to_mask:
		c = channel % 16
		xs = (0, 1024) if c < 8 else (1024, 2048)
		ys = (c * 128, (c + 1) * 128)
		if channel > 16:
			xs = ys
			ys = (1024, 2048) if c < 8 else (0, 1024)

		cleaned[ys[0]:ys[1], xs[0]:xs[1]] = np.nan

	return cleaned
		

def helium_background_subtraction(cleaned, background_frame,
	multicomponent_frame):
	"""[summary]

	Parameters
	----------
	cleaned : [type]
			[description]
	background_frame : [type]
			[description]
	multicomponent_frame : [type]
			[description]

	Returns
	-------
	[type]
			[description]
	"""
	comps = np.unique(multicomponent_frame)
	img_meds = [sigma_clipped_stats(
		cleaned[multicomponent_frame == comp],
		cenfunc = np.nanmedian,
		stdfunc = np.nanstd)[1] for comp in comps]
	bkg_meds = [sigma_clipped_stats(
		background_frame[multicomponent_frame == comp],
		cenfunc = np.nanmedian,
		stdfunc = np.nanstd)[1] for comp in comps]
	scale_factors = np.array(img_meds)/np.array(bkg_meds)
	new_arr = np.zeros(background_frame.shape)
	for i, comp in enumerate(comps):
		working_mask = (multicomponent_frame == comp)
		new_arr[working_mask] = \
			background_frame[working_mask]*scale_factors[i]

	return cleaned - new_arr, scale_factors
	
def destripe_image(image, sigma = 3, iters=5):
	"""Destripe the detector by subtracting the median
	of each row/column from each sector.
	This will work best after you subtract a background sky image.

	legacy code from WIRC+Pol; updated by Shreyas"""

	quads = []
	clean_imm = np.array(image)

	for i in range(4):
		k = 4 - i
		xr = (0, 1024) if i == 0 or i == 3 else (1024, 2048)
		yr = (0, 1024) if i == 0 or i == 1 else (1024, 2048)
		quad = image[xr[0]:xr[1], yr[0]:yr[1]]
		quad = np.rot90(quad, k = k, axes = (0, 1))
		quadm = sigma_clip(quad, sigma = 5)
		quad[quadm.mask] = np.nan
		quads.append(quad)

	mn_quad = np.nanmedian(quads, axis=0)

	for i in range(1024):
		to_sub = sigma_clipped_stats(
			mn_quad[:,i],sigma=sigma,maxiters=iters)[1]
		clean_imm[:1024,i]  -= to_sub
		clean_imm[1024:,-i] -= to_sub
		clean_imm[-i,:1024] -= to_sub
		clean_imm[i,1024:]  -= to_sub

	return clean_imm

def nonlinearity_correction(image, header, nonlinearity_arr):
	"""[summary]

	Parameters
	--------------------
	image : [type]
			[description]
	header : [type]
			[description]
	nonlinearity_arr : [type]
			[description]

	Returns
	--------------------
	[type]
			[description]
	"""
	assert np.shape(nonlinearity_arr) == np.shape(image)
	n_coadd = header['COADDS']
	image_copy = np.array(image, dtype = float) #copy image
	image_copy /= n_coadd
	image_copy = (-1 + np.sqrt(1 + 4*nonlinearity_arr*image_copy)) / \
		(2*nonlinearity_arr) #quadratic formula with correct root
	return image_copy * n_coadd

def construct_multicomponent_frame(calib_dir, dump_dir, rstepsize = 10):
	"""Create multicomponent frame for removing the undesired ring-effect that appears in helium background images.

	Parameters
	--------------------
	calib_dir : string
			Path to the directory in which the [*calibrated data*] will be stored
	dump_dir : string
			Path to directory in which the multicomponent framed background will be stored
	rstepsize : int, optional
			Radial step size in pixels across the detector, by default 10

	Returns
	--------------------
	[type]
			[description]
	"""
	home = (1037, 2120)
	mcf = np.zeros((2048, 2048))
	dmax = 2400
	print("Constructing multicomponent frame...")
	for i in range(mcf.shape[0]):
		for j in range(mcf.shape[1]):
			d = dist(home, (j,i))
			dist_tag = int(d // rstepsize) + 1
			mcf[i,j] = dist_tag
	print("Multicomponent frame constructed!")
	save_multicomponent_frame(mcf, dump_dir)	
	return mcf
	
def dist(p1, p2):
	"""Calculate the distance bwteeen two point coordinates using the Pythagorean Theorem.

	Parameters
	--------------------
	p1 : tuple of floats
			Tuple (int1, int2) representing a point of xy coordinate
	p2 : tuple of floats
			Tuple (int1, int2) representing a point of xy coordinate

	Returns
	--------------------
	float
			Distance between p1 and p2
	"""
	return np.sqrt((p2[1] - p1[1])**2 + (p2[0] - p1[0])**2)

