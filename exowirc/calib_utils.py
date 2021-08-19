import numpy as np
from scipy.ndimage import median_filter
from astropy.io import fits
from astropy import time as ap_time, coordinates as coord, units as u
from astropy.stats import median_absolute_deviation, sigma_clip, \
	sigma_clipped_stats
import cv2

np.set_printoptions(threshold=np.inf)

from .io_utils import get_science_img_list, load_calib_files, \
	get_img_name, save_image, save_multicomponent_frame, save_covariates

def calibrate_all(raw_dir, calib_dir, dump_dir, science_ranges, dark_ranges,
	dark_for_flat_range, flat_range, destripe = True, style = 'wirc',
	background_mode = None, bkg_filename = None,
	correct_nonlinearity = False, remake_darks_and_flats = False,
	nonlinearity_fname = None, mask_channels = []):
	"""Creates a sequence of science images by calibrating raw images with the dark image data ("darks") and the flat image data ("flats"). The darks remove ambient thermal noise. The flats correct for the varying sensitivity of the pixels accross the detector. Dead pixels and hot pixels are also removed. The resulting calibrated science images are stored inside the calib folder, and metadata of the covariates, which are objects and conditions may affect the stellar brightness when images were taken, are stored inside the dump directory along with the pickled photometry.

	Parameters
	------------
	raw_dir : string
			Path to the directory in which all the raw data are stored
	calib_dir : string
			Path to the directory in which the calibrated image data will be stored
	dump_dir : string
			Path to directory in which the saved covariates background values and pickled raw photometry values will be stored
	science_ranges : list of tuples
			A list of tuples (int1, int2) where int1 and int2 are the image numbers for a linear sequence of raw science images to be corrected. 
	dark_ranges : list of tuples
			List of (int1, int2) tuples corresponding to starting and ending image numbers of the dark image sequence(s)
	dark_for_flat_range : list of tuples
			List of (int1, int2) tuples corresponding to starting and ending image numbers of the dark image sequence(s) to be used for the flat sequences. 
	flat_range : tuple of ints
			Tuple (int1, int2) corresponding to starting and ending image numbers of the flat image sequence(s).
	destripe : boolean, optional
			Flag that indicates whether raw images should be 'destriped'. The four quadrants of the telescope detector produce a striping effect on the image output. Setting the flag subtracts the median value from each row or column in each quadrant.
	style : string, optional
			Prefix convention used in naming the image file. Usually 'image' or 'wirc' unless otherwise specified during observations
	background_mode : string or None, optional
			Either None, 'median', 'global', or 'helium' indicating the background subtraction procedure. 'median' subtracts a sigma-clipped median from the whole image, 'global' subtracts a calibrated background frame from each image, and 'helium' subtracts a calibrated background frame combined with ta multicomponent frame to capture the unique helium filter structure.
	bkg_filename : string, optional
			Path to the reduced sky background frame
	correct_nonlinearity : boolean, optional
			Flag that indicates whether or not to do nonlinearity correction on the image data (in the case of bright targets approaching the saturation limit of the detector)
	remake_darks_and_flats : boolean, optional
			Flag that indicates whether or not to remake the combined darks and flats
	nonlinearity_fname : string or None, optional
			Path to the file with the nonlinearity correction coefficients (a 2048x2048 numpy array with nonlinearity coefficients per pixel)
	mask_channels : list, optional
			A list of channels on the detector that should be masked. Number convention to specify which oof the 8 channels per quadrant on the detector are defined in the mask_bad_channels function
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

def calibrate_sequence(raw_dir, calib_dir, science_sequence, flat, dark, bp, hp,
	bkg, destripe, style, background_mode, correct_nonlinearity,
	nonlinearity_fname, mcf, covariates, mask_channels):
	"""Calibrates all the raw science sequence images, and returns an array of scalar values for each of the specified covariates in each image
	
	Parameters
	------------
	raw_dir : string
			Path to the directory in which all the raw data are stored
	calib_dir : string
			Path to the directory into which the calibrated data will be stored
	science_sequence : tuple
			A  tuple (int1, int2) where int1 and int2 are the image numbers for a linear sequence of raw science images to be corrected. 
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
			Flag that indicates whether raw images should be 'destriped'. Setting the flag subtracts the median value from each row or column in each quadrant. Default is False.
	style : string, optional
			Prefix convention used in naming the image. Usually 'image' or 'wirc' unless otherwise specified during observations
	background_mode : string or None, optional
			Either None, 'median', 'global', or 'helium' indicating the background subtraction procedure. 'median' subtracts a sigma-clipped median from the whole image, 'global' subtracts a calibrated background frame from each image, and 'helium' subtracts a calibrated background frame combined with a multicomponent frame to capture the unique helium filter structure.
	correct_nonlinearity : boolean, optional
			Flag that indicates whether or not to do nonlinearity correction on the image data (in the case of bright targets approaching the saturation limit of the detector)
	nonlinearity_fname : string, optional
			Path to the file name with nonlinearity array, by default None
    mcf : numpy.ndarray, optional
            2048 x 2048 numpy array representing radial distances from the filter center of the helium arc, used for correcting brightness variation due to the helium arc structure, by default None
    covariates : dictionary, optional
            Dictionary containing the covariate types as keys with empty values to be filled, by default None
	mask_channels : list, optional
			A list of channels on the detector that should be masked. Number convention to specify which oof the 8 channels per quadrant on the detector are defined in the mask_bad_channels function

	Returns
	--------------
    covariates : dictionary
            Covariates values of the specified covariance type for each science image in the sequence
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
	"""Checks that the dark and flat files have already been created and all related metadata stored in the image fits header are saved.

	Parameters
	--------------------
	dirname : string
			Path to the directory in which all combined darks and flats are stored
	dark_seqs : list of tuples
			List of (int1, int2) tuples corresponding to starting and ending image numbers of the dark image sequence(s). List length may be longer than 1 if the raw images were taken in multiple sequences and settings.
	flat_seq : tuple of ints
			List of (int1, int2) tuples corresponding to starting and ending image numbers of the flat image sequence
	style : string, optional
			Prefix convention used in naming the image. Usually 'image' or 'wirc' unless otherwise specified during observations
	Returns
	-----------------
	flat : string
			Path to the combined flat image
	darks : list of strings
			List of paths to the combined dark images
	bpname : string
			Path to the bad pixel file from the combined flat
	hps : list of strings
			Paths to the hot pixel file from the combined darks
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
	"""Checks that the sky background image has already been created and saved

	Parameters
	----------
	imname : string
			Path to the background image file

	Returns
	-------
	imname : string
			Path to the background image file
	"""
	hdul = fits.open(imname)
	hdul.close()
	print("Loaded saved background file...")
	return imname

###Flats and Darks###

def make_darks_and_flats(dirname, calib_dir, dark_seqs, dark_for_flat_seq,
	flat_seq, style, remake_darks_and_flats = True):
	"""Creates combined dark, dark for flat, and combined flat for calibrating the raw science images.
	
	Parameters
	-------------
	dirname : string
			Path to the directory in which all darks and flats are stored
	calib_dir : string
			Path to the directory in which the dark and flat image data will be stored
	dark_seqs : list of tuples
			List of (int1, int2) tuples. List is usually of length 1 if all raw images were taken with the [*same exposure time*] in one setting. List length may be longer than 1 if the raw images were taken with different [exposure times] in multiple sequences and settings. Each tuple defines a single linear sequence of darks from which a combined dark will be constructed
	dark_for_flat_seq : tuple of ints
			Tuple (int1, int2) that defines a linear sequence of darks from which a combined dark will be created to correct the flat
	flat_seq : tuple of ints
			Tuple (int1, int2) that defines a linear sequence of flat from which a combined flat will be created
	style : string, optional
			Prefix convention used in naming the image. Usually 'image' or 'wirc' unless otherwise specified during observations
	remake_darks_and_flats : boolean, optional
			Flag that indicates whether or not to remake the combined darks and flats. Default is True

	Returns
	--------------
	flat : string
			Path to the combined flat image
	darks : list of strings
			List of paths to the combined darks. Usually just one element.
	bp : string
			Path to the bad pixel file from the combined flat
	hps : list of strings
			Paths to the hot pixel files from the combined darks. Also usually just one element.
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
	"""Constructs a combined frame, given either a dark or flat image sequence

	Parameters
	-------------
	dirname : string
			Path to the directory in which raw frames are stored
	calib_dir : string
			Path to the directory you want combined frame saved (this should be the same as the calibrated image directory)
	seq_start : int
			Starting image number for the sequence
	seq_end : int
			Ending image number for the sequence
	calibration : string
			Either 'dark' or 'flat'
	dark_for_flat_name : string
			Path to the dark used to correct flat-fielding images; must not be None if making a flat
	style : string, optional
			The prefix for the image number. usually 'image' or 'wirc' unless otherwise specified during observations. 

	Returns
	--------------
	savename : string
			Path to the combined image frame
	bpname : string
			Path to the bad/hot pixel file from the combined frame
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
	"""
	Identify pixel values that are more than sig_hot_pix away from the median and then mask them (turn them into NAN with sigma clipping). Originally implemented by the WIRC+Pol team

	Parameters
	--------------------
	dark_stack : numpy.ndarray of floats
			3D array of the stacked dark images
	sig_hot_pix : int, optional
			sigma bound for the hot pixel value, by default 5

	Returns
	--------------------
	numpy array of booleans
			Each boolean value corresponds to a pixel in the 2048x2048 array. 0 indicates a good pixel, 1 indicates a hot pixel
	"""

	MAD = median_absolute_deviation(dark_stack, axis = 2)  

	# median_absolute_deviation FROM ASTROPY module measure std in images. this will tell us where are the hot pixels. 2 means it's a 2 D matrix - find the hot pix in the 2D array 
	hot_px = sigma_clip(MAD, sigma = sig_hot_pix)   #sigma_clip another AstroPy module goes through img and clips out any img above the treshhold
	# print("\n")
	# print("hotpixel \n")
	# print(type(hot_px.mask))
	# print(np.array(hot_px.mask, dtype = 'int'))
	# print("\n")
	return np.array(hot_px.mask, dtype = 'int') # hot_px.mask -> mask is a built-in python object returning the same shape as obj you're applying to. (boolean arr?) all entries are true . return 2D array where hot pixel found, ret true, where good pixel found, false

def stddevFilter(img, box_size):
	"""
	Compute the local standard deviation of the  pixel count in an image in a moving box of given size. Originally implemented by the WIRC+Pol team (stackoverflow 28931265)

	Parameters
	--------------------
	img : numpy.ndarray of floats
			2048 x 2048 2D image array of pixel counts
	box_size : int
			Side length of a box for the moving median across the image used for calculating the local standard deviation

	Returns
	--------------------
	float
			Standard deviation of the pixel value from the median value within the moving box of an image
	"""
	wmean, wsqrmean = (cv2.boxFilter(x, -1, (box_size, box_size), 
		borderType=cv2.BORDER_REFLECT) for x in (img, img*img))
	return np.sqrt(wsqrmean - wmean*wmean)

def get_bad_px(flat, local_sig_bad_pix = 3, global_sig_bad_pix = 9,
	local_box_size = 11):
	"""
	Originally implemented by the WIRC+Pol team
	Parameters
	----------
	flat : numpy.ndarray of floats
			Array containing data of the .fits file for the flat image
	local_sig_bad_pix : int, optional
			Pixel values with this number of standard deviations above the local surrounding pixels will be marked as bad pixels, by default 3
	global_sig_bad_pix : int, optional
			Pixel values with this many standard deviations above the median of the whole image will be marked as bad, by default 9
	local_box_size : int, optional
			Pixel box size from which the local average standard deviation of pixels is taken, by default 11

	Returns
	-------
	numpy.ndarray of booleans
			2048 x 2048 array where each boolean value correspond to a pixel. 0 indicates a good pixel and 1 indicates a broken pixel, i.e. bad.
	"""

	median_flat = median_filter(flat, local_box_size)
	stddev_im = stddevFilter(flat, local_box_size)
	local_bad_pix = np.abs(median_flat - flat) > local_sig_bad_pix*stddev_im
	pix_to_pix = flat/median_flat
	global_bad_px = sigma_clip(pix_to_pix, sigma = global_sig_bad_pix).mask
	non_positive = flat <= 0

	bad_px = np.logical_or(global_bad_px, local_bad_pix)
	bad_px = np.logical_or(bad_px, non_positive)
	# print("\n")
	# print("return \n")
	# print(type(np.array(bad_px, dtype = 'int')))
	# print(np.array(bad_px, dtype = 'int'))
	# print("\n")
	return np.array(bad_px, dtype = 'int')

def clean_bad_pix(image, bad_px_map, replacement_box = 5):
	"""
	Takes input image data and cleans any bad pixels in the bad pixel map by turning them into the median of their neighbors
	Originally implemented by WIRC+Pol team

	Parameters
	----------
	image : numpy.ndarray of floats
			2D array of floats containing data of the .fits file
	bad_px_map : numpy.ndarray of booleans
			Boolean flag for each pixel indicating whether or not the pixel is bad, i.e. broken
	replacement_box : int, optional
			Box size around which we use to replace the bad pixel so if you have a bad pixel, then we take a box of 5 pix by 5 pix (by default) and take the avg of those to replace the bad pixel with

	Returns
	-------
	numpy.ndarray of floats
			masked 2D array of floats containing data of the .fits file, where bad pixels are masked
	"""

	# print("\n")
	# print("image \n")
	# print(type(image))
	# print(image)
	# print("\n")

	# print("\n")
	# print("bad_px_map \n")
	# print(type(bad_px_map))
	# print(bad_px_map)
	# print("\n")

	bad_px_map = np.logical_or(bad_px_map, image <= 0)
	bad_px_map = np.logical_or(bad_px_map, ~np.isfinite(image))
	med_fil = median_filter(image, size = replacement_box)
	cleaned = image*~bad_px_map + med_fil*bad_px_map

	# print("\n")
	# print("return \n")
	# print(type(cleaned))
	# print(cleaned)
	# print("\n")
	return cleaned

###Background construction###

def make_calibrated_bkg_image(data_dir, calib_dir, bkg_seq, dark_ranges, 
	dark_for_flat_range, flat_range, naming_style = 'wirc', 
	nonlinearity_fname = None, sigma_lower = 5, 
	sigma_upper = 3, plot = False, remake_bkg = False,
	remake_darks_and_flats = False):
	"""
	Create a calibrated sky background image from the sky background frames

	Parameters
	--------------------
	data_dir : string
			Path to the data directory
	calib_dir : string
			Path to the calibrated directory in which the calibrated background image will be stored
	bkg_seq : list of tuples
			List of (int1, int2) tuples corresponding to the starting and ending background image numbers used to construct background frames
	dark_ranges : list of tuples
			List of (int1, int2) tuples corresponding to starting and ending image numbers of the dark image sequence(s)
	dark_for_flat_range : list of tuples
			List of (int1, int2) tuples corresponding to starting and ending image numbers of the dark image sequence(s) to be used for the flat sequences. 
	flat_range : tuple of ints
			Tuple (int1, int2) corresponding to starting and ending image numbers of the flat image sequence.
	naming_style : string, optional
			Prefix convention used in naming the image, by default 'wirc', now more commonly 'image'
	nonlinearity_fname : string, optional
			Path to the file name with nonlinearity array, by default None
	sigma_lower : int, optional
			Sigma floor to be clipped where outliers are removed from the sky background frame, by default 5
	sigma_upper : int, optional
			Sigma ceiling to be clipped where outliers are removed from the sky background frame, by default 3
	plot : bool, optional
			Flag that indicates whether or not to plot all the individual background frames, by default False
	remake_bkg : bool, optional
			Flag that indicates whether or not to remake the sky background image, by default False
	remake_darks_and_flats : bool, optional
			Flag that indicates whether or not to remake the darks and flats, by default False

	Returns
	--------------------
	string
			Path to the final calibrated background image
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

	print("\n")
	print("return imname \n")
	print(type(imname))
	print(imname)
	print("\n")
	return imname

###Image calibration###

def get_bjd(header):
	"""
	Get the UT timestamp of the image from the image header and return it as a BJD time.

	Parameters
	----------
	header : astropy.io.fits.header.Header
			Header containing metadata of the image taken

	Returns
	-------
	float
			Datetime in BJD floating point value for the image 
	"""
	# print("\n")
	# print("header \n")
	# print(type(header))
	# print(header)
	# print("\n")
	date_in = header['UTSHUT']
	target_pos = coord.SkyCoord(header['RA'], header['DEC'],
		unit = (u.hourangle, u.deg), frame='icrs')
	palomar = coord.EarthLocation.of_site('Palomar')
	time=ap_time.Time(date_in, format='isot', scale='utc', location=palomar)
	half_exptime=0.5*header['EXPTIME']*header['COADDS']/(24*3600)
	ltt_bary = time.light_travel_time(target_pos)
	time = time.tdb+ltt_bary
	bjd_tdb = time.jd+half_exptime

	# print("\n")
	# print("return \n")
	# print(type(bjd_tdb))
	# print(bjd_tdb)
	# print("\n")
	return bjd_tdb
	
def calibrate_image(im_name, flat, dark, bp, hp, correct_nonlinearity = False,
	nonlinearity_array = None, destripe = False, background_mode = None,
	background_frame = None, multicomponent_frame = None,
	covariate_dict = None, mask_channels = []):
	"""
	Helper function for calibrate_sequence() to calibrate an individual raw science image.

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
			Flag indicating whether nonlinearity array calibration is needed due to the presence of very bright/saturated objects/stars in the image, by default False
	nonlinearity_array : array, optional
			The array storing coefficients for the nonlinearity correction, by default None
	destripe : bool, optional
			Flag indicating if the detector should be destriped (setting the flag subtracts the median value from each row or column in each quadrant), by default False.
	background_mode : string, optional
			'median', 'global', or 'helium', indicating the type of the sky background subtraction, by default None
	background_fname : string, optional
			Path to the filename for the sky background image, by default None
	multicomponent_frame : numpy.ndarray, optional
			2048 x 2048 numpy array representing radial distances from the filter center of the helium arc, used for correcting brightness variation due to the helium arc structure, by default None
	covariate_dict : dictionary, optional
			Dictionary containing the covariate types as keys with empty values to be filled, by default None
	mask_channels : list, optional
			A list of channels on the detector that should be masked. Number convention to specify which oof the 8 channels per quadrant on the detector are defined in the mask_bad_channels function


	Returns
	--------------------
	cleaned : numpy.ndarray of floats
			Array of the cleaned image data 
	covariate_dict : dictionary
			Dictionary of covariates values for the specified covariance types
	"""

	print("\n")
	print("mult frame \n")
	print(type(multicomponent_frame))
	print(multicomponent_frame)
	print("\n")

	print("\n")
	print("background_frame \n")
	print(type(background_frame))
	print(background_frame)
	print("\n")
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

	# print("\n")
	# print("return \n")
	# print(type(cleaned))
	# print(cleaned)
	# print("\n")

	# print("\n")
	# print("return \n")
	# print(type(covariate_dict))
	# print(covariate_dict)
	# print("\n")

	return cleaned, covariate_dict

def mask_bad_channels(cleaned, to_mask):
	"""Mask out any bad channel in the pixel array by turning all pixel counts in the malfunctioning channel into NaN. Ordering of the channels is as follows:

	Bottom left quad, bottom to top = 0-7
	Top right, bottom to top = 8-15
	Top left, left to right = 16-23
	Bottom right, left to right = 24-31

	(Bottom to top, left to right)

	Parameters
	-----------------
	cleaned : numpy.ndarray
			2048 x 2048 corrected science image that has been dark subtracted, flat fielded, bad/hot pixels removed
	to_mask : array of integers
			each integer corresponds to the channel to be masked

	Returns
	-----------------
	numpy.ndarray
			the masked image array
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
	"""Remove the radial arc and sky background effect that is present in helium images

	Parameters
	--------------------
	cleaned : numpy.ndarray
			2048 x 2048 corrected science image array that are dark and flat corrected and hot-pixel and bad-pixel removed
	background_frame : numpy.ndarray
			Data of the background sky image used for calibrating for sky background brightness and structure
	multicomponent_frame : numpy.ndarray
			Special background frame used for reducing the data taken with helium filter

	Returns
	--------------------
	numpy.ndarray
			Corrected science images adjusted by background frame and multicomponent background frame
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
	"""
	Destripe the detector by subtracting the median of each row/column from each quadrant. This works best after subtracting a background sky image.

	Originally implemented by the WIRC+Pol team; modified by Shreyas

	Parameters
	--------------------
	image : array of floats
			Array of pixel data corresponding to brightness values of the science image before the destriping
	sigma : int, optional
			Sigma floor to be clipped where outliers are removed from the sky background, by default 3
	iters : int, optional
			Number of iterations to remove the striping artifacts from the detector, by default 5

	Returns
	--------------------
	numpy.ndarray
			Destripped science image where the striping effects of the detector are removed.
	"""

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
	"""
	Correct an image whose pixel values have been skewed by a very bright target(s) that over-saturates the detector

	Parameters
	--------------------
	image : numpy.ndarray of floats
			2D array of floats containing image data of the .fits file
	header : astropy.io.fits.header.Header
			Header of the fits file, containing metadata of the image
	nonlinearity_arr : array of float
			2D numpy array containing coefficients for the quadratic nonlinearity correction

	Returns
	--------------------
	numpy.ndarray
			The corrected image data where nonlinearity have been adjusted
	"""
	assert np.shape(nonlinearity_arr) == np.shape(image)
	n_coadd = header['COADDS']
	image_copy = np.array(image, dtype = float) #copy image
	image_copy /= n_coadd
	image_copy = (-1 + np.sqrt(1 + 4*nonlinearity_arr*image_copy)) / \
		(2*nonlinearity_arr) #quadratic formula with correct root
	return image_copy * n_coadd

def construct_multicomponent_frame(calib_dir, dump_dir, rstepsize = 10):
	"""
	Create multicomponent frame for removing the arc-effect that appears in helium-background images.

	Parameters
	--------------------
	calib_dir : string
			Path to the data directory
	dump_dir : string
			Path to directory in which the multicomponent framed background will be stored
	rstepsize : int, optional
			Radial step size in pixels across the detector, by default 10

	Returns
	--------------------
	numpy.ndarray
			distances of each pixel from the radial filter center
	"""
	home = (1037, 2120)  # center of the circle/arc
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
	"""
	Calculate the radial distance bwteeen two point coordinates p1 and p2 that each have x and y coordinates.

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
	return np.sqrt((p2[1] - p1[1]) ** 2 + (p2[0] - p1[0]) ** 2)

