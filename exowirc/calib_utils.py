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
	nonlinearity_fname = None):
	"""Calibrates all science images. 
	
	Parameters
	------
	raw_dir : string
		path to the directory in which all the raw data are stored
	calib_dir : string
		path to the directory into which the calibrated data will be
		stored
	dump_dir : string
		path to directory in which the saved background values will be
		stored
	science_ranges : list of tuples
		list of (int1, int2) tuples, where each tuple defines a
		single linear sequence of raw science frames to be calibrated
	dark_ranges : list of tuples
		list of (int1, int2) tuples of length 1 (if all science
		sequences were taken with the same exposure time) or length
		len(science_ranges) (if the science sequences were taken with
		different exposure times). Each tuple defines a single linear
		sequence of darks from which a combined dark will be constructed
		and used to calibrate the science sequence
	dark_for_flat_range : tuple of ints
		a tuple (int1, int2) that defines a linear sequence of darks
		from which a combined dark will be created to correct the flat
	flat_range : tuple of ints
		a tuple (int1, int2) that defines a linear sequence of flats
		from which a combined flat will be created
	destripe : boolean, optional
		flag that indicates whether raw images should be 'destriped'
		(aka bias levels from each readout subtracted) or not
	style : string, optional
		the prefix for the image number. usually 'image' or 'wirc'
		unless otherwise specified during observations
	background mode : string or None, optional
		either None, 'median', 'global', or 'helium' indicating the 
		background subtraction procedure
	bkg_filename : string, optional
		path to a reduced background frame
	correct_nonlinearity : boolean, optional
		whether or not to do nonlinearity correction on the data
	remake_darks_and_flats : boolean, optional
		whether or not to remake the darks and flats
        nonlinearity_fname : string or None, optional
		path to the file with the nonlinearity correction coefficients
	
	Returns
	-------
	calib_dir : string
		path to the directory contained calibrated science images
	"""
	assert (len(science_ranges) == len(dark_ranges)) or \
		(len(dark_ranges) == 1)

	#making/loading darks and flats
	flat, darks, bp, hps = make_darks_and_flats(raw_dir, dark_ranges,
		dark_for_flat_range, flat_range, style, remake_darks_and_flats)

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
			nonlinearity_fname, mcf, covariates)

	save_covariates(dump_dir, covariates)

	print("CALIBRATION COMPLETE")

	return calib_dir

def calibrate_sequence(raw_dir, calib_dir, science_sequence, flat, dark, bp, hp,
	bkg, destripe, style, background_mode, correct_nonlinearity,
	nonlinearity_fname, mcf, covariates):
	"""Calibrates all images in a science sequence.
	
	Parameters
	------
	raw_dir : string
		path to the directory in which all the raw data are stored
	calib_dir : string
		path to the directory into which the calibrated data will be
		stored
	science_sequence : tuple of ints
		a tuple (int1, int2) that defines a linear sequence of raw
		science images to be corrected
	flat : string
		path to the combined flat that will be used to calibrate the
		science sequence
	dark : string
		path to the combined dark that will be used to calibrate the
		science sequence
	bp : string
		path to the bad pixel file that will be used to calibrate the
		science sequence
	hp : string
		path to the hot pixel file that will be used to calibrate the
		science sequence
	bkg : string or None
		path to the sky background file that will be used to calibrate
		the science sequence
	destripe : boolean
		flag that indicates whether raw images should be 'destriped'
		(aka bias levels from each readout subtracted) or not
	median_sub : boolean
		flag that indicates whether the sigma-clipped median of the
		image should be subtracted during calibration or not
	style : string
		the prefix for the image number. usually 'image' or 'wirc'
		unless otherwise specified during observations
	save_bkg : boolean
		flag that indicates whether median values subtracted as
		background should be saved or not. If True, saves to dump_dir

	Returns
	-------
	bkgs : array of floats
		if save_bkg is True, this will be an array of floats containing
		the saved median background of each image. if save_bkg is False,
		this will just be an empty array.
	"""
	flat, dark, bp, hp, nonlinearity_array, correct_nonlinearity = \
		load_calib_files(flat,dark,bp,hp,nonlinearity_fname)
	with fits.open(bkg) as hdul:
		background_frame = hdul[0].data
	
	for i in science_sequence:
		image = get_img_name(raw_dir, i, style = style) 
		print(f"Reducing {image}...")
		calib, covariates = calibrate_image(image, flat, dark, bp, hp,
			correct_nonlinearity = correct_nonlinearity,
			nonlinearity_array = nonlinearity_array,
			destripe = destripe, background_mode = background_mode,
			background_frame = background_frame,
			multicomponent_frame = mcf, covariate_dict = covariates)
		outname = get_img_name(calib_dir, i, style = style)
		save_image(calib, outname)

	return covariates 

###Checking saved versions###
def check_saved(dirname, dark_seqs, flat_seq, style):
	darks = []
	hps = []
	for dark in dark_seqs:
		num = dark[-1]
		zeros = '0'*(4-len(str(num)))
		darkname = f'{dirname}{style}{zeros}{num}' + \
			'_combined_dark.fits'
		hpname = f'{dirname}{style}{zeros}{num}' + \
			'_combined_hp_map_test.fits'
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
		'_combined_bp_map_test.fits'
	hdul = fits.open(flatname)
	hdul.close()
	hdul = fits.open(bpname)
	hdul.close()

	print("Loaded saved darks and flats...")
	return flatname, darks, bpname, hps

def check_saved_background(imname):
	hdul = fits.open(imname)
	hdul.close()
	print("Loaded saved background file...")
	return imname

###Flats and Darks###

def make_darks_and_flats(dirname, dark_seqs, dark_for_flat_seq,
	flat_seq, style, remake_darks_and_flats = True):
	"""Creates combined dark, dark for flat, and combined flat.
	
	Parameters
	------
	dirname : string
		path to the directory in which all darks and flats are stored
	dark_seqs : list of tuples
		list of (int1, int2) tuples, where each tuple defines a
		single linear sequence of darks from which a combined dark will
		be created
	dark_for_flat_seq : tuple of ints
		a tuple (int1, int2) that defines a linear sequence of darks
		from which a combined dark will be created to correct the flat
	flat_seq : tuple of ints
		a tuple (int1, int2) that defines a linear sequence of flats
		from which a combined flat will be created
	style : string, optional
		the prefix for the image number. usually 'image' or 'wirc'
		unless otherwise specified during observations. 

	Returns
	-------
	flat : string
		path to the combined flat
	darks : list of strings
		list of paths to the combined darks
	bp : string
		path to the bad pixel file from the combined flat
	hps : list of strings
		paths to the hot pixel files from the combined darks
	"""
	#check if saved versions exist
	if not remake_darks_and_flats:
		try:
			return check_saved(dirname, dark_seqs,
				flat_seq, style)
		except FileNotFoundError as e:
			print("Can't find saved darks/flats -- remaking...")

	temp,_ = make_combined_image(dirname, *dark_for_flat_seq,
		style = style, calibration = 'dark')
	print('DARK FOR FLAT CREATED')
	flat,bp = make_combined_image(dirname, *flat_seq,
		style = style, calibration = 'flat', dark_for_flat_name = temp)
	print('COMBINED FLAT CREATED')
	
	darks = []
	hps = []
	for seq in dark_seqs:
		dark, hp = make_combined_image(dirname, *seq, style = style,
			calibration = 'dark')
		darks.append(dark)
		hps.append(hp)
		print('COMBINED DARK CREATED')
			
	return flat, darks, bp, hps

def make_combined_image(dirname, seq_start, seq_end, calibration = 'dark',
	dark_for_flat_name = None, style = 'wirc'):
	"""Given a dark or flat sequence, constructs a combined frame.

	Parameters
	------
	dirname : string
		path to the directory in which darks are stored
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
	-------
	savename : string
		path to the combined frame
	bpname : string
		path to the bad/hot pixel file from the combined frame
	"""
	zeros = '0'*(4 - len(str(seq_end)))

	image_list = [get_img_name(dirname, i, style = style) for \
		i in range(seq_start, seq_end + 1)]
	stack = np.zeros([2048, 2048, len(image_list)])
	for i,name in enumerate(image_list):
		with fits.open(name) as hdul:
			temp = hdul[0].data
		print(f"Stacking image {name}...")
		if calibration == 'dark':
			stack[:,:,i] = temp
		else:
			with fits.open(dark_for_flat_name) as hdul:
				dark_for_flat = hdul[0].data
			dark_corr = temp - dark_for_flat
			stack[:,:,i] = dark_corr/np.nanmedian(
				dark_corr.flatten())
	combined = np.nanmedian(stack, axis = 2)
	if calibration == 'dark':
		print("Generating hot pixel map...")
		hp = get_hot_px(stack) 
		bpname = f'{dirname}{style}{zeros}{seq_end}'
		bpname += f'_combined_hp_map_test.fits'
		save_image(hp, bpname)
	else:
		print("Generating bad pixel map...")
		bp = get_bad_px(combined)
		bpname = f'{dirname}{style}{zeros}{seq_end}'
		bpname += f'_combined_bp_map_test.fits'
		save_image(bp, bpname)
		bp = np.array(bp, dtype = 'bool')
		med = np.nanmedian(combined[~bp])
		combined = combined/med
	savename = f'{dirname}{style}{zeros}{seq_end}'
	savename += f'_combined_{calibration}.fits'
	save_image(combined, savename)
	return savename, bpname

def get_hot_px(dark_stack, sig_hot_pix = 5):
	"original implement by WIRC+Pol team"
	MAD = median_absolute_deviation(dark_stack, axis = 2)
	hot_px = sigma_clip(MAD, sigma = sig_hot_pix)
	return np.array(hot_px.mask, dtype = 'int')

def stddevFilter(img, box_size):
	""" from stackoverflow 28931265, implemented by WIRC+Pol team
	computes standard deviation of image in a moving box of given size. 
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

	image_list = [get_img_name(data_dir, i,
		style = naming_style) for \
		i in range(bkg_seq[0], bkg_seq[1] + 1)]
	imname = get_img_name(calib_dir, bkg_seq[-1],
		style = naming_style, img_type = 'calibrated_background_test')
	if not remake_bkg:
		try:
			return check_saved_background(imname)
		except FileNotFoundError as e:
			print("Can't find saved background -- remaking...")

	print("Creating background frame...")
	#create/load up flats and darks
	flat, darks, bp, hps = make_darks_and_flats(data_dir, dark_ranges,
		dark_for_flat_range, flat_range, naming_style,
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
	covariate_dict = None):
	"image is file flat dark are numpy arrays"
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

def helium_background_subtraction(cleaned, background_frame,
	multicomponent_frame):
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

	legacy code from WIRC+Pol"""

	quad1 = image[:1024,:1024]
	quad2 = np.rot90(image[1024:,:1024], k=3, axes=(0, 1))
	quad3 = np.rot90(image[1024:,1024:], k=2, axes=(0, 1))
	quad4 = np.rot90(image[:1024,1024:], k=1, axes=(0, 1))
	mn_quad = np.median([quad1,quad2,quad3,quad4],axis=0)
	clean_imm = np.array(image)

	for i in range(1024):
		to_sub = sigma_clipped_stats(
			mn_quad[:,i],sigma=sigma,maxiters=iters)[1]
		clean_imm[:1024,i]  -= to_sub
		clean_imm[1024:,-i] -= to_sub
		clean_imm[-i,:1024] -= to_sub
		clean_imm[i,1024:]  -= to_sub

	return clean_imm

def nonlinearity_correction(image, header, nonlinearity_arr):
	assert np.shape(nonlinearity_arr) == np.shape(image)
	n_coadd = header['COADDS']
	image_copy = np.array(image, dtype = float) #copy image
	image_copy /= n_coadd
	image_copy = (-1 + np.sqrt(1 + 4*nonlinearity_arr*image_copy)) / \
		(2*nonlinearity_arr) #quadratic formula with correct root
	return image_copy * n_coadd

def construct_multicomponent_frame(calib_dir, dump_dir, rstepsize = 10):
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
	return np.sqrt((p2[1] - p1[1])**2 + (p2[0] - p1[0])**2)

