import numpy as np
from astropy.stats import mad_std, sigma_clipped_stats
from scipy.stats import sigmaclip
from functools import reduce
from astropy.io import fits
from scipy.optimize import curve_fit
from photutils.utils import calc_total_error
import photutils

from .plot_utils import plot_sources 
from .io_utils import get_science_img_list, init_phot_dirs, load_calib_img, \
	load_bkgs, load_multicomponent_frame, save_phot_data 

def find_sources(image, fwhm = 20., sigma_threshold = 20.):
	"""Using the photutils DAOStarFinder algorithm, automatically
	identify sources of a certain FWHM and a certain SNR.

	Parameters
	------------
	image : numpy.ndarray, shape(2048, 2048)
			The finding frame in which the sources will be located
	fwhm : float, optional
			The approximate FWHM of all the sources to be detected
	sigma_theshold : float, optional
			The number of sigmas that a source needs to be to be detected

	Returns
	-------------
	sources : astropy.Table
			A Table of the detected sources with xcentroid, ycentroid, and some other auxilliary stats
	"""
	bkg_sigma = mad_std(image)
	daofind = photutils.DAOStarFinder(fwhm = fwhm,
		threshold = sigma_threshold*bkg_sigma)
	sources = daofind(image)
	return sources

def accurate_cent(data, xc, yc, radius=3.0, max_iter=100, max_pos_error=1.e-2):
	"""Iterative flux-weighted centroiding for getting the apertures placed
	precisely.

	Parameters
	------
	data : array_like, shape(N, N)
			The cutout on which the centroiding will be performed
	xc : float
			Initial guess at the x coordinate of the centroid
	yc : float
			Initial guess at the y coordinate of the centroid
	radius : float, optional
			Pixel radius from the initial guess centroid that the calculated centroid is expected to reside on the image, default is 3.
	max_iter : int, optional
			The max number of iterations of centroiding to perform (The actual number of iterations may be less). Default is 100.
	max_pos_error : float, optional
			The maximimum positional error allowed before returning the calculated centroid, default is 0.01.

	Returns
	-------
	xc : float
			x centroid
	yc : float
			y centroid
	"""
	data = np.array(data)
	xs = np.array([np.arange(data.shape[1])] * data.shape[0])
	ys = xs.transpose()

	for i in range(max_iter):
		aperture = photutils.CircularAperture((xc, yc), radius)
		flux = (photutils.aperture_photometry(data,
			aperture)['aperture_sum'][0])
		old_xc = xc
		old_yc = yc
		xc = (photutils.aperture_photometry(data * xs,
			aperture)['aperture_sum'][0] / flux)
		yc = (photutils.aperture_photometry(data * ys,
			aperture)['aperture_sum'][0] / flux)
		if np.isnan(xc) or np.isnan(yc):
			break
		last_pos_err = np.sqrt((xc - old_xc) ** 2 + (yc - old_yc) ** 2)
		if last_pos_err < max_pos_error:
			break

	if last_pos_err < max_pos_error:
		return xc, yc
	print("couldn't converge on the source")
	return old_xc, old_yc

def get_aperture_sum(sources, image, radii = [10.], error = None,
	ann_rads = (25, 50), target_ind = 0):
	"""Given a list of sources, re-calculates image centroids via flux-weighted centroiding and performs aperture photometry on all the sources. All counts in the aperture are summed, and the local background is
	estimated using an annulus.

	Parameters
	------
	sources : astropy.Table or dict
			Table of source locations with x and y centroids
	image : numpy.ndarray, shape(2048, 2048)
			The finding frame in which the sources will be located
	radius : numpy.ndarray, optional
			Radii of the apertures for the photometry
	error : None or numpy.ndarray, shape(2048, 2048), optional
			If None, errors will not be calculated during photometry. If numpy.ndarray, the errors will be used to produce error estimates on the photometry
	ann_rads : tuple, optional
			Tuple of form (float1, float2), where float1 specifies the inner radius and float2 specifies the outer radius of the annulus that will be used for local background subtraction
	target_ind : int
			Index location of the target in the list of sources. Default to 0 because index loaction of the source coordinate is conventionally marked at the 0th index, and function find_my_source() moves the source with target coordinate to the top of the list

	Returns
	-------
	phot_table : astropy.Table
			Table with aperture sums for each radius and errors (if specified by input params)
	xs : numpy.ndarrays
			Array of x centroids of all the sources
	ys : numpy.ndarrays
			Array of y centroids of all the sources
	widths : numpy.ndarrays
			Array of PSF widths for all the sources
	"""
	radii = list(radii)
	max_rad = max(radii)
	image = np.nan_to_num(image)
	img_arrs = make_img_arrs(sources, max_rad*2, image)
	xs = []
	ys = []
	widths = []
	
	#calculating centroids and widths for all sources
	for i, arr in enumerate(img_arrs):
		x_coord_ref = int(np.array(sources['xcentroid'])[i])
		y_coord_ref = int(np.array(sources['ycentroid'])[i])
		#when background subtraction is poor, flux-weighted
		#centroiding will fail. below is a quick fix.
		temp_arr = arr - sigma_clipped_stats(arr, sigma = 2)[1]
		xval, yval = accurate_cent(temp_arr, arr.shape[0]/2,
			arr.shape[1]/2, max_rad)
		x_centroid = x_coord_ref + xval - arr.shape[0]/2
		y_centroid = y_coord_ref + yval - arr.shape[1]/2
		width = fit_cut(arr, xval, yval)
		if np.isnan(x_centroid) or np.isnan(y_centroid):
			x_centroid = x_coord_ref
			y_centroid = y_coord_ref
		if np.abs(x_centroid - 1024) > 1024 or \
			np.abs(y_centroid - 1024) > 1024:
			print("Critical failure in centroiding")
			print(x_centroid, y_centroid)
			print(x_coord_ref, y_coord_ref)
			print(xval, yval)
		xs.append(x_centroid)
		ys.append(y_centroid)
		widths.append(width)

	xs = np.array(xs)
	ys = np.array(ys)
	widths = np.array(widths)
	positions = [(x,y) for x, y in zip(xs, ys)]
	apertures = [photutils.CircularAperture(positions, r = rad) \
		for rad in radii]
	phot_table = photutils.aperture_photometry(image, apertures,
		error = error)

	#estimating local background using an annulus
	ann = [photutils.CircularAnnulus(positions, r_in = ann_rads[0], 
		r_out = ann_rads[1]) for rad in radii]
	ann_masks = [annulus.to_mask() for annulus in ann]
	local_bkgs = []
	for masks in ann_masks:
		local_bkgs_temp = []
		for i, mask in enumerate(masks):
			try:
				mask_data = mask.multiply(image)[mask.data > 0]
				flat_mask_data = mask_data.flatten()
				clipped_data, low, up = sigmaclip(
					flat_mask_data, low = 2.0, high = 2.0)
				local_bkgs_temp.append(np.median(clipped_data))
			except TypeError as e:
				print("Annulus local background failed")
				local_bkgs_temp.append(0.)
		local_bkgs_temp = np.array(local_bkgs_temp)
		local_bkgs.append(local_bkgs_temp)
	
	#perform the aperture photometry	- removing background from target aperature - and calculate the final flux value and add it to the photometry table
	ap_areas = np.array([aper.area for aper in apertures])
	for i, aperture_area in enumerate(ap_areas):
		table_ind = 'aperture_sum_' + str(i)
		extra_bkg_in_ap = aperture_area*local_bkgs[i]
		phot_table[table_ind] = phot_table[table_ind] - extra_bkg_in_ap

	return phot_table, np.array(xs), np.array(ys), np.array(widths)

# *p destructure a list into a, b, and c
def gauss(x, *p):
	"""Calculate a gaussian curve. Supports the scipy function curve_fit().

	Parameters
	----------
	x : float
			data point

	Returns
	-------
	float
			y data point fitted to the gaussian curve described by parameters *p
	"""
	# unpack the first three args of p as a, b, and c
	a, b, c = p
	return a*np.exp(-(x - b)**2 / (2 * c**2))

def fit_cut(arr, xval, yval):
	"""Fit a gaussian profile to the star's PSF and then cut the guassian profile to calculate the FWHM in function get_aperature_sum().

	Parameters
	----------
	arr : numpy.ndarray
			A localized 2D numpy image array of pixel brightness around the star. Supports the function make_image_arrs()
	xval : int
			x-coordinate
	yval : int
			y-coordinate

	Returns
	-------
	flat
			THE FWHM from the gaussian fit of the star's PSF
	"""
	p0 = [1000, arr.shape[0]/2, 5]
	popt, _ = curve_fit(gauss, np.arange(arr.shape[0]), arr[int(xval),:],
		p0, maxfev = 100000)
	return np.abs(popt[-1])

def make_img_arrs(sources, rad, img):
	"""Given a list of sources and the image, make a list of smaller cutouts centered on each source. The cutouts go from (center - radius) to (center + rad) in both the x and the y directions.

	Parameters
	------------
	sources : astropy.Table or dict
			Table of source locations with x and y centroids
	rad : int
			Half the size of one side of the cutout (or, if you prefer, the radius of the inscribed circle for the bounding box).
			Needs to be an int for indexing purposes.
	image : array_like, shape(2048, 2048)
			The image from which the cutouts will be cut out

	Returns
	-------------
	img_arrs : list of arrays, shape (2 * radius, 2 * radius)
			Array of cutouts for all the sources
	"""
	xpositions = np.array(sources['xcentroid'])
	ypositions = np.array(sources['ycentroid'])
	img_arrs = []
	rad = int(rad)
	for i, pos in enumerate(xpositions):
		xpos = int(pos)
		ypos = int(ypositions[i])
		img_arrs.append(img[ypos-rad:ypos+rad,xpos-rad:xpos+rad])
	return img_arrs

def init_data(n_sources, n_images, radii):
	"""Initializes all the data storage arrays and dictionaries.
	
	Parameters
	------------
	n_sources : int
			Number of sources on which the photometry is being performed
	n_images : int
			Number of images to perform photometry on
	radii : numpy.ndarray
			The radii of the apertures for photometry 

	Returns
	------------
	xpos_arr : numpy.ndarray, shape(n_sources, n_images)
			Storage array for x centroid positions
	ypos_arr : numpy.ndarray, shape(n_sources, n_images)
			Storage array for y centroid positions
	psf_widths : numpy.ndarray, shape(n_sources, n_images)
			Storage array for the widths of the source PSFs
	phot_dict : dict of numpy.ndarray, shape(n_sources, n_images)
			Dictionary mapping aperture radius to aperture sum for each source and image number
	err_dict : dict of numpy.ndarray, shape(n_sources, n_images)
			Dictionary mapping aperture radius to error on aperture sum for each source and image number
	"""
	xpos_arr = np.zeros([n_sources, n_images])
	ypos_arr = np.zeros([n_sources, n_images])
	psf_widths = np.zeros([n_sources, n_images])
	phot_dict = {r: np.zeros([n_sources, n_images]) for r in radii}
	err_dict = {r: np.zeros([n_sources, n_images]) for r in radii}
	return xpos_arr, ypos_arr, psf_widths, phot_dict, err_dict

def clean_sources(sources, fwhm, bad_channel = False):
	"""Cleans (i.e. removes) all sources close to the edge of the detector as well as sources that overlap with the apertures.
	
	Parameters
	------
	sources : astropy.Table or dict
			Table of source locations with x and y centroids
	fwhm : float
			Approximate FWHM of source PSFs. If sources are less than a FWHM from a detector edge, they'll automatically be removed.

	Returns
	-------
	sources : astropy.Table or dict
			Cleaned Table of source locations with bad sources removed.
	"""
	#First clean sources that are close to the detector edge
	xvals = np.array(sources['xcentroid']).astype(int)
	yvals = np.array(sources['ycentroid']).astype(int)
	to_remove = np.array([])
	badvals1 = np.where(xvals < fwhm)
	badvals2 = np.where(yvals < fwhm)
	badvals3 = np.where(xvals > (2048-fwhm))
	badvals4 = np.where(yvals > (2048-fwhm))
	to_remove = np.append(to_remove, badvals1)
	to_remove = np.append(to_remove, badvals2)
	to_remove = np.append(to_remove, badvals3)
	to_remove = np.append(to_remove, badvals4)

	if bad_channel:
		a = np.where(xvals > 1024)
		b = np.where(xvals < 2048)
		c = np.where(yvals > 1152)
		d = np.where(yvals < 1280)
		bad_channel_vals = reduce(np.intersect1d, (a, b, c, d))
		to_remove = np.append(to_remove, bad_channel_vals)

	#Then clean sources that overlap apertures
	badvals = []
	for i, xval in enumerate(xvals):
		#for every source, calculate distances to the other sources
		yval = yvals[i]
		distances = np.sqrt((xvals - xval)**2 + (yvals - yval)**2)
		for j in range(len(xvals)):
			if distances[j] < fwhm and distances[j] != 0.:
				#if this source encounters an overlap anywhere
				#remove the overlapper
				badvals.append(j)
	to_remove = np.append(to_remove, np.array(badvals))
	to_remove = np.array(to_remove, dtype = int)
	sources.remove_rows(np.unique(to_remove))
	return sources

def find_my_source(sources, target_coords, tolerance = 10):
	"""Given a list of sources and an initial guess for the coordinates of the target, determines the index of the target source.
	
	Parameters
	------
	sources : astropy.Table or dict
			Table of source locations with x and y centroids
	target_coords : tuple, shape:(2)
			An (x, y) tuple describing approximately where the target is located
	tolerance : float, optional
			The maximum distance away (in pixels) from the guess for a source to be considered correctly identified. If it's small, your guess better be really good. If it's big, be careful of additional nearby sources.

	Returns
	-------
	index : int or None
			If the source is found, this is the index of the source in the sources dict. If not, None will be returned.
	"""
	x, y = target_coords
	xs = np.array(sources['xcentroid'])
	ys = np.array(sources['ycentroid'])
	distances = np.sqrt((x - xs)**2 + (y - ys)**2)
	index = np.argmin(distances)
	min_distance = min(distances)
	if min_distance < tolerance:
		print("Found your source -- it's index", index)
		return index
	else:
		print("Unable to converge on source...")
		return None

def perform_photometry(calib_dir, dump_dir, img_dir, science_ranges,
	target_coords, finding_fwhm = 15., extraction_rads = [20.],
	style = 'wirc', source_detection_sigma = 50, max_num_compars = 10,
	gain = 1.2, bkg_fname = None, background_mode = None,
	ann_rads = (20, 50), target_and_compars = None, bad_channel = False):
	"""Given a list of science images, perform aperture photometry. First, sources are automatically detected and cleaned. Then run the aperture photometry with local background subtraction using a sigma-clipped annulus. The aperture sums and errors for each radius, as well as diagnostics like x centroid, y centroid, and PSF width, are pickled and saved for fitting.
	
	Parameters
	------------
	calib_dir : string
			Path to the directory holding the calibrated science images.
	dump_dir : string
			Path to the directory into which the pickled results will be saved.
	img_dir : string
			Path to the directory holding all the diagnostic plots that will be automatically generated.
	science_ranges : list of tuples
			List of (int1, int2) tuples, where each tuple defines a single linear sequence of science images
	target_coords : tuple, shape:(2)
			An (x, y) tuple describing approximately where the target is located on the detector
	finding_fwhm : float, optional
			The PSF FWHM used for automatic source detection
	extraction_rads : array_like, optional
			A list of radii defining target aperture sizes to be used for the photometry
	style : string, optional
			The prefix for the image number. usually 'image' or 'wirc' unless otherwise specified during observations
	source_detection_sigma : float, optional
			The number of sigmas that a source needs to be above the background to be detected
	max_num_compars : int, optional
			The maximum number of comparison stars to use.
	gain : float, optional
			The gain for the WIRC detector. Made it a free parameter but it's unlikely to change anytime soon, so probably leave this alone.
	bkg_frame : int, optional
			The number of the background frame used for calibration, if necessary. If background frame subtraction was performed during calibration and you want accurate photometric errors, you definitely should set this parameter.
	global_bkg_sub : boolean, optional
			If you chose to subtract a background frame or a sigma-clipped background from the science images during calibration, set this flag to True so that you can calculate accurate photometric errors.
	ann_rads : tuple, optional
			Tuple of form (float1, float2), where float1 specifies the inner radius and float2 specifies the outer radius of the annulus that will be used for local background subtraction
	target_and_compars : list of shape:(2) lists
			A list of [x, y] coordinates for manually selected target stars and comparison stars. Only select this if you don't want automatic source detection. The target star is assumed to come first in the list.

	Returns
	-------------
	clean_fnames : list of Strings
			A list of the paths to the pickled and saved photometry and error arrays as well as to the auxilliary arrays for the centroids and widths
	"""
	#initializing dirs and finding frame 
	to_extract = get_science_img_list(science_ranges)  # full range of science images
	n_images = len(to_extract)
	dump_dir_phot = init_phot_dirs(dump_dir, img_dir, extraction_rads)  # init_phot_dirs() create photometry folder
	finding_frame = load_calib_img(calib_dir, to_extract[0], style = style)  # load_calib_img() extract the first calibrated science img of the night, which is the finding_frame that identifies where the star and comparison stars are
	print(to_extract)  # range of science images to do photometry on

	#getting list of sources
	print(ann_rads)
	max_lengthscale = ann_rads[1]  # don't go past 50 (ann_rads: [20, 50])
	if target_and_compars is None:  # target_and_compars: list of lists of xy pixel coord of every compars in the image
		sources = find_sources(finding_frame, fwhm = finding_fwhm, sigma_threshold = source_detection_sigma)  # find them
		if sources is None:
			print("NO SOURCES FOUND!!") ## 
		source_ind_temp = find_my_source(sources, target_coords)
		sources = clean_sources(sources, max_lengthscale, bad_channel = bad_channel)
		source_ind = find_my_source(sources, target_coords)
		plot_sources(img_dir, finding_frame, sources, finding_fwhm, ann_rads)  # create an img of the annulus drawn on the stars
		n_sources = len(sources)
	else:  # in the case target_and_compars is manually defined, create a table of the sources
		x_stars = np.array([tup[0] for tup in target_and_compars])  # x_stars: x pixel value
		y_stars = np.array([tup[1] for tup in target_and_compars])  # 
		sources = {'xcentroid': x_stars, 'ycentroid': y_stars}  # sources table: used for photometry, contains location of all the stars
		source_ind = 0  # target star is index 0 of the list
		n_sources = len(x_stars)

	#initializing data storage arrays
	xpos, ypos, psf_widths, phot_dict, err_dict = init_data(n_sources,
		n_images, extraction_rads)
	if background_mode == 'helium' or background_mode == 'global':	
		bkg_arr = np.ones(np.shape(finding_frame))
		if bkg_fname is not None:	
			with fits.open(bkg_fname) as hdul:
				bkg_frame = hdul[0].data  # put image data into bkg_frame

	if background_mode is not None:
		bkgs = np.array(load_bkgs(dump_dir))

	#performing the extraction	
	for i, n_img in enumerate(to_extract):
		print('Extracting image ', n_img)
		image = load_calib_img(calib_dir, n_img, style = style)
		if background_mode == 'helium':
			mcf = load_multicomponent_frame(dump_dir)
			bkg_error_array = construct_bkg(bkg_arr, bkgs[i], mcf)
			bkg_error_array = np.sqrt(bkg_error_array/gain)  ## gain: prop of detector sensitivity for num of photoelectrons to make 1 count (Analog to Digital Unit ADU: 1 ADU per 1.2e-). photoelectric effect
			error = calc_total_error(image, bkg_error_array, gain)  # error in the sky background--std of the total img
		elif background_mode == 'global' or background_mode == 'median':
			#error is sqrt(N)
			if background_mode == 'global':
				bkg_arr = np.sqrt(bkg_arr)
			else:  # background = 'median', for case where we don't have a background file
				bkg_arr = np.ones((2048, 2048))
			bkg_errors = np.sqrt(bkgs / gain)
			bkg_error_array = bkg_arr * bkg_errors[i]
			error = calc_total_error(image,	bkg_error_array, gain)
		else:
			error = np.sqrt(image / gain)

		phot_table, xs, ys, widths = get_aperture_sum(sources, image, radii = extraction_rads, error = error, ann_rads = ann_rads, target_ind = source_ind)  # !!! get brightness, position, widths of the target and compars stars

		xpos[:,i] = xs  ## xpos is a 2D list. Fill all the rows, for col correspond to science image i, fill exposition with that star. xs is a 1D list
		ypos[:,i] = ys
		psf_widths[:,i] = widths ## "seeing"--clouds diffuse light, etc
		## for ea sci img record brightness of star, which depends on the aperature size. measure ea sci img for each aperture radius size.
		for j, rad in enumerate(extraction_rads):
			table_ind = 'aperture_sum_' + str(j)
			table_err_ind = 'aperture_sum_err_' + str(j)
			phot_dict[rad][:,i] = phot_table[table_ind]
			err_dict[rad][:,i] = phot_table[table_err_ind]

	#saving files for each extraction radius seperately - pickling to efficiently store data (compression? zip?)
	for i, rad in enumerate(extraction_rads):
		raw_phot = phot_dict[rad]
		errs = err_dict[rad]
		xpos_temp, ypos_temp, psf_widths_temp, raw_phot, errs = \
			get_source_first(source_ind, xpos, ypos, psf_widths,
			raw_phot, errs)
		dump_dir_temp = dump_dir_phot + str(rad) + '/'
		xpos_temp, ypos_temp, psf_widths_temp, raw_phot, errs = \
			reject_bad_trends(xpos_temp, ypos_temp,
			psf_widths_temp, raw_phot, errs, max_num_compars)

		fnames = save_phot_data(dump_dir_temp,
			xpos_temp, ypos_temp, psf_widths_temp,
			raw_phot, errs)
	
	print('DATA SAVED! EXTRACTION COMPLETE')
	
	return fnames

def construct_bkg(background, scale_factors, multicomponent_frame):
	"""Create a background frame specifically for the helium data.

	Parameters
	-----------------
	background : numpy.ndarray
			A stack of the background sky image data
	scale_factors : numpy.ndarray
			An array of scale factors for each of the sky images that are part of the background dither sequence, relating the individual image median value bacto that of the first frame in the background sequence
	multicomponent_frame : numpy.ndarray  
			2048 x 2048 numpy array representing radial distances from the filter center of the helium arc, used for correcting brightness variation due to the helium arc structure
            
	Returns
	--------------
	numpy.ndarray
			An updated stack of sky background image data adjusted in brightness by the scale factors and with the multicomponent frame
	"""
	new_bkg = np.zeros(background.shape)
	for i in range(scale_factors.shape[0]):
		working_mask = (multicomponent_frame == i + 1)
		new_bkg[working_mask] = \
			background[working_mask]*scale_factors[i]
	return new_bkg

def get_source_first(source_ind, xpos, ypos, widths, phot, errs):
	"""Moves the source to be index 0 in each of the photometry and auxilliary arrays. Bubbling up target star to be index 0 in the list of stars.
	
	Parameters
	------
	source_ind : int, optional	
			Index of the target trace for all of the arrays.
	xpos : array_like
			Array containing all the x coordinates for the centroids of every source
	ypos : array_like
			Array containing all the y coordinates for the centroids of every source
	widths : array_like
			Array containing the widths for every source PSF
	phot : array_like
			Array containing the photometry for each of the sources
	errs : 	array_like
			Array containing the raw errors for each of the sources

	Returns
	-------
	xpos : array_like
			Array containing all the x coordinates for the centroids of every source with the target as index 0
	ypos : array_like
			Array containing all the y coordinates for the centroids of every source with the target as index 0
	widths : array_like
			Array containing the widths for every source PSF with the target as index 0
	phot : array_like
			Array containing the photometry for each of the sources with the target as index 0
	errs : 	array_like
			Array containing the raw errors for each of the sources with the target as index 0
	"""
	source_xpos = xpos[source_ind]
	source_ypos = ypos[source_ind]
	source_widths = widths[source_ind]
	source_phot = phot[source_ind]
	source_errs = errs[source_ind]
	temp_xpos = np.delete(xpos, source_ind, axis = 0)
	temp_ypos = np.delete(ypos, source_ind, axis = 0)
	temp_widths = np.delete(widths, source_ind, axis = 0)
	temp_phot = np.delete(phot, source_ind, axis = 0)
	temp_errs = np.delete(errs, source_ind, axis = 0)
	xpos = np.vstack((source_xpos, temp_xpos))
	ypos = np.vstack((source_ypos, temp_ypos)) 
	widths = np.vstack((source_widths, temp_widths)) 
	phot = np.vstack((source_phot, temp_phot))
	errs = np.vstack((source_errs, temp_errs))
	return xpos, ypos, widths, phot, errs

def reject_bad_trends(xpos, ypos, widths, phot, errs, max_num_compars = 10):
	"""Using the residual sum of squares as a best-fit statistic, retains
	only a specified number of the best comparison stars. Removes bad comparison stars which do not vary in brightness in sync with the target star
	
	Parameters
	------
	xpos : array_like
			Array containing all the x coordinates for the centroids of every source
	ypos : array_like
			Array containing all the y coordinates for the centroids of every source
	widths : array_like
			Array containing the widths for every source PSF
	phot : array_like
			Array containing the photometry measurements for each of the sources
	errs : 	array_like
			Array containing the raw errors for each of the sources
	max_num_compars : int, optional
			The maximum number of comparison stars that you want to retain. Default is 10. Note that in sparse fields, there may be fewer comparison stars than this.

	Returns
	-------
	xpos : array_like
			Array containing all the x coordinates for the centroids ofevery source with only max_num_compars sources retained
	ypos : array_like
			Array containing all the y coordinates for the centroids of every source with only max_num_compars sources retained
	widths : array_like
			Array containing the widths for every source PSF with only max_num_compars sources retained
	phot : array_like
			Array containing the photometry for each of the sources with only max_num_compars sources retained
	errs : 	array_like
			Array containing the raw errors for each of the sources with only max_num_compars sources retained
	"""
	print("Rejecting bad trends...")
	total_num_compars = len(phot) - 1
	max_num_compars = min(max_num_compars, total_num_compars)
	init_size = len(phot)
	template = phot[0]
	template_subtracted = np.array([arr - template for arr in phot])
	template_sub_sums = np.sum(template_subtracted**2, axis = 1)
	sorted_template_sub_sums = np.sort(template_sub_sums)
	cutoff_template_sub_sums = sorted_template_sub_sums[max_num_compars]
	safe = np.where(template_sub_sums <= cutoff_template_sub_sums)
	safe = safe[0]
	phot = phot[safe,:]
	errs = errs[safe,:]
	xpos = xpos[safe,:]
	ypos = ypos[safe,:]
	widths = widths[safe,:]
	final_size = len(phot)
	print("Initial Number of Curves: ", init_size)
	print("Final Number of Curves: ", final_size)
	return xpos, ypos, widths, phot, errs

