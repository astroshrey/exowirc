from pathlib import Path
from astropy.io import fits
import numpy as np
import pickle

#calibration io

def save_image(data, imname):
	hdu = fits.PrimaryHDU(data)
	hdu.writeto(imname, overwrite = True)
	return None

def load_calib_img(calib_dir, img_number, style = 'wirc', img_type = ''):
	fname = get_img_name(calib_dir, img_number, style = style,
		img_type = img_type)
	with fits.open(fname) as hdul:
                data = hdul[0].data
	return data

def save_multicomponent_frame(mcf, dump_dir):
	frame = pickle.dump(mcf, open(
		dump_dir + 'multicomponent_frame.p', 'wb'))
	return frame

def load_multicomponent_frame(dump_dir):
	mcf = pickle.load(open(dump_dir + 'multicomponent_frame.p', 'rb'))
	return mcf

def load_calib_files(flat, dark, bp, hp, nonlinearity_fname = None):
	with fits.open(flat) as hdul:
		flat = hdul[0].data
	with fits.open(dark) as hdul:
		dark = hdul[0].data
	with fits.open(bp) as hdul:
		bp = np.array(hdul[0].data, dtype = 'bool')
	with fits.open(hp) as hdul:
		hp = np.array(hdul[0].data, dtype = 'bool')

	if nonlinearity_fname is not None:
		with fits.open(nonlinearity_fname) as hdu:
			nonlinearity_array = hdu[1].data
		correct_nonlinearity = True
	else:
		nonlinearity_array = None
		correct_nonlinearity = False

	return flat, dark, bp, hp, nonlinearity_array, correct_nonlinearity

def get_science_img_list(science_ranges):
	to_extract = np.array([])
	for seq in science_ranges:
		range_i = np.arange(seq[0], seq[1] + 1,
			dtype = int)
		to_extract = np.append(to_extract, range_i)
	return np.array(to_extract, dtype = int)

def load_bkgs(dump_dir):
	fname = dump_dir + 'bkgs.p'
	return pickle.load(open(fname, 'rb'))

##directories
def init_phot_dirs(dump_dir, img_dir, rads):
	dump_dir_phot = dump_dir + 'phot/'
	Path(dump_dir_phot).mkdir(exist_ok = True)
	for rad in rads:
		dump_dir_temp = dump_dir + 'phot/' + str(rad) + '/'
		Path(dump_dir_temp).mkdir(exist_ok = True)
	return dump_dir_phot

def init_output_direcs(path, test_name):
	"""Initializes all output directories"""
  
	calib_dir = path + 'calibrated_' + test_name + '/'
	dump_dir = path + 'dump_files_' + test_name + '/'
	img_dir = path + 'image_files_' + test_name + '/'
	Path(calib_dir).mkdir(exist_ok = True)
	Path(dump_dir).mkdir(exist_ok = True)
	Path(img_dir).mkdir(exist_ok = True)
	print("OUTPUT DIRECTORIES INITIALIZED")
	return calib_dir, dump_dir, img_dir

##filenames
def get_img_name(direc, number, style = 'wirc', img_type = ''):
	"""Gets the image name in WIRC convention

	This function will take an image directory and number and
	gives the image name.
	Args:
		number (int): The image number.
		direc (str): The directory in which the image is stored.
		style (str): Either 'wirc' or 'image' to precede the number.
		img_type (str): For instance 'master_dark', etc.
	
	Returns:
		str: The correct path to the required file.
	"""
	num = str(number)
	num_zeros = 4 - len(num)
	if img_type != '':
		img_type = '_' + img_type
	return direc + style + '0'*num_zeros + num + img_type + '.fits'

def get_bkg_file_name(direc, bkg_num, style = 'wirc'):
	return get_img_name(direc, bkg_num, style = style,
		img_type = 'calibrated_background')

def get_calib_file_names(direc, dark_num, flat_num, style = 'wirc'):
	bp = get_img_name(direc, flat_num, style = style, img_type = 'bp_map')
	hp = get_img_name(direc, dark_num, style = style, img_type = 'hp_map')
	dark = get_img_name(direc, dark_num, style = style,
		img_type = 'combined_dark')
	flat = get_img_name(direc, flat_num, style = style,
		img_type = 'combined_flat')
	return bp, hp, dark, flat

def load_phot_data(dump_dir, aperture):
	phot_dir = f'phot/{aperture}/'
	x = pickle.load(open(dump_dir + 'bjd.p', 'rb'))
	ys = pickle.load(open(dump_dir + phot_dir + 'raw_phot.p', 'rb'))
	yerrs = pickle.load(open(dump_dir + phot_dir + 'errs.p', 'rb'))
	yerrs /= ys
	temp = ys.T/np.median(ys, axis = 1)
	ys = temp.T
	bkgs = pickle.load(open(dump_dir + 'bkgs.p', 'rb'))
	centroid_x = pickle.load(open(dump_dir + phot_dir + 'xpos.p', 'rb'))
	centroid_y = pickle.load(open(dump_dir + phot_dir + 'ypos.p', 'rb'))
	airmass = pickle.load(open(dump_dir + 'AIRMASS.p', 'rb'))
	widths = pickle.load(open(dump_dir + phot_dir + 'widths.p', 'rb'))

	return x, ys, yerrs, bkgs, centroid_x, centroid_y, \
		airmass, widths

def save_phot_data(dump_dir, xpos, ypos, widths, raw_phot, errs, tag = ''):
	if tag != '':
		tag = '_' + tag
	save_files = ['xpos', 'ypos', 'widths', 'raw_phot', 'errs']
	save_files = [dump_dir + fname + tag + '.p' for fname in
		save_files]
	pickle.dump(xpos, open(save_files[0], 'wb'))  
	pickle.dump(ypos, open(save_files[1], 'wb'))        
	pickle.dump(widths, open(save_files[2], 'wb'))        
	pickle.dump(raw_phot, open(save_files[3], 'wb'))
	pickle.dump(errs, open(save_files[4], 'wb'))
	return save_files

def save_covariates(dump_dir, covariate_dict):
	for key in covariate_dict.keys():
		fname = f'{dump_dir}{key}.p'
		pickle.dump(np.array(covariate_dict[key]), open(fname, 'wb'))
	return None


