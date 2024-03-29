Helium Tutorial
***************

The documentation below describes the steps to analyze images applied with a helium filter. To analyze science images taken with the J-band filter and the K-band filter, please navigate to the :doc:`jband-tutorial` page and the :doc:`kband-tutorial` page respectively.

To begin utilizing all functions within the library, create five flags corresponding to the core functions of the library:

.. code-block:: Python

  remake_darks_and_flats = True
  remake_bkg = True
  calibrate_data = True
  photometric_extraction = True
  fit_for_eclipse = True

In order to run a step, the boolean flag in the python script must be set to True. To turn off any step that you do not wish to run, simply set the boolean flag to False.

.. code-block:: Python

  remake_darks_and_flats = False

Specify the full paths to the raw image data as well as the output directory. Also, give a prefix for the output test files and specify a naming style. For example:

.. code-block:: Python

  data_dir = "absolute path to the data directory/"
  output_dir = "absolute path to the output directory/"
  test_name = 'WASP69_Helium'
  naming_style = 'wirc'

Indicate the starting and ending indices of the science images and dark sequences to be analyzed:

.. code-block:: Python

  science_seqs = [(73, 417)]
  dark_seqs = [(438, 457)]

If multiple discontinuous series of science sequences or dark sequences are to be used, then simply include the seperate sequences as tuple pairs of the starting and ending index. For example:

.. code-block:: Python

  science_seqs = [(43, 285), (369, 572)]

Similarly, include the starting and ending indices for the flat image sequences as well as for the dark for flat sequences:

.. code-block:: Python

  flat_seq = (6, 25)
  dark_for_flat_seq = (438, 457)

Specifically for images with background mode set to global (this is the typical mode in which we use a series of dithered images to construct a sky background profile), include the background sequence:

.. code-block:: Python

  bkg_seq = (68, 71)
  
Set the lower and upper sigma thresholds for which to remove pixels from the background image. We typically use a lower bound of 5 sigma, and a large upper bound of 1000 to include all the highest outliers.

.. code-block:: Python
  bkg_sigma_lower = 5
  bkg_sigma_upper = 1000
  background_mode = 'helium'

Optionally indicate the path to the file containing the array of pixel coordinates and their corresponding nonlinearity coefficients if the image pixels have oversaturated brightness (this is typically not necessary):

.. code-block:: Python
  
  nonlinearity_fname = 'absolute path to the directory/'

Covariates are quantities to be used for systematic noise correction. Add the covariates whose data you would like to examine in the covariate_names list. For example:

.. code-block:: Python

  covariate_names = ['d_from_med', 'water_proxy', 'airmass']

A full list of covariates that may be selected include:

  |   'd_from_med',
  |   ‘airmass',
  |   'background',
  |   'psf_width',
  |   'x_cent',
  |   'y_cent',
  |   'd_from_med’,
  |   ‘water_proxy’

‘water_proxy’ is a standardly tracked covariate for images taken with the helium filter.

Provide the estimated pixel coordinate of the target source in the science image:

.. code-block:: Python

  source_coords = [265, 1836]

A cluster of pixels may be identified as a star if its point spread function (PSF) has a full-width-half-max above a threshold value. Optionally set an estiamte of this value in the variable finding_fwhm. If finding_fwhm is not set, the value is defaulted to 15.

.. code-block:: Python

  finding_fwhm = 20.

Optionally, provide a list of aperature radii sizes to test for photometric extraction. If a list for extraction_rads is not provided, the value of the raddi list is defaulted to [20.].

.. code-block:: Python

  extraction_rads = range(5, 25)

A tuple of the inner and outer pixel radii of the annulus ring that surrounds the target star may optionally be specified for performing the local background subtraction. If there is no specification of ann_rads, then the default radii values of the tuple is (20, 50).

.. code-block:: Python

  ann_rads = (25, 50)

Optionally, estimate a sigma threshhold for detecting the source stars (this is the sigma threshold above the background for identifying the bright pixels corresponding to stars). The default source_detection_sigma value is 50.

.. code-block:: Python

  source_detection_sigma = 50.

The source_detection_sigma value may be readjusted after running the photometric analysis. To determine whether to lower or to raise the source_detection_sigma value, navigate to the output dump directory and search for image file source_plot.png generated from the photometry step. If you find that the source star is not circled (not detected) because it is too faint, the threshold should be lowered.

Set a maximum number of comparison stars to use in the photometry process. If the max_num_compars is not specified, it is defaulted to 10. However, note that the usable number is often smaller than 10 in sparse fields.

.. code-block:: Python

  max_num_compars = 10

Define parameters for the fitting of the planet transit shape. Ideally, these will be informed by existing constraints from other photometric analysis, but for transits detected at high SNR the fits should be robust for wide uniform priors:

.. code-block:: Python

  phase = 'primary'
  texp = 1./1440.
  r_star_prior = ('normal', 0.813, 0.028) #Anderson+14
  period_prior = ('normal', 3.8681382, 0.0000017) #Anderson+14
  t0_prior = ('normal', 2455748.83344, 0.00018) #Anderson+14
  a_rs_prior = ('normal', 12.00, 0.46) #Anderson+14
  b_prior = ('normal', 0.686, 0.023) #Anderson+14
  ror_prior = ('uniform', 0., 0.25)
  jitter_prior = ('uniform', 1e-6, 1e-2)

Define the parameters to reject outliers from the final target photometry:

.. code-block:: Python

  sigma_cut = 4
  filter_width = 31

Define the parameters for how many steps to run the exoplanet PyMC3 posterior sampler:

.. code-block:: Python

  tune = 1000            #number of burn-in steps per chain
  draws = 1500           #number of steps per chain
  target_accept = 0.99   #basically step-size tuning, closer to 1 -> small steps


Congrats! You have now defined all of the necessary input parameters to reduce and analyze your WIRC data. Now begins the code segment to execute the functions you have just defined the inputs for. First, import all the necessary library functions:

.. code-block:: Python

  import exowirc.calib_utils as cu
  import exowirc.fit_utils as fu
  import exowirc.photo_utils as pu
  import exowirc.io_utils as iu
  import numpy as np
  import warnings

Specify __main__ as the entry point to begin reducing the dataset:

.. code-block:: Python

  if __name__ == '__main__':

Initialize the output directories for storing the output of the calibrations and analyses:

.. code-block:: Python

  calib_dir, dump_dir, img_dir = 
    iu.init_output_direcs(output_dir, test_name)

The calib_dir stores the calibrated image data that are later used for photometric analysis. The dump_dir stores the diagnostic information about the images that were generated by running the functions, which will later be used in the photometric analysis and fitting, along with the results of the fit. The img_dir stores the scientific analysis plots.

Construct a background image by using the make_calibrated_bkg_image function with all the following parameters:

.. code-block:: Python
 
  if remake_bkg:
    with warnings.catch_warnings():
      warnings.simplefilter("ignore")
      bkg = cu.make_calibrated_bkg_image(
        data_dir,
        calib_dir,
        bkg_seq,
        dark_seqs,
        dark_for_flat_seq,
        flat_seq,
        naming_style = naming_style,
        nonlinearity_fname = nonlinearity_fname,
        sigma_lower = bkg_sigma_lower, 
        sigma_upper = bkg_sigma_upper, 
        remake_darks_and_flats = remake_darks_and_flats,
        remake_bkg = remake_bkg)

After constructing the background image, calibrate the sceince images by calling the calibrate_all function with the following parameters:

.. code-block:: Python

    if calibrate_data:
      with warnings.catch_warnings():
        warnings.simplefilter("ignore")
      	cu.calibrate_all(
          data_dir,
          calib_dir,
          dump_dir,
          science_seqs,
          dark_seqs,
          dark_for_flat_seq,
          flat_seq,
          style = naming_style,
          background_mode = background_mode,
          bkg_filename = bkg)

With the science images all calibrated and the noise removed, they are now ready for photometric analysis. Perform photometry by calling the perform_photometry function if the photometric_extraction flag is turned on, and pass in all the necessary parameters:
  
.. code-block:: Python

  if photometric_extraction:
    with warnings.catch_warnings():
      warnings.simplefilter("ignore")
      pu.perform_photometry(
        calib_dir,
        dump_dir,
        img_dir,
        science_seqs,
        source_coords,
        style = naming_style,
        finding_fwhm = finding_fwhm,
        extraction_rads = extraction_rads,
        background_mode = background_mode,
        ann_rads = ann_rads,
        source_detection_sigma = source_detection_sigma,
        max_num_compars = max_num_compars,
        bkg_fname = bkg)


Finally, fit the extracted photometry for the transit profile by calling the fit_for_eclipse function with all necessary parameters:

.. code-block:: Python

  if fit_for_eclipse:
    with warnings.catch_warnings():
      warnings.simplefilter("ignore")
      best_ap = fu.quick_aperture_optimize(
        dump_dir,
        img_dir,
        extraction_rads,
        filter_width = filter_width,
        sigma_cut = sigma_cut)
      fu.fit_lightcurve(
        dump_dir,
        img_dir,
        best_ap,
	background_mode,
        covariate_names,
        texp,
        r_star_prior,
        t0_prior,
        period_prior,
        a_rs_prior,
        b_prior,
        jitter_prior,
        phase = phase,
        ror_prior = ror_prior,
        tune = tune,
        draws = draws,
        target_accept = target_accept,
        sigma_cut = sigma_cut,
        filter_width = filter_width)


This concludes the helium tutorial.
