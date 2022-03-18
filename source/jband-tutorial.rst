J-Band Tutorial
***************

The documentation below describes the steps to analyze images taken with a j-band filter. To analyze science images taken with the k-band filter and the helium filter, please navigate to the :doc:`kband-tutorial` and :doc:`helium-tutorial` pages respectively. 

To begin utilizing all functions within the library, create five flags corresponding to the five core functions of the library:

.. code-block:: Python

  remake_darks_and_flats = True
  remake_bkg = True
  calibrate_data = True
  photometric_extraction = True
  fit_for_eclipse = True

In order to run a step, the boolean flag in the python script must be set to True. To turn off any step that you do not wish to run, simply set the boolean flag to False.

.. code-block:: Python

  remake_bkg = False

Specify the full paths to the raw image data as well as the output directory. Also, give a prefix for the output test files and specify a naming style. For example:

.. code-block:: Python

  data_dir = "absolute path to the data directory/"
  output_dir = "absolute path to the output directory/"
  test_name = 'test1'
  naming_style = 'image'

Note that the output directory must be manually created before running the code.

If the data requires correcting nonlinearity in the target's brightness due to detector saturation (this is uncommon and typically not required), indicate the absolute path to the nonlinearity file. Otherwise, leave the variable assignment as None or remove it.

.. code-block:: Python

  nonlinearity_fname = None

Indicate the starting and ending indices of the science images and dark sequences to be analyzed:

.. code-block:: Python

  science_seqs = [(65, 458)]  
  dark_seqs = [(458, 477)] 

The science image range may be narrowed when running tests of the calibration or photometry steps. If more than one range of images is analyzable, simply include the extra tuple in the array.

Similarly, include the starting and ending indices for the flat image sequences as well as for the dark for flat sequences:

.. code-block:: Python

  flat_seq = (22, 41)
  dark_for_flat_seq = (2, 21)

Depending on the analysis, indicate the background mode. The available modes include 'median', 'global', and 'helium'. J-band filter observations typically use 'global' or 'median' mode:

.. code-block:: Python

  background_mode = 'median'

To follow an example of applying the 'global' background mode, please check out the :doc:`helium-tutorial` page.

Provide the estimated pixel coordinate of the target source in the science images:

.. code-block:: Python

  source_coords = [1210, 671]

A cluster of pixels may be identified as a star if its point spread function (PSF) has a full-width-half-max above a threshold value. Optionally set an estimate of this value in the variable finding_fwhm. If finding_fwhm is not set, the value is defaulted to 15.

.. code-block:: Python

  finding_fwhm = 20.

Optionally, provide a list of aperture radii sizes. If a list for extraction_rads is not provided, the value of the raddi list is defaulted to [20.].

.. code-block:: Python

  extraction_rads = range(5, 25)

A tuple of the inner and outer pixel radii of the annulus surrounding the target star may also optionally be specified for performing the local background subtraction. If there is no specification of ann_rads, the default radii values of the tuple is (20, 50).

.. code-block:: Python

  ann_rads = (25, 50)

Optionally, estimate a sigma threshhold for detecting the source stars (this is the sigma threshold above the background for identifying the bright pixels corresponding to stars). The default source_detection_sigma value is 50.

.. code-block:: Python

  source_detection_sigma = 100.

The source_detection_sigma value may be readjusted after running the photometric analysis. To determine whether to lower or to raise the source_detection_sigma value, navigate to the output dump directory and search for image file source_plot.png generated from the photometry step. If you find that the source star is not circled (not detected) because it is too faint, the threshold should be lowered.

Set a maximum number of comparison stars to use in the photometry process. If the max_num_compars is not specified, it is defaulted to 10. However, note that the usable number is often smaller than 10 in sparse fields.

.. code-block:: Python

  max_num_compars = 8

>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Define parameters for the fitting of the planet transit shape. Ideally, these will be informed by existing constraints from other photometric analysis, but for transits detected at high SNR the fits should be robust for wide uniform priors:

.. code-block:: Python

  phase = 'primary'
  texp = (50./60.)/1440. #days
  r_star_prior = ('normal', 1.01, 0.045) #Berger+18
  period_prior = ('normal', 125.8518, 0.0076) #Schmit +14
  t0_prior = ('uniform', 2458719.4, 2458720.)
  a_rs_prior = ('normal', 108.6, 1.1) #Schmitt+14
  b_prior = ('normal', 0.394, 0.029) #Schmitt+14
  ror_prior = ('uniform', 0., 0.15)
  jitter_prior = ('uniform', 1e-6, 1e-2)
  
Define the parameters to reject outliers from the final target photometry:

.. code-block:: Python

  sigma_cut = 5
  filter_width = 31

Define the parameters for how many steps to run the exoplanet PyMC3 posterior sampler:

.. code-block:: Python

  tune = 1500            #number of burn-in steps per chain
  draws = 2500           #number of steps per chain
  target_accept = 0.99   #basically step-size tuning, closer to 1 -> small steps

>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Congrats! You have now defined all of the necessary input parameters to reduce and analyze your WIRC data. Now begins the code segment to execute the functions you have just defined the inputs for. First, specify the following code as the main program to execute:

.. code-block:: Python

  if __name__ == '__main__':

Next, initialize the output directories for storing the output of the calibrations and analyses:

.. code-block:: Python

  calib_dir, dump_dir, img_dir = 
    iu.init_output_direcs(output_dir, test_name)

The calib_dir stores the calibrated image data that are later used for photometric analysis. The dump_dir stores the diagnostic information about the images that were generated by running the functions, which will later be used in the photometric analysis and fitting, along with the results of the fit. The img_dir stores the scientific analysis plots.

If you are using background_mode = 'global' and constructing a sky background frame to remove from the raw images, now is where you would construct that frame from the dithered sky background images. In this case, however, we are using the simple median subtraction method, so we can go right to the calibration of the image data:

.. code-block:: Python

  if calibrate_data:
    with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    cu.calibrate_all(data_dir, 
      calib_dir, 
      dump_dir,
      science_seqs, 
      dark_seqs, 
      dark_for_flat_seq,
      flat_seq, 
      style = naming_style, 
      background_mode = background_mode,
      remake_darks_and_flats = remake_darks_and_flats)

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
          max_num_compars = max_num_compars)

As in the calibration step, some parameters in the photometry step have default values provided for them, which could be adjusted by users if better suited for the specific science case. Full descriptions of the complete perform_photometry function, as well as documentation for all other ExoWIRC library functions, are available in the API section in this website.

Finally, fit the extracted photometry for the transit profile by calling the fit_for_eclipse function with all necessary parameters:

.. code-block:: Python

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

This concludes the J-band tutorial. Download the J-band sample script (and ask for some WIRC J-band data) and try it out!
