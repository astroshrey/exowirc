J-Band Tutorial
***************

The documentation below describes the steps to analyze images applied with a j-band filter. To analyze science images taken with the k-band filter and the helium filter, please navigate to the :doc:`kband-tutorial` and :doc:`helium-tutorial` page respectively. 

To begin utilizing all functions within the library, create five flags corresponding to the five core endpoints of the library:

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

If the data is provided with a file for correcting nonlinearity in the target's brightness, indicate the absolute path to the nonlinearity file. Otherwise, leave the variable assignment as None.

.. code-block:: Python

  nonlinearity_fname = None

Indicate the starting and ending indices of the science images and dark sequences to be analyzed:

.. code-block:: Python

  science_seqs = [(65, 458)]  
  dark_seqs = [(458, 477)] 

The range may be narrowed when running tests. If more than one range is analyzable, simply include the tuple in the array.

Similarly, include the starting and ending indices for the flat image sequences as well as for the dark for flat sequences:

.. code-block:: Python

  flat_seq = (22, 41)
  dark_for_flat_seq = (2, 21)

Depending on the analysis, indicate the background mode. The available modes include 'median', 'global', and 'helium'. J-band filter typically uses 'median' or 'global' mode:

.. code-block:: Python

  background_mode = 'median'

To see follow an example of applying the 'global' mode,please check out the :doc:`jband-tutorial` page.

Provide the estimated pixel coordinate of the target source in the science image:

.. code-block:: Python

  source_coords = [1210, 671]

A pixel (or cluster of pixels) may be identified as a star if its point spread function (PSF) has a full-width-half-max above a threshold value. Optionally set an estimate of this value in the variable finding_fwhm. If finding_fwhm is not set, the value is defaulted to 15.

.. code-block:: Python

  finding_fwhm = 20.

Optionally, provide a list of aperture radii sizes. If a list for extraction_rads is not provided, the value of the raddi list is defaulted to [20.].

.. code-block:: Python

  extraction_rads = range(5, 25)

A tuple of the inner and outer pixel radii of the annulus surrounding the target star may also optionally be specified for performing the local background subtraction. If there is no specification of ann_rads, the default radii values of the tuple is (20, 50).

.. code-block:: Python

  ann_rads = (25, 50)

A source or target star will have a much higher pixel brightness value compared to the pixel brightness values of other non-source stars. 

Optionally, estimate a sigma threshold for detecting the source stars. The default source_detection_sigma value is 50. 

.. code-block:: Python

  source_detection_sigma = 100.

The source_detection_sigma value may be readjusted after running the photometric analysis. To determine whether to lower or to raise the source_detection_sigma value, navigate to the output dump directory and search for image file source_plot.png generated from the photometry step.

If the source_detection.png circled too many source stars, then lower the sigma value, and if the image circled too little source stars, raise the sigma value. Keep the number of comparison stars circled in the image to be around 10.

Set a maximum number of comparison stars to use in the photometry process. If the max_num_compars is not specified, it is defaulted to 10. However, note that the number is often scarcer than 10 in sparse fields.

.. code-block:: Python

  max_num_compars = 8

>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Define planet params for the transit shape:

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

Define fitting params for the pymc3 library:

.. code-block:: Python

  tune = 1000            #number of burn-in steps per chain
  draws = 1500           #number of steps per chain
  target_accept = 0.99   #basically step-size tuning, closer to 1 -> small steps

>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Now begins the code segment of the sample k-band script:

.. code-block:: Python

  if __name__ == '__main__':

First, initialize the output directories for storing the output of the calibrations and analyses:

.. code-block:: Python

  calib_dir, dump_dir, img_dir = 
    iu.init_output_direcs(output_dir, test_name)

The calib_dir stores the calibrated image data that are later used for photometric analysis. The dump_dir stores the side-effect information about the images that were generated by running the functions, which may later be used in the photometric analysis or fitting later on. The img_dir stores the graph and image outputs that are useful for science.

Calibrate the science images if the calibrate_data flag is turned on by passing in the science sequence images, the dark images, the flat images, and the dark for flat images into the calibrate_all() function along with the three directories and other optional parameters:

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
				remake_darks_and_flats = remake_darks_and_flats)

After the science images are all calibrated,with the background noises removed, they are ready for photometric analysis. Perform photometry by calling the perform_photometry() function if the photometric_extraction flag is turned on, and pass in the three basic directories as well as the science sequence images and an array of the estimated coordinates of the stars in the science sequence images:

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

As in the calibration step, some parameters in the photometry steop have default values provided for them, which could be adjusted by users if better suited or more precise values are known.

Finally, fit_for_eclipse:

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