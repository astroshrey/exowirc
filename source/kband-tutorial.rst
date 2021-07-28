K-Band Tutorial
***************

The documentation below desceribes the steps to analyze images applied with a k-band filter. To analyze sceince images taken with the j-band filter and the helium filter, please navigate to the :doc:`jband-tutorial` page and the :doc:`helium-tutorial` page respectively.

To begin utilizing all functions within the library, create five flags corresponding to the five core endpoints of the library:

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
  test_name = 'test_name'
  naming_style = 'image'


Indicate the starting and ending indices of the science images and dark sequences to be analyzed:

.. code-block:: Python

  science_seqs = [(291, 529)] 
  dark_seqs = [(530, 540)]

If multiple discontinuous series of science sequences or dark sequences are to be used, then simply include the seperate sequences as tuple pairs of the starting and ending index. For example:

.. code-block:: Python

  science_seqs = [(280, 477), (510, 693)] 

Similarly, include the starting and ending indices for the flat image sequences as well as for the dark for flat sequences:

.. code-block:: Python

  flat_seq = (66, 86)
  dark_for_flat_seq = (1, 21)

Specifically for images with background mode set to global, include the background sequnece:

.. code-block:: Python

  bkg_seq = (285, 289)

Indicate the path to the file containing the array of coordinates and their corresponding nonlinearity coefficients for calibrating pixels with oversaturated brightness.

>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
nonlinearity_fname is not used anywhere in the three sample files. Can you show me a use case, where the field is not None? 

>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Covariates are quantified invariances used for noise correction. What does enumerating the covariates good for?

covariate_names = ['d_from_med', 'airmass', 'background']

>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Is the background mode 'median', 'global' dependent upon the type of sample file it is running, like only jband can have 'median' background mode, and only kband can have 'global' background mode?

>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Provide the estimated pixel coordinate of the target source in the science image:

.. code-block:: Python

  source_coords = [359, 449]

A pixel (or cluster of pixels) may be identified as a star if its point spread function (PSF) has a full-width-half-max above a threshhold value. Optionally set an estiamte of this value in the variable finding_fwhm. If finding_fwhm is not set, the value is defaulted to 15.

.. code-block:: Python

  finding_fwhm = 20.

Optionally, provide a list of aperature radii sizes. If a list for extraction_rads is not provided, the value of the raddi list is defaulted to [20.].

.. code-block:: Python

  extraction_rads = range(10, 25)

A tuple of the inner and outer pixel radii of the annulus surrounding the target star may also optionally be specified for performing the local background subtraction. If there is no specification of ann_rads, the default radii values of the tuple is (20, 50).

.. code-block:: Python

  ann_rads = (25, 50)

A target or calibrator star source will have a much higher pixel brightness value compared to the pixel brightness values of other non-source stars. Optionally set a sigma threshhold for detecting the source stars. The default source_detection_sigma value is 50.

>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

.. code-block:: Python

  source_detection_sigma = 600.

Why is the source detection sigma a lot higher? (600 vs 100) Is the global filter a lot brighter than the median filter?

>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Set a maximum number of comparison stars to use in the photometry process. If the max_num_compars is not specified, it is defaulted to 10. However, note that the number is often scarcer than 10 in sparse fields.

.. code-block:: Python

  max_num_compars = 10

>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>




