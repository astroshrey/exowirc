Quickstart
**********

Exowirc v.1.0 is composed of five major calibration steps: remake_darks_and_flats, remake_bkg, calibrate_data, photometric_extraction, and fit_for_eclipse. 

The documentation below desceribes the steps to analyze the light curve of a j-band transit.

In order to run a step, set the boolean flag in the python script to True.

.. code-block:: Python

  remake_darks_and_flats = True

Turn off any calibration step that you do not wish to run by setting the boolean flag to False.

.. code-block:: Python

  fit_for_eclipse = False

Specify the full paths to the raw image data as well as the output directory. Also, give a prefix for the output test files and specify a naming style. For example:

.. code-block:: Python

  data_dir = "absolute path to the data directory/"
  output_dir = "absolute path to the output directory/"
  test_name = 'test1'
  naming_style = 'image'

Note that the output directory must be manually created before running the code.

If the data is provided with a file for correcting nonlinearity in the target's brightness, indicate the absolute path to the nonliearity file. Otherwise, leave the variable assignment as None.

.. code-block:: Python

  nonlinearity_fname = None

Indicate the starting and ending indices of the science images and dark sequences to be analyzed. For example:

.. code-block:: Python

  science_seqs = [(65, 458)]  
  dark_seqs = [(458, 477)] 

The range may be narrowed when running tests. If more than one range is analyzable, simple include the tuple in the array.

Similarly, include the starting and ending indices for the flat image sequences as well as for the dark for flat sequences.

.. code-block:: Python

  flat_seq = (22, 41)
  dark_for_flat_seq = (2, 21)

Finally, depending on the desired analysis mode, indicate the background mode. The available modes include 'median', 'global', and 'helium':

.. code-block:: Python

  background_mode = 'median'







