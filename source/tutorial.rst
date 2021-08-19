Tutorial
=========

Exowirc v.1.0 is composed of five major calibration steps: remake_darks_and_flats, remake_bkg, calibrate_data, photometric_extraction, and fit_for_eclipse. 

Import all the necessary libraries before starting:

.. code-block:: Python

  import exowirc.calib_utils as cu
  import exowirc.fit_utils as fu
  import exowirc.photo_utils as pu
  import exowirc.io_utils as iu
  import numpy as np
  import warnings


The following tutorials cover the different methods of processing the images with different WIRC filters.

.. toctree::
   :maxdepth: 3

   jband-tutorial
   kband-tutorial
   helium-tutorial