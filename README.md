# exowirc

Here, you'll find all things for exoplanet observations on Palomar/WIRC, including data reduction software, observing procedures, exposure time calculators, and the current data inventory.

To install the data reduction package, first download/clone the repo with:

`git clone https://github.com/astroshrey/exowirc.git`

And then run:

`python3 setup.py install`

This should take care of all package dependencies.

To run the sample scripts, first make sure the paths at the top of the script point to where the raw data are stored and where you want the data products to go. Then, run the script, for example:

`python3 helium_reduction.py`
