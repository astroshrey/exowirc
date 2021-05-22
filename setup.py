import setuptools

setuptools.setup(
	name = "exowirc",
	version = '0.0.1',
	url = "https://github.com/astroshrey/exowirc",
	author = "Shreyas Vissapragada",
	author_email = "svissapr@caltech.edu",
	packages = setuptools.find_packages(),
	license = "MIT",
	install_requires=["numpy>=1.18", "scipy>=1.6", "astropy>=4.2", 
		"matplotlib>=3.4", "opencv-python>=3.4", "celerite2>=0.2", 
		"exoplanet>=0.5", "lightkurve>=2.0", "pymc3>=3.11",
		"arviz>=0.11", "photutils>=0.7", "corner>=2.2",
		"setuptools>=46"]
	)
