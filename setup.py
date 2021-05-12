import setuptools

setuptools.setup(
	name = "exowirc",
	version = '0.0.1',
	url = "https://github.com/astroshrey/exowirc",
	author = "Shreyas Vissapragada",
	author_email = "svissapr@caltech.edu",
	packages = setuptools.find_packages(),
	license = "MIT",
	install_requires=["numpy", "scipy", "astropy", "matplotlib", 
		"opencv-python", "celerite2", 
		"exoplanet", "lightkurve", "pymc3",
		"arviz", "photutils", "corner", "setuptools"]
	)
