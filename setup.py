import setuptools

setuptools.setup(
	name = "exowirc",
	url = "https://github.com/astroshrey/exowirc",
	version = '0.0.1',
	author = "Shreyas Vissapragada",
	author_email = "svissapr@caltech.edu",
	packages = ["exowirc"],
	license = "MIT",
	install_requires=["numpy", "scipy", "astropy", "matplotlib", 
		"cv2", "exoplanet", "theano", "lightkurve", "pymc3",
		"arviz", "photutils", "corner"]
	)
