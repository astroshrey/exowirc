import setuptools

setuptools.setup(
	name = "exowirc",
	version = '0.0.1',
	url = "https://github.com/astroshrey/exowirc",
	description = "Image data reduction and calibration tool for the WIRC instrument",
	long_description = open("README.txt").read() + "\n\n" + open("CHANGELOG.txt").read(),
	author = "Shreyas Vissapragada",
	author_email = "svissapr@caltech.edu",
	packages = setuptools.find_packages(),
	license = "MIT",
	install_requires=["numpy>=1.20","scipy>=1.6","astropy>=4.2","matplotlib>=3.4","opencv-python>=4.5","celerite2>=0.2","exoplanet>=0.5","lightkurve>=2.0","pymc3>=3.11","arviz>=0.11","photutils>=1.1","corner>=2.2","setuptools>=46","pymc3_ext>=0.1.0"],
	)
