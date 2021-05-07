# Guidelines for observing exoplanets with Palomar/WIRC

Here's how you observe your favorite transiting exoplanet with Palomar/WIRC.

### General Considerations

- Try to avoid hitting the 'x' button on things. Instead, if there's a button you pressed to bring the window up, click the same button to take the window away.

- Whenever you are entering text into the software, be *sure* to hit Enter before you run whatever you're going to run. Otherwise, the software does not know that you've updated a value.

- When you are switching filters, never use the joint control, always change individually. If you're switching filters in both the fore and aft wheels, do one first, wait until the status light returns to green, and then do the other.

- When in doubt, ask the telescope operator (TO) or support astronomer (SA).

### Starting Up

- If you're at Palomar, you don't need to do anything! The three screens should already be visible.

- If you're in the Caltech remote observing room...[fill in]

- If you're connecting remotely from your home computer...[fill in]

### Calibrations

Take most of your calibrations at the start of the night, to save yourself the pain of forgetting calibrations at the end of the night.

1. Take darks for your flats first. Ensure the mirror cover is on and the dome is dark. Change the filters to BrGamma/J. Then, take a series of 2 s (for J and He band), 1 s (for H band), or 0.92 s (for Ks band) darks. We typically take 1 test dark to make sure everything looks OK, and then we take 20 more.

2. Take dome flats. Change the fore filter wheel to open, and the aft wheel to whatever you need flats in. Ask the TO to take the mirror cover off and turn the low lamp (for J/H/Ks) or the "super high lamp" (for He band). For Ks band, you will need to cover the mirror by about 50% (maybe more) to avoid substantial nonlinearity. Once things are set up, take 1 test flat, ensure things are not nonlinear (< 30K counts), and then take 20 more.

3. If you know the exposure time of your target already (maybe because you've already observed it), go ahead and take darks for your target now. Same as before -- ensure the mirror cover is on and the dome is dark, take 1 test dark to make sure things look OK, and take at least 10-20 more. 

### Getting on sky

Now that your calibrations are done, you're ready to get on sky!

1. Ask the TO to open the dome and go to a 11th-12th mag SAO star to check pointing and focusing. If you're worried about pointing you may want to ask for a brighter SAO star, but it's usually fine.

2. While the dome opens and the telescope slews to your target, change the filter wheel to Open/J, where it will be easiest to see the field and focus.

3. Also, while you're waiting, add targets to your list for the night. To do this, go to the telescope screen, then go to the window to add targets (should be at the bottom right). Search for a target file, and load up the one in the directory with your name on it, or just go to 'vissapragada/WIRCList.txt'. When loading, make sure you click 'CSV' in format, and 'space' in the delimiter.
