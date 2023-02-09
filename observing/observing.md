# Guidelines for observing exoplanets with Palomar/WIRC

Here's how you observe your favorite transiting exoplanet with Palomar/WIRC.

### General Considerations

- Try to avoid hitting the 'x' button on things. Instead, if there's a button you pressed to bring the window up, click the same button to take the window away.

- Whenever you are entering text into the software, be *sure* to hit Enter before you run whatever you're going to run. Otherwise, the software does not know that you've updated a value.

- When you are switching filters, never use the joint control, always change individually. If you're switching filters in both the fore and aft wheels, do one first, wait until the status light returns to green, and then do the other.

- When in doubt, ask the telescope operator (TO) or support astronomer (SA).

- Mask should always be OUT for any exoplanet photometry work.

-Use DS9 to occasionally check that images look good and that timestamps are being written in the header.


### Starting Up

- If you're at Palomar, you don't need to do anything! The three screens should already be visible.

- If you're in the Caltech remote observing room...[fill in]

- If you're connecting remotely from your home computer, first, call in to the Palomar Zoom room. Then, connect to the VNC Viewer on portal.palomar.caltech.edu:10, 11, and 12. These will be the instrument, analysis, and telescope windows respectively. The instrument window will have interfaces for WIRC, the analysis window will have DS9 open, and the telescope window should have the FACSUM display along with a handy picture of the all-sky cam. If you don't see the right windows, ask the TO/SA! If the VNC won't connect after ~10 seconds, you're probably being blocked by the Palomar firewall. Make sure you gave them the right IP address before the night started. If all else fails, connect to the Caltech VPN and then try joining; Caltech IP addresses are allowed by default.

- If you're in the Yale remote room, the steps to connect are similar to the ones for connecting from another remote room. The IP address should already be ok for Palomar's firewall, but you can always get the updated IP address by Googling "what's my IP address."

### Calibrations

Take most of your calibrations at the start of the night, to save yourself the pain of forgetting calibrations at the end of the night.

1. Take darks for your flats first. Ensure the mirror cover is on and the dome is dark. Change the filters to BrGamma/J. Then, take a series of 2 s (for J and He band), 1 s (for H band), or 0.92 s (for Ks band) darks. We typically take 1 test dark to make sure everything looks OK, and then we take 20 more. For He observations, take 41 total darks for your dome flats. Note that the detector has more bad pixels towards the upper right-hand side (as of 09-Feb-2023).

2. Take dome flats. Change the fore filter wheel to open, and the aft wheel to whatever you need flats in. Ask the TO to take the mirror cover off and turn the "low lamp" (for J/H/Ks) or the "super high lamp" (for He band). For Ks band, you will need to cover the mirror by about 50% (maybe more) to avoid substantial nonlinearity. Once things are set up, take 1 test flat, ensure things are not nonlinear (< 30K counts), and then take 20 more. For He observations, take 41 total dome flats. For He observations, the super high lamp flats should have 8-10k counts.

3. If you know the exposure time of your target already (maybe because you've already observed it), go ahead and take darks for your target now. Same as before -- ensure the mirror cover is on and the dome is dark, take 1 test dark to make sure things look OK, and take at least 10-20 more.

4. For He observations, take a He arc image with open (AFT) and He (fore) -- you'll have to ask the TO to start this lamp. You should take a 10s exposure and look for 15-17k counts in the middle of the bright (i.e. sensitive) annulus.

### Getting on sky

Now that your calibrations are done, you're ready to get on sky!

1. Ask the TO to open the dome and go to a 11th-12th mag SAO star to check pointing and focusing. If you're worried about pointing you may want to ask for a brighter SAO star, but it's usually fine.

2. While the dome opens and the telescope slews to your target, change the filter wheel to Open/J, where it will be easiest to see the field and focus.

3. Also, while you're waiting, add targets to your list for the night. To do this, go to the telescope screen, then go to the "Astronomical Target List" window. It should be up by default, but if not, it's the third graphical button at the top of the FACSUM display, hovering over each button should give you a window description. Click "Import Astronomical Target List" on the right-hand side of the window. Go to /observer/observer/targets/, and load up the one in the directory with your name on it, or just go to 'vissapragada/WIRClist.txt'. When loading, make sure you click the right format for the file, 'vissapragada/WIRClist.txt' is 'CSV' in format, and 'space' in the delimiter. Add your target name and coordinates in the format [target name][delimiter][RA][delimiter][DEC][delimiter][epoch]. For example, the J2000 coordinates of Kepler-460 would be input as "Kepler-460 19:13:53.96 +40:39:04.82 2000". Then click "parse imported text file". This should populate the "Astronomical Targets Table" tab. When you want to deliver a target to the telescope, click it in the table, then click "Load to Telescope". In the FACSUM window, you should see "Successful Completion" in the TCS response. Then, tell the TO you've loaded the target, and they'll take you there.

4. Check pointing: is the field reasonably centered on a bright star? If not, you might need to go to a few different pointing stars to ensure the telescope is looking where you think it's looking. You should take a 1s exposure in open (AFT) and J-band (FORE).

5. Focus: First, make sure your target isn't too bright, you'll want to look at something ~11th mag. Then, ensure that the autofocus is connected to TCS on the instrument window -- otherwise, the physical position of the focus will not change during the autofocus sequence. In the instrument window, go to "Tools", then open the "Guider Controls", then click "Cross Section Graph". Right click in the image display, make sure you're on "Guide Box Selection Mode", and then click the target to update the cross section graph. Make sure the star isn't too bright (~< 25,000 peak counts). Then in the instrument window, go to "Tools", then open "Autofocus Controls". Start at something like 33, and try a coarse focus with 10 steps of 0.4 mm each. The goal is to try to bracket the correct focus on both sides. With the guide box on your source, click "Perform Autofocus". If you're relatively new to WIRC, make sure you hit enter in each of the text boxes for the focus and the increment, this is a pretty common thing to lose time on. After the autofocus is done, inspect the graph, and choose a focus as close to the bottom (i.e. the smallest FWHM) as possible. Then, in the telescope window, go to "Telescope Controls" (fourth graphical button on the top), and click "Connect to Telescope" if it's not already connected. Finally, input your focus, and click "Set Focus". Ensure that the right focus has been reached on the TCS.

6. Move to your target! On the telescope window, click on your target and find the "send to telescope" button. Ask the TO to move the telescope to your target.

7. Take a test image (1s in J-band, regardless of the observations that you'll actually do), and compare to a reference image of the field so you can be sure you've got the right target.

8. You can position the star by using the "move target to bullseye" by drawing a bullseye on the image in the instrument window. Due to a bug in the program, the target does not always move to the bullseye and sometimes requires multiple tries. The issue is exacerbated at high declination. You can take a reconaissance image each time the telescope moves with a 1s J-band exposure. As an alternative, you can use arrows on the control pad while remembering that the plate scale is 0.25"/pixel.

9. For He observations, make sure that the target star is positioned in the sensitive part of the detector. You can open the He arc image from calibrations in DS9 to ensure that your target star is on an appropriate pixel.

10. Do a test exposure with the diffuser. Do not skip this step before dithering, as the WIRC dither script will not work correctly without the correct exposure time in the preceding images.

11. Dither using the WIRC dither script (can be opened in the instrument window). Change the scale factor to 60" and begin substracting the background with the first image for which the PSFs look choppy on the cross-section graph.

12. Select the guide stars with the + button, then hit start measurement and guide. Click "continuous read" and begin taking science exposures.

### After Observing

1. Download the data with 1-2 days of observing.

2. Use exowirc!
