# exowirc
data reduction for exoplanet observations on Palomar/WIRC

Included also are some example scripts for how data reduction could be carried out.

Here's a useful command for installation from the static .cfg file:
``python3 -c "import configparser; c = configparser.ConfigParser(); c.read('setup.cfg'); print(c['options']['install_requires'])" | xargs pip install``

