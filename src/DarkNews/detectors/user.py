# load in main code as detector.user.py
# valid python syntax can be used

# define your detector here, this will be loaded by the experiment class
# in the main code, pass it with the flag
# --exp detector/user


# set the name of your experiment
name = "my experiment"

# path to flux file
# ALL FLUXES ARE NORMALIZED SO THAT THE UNITS ARE    nus/cm^2/GeV/POT      
fluxfile = "fluxes/my_experiment_fluxes.txt"

# flux normalization factor
# the flux will be scaled by this factor
flux_norm = 1.0

# neutrino energy range as a tuple
erange = (0.05, 20)

# Detector materials -- 
nuclear_targets = ['H1','He2','Li3']
fiducial_mass = 1.0 # tons
fiducial_mass_per_target = [fiducial_mass*12/14, fiducial_mass*2/14] # tons

# total number of protons on target
POTs = 1e20