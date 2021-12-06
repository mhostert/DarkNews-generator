# load in main code as detector.miniboone
# valid python syntax can be used

# configuration file for MiniBooNE experiment

name = "MiniBooNE"

# path to flux file
# ALL FLUXES ARE NORMALIZED SO THAT THE UNITS ARE    nus/cm^2/GeV/POT      
fluxfile = "fluxes/MiniBooNE_nu_mode_flux.dat"

# flux normalization factor
flux_norm = 1.0/0.05

# neutrino energy range
erange = (0.1, 9)

# Detector materials -- homogeneous CH2
nuclear_targets = ['C12','H1']
fiducial_mass = 818.0 # tons
fiducial_mass_per_target = [fiducial_mass*12/14, fiducial_mass*2/14] # tons

# total number of protons on target
POTs = 18.75e20