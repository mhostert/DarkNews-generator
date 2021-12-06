# load in main code as detector.minerva_me
# valid python syntax can be used

# configuration file for Minerva Medium Energy experiment

name = "Minerva ME"

# path to flux file
# ALL FLUXES ARE NORMALIZED SO THAT THE UNITS ARE    nus/cm^2/GeV/POT      
fluxfile = "fluxes/MINERVA_LE_numu_flux.dat"

# flux normalization factor
flux_norm = 1e-4

# neutrino energy range
erange = (0.1, 19)

# Detector materials -- homogeneous CH 
nuclear_targets = ['C12','H1'] 
fiducial_mass = 6.10   # tons
fiducial_mass_per_target = [fiducial_mass*6/7,fiducial_mass*1/7]

# total number of protons on target
POTs = 0.73 * 1.16e21

# usefule?
det_size = 2.5/2.0 # meters
