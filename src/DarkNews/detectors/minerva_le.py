# load in main code as detector.minerva_le
# valid python syntax can be used

# configuration file for Minerva Low Energy experiment

name = "Minerva LE"

# path to flux file
# ALL FLUXES ARE NORMALIZED SO THAT THE UNITS ARE    nus/cm^2/GeV/POT      
fluxfile = "fluxes/MINERVA_LE_numu_flux.dat"

# flux normalization factor
flux_norm = 2 * 1e-4 * 1e-6

# neutrino energy range
erange = (0.1, 19)

# Detector materials -- homogeneous CH 
nuclear_targets = ['C12','H1']   # mystery target
fiducial_mass = 6.10   # tons
fiducial_mass_per_target = [fiducial_mass*6/7,fiducial_mass*1/7]

# total number of protons on target
POTs = 0.73 * 3.43e20
