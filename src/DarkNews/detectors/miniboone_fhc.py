# load in main code as detector.miniboone_fhc
# valid python syntax can be used

name = "MiniBooNE_FHC"

# path to flux file
# ALL FLUXES ARE NORMALIZED SO THAT THE UNITS ARE    nus/cm^2/GeV/POT      
fluxfile = "fluxes/MiniBooNE_FHC.dat"

# flux normalization factor
flux_norm = 1

# neutrino energy range
erange = (0.1, 9)

# Detector materials -- homogeneous CH2
nuclear_targets = ['C12','H1']
fiducial_mass = 818.0 # tons
fiducial_mass_per_target = [fiducial_mass*12/14, fiducial_mass*2/14] # tons

# total number of protons on target
POTs = 18.75e20