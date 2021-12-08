# load in main code as detector.minos_me_fhc
# valid python syntax can be used

# configuration file for MiniBooNE experiment

name = "NUMI_FHC_LE"

# path to flux file
# ALL FLUXES ARE NORMALIZED SO THAT THE UNITS ARE    nus/cm^2/GeV/POT      
fluxfile = "fluxes/NUMI_FHC_LE.dat"

# flux normalization factor
flux_norm = 1

# neutrino energy range
erange = (0.1, 9)

# Detector materials -- homogeneous CH2
nuclear_targets = ['C12','Fe56']
fiducial_mass = 28.6 # tons

massFe=	0.8 * fiducial_mass
massC=	0.2 * fiducial_mass


fiducial_mass_per_target = [massC, massFe] # tons

# total number of protons on target
POTs = 9.69e20 # in FHC