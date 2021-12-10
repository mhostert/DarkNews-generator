# load in main code as detector.uboone
# valid python syntax can be used

# configuration file for uBOONE experiment

name = "MicroBooNE"

# path to flux file
# ALL FLUXES ARE NORMALIZED SO THAT THE UNITS ARE    nus/cm^2/GeV/POT      
fluxfile = "fluxes/MiniBooNE_FHC.dat"

# flux normalization factor
flux_norm = (541.0/463.0)**2

# neutrino energy range
erange = (0.05, 9)


# Detector materials -- homogeneous Argon40
nuclear_targets = ['Ar40']
fiducial_mass = 85.0 # tons
fiducial_mass_per_target = [fiducial_mass] 

# total number of protons on target
POTs = 12.25e20