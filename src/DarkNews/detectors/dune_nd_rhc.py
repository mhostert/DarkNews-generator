
# fluxes from https://arxiv.org/pdf/2103.04797
# based on a "1.2-MW, 120-GeV primary proton beam and a 2.2m long, 16mm diameter cylindrical graphite target"
# near detector location 574 m from start of Horn 1


# load in main code as detector.dune_nd_fhc
# valid python syntax can be used

# configuration file for DUNE ND

name = "dune_nd_rhc"

# path to flux file
# ALL FLUXES ARE NORMALIZED SO THAT THE UNITS ARE    nus/cm^2/GeV/POT      
fluxfile = "fluxes/DUNE_ND_RHC.dat"

# flux normalization factor
flux_norm = 1

# neutrino energy range
erange = (0.05, 40)


# Detector materials -- homogeneous Argon40
nuclear_targets = ['Ar40']
fiducial_mass = 30.0 # tons
fiducial_mass_per_target = [fiducial_mass] 

# total number of protons on target
POTs = 1e22