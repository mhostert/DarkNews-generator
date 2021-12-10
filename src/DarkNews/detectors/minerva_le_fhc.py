# load in main code as detector.minerva_le_fhc
# valid python syntax can be used

name = "MINERVA_FHC_LE"

# path to flux file
# ALL FLUXES ARE NORMALIZED SO THAT THE UNITS ARE    nus/cm^2/GeV/POT      
fluxfile = "fluxes/NUMI_FHC_LE.dat"

# flux normalization factor
flux_norm = 1

# neutrino energy range
erange = (0.1, 19)

# Detector materials -- homogeneous CH 
nuclear_targets = ['C12','H1']   
fiducial_mass = 6.10   # tons
fiducial_mass_per_target = [fiducial_mass*6/7,fiducial_mass*1/7]

# total number of protons on target
POTs = 0.73 * 3.43e20
