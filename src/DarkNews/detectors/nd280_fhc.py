# load in main code as detector.uboone
# valid python syntax can be used

name = "ND280_FHC"

# path to flux file
# ALL FLUXES ARE NORMALIZED SO THAT THE UNITS ARE    nus/cm^2/GeV/POT      
fluxfile = "fluxes/ND280_FHC.dat"

# flux normalization factor
flux_norm = 1

# neutrino energy range
erange = (0.05, 9)

# Detector materials -- homogeneous Argon40
nuclear_targets = ['H1','C12','O16','Cu63', 'Zn64', 'Pb208']

massPb=12.0
massZn=0.8
massCu=0.4
massO=3.32
massC=5.4
massH=0.42

fiducial_mass_per_target = [massH, massC, massO, massCu, massZn, massPb] 

fiducial_mass = sum(fiducial_mass_per_target)

# total number of protons on target
POTs = 1.97e21 # in FHC and 1.63 in RHC