# load in main code as detector.nova_le_fhc
# valid python syntax can be used

name = "NOvA_FHC"

# path to flux file
# ALL FLUXES ARE NORMALIZED SO THAT THE UNITS ARE    nus/cm^2/GeV/POT      
fluxfile = "fluxes/NOvA_FHC.dat"

# flux normalization factor
flux_norm = 1

# neutrino energy range
erange = (0.05, 9)

# Detector materials -- homogeneous CH2
nuclear_targets = ['H1','C12','O16','Cl35', 'Ti48']
fiducial_mass = 193.0 # tons

massTi 	= 3.20e-2 	* fiducial_mass
massCl 	= 16.10e-2 	* fiducial_mass
massO 	= 3.00e-2 	* fiducial_mass
massC 	= 66.70e-2 	* fiducial_mass
massH 	= 10.80e-2 	* fiducial_mass

fiducial_mass_per_target = [massH, massC, massO, massCl, massTi] # tons

# total number of protons on target
POTs = 1.36e21 #in FHC + 1.25e21 in RHC

