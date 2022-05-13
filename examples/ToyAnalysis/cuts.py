import numpy as np
import pandas as pd

from DarkNews import *
from .fourvec import *
from ToyAnalysis import analysis as an

def efficiency(samples, weights, xmin, xmax):

	histTOT = np.histogram(samples, 
												weights=weights,  
												bins=100, 
												density=False, 
												range=(np.min(samples),np.max(samples)))

	histCUT = np.histogram(samples, 
												weights=weights, 
												bins=100, 
												density=False, 
												range=(xmin,xmax))

	return np.sum(histCUT[0])/np.sum(histTOT[0])


def MB_efficiency(samples, weights):

	costhetaZ = np.array([ fourvec.dot3(samples[i],fourvec.kz)/np.sqrt( fourvec.dot3(samples[i],samples[i]) ) for i in range(np.shape(samples)[0])])
	EZ = np.array([ fourvec.dot4(samples[i],fourvec.k0) for i in range(np.shape(samples)[0])])
	# EZ = EZ - const.Me
	PZ =  np.array([ np.sqrt( fourvec.dot3(samples[i],samples[i])) for i in range(np.shape(samples)[0])])
	
	mee =  np.array([ np.sqrt( fourvec.dot4(samples[i],samples[i])) for i in range(np.shape(samples)[0])])
	mee_cut = 0.03203 + 0.007417*EZ + 0.02738*EZ**2

	# EnuQE = (const.mproton*EZ - const.Me**2/2.0)/(const.mproton - EZ + PZ*costhetaZ)
	EnuQE = (const.mproton*EZ)/(const.mproton - EZ*(1.0 - costhetaZ) )

	mask =  (1.5 > EnuQE) &\
				 (EnuQE > 0.2) \
				# & (mee < mee_cut)
				
					 
	return np.sum((weights)*mask)/np.sum(weights)
	

#########################################
# EXPERIMENTAL PARAMETERS

############### MiniBooNE ###############
# signal def
MB_THRESHOLD = 0.03 # GeV
MB_ANGLE_MAX = 8 # degrees
# cuts
MB_ENU_MIN = 0.14 # GeV
MB_ENU_MAX = 1.5 # GeV

MB_Evis_MIN = 0.14 # GeV
MB_Evis_MAX = 3 # GeV

MB_Q2   = 1e10 # GeV^2
MB_ANALYSIS_TH = 0.2 # GeV
# resolutions
MB_STOCHASTIC = 0.12
MB_NOISE = 0.01
MB_ANGULAR = 2*np.pi/180.0


############### MicroBooNE ###############

# signal def
muB_THRESHOLD = 0.01 # GeV
muB_ANGLE_MAX = 5 # degrees
# cuts
muB_ENU_MIN = 0.14 # GeV
muB_ENU_MAX = 1.5 # GeV

muB_Evis_MIN = 0.04 # GeV
muB_Evis_MAX = 1.4 # GeV

# resolutions
muB_STOCHASTIC = 0.12
muB_NOISE = 0.01
muB_ANGULAR = 2*np.pi/180.0


############### MINERVA ###############
# signal def
MV_THRESHOLD = 0.030 # GeV
MV_ANGLE_MAX = 8 # degrees
# cuts
MV_ETHETA2     = 0.0032 # GeV
MV_Q2          = 0.02 # GeV^2
MV_ANALYSIS_TH = 0.8 # GeV
# resolutions
MV_STOCHASTIC = 0.034
MV_NOISE      = 0.059
MV_ANGULAR    = 1*np.pi/180.0





def MB_smear(samples, m):
	
	if np.size(m)!=1:
		print("ERROR! Invalid particle mass for 4 momentum smearing.")

	if an.verbose:
		print("Smearing...")
	size_samples = np.shape(samples)[0]

	E = samples['t']
	px = samples['x']
	py = samples['y']
	pz = samples['z']

	P = np.sqrt(df_dot3(samples, samples))

	# P = np.array([ np.sqrt(fourvec.dot3(samples[],samples[])) for i in range(size_samples)])

	sigma_E = MB_STOCHASTIC*np.sqrt(E) + MB_NOISE
	sigma_angle = MB_ANGULAR

	T = E - m
	theta = np.arccos(pz/P)
	phi = np.arctan2(py,px)

	T = np.array([ np.random.normal(T[i], sigma_E[i]) for i in range(size_samples)])
	# force smearing to be positive for T
	T[T < 0] = 1e-8
	
	theta = np.array([ np.random.normal(theta[i], sigma_angle) for i in range(size_samples)])
	phi = np.array([ np.random.normal(phi[i], sigma_angle) for i in range(size_samples)])
	E = T + m*np.ones((size_samples,))
	P = np.sqrt(E**2 - m**2)


	smeared = np.array([ [E[i], 
						P[i]*np.sin(theta[i])*np.cos(phi[i]),
						P[i]*np.sin(theta[i])*np.sin(phi[i]),
						P[i]*np.cos(theta[i])] for i in range(size_samples)])

	aux_df = pd.DataFrame(np.stack([smeared[:,0],smeared[:,1],smeared[:,2],smeared[:,3]], axis=-1), columns=['t','x','y','z'])

	return aux_df



def MicroBooNE_smear(samples, m):
	
	if np.size(m)!=1:
		print("ERROR! Invalid particle mass for 4 momentum smearing.")

	if an.verbose:
		print("Smearing...")
	size_samples = np.shape(samples)[0]

	E = samples['t']
	px = samples['x']
	py = samples['y']
	pz = samples['z']

	P = np.sqrt(df_dot3(samples, samples))

	# P = np.array([ np.sqrt(fourvec.dot3(samples[],samples[])) for i in range(size_samples)])

	sigma_E = muB_STOCHASTIC*np.sqrt(E) + muB_NOISE
	sigma_angle = muB_ANGULAR

	T = E - m
	theta = np.arccos(pz/P)
	phi = np.arctan2(py,px)

	#apply exponentially modified gaussian with exponential rate lambda = 1/K = 1 --> K=1
	K=1 

	#T = np.array([ np.random.normal(T[i], sigma_E[i]) for i in range(size_samples)])
	T = np.array([ exponnorm.rvs(K, loc = T[i], scale = sigma_E[i]) for i in range(size_samples)])
	# force smearing to be positive for T
	T[T < 0] = 1e-8
	
	#theta = np.array([ np.random.normal(theta[i], sigma_angle) for i in range(size_samples)])
	theta = np.array([ exponnorm.rvs(K, loc = theta[i], scale = sigma_angle) for i in range(size_samples)])

	#phi = np.array([ np.random.normal(phi[i], sigma_angle) for i in range(size_samples)])
	phi = np.array([ exponnorm.rvs(K, loc = phi[i], scale = sigma_angle) for i in range(size_samples)])

	E = T + m*np.ones((size_samples,))
	P = np.sqrt(E**2 - m**2)


	smeared = np.array([ [E[i], 
						P[i]*np.sin(theta[i])*np.cos(phi[i]),
						P[i]*np.sin(theta[i])*np.sin(phi[i]),
						P[i]*np.cos(theta[i])] for i in range(size_samples)])

	aux_df = pd.DataFrame(np.stack([smeared[:,0],smeared[:,1],smeared[:,2],smeared[:,3]], axis=-1), columns=['t','x','y','z'])

	return aux_df

