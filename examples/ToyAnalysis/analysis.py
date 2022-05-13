import numpy as np
import vegas as vg
import gvar as gv
import random
import scipy 
from collections import defaultdict
from functools import partial

from . import cuts
from .fourvec import *
from DarkNews import *

from ToyAnalysis import toy_logger 
from ToyAnalysis import *

#CYTHON
import pyximport
pyximport.install(
    language_level=3,
    pyimport=False,
    )
from . import Cfourvec as Cfv

# verbose option
verbose = False


def compute_MB_spectrum(df, THRESHOLD=0.030, ANGLE_MAX=13, EVENT_TYPE='asymmetric'):

	## I'm translating into these variables that were called in the rest of the code
	## so as to avoid legacy issues...
	pN   = df['P_decay_N_parent']
	pnu   = df['P_decay_N_daughter']
	pZ   = df['P_decay_ell_minus']+df['P_decay_ell_plus']
	plm  = df['P_decay_ell_minus']
	plp  = df['P_decay_ell_plus']
	pHad = df['P_recoil']

	w = df['w_rate'].to_numpy()

	sample_size = np.shape(plp)[0]

	########################## PROCESS FOR DISTRIBUTIONS ##################################################

	### Apply gaussian detector smearing to each electron
	plp = cuts.MB_smear(plp,const.m_e)
	plm = cuts.MB_smear(plm,const.m_e)

	pZ = plp+plm

	# compute some funky true kinematical variables
	costhetaN = pN['z']/np.sqrt( df_dot3(pN,pN) )
	costhetanu = pnu['z']/np.sqrt( df_dot3(pnu,pnu) )
	costhetaHad = pHad['z']/np.sqrt( df_dot3(pHad,pHad) )
	
	EN   = pN['t']
	EZ = pZ['t']
	Elp  = plp['t']
	Elm  = plm['t']
	EHad = pHad['t']

	Mhad = np.sqrt( df_dot4(pHad, pHad) )
	Mn = np.sqrt( df_dot4(pN,pN))
	Q2 = -(2*Mhad*Mhad-2*EHad*Mhad)
	Q = np.sqrt(Q2)
	# print(plm)
	# compute some reco kinematical variables from smeared electrons
	invmass = np.sqrt( df_dot4(pZ,pZ) )
	costheta_sum = pZ['z']/np.sqrt( df_dot3(pZ,pZ) )
	costhetalp = plp['z']/np.sqrt( df_dot3(plp,plp) )
	costhetalm = plm['z']/np.sqrt( df_dot3(plm,plm) )
	Delta_costheta = df_dot3(plm,plp)/np.sqrt(df_dot3(plm,plm))/np.sqrt(df_dot3(plp,plp))

	############################################################################
	# This takes the events and asks for them to be either overlapping or asymmetric
	# THRESHOLD -- how low energy does Esubleading need to be for event to be asymmetric
	# ANGLE_MAX -- how wide opening angle needs to be in order to be overlapping
	# EVENT_TYPE -- what kind of "mis-identificatin" selection to be used:
	#	  		-- 'asymmetric' picks events where one of the letpons (independent of charge) is below a hard threshold
	#	  		-- 'overlapping' picks events where the two leptons are overlapping
	#			-- 'both' for *either* asymmetric or overlapping condition to be true
	#	  		-- 'separated' picks events where both letpons are above threshold and non-overlapping
	Evis, theta_beam, w, eff_s, ind1 = signal_events(plp, plm, w, THRESHOLD=0.03, ANGLE_MAX=13.0, EVENT_TYPE=EVENT_TYPE)
	if verbose:
		print("Signal spoofing efficiency at MB: ", eff_s)

	############################################################################
	# Applies MiniBooNE analysis cuts on the surviving LEE candidate events
	# Evis, theta_beam, w, eff_c, ind2 = MB_expcuts(Evis, theta_beam, w)
	Evis2, theta_beam2, w2, eff_c2, ind2 = MB_expcuts(Evis, theta_beam, w)
	# print("Analysis cuts efficiency at MB: ", eff_c)

	# overall efficiency of all steps of our analysis so far
	my_eff = eff_c2*eff_s
	if verbose:
		print("Pre-selection efficiency: ", my_eff)

	############################################################################
	# Compute reconsructed neutrino energy
	# this assumes quasi-elastic scattering to mimmick MiniBooNE's assumption that the underlying events are nueCC.
	df['reco_Enu'] = const.m_proton * (Evis) / ( const.m_proton - (Evis)*(1.0 - np.cos(theta_beam)))

	###########################################################################
	# Now, a trick to get the approximate PMT and particle ID efficiencies
	# I am basically subtracting the event selection efficiency from the overall MiniBooNE-provided efficiency
	eff = np.array([0.0,0.089,0.135,0.139,0.131,0.123,0.116,0.106,0.102,0.095,0.089,0.082,0.073,0.067,0.052,0.026])
	enu = np.array([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.3,1.5,1.7,1.9,2.1])
	enu_c =  enu[:-1]+(enu[1:] - enu[:-1])/2
	eff_func = scipy.interpolate.interp1d(enu_c, eff, fill_value=(eff[0],eff[-1]), bounds_error=False, kind='nearest')
	eff_miniboone = np.sum(eff_func(df['reco_Enu'])*w)/np.sum(w)

	eff_final = eff_miniboone#*my_eff
	if verbose:
		print("FINAL efficiency: ", eff_final)

	############################################################################
	# Now, apply efficiency as a function of energy to event weights
	w2 = eff_func(df['reco_Enu'])*w2


	# eff = np.array([0.0,0.089,0.135,0.139,0.131,0.123,0.116,0.106,0.102,0.095,0.089,0.082,0.073,0.067,0.052,0.026])
	# enu = np.array([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.3,1.5,1.7,1.9,2.1])
	# enu_c =  enu[:-1]+(enu[1:] - enu[:-1])/2
	# eff_func = scipy.interpolate.interp1d(enu_c,eff,fill_value=(eff[0],eff[-1]),bounds_error=False,kind='nearest')


	# # Total nueCCQE efficiency 
	# E_final_eff, nueCCQE_final_eff = np.genfromtxt('digitized/Patterson_miniboone/nueCCQE_final_eff.dat',unpack=True)
	# nueCCQE_final_eff_func = scipy.interpolate.interp1d(E_final_eff*1e-3,nueCCQE_final_eff,fill_value=(nueCCQE_final_eff[0],nueCCQE_final_eff[-1]),bounds_error=False,kind='nearest')

	# # nueCCQE efficiency of pi/e & mu/e cuts 
	# E_mupiPID_eff, nueCCQE_mupiPID_eff = np.genfromtxt('digitized/Patterson_miniboone/nueCCQE_mu_pi_eff.dat',unpack=True)
	# nueCCQE_mupiPID_eff_func = scipy.interpolate.interp1d(E_mupiPID_eff*1e-3,nueCCQE_mupiPID_eff,fill_value=(nueCCQE_mupiPID_eff[0],nueCCQE_mupiPID_eff[-1]),bounds_error=False,kind='nearest')

	# # nueCCQE efficiency of mu/e PID cut
	# E_muPID_eff, nueCCQE_muPID_eff = np.genfromtxt('digitized/Patterson_miniboone/nueCCQE_mu_eff.dat',unpack=True)
	# nueCCQE_muPID_eff_func = scipy.interpolate.interp1d(E_muPID_eff*1e-3,nueCCQE_muPID_eff,fill_value=(nueCCQE_muPID_eff[0],nueCCQE_muPID_eff[-1]),bounds_error=False,kind='nearest')

	# # Computing the efficiency of mu PID cut TIMES PMT efficiency which is inside eff_fuc
	# # Assumes eff_func = eff_PMT * nueCCQE_final_eff_func
	# eff_miniboone = np.sum(w*nueCCQE_muPID_eff_func(Enu)*eff_func(Enu)/nueCCQE_final_eff_func(Enu))/np.sum(w)
	# print("MB efficiency: ", eff_miniboone)

	# eff_final = eff_miniboone*my_eff
	# print("FINAL efficiency: ", eff_final)

	# # w = eff_func(Enu)*w
	# w = w*nueCCQE_muPID_eff_func(Enu)*eff_func(Enu)/nueCCQE_final_eff_func(Enu)

	############################################################################
	# return reco observables of LEE -- regime is still a true quantity...

	df['reco_w'] = w2
	df['reco_Evis'] = Evis2
	df['reco_theta_beam'] = theta_beam2
	df['reco_costheta_beam'] = np.cos(theta_beam2*np.pi/180)
	df['reco_eff'] = eff_final
	if verbose:
		print(df['reco_Evis'])
	# observables = {		'Enu': Enu, 
	# 					'Evis' : Evis, 
	# 					'theta_beam' : theta_beam, 
	# 					'costheta_beam' : np.cos(theta_beam*np.pi/180), 
	# 					'w' : w, 
	# 					'eff' : eff_final, 
	# 					'scattering_regime': df['scattering_regime'][ind1],#[ind2], 
	# 					'helicity': df['helicity'][ind1]}#[ind2]}
	return df



def compute_muB_spectrum(df, THRESHOLD=0.01, ANGLE_MAX=5, event_type='both', muB_eff=0.10):

	## I'm translaticng into these variables that were called in the rest of the code
	## so as to avoid legacy issues...
	plm  = df['P_decay_ell_minus']
	plp  = df['P_decay_ell_plus']

	w = df['w_rate'].to_numpy()

	sample_size = np.shape(plp)[0]

	########################## PROCESS FOR DISTRIBUTIONS ##################################################
	### Apply gaussian detector smearing to each electron
	plp = cuts.MicroBooNE_smear(plp,const.m_e)
	plm = cuts.MicroBooNE_smear(plm,const.m_e)

	############################################################################
	## pre-selection

	Evis, theta_beam, w, eff_s, ind1 = signal_events(plp,plm, w, THRESHOLD=THRESHOLD, ANGLE_MAX=ANGLE_MAX, EVENT_TYPE=event_type)
	if verbose:
		print("Signal spoofing efficiency at muB: ", eff_s)

	############################################################################
	## selection
	Evis, theta_beam, w, eff_c, ind2 = muB_expcuts(Evis, theta_beam, w, event_type=event_type)
	if verbose:
		print("Analysis cuts efficiency at muB: ", eff_c)

	# overall efficiency of all steps of our analysis so far
	my_eff = eff_c*(eff_s)
	if verbose:
		print("Pre-selection efficiency: ", my_eff)

	############################################################################
	# Compute reconsructed neutrino energy
	# this assumes quasi-elastic scattering to mimmick MiniBooNE's assumption that the underlying events are nueCC.
	Enu = const.m_proton * (Evis) / ( const.m_proton - (Evis)*(1.0 - np.cos(theta_beam)))

	############################################################################
	# Total nueCCQE efficiency -- using efficiencies provided by MicroBooNE
	E_numi_eff_edge, nueCCQE_numi_eff_edge = np.genfromtxt('digitized/2109-06832/efficiency_Ee.dat',unpack=True)
	E_numi_eff = (E_numi_eff_edge[1:] - E_numi_eff_edge[:-1])/2 + E_numi_eff_edge[:-1]
	nueCCQE_numi_eff = nueCCQE_numi_eff_edge[:-1]
	nueCCQE_numi_eff_func = scipy.interpolate.interp1d(E_numi_eff,nueCCQE_numi_eff,fill_value=(nueCCQE_numi_eff[0],nueCCQE_numi_eff[-1]),bounds_error=False,kind='nearest')

	muB_eff = np.sum(w*nueCCQE_numi_eff_func(Evis))/np.sum(w)

	eff_final = my_eff*muB_eff
	if verbose:
		print("FINAL efficiency: ", eff_final)

	############################################################################
	# Now, apply efficiency as a function of energy to event weights
	w = nueCCQE_numi_eff_func(Evis)*w

	############################################################################
	# return reco observables of LEE -- regime is still a true quantity...
	observables = {'Enu': Enu, 
					'Evis' : Evis, 
					'theta_beam' : theta_beam, 
					'costheta_beam' : np.cos(theta_beam*np.pi/180),
					'w' : w, 'eff' : eff_final, 
					'scattering_regime': df['scattering_regime'][ind1][ind2], 
					'helicity': df['helicity'][ind1][ind2],
					'event_type': event_type}
	
	return observables


def signal_events(pep, pem, w, THRESHOLD=0.03, ANGLE_MAX=13.0, EVENT_TYPE='both'):

	################### PROCESS KINEMATICS ##################
	size_samples = np.shape(pep)[0]

	# electron kinematics
	Eep = pep['t']

	Eem = pem['t']

	# angle of separation between ee
	cosdelta_ee = (df_dot3(pep,pem)/np.sqrt( df_dot3(pem,pem))/np.sqrt( df_dot3(pep,pep)))
	theta_ee = np.arccos(cosdelta_ee)*180.0/np.pi

	# two individual angles
	costheta_ep = (pep['z']/np.sqrt( df_dot3(pep,pep))).to_numpy()
	theta_ep = np.arccos(costheta_ep)*180.0/np.pi

	costheta_em = (pem['z']/np.sqrt( df_dot3(pem,pem))).to_numpy()
	theta_em = np.arccos(costheta_em)*180.0/np.pi

	# this is the angle of the combination of ee with the neutrino beam
	costheta_comb = ((pem['z']+pep['z'])/np.sqrt( df_dot3(pem+pep,pem+pep))).to_numpy()
	theta_comb = np.arccos(costheta_comb)*180.0/np.pi


	########################################
	mee = df_inv_mass(pep,pem)
	mee_cut = 0.03203 + 0.007417*(Eep + Eem) + 0.02738*(Eep + Eem)**2
	inv_mass_cut = (mee < mee_cut)

	asym_p_filter = (Eem - const.m_e < THRESHOLD) & (Eep - const.m_e > THRESHOLD) & inv_mass_cut
	asym_m_filter = (Eem - const.m_e > THRESHOLD) & (Eep - const.m_e < THRESHOLD) & inv_mass_cut
	asym_filter = (asym_p_filter | asym_m_filter) & inv_mass_cut
	ovl_filter = (Eep - const.m_e > THRESHOLD) & (Eem - const.m_e > THRESHOLD) & (theta_ee < ANGLE_MAX) & inv_mass_cut
	sep_filter = (Eep - const.m_e > THRESHOLD) & (Eem - const.m_e > THRESHOLD) & (theta_ee > ANGLE_MAX) & inv_mass_cut
	inv_filter = (Eep - const.m_e < THRESHOLD) & (Eem - const.m_e < THRESHOLD) & inv_mass_cut
	both_filter = (asym_m_filter | asym_p_filter | ovl_filter)

	w_asym_p = w[asym_p_filter]
	w_asym_m = w[asym_m_filter]
	w_asym = w[asym_m_filter | asym_p_filter]
	w_ovl = w[ovl_filter]
	w_sep = w[sep_filter]
	w_inv = w[inv_filter]
	w_tot = np.sum(w)

	eff_asym	= np.sum(w_asym)/w_tot	
	eff_ovl		= np.sum(w_ovl)/w_tot	
	eff_sep		= np.sum(w_sep)/w_tot	
	eff_inv		= np.sum(w_inv)/w_tot	

	if verbose:
		print("Efficiency for asym -> ", eff_asym*100,"%")
		print("Efficiency for ovl -> ", eff_ovl*100,"%")
		print("Efficiency for sep -> ", eff_sep*100,"%")
		print("Efficiency for inv -> ", eff_inv*100,"%")

	if EVENT_TYPE=='overlapping':
		######################### FINAL OBSERVABLES ##########################################
		Evis = np.full_like(Eep, None)
		theta_beam = np.full_like(Eep, None)

		# visible energy
		Evis[ovl_filter] = (Eep*ovl_filter + Eem*ovl_filter)[ovl_filter]

		# angle to the beam
		theta_beam[ovl_filter] = theta_comb[ovl_filter]

		w[~ovl_filter] *= 0.0

		return Evis, theta_beam, w, eff_ovl, ovl_filter

	elif EVENT_TYPE=='asymmetric':
		######################### FINAL OBSERVABLES ##########################################
		Evis = np.full_like(Eep, None)
		theta_beam = np.full_like(Eep, None)

		# visible energy
		Evis[asym_filter] = (Eep*asym_p_filter + Eem*asym_m_filter)[asym_filter]

		# angle to the beam
		theta_beam[asym_filter] = (theta_ep*asym_p_filter + theta_em*asym_m_filter)[asym_filter]

		w[~asym_filter] *= 0.0

		return Evis, theta_beam, w, eff_asym, asym_filter

	elif EVENT_TYPE=='both':
		######################### FINAL OBSERVABLES ##########################################
		Evis = np.full_like(Eep, None)
		theta_beam = np.full_like(Eep, None)
		
		# visible energy
		Evis[both_filter] = (Eep*asym_p_filter + Eem*asym_m_filter + (Eep+Eem)*ovl_filter)[both_filter]
		# angle to the beam
		theta_beam[both_filter] = (theta_ep*asym_p_filter + theta_em*asym_m_filter + theta_comb*ovl_filter)[both_filter]

		w[~both_filter] *= 0.0

		return Evis, theta_beam, w, eff_ovl+eff_asym, both_filter 

	elif EVENT_TYPE=='separated':
		######################### FINAL OBSERVABLES ##########################################
		Eplus = np.full_like(Eep, None)
		Eminus = np.full_like(Eep, None)
		theta_beam_plus = np.full_like(Eep, None)
		theta_beam_minus = np.full_like(Eep, None)

		# visible energy
		Eplus[sep_filter] = Eep[sep_filter]
		Eminus[sep_filter] = Eem[sep_filter]

		# angle to the beam
		theta_beam_plus[sep_filter] = theta_ep[sep_filter]
		theta_beam_minus[sep_filter] = theta_em[sep_filter]
		theta_sep[sep_filter] = theta_ee[sep_filter]

		return Eplus, Eminus, theta_beam_plus, theta_beam_minus, theta_sep, w_sep, eff_sep, sep_filter
		
	elif EVENT_TYPE=='invisible':
		######################### FINAL OBSERVABLES ##########################################
		# visible energy
		Eplus = np.full_like(Eep, None)
		Eminus = np.full_like(Eep, None)
		theta_beam_plus = np.full_like(Eep, None)
		theta_beam_minus = np.full_like(Eep, None)

		# visible energy
		Eplus[inv_filter] = Eep[inv_filter]
		Eminus[inv_filter] = Eem[inv_filter]

		# angle to the beam
		theta_beam_plus[inv_filter] = theta_ep[inv_filter]
		theta_beam_minus[inv_filter] = theta_em[inv_filter]
		theta_sep[inv_filter] = theta_ee[inv_filter]

		return Eplus, Eminus, theta_beam_plus, theta_beam_minus, theta_sep, w_inv, eff_inv, inv_filter
	
	else:
		print(f"Error! Could not find event type {EVENT_TYPE}.")
		return

def true_events(pep, pem, w, THRESHOLD, ANGLE_MAX, TYPE):

	################### PROCESS KINEMATICS ##################
	size_samples = np.shape(pep)[0]

	# electron kinematics
	Eep = pep[0,:]
	Eem = pem[0,:]

	# angle of separation between ee
	cosdelta_ee = Cfv.dot3(pep,pem)/np.sqrt( Cfv.dot3(pem,pem))/np.sqrt( Cfv.dot3(pep,pep) )
	theta_ee = np.arccos(cosdelta_ee)*180.0/np.pi

	# two individual angles
	costheta_ep = Cfv.get_cosTheta(pep)
	theta_ep = np.arccos(costheta_ep)*180.0/np.pi

	costheta_em = Cfv.get_cosTheta(pem)
	theta_em = np.arccos(costheta_em)*180.0/np.pi

	# this is the angle of the combination of ee with the neutrino beam
	costheta_comb = Cfv.get_cosTheta(pem+pep)
	theta_comb = np.arccos(costheta_comb)*180.0/np.pi

	mee =  np.sqrt( Cfv.dot4(samples[i],samples[i]))
	mee_cut = 0.03203 + 0.007417*(Eem+Eep) + 0.02738*(Eem+Eep)**2


	########################################
	w_asym = defaultdict(float)
	w_ovl = defaultdict(float)
	w_sep = defaultdict(float)
	w_inv = defaultdict(float)
	indices_asym_p=[]
	indices_asym_m=[]
	indices_ovl=[]
	indices_sep=[]
	indices_inv=[]

	## All obey experimental threshold
	## 
	for i in range(size_samples):
		# asymmetric positive one
		if ((Eem[i] - const.m_e < THRESHOLD) and 
				(Eep[i] - const.m_e > THRESHOLD)):
		   #w_asym += w[i]
			w_asym['pel'] +=w['pel'][i] 
			w_asym['coh'] +=w['coh'][i] 
			indices_asym_p.append(i)

		# asymmetric minus one
		if ((Eep[i] - const.m_e < THRESHOLD) and 
				(Eem[i] - const.m_e > THRESHOLD)):
			#w_asym += w[i]
			w_asym['pel'] +=w['pel'][i] 
			w_asym['coh'] +=w['coh'][i] 
			indices_asym_m.append(i)

		# overlapping
		if ((Eep[i] - const.m_e > THRESHOLD) and 
				(Eem[i] - const.m_e > THRESHOLD) and
		 		(np.arccos(cosdelta_ee[i])*180.0/np.pi < ANGLE_MAX)):
			#w_ovl += w[i]
			w_ovl['pel'] +=w['pel'][i] 
			w_ovl['coh'] +=w['coh'][i] 
			indices_ovl.append(i)

		# separated
		if ((Eep[i] - const.m_e > THRESHOLD) and \
				(Eem[i] - const.m_e > THRESHOLD) and \
		 	 (np.arccos(cosdelta_ee[i])*180.0/np.pi > ANGLE_MAX)):
			#w_sep += w[i]
			w_sep['pel'] +=w['pel'][i] 
			w_sep['coh'] +=w['coh'][i] 
			indices_sep.append(i)

		# invisible
		if ((Eep[i] - const.m_e < THRESHOLD) and \
				(Eem[i] - const.m_e < THRESHOLD)):
			#w_inv += w[i]
			w_inv['pel'] +=w['pel'][i] 
			w_inv['coh'] +=w['coh'][i] 
			indices_inv.append(i)


	indices_asym_p = np.array(indices_asym_p)
	indices_asym_m = np.array(indices_asym_m)
	indices_ovl = np.array(indices_ovl)
	indices_sep = np.array(indices_sep)
	indices_inv = np.array(indices_inv)

	#eff_asym = w_asym/w_tot	
	#eff_ovl 	= w_ovl/w_tot	
	#eff_sep 	= w_sep/w_tot	
	#eff_inv 	= w_inv/w_tot

	eff_asym = defaultdict(float)
	eff_ovl = defaultdict(float)
	eff_sep = defaultdict(float)
	eff_inv = defaultdict(float)	

	for keys in weights.keys():
		eff_asym[keys] = w_asym[keys]/np.sum(w[keys])		
		eff_ovl[keys] = w_ovl[keys]/np.sum(w[keys])
		eff_sep[keys] = w_sep[keys]/np.sum(w[keys])		
		eff_asym[keys] = w_asym[keys]/np.sum(w[keys])
	

	if TYPE=='overlapping':
		return indices_ovl, eff_ovl
	elif TYPE=='asymmetric':
		return indices_asym_p, indices_asym_m, eff_ovl
	elif TYPE=='separated':
		return indices_sep, eff_sep


def MB_expcuts(Evis, theta, weights):

	## Experimental cuts
	w_tot   = np.sum(weights)

	Pe  = np.sqrt(Evis**2 - const.m_e**2)
	Enu = (const.m_neutron*Evis - 0.5*const.m_e**2)/(const.m_neutron - Evis + Pe*np.cos(theta*np.pi/180.0))
	Q2  = 2*const.m_neutron*(Enu - Evis)

	# Cuts
	in_energy_range = (Evis > cuts.MB_Evis_MIN) & (Evis < cuts.MB_Evis_MAX)
	# there could be an angular cuts, but miniboone's acceptance is assumed to be 4*pi
	final_selected = in_energy_range

	eff = np.sum(weights[final_selected])/np.sum(weights)

	Evis[~final_selected] = None
	theta[~final_selected] = None
	weights[~final_selected] *= 0.0

	return Evis, theta, weights, eff, final_selected

def muB_expcuts(Evis, theta, weights, event_type='overlapping'):

	# Cuts
	in_energy_range = (Evis > cuts.muB_Evis_MIN) & (Evis < cuts.muB_Evis_MAX)
	
	final_selected = in_energy_range

	eff = np.sum(weights[final_selected])/np.sum(weights)

	return Evis[final_selected], theta[final_selected], weights[final_selected], eff, final_selected

def MV_expcuts(Evis, theta, weights):

	## Experimental cuts
	#w = 0.0
	indices = []
	eff = defaultdict(float)
	w = defaultdict(float)
	weights_indices = defaultdict(partial(np.ndarray,0))
	#w_tot = np.sum(weights)

	Pe = np.sqrt(Evis**2 - const.m_e**2)
	Enu = (const.m_neutron*Evis - 0.5*const.m_e**2)/ (const.m_neutron - Evis + Pe*np.cos(theta*np.pi/180.0))
	Q2 = 2*const.m_neutron*(Enu - Evis)

	for i in range(np.size(Evis)):
		# asymmetric positive one
		if ((Evis[i] > cuts.MV_ANALYSIS_TH) and \
				(Evis[i]*(theta[i]*np.pi/180.0)**2 < cuts.MV_ETHETA2) and \
				(Q2[i] < cuts.MV_Q2)):
			for keys in weights.keys():
				w[keys] += weights[keys][i]
			indices.append(i)

	for keys in weights.keys():
		eff[keys] = w[keys]/np.sum(weights[keys])
		weights_indices[keys] = np.append(weights_indices[keys], [weights[keys][i] for i in indices]) 

	return Evis[indices], theta[indices], weights_indices, eff

def CH_expcuts(Evis, theta, weights):

	## Experimental cuts
	#w = 0.0
	indices = []
	eff = defaultdict(float)
	w = defaultdict(float)
	weights_indices = defaultdict(partial(np.ndarray,0))
	#w_tot = np.sum(weights)

	Pe = np.sqrt(Evis**2 - const.m_e**2)
	Enu = (const.m_neutron*Evis - 0.5*const.m_e**2)/ (const.m_neutron - Evis + Pe*np.cos(theta*np.pi/180.0))
	Q2 = 2*const.m_neutron*(Enu - Evis)

	for i in range(np.size(Evis)):
		# asymmetric positive one
		if ((Evis[i] > cuts.CH_ANALYSIS_TH) and (Evis[i] < cuts.CH_ANALYSIS_EMAX) and \
				(Evis[i]*(theta[i]*np.pi/180.0)**2 < cuts.CH_ETHETA2)):
			for keys in weights.keys():
				w[keys] += weights[keys][i]
			indices.append(i)

	for keys in weights.keys():
		eff[keys] = w[keys]/np.sum(weights[keys])
		weights_indices[keys] = np.append(weights_indices[keys], [weights[keys][i] for i in indices]) 

	return Evis[indices], theta[indices], weights_indices, eff

