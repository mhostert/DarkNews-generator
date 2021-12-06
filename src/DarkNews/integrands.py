import numpy as np
import scipy
import vegas as vg
import random 
import logging
mylogger = logging.getLogger(__name__)


#CYTHON
import pyximport
pyximport.install(
    language_level=3,
    pyimport=False,
    )
from . import Cfourvec as Cfv

from . import pdg 
from . import const
from .const import *
from . import fourvec
from . import phase_space
from . import decay_rates as dr
from . import amplitudes as amps

def Power(x,n):
	return x**n
def lam(a,b,c):
	return a**2 + b**2 + c**2 -2*a*b - 2*b*c - 2*a*c
def Sqrt(x):
	return np.sqrt(x)


class UpscatteringXsec(vg.BatchIntegrand):

	def __init__(self, dim, Enu, MC_case):
		self.dim = dim
		self.Enu = Enu
		self.MC_case = MC_case
		

	def __call__(self, x):

		ups_case = self.MC_case.ups_case

		##############################################
		# Upscattering Kinematics
		Enu = self.Enu
		M = self.MC_case.target.mass

		Q2lmin = np.log(phase_space.upscattering_Q2min(Enu, ups_case.m_ups, M))
		Q2lmax = np.log(phase_space.upscattering_Q2max(Enu, ups_case.m_ups, M))

		Q2l = (Q2lmax - Q2lmin) * x[:, 0] + Q2lmin
		Q2 = np.exp(Q2l)

		s = M**2 + 2*Enu*M # massless projectile
		t = -Q2
		u = 2*M**2 + ups_case.m_ups**2 - s - t # massless projectile


		##############################################
		# Upscattering amplitude squared (spin summed -- not averaged)
		diff_xsec = amps.upscattering_dxsec_dQ2([s,t,u], self.MC_case.ups_case)

		# Vegas jacobian -- from unit cube to physical vars
		vegas_jacobian = (Q2lmax - Q2lmin)*np.exp(Q2l)
		diff_xsec *= vegas_jacobian

		##############################################
		# return all differential quantities of interest
		self.int_dic = {}
		self.int_dic['diff_xsec'] = diff_xsec
		
		##############################################
		# storing normalization for integrands to be of O(1) numbers		
		self.norm = {}
		self.norm['diff_xsec'] = np.mean(self.int_dic['diff_xsec'])/len(x[0,:])
		
		# normalization
		self.int_dic['diff_xsec'] /= self.norm['diff_xsec']

		return self.int_dic

class UpscatteringHNLDecay(vg.BatchIntegrand):

	def __init__(self, dim, Emin, Emax, MC_case):
		self.dim = dim
		self.Emax = Emax
		self.Emin = Emin
		self.MC_case = MC_case

	def __call__(self, x):

		self.int_dic = {}
		self.norm = {}

		ups_case = self.MC_case.ups_case
		decay_case = self.MC_case.decay_case

		M = self.MC_case.target.mass
		m_parent = ups_case.m_ups
		m_daughter = decay_case.m_daughter
		mzprime = ups_case.mzprime

		i_var = 0		
		# neutrino energy
		Enu = (self.Emax - self.Emin) * x[:, i_var] + self.Emin
		i_var += 1 

		##############################################
		# Upscattering Kinematics

		Q2lmin = np.log(phase_space.upscattering_Q2min(Enu, m_parent, M))
		Q2lmax = np.log(phase_space.upscattering_Q2max(Enu, m_parent, M))

		Q2l = (Q2lmax - Q2lmin) * x[:, i_var] + Q2lmin
		i_var += 1 
		

		Q2 = np.exp(Q2l)
		s_scatt = M**2 + 2*Enu*M # massless projectile
		t_scatt = -Q2
		u_scatt = 2*M**2 + m_parent**2 - s_scatt + Q2 # massless projectile


		##############################################
		# Upscattering differential cross section (spin averaged)
		diff_xsec = amps.upscattering_dxsec_dQ2([s_scatt,t_scatt,u_scatt], self.MC_case.ups_case)

		# Vegas jacobian -- from unit cube to physical vars
		vegas_jacobian = (Q2lmax - Q2lmin)*np.exp(Q2l) * (self.Emax - self.Emin)
		diff_xsec *= vegas_jacobian

		self.int_dic['diff_flux_avg_xsec'] = diff_xsec * self.MC_case.flux(Enu)
		self.int_dic['diff_event_rate']    = diff_xsec * self.MC_case.flux(Enu)


		if decay_case.on_shell:
			##############################################
			# decay nu_parent -> nu_daughter mediator

			# angle between nu_daughter and z axis
			cost = -1.0 + (2.0) * x[:, i_var]
			i_var += 1 

			params = decay_case.TheoryModel
			vertexSQR = params.UD4**2 * (params.Ue4**2 + params.Umu4**2 + params.Utau4**2)*params.gD**2

			self.int_dic['diff_decay_rate_0'] = dr.diff_gamma_Ni_to_Nj_V(cost=cost, 
												vertex_ij = np.sqrt(vertexSQR), 
												mi=m_parent, 
												mj=m_daughter, 
												mV=mzprime, 
												HNLtype = decay_case.HNLtype, 
												h=decay_case.h_parent)
			self.int_dic['diff_decay_rate_0'] *= 2 # Vegas jacobian
			

			##############################################
			# mediator decay M --> ell+ ell-
			self.int_dic['diff_decay_rate_1'] = dr.gamma_V_to_ell_ell(vertex=const.eQED*decay_case.TheoryModel.epsilon, 
													mV=mzprime, 
													m_ell=decay_case.mm)\
													*np.full_like(self.int_dic['diff_decay_rate_0'],1.0)


		else:
			##############################################
			# decay nu_parent -> nu_daughter ell+ ell-

			m1 = decay_case.m_parent
			m2 = decay_case.mm
			m3 = decay_case.mp
			m4 = decay_case.m_daughter
			masses = np.array([m1,m2,m3,m4])

			# limits
			tmax = phase_space.three_body_tmax(*masses)
			tmin = phase_space.three_body_tmin(*masses)
			t = (tmax - tmin)*x[:,i_var] + tmin
			i_var += 1 

			umax  = phase_space.three_body_umax(*masses,t)
			umin = phase_space.three_body_umin(*masses,t)
			u = (umax - umin)*x[:,i_var] + umin
			i_var += 1 

			v = np.sum(masses**2) - u - t

			c3 = (2.0)*x[:,i_var]-1.0				
			i_var += 1 
			phi34 = (2.0*np.pi)*x[:,i_var]	
			i_var += 1 

			dgamma = dr.diff_gamma_Ni_to_Nj_ell_ell([t,u,v,c3,phi34], decay_case)

			# integrating in phi34 and ct3 explicitly
			dgamma /= 2*np.pi*2.0

			## JACOBIAN FOR DECAY 
			dgamma *= (tmax - tmin)
			dgamma *= (umax - umin)
			self.int_dic['diff_decay_rate_0'] = dgamma


		##############################################
		# storing normalization for integrands to be of O(1) numbers
		self.norm = {}
		self.norm['diff_flux_avg_xsec'] = np.mean(self.int_dic['diff_flux_avg_xsec'])/len(x[0,:])
		self.norm['diff_event_rate'] = self.norm['diff_flux_avg_xsec']


		# loop over decay processes
		for decay_step in (k for k in self.int_dic.keys() if 'decay_rate' in k):
			
			# normalization for decay process
			self.norm[decay_step]  = np.mean(self.int_dic[decay_step])/len(x[0,:])

			# multiply differential event rate by dGamma_i/dPS
			self.norm['diff_event_rate']    *= self.norm[decay_step]
			self.int_dic['diff_event_rate'] *= self.int_dic[decay_step]

		# normalize integrands to be O(1)
		for k in self.norm.keys():
			self.int_dic[k] /= self.norm[k]

		logger.debug(f"Normalization factors: {self.norm}.")
		# return all differential quantities of interest
		return self.int_dic


def get_four_momenta_from_vsamples_onshell(vsamples=None, MC_case=None, w=None, I=None):


	mh = MC_case.ups_case.m_ups
	MA = MC_case.ups_case.MA

	mf = MC_case.decay_case.m_daughter
	mm = MC_case.decay_case.mm
	mp = MC_case.decay_case.mm
	mzprime = MC_case.decay_case.mzprime

	########################	
	### scattering
	# Ni(k1) target(k2) -->  Nj(k3) target(k4)
	# energy of projectile
	Eprojectile = (MC_case.EMAX - MC_case.EMIN) * vsamples[0] + MC_case.EMIN
	scatter_samples = {  'Eprojectile': Eprojectile, 'unit_Q2': vsamples[1]} 
	masses_scatter = {	'm1': 0.0,	# nu_projectile
						'm2': MA,		# target
						'm3': mh,		# nu_upscattered
						'm4': MA		# final target
					}
	
	P1LAB, P2LAB, P3LAB, P4LAB = phase_space.two_to_two_scatter(scatter_samples, **masses_scatter)

	# N boost parameters
	boost_scattered_N = {'EP_LAB':   P3LAB.T[0],
						'costP_LAB': Cfv.get_cosTheta(P3LAB),
						'phiP_LAB':  np.arctan2(P3LAB.T[2], P3LAB.T[1])}
	

	########################
	### HNL decay		
	N_decay_samples = {'unit_cost' : np.array(vsamples[2])}
	# Ni (k1) --> Nj (k2)  Z' (k3)
	masses_decay = {'m1': mh,		# Ni
					'm2': mf,		# Nj
					'm3': mzprime,	# Z'
					}
	# Phnl, pe-, pe+, pnu
	P1LAB_decay, P2LAB_decay, P3LAB_decay = phase_space.two_body_decay(N_decay_samples, boost=boost_scattered_N, **masses_decay)

	# Z' boost parameters
		# N boost parameters
	boost_Z = {'EP_LAB':   P3LAB_decay.T[0],
				'costP_LAB': Cfv.get_cosTheta(P3LAB_decay),
				'phiP_LAB':  np.arctan2(P3LAB_decay.T[2], P3LAB_decay.T[1])}

	########################
	### Z' decay		
	Z_decay_samples = {} # all uniform
	# Z'(k1) --> ell- (k2)  ell+ (k3)
	masses_decay = {'m1': mzprime,	# Ni
					'm2': mp,	# \ell+
					'm3': mm,	# \ell-
					}
	# Phnl, pe-, pe+, pnu
	P1LAB_decayZ, P2LAB_decayZ, P3LAB_decayZ = phase_space.two_body_decay(Z_decay_samples, boost=boost_Z, **masses_decay)


	# returing dictionary
	return  {"P_projectile" :	P1LAB,
			"P_target" :		P2LAB,
			"P_scattered" :		P3LAB,
			"P_recoil" :		P4LAB,
			#
			"P_decay_N_parent" :	P1LAB_decay, 
			"P_decay_N_daughter" :	P2LAB_decay,
			"P_decay_Z_parent" :	P3LAB_decay, 
			#
			"P_decay_ell_minus" :	P2LAB_decayZ, 
			"P_decay_ell_plus" :	P3LAB_decayZ, 
			}

def get_four_momenta_from_vsamples_offshell(vsamples=None, MC_case=None):

		mh = MC_case.ups_case.m_ups
		MA = MC_case.ups_case.MA

		mf = MC_case.decay_case.m_daughter
		mm = MC_case.decay_case.mm
		mp = MC_case.decay_case.mm


		########################		
		# scattering
		# Ni(k1) target(k2) -->  Nj(k3) target(k4)
		Eprojectile = (MC_case.EMAX - MC_case.EMIN) * vsamples[0] + MC_case.EMIN
		scatter_samples = {  	
								'Eprojectile': Eprojectile,
								'unit_Q2': vsamples[1]
							} 

		masses_scatter = {	'm1': 0.0, # nu_projectile
							'm2': MA,  # target
							'm3': mh,  # nu_upscattered
							'm4': MA   # final target
						}
		
		P1LAB, P2LAB, P3LAB, P4LAB = phase_space.two_to_two_scatter(scatter_samples, **masses_scatter)

		# N boost parameters
		boost_scattered_N = {'EP_LAB':   P3LAB.T[0],
							'costP_LAB': Cfv.get_cosTheta(P3LAB),
							'phiP_LAB':  np.arctan2(P3LAB.T[2], P3LAB.T[1])}
		
		
		########################
		# HNL decay		
		N_decay_samples = {
							'unit_t' : vsamples[2],
							'unit_u' : vsamples[3],
							'unit_c3' : vsamples[4],
							'unit_phi34' : vsamples[5],
							}
		
		# Ni (k1) --> ell-(k2)  ell+(k3)  Nj(k4)
		masses_decay = {'m1': mh, # Ni
						'm2': mm, # ell-
						'm3': mp, # ell+
						'm4': mf} # Nj
		# Phnl, pe-, pe+, pnu
		P1LAB_decay, P2LAB_decay, P3LAB_decay, P4LAB_decay = phase_space.three_body_decay(N_decay_samples, boost=boost_scattered_N, **masses_decay)


		# returing dictionary
		return  {"P_projectile" :	P1LAB,
				"P_target" :		P2LAB,
				"P_scattered" :		P3LAB,
				"P_recoil" :		P4LAB,
				#
				"P_decay_N_parent" :	P1LAB_decay, 
				"P_decay_ell_minus" :	P2LAB_decay, 
				"P_decay_ell_plus" :	P3LAB_decay, 
				"P_decay_N_daughter" :	P4LAB_decay,
				}
