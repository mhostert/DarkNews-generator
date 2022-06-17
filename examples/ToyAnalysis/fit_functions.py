import numpy as np
import scipy
import pandas as pd
import os

from . import analysis_decay as av

#from ToyAnalysis import PATH_TO_DATA_RELEASE
PATH_TO_DATA_RELEASE = os.path.dirname(os.path.realpath(__file__)) + '/data'

##############################################################################################################################
##############################################################################################################################
##############################################################################################################################
#GENERAL CONSTANTS AND LAMBDA FUNCTIONS

# default value of couplings
ud4_def = 1.0/np.sqrt(2.)
ud5_def = 1.0/np.sqrt(2.)
gD_def = 2.
umu4_def = np.sqrt(1.0e-12)
umu5_def = np.sqrt(1.0e-12)
epsilon_def = 1e-4

vmu5_def = gD_def * ud5_def * (umu4_def*ud4_def + umu5_def*ud5_def) / np.sqrt(1 - umu4_def**2 - umu5_def**2)
v4i_def = gD_def * ud4_def * umu4_def
vmu4_def = gD_def * ud4_def * ud4_def * umu4_def / np.sqrt(1-umu4_def**2)


# Couplings: functions for simulation
vmu4_f = lambda umu4 : gD_def * ud4_def * ud4_def * umu4 / np.sqrt(1-umu4**2)
v4i_f = lambda umu4 : gD_def * ud4_def * umu4
epsilon = 1e-4

# Bins for fast histogram preparation for Reconstructed Neutrino Energy
bin_e_def = np.array([0.2, 0.3, 0.375, 0.475, 0.55, 0.675, 0.8, 0.95, 1.1, 1.3, 1.5, 3.])

# Data from MiniBooNE: Covariance Matrix
nue_data = np.genfromtxt(f'{PATH_TO_DATA_RELEASE}/MB_data_release/nue2020/numode/miniboone_nuedata_lowe.txt')
numu_data = np.genfromtxt(f'{PATH_TO_DATA_RELEASE}/MB_data_release/nue2020/numode/miniboone_numudata.txt')
nue_bkg = np.genfromtxt(f'{PATH_TO_DATA_RELEASE}/MB_data_release/nue2020/numode/miniboone_nuebgr_lowe.txt')
numu_bkg = np.genfromtxt(f'{PATH_TO_DATA_RELEASE}/MB_data_release/nue2020/numode/miniboone_numu.txt')
fract_covariance = np.genfromtxt(f'{PATH_TO_DATA_RELEASE}/MB_data_release/nue2020/numode/miniboone_full_fractcovmatrix_nu_lowe.txt')


##############################################################################################################################
##############################################################################################################################
##############################################################################################################################
#FITTING FUNCTIONS

##############################################################################################################################
# Function that rounds floats to "sig" significative decimals and returns and int if it's an integer. Works safe on arrays.
def round_sig(x, sig=1):
    if x is None:
        return x
    elif  isinstance(x,int) | isinstance(x,float):
        if int(x) == x:
            return int(x)
        else:
            z = int(np.floor(np.log10(np.abs(x))))
            return round(x, sig-z-1)
    else:
        n = len(x)
        y = np.floor(np.log10(np.abs(x)))
        z = np.array([int(i) for i in y])
        return np.array([(round(x[i], sig-z[i]-1)) if (int(x[i])!=x[i]) else (int(x[i])) for i in range(n)])


##############################################################################################################################
# Function that computes the logarithm of di/xi, returning 0 if one of them is null.
def safe_log(di,xi):
    mask = (di*xi>0)
    d = np.empty_like(di*xi)
    d[mask] = di[mask]*np.log(di[mask]/xi[mask])
    d[~mask] = di[~mask]*0.
    return d

##############################################################################################################################
# General chi2 function. Receives an array NP_MC (test histogram), NPevents (number of events to renormalize histogram), background, signal and systematics for both.
def chi2_binned_rate(NP_MC, NPevents, back_MC,D, sys=[0.1,0.1]):
    err_flux = sys[0]
    err_back = sys[1]
    
    # shape of new physics prediction normalized to NPevents
    if np.sum(NP_MC)!=0:
    	NP_MC  = (NP_MC/np.sum(NP_MC)) * NPevents

    dpoints = len(D)
    
    def chi2bin(nuis):
        alpha=nuis[:dpoints]
        beta = nuis[dpoints:]

        mu = NP_MC*(1+alpha) + back_MC*(1+beta)
        
        return 2*np.sum(mu - D + safe_log(D, mu) ) + np.sum(alpha**2/(err_flux**2)) + np.sum(beta**2 /(err_back**2))
    
    cons = ({'type': 'ineq', 'fun': lambda x: x})
    
    res = scipy.optimize.minimize(chi2bin, np.zeros(dpoints*2),constraints=cons)
    
    return chi2bin(res.x)


##############################################################################################################################
# Function like chi2_binned_rate, but filters the decay coming from 3+1 model. Useful to fit when couplings are allowed to change.
# It considers the value of |U_{mu4}| and r_eps = epsilon / epsilon_def (the ratio between the value of epsilon and the its value when the generation of events was performed)
def chi2_binned_rate_3p1(df, umu4,back_MC,D, on_shell=True,r_eps=1., sys=[0.1,0.1],type_fit='angle',decay_limit = 10000):
    err_flux = sys[0]
    err_back = sys[1]

    if on_shell:
        coupling_factor = v4i_f(umu4)/v4i_def
    else:
	    coupling_factor = (v4i_f(umu4)*r_eps)/(v4i_def)
    
    df_decay = av.select_MB_decay_expo(df,coupling_factor=coupling_factor)
    sum_w_post_smearing = np.abs(np.sum(df_decay['reco_w']))
    total_Nevent_MB = 400 * np.abs((1/df_decay['reco_eff'][0]))
    if type_fit=='angle':
    	histograms = np.histogram(np.cos(df_decay['reco_theta_beam'].values*np.pi/180.), weights=df_decay['reco_w'], bins=np.linspace(-1,1,21), density = False)
    else:
    	histograms = np.histogram(df_decay['reco_Enu'], weights=df_decay['reco_w'], bins=bin_e_def, density = False)
    NP_MC = histograms[0]
    NPevents = (vmu4_f(umu4) *r_eps / vmu4_def)**2 * sum_w_post_smearing
    
    # shape of new physics prediction normalized to NPevents
    if np.sum(NP_MC)!=0:
    	NP_MC  = (NP_MC/np.sum(NP_MC)) * NPevents

    dpoints = len(D)
    
    def chi2bin(nuis):
        alpha=nuis[:dpoints]
        beta = nuis[dpoints:]

        mu = NP_MC*(1+alpha) + back_MC*(1+beta)
        
        return 2*np.sum(mu - D + safe_log(D, mu) ) + np.sum(alpha**2/(err_flux**2)) + np.sum(beta**2 /(err_back**2))

    
    cons = ({'type': 'ineq', 'fun': lambda x: x})
    
    res = scipy.optimize.minimize(chi2bin, np.zeros(dpoints*2),constraints=cons)

    l_decay = get_decay_length(df,coupling_factor=coupling_factor)
    
    return chi2bin(res.x) if (l_decay<decay_limit) else (chi2bin(res.x) + l_decay**1.5)


##############################################################################################################################
# Function like chi2_binned_rate, but filters the decay coming from 3+2 model. Useful to fit when couplings are allowed to change.
# It considers the value of |V_{mu5}| and r_eps = epsilon / epsilon_def (the ratio between the value of epsilon and the its value when the generation of events was performed)
def chi2_binned_rate_3p2(df_decay, vmu5, back_MC,D, sys=[0.1,0.1],r_eps=1.,type_fit='angle',decay_limit = 10000):
    err_flux = sys[0]
    err_back = sys[1]
    
    sum_w_post_smearing = np.abs(np.sum(df_decay['reco_w']))
    if type_fit=='angle':
    	histograms = histograms = np.histogram(np.cos(df_decay['reco_theta_beam'].values*np.pi/180.), weights=df_decay['reco_w'], bins=np.linspace(-1,1,21), density = False)
    else:
    	histograms = np.histogram(df_decay['reco_Enu'], weights=df_decay['reco_w'], bins=bin_e_def, density = False)
    NP_MC = histograms[0]
    NPevents = (vmu5 * r_eps / vmu5_def)**2 * sum_w_post_smearing
    
    # shape of new physics prediction normalized to NPevents
    if np.sum(NP_MC)!=0:
    	NP_MC  = (NP_MC/np.sum(NP_MC)) * NPevents

    dpoints = len(D)
    
    def chi2bin(nuis):
        alpha=nuis[:dpoints]
        beta = nuis[dpoints:]

        mu = NP_MC*(1+alpha) + back_MC*(1+beta)
        
        return 2*np.sum(mu - D + safe_log(D, mu) ) + np.sum(alpha**2/(err_flux**2)) + np.sum(beta**2 /(err_back**2))

    
    cons = ({'type': 'ineq', 'fun': lambda x: x})
    
    res = scipy.optimize.minimize(chi2bin, np.zeros(dpoints*2),constraints=cons)

    l_decay = get_decay_length(df,coupling_factor=coupling_factor)
    
    return chi2bin(res.x) if (l_decay<decay_limit) else (chi2bin(res.x) + l_decay**1.5)



##############################################################################################################################
# Chi2 function that fits the reconstructed neutrino energy using MiniBooNE's covariance matrix method.
def chi2_MiniBooNE_2020(NP_MC, NPevents):

    # shape of new physics prediction normalized to NPevents
    if np.sum(NP_MC)!= 0:
    	NP_MC  = (NP_MC/np.sum(NP_MC)) * NPevents

    NP_diag_matrix  = np.diag(np.concatenate([NP_MC,nue_bkg*0.0,numu_bkg*0.0]))
    tot_diag_matrix = np.diag(np.concatenate([NP_MC,nue_bkg,numu_bkg]))

    rescaled_covariance = np.dot(tot_diag_matrix,np.dot(fract_covariance,tot_diag_matrix))
    rescaled_covariance += NP_diag_matrix # this adds the statistical error on data

    # collapse background part of the covariance
    n_signal = len(NP_MC)
    n_numu = len(numu_bkg)

    # procedure described by MiniBooNE itself
    error_matrix = np.zeros([n_signal+n_numu,n_signal+n_numu])
    error_matrix[0:n_signal,0:n_signal] = rescaled_covariance[0:n_signal,0:n_signal] + rescaled_covariance[n_signal:2*n_signal,0:n_signal] + rescaled_covariance[0:n_signal,n_signal:2*n_signal] + rescaled_covariance[n_signal:2*n_signal,n_signal:2*n_signal]
    error_matrix[n_signal:(n_signal+n_numu),0:n_signal] = rescaled_covariance[2*n_signal:(2*n_signal+n_numu),0:n_signal] + rescaled_covariance[2*n_signal:(2*n_signal+n_numu),n_signal:2*n_signal]
    error_matrix[0:n_signal,n_signal:(n_signal+n_numu)] = rescaled_covariance[0:n_signal,2*n_signal:(2*n_signal+n_numu)] + rescaled_covariance[n_signal:2*n_signal,2*n_signal:(2*n_signal+n_numu)]
    error_matrix[n_signal:(n_signal+n_numu),n_signal:(n_signal+n_numu)] = rescaled_covariance[2*n_signal:2*n_signal+n_numu,2*n_signal:(2*n_signal+n_numu)]

    if not(np.abs(np.sum(error_matrix) - np.sum(rescaled_covariance)) < 1.e-3):
    	return -1

    # compute residuals
    residuals = np.concatenate([nue_data - (NP_MC + nue_bkg), (numu_data - numu_bkg)])

    inv_cov = np.linalg.inv(error_matrix)
    
	# calculate chi^2
    chi2 = np.dot(residuals,np.dot(inv_cov,residuals)) #+ np.log(np.linalg.det(error_matrix))

    return chi2


##############################################################################################################################
# Chi2 function that fits the reconstructed neutrino energy using MiniBooNE's covariance matrix method.
# This method recomputes the number of events considering the value of |U_{mu4}| and r_eps = epsilon / epsilon_def (the ratio between the value of epsilon and the its value when the generation of events was performed)
def chi2_MiniBooNE_2020_3p1(df, umu4, on_shell=True, decay_limit = False, r_eps=1.,l_decay_proper_cm=0):

    if on_shell:
        coupling_factor = v4i_f(umu4)/v4i_def
    else:
	    coupling_factor = (v4i_f(umu4)*r_eps)/(v4i_def)
	
    df_decay = av.select_MB_decay_expo_prob(df,coupling_factor=coupling_factor,l_decay_proper_cm=l_decay_proper_cm)
    sum_w_post_smearing = np.abs(np.sum(df_decay['reco_w']))
    hist = np.histogram(df_decay['reco_Enu'], weights=df_decay['reco_w'], bins=bin_e_def, density = False)
    NP_MC = hist[0]
    NPevents = (vmu4_f(umu4) *r_eps / vmu4_def)**2 * sum_w_post_smearing
        
    # shape of new physics prediction normalized to NPevents
    if np.sum(NP_MC)!= 0:
    	NP_MC  = (NP_MC/np.sum(NP_MC)) * NPevents


    ####
    NP_diag_matrix  = np.diag(np.concatenate([NP_MC,nue_bkg*0.0,numu_bkg*0.0]))
    tot_diag_matrix = np.diag(np.concatenate([NP_MC,nue_bkg,numu_bkg]))

    rescaled_covariance = np.dot(tot_diag_matrix,np.dot(fract_covariance,tot_diag_matrix))
    rescaled_covariance += NP_diag_matrix # this adds the statistical error on data

    # collapse background part of the covariance
    n_signal = len(NP_MC)
    n_numu = len(numu_bkg)

	# procedure described by MiniBooNE itself
    error_matrix = np.zeros([n_signal+n_numu,n_signal+n_numu])
    error_matrix[0:n_signal,0:n_signal] = rescaled_covariance[0:n_signal,0:n_signal] + rescaled_covariance[n_signal:2*n_signal,0:n_signal] + rescaled_covariance[0:n_signal,n_signal:2*n_signal] + rescaled_covariance[n_signal:2*n_signal,n_signal:2*n_signal]
    error_matrix[n_signal:(n_signal+n_numu),0:n_signal] = rescaled_covariance[2*n_signal:(2*n_signal+n_numu),0:n_signal] + rescaled_covariance[2*n_signal:(2*n_signal+n_numu),n_signal:2*n_signal]
    error_matrix[0:n_signal,n_signal:(n_signal+n_numu)] = rescaled_covariance[0:n_signal,2*n_signal:(2*n_signal+n_numu)] + rescaled_covariance[n_signal:2*n_signal,2*n_signal:(2*n_signal+n_numu)]
    error_matrix[n_signal:(n_signal+n_numu),n_signal:(n_signal+n_numu)] = rescaled_covariance[2*n_signal:2*n_signal+n_numu,2*n_signal:(2*n_signal+n_numu)]

    if not(np.abs(np.sum(error_matrix) - np.sum(rescaled_covariance)) < 1.e-3):
    	return -1

    # compute residuals
    residuals = np.concatenate([nue_data - (NP_MC + nue_bkg), (numu_data - numu_bkg)])

    inv_cov = np.linalg.inv(error_matrix)
    
	# calculate chi^2
    chi2 = np.dot(residuals,np.dot(inv_cov,residuals))

    if decay_limit:
        l_decay = const.get_decay_rate_in_cm(np.sum(df_decay.w_decay_rate_0))
        return chi2 if (l_decay<decay_limit) else (chi2 + l_decay**1.5)
    else:
        return chi2

    

##############################################################################################################################
# Chi2 function that fits the reconstructed neutrino energy using MiniBooNE's covariance matrix method.
# This method recomputes the number of events considering the value of |V_{mu4}| and r_eps = epsilon / epsilon_def (the ratio between the value of epsilon and the its value when the generation of events was performed)
def chi2_MiniBooNE_2020_3p2(df_decay, vmu5,r_eps=1.):

    sum_w_post_smearing = np.abs(np.sum(df_decay['reco_w']))
    histograms = np.histogram(df_decay['reco_Enu'], weights=df_decay['reco_w'], bins=bin_e_def, density = False)
    NP_MC = histograms[0]
    NPevents = (vmu5 * r_eps / vmu5_def)**2 * sum_w_post_smearing
        
    # shape of new physics prediction normalized to NPevents
    if np.sum(NP_MC)!= 0:
    	NP_MC  = (NP_MC/np.sum(NP_MC)) * NPevents


    ####
    NP_diag_matrix  = np.diag(np.concatenate([NP_MC,nue_bkg*0.0,numu_bkg*0.0]))
    tot_diag_matrix = np.diag(np.concatenate([NP_MC,nue_bkg,numu_bkg]))

    rescaled_covariance = np.dot(tot_diag_matrix,np.dot(fract_covariance,tot_diag_matrix))
    rescaled_covariance += NP_diag_matrix # this adds the statistical error on data

    # collapse background part of the covariance
    n_signal = len(NP_MC)
    n_numu = len(numu_bkg)

    # procedure described by MiniBooNE itself
    error_matrix = np.zeros([n_signal+n_numu,n_signal+n_numu])
    error_matrix[0:n_signal,0:n_signal] = rescaled_covariance[0:n_signal,0:n_signal] + rescaled_covariance[n_signal:2*n_signal,0:n_signal] + rescaled_covariance[0:n_signal,n_signal:2*n_signal] + rescaled_covariance[n_signal:2*n_signal,n_signal:2*n_signal]
    error_matrix[n_signal:(n_signal+n_numu),0:n_signal] = rescaled_covariance[2*n_signal:(2*n_signal+n_numu),0:n_signal] + rescaled_covariance[2*n_signal:(2*n_signal+n_numu),n_signal:2*n_signal]
    error_matrix[0:n_signal,n_signal:(n_signal+n_numu)] = rescaled_covariance[0:n_signal,2*n_signal:(2*n_signal+n_numu)] + rescaled_covariance[n_signal:2*n_signal,2*n_signal:(2*n_signal+n_numu)]
    error_matrix[n_signal:(n_signal+n_numu),n_signal:(n_signal+n_numu)] = rescaled_covariance[2*n_signal:2*n_signal+n_numu,2*n_signal:(2*n_signal+n_numu)]

    if not(np.abs(np.sum(error_matrix) - np.sum(rescaled_covariance)) < 1.e-3):
    	return -1

    # compute residuals
    residuals = np.concatenate([nue_data - (NP_MC + nue_bkg), (numu_data - numu_bkg)])

    inv_cov = np.linalg.inv(error_matrix)

    # calculate chi^2
    chi2 = np.dot(residuals,np.dot(inv_cov,residuals))

    return chi2
