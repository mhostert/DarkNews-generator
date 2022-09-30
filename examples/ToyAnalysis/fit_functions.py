import numpy as np
import scipy
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.patches as mpatches
from matplotlib import cm
from scipy.stats import chi2 as chi2_scipy
#from pathos.pools import ProcessPool
from pathos.multiprocessing import ProcessingPool as Pool
from matplotlib import ticker

from DarkNews import const 
from DarkNews import NuclearTarget
from . import analysis_decay as av
from . import analysis as a
from . import grid_fit
from . import plot_tools

import importlib.resources as resources

#from ToyAnalysis import PATH_TO_DATA_RELEASE
# PATH_TO_DATA_RELEASE = os.path.dirname(os.path.realpath(__file__)) + '/data'

# targets
proton = NuclearTarget("H1")
C12 = NuclearTarget("C12")
Ar40 = NuclearTarget("Ar40")

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


# couplings functions for simulation
vmu4_f = lambda umu4 : gD_def * ud4_def * ud4_def * umu4 / np.sqrt(1-umu4**2)
v4i_f = lambda umu4 : gD_def * ud4_def * umu4
epsilon = 1e-4
r_eps = epsilon / epsilon_def 

# bins for fast histogram preparation
bin_e_def = np.array([0.2, 0.3, 0.375, 0.475, 0.55, 0.675, 0.8, 0.95, 1.1, 1.3, 1.5, 3.])

# data for plots
plotvars = {'3+1' : ['mzprime', 'm4'], '3+2' : ['m5', 'delta']}
plotaxes = {'3+1' : [r'$m_{Z\prime} [\mathrm{GeV}]$', r'$m_{4} [\mathrm{GeV}]$'], '3+2' : [r'$m_{5} [\mathrm{GeV}]$', r'$\Delta$']}

# Location
#loc = 'ToyAnalysis/data'
loc = 'data'
# obtain data from MB for the fitting
data_MB_source = {'Enu' : grid_fit.get_data_MB(varplot='reco_Enu'), 'angle' : grid_fit.get_data_MB(varplot='reco_angle')}

# Normalization (temporal variable)
NORMALIZATION = 1

def round_sig(x, sig=1):
    if isinstance(x,float) | isinstance(x,int):
        z = int(np.floor(np.log10(np.abs(x))))
        return round(x, sig-z-1)
    else:
        n = len(x)
        y = np.floor(np.log10(np.abs(x)))
        z = np.array([int(i) for i in y])
        return np.array([round(x[i], sig-z[i]-1) for i in range(n)])


def get_decay_length(df,coupling_factor=1.):
    # get momenta and decay length for decay_N
    pN = df.P_decay_N_parent.values
    l_decay_proper_cm = const.get_decay_rate_in_cm(np.sum(df.w_decay_rate_0)) / coupling_factor**2

    # compute the position of decay
    x,y,z = av.decay_position(pN, l_decay_proper_cm)[1:]

    return np.sqrt(x*x+y*y+z*z).mean()


def chi2_test(NP_MC,NPevents,D,sys=[0.1,0.1]):
    NP_MC  = (NP_MC/np.sum(NP_MC)) * NPevents
    sys = sys[0]
    chi2 = ((NP_MC - D)**2) / (D + sys**2)

    return chi2.sum()


def safe_log(di,xi):
    mask = (di*xi>0)
    d = np.empty_like(di*xi)
    d[mask] = di[mask]*np.log(di[mask]/xi[mask])
    #d[~mask] = di[~mask]*1e100
    d[~mask] = di[~mask]*0.
    return d

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


def chi2_MiniBooNE_2020(NP_MC, NPevents):

    # shape of new physics prediction normalized to NPevents
    if np.sum(NP_MC)!= 0:
        NP_MC  = (NP_MC/np.sum(NP_MC)) * NPevents


    ####
    # using __init__ path definition
    # bin_e = np.genfromtxt(f'{PATH_TO_DATA_RELEASE}//miniboone_binboundaries_nue_lowe.txt')
    bin_e = np.genfromtxt(resources.open_text("ToyAnalysis.include.MB_data_release.numode", 'miniboone_binboundaries_nue_lowe.txt'))

    bin_w = -bin_e[:-1]  + bin_e[1:]
    bin_c = bin_e[:-1] + bin_w/2

    nue_data = np.genfromtxt(resources.open_text('ToyAnalysis.include.MB_data_release.numode', 'miniboone_nuedata_lowe.txt'))
    numu_data = np.genfromtxt(resources.open_text('ToyAnalysis.include.MB_data_release.numode', 'miniboone_numudata.txt'))

    nue_bkg = np.genfromtxt(resources.open_text('ToyAnalysis.include.MB_data_release.numode', 'miniboone_nuebgr_lowe.txt'))
    numu_bkg = np.genfromtxt(resources.open_text('ToyAnalysis.include.MB_data_release.numode', 'miniboone_numu.txt'))

    fract_covariance = np.genfromtxt(resources.open_text('ToyAnalysis.include.MB_data_release.numode', 'miniboone_full_fractcovmatrix_nu_lowe.txt'))

    MB_LEE = nue_data - nue_bkg

    NP_diag_matrix  = np.diag(np.concatenate([NP_MC,nue_bkg*0.0,numu_bkg*0.0]))
    tot_diag_matrix = np.diag(np.concatenate([NP_MC,nue_bkg,numu_bkg]))


    rescaled_covariance = np.dot(tot_diag_matrix,np.dot(fract_covariance,tot_diag_matrix))
    rescaled_covariance += NP_diag_matrix # this adds the statistical error on data

    # collapse background part of the covariance
    n_signal = len(NP_MC)
    n_background = len(nue_bkg)
    n_numu = len(numu_bkg)


    # procedure described by MiniBooNE itself
    error_matrix = np.zeros([n_signal+n_numu,n_signal+n_numu])
    error_matrix[0:n_signal,0:n_signal] = rescaled_covariance[0:n_signal,0:n_signal] + rescaled_covariance[n_signal:2*n_signal,0:n_signal] + rescaled_covariance[0:n_signal,n_signal:2*n_signal] + rescaled_covariance[n_signal:2*n_signal,n_signal:2*n_signal]
    error_matrix[n_signal:(n_signal+n_numu),0:n_signal] = rescaled_covariance[2*n_signal:(2*n_signal+n_numu),0:n_signal] + rescaled_covariance[2*n_signal:(2*n_signal+n_numu),n_signal:2*n_signal]
    error_matrix[0:n_signal,n_signal:(n_signal+n_numu)] = rescaled_covariance[0:n_signal,2*n_signal:(2*n_signal+n_numu)] + rescaled_covariance[n_signal:2*n_signal,2*n_signal:(2*n_signal+n_numu)]
    error_matrix[n_signal:(n_signal+n_numu),n_signal:(n_signal+n_numu)] = rescaled_covariance[2*n_signal:2*n_signal+n_numu,2*n_signal:(2*n_signal+n_numu)]

    #assert(np.abs(np.sum(error_matrix) - np.sum(rescaled_covariance)) < 1.e-3)
    
    if not(np.abs(np.sum(error_matrix) - np.sum(rescaled_covariance)) < 1.e-3):
        return -1

    # compute residuals
    residuals = np.concatenate([nue_data - (NP_MC + nue_bkg), (numu_data - numu_bkg)])

    inv_cov = np.linalg.inv(error_matrix)

    # calculate chi^2
    chi2 = np.dot(residuals,np.dot(inv_cov,residuals)) #+ np.log(np.linalg.det(error_matrix))

    return chi2


def cov_matrix_MB():

    # shape of new physics prediction normalized to NPevents
    # using __init__ path definition
    bin_e = np.genfromtxt(resources.open_text('ToyAnalysis.include.MB_data_release.numode', 'miniboone_binboundaries_nue_lowe.txt'))
    bin_w = -bin_e[:-1]  + bin_e[1:]
    
    nue_data = np.genfromtxt(resources.open_text('ToyAnalysis.include.MB_data_release.numode', 'miniboone_nuedata_lowe.txt'))
    numu_data = np.genfromtxt(resources.open_text('ToyAnalysis.include.MB_data_release.numode', 'miniboone_numudata.txt'))
    
    nue_bkg = np.genfromtxt(resources.open_text('ToyAnalysis.include.MB_data_release.numode', 'miniboone_nuebgr_lowe.txt'))
    numu_bkg = np.genfromtxt(resources.open_text('ToyAnalysis.include.MB_data_release.numode', 'miniboone_numu.txt'))
    
    fract_covariance = np.genfromtxt(resources.open_text('ToyAnalysis.include.MB_data_release.numode', 'miniboone_full_fractcovmatrix_nu_lowe.txt'))

    
    NP_diag_matrix  = np.diag(np.concatenate([nue_data-nue_bkg,nue_bkg*0.0,numu_bkg*0.0]))
    tot_diag_matrix = np.diag(np.concatenate([nue_data-nue_bkg,nue_bkg,numu_bkg]))


    rescaled_covariance = np.dot(tot_diag_matrix,np.dot(fract_covariance,tot_diag_matrix))
    rescaled_covariance += NP_diag_matrix # this adds the statistical error on data

    # collapse background part of the covariance
    n_signal = 11
    n_background = 11
    n_numu = 8


    # procedure described by MiniBooNE itself
    error_matrix = np.zeros([n_signal+n_numu,n_signal+n_numu])
    error_matrix[0:n_signal,0:n_signal] = rescaled_covariance[0:n_signal,0:n_signal] + rescaled_covariance[n_signal:2*n_signal,0:n_signal] + rescaled_covariance[0:n_signal,n_signal:2*n_signal] + rescaled_covariance[n_signal:2*n_signal,n_signal:2*n_signal]
    error_matrix[n_signal:(n_signal+n_numu),0:n_signal] = rescaled_covariance[2*n_signal:(2*n_signal+n_numu),0:n_signal] + rescaled_covariance[2*n_signal:(2*n_signal+n_numu),n_signal:2*n_signal]
    error_matrix[0:n_signal,n_signal:(n_signal+n_numu)] = rescaled_covariance[0:n_signal,2*n_signal:(2*n_signal+n_numu)] + rescaled_covariance[n_signal:2*n_signal,2*n_signal:(2*n_signal+n_numu)]
    error_matrix[n_signal:(n_signal+n_numu),n_signal:(n_signal+n_numu)] = rescaled_covariance[2*n_signal:2*n_signal+n_numu,2*n_signal:(2*n_signal+n_numu)]

    #assert(np.abs(np.sum(error_matrix) - np.sum(rescaled_covariance)) < 1.e-3)
    
    if not(np.abs(np.sum(error_matrix) - np.sum(rescaled_covariance)) < 1.e-3):
        return -1

    #inv_cov = np.linalg.inv(error_matrix)

    return error_matrix


def chi2_binned_rate_3p1(df, couplings, coupling_factor, back_MC,D, sys=[0.1,0.1],type_fit='angle',decay_limit = 10000):
    err_flux = sys[0]
    err_back = sys[1]
    
    df_decay = av.select_MB_decay_expo(df,coupling_factor=coupling_factor)
    sum_w_post_smearing = np.abs(np.sum(df_decay['reco_w']))
    total_Nevent_MB = 400 * np.abs((1/df_decay['reco_eff'][0]))
    if type_fit=='angle':
        histograms = plot_tools.get_histogram1D(df_decay, NEVENTS=total_Nevent_MB, varplot='reco_angle',loc='../')
    else:
        histograms = plot_tools.get_histogram1D(df_decay, NEVENTS=total_Nevent_MB, varplot='reco_Enu',loc='../')
    NP_MC = histograms[0]
    NPevents = ((couplings*epsilon*couplings/ud4_def) / couplings_default_3p1)**2 * sum_w_post_smearing
    
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


def chi2_binned_rate_3p2(df, couplings, coupling_factor, back_MC,D, sys=[0.1,0.1],type_fit='angle',decay_limit = 10000):
    err_flux = sys[0]
    err_back = sys[1]
    
    df_decay = av.select_MB_decay(df,coupling_factor=coupling_factor)
    sum_w_post_smearing = np.abs(np.sum(df_decay['reco_w']))
    total_Nevent_MB = 400 * np.abs((1/df_decay['reco_eff'][0]))
    if type_fit=='angle':
        histograms = plot_tools.get_histogram1D(df_decay, NEVENTS=total_Nevent_MB, varplot='reco_angle',loc='../')
    else:
        histograms = plot_tools.get_histogram1D(df_decay, NEVENTS=total_Nevent_MB, varplot='reco_Enu',loc='../')
    NP_MC = histograms[0]
    NPevents = ((couplings*couplings_heavy) / couplings_default)**2 * sum_w_post_smearing
    
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


def chi2_MiniBooNE_2020_3p1(df, umu4, on_shell=True, v4i_f=v4i_f, v4i_def=v4i_def, vmu4_f=vmu4_f, vmu4_def=vmu4_def, r_eps = r_eps, l_decay_proper_cm=1):

    df = df.copy(deep=True)
    
    if on_shell:
        factor = (v4i_f(umu4)/v4i_def)**2
    else:
        factor = (r_eps*v4i_f(umu4)/v4i_def)**2
    
    decay_l = l_decay_proper_cm / factor
    df_decay = av.decay_selection(df, decay_l, 'miniboone', weights='w_event_rate')
    df_decay = a.compute_spectrum(df_decay, EVENT_TYPE='both')
    
    df_decay = df_decay[df_decay.reco_w>0]
    sum_w_post_smearing = np.abs(np.sum(df_decay['reco_w']))
    hist = np.histogram(df_decay['reco_Enu'], weights=df_decay['reco_w'], bins=bin_e_def, density = False)
    NP_MC = hist[0]
    NPevents = (vmu4_f(umu4)/vmu4_def)**2 * sum_w_post_smearing * r_eps**2
        
    return chi2_MiniBooNE_2020(NP_MC,NPevents)

    


def chi2_MiniBooNE_2020_3p2(df, vmu5,vmu5_def=vmu5_def,r_eps=1.):

    df_decay = df.copy(deep=True)
    
    sum_w_post_smearing = np.abs(np.sum(df_decay['reco_w'])) * r_eps**2
    histograms = np.histogram(df_decay['reco_Enu'], weights=df_decay['reco_w'], bins=bin_e_def, density = False)
    NP_MC = histograms[0]
    NPevents = (vmu5 / vmu5_def)**2 * sum_w_post_smearing
        
    return chi2_MiniBooNE_2020(NP_MC,NPevents)


def chi2_MiniBooNE_2020_3p2_nodecay(NP_MC, NPevents):

    NP_MC = histograms[0]
        
    # shape of new physics prediction normalized to NPevents
    if np.sum(NP_MC)!= 0:
        NP_MC  = (NP_MC/np.sum(NP_MC)) * NPevents

    ####
    # using __init__ path definition
    bin_e = np.genfromtxt(resources.open_text('ToyAnalysis.include.MB_data_release.numode', 'miniboone_binboundaries_nue_lowe.txt'))
    bin_w = -bin_e[:-1]  + bin_e[1:]
    bin_c = bin_e[:-1] + bin_w/2
    
    nue_data = np.genfromtxt(resources.open_text('ToyAnalysis.include.MB_data_release.numode', 'miniboone_nuedata_lowe.txt'))
    numu_data = np.genfromtxt(resources.open_text('ToyAnalysis.include.MB_data_release.numode', 'miniboone_numudata.txt'))
    
    nue_bkg = np.genfromtxt(resources.open_text('ToyAnalysis.include.MB_data_release.numode', 'miniboone_nuebgr_lowe.txt'))
    numu_bkg = np.genfromtxt(resources.open_text('ToyAnalysis.include.MB_data_release.numode', 'miniboone_numu.txt'))
    
    fract_covariance = np.genfromtxt(resources.open_text('ToyAnalysis.include.MB_data_release.numode', 'miniboone_full_fractcovmatrix_nu_lowe.txt'))

    MB_LEE = nue_data - nue_bkg

    NP_diag_matrix  = np.diag(np.concatenate([NP_MC,nue_bkg*0.0,numu_bkg*0.0]))
    tot_diag_matrix = np.diag(np.concatenate([NP_MC,nue_bkg,numu_bkg]))


    rescaled_covariance = np.dot(tot_diag_matrix,np.dot(fract_covariance,tot_diag_matrix))
    rescaled_covariance += NP_diag_matrix # this adds the statistical error on data

    # collapse background part of the covariance
    n_signal = len(NP_MC)
    n_background = len(nue_bkg)
    n_numu = len(numu_bkg)


    # procedure described by MiniBooNE itself
    error_matrix = np.zeros([n_signal+n_numu,n_signal+n_numu])
    error_matrix[0:n_signal,0:n_signal] = rescaled_covariance[0:n_signal,0:n_signal] + rescaled_covariance[n_signal:2*n_signal,0:n_signal] + rescaled_covariance[0:n_signal,n_signal:2*n_signal] + rescaled_covariance[n_signal:2*n_signal,n_signal:2*n_signal]
    error_matrix[n_signal:(n_signal+n_numu),0:n_signal] = rescaled_covariance[2*n_signal:(2*n_signal+n_numu),0:n_signal] + rescaled_covariance[2*n_signal:(2*n_signal+n_numu),n_signal:2*n_signal]
    error_matrix[0:n_signal,n_signal:(n_signal+n_numu)] = rescaled_covariance[0:n_signal,2*n_signal:(2*n_signal+n_numu)] + rescaled_covariance[n_signal:2*n_signal,2*n_signal:(2*n_signal+n_numu)]
    error_matrix[n_signal:(n_signal+n_numu),n_signal:(n_signal+n_numu)] = rescaled_covariance[2*n_signal:2*n_signal+n_numu,2*n_signal:(2*n_signal+n_numu)]

    #assert(np.abs(np.sum(error_matrix) - np.sum(rescaled_covariance)) < 1.e-3)
    
    if not(np.abs(np.sum(error_matrix) - np.sum(rescaled_covariance)) < 1.e-3):
        return -1

    # compute residuals
    residuals = np.concatenate([nue_data - (NP_MC + nue_bkg), (numu_data - numu_bkg)])

    inv_cov = np.linalg.inv(error_matrix)
    # print(error_matrix)
    # calculate chi^2
    chi2 = np.dot(residuals,np.dot(inv_cov,residuals)) #+ np.log(np.linalg.det(error_matrix))


    return chi2





class exp_plotter:
    
    def __init__(self, path='test',path_gen='test', points=20, model='3+2', delta=np.geomspace(0.05, 79, 20), m4=np.geomspace(0.01, 0.75, 20), mzprime=1.25, m5_max=10.0,neutrino_type="majorana", exp='miniboone',nb_cores=8, mode_chi2=True,sys_angle=0,method='method1',minimize=True):
        
        self.path = path
        self.path_gen = path_gen
        self.path_gen_couplings = self.path_gen + '_couplings'
        self.points = points
        self.model = model
        self.datasets_list = {}
        self.datasets_list_couplings = {}
        self.same_dataset = True
        self.Npoints = points**2
        self.dof_enu = 6.7 #to initialize it
        self.dof_angle = 18 #to initialize it
        self.neutrino_type = neutrino_type
        self.exp = exp
        self.nb_cores = nb_cores
        self.mode_chi2 = mode_chi2
        self.sys_angle = sys_angle
        self.coupling_mode = 'm5'
        self.method = method
        
        if self.model=='3+2':
            self.delta = delta
            self.m4 = m4
            if len(m4)==len(delta):
                self.m5 = self.m4 * (self.delta + 1)
            self.mzprime = mzprime
            self.m5_max = m5_max
            self.columns = ['delta','m4','m5','N_events','chi2', 'sum_w_rate_df', 'sum_w_flux_avg_sigma_df','sum_w_post_smearing']
            self.columns_couplings = ['m4', 'couplings','chi2','m5','delta']
        elif self.model=='3+1':
            self.m4 = m4
            self.mzprime = mzprime
            self.columns = ['mzprime','m4','N_events','chi2', 'sum_w_rate_df', 'sum_w_flux_avg_sigma_df','sum_w_post_smearing']
            self.columns_couplings = ['m4', 'couplings','chi2','mzprime']
        
        self.path_enu = self.path+'__grid_run__/chi2_enu_fit_'+self.method+'.dat'
        self.path_angle = self.path+'__grid_run__/chi2_angle_fit_'+self.method+'.dat'
        self.path_enu_original = self.path+'__grid_run__/chi2_enu_fit_'+self.method+'_original.dat'
        self.path_angle_original = self.path+'__grid_run__/chi2_angle_fit_'+self.method+'_original.dat'
        self.path_enu_coup = self.path+'__grid_run__/chi2_enu_couplings_'+self.method+'.dat'
        self.path_angle_coup = self.path+'__grid_run__/chi2_angle_couplings_'+self.method+'.dat'
        self.path_enu_coup_original = self.path+'__grid_run__/chi2_enu_couplings_'+self.method+'_original.dat'
        self.path_angle_coup_original = self.path+'__grid_run__/chi2_angle_couplings_'+self.method+'_original.dat'

        self.paths_normal = {'Enu' : self.path_enu, 'angle' : self.path_angle}
        self.paths_coup = {'Enu' : self.path_enu_coup, 'angle' : self.path_angle_coup}
        self.paths_coup_original = {'Enu' : self.path_enu_coup_original, 'angle' : self.path_angle_coup_original}
        self.paths_original = {'Enu' : self.path_enu_original, 'angle' : self.path_angle_original}

        self.paths = {'normal' : self.paths_normal, 'couplings' : self.paths_coup, 'original' : self.paths_original, 'original_couplings' : self.paths_coup_original}

        self.plots_paths_fitting_enu = {'chi2': self.path+'__grid_run__/plots/chi2_enu_'+self.neutrino_type+'_'+self.method+'.pdf', 'chi2_sigmas': self.path+'__grid_run__/plots/chi2_enu_'+self.neutrino_type+'_'+self.method+'_sigmas.pdf', 'nevents': self.path+'__grid_run__/plots/nevents_enu_'+self.neutrino_type+'_'+self.method+'.pdf'}
        self.plots_paths_fitting_angle = {'chi2': self.path+'__grid_run__/plots/chi2_angle_'+self.neutrino_type+'_'+self.method+'.pdf', 'chi2_sigmas': self.path+'__grid_run__/plots/chi2_angle_'+self.neutrino_type+'_'+self.method+'_sigmas.pdf', 'nevents': self.path+'__grid_run__/plots/nevents_angle_'+self.neutrino_type+'_'+self.method+'.pdf'}
        self.plots_paths_fitting = {'Enu' : self.plots_paths_fitting_enu, 'angle' : self.plots_paths_fitting_angle}

        self.hnl_type = r'Dirac' if self.neutrino_type=='dirac' else r'Majorana'
        self.plots_titles_fitting_enu = {'chi2': r'$\chi^2/dof$ for $E_\nu$', 'chi2_sigmas':r'$\Delta \chi^2$ for $E_\nu$, MiniBooNE, ' + self.hnl_type if self.model=='3+1' else r'$\Delta \chi^2$ for $E_\nu$, $m_{Z\prime}=' + str(self.mzprime) + r'\ \mathrm{GeV}$, MiniBooNE, ' + self.hnl_type , 'nevents': r'$N_{events}$ for $E_\nu$' }
        self.plots_titles_fitting_angle = {'chi2': r'$\chi^2/dof$ for $\theta_{ee}^{\mathrm{beam}}$', 'chi2_sigmas': r'$\Delta \chi^2$ for $\theta_{ee}^{\mathrm{beam}}$, MiniBooNE, ' + self.hnl_type  if self.model=='3+1' else r'$\Delta \chi^2$ for $\theta_{ee}^{\mathrm{beam}}$, $m_{Z\prime}=' + str(self.mzprime) + r'\ \mathrm{GeV}$, MiniBooNE, ' + self.hnl_type, 'nevents': r'$N_{events}$ for $\theta_{ee}^{\mathrm{beam}}$' }
        self.plots_titles_fitting = {'Enu' : self.plots_titles_fitting_enu, 'angle' : self.plots_titles_fitting_angle}

        self.plots_paths_fitting_enu = {'chi2': self.path+'__grid_run__/plots/chi2_enu_'+self.neutrino_type+'_'+self.method+'.pdf', 'chi2_sigmas': self.path+'__grid_run__/plots/chi2_enu_'+self.neutrino_type+'_'+self.method+'_sigmas.pdf', 'nevents': self.path+'__grid_run__/plots/nevents_enu_'+self.neutrino_type+'_'+self.method+'.pdf'}
        self.plots_paths_fitting_angle = {'chi2': self.path+'__grid_run__/plots/chi2_angle_'+self.neutrino_type+'_'+self.method+'.pdf', 'chi2_sigmas': self.path+'__grid_run__/plots/chi2_angle_'+self.neutrino_type+'_'+self.method+'_sigmas.pdf', 'nevents': self.path+'__grid_run__/plots/nevents_angle_'+self.neutrino_type+'_'+self.method+'.pdf'}
        self.plots_paths_fitting = {'Enu' : self.plots_paths_fitting_enu, 'angle' : self.plots_paths_fitting_angle}

        self.hnl_type = r'Dirac' if self.neutrino_type=='dirac' else r'Majorana'
        self.plots_titles_fitting_enu = {'chi2': r'$\chi^2/dof$ for $E_\nu$', 'chi2_sigmas':r'$\Delta \chi^2$ for $E_\nu$, MiniBooNE, ' + self.hnl_type if self.model=='3+1' else r'$\Delta \chi^2$ for $E_\nu$, $m_{Z\prime}=' + str(self.mzprime) + r'\ \mathrm{GeV}$, MiniBooNE, ' + self.hnl_type , 'nevents': r'$N_{events}$ for $E_\nu$' , 'chi2_title' : r'3+2, MiniBooNE, ' + self.hnl_type if self.model=='3+1' else r'$m_{Z\prime}=' +str(self.mzprime) + r' \ \mathrm{GeV}$, 3+2, MiniBooNE, ' + self.hnl_type}
        self.plots_titles_fitting_angle = {'chi2': r'$\chi^2/dof$ for $\theta_{ee}^{\mathrm{beam}}$', 'chi2_sigmas': r'$\Delta \chi^2$ for $\theta_{ee}^{\mathrm{beam}}$, MiniBooNE, ' + self.hnl_type  if self.model=='3+1' else r'$\Delta \chi^2$ for $\theta_{ee}^{\mathrm{beam}}$, $m_{Z\prime}=' + str(self.mzprime) + r'\ \mathrm{GeV}$, MiniBooNE, ' + self.hnl_type, 'nevents': r'$N_{events}$ for $\theta_{ee}^{\mathrm{beam}}$', 'chi2_title' : r'3+2, MiniBooNE, ' + self.hnl_type if self.model=='3+1' else r'$m_{Z\prime}=' +str(self.mzprime) + r' \ \mathrm{GeV}$, 3+2, MiniBooNE, ' + self.hnl_type}
        self.plots_titles_fitting = {'Enu' : self.plots_titles_fitting_enu, 'angle' : self.plots_titles_fitting_angle}

        self.minimize = minimize
        
    
    
    def set_couplings_case(name='light'):
        self.path_enu_coup = self.path+'__grid_run__/chi2_enu_couplings_'+self.method+'_'+name+'.dat'
        self.path_angle_coup = self.path+'__grid_run__/chi2_angle_couplings_'+self.method+'_'+name+'.dat'


    def run_grid(self, really_run=True):
        
        if self.model=='3+2':
            grid_run_object = ThreePlusTwoPipeline(
                m4=self.m4,
                delta=self.delta,
                m5_max=self.m5_max,
                mzprime=self.mzprime, 
                sort_fields=["m5", "Delta"],
                grid_run_path=self.path_gen, 
                D_or_M=self.neutrino_type, 
                exp=self.exp,
                plot_path="my_plots"
            )
        elif self.model=='3+1':
            grid_run_object = ThreePlusOnePipeline(
                mzprime=self.mzprime,
                m4=self.m4,
                sort_fields=["mzprime", "m4"],
                grid_run_path=self.path_gen, 
                D_or_M=self.neutrino_type, 
                exp=self.exp,
                plot_path="my_plots"
            )
        
        if really_run:
            grid_run_object.run(nb_cores=self.nb_cores)
        
        grid_run_object.load_datasets_list()
        #grid_run_object.fill_observables_df(nb_cores=self.nb_cores)
        self.datasets_list = grid_run_object.get_datasets_list(extension='pckl')
        self.Npoints = len(self.datasets_list)
    
    
    def run_grid_couplings(self, really_run=True, delta=1,mzprime=1.25,m4=np.geomspace(0.01, 0.75, 100)):
        
        if self.model=='3+2':
            delta_ar = np.array([delta])
            grid_run_object = ThreePlusTwoPipeline(
                m4=m4,
                delta=delta_ar,
                m5_max=self.m5_max,
                mzprime=self.mzprime, 
                sort_fields=["m5", "Delta"],
                grid_run_path=self.path_gen_couplings, 
                D_or_M=self.neutrino_type, 
                exp=self.exp,
                plot_path="my_plots"
            )
        elif self.model=='3+1':
            mzprime_ar = np.array([mzprime])
            grid_run_object = ThreePlusOnePipeline(
                mzprime=mzprime_ar,
                m4=m4,
                sort_fields=["mzprime", "m4"],
                grid_run_path=self.path_gen_couplings, 
                D_or_M=self.neutrino_type, 
                exp=self.exp,
                plot_path="my_plots"
            )
        
        if really_run:
            grid_run_object.run(nb_cores=self.nb_cores)
        
        grid_run_object.load_datasets_list()
        self.datasets_list_couplings = grid_run_object.get_datasets_list(extension='pckl')
        self.same_dataset = False
    
    
    #def compute_chi2_grid(self, dataset, type_fit='Enu', num=500, sys=0.1, back_MC=0.,D=1.):
    def compute_chi2_grid(self, dataset, type_fit, sys, back_MC,D,num=500):
        
        desired_MB_events = 400
        
        # read data
        df = pd.read_pickle(dataset['dataset'])
        sum_w_rate_df = np.sum(df['w_rate'])
        sum_w_flux_avg_sigma_df = np.sum(df['w_flux_avg_sigma'])
        m4 = dataset['m4']
        
        if self.model=='3+2':
            delta = (dataset['m5']-dataset['m4'])/dataset['m4']
            m5 = dataset['m5']
        elif self.model=='3+1':
            mzprime = dataset['mzprime']

        # compute spectrum
        bag_reco_MB = av.compute_MB_spectrum(df, EVENT_TYPE='both')
        bag_reco_MB = av.select_MB_decay(bag_reco_MB)
        sum_w_post_smearing = np.sum(bag_reco_MB['reco_w'])

        # compute histograms
        total_Nevent_MB = desired_MB_events * (1/bag_reco_MB['reco_eff'][0])
                    
        if type_fit=='Enu':
            histograms = plot_tools.get_histogram1D(bag_reco_MB, NEVENTS=total_Nevent_MB, varplot='reco_Enu',loc=loc)
            if self.mode_chi2:
                self.dof_enu = 6.7
            dof = self.dof_enu
        elif type_fit=='angle':
            histograms = plot_tools.get_histogram1D(bag_reco_MB, NEVENTS=total_Nevent_MB, varplot='reco_angle',loc=loc)
            dof = self.dof_angle

        NP_MC = histograms[0]
                    
        if self.minimize:
            init_guess = np.array([540])
            if not(self.mode_chi2) or (type_fit=='angle'):
                chi2f = lambda nevents : chi2_binned_rate(NP_MC, nevents, back_MC, D,sys=sys)
                res = scipy.optimize.minimize(chi2f, init_guess)
                chi2min = res.fun
                N_events = res.x[0]
                
            if (self.mode_chi2) & (type_fit=='Enu'):
                chi2f = lambda nevents : chi2_MiniBooNE_2020(NP_MC, nevents)
                res = scipy.optimize.minimize(chi2f, init_guess)
                chi2min = res.fun
                N_events = res.x[0]
                
        else:
            # fitting MB
            NPevents_trials = np.linspace(100, 700, num)
            chi2_arr = 1e8*np.ones(num)

            if not(self.mode_chi2) or (type_fit=='angle'):
                for j in range(num):
                    chi2_arr[j] = chi2_binned_rate(NP_MC, NPevents_trials[j], back_MC, D,sys=sys)
                
            if (self.mode_chi2) & (type_fit=='Enu'):
                for j in range(num):
                    chi2_arr[j] = chi2_MiniBooNE_2020(NP_MC, NPevents_trials[j])   

            chi2min = chi2_arr.min()		
            NPevents_trials[chi2_arr==chi2min]		
            N_events = NPevents_trials[chi2_arr==chi2min][0]

        chi2min /= dof

        
        if self.model=='3+2':
            return [m4, m5, delta, chi2min, N_events,sum_w_rate_df,sum_w_flux_avg_sigma_df,sum_w_post_smearing]
        elif self.model=='3+1':
            return [mzprime, m4, chi2min, N_events,sum_w_rate_df,sum_w_flux_avg_sigma_df,sum_w_post_smearing]
    
    
    def fit_grid(self, definition_number=500, type_fit='Enu'):
            
        num = definition_number
        desired_MB_events = 400
        
        # obtain data from MB for the fitting
        data_MB = data_MB_source[type_fit]
        
        self.dof_enu = 6.7
        if type_fit=='angle':
            self.dof_angle = len(data_MB[0]) - 2
        
        back_MC = data_MB[1]
        D = data_MB[0] + data_MB[1]

        if self.sys_angle!=0 & type_fit=='angle':
            sys = [self.sys_angle, self.sys_angle]
        else:
            sys = [data_MB[2], data_MB[3]]
    
        
        chi2 = lambda dataset: self.compute_chi2_grid(dataset, type_fit=type_fit, sys=sys, back_MC=back_MC,D=D, num=num)
        
        pool = Pool(self.nb_cores)
                                
        chi2_lists = pool.map(chi2,self.datasets_list)
        
        chi2_df = pd.DataFrame(data=chi2_lists,columns=self.columns)

        chi2_df.to_csv(self.paths['normal'][type_fit],sep='\t',float_format='%.5e',index=False)
        
                

    def purge_grid(self, type_fit='Enu'):
        
        try:
            path_data_source = self.paths['normal'][type_fit]
            path_data_source_ur = self.paths['original'][type_fit]
            data = pd.read_csv(path_data_source,sep='\t')
            
            
        except:
            print('You have to do the fitting first!')
            return 0
        
        
        try:
            _ = data.loc[0,'ID']
        except:
            n_data = len(data)
            data['ID'] = np.arange(n_data)
        
        data_purged = data[(data['chi2'] >= 0) & (data['sum_w_post_smearing'] < 1e30)]
        data_purged.loc[:,'sum_w_post_smearing'] = np.abs(data_purged['sum_w_post_smearing'].values)
        
        
        data_purged.to_csv(path_data_source,sep='\t',float_format='%.5e',index=False)
        data.to_csv(path_data_source_ur,sep='\t',float_format='%.5e',index=False)
        
        
    
    def find_min(self, fit_source='Enu'):
        
        try:
            path_data_source = self.paths['normal'][fit_source]
            data_source = pd.read_csv(path_data_source,sep='\t')
            
            
        except:
            print('You have to do the fitting first!')
            return 0
        
        
        if self.model=='3+1':
            X_source = data_source['mzprime'].values
            Y_source = data_source['m4'].values
        elif self.model=='3+2':
            X_source = data_source['m5'].values
            Y_source = data_source['delta'].values
        
        Z_source = data_source['chi2'].values

        
        zmin = Z_source.min()
        mask_min = Z_source == zmin
        xmin_point, ymin_point = X_source[mask_min][0], Y_source[mask_min][0]
        
        return [xmin_point, ymin_point]
    
    
    def plot_fitting(self,type_fit='Enu',leg_loc='upper left',save=True):
    
        try:
            path_data = self.paths['normal'][type_fit]
            plot_path1 = self.plots_paths_fitting[type_fit]['chi2']
            plot_path2 = self.plots_paths_fitting[type_fit]['chi2_sigmas']
            plot_path3 = self.plots_paths_fitting[type_fit]['nevents']
            dof = self.dof_enu if type_fit=='Enu' else self.dof_angle
            data = pd.read_csv(path_data,sep='\t')

            plot_title1_cbar = self.plots_titles_fitting[type_fit]['chi2']
            plot_title1 = self.plots_titles_fitting[type_fit]['chi2_title']
            plot_title2 = self.plots_titles_fitting[type_fit]['chi2_sigmas']
            plot_title3_cbar = self.plots_titles_fitting[type_fit]['nevents']

        except:
            print('You have to do the fitting first!')
            return 0
        
        X = data[plotvars[self.model][0]].values
        Y = data[plotvars[self.model][1]].values
        xlabel = plotaxes[self.model][0]
        ylabel = plotaxes[self.model][1]

        
        Z = data['chi2'].values
        W = data['N_events'].values
        
        marker1 = '*'
        marker2 = 's'
        
        xmin_enu, ymin_enu = self.find_min(fit_source='Enu')
        xmin_angle, ymin_angle = self.find_min(fit_source='angle')
        
        plt.rcParams["figure.figsize"] = (6,4)
        levels = 10
        plt.tricontourf(X,Y,Z,levels=levels,cmap='viridis')
        plt.plot(xmin_enu,ymin_enu,color='orange',marker=marker1,markersize=12)
        plt.plot(xmin_angle,ymin_angle,color='orange',marker=marker2,markersize=12)
        cbar = plt.colorbar()
        cbar.set_label(plot_title1_cbar,size=15)
        plt.title(plot_title1,fontsize=15)
        plt.xlabel(xlabel,fontsize=15)
        plt.ylabel(ylabel,fontsize=15)
        plt.xscale('log')
        plt.yscale('log')
        if save:
            plt.savefig(plot_path1,dpi=400)
        else:
            plt.show()
        plt.clf()
        
        num_colors = 12
        viridis = cm.get_cmap('viridis', num_colors)
        
        min_chi2 = Z.min()*dof
        delta_chi2 = Z*dof - min_chi2

        bar_1 = mpatches.Patch(color=viridis(range(num_colors))[1], label=r'1 $\sigma$')
        bar_2 = mpatches.Patch(color=viridis(range(num_colors))[4], label=r'2 $\sigma$')
        bar_3 = mpatches.Patch(color=viridis(range(num_colors))[8], label=r'3 $\sigma$')

        plt.rcParams["figure.figsize"] = (6,4)
        levels = [0,2.3,6.18,11.83]
        plt.tricontourf(X,Y,delta_chi2,levels=levels,cmap='viridis')
        plt.tricontour(X,Y,delta_chi2,levels=levels,colors='black',linewidths=0.5)
        plt.plot(xmin_enu,ymin_enu,color='orange',marker=marker1,markersize=12)
        plt.plot(xmin_angle,ymin_angle,color='orange',marker=marker2,markersize=12)
        plt.legend(handles=[bar_1, bar_2, bar_3],fontsize=10,loc=leg_loc)
        plt.title(plot_title2+', '+self.model,fontsize=15)
        plt.xlabel(xlabel,fontsize=15)
        plt.ylabel(ylabel,fontsize=15)
        plt.xscale('log')
        plt.yscale('log')
        if save:
            plt.savefig(plot_path2,dpi=400)
        else:
            plt.show()
        plt.clf()
        
        
        plt.rcParams["figure.figsize"] = (6,4)
        levels = 15

        plt.tricontourf(X,Y,W,levels=levels,cmap='viridis')
        plt.plot(xmin_enu,ymin_enu,color='orange',marker=marker1,markersize=12)
        plt.plot(xmin_angle,ymin_angle,color='orange',marker=marker2,markersize=12)
        cbar = plt.colorbar()
        cbar.set_label(plot_title3_cbar,size=15)
        plt.title(plot_title1,fontsize=15)
        plt.xlabel(xlabel,fontsize=15)
        plt.ylabel(ylabel,fontsize=15)
        plt.xscale('log')
        plt.yscale('log')
        if save:
            plt.savefig(plot_path3,dpi=400)
        else:
            plt.show()
        plt.clf()
    
    
    def compute_chi2_grid_couplings(self, dataset, type_fit, sys, back_MC,D,couplings, num=500):
        
        desired_MB_events = 400
        length = len(couplings)
        if self.model=='3+2':
            data = np.zeros((length,5))
        else:
            data = np.zeros((length,4))
        
        # read data
        df = pd.read_pickle(dataset[0])
        
        if self.model=='3+2':
            m4 = dataset[2]
            var = dataset[1]
        elif self.model=='3+1':
            m4 = dataset[1]
            var = dataset[2]

        # compute spectrum
        bag_reco_MB = av.compute_MB_spectrum(df, EVENT_TYPE='both')
        bag_reco_MB = av.select_MB_decay(bag_reco_MB)
        sum_w_post_smearing = np.abs(np.sum(bag_reco_MB['reco_w']))

        # compute histograms
        total_Nevent_MB = desired_MB_events * np.abs((1/bag_reco_MB['reco_eff'][0]))
                    
        if type_fit=='Enu':
            histograms = plot_tools.get_histogram1D(bag_reco_MB, NEVENTS=total_Nevent_MB, varplot='reco_Enu',loc=loc)
            if self.mode_chi2:
                self.dof_enu = 6.7
            dof = self.dof_enu
        elif type_fit=='angle':
            histograms = plot_tools.get_histogram1D(bag_reco_MB, NEVENTS=total_Nevent_MB, varplot='reco_angle',loc=loc)
            dof = self.dof_angle

        # fitting MB
        NP_MC = histograms[0]
        
        for i in range(length):
            data[i,0] = m4
            data[i,1] = couplings[i]
            data[i,3] = var
            if self.model=='3+2':
                data[i,4] = var/m4 - 1
            NPevents = (couplings[i] * 1e+8)**2 * sum_w_post_smearing * NORMALIZATION
            if not(self.mode_chi2) or (type_fit=='angle'):
                data[i,2] = chi2_binned_rate(NP_MC, NPevents, back_MC, D,sys=sys) / dof
            elif (self.mode_chi2) & (type_fit=='Enu'):
                data[i,2] = chi2_MiniBooNE_2020(NP_MC, NPevents) / dof


        return data
    
    def fit_grid_couplings(self, definition_number=500, type_fit='Enu', i_m=0, i_mzprime=0,couplings=np.geomspace(0.0001,0.1,20),mode='m5'):
            
        num = definition_number
        desired_MB_events = 400
        self.coupling_mode = mode
        
        if self.same_dataset:
            ds = pd.DataFrame(self.datasets_list)
                    
                    
            if self.model=='3+2':
                if mode == 'm4':
                    fixed_mode = 'm5'
                    fixed_axis = ds['m5'].unique()
                    fixed_axis = np.sort(fixed_axis)
                elif mode == 'm5':
                    ds['delta'] = round_sig(ds['m5'].values / ds['m4'].values - 1)
                    fixed_mode = 'delta'
                    fixed_axis = ds['delta'].unique()
                    fixed_axis = np.sort(fixed_axis)
                
                fixed = fixed_axis[i_m]
                filtered_dataset = ds[ds[fixed_mode]==fixed]
            elif self.model=='3+1':
                fixed_axis = ds['mzprime'].unique()
                fixed_axis = np.sort(fixed_axis)
                mzprime = fixed_axis[i_mzprime]
                fixed = fixed_axis[i_m]
                filtered_dataset = ds[ds['mzprime']==fixed]
            
            filtered_dataset = filtered_dataset.values
        else:
            filtered_dataset = self.datasets_list_couplings
        
        # obtain data from MB for the fitting
        data_MB = data_MB_source[type_fit]

        self.dof_enu = 6.7
        if type_fit=='angle':
            self.dof_angle = len(data_MB[0]) - 2

        back_MC = data_MB[1]
        D = data_MB[0] + data_MB[1]

        if self.sys_angle!=0 & type_fit=='angle':
            sys = [self.sys_angle, self.sys_angle]
        else:
            sys = [data_MB[2], data_MB[3]]

        chi2 = lambda dataset: self.compute_chi2_grid_couplings(dataset, type_fit=type_fit, num=num, sys=sys, back_MC=back_MC,D=D,couplings=couplings)

        pool = Pool(self.nb_cores)
                                    
        chi2_lists = pool.map(chi2,filtered_dataset)
        chi2_enu_lists = np.concatenate(tuple(chi2_enu_lists))
        chi2_df = pd.DataFrame(data=chi2_lists,columns=self.columns_couplings)
        chi2_df.to_csv(self.paths['couplings'][type_fit],sep='\t',float_format='%.5e',index=False)

    
    
    def find_min_couplings(self, fit_source='Enu'):
        
        try:
            path_data_source = self.paths['couplings'][fit_source]
            
            data_source = pd.read_csv(path_data_source,sep='\t')
            data_source = data_source.values
            
            
        except:
            print('You have to do the fitting first!')
            return 0
        
        
        if self.model=='3+1' or self.coupling_mode == 'm4':
            X_source = data_source[:,0]
            Y_source = data_source[:,1]
        elif self.coupling_mode == 'm5':
            X_source = data_source[:,3]
            Y_source = data_source[:,1]
        
        Z_source = data_source[:,2]

        
        zmin = Z_source.min()
        mask_min = Z_source == zmin
        xmin_point, ymin_point = X_source[mask_min][0], Y_source[mask_min][0]
        
        return [xmin_point, ymin_point]
    
    
    def purge_grid_couplings(self, type_fit='Enu'):
        
        try:
            path_data_source = self.paths['couplings'][type_fit]
            path_data_source_ur = self.paths['original_couplings'][type_fit]
            data = pd.read_csv(path_data_source,sep='\t')
            
            
        except:
            print('You have to do the fitting first!')
            return 0
        
        try:
            _ = data.loc[0,'ID']
        except:
            n_data = len(data)
            data['ID'] = np.arange(n_data)
        
        data_purged = data[(data['chi2'] >= 0)&(data['sum_w_post_smearing']!=0)]
                
        data_purged.to_csv(path_data_source,sep='\t',float_format='%.5e',index=False)
        data.to_csv(path_data_source_ur,sep='\t',float_format='%.5e',index=False)
        
    
    def get_grid(self, type_fit='Enu', grid='normal'): #the grid options are: normal, couplings, original or original_couplings
        
        try:
            path_data = self.paths[grid][type_fit]
            data = pd.read_csv(path_data,sep='\t')
            
            return data
            
            
        except:
            print('You have to do the fitting first!')
            return 0
        
        
    
    def plot_couplings_mass(self,type_fit='Enu', leg_loc='upper left',coupling_exp=0,save=True,min_enu=['function','function'],min_angle=['function','function'],chi2_min=None,xlims=['none','none']):
    
        coupling_factor = 10**coupling_exp
        
        try:
            if type_fit=='Enu':
                path_data = self.path_enu_coup
                data = pd.read_csv(path_data,sep='\t')
                plot_title1_cbar = r'$\chi^2/dof$ for $E_\nu$'
                plot_path1 = self.path+'__grid_run__/plots/chi2_enu_couplings_'+self.neutrino_type+'_'+method+'.pdf'
                if self.model=='3+1':
                    plot_title2 = r'$\Delta \chi^2$ for $E_\nu$, $m_{Z \prime} = $' + str(data.loc[0,'mzprime']) + ' GeV, MiniBooNE, ' + self.hnl_type
                else:
                    if self.coupling_mode == 'm4':
                        coupling_name = r'$m_4$'
                        plot_title2 = r'$\Delta \chi^2$ for $E_\nu$, $m_{Z\prime}=$' + str(self.mzprime) + r'$ \ \mathrm{GeV}$,$m_{5} = $' + str(data.loc[0,'m5']) + ' GeV,MiniBooNE, ' + self.hnl_type
                    elif self.coupling_mode == 'm5':
                        coupling_name = r'$m_5$'
                        plot_title2 = r'$\Delta \chi^2$ for $E_\nu$, $m_{Z\prime}=$' + str(self.mzprime) + r'$ \ \mathrm{GeV}$,$\Delta = $' + str(round_sig([data.loc[0,'delta']])) + ',MiniBooNE, ' + self.hnl_type
                plot_path2 = self.path+'__grid_run__/plots/chi2_enu_couplings_'+self.neutrino_type+'_'+method+'_sigmas.pdf'
                dof = self.dof_enu
                
            elif type_fit=='angle':
                path_data = self.path_angle_coup
                data = pd.read_csv(path_data,sep='\t')
                plot_title1_cbar = r'$\chi^2/dof$ for $\theta_{ee}^{\mathrm{beam}}$'
                plot_path1 = self.path+'__grid_run__/plots/chi2_angle_couplings_'+self.neutrino_type+'_'+method+'.pdf'
                if self.model=='3+1':
                    plot_title2 = r'$\Delta \chi^2$ for $\theta_{ee}^{\mathrm{beam}}$, $m_{Z \prime} = $' + str(data.loc[0,'mzprime']) + ' GeV, MiniBooNE, ' + self.hnl_type
                else:
                    if self.coupling_mode == 'm4':
                        coupling_name = r'$m_4$'
                        plot_title2 = r'$\Delta \chi^2$ for $\theta_{ee}^{\mathrm{beam}}$, $m_{Z\prime}=$' + str(self.mzprime) + r'$ \ \mathrm{GeV}$,$m_{5} = $' + str(data.loc[0,'m5']) + ' GeV,MiniBooNE, ' + self.hnl_type
                    elif self.coupling_mode == 'm5':
                        coupling_name = r'$m_5$'
                        plot_title2 = r'$\Delta \chi^2$ for $\theta_{ee}^{\mathrm{beam}}$, $m_{Z\prime}=$' + str(self.mzprime) + r'$ \ \mathrm{GeV}$,$\Delta = $' + str(round_sig([data.loc[0,'delta']])) + ',MiniBooNE, ' + self.hnl_type
                        
                plot_path2 = self.path+'__grid_run__/plots/chi2_angle_couplings_'+self.neutrino_type+'_'+method+'_sigmas.pdf'
                dof = self.dof_angle
                
            
        except:
            print('You have to do the fitting first!')
            return 0
        
        ce_string = str(coupling_exp)
        
        if self.model=='3+1':
            factor_dmu = 4.
            X = data['m4'].values
            Y = data['couplings'].values  / np.sqrt(4 * np.pi)
            Y = Y * Y * coupling_factor
            xlabel = r'$m_{4} [\mathrm{GeV}]$'
            if coupling_factor==1:
                ylabel = r'$\alpha_D (\epsilon V_{\mu 4})^2$'
            else:
                if len(ce_string) == 1:
                    ylabel = r'$\alpha_D (\epsilon V_{\mu 4})^2 \times$ ' + fr'$10^{coupling_exp}$'
                else:
                    ylabel = r'$\alpha_D (\epsilon V_{\mu 4})^2 \times$ ' + fr'$10^{ce_string[0]}$'+fr'$^{ce_string[1]}$'
            plot_title1 = r'3+1, $m_{Z \prime} = $' + str(data.loc[0,'mzprime']) +  r' $\mathrm{GeV}$,MiniBooNE, ' + hnl_type
        elif self.model=='3+2':
            factor_dmu = 1.
            X = data[self.coupling_mode].values
            Y = data['couplings'].values  / np.sqrt(np.pi)
            Y = Y * Y * coupling_factor
            xlabel = coupling_name + r' $[\mathrm{GeV}]$'
            if coupling_factor==1:
                ylabel = r'$\alpha_D (\epsilon V_{\mu 5})^2$'
            else:
                if len(ce_string) == 1:
                    ylabel = r'$\alpha_D (\epsilon V_{\mu 5})^2 \times$ ' + fr'$10^{coupling_exp}$'
                else:
                    ylabel = r'$\alpha_D (\epsilon V_{\mu 5})^2 \times$ ' + fr'$10^{ce_string[0]}$'+fr'$^{ce_string[1]}$'
            
            if not(self.same_dataset):
                plot_title1 = r'3+2, $m_{Z\prime}=$' +str(self.mzprime) + r' $\mathrm{GeV}$, $\Delta = $' + str(round_sig([data.loc[0,'delta']])) + ', MiniBooNE, ' + hnl_type
            else:
                if self.coupling_mode == 'm4':
                    plot_title1 = r'3+2, $m_{Z\prime}=$' +str(self.mzprime) + r' $\mathrm{GeV}$, $m_{5} = $' + str(data.loc[0,'m5']) + r' $\mathrm{GeV}$,MiniBooNE, ' + hnl_type
                elif self.coupling_mode == 'm5':
                    plot_title1 = r'3+2, $m_{Z\prime}=$' +str(self.mzprime) + r' $\mathrm{GeV}$, $\Delta = $' + str(round_sig([data.loc[0,'delta']])) + ',MiniBooNE, ' + hnl_type
        
        Z = data['chi2'].values
        
        marker1 = '*'
        marker2 = 's'
        
        if min_enu[0] == 'function':
            xmin_enu, ymin_enu = self.find_min_couplings(fit_source='Enu')
        elif min_enu[0] != 'none':
            xmin_enu, ymin_enu = min_enu
        
        if min_angle[0] == 'function':
            xmin_angle, ymin_angle = self.find_min_couplings(fit_source='angle')
        elif min_angle[0] != 'none':
            xmin_angle, ymin_angle = min_angle
            
        if min_enu[0] != 'none':
            ymin_enu *= ymin_enu  / (factor_dmu * np.pi) * coupling_factor
        if min_angle[0] != 'none':
            ymin_angle *= ymin_angle  / (factor_dmu * np.pi) * coupling_factor
        
        plt.rcParams["figure.figsize"] = (6,4)
        levels = 10
        plt.tricontourf(X,Y,Z,levels=levels,cmap='viridis')
        if min_enu[0] != 'none':
            plt.plot(xmin_enu,ymin_enu,color='orange',marker=marker1,markersize=12)
        if min_angle[0] != 'none':
            plt.plot(xmin_angle,ymin_angle,color='orange',marker=marker2,markersize=12)
        cbar = plt.colorbar()
        cbar.set_label(plot_title1_cbar,size=15)
        plt.title(plot_title1,fontsize=10)
        plt.xlabel(xlabel,fontsize=15)
        plt.ylabel(ylabel,fontsize=15)
        plt.xscale('log')
        plt.yscale('log')
        if xlims[0] != 'none':
            plt.xlim(xlims[0],xlims[1])
            
        if save:
            plt.savefig(plot_path1,dpi=400)
        else:
            plt.show()
        plt.clf()
        
        num_colors = 12
        viridis = cm.get_cmap('viridis', num_colors)
        
        if chi2_min == None:
            min_chi2 = Z.min()*dof
        else:
            min_chi2 = chi2_min*dof
            
        delta_chi2 = Z*dof - min_chi2

        bar_1 = mpatches.Patch(color=viridis(range(num_colors))[1], label=r'1 $\sigma$')
        bar_2 = mpatches.Patch(color=viridis(range(num_colors))[4], label=r'2 $\sigma$')
        bar_3 = mpatches.Patch(color=viridis(range(num_colors))[8], label=r'3 $\sigma$')

        plt.rcParams["figure.figsize"] = (6,4)
        levels = [0,2.3,6.18,11.83]
        plt.tricontourf(X,Y,delta_chi2,levels=levels,cmap='viridis')
        plt.tricontour(X,Y,delta_chi2,levels=levels,colors='black',linewidths=0.5)
        if min_enu[0] != 'none':
            plt.plot(xmin_enu,ymin_enu,color='orange',marker=marker1,markersize=12)
        if min_angle[0] != 'none':
            plt.plot(xmin_angle,ymin_angle,color='orange',marker=marker2,markersize=12)
        plt.legend(handles=[bar_1, bar_2, bar_3],fontsize=10,loc=leg_loc)
        plt.title(plot_title2+', '+self.model,fontsize=10)
        plt.xlabel(xlabel,fontsize=15)
        plt.ylabel(ylabel,fontsize=15)
        plt.xscale('log')
        plt.yscale('log')
        if xlims[0] != 'none':
            plt.xlim(xlims[0],xlims[1])
        if save:
            plt.savefig(plot_path2,dpi=400)
        else:
            plt.show()
        plt.clf()
        
        
                
    
    def plot_strength(self, type_fit='Enu',leg_loc='upper left',levels=15,save=True):
    
        if self.neutrino_type=='dirac':
            hnl_type = r'Dirac'
        else:
            hnl_type = r'Majorana'
        
        try:
            if type_fit=='Enu':
                path_data = self.path_enu
                if self.model=='3+1':
                    plot_title_cbar = r'$\alpha_D (\epsilon V_{\mu 4})^2$ / $10^{-15}$ for $E_\nu$'
                elif self.model=='3+2':
                    plot_title_cbar = r'$\alpha_D (\epsilon V_{\mu 5})^2$ / $10^{-11}$ for $E_\nu$'
                plot_path = self.path+'__grid_run__/plots/strength_enu_'+self.neutrino_type+'_'+self.method+'.pdf'
                                
            elif type_fit=='angle':
                path_data = self.path_angle
                if self.model=='3+1':
                    plot_title_cbar = r'$\alpha_D (\epsilon V_{\mu 4})^2$ / $10^{-15}$ for $\theta_{ee}^{\mathrm{beam}}$'
                elif self.model=='3+2':
                    plot_title_cbar = r'$\alpha_D (\epsilon V_{\mu 5})^2$ / $10^{-11}$ for $\theta_{ee}^{\mathrm{beam}}$'
                plot_path = self.path+'__grid_run__/plots/strength_angle_'+self.neutrino_type+'_'+self.method+'.pdf'
                                
            data = pd.read_csv(path_data,sep='\t')
        except:
            print('You have to do the fitting first!')
            return 0
        
        
        Z = data['sum_w_post_smearing'].values * NORMALIZATION
        W = data['N_events'].values
        
        
        if self.model=='3+1':
            X = data['mzprime'].values
            Y = data['m4'].values
            xlabel = r'$m_{Z\prime}$ (GeV)'
            ylabel = r'$m_{4} (GeV)$'
            plot_title = r'3+1, MiniBooNE, ' + hnl_type
            V = np.sqrt(np.abs(W / Z)) * 1e-8  / np.sqrt(4 * np.pi)
            V = V * V * 1e15
        elif self.model=='3+2':
            X = data['m5'].values
            Y = data['delta'].values
            xlabel = r'$m_{5}$ (GeV)'
            ylabel = r'$\Delta$'
            plot_title = r'$m_{Z\prime}=' +str(self.mzprime) + r' \ \mathrm{GeV}$, 3+2, MiniBooNE, ' + hnl_type
            V = np.sqrt(np.abs(W / Z)) * 1e-8  / np.sqrt(np.pi)
            V = V * V * 1e11
        
        
        
        marker1 = '*'
        marker2 = 's'
        
        xmin_enu, ymin_enu = self.find_min(fit_source='Enu')
        xmin_angle, ymin_angle = self.find_min(fit_source='angle')
        
        plt.rcParams["figure.figsize"] = (6,4)
        
        plt.tricontourf(X,Y,V,levels=levels,locator=ticker.LogLocator(),cmap='viridis')
        plt.plot(xmin_enu,ymin_enu,color='orange',marker=marker1,markersize=12)
        plt.plot(xmin_angle,ymin_angle,color='orange',marker=marker2,markersize=12)
        cbar = plt.colorbar()
        cbar.set_label(plot_title_cbar,size=15)
        plt.title(plot_title,fontsize=12)
        plt.xlabel(xlabel,fontsize=15)
        plt.ylabel(ylabel,fontsize=15)
        plt.xscale('log')
        plt.yscale('log')
        if save:
            plt.savefig(plot_path,dpi=400)
        else:
            plt.show()
        plt.clf()
    
    
    def compute_p_value(self, fit_source='Enu',fit_point='Enu'):
        
        df = 2
        
        try:
            paths = {'Enu' : self.path_enu, 'angle' : self.path_angle}
            dof = self.dof_enu if fit_source=='Enu' else self.dof_angle

            path_data_source = paths[fit_source]
            path_data_point = paths[fit_point]
            
            data_source = pd.read_csv(path_data_source,sep='\t')
            data_point = pd.read_csv(path_data_point,sep='\t')
            
        except:
            print('You have to do the fitting first!')
            return 0
        
        
        ID_source = data_source['ID'].values
        ID_point = data_point['ID'].values
        
        Z_source = data_source['chi2'].values
        Z_point = data_point['chi2'].values
        
        zmin = Z_point.min()
        mask_min = Z_point == zmin
        idmin_point = ID_point[mask_min][0]
        
        mask_source = ID_source == idmin_point
        
        try:
            x = Z_source[mask_source][0]*dof
        
            return chi2_scipy.sf(x,df,loc=0,scale=1)
        
        except:
            return 'NAN'
    
    
    def compute_best_fits(self, fit_source='Enu'):
        
        try:
            if fit_source=='Enu':
                path_data_source = self.path_enu
            elif fit_source=='angle':
                path_data_source = self.path_angle
            
            
            data_source = pd.read_csv(path_data_source,sep='\t')
            
            
        except:
            print('You have to do the fitting first!')
            return 0
        
        
        if self.model=='3+1':
            X_source = data_source['mzprime'].values
            Y_source = data_source['m4'].values
        elif self.model=='3+2':
            X_source = data_source['m5'].values
            Y_source = data_source['delta'].values
        
        Nevents = data_source['N_events'].values
        Z_source = data_source['chi2'].values
        
        zmin = Z_source.min()
        mask_min = Z_source == zmin
        xmin, ymin, nevents = X_source[mask_min][0], Y_source[mask_min][0], Nevents[mask_min][0]
        
        return np.array([xmin,ymin,nevents])
    
    
    def matrix_p_value(self):
        
        indices = ['Enu','angle']
        
        if self.model=='3+2':
            path_p = self.path+'__grid_run__/p_values_z_' + str(self.mzprime)+ '_3+2_' + self.neutrino_type+'_'+self.method + '.dat'
        elif self.model=='3+1':
            path_p = self.path+'__grid_run__/p_values_3+1_' + self.neutrino_type+'_'+self.method + '.dat'
        
        p_array = np.array([[self.compute_p_value(fit_source=i,fit_point=j) for j in indices] for i in indices])
        
        p_df = pd.DataFrame(data=p_array, index=['Plot Enu', 'Plot angle'], columns=['Best fit Enu', 'Best fit angle'])
        
        p_df.to_csv(path_p,sep='\t')
        
        return p_df
    
    
    def best_fits(self):
        
        indices = ['Enu','angle']
        if self.model=='3+2':
            columns = ['m5','delta','N_events']
            path_bf = self.path+'__grid_run__/bf_z_' + str(self.mzprime)+ '_3+2_' + self.neutrino_type+'_'+self.method + '.dat'
        elif self.model=='3+1':
            columns = ['mzprime','m4','N_events']
            path_bf = self.path+'__grid_run__/bf_3+1_' + self.neutrino_type+'_'+self.method + '.dat'
        
        bf_array = np.array([self.compute_best_fits(fit_source=fs) for fs in indices])
        
        bf_df = pd.DataFrame(data=bf_array,index=indices,columns=columns)
        
        bf_df.to_csv(path_bf,sep='\t')
        
        return bf_df
        
        
    def plot_bf_distribution(self, fit_source='Enu', kde=False, rounding=2):
        
        df_bf = self.best_fits()
        
        
        n = len(self.datasets_list)
        
        
        if self.model=='3+2':
            m5_bf = df_bf.loc[fit_source,'m5']
            delta_bf = df_bf.loc[fit_source,'delta']
            for i in range(n):
                m5 = self.datasets_list[i]['m5']
                m4 = self.datasets_list[i]['m4']
                delta = round_sig((m5 - m4) / m4, sig=rounding)
                m5 = round_sig(m5 , sig=rounding)
                if (round_sig(m5_bf,sig=rounding) == m5) & (round_sig(delta_bf,sig=rounding) == delta):
                    location = self.datasets_list[i]['dataset']
        elif self.model=='3+1':
            m4_bf = df_bf.loc[fit_source,'m4']
            mzprime_bf = df_bf.loc[fit_source,'mzprime']
            for i in range(n):
                mzprime = round_sig(self.datasets_list[i]['mzprime'], sig=rounding)
                m4 = round_sig(self.datasets_list[i]['m4'], sig=rounding)
                if (round_sig(m4_bf,sig=rounding) == m4) & (round_sig(mzprime_bf,sig=rounding) == mzprime):
                    location = self.datasets_list[i]['dataset']
        
        
        df = pd.read_pickle(location)
        PATH = self.path + '__grid_run__/plots/' + fit_source + '_' + self.method
        nevents = df_bf.loc[fit_source,'N_events']
        
        if fit_source == 'Enu':
            source = r'$E_\nu$'
        elif fit_source == 'angle':
            source = r'$\theta_{ee}^{\mathrm{beam}}$'
        
        title = 'MiniBooNE: best fit for ' + source + ', ' + self.model + ', ' + self.hnl_type
        
        plot_tools.plot_all_rates(df, fit_source, Nevents=nevents, title=title, plot_muB = False, path=PATH,loc='../')
    
        
    def plot_bf_xsection(self, definition_number=500, fit_source='Enu', regime = 'coherent', gD=1.0, epsilon=1e-2, Umu4=1e-6, UD4=1.0):
        
        N_events_LEE = 560.6
        
        if self.neutrino_type=='dirac':
            hnl_type = r'Dirac'
        else:
            hnl_type = r'Majorana'
        
        df_bf = self.best_fits()
        
        if fit_source=='Enu':
            source = r'$E_\nu$'
        elif fit_source=='angle':
            source = r'$\theta_{ee}^{\mathrm{beam}}$'
        
        m4_bf = df_bf.loc[fit_source,'m4']
        mzprime_bf = df_bf.loc[fit_source,'mzprime']
        N_events = df_bf.loc[fit_source,'N_events']
        
        xsecs = self.compute_xsecs(m4_bf, mzprime_bf, gD=gD, epsilon=epsilon, Umu4=Umu4, UD4=UD4)
        
        theseMCs = xsecs[0]
        mc_lowT = xsecs[1]
        
        fig, ax = plot_tools.std_fig()
        enu_axis = np.geomspace(mc_lowT.Ethreshold*0.99, 5, 100)

        all_args={
            f'H1_conserving_{regime}': {'color': 'black', 'ls': '-'},
            f'C12_conserving_{regime}': {'color': 'royalblue', 'ls': '-'},
            f'Ar40_conserving_{regime}': {'color': 'violet', 'ls': '-'},
            f'H1_flipping_{regime}': {'color': 'black', 'ls': '--'},
            f'C12_flipping_{regime}': {'color': 'royalblue', 'ls': '--'},
            f'Ar40_flipping_{regime}': {'color': 'violet', 'ls': '--'},
            }    

        for key, mc in theseMCs.items():
            if regime in key:
                if regime == 'coherent' and "H1" in key:
                    continue
                else:
                    sigmas = mc.sigmas

                    ls = '-' if ('conserving' in key) else '--'
                    args = all_args[key]
                    p = mc.ups_case.TheoryModel
                    #norm = (p.Umu4*p.epsilon*const.eQED*p.gD*p.UD4*mc.target.Z)**2
                    norm = N_events_LEE / N_events
                    #                 print(norm/(mc.ups_case.Vij**2*mc.ups_case.Vhad**2))
                    if 'conserving' in key:
                        ax.plot(enu_axis, sigmas/norm, label=key.replace("_", " ").replace("conserving", "HC").replace("coherent", "coh"), **args)
                    else:
                        ax.plot(enu_axis, sigmas/norm, **args)

        ax.set_title(fr'$m_{{Z^\prime}}= {mc.ups_case.mzprime:.2f}$ GeV,  $m_4 = {mc.ups_case.m_ups*1e3:.0f}$ MeV' + ', ' + hnl_type + ', bf for ' + source, fontsize=12)
        ax.set_yscale("log")
        ax.set_xscale("log")

        ax.set_xlabel(r"$E_\nu$ (GeV)")
        #ax.set_ylabel(r"$\sigma/(Z e \epsilon V_{\mu 4})^2$ (cm$^2$)", fontsize=12)
        ax.set_ylabel(r"$\sigma$ (cm$^2$)", fontsize=12)
        ax.set_xlim(0.1,np.max(enu_axis))
        #     ax.set_ylim(1e-34,1e-26)

        ax.legend(loc="best", frameon=False)
        ax.grid(which='major', lw=0.5)

        fig.savefig(self.path + "__grid_run__/plots/xsec_3+1_" + fit_source + f"_{mc.ups_case.mzprime:.2f}_{mc.ups_case.m_ups*1e3:.0f}_{regime}_" + self.neutrino_type +'_'+self.method + ".pdf",dpi=400)
        
        
    def do_whole_analysis(self,really_run='False',processes=[True,False,False,True,True,False,False,False,False,False,False,True,True,True,True], chi2_enu=True,chi2_angle=True,couplings_light_enu=np.geomspace(0.1e-23,2.5e-23,40),couplings_light_angle=np.geomspace(0.1e-23,2.5e-23,40),couplings_heavy_enu=np.geomspace(0.1e-23,2.5e-23,40),couplings_heavy_angle=np.geomspace(0.1e-23,2.5e-23,40),couplings_mode='m5',im_light=2,im_heavy=18,coupling_exp_light=13,coupling_exp_heavy=13,rounding=2):
        
        if processes[0]:
            print('Running grid')
            self.run_grid(really_run=really_run)
        
        if chi2_enu or processes[1]:
            print('Fitting Enu')
            self.fit_grid(type_fit='Enu')
        if chi2_angle or processes[2]:
            print('Fitting angle')
            self.fit_grid(type_fit='angle')
            
        if processes[3]:
            print('Plotting fittings')
            self.purge_grid(type_fit='Enu')
            self.purge_grid(type_fit='angle')
            self.plot_fitting(type_fit='Enu')
            self.plot_fitting(type_fit='angle')
        
        if processes[4]:
            print('Plotting strength fittings')
            self.plot_strength(type_fit='Enu')
            self.plot_strength(type_fit='angle')
        
        if processes[5]:
            print('Fitting couplings Enu for light particle')
            self.set_couplings_case(name='light')
            self.fit_grid_couplings(type_fit='Enu', i_m=im_light,couplings=couplings_light_enu,mode=couplings_mode)
        if processes[6]:
            print('Fitting couplings angle for light particle')
            self.set_couplings_case(name='light')
            self.fit_grid_couplings(type_fit='angle', i_m=im_light,couplings=couplings_light_angle,mode=couplings_mode)
        if processes[7]:
            print('Plotting fit of couplings for light particle')
            self.set_couplings_case(name='light')
            self.purge_grid_couplings(type_fit='Enu')
            self.purge_grid_couplings(type_fit='angle')
            self.plot_couplings_mass(type_fit='Enu',leg_loc='upper left',coupling_exp=coupling_exp_light)
            self.plot_couplings_mass(type_fit='angle',leg_loc='upper left',coupling_exp=coupling_exp_light)
        
        if processes[8]:
            print('Fitting couplings Enu for heavy particle')
            self.set_couplings_case(name='heavy')
            self.fit_grid_couplings(type_fit='Enu', i_m=im_heavy,couplings=couplings_heavy_enu,mode=couplings_mode)
        if processes[9]:
            print('Fitting couplings angle for heavy particle')
            self.set_couplings_case(name='heavy')
            self.fit_grid_couplings(type_fit='angle', i_m=im_heavy,couplings=couplings_heavy_angle,mode=couplings_mode)
        if processes[10]:
            print('Plotting fit of couplings for heavy particle')
            self.set_couplings_case(name='heavy')
            self.purge_grid_couplings(type_fit='Enu')
            self.purge_grid_couplings(type_fit='angle')
            self.plot_couplings_mass(type_fit='Enu',leg_loc='upper left',coupling_exp=coupling_exp_heavy)
            self.plot_couplings_mass(type_fit='angle',leg_loc='upper left',coupling_exp=coupling_exp_heavy)
        
        if processes[11]:
            print('Computing p-values')
            self.matrix_p_value()
        if processes[12]:
            print('Printing best fits')
            self.best_fits()
        
        if processes[13]:
            print('Plotting distribution for best fits, Enu')
            self.plot_bf_distribution(fit_source='Enu', kde=True,rounding=rounding)
        if processes[14]:
            print('Plotting distribution for best fits, angle')
            self.plot_bf_distribution(fit_source='angle', kde=True,rounding=rounding)
    
    
    def do_couplings_analysis(self,really_run=True,descriptor='light',delta=1,mzprime=1.25,m4=np.geomspace(0.01, 0.75, 20),couplings_mode='m5',couplings_enu=np.geomspace(1e-5,1e-1,50),couplings_angle=np.geomspace(1e-5,1e-1,50),fit_enu=True,fit_angle=True):
        
        method_plot = self.method + '_' + descriptor
        
        print('Running grid')
        self.run_grid_couplings(really_run=really_run, delta=delta,mzprime=mzprime,m4=m4)
        
        if fit_enu:
            print('Fitting couplings Enu for ' + descriptor + ' particle')
            self.fit_grid_couplings(type_fit='Enu', couplings=couplings_enu,mode=couplings_mode,method=method_plot)
        
        if fit_angle:
            print('Fitting couplings angle for ' + descriptor + ' particle')
            self.fit_grid_couplings(type_fit='angle', couplings=couplings_angle,mode=couplings_mode,method=method_plot)
        
        print('Plotting fit of couplings for ' + descriptor + ' particle')
        self.purge_grid_couplings(type_fit='Enu', method=method_plot)
        self.purge_grid_couplings(type_fit='angle', method=method_plot)
        self.plot_couplings_mass(type_fit='Enu',leg_loc='upper left',method=method_plot,coupling_factor=1)
        self.plot_couplings_mass(type_fit='angle',leg_loc='upper left', method=method_plot, coupling_factor=1)
