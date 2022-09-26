import numpy as np
import pandas as pd
import scipy
import os
from pathos.multiprocessing import ProcessingPool as Pool

import importlib.resources as resources

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import cm

from DarkNews import const 
from DarkNews.GenLauncher import GenLauncher

from . import fit_functions as ff
from . import analysis as av2
from . import analysis_decay as av

# default values
UD4_def = 1.0/np.sqrt(2.)
UD5_def = 1.0/np.sqrt(2.)
gD_def = 2.
Umu4_def = np.sqrt(1.0e-12)
Umu5_def = np.sqrt(1.0e-12)
epsilon_def = 8e-4

def get_data_MB(varplot='reco_Evis'):

    if varplot=='reco_Evis':
        _, data = np.genfromtxt(resources.open_text("ToyAnalysis.include.miniboone_2020", 'Evis_data.dat'), unpack=True)
        _, bkg = np.genfromtxt(resources.open_text("ToyAnalysis.include.miniboone_2020", 'Evis_bkg.dat'), unpack=True)
        signal = data - bkg
        sys_signal = 0.1
        sys_bkg = 0.1

    elif varplot=='reco_Enu':
        # miniboone nu data 2020
        _, data = np.genfromtxt(resources.open_text("ToyAnalysis.include.miniboone_2020", 'Enu_data.dat'), unpack=True)
        _, bkg = np.genfromtxt(resources.open_text("ToyAnalysis.include.miniboone_2020", 'Enu_constrained_bkg.dat'), unpack=True)
        _, error_low = np.genfromtxt(resources.open_text("ToyAnalysis.include.miniboone_2020", 'Enu_lower_error_bar_constrained_bkg.dat'), unpack=True)
        signal = data - bkg
        sys_bkg = (bkg - error_low)/bkg
        sys_signal = 0.1
        bin_e = np.genfromtxt(resources.open_text("ToyAnalysis.include.miniboone_2020", 'Enu_bin_edges.dat'), unpack=True)
        bin_w = (bin_e[1:] - bin_e[:-1])
        signal *= bin_w*1e3
        bkg *= bin_w*1e3

    elif varplot=='reco_angle':
        _, data = np.genfromtxt(resources.open_text('ToyAnalysis.include.miniboone_2020', 'cosTheta_data.dat'), unpack=True)
        _, bkg = np.genfromtxt(resources.open_text('ToyAnalysis.include.miniboone_2020', 'cosTheta_bkg.dat'), unpack=True)
        signal = data - bkg
        sys_signal = 0.1
        sys_bkg = 0.1

    return [signal,bkg,sys_signal,sys_bkg]


def get_events_df(model='3+1',experiment='miniboone_fhc',neval=100000, HNLtype="dirac",mzprime=1.25,m4=0.8,m5=1.0,UD4=UD4_def,UD5=UD5_def,Umu4=Umu4_def,Umu5=Umu5_def,gD=gD_def,epsilon=epsilon_def, **kwargs):
    if model=='3+1':
        gen = GenLauncher(mzprime=mzprime, m4=m4, Umu4=Umu4, UD4=UD4, gD=gD,epsilon=epsilon, neval=neval, HNLtype=HNLtype, experiment=experiment,sparse=True,print_to_float32=True, pandas=False, parquet=True, **kwargs)
    elif model=='3+1':
        gen = GenLauncher(mzprime=mzprime, m4=m4, m5=m5, Umu4=Umu4, Umu5=Umu5, UD4=UD4, UD5=UD5, gD=gD,epsilon=epsilon, neval=neval, HNLtype=HNLtype, experiment=experiment,sparse=True,print_to_float32=True, pandas=False, parquet=True, **kwargs)
    gen.run(loglevel="ERROR")
    df = gen.df
    decay_l = const.get_decay_rate_in_cm(np.sum(df.w_decay_rate_0))

    df = df[df.w_event_rate>0]

    if experiment=='miniboone_fhc':
        df = av2.compute_spectrum(df, EXP='miniboone',EVENT_TYPE='both')
    elif experiment=='miniboone_fhc':
        df = av.set_params(df)
    df = df[df.reco_w>0]

    if experiment=='miniboone_fhc':
        df = av.select_MB_decay_expo_prob(df,l_decay_proper_cm=decay_l)
    elif experiment=='microboone':
        df = av.select_muB_decay(df,l_decay_proper_cm=decay_l)
    df.reset_index(inplace=True)

    return df

class grid_analysis:

    def __init__(self,model='3+1',
                    experiment='miniboone_fhc',
                    neval=100000, 
                    HNLtype="dirac",
                    x_label='mzprime',
                    y_label='m4',
                    x_range=(0.02,10,10),
                    y_range=(0.01,2,10),
                    log_interval_x=True,
                    log_interval_y=True,
                    mzprime=None,
                    m4=None,
                    m5=None,
                    delta=None,
                    UmuN_max=1e-2,
                    UD4=UD4_def,
                    UD5=UD5_def,
                    Umu4=Umu4_def,
                    Umu5=Umu5_def,
                    gD=gD_def,
                    epsilon=8e-4,
                    cores=1,
                    output_file=None):
        
        # initialize model parameters
        self.model = model
        self.experiment = experiment
        self.neval = neval
        self.HNLtype = HNLtype
        self.cores = cores
        self.x_label = x_label
        self.y_label = y_label
        self.UmuN_max = UmuN_max
        self.df = None
        self.log_interval_x = log_interval_x
        self.log_interval_y = log_interval_y
        
        if output_file:
            self.output_file = output_file
        else:
            self.output_file = f'fit_{model[0]}p{model[2]}_{HNLtype}_{experiment}.dat'
        keys_3p1 = ['mzprime','m4']
        keys_3p2 = ['mzprime','m4','m5','delta']
        if log_interval_x:
            interval_func_x = np.geomspace
        else:
            interval_func_x = np.linspace
        if log_interval_y:
            interval_func_y = np.geomspace
        else:
            interval_func_y = np.linspace

        if (model == '3+1'):
            self.params = {key: None for key in keys_3p1}
            self.params[x_label] = ff.round_sig(interval_func_x(*x_range),4)
            self.params[y_label] = ff.round_sig(interval_func_y(*y_range),4)
        elif (model == '3+2'):
            self.params = {key: None for key in keys_3p2}
            self.params['mzprime'] = ff.round_sig(mzprime,4)
            self.params['m4'] = ff.round_sig(m4,4)
            self.params['m5'] = ff.round_sig(m5,4)
            self.params['delta'] = ff.round_sig(delta,4)
            self.params[x_label] = ff.round_sig(interval_func_x(*x_range),4)
            self.params[y_label] = ff.round_sig(interval_func_y(*y_range),4)

        # initialize grid parameters
        self.grid_params = {'x_label' : x_label, 'y_label' : y_label, 'x_points' : x_range[2], 'y_points' : y_range[2], 'UmuN_max' : UmuN_max}
        if (model == '3+1'):
            self.cols = ['mzprime','m4','chi2','N_events','couplings','decay_length','sum_w_post_smearing','v_4i']
        elif (model == '3+2'):
            self.cols=['m5','m4','delta','sum_w_post_smearing','couplings','chi2','decay_length','N_events','mzprime']
        
        # initialize data from experiment
        if self.experiment=='miniboone_fhc':
            self.data_enu = get_data_MB()
            self.back_MC_enu = self.data_enu[1]
            self.D_enu = self.data_enu[0] + self.data_enu[1]
            self.sys_enu = [self.data_enu[2], self.data_enu[3]]
        
        # initialize default coupling parameters for generation of events
        if (model == '3+1'):
            self.couplings_def = {'UD4' : UD4_def, 'gD' : gD_def, 'Umu4' : Umu4_def, 'epsilon' : epsilon_def if epsilon_def else 8e-4}
            self.v4i_def = self.couplings_def['gD'] * self.couplings_def['UD4'] * self.couplings_def['Umu4']
            self.vmu4_def = self.couplings_def['gD'] * self.couplings_def['UD4'] * self.couplings_def['UD4'] * self.couplings_def['Umu4'] / np.sqrt(1-self.couplings_def['Umu4']**2)
        elif (model == '3+2'):
            self.couplings_def = {'UD4' : UD4_def, 'UD5' : UD5_def, 'gD' : gD_def, 'Umu4' : Umu4_def, 'Umu5' : Umu5_def, 'epsilon' : epsilon_def if epsilon_def else 1e-2}
            self.vmu5_def = self.couplings_def['gD'] * self.couplings_def['UD5'] * (self.couplings_def['Umu4']*self.couplings_def['UD4'] + self.couplings_def['Umu5']*self.couplings_def['UD5']) / np.sqrt(1 - self.couplings_def['Umu4']**2 - self.couplings_def['Umu5']**2)
        
        # initialize coupling parameters to be considered
        if (model == '3+1'):
            self.couplings = {'UD4' : UD4 if UD4 else UD4_def, 'gD' : gD if gD else gD_def, 'Umu4' : Umu4 if Umu4 else Umu4_def, 'epsilon' : epsilon if epsilon else 8e-4}
            self.v4i = lambda umu4 : self.couplings['gD'] * self.couplings['UD4'] * umu4
            self.vmu4 = lambda umu4 : self.couplings['gD'] * self.couplings['UD4'] * self.couplings['UD4'] * umu4 / np.sqrt(1-umu4**2)
        elif (model == '3+2'):
            self.couplings = {'UD4' : UD4 if UD4 else UD4_def, 'UD5' : UD5 if UD5 else UD5_def, 'gD' : gD if gD else gD_def, 'Umu4' : Umu4 if Umu4 else Umu4_def, 'Umu5' : Umu5 if Umu5 else Umu5_def, 'epsilon' : epsilon if epsilon else 1e-2}
            self.vmu5 = lambda umu : self.couplings['gD'] * self.couplings['UD5'] * (umu[0]*self.couplings['UD4'] + umu[1]*self.couplings['UD5']) / np.sqrt(1 - umu[0]**2 - umu[1]**2)
        
        self.r_eps = self.couplings['epsilon']/self.couplings_def['epsilon']
    


    def generate_events(self,location='data', **kwargs):
        if self.model == '3+1':
            def chi2_grid_run(k_y):
                for k_x in range(self.grid_params['x_points']):
            
                    x_s = self.params[self.x_label][k_x]
                    y_s = self.params[self.y_label][k_y]
                    
                    if self.x_label == 'm4':
                        m4s = x_s
                        mzs = y_s
                    else:
                        m4s = y_s
                        mzs = x_s

                    try:
                        pd.read_parquet(location + '/' + self.experiment + f'/3plus1/m4_{m4s}_mzprime_{mzs}_'+self.HNLtype+'/pandas_df.parquet', engine='pyarrow')
                    except:
                        try:
                            gen = GenLauncher(mzprime=mzs, m4=m4s, Umu4=self.couplings_def['Umu4'], UD4=self.couplings_def['UD4'], gD=self.couplings_def['gD'],epsilon=self.couplings_def['epsilon'], neval=self.neval, HNLtype=self.HNLtype, experiment=self.experiment,sparse=True,print_to_float32=True, pandas=False, parquet=True, **kwargs)
                            gen.run(loglevel="ERROR")
                        except:
                            continue

        elif self.model == '3+2':
            def chi2_grid_run(k_y):
                for k_x in range(self.grid_params['x_points']):
                    if (self.x_label == 'm5') & (self.y_label == 'delta'):
                        m5s = self.params[self.x_label][k_x]
                        deltas = self.params[self.y_label][k_y]
                        m4s = ff.round_sig(m5s / (deltas + 1),4)
                        mzs = self.params['mzprime']
                    elif (self.x_label == 'delta') & (self.y_label == 'm5'):
                        deltas = self.params[self.x_label][k_x]
                        m5s = self.params[self.y_label][k_y]
                        m4s = ff.round_sig(m5s / (deltas + 1),4)
                        mzs = self.params['mzprime']
                    elif (self.x_label == 'm5') & (self.y_label == 'm4'):
                        m5s = self.params[self.x_label][k_x]
                        m4s = self.params[self.y_label][k_y]
                        mzs = self.params['mzprime']
                    elif (self.x_label == 'mzprime') & (self.y_label == 'm4'):
                        mzs = self.params[self.x_label][k_x]
                        m4s = self.params[self.y_label][k_y]
                        if self.params['delta']:
                            m5s = ff.round_sig(m4s*(self.params['delta']+1),4)
                        else:
                            m5s = self.params['m5']
                    elif (self.x_label == 'm4') & (self.y_label == 'mzprime'):
                        m4s = self.params[self.x_label][k_x]
                        mzs = self.params[self.y_label][k_y]
                        if self.params['delta']:
                            m5s = ff.round_sig(m4s*(self.params['delta']+1),4)
                        else:
                            m5s = self.params['m5']
                    elif (self.x_label == 'm5') & (self.y_label == 'mzprime'):
                        m5s = self.params[self.x_label][k_x]
                        mzs = self.params[self.y_label][k_y]
                        if self.params['delta']:
                            m4s = ff.round_sig(m5s/(self.params['delta']+1),4)
                        else:
                            m4s = self.params['m4']
                    elif (self.x_label == 'mzprime') & (self.y_label == 'm5'):
                        mzs = self.params[self.x_label][k_x]
                        m5s = self.params[self.y_label][k_y]
                        if self.params['delta']:
                            m4s = ff.round_sig(m5s/(self.params['delta']+1),4)
                        else:
                            m4s = self.params['m4']
                    elif (self.x_label == 'mzprime') & (self.y_label == 'delta'):
                        mzs = self.params[self.x_label][k_x]
                        deltas = self.params[self.y_label][k_y]
                        if self.params['m4']:
                            m4s = self.params['m4']
                            m5s = ff.round_sig(m4s*(self.params['delta']+1),4)
                        else:
                            m5s = self.params['m5']
                            m4s = ff.round_sig(m5s/(self.params['delta']+1),4)
                    
                    if (m5s <= m4s):
                        continue
                    try:
                        pd.read_parquet(location + '/' + self.experiment + f'/3plus2/m5_{m5s}_m4_{m4s}_mzprime_{mzs}_'+self.HNLtype+'/pandas_df.parquet', engine='pyarrow')
                    except:
                        try:
                            gen = GenLauncher(mzprime=mzs, m5=m5s, m4=m4s, Umu4=self.couplings_def['Umu4'], Umu5=self.couplings_def['Umu5'], UD4=self.couplings_def['UD4'], UD5=self.couplings_def['UD5'], gD=self.couplings_def['gD'],epsilon=self.couplings_def['epsilon'], neval=self.neval, HNLtype=self.HNLtype, experiment=self.experiment,sparse=True,print_to_float32=True, pandas=False, parquet=True, **kwargs)
                            gen.run(loglevel="ERROR")
                        except:
                            continue

        pool = Pool(self.cores)
        pool.map(chi2_grid_run,range(self.grid_params['y_points']))

        return "Success!"
    

    def fit_events(self,location='data',location_fit='data_fitting'):
        os.makedirs(location_fit, exist_ok=True)
        if self.model == '3+1':
            A_cut = self.UmuN_max
            data_list = [location_fit+f'/chi2_enu_3p1_y_axis_{k}.dat' for k in range(self.grid_params['y_points'])]
            def chi2_grid(k_y):
                data_tot_enu = []
                try:
                    pd.read_csv(data_list[k_y],sep='\t')
                    return "Success!"
                except:
                    for k_x in range(self.grid_params['x_points']):
                
                        x_s = self.params[self.x_label][k_x]
                        y_s = self.params[self.y_label][k_y]
                        
                        if self.x_label == 'm4':
                            m4s = x_s
                            mzs = y_s
                        else:
                            m4s = y_s
                            mzs = x_s

                        data_enu = [[mzs,m4s,0,0,0,0,0,0]]
                        
                        try:
                            df = pd.read_parquet(location + '/' + self.experiment + f'/3plus1/m4_{m4s}_mzprime_{mzs}_'+self.HNLtype+'/pandas_df.parquet', engine='pyarrow')
                        except:
                            continue
                        
                        decay_l = const.get_decay_rate_in_cm(np.sum(df.w_decay_rate_0))
                        data_enu[0][5] = decay_l
                        df = av2.compute_spectrum(df, EVENT_TYPE='both')
                        df = df[df.reco_w > 0]
                        
                        if m4s > mzs:
                            on_shell = True
                        else:
                            on_shell = False
                        
                        gen_num = np.abs(np.sum(df['reco_w'])) * self.r_eps**2
                        estim_umu4 = np.sqrt(560./gen_num) * self.vmu4_def # we are assuming vmu4 is close to umu4
                        
                        init_guess = np.min([estim_umu4,A_cut])
                        init_guess = np.arccos(np.sqrt(np.abs(init_guess/A_cut)))
                        chi2_enu = lambda theta: ff.chi2_MiniBooNE_2020_3p1(df,A_cut*np.cos(theta)**2,on_shell=on_shell, r_eps=self.r_eps,l_decay_proper_cm=decay_l)
                        
                        res_enu = scipy.optimize.minimize(chi2_enu, init_guess)
                        
                        umu4_bf = A_cut*np.cos(res_enu.x[0])**2
                        if on_shell:
                            df_decay = av.select_MB_decay_expo_prob(df,coupling_factor=self.v4i(umu4_bf)/self.v4i_def,l_decay_proper_cm=decay_l)
                        else:
                            df_decay = av.select_MB_decay_expo_prob(df,coupling_factor=self.v4i(umu4_bf)/self.v4i_def * self.r_eps,l_decay_proper_cm=decay_l)
                        sum_w_post_smearing = np.abs(np.sum(df_decay['reco_w'])) * self.r_eps**2
                        vmu4_bf = self.vmu4(umu4_bf)
                        
                        data_enu[0][3] = (vmu4_bf / self.vmu4_def)**2 * sum_w_post_smearing
                        data_enu[0][2] = res_enu.fun
                        data_enu[0][6] = sum_w_post_smearing
                        data_enu[0][7] = self.v4i(vmu4_bf)
                        data_enu[0][4] = vmu4_bf
                        
                                        
                        data_tot_enu += data_enu

                    data_enu = pd.DataFrame(data=data_tot_enu,columns=self.cols)
                    data_enu.to_csv(data_list[k_y],sep='\t',index=False)            
                    return "Success!"
        

        elif self.model == '3+2':
            v_cut = self.vmu5(self.UmuN_max,self.UmuN_max)
            data_list = [location_fit+f'/chi2_enu_3p2_y_axis_{k}.dat' for k in range(self.grid_params['y_points'])]
            def chi2_grid(k_y):
                data_tot_enu = []
                try:
                    pd.read_csv(data_list[k_y],sep='\t')
                    return "Success!"
                except:
                    for k_x in range(self.grid_params['x_points']):
                        if (self.x_label == 'm5') & (self.y_label == 'delta'):
                            m5s = self.params[self.x_label][k_x]
                            deltas = self.params[self.y_label][k_y]
                            m4s = ff.round_sig(m5s / (deltas + 1),4)
                            mzs = self.params['mzprime']
                        elif (self.x_label == 'delta') & (self.y_label == 'm5'):
                            deltas = self.params[self.x_label][k_x]
                            m5s = self.params[self.y_label][k_y]
                            m4s = ff.round_sig(m5s / (deltas + 1),4)
                            mzs = self.params['mzprime']
                        elif (self.x_label == 'm5') & (self.y_label == 'm4'):
                            m5s = self.params[self.x_label][k_x]
                            m4s = self.params[self.y_label][k_y]
                            mzs = self.params['mzprime']
                        elif (self.x_label == 'mzprime') & (self.y_label == 'm4'):
                            mzs = self.params[self.x_label][k_x]
                            m4s = self.params[self.y_label][k_y]
                            if self.params['delta']:
                                m5s = ff.round_sig(m4s*(self.params['delta']+1),4)
                            else:
                                m5s = self.params['m5']
                        elif (self.x_label == 'm4') & (self.y_label == 'mzprime'):
                            m4s = self.params[self.x_label][k_x]
                            mzs = self.params[self.y_label][k_y]
                            if self.params['delta']:
                                m5s = ff.round_sig(m4s*(self.params['delta']+1),4)
                            else:
                                m5s = self.params['m5']
                        elif (self.x_label == 'm5') & (self.y_label == 'mzprime'):
                            m5s = self.params[self.x_label][k_x]
                            mzs = self.params[self.y_label][k_y]
                            if self.params['delta']:
                                m4s = ff.round_sig(m5s/(self.params['delta']+1),4)
                            else:
                                m4s = self.params['m4']
                        elif (self.x_label == 'mzprime') & (self.y_label == 'm5'):
                            mzs = self.params[self.x_label][k_x]
                            m5s = self.params[self.y_label][k_y]
                            if self.params['delta']:
                                m4s = ff.round_sig(m5s/(self.params['delta']+1),4)
                            else:
                                m4s = self.params['m4']
                        elif (self.x_label == 'mzprime') & (self.y_label == 'delta'):
                            mzs = self.params[self.x_label][k_x]
                            deltas = self.params[self.y_label][k_y]
                            if self.params['m4']:
                                m4s = self.params['m4']
                                m5s = ff.round_sig(m4s*(self.params['delta']+1),4)
                            else:
                                m5s = self.params['m5']
                                m4s = ff.round_sig(m5s/(self.params['delta']+1),4)
                        
                        if (m5s <= m4s):
                            continue

                        try:
                            pd.read_parquet(location + '/' + self.experiment + f'/3plus2/m5_{m5s}_m4_{m4s}_mzprime_{mzs}_'+self.HNLtype+'/pandas_df.parquet', engine='pyarrow')
                        except:
                            continue
                        
                        data_enu = [[m5s,m4s,deltas,0,0,0,0,0,mzs]]
                        decay_l = const.get_decay_rate_in_cm(np.sum(df.w_decay_rate_0))
                        
                        df = av2.compute_spectrum(df, EVENT_TYPE='both')
                        df = df[df.reco_w>0]
                        
                        if (m5s - m4s >= mzs):
                            df = av.select_MB_decay_expo_prob(df,coupling_factor=1,l_decay_proper_cm=decay_l)
                            data_enu[0][6] = decay_l
                        else:
                            df = av.select_MB_decay_expo_prob(df,coupling_factor= self.r_eps,l_decay_proper_cm=decay_l)
                            data_enu[0][6] = decay_l / self.r_eps**2
                        df = df[df.reco_w>0]
                        sum_w_post_smearing = np.abs(np.sum(df['reco_w'])) * self.r_eps**2
                        
                        data_enu[0][3] = sum_w_post_smearing
                        
                        if sum_w_post_smearing!=0:
                            guess0 = np.sqrt(np.sqrt(560./sum_w_post_smearing)*self.vmu5_def/v_cut)
                        else:
                            guess0 = 1
                        guess = np.min([guess0,1])
                        theta_guess = np.arccos(guess)
                        init_guess = np.array([theta_guess])
                        chi2_enu = lambda theta: ff.chi2_MiniBooNE_2020_3p2(df, v_cut*np.cos(theta)**2,r_eps=self.r_eps)
                        
                        res_enu = scipy.optimize.minimize(chi2_enu, init_guess)
                        
                        vmu5_bf = v_cut*np.cos(res_enu.x[0])**2
                                
                        data_enu[0][4] = vmu5_bf
                        data_enu[0][5] = res_enu.fun
                        
                        
                        data_enu[0][7] = (vmu5_bf / self.vmu5_def)**2 * sum_w_post_smearing
                        
                        data_tot_enu += data_enu
                                
                    data_enu = pd.DataFrame(data=data_tot_enu,columns=self.cols)
                    data_enu.to_csv(data_list[k_y],sep='\t',index=False)            
                    return "Success!"

        pool2 = Pool(self.cores)
        pool2.map(chi2_grid,range(self.grid_params['y_points']))
        
        data_enu = pd.DataFrame(data=[],columns=self.cols)
        for k in range(self.grid_params['y_points']):
            try:
                data = pd.read_csv(data_list[k],sep='\t')
                data_enu = pd.concat([data_enu, data], ignore_index=True)
            except:
                continue
        
        data_enu = data_enu[data_enu.chi2>0]
        path_grid = location_fit + '/' + self.output_file
        data_enu.to_csv(path_grid,sep='\t',index=False)
        self.df = data_enu
    

    def plot(self,leg_loc = 'lower left',plot_path=False, save=True, title=False, x_limits=[False,(0,0)], y_limits=[False,(0,0)]):
        
        # make a copy to handle the data and delete nan values
        data = self.df.copy(deep=True)
        data = data.dropna()
        
        # values for axes
        X = data[self.x_label].values
        Y = data[self.y_label].values
        Z = data["chi2"].values - data["chi2"].min()

        # looking for best fit
        mask_min = Z == Z.min()
        xmin_enu, ymin_enu = X[mask_min][0], Y[mask_min][0]

        X = list(X)
        Y = list(Y)
        Z = list(Z)

        # setting the contour features
        num_colors = 12
        viridis = cm.get_cmap('viridis', num_colors)
        bar_1 = mpatches.Patch(color=viridis(range(num_colors))[1], label=r'1 $\sigma$')
        bar_2 = mpatches.Patch(color=viridis(range(num_colors))[4], label=r'2 $\sigma$')
        bar_3 = mpatches.Patch(color=viridis(range(num_colors))[8], label=r'3 $\sigma$')

        # setting the general plot features
        plt.rcParams["figure.figsize"] = (6,4)
        levels = [0,2.3,6.18,11.83]
        plot_labels = {'mzprime' : r'$m_{Z \prime}$', 'm4' : r'$m_4$', 'm5' : r'$m_5$', 'delta' : r'$\Delta$'}
        if not(plot_path):
            plot_path = './fit_' + self.model[0] + 'p' + self.model[2] + '_' + self.HNLtype + '_' + self.experiment + '.jpg'
        if not(title):
            title = r'Fitting for $E_\nu$, ' + self.model + ', ' + self.HNLtype + ', ' + self.experiment
        
        # plot
        plt.tricontourf(X,Y,Z,levels=levels,cmap='viridis')
        plt.tricontour(X,Y,Z,levels=levels,colors='black',linewidths=0.5)
        plt.plot(xmin_enu,ymin_enu,color='orange',marker='*',markersize=12)
        plt.legend(handles=[bar_1, bar_2, bar_3],fontsize=10,loc=leg_loc)
        plt.title(title,fontsize=15)
        plt.xlabel(plot_labels[self.x_label],fontsize=15)
        plt.ylabel(plot_labels[self.y_label],fontsize=15)
        if self.log_interval_x:
            plt.xscale('log')
        if self.log_interval_y:
            plt.yscale('log')
        if x_limits[0]:
            plt.xlim(*x_limits[1])
        if y_limits[0]:
            plt.ylim(*y_limits[1])
        if save:
            plt.savefig(plot_path,dpi=400)
        plt.show()
        plt.clf()











class grid_analysis_couplings:

    def __init__(self,model='3+1',experiment='miniboone_fhc',neval=100000, HNLtype="dirac",x_label='mzprime', x_range=(0.02,10,10),coupling_range=(1e-4,1e-2,10),log_interval_x=True,log_interval_coupling=True,mzprime=None,m4=None,m5=None,delta=None,UD4=UD4_def,UD5=UD5_def,Umu4=Umu4_def,Umu5=Umu5_def,gD=gD_def,epsilon=8e-4,cores=1,output_file=None):
        
        # initialize model parameters
        self.model = model
        self.experiment = experiment
        self.neval = neval
        self.HNLtype = HNLtype
        self.cores = cores
        self.x_label = x_label
        self.df = None
        self.log_interval_x = log_interval_x
        self.log_interval_coupling = log_interval_coupling
        
        if output_file:
            self.output_file = output_file + '.dat'
        else:
            self.output_file = 'coupling_fit_' + model[0] + 'p' + model[2] + '_' + HNLtype + '_' + experiment + '.dat'
        keys_3p1 = ['mzprime','m4']
        keys_3p2 = ['mzprime','m4','m5','delta']
        if log_interval_x:
            interval_func_x = np.geomspace
        else:
            interval_func_x = np.linspace
        if log_interval_coupling:
            interval_func_coupling = np.geomspace
        else:
            interval_func_coupling = np.linspace

        if (model == '3+1'):
            self.y_label = 'Umu4'
            self.params = {key: None for key in keys_3p1}
            self.params['mzprime'] = ff.round_sig(mzprime,4)
            self.params['m4'] = ff.round_sig(m4,4)
            self.params[x_label] = ff.round_sig(interval_func_x(*x_range),4)
            self.params['coupling'] = ff.round_sig(interval_func_coupling(*coupling_range),4)
        elif (model == '3+2'):
            self.y_label = 'vmu5'
            self.params = {key: None for key in keys_3p2}
            self.params['mzprime'] = ff.round_sig(mzprime,4)
            self.params['m4'] = ff.round_sig(m4,4)
            self.params['m5'] = ff.round_sig(m5,4)
            self.params['delta'] = ff.round_sig(delta,4)
            self.params[x_label] = ff.round_sig(interval_func_x(*x_range),4)
            self.params['coupling'] = ff.round_sig(interval_func_coupling(*coupling_range),4)

        # initialize grid parameters
        self.grid_params = {'x_label' : x_label, 'y_label' : self.y_label, 'x_points' : x_range[2], 'y_points' : coupling_range[2]}
        if (model == '3+1'):
            self.cols = ['m4','mzprime','decay_length','vmu4','Nevents','chi2','sum_w_post_smearing']
        elif (model == '3+2'):
            self.cols = ['m4','m5','delta','decay_length','vmu5','Nevents','chi2','mzprime']
        
        # initialize data from experiment
        if self.experiment=='miniboone_fhc':
            self.data_enu = get_data_MB()
            self.back_MC_enu = self.data_enu[1]
            self.D_enu = self.data_enu[0] + self.data_enu[1]
            self.sys_enu = [self.data_enu[2], self.data_enu[3]]
        
        # initialize default coupling parameters for generation of events
        if (model == '3+1'):
            self.couplings_def = {'UD4' : UD4_def, 'gD' : gD_def, 'Umu4' : Umu4_def, 'epsilon' : epsilon_def if epsilon_def else 8e-4}
            self.v4i_def = self.couplings_def['gD'] * self.couplings_def['UD4'] * self.couplings_def['Umu4']
            self.vmu4_def = self.couplings_def['gD'] * self.couplings_def['UD4'] * self.couplings_def['UD4'] * self.couplings_def['Umu4'] / np.sqrt(1-self.couplings_def['Umu4']**2)
        elif (model == '3+2'):
            self.couplings_def = {'UD4' : UD4_def, 'UD5' : UD5_def, 'gD' : gD_def, 'Umu4' : Umu4_def, 'Umu5' : Umu5_def, 'epsilon' : epsilon_def if epsilon_def else 1e-2}
            self.vmu5_def = self.couplings_def['gD'] * self.couplings_def['UD5'] * (self.couplings_def['Umu4']*self.couplings_def['UD4'] + self.couplings_def['Umu5']*self.couplings_def['UD5']) / np.sqrt(1 - self.couplings_def['Umu4']**2 - self.couplings_def['Umu5']**2)
        
        # initialize coupling parameters to be considered
        if (model == '3+1'):
            self.couplings = {'UD4' : UD4 if UD4 else UD4_def, 'gD' : gD if gD else gD_def, 'Umu4' : Umu4 if Umu4 else Umu4_def, 'epsilon' : epsilon if epsilon else 8e-4}
            self.v4i = lambda umu4 : self.couplings['gD'] * self.couplings['UD4'] * umu4
            self.vmu4 = lambda umu4 : self.couplings['gD'] * self.couplings['UD4'] * self.couplings['UD4'] * umu4 / np.sqrt(1-umu4**2)
        elif (model == '3+2'):
            self.couplings = {'UD4' : UD4 if UD4 else UD4_def, 'UD5' : UD5 if UD5 else UD5_def, 'gD' : gD if gD else gD_def, 'Umu4' : Umu4 if Umu4 else Umu4_def, 'Umu5' : Umu5 if Umu5 else Umu5_def, 'epsilon' : epsilon if epsilon else 1e-2}
            self.vmu5 = lambda umu : self.couplings['gD'] * self.couplings['UD5'] * (umu[0]*self.couplings['UD4'] + umu[1]*self.couplings['UD5']) / np.sqrt(1 - umu[0]**2 - umu[1]**2)
        
        self.r_eps = self.couplings['epsilon']/self.couplings_def['epsilon']

    
    def generate_events(self,location='data', **kwargs):
        if self.model == '3+1':
            def chi2_grid_run(k_x):
                if self.x_label == 'm4':
                    m4s = self.params[self.x_label][k_x]
                    mzs = self.params['mzprime']
                elif self.x_label == 'mprime':
                    mzs = self.params[self.x_label][k_x]
                    m4s = self.params['m4']
                
                try:
                    pd.read_parquet(location + '/' + self.experiment + f'/3plus1/m4_{m4s}_mzprime_{mzs}_'+self.HNLtype+'/pandas_df.parquet', engine='pyarrow')
                except:
                    try:
                        gen = GenLauncher(mzprime=mzs, m4=m4s, Umu4=self.couplings_def['Umu4'], UD4=self.couplings_def['UD4'], gD=self.couplings_def['gD'],epsilon=self.couplings_def['epsilon'], neval=self.neval, HNLtype=self.HNLtype, experiment=self.experiment,sparse=True,print_to_float32=True, pandas=False, parquet=True, **kwargs)
                        gen.run(loglevel="ERROR")
                    except:
                        return "Nothing done"

        elif self.model == '3+2':
            def chi2_grid_run(k_x):
                if not(self.params['delta']):
                    if self.x_label == 'm4':
                        m4s = self.params[self.x_label][k_x]
                        mzs = self.params['mzprime']
                        m5s = self.params['m5']
                    elif self.x_label == 'mprime':
                        mzs = self.params[self.x_label][k_x]
                        m4s = self.params['m4']
                        m5s = self.params['m4']
                    elif self.x_label == 'm5':
                        m5s = self.params[self.x_label][k_x]
                        m4s = self.params['m4']
                        mzs = self.params['mzprime']
                else:
                    if self.x_label == 'm4':
                        m4s = self.params[self.x_label][k_x]
                        mzs = self.params['mzprime']
                        m5s = m4s*(self.params['delta'] + 1)
                    elif self.x_label == 'mprime':
                        if self.params['m4']:
                            mzs = self.params[self.x_label][k_x]
                            m4s = self.params['m4']
                            m5s = m4s*(self.params['delta'] + 1)
                        else:
                            mzs = self.params[self.x_label][k_x]
                            m5s = self.params['m5']
                            m4s = m5s/(self.params['delta'] + 1)
                    elif self.x_label == 'm5':
                        m5s = self.params[self.x_label][k_x]
                        m4s = m5s/(self.params['delta'] + 1)
                        mzs = self.params['mzprime']
                    elif self.x_label == 'delta':
                        deltas = self.params[self.x_label][k_x]
                        mzs = self.params['mzprime']
                        if self.params['m4']:
                            m4s = self.params['m4']
                            m5s = m4s*(deltas + 1)
                        else:
                            m5s = self.params['m5']
                            m4s = m5s/(deltas + 1)
                
                try:
                    pd.read_parquet(location + '/' + self.experiment + f'/3plus2/m5_{m5s}_m4_{m4s}_mzprime_{mzs}_'+self.HNLtype+'/pandas_df.parquet', engine='pyarrow')
                except:
                    try:
                        gen = GenLauncher(mzprime=mzs, m5=m5s, m4=m4s, Umu4=self.couplings_def['Umu4'], Umu5=self.couplings_def['Umu5'], UD4=self.couplings_def['UD4'], UD5=self.couplings_def['UD5'], gD=self.couplings_def['gD'],epsilon=self.couplings_def['epsilon'], neval=self.neval, HNLtype=self.HNLtype, experiment=self.experiment,sparse=True,print_to_float32=True, pandas=False, parquet=True, **kwargs)
                        gen.run(loglevel="ERROR")
                    except:
                        return "Nothing done"

        pool3 = Pool(self.cores)
        pool3.map(chi2_grid_run,range(self.grid_params['x_points']))


    def fit_events(self,location='data',location_fit='data_fitting'):
        os.makedirs(location_fit, exist_ok=True)
        if self.model == '3+1':
            data_list = [location_fit+f'/chi2_enu_3p1_coupling_{k}.dat' for k in range(self.grid_params['x_points'])]
            def chi2_grid(k_x):
                bin_e = np.array([0.2, 0.3, 0.375, 0.475, 0.55, 0.675, 0.8, 0.95, 1.1, 1.3, 1.5, 3.])
                umu4 = self.params['coupling']
                n = self.grid_params['y_points']
                if self.x_label == 'm4':
                    m4s = self.params[self.x_label][k_x]
                    mzs = self.params['mzprime']
                elif self.x_label == 'mprime':
                    mzs = self.params[self.x_label][k_x]
                    m4s = self.params['m4']
                
                data = [[m4s,mzs,0,0,0,0,0] for i in range(n)]
                data_error = [[m4s,mzs,0,0,0,-1,0] for i in range(n)]

                try:
                    df = pd.read_parquet(location + '/' + self.experiment + f'/3plus1/m4_{m4s}_mzprime_{mzs}_'+self.HNLtype+'/pandas_df.parquet', engine='pyarrow')
                    l_decay = const.get_decay_rate_in_cm(np.sum(df.w_decay_rate_0))
                    if df.w_event_rate.sum()==0:
                        return data_error
                    df = av2.compute_spectrum(df, EVENT_TYPE='both')
                    df = df[df.reco_w>0]
                    
                    if df.reco_w.sum()==0:
                        return data_error
                
                    for i in range(n):
                        data[i][3] = self.vmu4(umu4[i])
                        if (m4s < mzs):
                            coupling_factor = umu4[i]/self.couplings_def['Umu4'] * self.r_eps
                        else:
                            coupling_factor = umu4[i]/self.couplings_def['Umu4']
                        df_decay = av.select_MB_decay_expo_prob(df,coupling_factor=coupling_factor,l_decay_proper_cm = l_decay)
                        df_decay = df_decay[df_decay.reco_w>0]
                        if (df_decay.reco_w.sum()==0) or (df_decay.w_decay_rate_0.sum()==0):
                            data[i][5] = -1
                            continue
                        sum_w_post_smearing = np.abs(np.sum(df_decay.reco_w.values))
                        hist = np.histogram(df_decay['reco_Enu'], weights=df_decay['reco_w'], bins=bin_e, density = False)
                        norm=np.sum(hist[0])/560.
                        if norm == 0:
                            data[i][5] = -1
                            continue
                        NP_MC = (hist[0])/norm
                        
                        l_decay_proper_cm = l_decay / coupling_factor**2
                        
                        data[i][2] = l_decay_proper_cm
                        data[i][4] = (data[i][3]/ self.vmu4_def * self.r_eps)**2 * sum_w_post_smearing
                        data[i][5] = ff.chi2_MiniBooNE_2020(NP_MC, data[i][4])
                        data[i][6] = sum_w_post_smearing
                    
                    data_enu = pd.DataFrame(data=data,columns=self.cols)
                    data_enu.to_csv(data_list[k_x],sep='\t',index=False)
                    
                    return "Success!"
                    
                except:
                    return "Failed to run events on the grid!"

        elif self.model == '3+2':
            data_list = [location_fit+f'/chi2_enu_3p2_coupling_{k}.dat' for k in range(self.grid_params['x_points'])]
            def chi2_grid(k_x):
                bin_e = np.array([0.2, 0.3, 0.375, 0.475, 0.55, 0.675, 0.8, 0.95, 1.1, 1.3, 1.5, 3.])
                couplings = self.params['coupling']
                n = self.grid_params['y_points']
                
                if not(self.params['delta']):
                    if self.x_label == 'm4':
                        m4s = self.params[self.x_label][k_x]
                        mzs = self.params['mzprime']
                        m5s = self.params['m5']
                    elif self.x_label == 'mprime':
                        mzs = self.params[self.x_label][k_x]
                        m4s = self.params['m4']
                        m5s = self.params['m4']
                    elif self.x_label == 'm5':
                        m5s = self.params[self.x_label][k_x]
                        m4s = self.params['m4']
                        mzs = self.params['mzprime']
                else:
                    if self.x_label == 'm4':
                        m4s = self.params[self.x_label][k_x]
                        mzs = self.params['mzprime']
                        m5s = m4s*(self.params['delta'] + 1)
                    elif self.x_label == 'mprime':
                        if self.params['m4']:
                            mzs = self.params[self.x_label][k_x]
                            m4s = self.params['m4']
                            m5s = m4s*(self.params['delta'] + 1)
                        else:
                            mzs = self.params[self.x_label][k_x]
                            m5s = self.params['m5']
                            m4s = m5s/(self.params['delta'] + 1)
                    elif self.x_label == 'm5':
                        m5s = self.params[self.x_label][k_x]
                        m4s = m5s/(self.params['delta'] + 1)
                        mzs = self.params['mzprime']
                    elif self.x_label == 'delta':
                        deltas = self.params[self.x_label][k_x]
                        mzs = self.params['mzprime']
                        if self.params['m4']:
                            m4s = self.params['m4']
                            m5s = m4s*(deltas + 1)
                        else:
                            m5s = self.params['m5']
                            m4s = m5s/(deltas + 1)
                data_error = [[m4s,m5s,deltas,0,0,0,-1,mzs] for i in range(n)]
                try:
                    df = pd.read_parquet(location + '/' + self.experiment + f'/3plus2/m5_{m5s}_m4_{m4s}_mzprime_{mzs}_'+self.HNLtype+'/pandas_df.parquet', engine='pyarrow')
                    if df.w_event_rate.sum()==0:
                        return data_error
                    decay_l = const.get_decay_rate_in_cm(np.sum(df.w_decay_rate_0))
                    
                    df = av2.compute_spectrum(df, EVENT_TYPE='both')
                    df = df[df.reco_w>0]
                    
                    if df.reco_w.sum()==0:
                        return data_error
                    
                    if (m5s-m4s<mzs):
                        df = av.select_MB_decay_expo_prob(df,coupling_factor=self.r_eps,l_decay_proper_cm=decay_l)
                        decay_l_norm = decay_l / (self.r_eps)**2
                        data = [[m4s[k],m5s[k],deltas,decay_l_norm,0,0,0,mzs] for i in range(n)]
                    else:
                        df = av.select_MB_decay_expo_prob(df,coupling_factor=1,l_decay_proper_cm=decay_l)
                        data = [[m4s[k],m5s[k],deltas,decay_l,0,0,0,mzs] for i in range(n)]
                    df = df[df.reco_w>0]
                    
                    if df.reco_w.sum()==0:
                        return data_error
                    
                    
                    sum_w_post_smearing = np.abs(np.sum(df.reco_w.values))
                    hist = np.histogram(df['reco_Enu'], weights=df['reco_w'], bins=bin_e, density = False)
                    norm=np.sum(hist[0])/550.
                    NP_MC = (hist[0])/norm

                    for i in range(n):
                        data[i][4] = couplings[i]
                        data[i][5] = (couplings[i]/ self.vmu5_def * self.r_eps)**2 * sum_w_post_smearing
                        data[i][6] = ff.chi2_MiniBooNE_2020(NP_MC, data[i][5])
                    
                    data_enu = pd.DataFrame(data=data,columns=self.cols)
                    data_enu.to_csv(data_list[k_x],sep='\t',index=False)
                    
                    return "Success!"
                    
                except:
                    return "Not success!"

        pool4 = Pool(self.cores)
        pool4.map(chi2_grid,range(self.grid_params['x_points']))

        data_enu = pd.DataFrame(data=[],columns=self.cols)
        for k in range(self.grid_params['x_points']):
            try:
                data = pd.read_csv(data_list[k],sep='\t')
                data_enu = pd.concat([data_enu, data], ignore_index=True)
            except:
                continue
        
        data_enu = data_enu[data_enu.chi2>0]
        path_grid = location_fit + '/' + self.output_file
        data_enu.to_csv(path_grid,sep='\t',index=False)
        self.df = data_enu
    

    def plot(self,coupling_factor=1e6,leg_loc = 'lower left',plot_path=False, save=True, title=False, x_limits=[False,(0,0)], y_limits=[False,(0,0)]):
        
        # make a copy to handle the data and delete nan values
        data = self.df.copy(deep=True)
        data = data.dropna()
        
        if self.model == '3+1':
            c_label = 'vmu4'
        elif self.model == '3+2':
            c_label = 'vmu5'

        # values for axes
        X = data[self.x_label].values
        Y = data[c_label].values
        Y = Y*Y*coupling_factor
        Z = data["chi2"].values - data["chi2"].min()

        # looking for best fit
        mask_min = Z == Z.min()
        xmin_enu, ymin_enu = X[mask_min][0], Y[mask_min][0]

        X = list(X)
        Y = list(Y)
        Z = list(Z)

        # setting the contour features
        num_colors = 12
        viridis = cm.get_cmap('viridis', num_colors)
        bar_1 = mpatches.Patch(color=viridis(range(num_colors))[1], label=r'1 $\sigma$')
        bar_2 = mpatches.Patch(color=viridis(range(num_colors))[4], label=r'2 $\sigma$')
        bar_3 = mpatches.Patch(color=viridis(range(num_colors))[8], label=r'3 $\sigma$')

        # setting the general plot features
        plt.rcParams["figure.figsize"] = (6,4)
        levels = [0,2.3,6.18,11.83]
        plot_labels = {'mzprime' : r'$m_{Z \prime}$', 'm4' : r'$m_4$', 'm5' : r'$m_5$', 'delta' : r'$\Delta$'}
        coupling_label = {'3+1': r'$|U_{\mu 4}|^2$', '3+2': r'$|V_{\mu 5}|^2$'}
        if not(plot_path):
            plot_path = './fit_' + self.model[0] + 'p' + self.model[2] + '_' + self.HNLtype + '_' + self.experiment + '.jpg'
        if not(title):
            title = r'Fitting for $E_\nu$, ' + self.model + ', ' + self.HNLtype + ', ' + self.experiment
        
        # plot
        plt.tricontourf(X,Y,Z,levels=levels,cmap='viridis')
        plt.tricontour(X,Y,Z,levels=levels,colors='black',linewidths=0.5)
        plt.plot(xmin_enu,ymin_enu,color='orange',marker='*',markersize=12)
        plt.legend(handles=[bar_1, bar_2, bar_3],fontsize=10,loc=leg_loc)
        plt.title(title,fontsize=15)
        plt.xlabel(plot_labels[self.x_label],fontsize=15)
        plt.ylabel(coupling_label[self.model],fontsize=15)
        if self.log_interval_x:
            plt.xscale('log')
        if self.log_interval_coupling:
            plt.yscale('log')
        if x_limits[0]:
            plt.xlim(*x_limits[1])
        if y_limits[0]:
            plt.ylim(*y_limits[1])
        
        ymin = int(np.ceil(np.log10(np.array(Y).min())))
        ymax = int(np.floor(np.log10(np.array(Y).max())))
        c_fact = int(round(np.log10(coupling_factor)))

        yt = 10.**np.arange(ymin,ymax + 1)
        yl = [(r'$10^{%d}$' % i) for i in range(ymin - c_fact, ymax + 1 - c_fact)]
        plt.yticks(ticks=yt,labels=yl)

        if save:
            plt.savefig(plot_path,dpi=400)
        plt.show()
        plt.clf()
