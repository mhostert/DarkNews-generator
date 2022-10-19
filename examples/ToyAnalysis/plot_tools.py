import os
import numpy as np
from scipy.interpolate import splprep, splev
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import matplotlib.patches as mpatches

from DarkNews import const
from DarkNews import plot_tools as pt

from ToyAnalysis import analysis
from ToyAnalysis import analysis_decay
from ToyAnalysis import fourvec as fv
from ToyAnalysis import toy_logger

import importlib.resources as resources

###########################

def plot_all_rates(df, case_name, Nevents=None, truth_plots=False, title=None, path=None, loc=''):

    toy_logger.info("Plot MiniBooNE signal in {PATH_MB}")
    if path:
        PATH_MB = path
    else:
        PATH_MB = Path(f'plots/{case_name}_miniboone/')

    if not os.path.exists(PATH_MB):
        os.makedirs(PATH_MB)

    # plot titles
    if not title:
        title = case_name

    # get observables at MiniBooNE
    bag_reco_MB = analysis_decay.decay_selection(
                                                analysis.compute_spectrum(df, EVENT_TYPE='both'),
                                                experiment='miniboone',
                                                l_decay_proper_cm=df.attrs['N5_ctau0'])

    batch_plot_signalMB(bag_reco_MB, PATH_MB, BP=case_name, title=title, NEVENTS=Nevents, loc=loc)

    # plot true variables for MiniBooNE
    if truth_plots:
        batch_plot(df, Path(f'{PATH_MB}/truth_level_plots/'), title=title)



def batch_plot_signalMB(obs, PATH, title='Dark News', Nevents= None, loc='', prefix=''):

    if Nevents is not None:
        total_Nevent_MB = Nevents*(1/obs['reco_eff'][0])
    else:
        total_Nevent_MB = obs['reco_w'].sum()
    print(f"MB events: {total_Nevent_MB:.2g}")

    #################### HISTOGRAMS 1D - STACKED ####################################################
    histogram1D_data_stacked(Path(PATH)/f"{prefix}_1D_Enu_data_stacked", obs, r"$E_{\rm \nu}/$GeV", title,
        varplot='reco_Enu', tot_events=total_Nevent_MB, loc=loc)
    histogram1D_data_stacked(Path(PATH)/f"{prefix}_1D_Evis_data_stacked", obs, r"$E_{\rm vis}/$GeV", title,
        varplot='reco_Evis', tot_events=total_Nevent_MB, loc=loc)
    histogram1D_data_stacked(Path(PATH)/f"{prefix}_1D_costheta_data_stacked", obs, r"$\cos\theta$", title,
        varplot='reco_costheta_beam', tot_events=total_Nevent_MB, loc=loc)


def batch_plot_signalMB_bf(obs, PATH, title='Dark News', NEVENTS=1, kde=False, BP = "",loc=''):

    #################### HISTOGRAMS 1D - STACKED ####################################################
    histogram1D_data_stacked(PATH/BP/"1D_Enu_data_stacked", obs, r"$E_{\rm \nu}/$GeV", title,
        varplot='reco_Enu', tot_events=NEVENTS,loc=loc)
    histogram1D_data_stacked(PATH/BP/"1D_Evis_data_stacked", obs, r"$E_{\rm vis}/$GeV", title,
        varplot='reco_Evis', tot_events=NEVENTS,loc=loc)
    histogram1D_data_stacked(PATH/BP/"1D_costheta_data_stacked", obs, r"$\cos\theta$", title,
        varplot='reco_costheta_beam', tot_events=NEVENTS,loc=loc)   


# Function for obtaining the histogram data for the simulation at MiniBooNE
def get_histogram1D(obs, NEVENTS=1, varplot='reco_Evis', get_bins=False,loc='../'):
    
    if varplot=='reco_Enu':
        TMIN, TMAX, nbins, tot_events = 0.2, 1.5, 10, NEVENTS*(obs['reco_eff'][0])
    elif varplot=='reco_Evis':
        TMIN, TMAX, nbins, tot_events = 0.1, 1.25, 10, NEVENTS*(obs['reco_eff'][0])
    elif varplot=='reco_angle':
        TMIN, TMAX, nbins, tot_events = -1.0, 1.0, 10, NEVENTS*(obs['reco_eff'][0])
    else:
        toy_logger.error('That is not a correct variable!')
        return 1

    coherent = (obs['scattering_regime'] == 'coherent')
    pel = (obs['scattering_regime'] == 'p-el')
    
    HC = (obs['helicity'] == 'conserving')
    HF = (obs['helicity'] == 'flipping')

    if varplot=='reco_Evis':
        

        # miniboone nu data for bins
        Enu_binc, _ = np.loadtxt(loc+"aux_data/miniboone_2020/Evis/data_Evis.dat", unpack=True)
        nbins=np.size(Enu_binc)
        Enu_binc *= 1e-3
        binw_enu = 0.05*np.ones((nbins))
        bin_e = np.append(0.1, binw_enu/2.0 + Enu_binc)

        hist_co = np.histogram(obs[varplot][coherent & HC], weights=obs['reco_w'][coherent & HC], bins=bin_e, density = False, range = (TMIN,TMAX) )
        hist_inco = np.histogram(obs[varplot][pel & HC], weights=obs['reco_w'][pel & HC], bins=bin_e, density = False, range = (TMIN,TMAX) )
        
        norm=np.sum(hist_inco[0]+hist_co[0])/tot_events
        
        h_co = hist_co[0]/norm
        h_inco = hist_inco[0]/norm
        h_tot = h_co + h_inco
        h_bins = hist_co[1]
        
        
    elif varplot=='reco_Enu':

        # miniboone nu data for bins
        bin_e = np.loadtxt(loc+"aux_data/miniboone_2020/Enu/bin_edges.dat")
        bin_w = (bin_e[1:] - bin_e[:-1])
        units = 1e3 # from GeV to MeV
        
        hist_co = np.histogram(obs[varplot][coherent & HC], weights=obs['reco_w'][coherent & HC], bins=bin_e, density = False, range = (TMIN,TMAX) )
        hist_inco = np.histogram(obs[varplot][pel & HC], weights=obs['reco_w'][pel & HC], bins=bin_e, density = False, range = (TMIN,TMAX) )
        
        norm = np.sum(hist_co[0]+hist_inco[0])/tot_events*bin_w*units
        
        h_co = hist_co[0]/norm
        h_inco = hist_inco[0]/norm
        h_tot = h_co + h_inco
        h_bins = hist_co[1]
        
            
    elif varplot=='reco_angle':

        # miniboone nu data for bins
        bincost_e = np.linspace(-1,1,21)

        hist_co = np.histogram(np.cos(obs['reco_theta_beam']*np.pi/180)[coherent & HC], weights=obs['reco_w'][coherent & HC], bins=bincost_e, density = False, range = (TMIN,TMAX) )
        hist_inco = np.histogram(np.cos(obs['reco_theta_beam']*np.pi/180)[pel & HC], weights=obs['reco_w'][pel & HC], bins=bincost_e, density = False, range = (TMIN,TMAX) )
        
        norm=np.sum(hist_inco[0]+hist_co[0])/tot_events
        
        h_co = hist_co[0]/norm
        h_inco = hist_inco[0]/norm
        h_tot = h_co + h_inco
        h_bins = hist_co[1]
        

    if get_bins:
        return [h_tot, h_co, h_inco, h_bins]
    else:
        return [h_tot, h_co, h_inco]


def get_data_MB(varplot='reco_Evis',loc='../'):
    
    if varplot=='reco_Evis':
        _, data = np.loadtxt(loc+"aux_data/miniboone_2020/Evis/data_Evis.dat", unpack=True)
        _, bkg = np.loadtxt(loc+"aux_data/miniboone_2020/Evis/bkg_Evis.dat", unpack=True)
        signal = data - bkg
        sys_signal = 0.1
        sys_bkg = 0.1
        
    elif varplot=='reco_Enu':
        # miniboone nu data 2020
        _, data = np.loadtxt(loc+"aux_data/miniboone_2020/Enu/data.dat", unpack=True)
        _, bkg = np.loadtxt(loc+"aux_data/miniboone_2020/Enu/constrained_bkg.dat", unpack=True)
        _, error_low = np.loadtxt(loc+"aux_data/miniboone_2020/Enu/lower_error_bar_constrained_bkg.dat", unpack=True)
        signal = data - bkg
        sys_bkg = (bkg - error_low)/bkg
        sys_signal = 0.1
        bin_e = np.loadtxt(loc+"aux_data/miniboone_2020/Enu/bin_edges.dat")
        bin_w = (bin_e[1:] - bin_e[:-1])
        signal *= bin_w*1e3
        bkg *= bin_w*1e3
            
    elif varplot=='reco_angle':
        _, data = np.loadtxt(loc+"aux_data/miniboone_2020/cos_Theta/data_cosTheta.dat", unpack=True)
        _, bkg = np.loadtxt(loc+"aux_data/miniboone_2020/cos_Theta/bkg_cosTheta.dat", unpack=True)
        signal = data - bkg
        sys_signal = 0.1
        sys_bkg = 0.1
        
    return [signal,bkg,sys_signal,sys_bkg]


# Main plotting function for signal at MiniBooNE (stacked histograms)
def histogram1D_data_stacked(plotname, df, XLABEL, TITLE, varplot='reco_costheta_beam', tot_events  = 1.0, rasterized=True,loc='../'):

    # Masks
    coherent = (df['scattering_regime'] == 'coherent')
    pel = (df['scattering_regime'] == 'p-el')
    HC = (df['helicity'] == 'conserving')
    HF = (df['helicity'] == 'flipping')
   
    # identifiers  
    cases = [coherent & HC, pel & HC, coherent & HF, pel & HF]
    case_names = [r"coherent conserving", r"p-el conserving", r"coherent flipping", r"p-el flipping"]
    case_shorthands = [r"coh HC", r"incoh HC", r"coh HF", r"incoh HF"]
    colors=['dodgerblue','lightblue', 'violet', 'pink']

    nevents = []
    legends = []
    tot_samples = np.size(df['reco_w'])
    for i in range(4):
        this_n_events = int(round(np.sum(df['reco_w'][cases[i]])/np.sum(df['reco_w'])*tot_events))
        nevents.append(this_n_events)
        legends.append(f'{case_shorthands[i]} ({this_n_events} events)')
        
    fsize = 10
    fig = plt.figure()
    ax = fig.add_axes(pt.std_axes_form, rasterized=rasterized)
    ax.patch.set_alpha(0.0)


    #####################
    # MiniBooNE data 
    if varplot=='reco_Evis':

        # miniboone nu data
        bin_c, data_MB_enu_nue = np.genfromtxt(resources.open_text("ToyAnalysis.include.miniboone_2020", "Evis_data.dat"), unpack=True)
        _, data_MB_bkg = np.genfromtxt(resources.open_text("ToyAnalysis.include.miniboone_2020", "Evis_bkg.dat"), unpack=True)
        bin_c *= 1e-3
        bin_w = 0.05*bin_c/bin_c
        bin_e = np.append(0.1, bin_w/2.0 + bin_c)
        units = 1

        data_plot(ax,\
                    bin_c,
                    bin_w, 
                    (data_MB_enu_nue-data_MB_bkg),
                    (np.sqrt(data_MB_enu_nue)), 
                    (np.sqrt(data_MB_enu_nue)))

    elif varplot=='reco_Enu':

        # miniboone nu data 2020
        _, data_MB = np.genfromtxt(resources.open_text("ToyAnalysis.include.miniboone_2020", "Enu_data.dat"), unpack=True)
        _, data_MB_bkg = np.genfromtxt(resources.open_text("ToyAnalysis.include.miniboone_2020", "Enu_constrained_bkg.dat"), unpack=True)
        _, MB_bkg_lower_error_bar = np.genfromtxt(resources.open_text("ToyAnalysis.include.miniboone_2020", "Enu_lower_error_bar_constrained_bkg.dat"), unpack=True)
        bin_e = np.genfromtxt(resources.open_text("ToyAnalysis.include.miniboone_2020", "Enu_bin_edges.dat"))
        
        
        data_MB = data_MB[:-1]
        bin_e = bin_e[:-1]
        data_MB_bkg = data_MB_bkg[:-1]
        MB_bkg_lower_error_bar = MB_bkg_lower_error_bar[:-1]
        bin_w = (bin_e[1:] - bin_e[:-1])
        bin_c = bin_e[:-1] + bin_w/2

        units = 1e3*bin_w # from GeV to MeV

        data_MB_enu_nue = (data_MB - data_MB_bkg)*units
        error_bar = np.sqrt( ((data_MB_bkg - MB_bkg_lower_error_bar)*units)**2
                                + np.sqrt(data_MB**2*units) )

        data_plot(ax,\
                    bin_c,
                    bin_w, 
                    data_MB_enu_nue/units,
                    error_bar/units, 
                    error_bar/units)


    elif varplot=='reco_costheta_beam':

        # miniboone nu data
        bin_c, data_MB_cost_nue = np.genfromtxt(resources.open_text("ToyAnalysis.include.miniboone_2020", "cosTheta_data.dat"), unpack=True)
        _, data_MB_bkg = np.genfromtxt(resources.open_text("ToyAnalysis.include.miniboone_2020", "cosTheta_bkg.dat"), unpack=True)
        bin_w = np.ones(len(bin_c))*0.1
        bin_e = np.linspace(-1,1,21)
        units = 1

        data_plot(ax,
                bin_c,
                bin_w, 
                (data_MB_cost_nue-data_MB_bkg),
                np.sqrt(data_MB_cost_nue), 
                np.sqrt(data_MB_cost_nue))


    df['reco_w'] = df['reco_w']/np.sum(df['reco_w'])*tot_events

    hists = []
    htotal = np.zeros(len(bin_w))
    handles =[]
    for i in range(4):
        # if nevents[i] > 1e-3*tot_events:
        case = cases[i]
        h, bins =  np.histogram(df[varplot][case], weights=df['reco_w'][case], bins=bin_e)
        h /= units
        ax.bar( bins[:-1], h, bottom=htotal, width=bin_w, label=legends[i],
                ec=None, fc=colors[i], alpha=0.8, align='edge', lw = 0.0, rasterized=rasterized)    
        hists.append(h)
        htotal += h
        ax.step(np.append(bins[:-1],10e10), 
                np.append(htotal, 0.0), 
                where='post',
                c='black', lw = 0.5,rasterized=rasterized)
        handles.append(mpatches.Patch(facecolor=colors[i], edgecolor='black', lw=0.5, label=legends[i]))


    ax.set_title(TITLE, fontsize=0.8*fsize)
    # ax.legend(frameon=False, loc='best')
    ax.legend(handles=handles, frameon=False, loc='best')
    ax.set_xlabel(XLABEL,fontsize=fsize)
    ax.set_xlim(np.min(bin_e),np.max(bin_e))

    if varplot=='reco_Enu':
        ax.set_ylim(0,ax.get_ylim()[1]*1.1)
        ax.set_ylabel(r"Excess events/MeV",fontsize=fsize)
    else:
        ax.set_ylim(-20,ax.get_ylim()[1]*1.1)
        ax.set_ylabel(r"Excess events",fontsize=fsize)
    pt.std_savefig(fig, plotname, dpi=400)

def errorband_plot(ax, X, BINW, DATA, ERRORLOW, ERRORUP, band=False, **kwargs):
    # ax.step(X, DATA, where='mid', **kwargs)
    ax.fill_between(X, DATA-ERRORLOW, DATA+ERRORUP, step='mid', alpha=0.3, **kwargs)

def data_plot(ax, X, BINW, DATA, ERRORLOW, ERRORUP, band=False, **kwargs):
    ax.errorbar(X, DATA, yerr= np.array([ERRORLOW,ERRORUP]), xerr = BINW/2.0, \
                            marker="o", markeredgewidth=0.5, capsize=1.0,markerfacecolor="black",\
                            markeredgecolor="black",ms=2, color='black', lw = 0.0, elinewidth=0.8, zorder=10, **kwargs)



def histogram1D(plotname, obs, w, TMIN, TMAX,  XLABEL, TITLE, nbins, regime=None, colors=None, legends=None, rasterized = True):

    fsize = 10
    fig = plt.figure()
    ax = fig.add_axes(pt.std_axes_form, rasterized=rasterized)
    ax.patch.set_alpha(0.0)

    # normalize
    w = w/np.sum(w)
    # colors = 
    nregimes = len(regime)
    bin_e = np.linspace(TMIN,TMAX, nbins+1, endpoint=True)
    bin_w = (bin_e[1:] - bin_e[:-1])
    if regime and not legends:
        legends = [f'case {i}' for i in range(nregimes)]
    if regime and not colors:
        color = cm.rainbow(np.linspace(0, 1, n))
        colors = [c for c in color]

    if regime:
        htotal = np.zeros((nbins))
        nregimes = len(regime)
        for i in range(np.shape(regime)[0]):
            case = regime[i]
            h, bins =  np.histogram(obs[case], weights=w[case], bins=bin_e)

            ax.bar( bins[:-1], h, bottom=htotal, label=legends[i], width=bin_w,
                    ec=None, facecolor=colors[i], alpha=0.8, align='edge', lw = 0.0, rasterized=rasterized) 
            ax.step(np.append(bins[:-1],10e10), 
                    np.append(htotal, 0.0), 
                    where='post',
                    c='black', lw = 0.5,
                    rasterized=rasterized)
            htotal += h
    else:
        h, bins =  np.histogram(obs, weights=w, bins=nbins, range = (TMIN,TMAX))
        ax.bar( bins[:-1], h, width=bin_w,
                    ec=None, fc='indigo', alpha=0.8, align='edge', lw = 0.0, rasterized=rasterized) 
            

    ax.set_title(TITLE, fontsize=0.8*fsize)
    ax.legend(frameon=False, loc='best')
    ax.set_xlabel(XLABEL,fontsize=fsize)
    ax.set_ylabel(r"PDF",fontsize=fsize)

    ax.set_xlim(TMIN,TMAX)
    ax.set_ylim(0.0,ax.get_ylim()[1]*1.1)
    pt.std_savefig(fig, plotname, dpi=400)
    plt.close()


def histogram2D(plotname, obsx, obsy, w,  xrange=None, yrange=None,  xlabel='x',  ylabel='y', title="Dark News", nbins=20, logx=False, logy=False):
    
    fsize = 11
    
    fig, ax = pt.std_fig(ax_form = [0.15,0.15,0.78,0.74])

    if logx:
        obsx = np.log10(obsx)
    if logy:
        obsy = np.log10(obsy)


    if not xrange:
        xrange = [np.min(obsx),np.max(obsx)]
    if not yrange:
        yrange = [np.min(obsy),np.max(obsy)]

    bar = ax.hist2d(obsx, obsy, bins=nbins, weights=w, range=[xrange,yrange],cmap="Blues",density=True)

    ax.set_title(title, fontsize=fsize)
    cbar_R = fig.colorbar(bar[3],ax=ax)
    cbar_R.ax.set_ylabel(r'a.u.', rotation=90)

    ax.set_xlabel(xlabel,fontsize=fsize)
    ax.set_ylabel(ylabel,fontsize=fsize)
    pt.std_savefig(fig, plotname, dpi=400)
    plt.close()



def batch_plot(df, PATH, title='Dark News'):
    
    # regimes
    coherent = (df['scattering_regime'] == 'coherent')
    pel = (df['scattering_regime'] == 'p-el')
    HC = (df['helicity'] == 'conserving')
    HF = (df['helicity'] == 'flipping')
    cases = [coherent & HC, pel & HC, coherent & HF, pel & HF]
    case_names = [r"coherent conserving", r"p-el conserving", r"coherent flipping", r"p-el flipping"]
    case_shorthands = [r"coh HC", r"incoh HC", r"coh HF", r"incoh HF"]
    colors=['dodgerblue','lightblue', 'violet', 'pink']
    regimes = cases
    args = {'regime': cases, 'colors': colors, 'legends': case_shorthands, 'rasterized': True, }

    if not os.path.exists(PATH):
        os.mkdir(PATH)


    # some useful definitions for four momenta
    for i in range(4):
        df['P_decay_ellell',i] = df['P_decay_ell_minus',f'{i}']+df['P_decay_ell_plus',f'{i}']
            
    # weights
    w     = df['w_event_rate','']
    w_pel = df['w_event_rate',''][pel]
    w_coh = df['w_event_rate',''][coherent]

    # variables
    df['E_N']   = df['P_decay_N_parent','0']
    df['E_Z']   = df['P_decay_ell_minus','0'] + df['P_decay_ell_plus','0']
    df['E_lp']  = df['P_decay_ell_plus','0']
    df['E_lm']  = df['P_decay_ell_minus','0']
    df['E_tot'] = df['E_lm'] + df['E_lp']
    df['E_asy'] = (df['E_lp'] - df['E_lm'])/(df['E_lp'] + df['E_lm'])
    df['E_Had'] = df['P_recoil','0']

    df['M_had'] = fv.df_inv_mass(df['P_recoil'], df['P_recoil'])
    df['Q2'] = -(2*df['M_had']**2-2*df['E_Had']*df['M_had'])
    
    df['costheta_N']   = fv.df_cos_azimuthal(df['P_decay_N_parent']) 
    df['costheta_nu']  = fv.df_cos_azimuthal(df['P_decay_N_daughter']) 
    df['costheta_Had'] = fv.df_cos_azimuthal(df['P_recoil']) 
    df['inv_mass']     = fv.df_inv_mass(df['P_decay_ellell'], df['P_decay_ellell'])
    
    df['costheta_sum'] = fv.df_cos_azimuthal(df['P_decay_ellell'])
    df['costheta_lp'] = fv.df_cos_azimuthal(df['P_decay_ell_plus'])
    df['costheta_lm'] = fv.df_cos_azimuthal(df['P_decay_ell_minus'])

    df['costheta_sum_had'] = fv.df_cos_opening_angle(df['P_decay_ellell'], df['P_recoil'])
    df['theta_sum_had'] = np.arccos(df['costheta_sum_had'])*180/np.pi
    
    df['theta_sum'] = np.arccos(df['costheta_sum'])*180/np.pi
    df['theta_lp'] = np.arccos(df['costheta_lp'])*180/np.pi
    df['theta_lm'] = np.arccos(df['costheta_lm'])*180/np.pi
    df['theta_nu'] = np.arccos(df['costheta_nu'])*180/np.pi

    df['Delta_costheta'] = fv.df_cos_opening_angle(df['P_decay_ell_minus'],df['P_decay_ell_plus'])
    df['Delta_theta'] = np.arccos(df['Delta_costheta'])*180/np.pi

    df['theta_proton'] = np.arccos(df['costheta_Had'][pel])*180/np.pi
    df['theta_nucleus'] = np.arccos(df['costheta_Had'][coherent])*180/np.pi
    
    df['T_proton'] = (df['E_Had'] - df['M_had'])[pel]
    df['T_nucleus'] = (df['E_Had'] - df['M_had'])[coherent]
    
    minus_lead = (df['P_decay_ell_minus','0'] >= df['P_decay_ell_plus','0'])
    plus_lead  = (df['P_decay_ell_minus','0'] < df['P_decay_ell_plus','0'])

    df['E_subleading'] = np.minimum(df['P_decay_ell_minus','0'], df['P_decay_ell_plus','0'])
    df['E_leading'] = np.maximum(df['P_decay_ell_minus','0'],df['P_decay_ell_plus','0'])

    df['theta_subleading'] = df['theta_lp']*plus_lead + df['theta_lm']*minus_lead
    df['theta_leading'] = df['theta_lp']*(~plus_lead) + df['theta_lm']*(~minus_lead)

    # CCQE neutrino energy
    df['E_nu_reco'] = const.m_proton * (df['P_decay_ell_plus','0'] + df['P_decay_ell_minus','0']) / ( const.m_proton - (df['P_decay_ell_plus','0'] + df['P_decay_ell_minus','0'])*(1.0 - (df['costheta_lm']*df['P_decay_ell_minus','0'] + df['costheta_lp'] * df['P_decay_ell_plus','0'])/(df['P_decay_ell_plus','0'] + df['P_decay_ell_minus','0'])  ))

  ###################### HISTOGRAM 2D ##################################################
    n2D = 40
    args_2d = {"title": title, "nbins": n2D}
    
    histogram2D(PATH+"/2D_EN_Etot.pdf", df['E_N'], df['E_tot'], w,
                                xrange=[0.0, 2.0],
                                yrange=[0.0, 2.0],
                                xlabel=r"$E_{N}$ (GeV)", 
                                ylabel=r"$E_{\ell^-}+E_{\ell^+}$ (GeV)",
                                **args_2d)

    histogram2D(PATH+"/2D_Ep_Em.pdf", df['E_lm'], df['E_lp'],w,
                                xrange=[0.0, 2.0],
                                yrange=[0.0, 2.0],
                                xlabel=r"$E_{\ell^-}$ (GeV)", 
                                ylabel=r"$E_{\ell^+}$ (GeV)",
                                **args_2d)

    histogram2D(PATH+"/2D_dtheta_Etot.pdf", df['Delta_theta'], df['E_tot'], w, \
                                              xrange=[0.0, 90],
                                              yrange=[0.0, 2.0],
                                              xlabel=r"$\Delta \theta_{\ell \ell}$ ($^\circ$)", 
                                              ylabel=r"$E_{\ell^+}+E_{\ell^-}$ (GeV)",
                                              **args_2d)

    histogram2D(PATH+"/2D_Easyabs_Etot.pdf", np.abs(df['E_asy']), df['Delta_costheta'], w,
                                xrange=[0.0, 1.0],
                                yrange=[0.0, 90.0],
                                xlabel=r"$|E_{\rm asy}|$", 
                                ylabel=r"$\Delta \theta_{\ell \ell}$ ($^\circ$)",
                                **args_2d)

    histogram2D(PATH+"/2D_Easyabs_Etot.pdf", np.abs(df['E_asy']), df['E_tot'], w,
                                xrange=[0.0, 1.0],
                                yrange=[0.0, 2.0],
                                xlabel=r"$|E_{\rm asy}|$", 
                                ylabel=r"$E_{\ell^+}+E_{\ell^-}$ (GeV)",
                                **args_2d)

    histogram2D(PATH+"/2D_Easyabs_Etot.pdf", np.abs(df['E_asy']), df['E_tot'], w,
                                xrange=[0.0, 1.0],
                                yrange=[0.0, 2.0],
                                xlabel=r"$|E_{\rm asy}|$", 
                                ylabel=r"$E_{\ell^+}+E_{\ell^-}$ (GeV)",
                                **args_2d)

    histogram2D(PATH+"/2D_Ehad_Etot.pdf", df['T_proton'][pel]*1e3, df['E_tot'][pel], w_pel,
                                xrange=[0.0, 1000],
                                yrange=[0.0, 2.0],
                                xlabel=r"$T_{\rm proton}$ (MeV)", 
                                ylabel=r'$E_{\ell^+} + E_{\ell^-}$ (GeV)', 
                                title=title +' proton-elastic only', nbins=n2D)


    histogram2D(PATH+"/2D_thetaLead_dtheta.pdf", df['theta_subleading'], df['theta_leading'], w,
                                                xrange=[0.0, 40.0],
                                                yrange=[0.0, 40.0],
                                                xlabel=r"$\theta_{\nu_\mu \ell_{\rm lead}}$ ($^\circ$)", 
                                                ylabel=r'$\Delta \theta$ ($^\circ$)', 
                                                **args_2d)

    #################### HISTOGRAMS 1D ####################################################    
    # momentum exchange
    histogram1D(PATH+"/1D_Q.pdf", np.sqrt(df['Q2']), w, 0.0, 1., r"$Q/$GeV", title, 10, **args)
    histogram1D(PATH+"/1D_Q2.pdf", df['Q2'], w, 0.0, 1.5, r"$Q^2/$GeV$^2$", title, 10, **args)
    
    histogram1D(PATH+"/1D_T_proton.pdf", df['T_proton'][pel]*1e3, w_pel, 0.0, 500.0, r"$T_{\rm p^+}$ (MeV)", 'el proton only', 50, **args)
    histogram1D(PATH+"/1D_theta_proton.pdf", df['theta_proton'][pel], w_pel, 0.0, 180, r"$\theta_{p^+}$ ($^\circ$)", 'el proton only', 50, **args)
    histogram1D(PATH+"/1D_T_nucleus.pdf", df['T_nucleus'][coherent]*1e3, w_coh, 0.0, 20, r"$T_{\rm Nucleus}$ (MeV)", 'coh nucleus only', 50, **args)
    histogram1D(PATH+"/1D_theta_nucleus.pdf", df['theta_nucleus'][coherent], w_coh, 0.0, 180, r"$\theta_{\rm Nucleus}$ ($^\circ$)", 'coh nucleus only', 50, **args)

    # energies
    histogram1D(PATH+"/1D_E_lp.pdf", df['E_lp'], w, 0.0, 2.0, r"$E_{\ell^+}$ GeV", title, 100, **args)
    histogram1D(PATH+"/1D_E_lm.pdf", df['E_lm'], w, 0.0, 2.0, r"$E_{\ell^-}$ GeV", title, 100, **args)
    histogram1D(PATH+"/1D_E_tot.pdf", df['E_tot'], w, 0.0, 2.0, r"$E_{\ell^-}+E_{\ell^+}$ GeV", title, 100, **args)

    histogram1D(PATH+"/1D_E_nu_truth.pdf", df['P_projectile','0'], w, 0.0, 2.0, r"$E_\nu^{\rm truth}/$GeV", title, 20, **args)
    histogram1D(PATH+"/1D_E_nu_QEreco.pdf", df['E_nu_reco'], w, 0.0, 2.0, r"$E_\nu^{\rm QE-reco}/$GeV", title, 20, **args)
    
    histogram1D(PATH+"/1D_E_N.pdf", df['E_N'], w, 0.0, 2.0, r"$E_N/$GeV", title, 20, **args)

    histogram1D(PATH+"/1D_E_leading.pdf", df['E_leading'], w, 0.0, 2.0, r"$E_{\rm leading}$ GeV", title, 100, **args)
    histogram1D(PATH+"/1D_E_subleading.pdf", df['E_subleading'], w, 0.0, 2.0, r"$E_{\rm subleading}$ GeV", title, 100, **args)
    
    # angles
    histogram1D(PATH+"/1D_costN.pdf", df['costheta_N'], w, -1.0, 1.0, r"$\cos(\theta_{\nu_\mu N})$", title, 20, **args)
    
    histogram1D(PATH+"/1D_cost_sum.pdf", df['costheta_sum'], w, -1.0, 1.0, r"$\cos(\theta_{(ee)\nu_\mu})$", title, 20, **args)
    histogram1D(PATH+"/1D_cost_sum_had.pdf", df['costheta_sum_had'], w, -1.0, 1.0, r"$\cos(\theta_{(ee) {\rm hadron}})$", title, 20, **args)
    
    histogram1D(PATH+"/1D_cost_nu.pdf", df['costheta_nu'], w, -1.0, 1.0, r"$\cos(\theta_{\nu_\mu \nu_{\rm out}})$", title, 40, **args)
    histogram1D(PATH+"/1D_theta_nu.pdf", df['theta_nu'], w, 0.0, 180.0, r"$\theta_{\nu_\mu \nu_{\rm out}}$", title, 40, **args)

    histogram1D(PATH+"/1D_cost_lp.pdf", df['costheta_lp'],  w, -1.0, 1.0, r"$\cos(\theta_{\nu_\mu \ell^+})$", title, 40, **args)
    histogram1D(PATH+"/1D_cost_lm.pdf", df['costheta_lm'], w, -1.0, 1.0, r"$\cos(\theta_{\nu_\mu \ell^-})$", title, 40, **args)

    histogram1D(PATH+"/1D_theta_lp.pdf", df['theta_lp'], w, 0.0, 180.0, r"$\theta_{\nu_\mu \ell^+}$", title, 40, **args)
    histogram1D(PATH+"/1D_theta_lm.pdf", df['theta_lm'], w, 0.0, 180.0, r"$\theta_{\nu_\mu \ell^-}$", title, 40, **args)

    histogram1D(PATH+"/1D_theta_lead.pdf", df['theta_leading'], w, 0.0, 180.0, r"$\theta_{\nu_\mu \ell_{\rm lead}}$ ($^\circ$)", title, 40, **args)
    histogram1D(PATH+"/1D_theta_sublead.pdf", df['theta_subleading'], w, 0.0, 180.0, r"$\theta_{\nu_\mu \ell_{\rm sublead}}$ ($^\circ$)", title, 40, **args)

    histogram1D(PATH+"/1D_deltacos.pdf", df['Delta_costheta'], w,  -1.0, 1.0, r"$\cos(\theta_{\ell^+ \ell^-})$", title, 40, **args)
    histogram1D(PATH+"/1D_deltatheta.pdf", df['Delta_theta'], w, 0, 180.0, r"$\theta_{\ell^+ \ell^-}$", title, 40, **args)

    # highe level vars
    histogram1D(PATH+"/1D_invmass.pdf", df['inv_mass'], w, 0.0, np.max(df['inv_mass']), r"$m_{\ell^+ \ell^-}$ [GeV]", title, 50, **args)

    histogram1D(PATH+"/1D_asym.pdf", df['E_asy'], w, -1.0, 1.0, r"$(E_{\ell^+}-E_{\ell^-})$/($E_{\ell^+}+E_{\ell^-}$)", title, 20, **args)
    histogram1D(PATH+"/1D_asym_abs.pdf", np.abs(df['E_asy']), w, 0.0, 1.0, r"$|E_{\ell^+}-E_{\ell^-}|$/($E_{\ell^+}+E_{\ell^-}$)", title, 20, **args)



def plot_closed_region(points, logx=False, logy=False):
    x,y = points
    if logy:
        if (y==0).any():
            raise ValueError("y values cannot contain any zeros in log mode.")
        sy = np.sign(y)
        ssy = ((np.abs(y)<1)*(-1) + (np.abs(y)>1)*(1))
        y  = ssy*np.log(y*sy)
    if logx:
        if (x==0).any():
            raise ValueError("x values cannot contain any zeros in log mode.")
        sx  = np.sign(x)
        ssx = ((x<1)*(-1) + (x>1)*(1))
        x  = ssx*np.log(x*sx)

    points = np.array([x,y]).T

    points_s     = (points - points.mean(0))
    angles       = np.angle((points_s[:,0] + 1j*points_s[:,1]))
    points_sort  = points_s[angles.argsort()]
    points_sort += points.mean(0)

    tck, u = splprep(points_sort.T, u=None, s=0.0, per=0, k=1)
    u_new = np.linspace(u.min(), u.max(), len(points[:,0]))
    x_new, y_new = splev(u_new, tck, der=0)
    
    if logx:
        x_new = sx*np.exp(ssx*x_new) 
    if logy:
        y_new = sy*np.exp(ssy*y_new) 

    return x_new, y_new