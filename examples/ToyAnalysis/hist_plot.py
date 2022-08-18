import numpy as np
import scipy 
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import rc, rcParams
from matplotlib.pyplot import *
from matplotlib.pyplot import cm
from collections import defaultdict
from functools import partial
import seaborn as sns
import matplotlib.patches as mpatches


from DarkNews import *
from . import analysis
#from ToyAnalysis import fourvec as fv
#from ToyAnalysis import toy_logger

fsize=11
rcparams={'axes.labelsize':fsize,'xtick.labelsize':fsize,'ytick.labelsize':fsize,\
                'figure.figsize':(1.2*3.7,1.3*2.3617)   }
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}\usepackage{amssymb}'
rc('text', usetex=True)
rc('font',**{'family':'serif', 'serif': ['Computer Modern Roman']})
rcParams.update(rcparams)
matplotlib.rcParams['hatch.linewidth'] = 0.3

axes_form  =[0.16,0.16,0.81,0.76]
def std_fig(ax_form=axes_form, rasterized=False):
    fig = plt.figure()
    ax = fig.add_axes(ax_form, rasterized=rasterized)
    ax.patch.set_alpha(0.0)
    return fig,ax

def get_hist(ax):
    n,bins = [],[]
    for rect in ax.patches:
        ((x0, y0), (x1, y1)) = rect.get_bbox().get_points()
        n.append(y1-y0)
        bins.append(x0) # left edge of each bin
    bins.append(x1) # also get right edge of last bin

    return n,bins

def plot_all_rates(df, case_name, Nevents=550, truth_plots=False, title=None, plot_muB = True, path='None',loc=''):
    
    toy_logger.info("Plot MiniBooNE signal in {PATH_MB}")
    if path=='None':
    	PATH_MB = f'plots/exp_rates/{case_name}_miniboone/'
    else:
    	PATH_MB = path
    	
    if not os.path.exists(PATH_MB):
        os.makedirs(PATH_MB)

    # plot titles
    if not title:
        title = case_name

    # get observables at MiniBooNE
    bag_reco_MB = analysis.compute_MB_spectrum(df, EVENT_TYPE='both')
    total_Nevent_MB = Nevents*(1/bag_reco_MB['reco_eff'][0])
    batch_plot_signalMB(bag_reco_MB, PATH_MB, BP=case_name, title=title, NEVENTS=total_Nevent_MB,loc=loc)

    # plot event rates at MuBooNE weighted appropriately for POT and efficiencies
    if plot_muB:
        PATH_muB = f'plots/exp_rates/{case_name}_microboone/'
        if not os.path.exists(PATH_muB):
            os.makedirs(PATH_muB)

        bag_reco_muB = analysis.compute_muB_spectrum(df, EVENT_TYPE='ovl', BP=case_name)
        # POT * n_targets(CH2)
        N_MB = 18.75e20 * 818.0/(8*const.m_proton + 6*const.m_neutron)
        N_muB = 12.25e20 * 85/(18*const.m_proton + 22*const.m_neutron)
        predicted_muB_events = (np.sum(bag_reco_muB['reco_w'])/np.sum(bag_reco_MB['reco_w'])) * N_muB/N_MB * (desired_MB_events)
        total_Nevent_muB = predicted_muB_events/bag_reco_muB['reco_eff'][0]
        batch_plot_signalmuB(bag_reco_muB, PATH_muB, title=title,NEVENTS=total_Nevent_muB)
    
    # plot true variables for MiniBooNE
    if truth_plots:
        batch_plot(df, f'{PATH_MB}/truth_level_plots/', title=title)



def batch_plot_signalMB(obs, PATH, title='Dark News', NEVENTS=1, kde=False, BP = "",loc=''):

    #################### HISTOGRAMS 1D - STACKED ####################################################
    histogram1D_data_stacked(PATH+"/"+BP+"1D_Enu_data_stacked", obs, r"$E_{\rm \nu}/$GeV", title,
        varplot='reco_Enu', tot_events=NEVENTS*(obs['reco_eff'][0]),loc=loc)
    histogram1D_data_stacked(PATH+"/"+BP+"1D_Evis_data_stacked", obs, r"$E_{\rm vis}/$GeV", title,
        varplot='reco_Evis', tot_events=NEVENTS*(obs['reco_eff'][0]),loc=loc)
    histogram1D_data_stacked(PATH+"/"+BP+"1D_costheta_data_stacked", obs, r"$\cos\theta$", title,
        varplot='reco_costheta_beam', tot_events=NEVENTS*(obs['reco_eff'][0]),loc=loc)

    ###################### HISTOGRAM 2D ##################################################
    
    n2D = 20
    # histogram2D(PATH+"/"+BP+"2D_reco_Evis_ctheta.pdf", [obs['reco_Evis'], obs['reco_w']],\
    #                                       [obs['reco_costheta_beam'],obs['reco_w']],\
    #                                       0.0, 2.0,\
                                            # -1.0,1.0,\
                                            # r"$E_{\rm vis}$ (GeV)", r'$\cos\theta$', title, n2D)
    

def batch_plot_signalMB_bf(obs, PATH, title='Dark News', NEVENTS=1, kde=False, BP = "",loc=''):

    #################### HISTOGRAMS 1D - STACKED ####################################################
    histogram1D_data_stacked(PATH+"/"+BP+"1D_Enu_data_stacked", obs, r"$E_{\rm \nu}/$GeV", title,
        varplot='reco_Enu', tot_events=NEVENTS,loc=loc)
    histogram1D_data_stacked(PATH+"/"+BP+"1D_Evis_data_stacked", obs, r"$E_{\rm vis}/$GeV", title,
        varplot='reco_Evis', tot_events=NEVENTS,loc=loc)
    histogram1D_data_stacked(PATH+"/"+BP+"1D_costheta_data_stacked", obs, r"$\cos\theta$", title,
        varplot='reco_costheta_beam', tot_events=NEVENTS,loc=loc)   


def batch_plot_signalmuB(obs, PATH, title='Dark News', NEVENTS=1, kde=False, BP = ""):
    
    #################### HISTOGRAMS 1D - STACKED ####################################################
    histogram1D_data_muB(PATH+"/"+BP+"1D_Enu_data_stacked", obs, r"$E_{\rm \nu}/$GeV", title,
        varplot='reco_Enu', tot_events=NEVENTS, kde=kde)
    histogram1D_data_muB(PATH+"/"+BP+"1D_Evis_data_stacked", obs, r"$E_{\rm vis}/$GeV", title,
        varplot='reco_Evis', tot_events=NEVENTS, kde=kde)
    histogram1D_data_muB(PATH+"/"+BP+"1D_costheta_data_stacked", obs, r"$\cos\theta$", title,
        varplot='reco_angle', tot_events=NEVENTS, kde=kde)


# Function for obtaining the histogram data for the simulation at MiniBooNE
def get_histogram1D(obs, NEVENTS=1, varplot='reco_Evis', get_bins=False,loc=''):
    
    if varplot=='reco_Enu':
        TMIN, TMAX, nbins, tot_events = 0.2, 1.5, 10, NEVENTS*(obs['reco_eff'][0])
    elif varplot=='reco_Evis':
        TMIN, TMAX, nbins, tot_events = 0.1, 1.25, 10, NEVENTS*(obs['reco_eff'][0])
    elif varplot=='reco_angle':
        TMIN, TMAX, nbins, tot_events = -1.0, 1.0, 10, NEVENTS*(obs['reco_eff'][0])
    else:
        toy_logger.error('That is not a correct variable!')
        return 1
    
    
    fsize = 10

    coherent = (obs['scattering_regime'] == 'coherent')
    pel = (obs['scattering_regime'] == 'p-el')
    
    HC = (obs['helicity'] == 'conserving')
    HF = (obs['helicity'] == 'flipping')

    if varplot=='reco_Evis':
        

        # miniboone nu data for bins
        Enu_binc, _ = np.loadtxt(loc+"digitized/miniboone_2020/Evis/data_Evis.dat", unpack=True)
        nbins=np.size(Enu_binc)
        Enu_binc *= 1e-3
        binw_enu = 0.05*np.ones((nbins))
        bin_e = np.append(0.1, binw_enu/2.0 + Enu_binc)

        hist_co = np.histogram(obs[varplot][coherent & HC], weights=np.abs(obs['reco_w'][coherent & HC]), bins=bin_e, density = False, range = (TMIN,TMAX) )
        hist_inco = np.histogram(obs[varplot][pel & HC], weights=np.abs(obs['reco_w'][pel & HC]), bins=bin_e, density = False, range = (TMIN,TMAX) )
        
        norm=np.sum(hist_inco[0]+hist_co[0])/tot_events
        
        if np.sum(hist_inco[0]+hist_co[0])!=0:
            h_co = hist_co[0]/norm
            h_inco = hist_inco[0]/norm
            h_tot = h_co + h_inco
            h_bins = hist_co[1]
        else:
            h_co = hist_co[0]
            h_inco = hist_inco[0]
            h_tot = h_co + h_inco
            h_bins = hist_co[1]
        
        
    elif varplot=='reco_Enu':

        # miniboone nu data for bins
        bin_e = np.loadtxt(loc+"digitized/miniboone_2020/Enu/bin_edges.dat")
        bin_w = (bin_e[1:] - bin_e[:-1])
        units = 1e3 # from GeV to MeV
        
        hist_co = np.histogram(obs[varplot][coherent & HC], weights=np.abs(obs['reco_w'][coherent & HC]), bins=bin_e, density = False, range = (TMIN,TMAX) )
        hist_inco = np.histogram(obs[varplot][pel & HC], weights=np.abs(obs['reco_w'][pel & HC]), bins=bin_e, density = False, range = (TMIN,TMAX) )
        
        norm = np.sum(hist_co[0]+hist_inco[0])/tot_events*bin_w*units
        
        if np.sum(hist_inco[0]+hist_co[0])!=0:
            h_co = hist_co[0]/norm
            h_inco = hist_inco[0]/norm
            h_tot = h_co + h_inco
            h_bins = hist_co[1]
        else:
            h_co = hist_co[0]
            h_inco = hist_inco[0]
            h_tot = h_co + h_inco
            h_bins = hist_co[1]
        
            
    elif varplot=='reco_angle':

        # miniboone nu data for bins
        bincost_e = np.linspace(-1,1,21)

        hist_co = np.histogram(np.cos(obs['reco_theta_beam']*np.pi/180)[coherent & HC], weights=np.abs(obs['reco_w'][coherent & HC]), bins=bincost_e, density = False, range = (TMIN,TMAX) )
        hist_inco = np.histogram(np.cos(obs['reco_theta_beam']*np.pi/180)[pel & HC], weights=np.abs(obs['reco_w'][pel & HC]), bins=bincost_e, density = False, range = (TMIN,TMAX) )
        
        norm=np.sum(hist_inco[0]+hist_co[0])/tot_events
        
        if np.sum(hist_inco[0]+hist_co[0])!=0:
            h_co = hist_co[0]/norm
            h_inco = hist_inco[0]/norm
            h_tot = h_co + h_inco
            h_bins = hist_co[1]
        else:
            h_co = hist_co[0]
            h_inco = hist_inco[0]
            h_tot = h_co + h_inco
            h_bins = hist_co[1]
        

    if get_bins:
        return [h_tot, h_co, h_inco, h_bins]
    else:
        return [h_tot, h_co, h_inco]


def get_data_MB(varplot='reco_Evis',loc=''):
    
    if varplot=='reco_Evis':
        _, data = np.loadtxt(loc+"digitized/miniboone_2020/Evis/data_Evis.dat", unpack=True)
        _, bkg = np.loadtxt(loc+"digitized/miniboone_2020/Evis/bkg_Evis.dat", unpack=True)
        signal = data - bkg
        sys_signal = 0.1
        sys_bkg = 0.1
        
    elif varplot=='reco_Enu':
        # miniboone nu data 2020
        _, data = np.loadtxt(loc+"digitized/miniboone_2020/Enu/data.dat", unpack=True)
        _, bkg = np.loadtxt(loc+"digitized/miniboone_2020/Enu/constrained_bkg.dat", unpack=True)
        _, error_low = np.loadtxt(loc+"digitized/miniboone_2020/Enu/lower_error_bar_constrained_bkg.dat", unpack=True)
        signal = data - bkg
        sys_bkg = (bkg - error_low)/bkg
        sys_signal = 0.1
        bin_e = np.loadtxt(loc+"digitized/miniboone_2020/Enu/bin_edges.dat")
        bin_w = (bin_e[1:] - bin_e[:-1])
        signal *= bin_w*1e3
        bkg *= bin_w*1e3
            
    elif varplot=='reco_angle':
        _, data = np.loadtxt(loc+"digitized/miniboone_2020/cos_Theta/data_cosTheta.dat", unpack=True)
        _, bkg = np.loadtxt(loc+"digitized/miniboone_2020/cos_Theta/bkg_cosTheta.dat", unpack=True)
        signal = data - bkg
        sys_signal = 0.1
        sys_bkg = 0.1
        
    return [signal,bkg,sys_signal,sys_bkg]


# Main plotting function for signal at MiniBooNE (stacked histograms)
def histogram1D_data_stacked(plotname, df, XLABEL, TITLE, varplot='reco_costheta_beam', tot_events  = 1.0, rasterized=True,loc=''):

    # Masks
    coherent = (df['scattering_regime'] == 'coherent')
    pel = (df['scattering_regime'] == 'p-el')
    HC = (df['helicity'] == 'conserving')
    HF = (df['helicity'] == 'flipping')
   
    # identifiers  
    cases = [coherent & HC, pel & HC, coherent & HF, pel & HF]
    case_names = [r"coherent conserving", r"p-el conserving", r"coherent flipping", r"p-el flipping"]
    case_shorthands = [r"coh HC", r"incoh HC", r"coh HF", r"incoh HF"]
    colors=['dodgerblue','violet', 'dodgerblue', 'violet']

    nevents = []
    legends = []
    tot_samples = np.size(df['reco_w'])
    for i in range(4):
        this_n_events = int(round(np.sum(df['reco_w'][cases[i]])/np.sum(df['reco_w'])*tot_events))
        nevents.append(this_n_events)
        legends.append(f'{case_shorthands[i]} ({this_n_events} events)')
        
    fsize = 10
    fig = plt.figure()
    ax = fig.add_axes(axes_form, rasterized=rasterized)
    ax.patch.set_alpha(0.0)


    #####################
    # MiniBooNE data 
    if varplot=='reco_Evis':

        # miniboone nu data
        bin_c, data_MB_enu_nue = np.loadtxt(loc+"digitized/miniboone_2020/Evis/data_Evis.dat", unpack=True)
        _, data_MB_bkg = np.loadtxt(loc+"digitized/miniboone_2020/Evis/bkg_Evis.dat", unpack=True)
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
        _, data_MB = np.loadtxt(loc+"digitized/miniboone_2020/Enu/data.dat", unpack=True)
        _, data_MB_bkg = np.loadtxt(loc+"digitized/miniboone_2020/Enu/constrained_bkg.dat", unpack=True)
        _, MB_bkg_lower_error_bar = np.loadtxt(loc+"digitized/miniboone_2020/Enu/lower_error_bar_constrained_bkg.dat", unpack=True)
        bin_e = np.loadtxt(loc+"digitized/miniboone_2020/Enu/bin_edges.dat")
        
        
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
        bin_c, data_MB_cost_nue = np.loadtxt(loc+"digitized/miniboone_2020/cos_Theta/data_cosTheta.dat", unpack=True)
        _, data_MB_bkg = np.loadtxt(loc+"digitized/miniboone_2020/cos_Theta/bkg_cosTheta.dat", unpack=True)
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

    
    plt.savefig(plotname+'.pdf',dpi=400)

    plt.close()

####### Deprecated
# Main plotting function for signal at MiniBooNE (stacked histograms)
# def histogram1D_data_stacked_seaborn(plotname, obs, XLABEL, TITLE, varplot='reco_costheta_beam', tot_events  = 1.0, kde = False):
    
#     coherent = (obs['scattering_regime'] == 'coherent')
#     pel = (obs['scattering_regime'] == 'p-el')

#     HC = (obs['helicity'] == 'conserving')
#     HF = (obs['helicity'] == 'flipping')
#     cases = [coherent & HC, pel & HC, coherent & HF, pel & HF]
#     case_names = [r"coherent conserving", r"p-el conserving", r"coherent flipping", r"p-el flipping"]
#     case_shorthands = [r"coh HC", r"incoh HC", r"coh HF", r"incoh HF"]
#     colors=['dodgerblue','violet', 'dodgerblue', 'violet']

#     nevents = []
#     legends = []
#     tot_samples = np.size(obs['reco_w'])
#     for i in range(4):
#         this_n_events = int(round(np.sum(cases[i])/tot_samples*tot_events))
#         nevents.append(this_n_events)
#         legends.append(mpatches.Patch(color=colors[i], label=f'{case_shorthands[i]} ({this_n_events} events)') )
        
#     fsize = 10
#     sns.set_style("white")
#     fig = plt.figure()
#     ax = fig.add_axes(axes_form, rasterized=False)
#     ax.patch.set_alpha(0.0)


#     #####################
#     # MiniBooNE data 
#     if varplot=='reco_Evis':

#         # miniboone nu data
#         bin_c, data_MB_enu_nue = np.loadtxt("digitized/miniboone_2020/Evis/data_Evis.dat", unpack=True)
#         _, data_MB_bkg = np.loadtxt("digitized/miniboone_2020/Evis/bkg_Evis.dat", unpack=True)
#         bin_c *= 1e-3
#         bin_w = 0.05*bin_c/bin_c
#         bin_e = np.append(0.1, bin_w/2.0 + bin_c)
#         hist_type = 'count'
#         units = 1
#         data_plot(ax,\
#                     bin_c,
#                     bin_w, 
#                     (data_MB_enu_nue-data_MB_bkg),
#                     (np.sqrt(data_MB_enu_nue)), 
#                     (np.sqrt(data_MB_enu_nue)))

#     elif varplot=='reco_Enu':

#         # miniboone nu data 2020
#         _, data_MB = np.loadtxt("digitized/miniboone_2020/Enu/data.dat", unpack=True)
#         _, data_MB_bkg = np.loadtxt("digitized/miniboone_2020/Enu/constrained_bkg.dat", unpack=True)
#         _, MB_bkg_lower_error_bar = np.loadtxt("digitized/miniboone_2020/Enu/lower_error_bar_constrained_bkg.dat", unpack=True)
#         bin_e = np.loadtxt("digitized/miniboone_2020/Enu/bin_edges.dat")
        
        
#         data_MB = data_MB[:-1]
#         bin_e = bin_e[:-1]
#         data_MB_bkg = data_MB_bkg[:-1]
#         MB_bkg_lower_error_bar = MB_bkg_lower_error_bar[:-1]
#         units = 1e3 # from GeV to MeV

#         bin_w = (bin_e[1:] - bin_e[:-1])
#         bin_c = bin_e[:-1] + bin_w/2

#         data_MB_enu_nue = (data_MB - data_MB_bkg)*bin_w*units
#         error_bar = np.sqrt( ((data_MB_bkg - MB_bkg_lower_error_bar)*bin_w*units)**2
#                                 + np.sqrt(data_MB**2*bin_w*units) )

#         hist_type = 'frequency'

#         data_plot(ax,\
#                     bin_c,
#                     bin_w, 
#                     data_MB_enu_nue/bin_w/units,
#                     error_bar/bin_w/units, 
#                     error_bar/bin_w/units)


#     elif varplot=='reco_costheta_beam':

#         # miniboone nu data
#         bin_c, data_MB_cost_nue = np.loadtxt("digitized/miniboone_2020/cos_Theta/data_cosTheta.dat", unpack=True)
#         _, data_MB_bkg = np.loadtxt("digitized/miniboone_2020/cos_Theta/bkg_cosTheta.dat", unpack=True)
#         bin_w = np.ones(len(bin_c))*0.1
#         bin_e = np.linspace(-1,1,21)
#         hist_type = 'count'
#         units = 1/0.1
#         data_plot(ax,
#                 bin_c,
#                 bin_w, 
#                 (data_MB_cost_nue-data_MB_bkg),
#                 np.sqrt(data_MB_cost_nue), 
#                 np.sqrt(data_MB_cost_nue))

#     df = pd.DataFrame(obs)
#     df['regime'] = df[['scattering_regime', 'helicity']].agg(' '.join, axis=1)
#     df['reco_w'] = df['reco_w']/np.sum(df['reco_w'])*tot_events/units
    
#     args = {
#         'x': varplot, 
#         'stat': hist_type,
#         'hue': 'regime', 
#         'hue_order': case_names,
#         'weights': 'reco_w',
#         'multiple': 'stack',
#         'bins': bin_e,
#         'binrange': (np.min(bin_e),np.max(bin_e)), 
#         'common_bins': True, 
#         'common_norm': True, 
#         'element' : 'bars',
#         'kde': kde,
#         'legend': False,
#         # 'bottom': data_MB_bkg,
#     }

#     sns.histplot(df, palette=4*['black'], fill=False, lw=0.1, zorder=2, **args)
#     sns.histplot(df, palette=colors, lw=0, zorder=1, **args)


#     ax.set_title(TITLE, fontsize=0.8*fsize)
#     ax.legend(handles=legends, frameon=False, loc='best')
#     ax.set_xlabel(XLABEL,fontsize=fsize)
#     ax.set_xlim(np.min(bin_e),np.max(bin_e))

#     if varplot=='reco_Enu':
#         ax.set_ylim(0,ax.get_ylim()[1]*1.1)
#         ax.set_ylabel(r"Excess events/MeV",fontsize=fsize)
#     else:
#         ax.set_ylim(-20,ax.get_ylim()[1]*1.1)
#         ax.set_ylabel(r"Excess events",fontsize=fsize)

#     if kde:
#         plt.savefig(plotname+'kde.pdf',dpi=400)
#     else:
#         plt.savefig(plotname+'.pdf',dpi=400)

#     plt.close()

##### FIX ME
# Main plotting function for signal at MicroBooNE (stacked histograms)
def histogram1D_data_muB(plotname, obs, XLABEL, TITLE, regime=None,varplot='reco_angle', tot_events = 1.0, kde = False):
    
    sns.set_style("white")
    
    fsize = 10

    coherent = (obs['scattering_regime'] == 'coherent')
    pel = (obs['scattering_regime'] == 'p-el')
    
    HC = (obs['helicity'] == 'conserving')
    HF = (obs['helicity'] == 'flipping')

    #####
    # FIX ME -- WE NEED TO ITERATE OVER RELEVANT CASES AND BUILD LABELS THAT WAY
    # THESE LABELS ARE WRITTEN BY HAND AND ARE OLD....
    label3=r"incoh $N_5\to N_4$"
    label4=r"coh $N_5\to N_4$"
    label2=r"incoh $N_6\to N_4$"
    label1=r"coh $N_6\to N_4$"
    ALPHA = 0.8


    color4='dodgerblue'
    color3='dodgerblue'
    color2='violet'
    color1='violet'

    if varplot=='reco_Evis':
        

        # microboone nu data for bins
        Enu_binc, data_MB_enu_nue = np.loadtxt("digitized/miniboone_2020/Evis/data_Evis.dat", unpack=True)
        _, data_MB_enu_nue_bkg = np.loadtxt("digitized/miniboone_2020/Evis/bkg_Evis.dat", unpack=True)
        Enu_binc *= 1e-3
        binw_enu = 0.05*Enu_binc/Enu_binc
        bin_e = np.append(0.1, binw_enu/2.0 + Enu_binc)
        nbins=np.size(Enu_binc)

        hist4 = np.histogram(obs[varplot][coherent & HC], weights=obs['reco_w'][coherent & HC], bins=bin_e, density = False)
        hist3 = np.histogram(obs[varplot][pel & HC], weights=obs['reco_w'][pel & HC], bins=bin_e, density = False)
        
        ans0 = hist4[1][:nbins]
        norm=np.sum(hist3[0]+hist4[0])/tot_events
        toy_logger.info('NORMALIZATION:',norm)
        full = (hist3[0]+hist4[0])/norm
        
        fig = plt.figure()
        ax = fig.add_axes(axes_form, rasterized=False)
        ax.patch.set_alpha(0.0)
        
        ax.step(np.append(ans0,10e10), 
            np.append(full, 0.0), 
            where='post',
            c='black', lw = 0.5,rasterized=True)
        
        norm=np.sum(hist3[0]+hist4[0])/tot_events

        h4 = hist4[0]/norm
        h4 = pd.DataFrame(data=h4, columns=['distribution'])
        h4['Scattering regime'] = 'coherent'
        h3 = hist3[0]/norm
        h3 = pd.DataFrame(data=h3, columns=['distribution'])
        h3['Scattering regime'] = 'incoherent'

        
        xaxis = (hist4[1][:-1]+hist4[1][1:])/2.
        sns.set_style("white")
        sns.set_style("ticks", {"xtick.major.size": 20, "ytick.major.size": 30})

        n3 = len(h3['distribution'])
        n4 = len(h4['distribution'])
        hue3 = pd.Series(['incoh' for i in range(n3)])
        hue4 = pd.Series(['coh' for i in range(n4)])
        myhue = pd.concat([hue3,hue4],ignore_index=True)

        myhist = pd.concat([h3['distribution'],h4['distribution']],ignore_index=True)
        myx = pd.Series(xaxis)
        myx = pd.concat([myx,myx],ignore_index=True)
        
        if kde:
            sns.histplot(x=myx, bins=hist4[1],weights=myhist,hue=myhue, multiple='stack',palette=['lightblue','darkblue'],thresh=0, kde=True)
        else:
            sns.histplot(x=myx, bins=hist4[1],weights=myhist,hue=myhue, multiple='stack',palette=['lightblue','darkblue'],thresh=0)

        top_bar = mpatches.Patch(color='lightblue', label=label4 + ' (%i events)'%(round(np.sum(hist4[0]/norm))))
        bottom_bar = mpatches.Patch(color='darkblue', label=label3 + ' (%i events)'%(round(np.sum(hist3[0]/norm))))
        plt.legend(handles=[top_bar, bottom_bar],fontsize=fsize)
        
        ax.set_ylabel(r"Excess events",fontsize=fsize)
        ax.set_title(TITLE, fontsize=0.8*fsize)

    elif varplot=='reco_Enu':

        # microboone nu data
        Enu_binc, data_MB_enu_nue = np.loadtxt("digitized/miniboone/Enu_excess_nue.dat", unpack=True)
        Enu_binc, data_MB_enu_nue_errorlow = np.loadtxt("digitized/miniboone/Enu_excess_nue_lowererror.dat", unpack=True)
        Enu_binc, data_MB_enu_nue_errorup = np.loadtxt("digitized/miniboone/Enu_excess_nue_uppererror.dat", unpack=True)
        binw_enu = np.array([0.1,0.075,0.1,0.075,0.125,0.125,0.15,0.15,0.2,0.2])
        bin_e = np.array([0.2,0.3,0.375,0.475,0.550,0.675,0.8,0.95,1.1,1.3,1.5])
        data_MB_enu_nue *=  binw_enu*1e3
        data_MB_enu_nue_errorlow *= binw_enu*1e3
        data_MB_enu_nue_errorup *= binw_enu*1e3
        units = 1e3
        nbins=np.size(binw_enu)
        
        
        hist4 = np.histogram(obs[varplot][coherent & HC], weights=obs['reco_w'][coherent & HC], bins=bin_e, density = False)
        hist3 = np.histogram(obs[varplot][pel & HC], weights=obs['reco_w'][pel & HC], bins=bin_e, density = False)
        
        ans0 = hist4[1][:nbins]
        norm=np.sum(hist3[0]+hist4[0])/tot_events*binw_enu*units
        toy_logger.info('NORMALIZATION:',norm)
        full = (hist3[0]+hist4[0])/norm
        
        fig = plt.figure()
        ax = fig.add_axes(axes_form, rasterized=False)
        ax.patch.set_alpha(0.0)
        
        ax.step(np.append(ans0,10e10), 
            np.append(full, 0.0), 
            where='post',
            c='black', lw = 0.5)
        
        h4 = hist4[0]/norm
        h4 = pd.DataFrame(data=h4, columns=['distribution'])
        h4['Scattering regime'] = 'coherent'
        h3 = hist3[0]/norm
        h3 = pd.DataFrame(data=h3, columns=['distribution'])
        h3['Scattering regime'] = 'incoherent'

        
        xaxis = (hist4[1][:-1]+hist4[1][1:])/2.
        sns.set_style("white")
        sns.set_style("ticks", {"xtick.major.size": 20, "ytick.major.size": 30})

        n3 = len(h3['distribution'])
        n4 = len(h4['distribution'])
        hue3 = pd.Series(['incoh' for i in range(n3)])
        hue4 = pd.Series(['coh' for i in range(n4)])
        myhue = pd.concat([hue3,hue4],ignore_index=True)

        myhist = pd.concat([h3['distribution'],h4['distribution']],ignore_index=True)
        myx = pd.Series(xaxis)
        myx = pd.concat([myx,myx],ignore_index=True)

        
        if kde:
            sns.histplot(x=myx, bins=hist4[1],weights=myhist,hue=myhue, multiple='stack',palette=['lightblue','darkblue'],thresh=0, kde=True)
        else:
            sns.histplot(x=myx, bins=hist4[1],weights=myhist,hue=myhue, multiple='stack',palette=['lightblue','darkblue'],thresh=0)


        top_bar = mpatches.Patch(color='lightblue', label=label4 + ' (%i events)'%(round(np.sum(hist4[0]/norm))))
        bottom_bar = mpatches.Patch(color='darkblue', label=label3 + ' (%i events)'%(round(np.sum(hist3[0]/norm))))
        plt.legend(handles=[top_bar, bottom_bar],fontsize=fsize)

        ax.set_ylabel(r"Excess events/MeV",fontsize=fsize)
        ax.set_title(TITLE, fontsize=0.8*fsize)

    elif varplot=='reco_angle':

        # microboone nu data
        cost_binc, data_MB_cost_nue = np.loadtxt("digitized/miniboone_2020/cos_Theta/data_cosTheta.dat", unpack=True)
        _, data_MB_cost_nue_bkg = np.loadtxt("digitized/miniboone_2020/cos_Theta/bkg_cosTheta.dat", unpack=True)
        nbins = np.size(cost_binc)
        binw_cost = np.ones(nbins)*0.1
        bincost_e = np.linspace(-1,1,21)

        hist4 = np.histogram(np.cos(obs['reco_theta_beam']*np.pi/180)[coherent & HC], weights=obs['reco_w'][coherent & HC], bins=bincost_e, density = False)
        hist3 = np.histogram(np.cos(obs['reco_theta_beam']*np.pi/180)[pel & HC], weights=obs['reco_w'][pel & HC], bins=bincost_e, density = False)
        
        ans0 = hist4[1][:nbins]
        norm=np.sum(hist3[0]+hist4[0])/tot_events#*binw_cost

        full = (hist3[0]+hist4[0])/norm
        
        fig = plt.figure()
        ax = fig.add_axes(axes_form, rasterized=False)
        ax.patch.set_alpha(0.0)
        
        ax.step(np.append(ans0,10e10), 
            np.append(full, 0.0), 
            where='post',
            c='black', lw = 0.5)
        
        h4 = hist4[0]/norm
        h4 = pd.DataFrame(data=h4, columns=['distribution'])
        h4['Scattering regime'] = 'coherent'
        h3 = hist3[0]/norm
        h3 = pd.DataFrame(data=h3, columns=['distribution'])
        h3['Scattering regime'] = 'incoherent'

        
        xaxis = (hist4[1][:-1]+hist4[1][1:])/2.
        sns.set_style("white")
        sns.set_style("ticks", {"xtick.major.size": 20, "ytick.major.size": 30})

        n3 = len(h3['distribution'])
        n4 = len(h4['distribution'])
        hue3 = pd.Series(['incoh' for i in range(n3)])
        hue4 = pd.Series(['coh' for i in range(n4)])
        myhue = pd.concat([hue3,hue4],ignore_index=True)

        myhist = pd.concat([h3['distribution'],h4['distribution']],ignore_index=True)
        myx = pd.Series(xaxis)
        myx = pd.concat([myx,myx],ignore_index=True)

        
        
        if kde:
            sns.histplot(x=myx, bins=hist4[1],weights=myhist,hue=myhue, multiple='stack',palette=['lightblue','darkblue'],thresh=0, kde=True)
        else:
            sns.histplot(x=myx, bins=hist4[1],weights=myhist,hue=myhue, multiple='stack',palette=['lightblue','darkblue'],thresh=0)


        ax.set_ylabel(r"Excess events",fontsize=fsize)
        ax.set_title(TITLE, fontsize=0.8*fsize)

    else:
        toy_logger.error('Error! No plot type specified.')



    #ax.legend(loc="best", frameon=False, fontsize=0.8*fsize)
    ax.set_xlabel(XLABEL,fontsize=fsize)

    #ax.set_xlim(TMIN,TMAX)
    # ax.set_yscale('log')

    ax.set_ylim(-20,ax.get_ylim()[1]*1.1)
    if varplot=='reco_Enu':
        ax.set_ylim(-0.2,ax.get_ylim()[1]*1.1)

    if kde:
        plt.savefig(plotname+'kde.pdf',dpi=400)
    else:
        plt.savefig(plotname+'.pdf',dpi=400)
    
    plt.close()


def data_plot(ax, X, BINW, DATA, ERRORLOW, ERRORUP):
    ax.errorbar(X, DATA, yerr= np.array([ERRORLOW,ERRORUP]), xerr = BINW/2.0, \
                            marker="o", markeredgewidth=0.5, capsize=1.0,markerfacecolor="black",\
                            markeredgecolor="black",ms=2, color='black', lw = 0.0, elinewidth=0.8, zorder=10)



def histogram1D(plotname, obs, w, TMIN, TMAX,  XLABEL, TITLE, nbins, regime=None, colors=None, legends=None, rasterized = True):

    fsize = 10
    fig = plt.figure()
    ax = fig.add_axes(axes_form, rasterized=rasterized)
    ax.patch.set_alpha(0.0)

    # normalize
    w = w/np.sum(w)
    # colors = 
    nregimes = len(regime)
    bin_e = np.linspace(TMIN,TMAX, nbins+1, endpoint=True)
    bin_w = (bin_e[1:] - bin_e[:-1])
    if not regime == None and legends == None:
        legends = [f'case {i}' for i in range(nregimes)]
    if not regime == None and colors == None:
        color = cm.rainbow(np.linspace(0, 1, n))
        colors = [c for c in color]

    if not regime == None:
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
    # ax.set_yscale('log')
    ax.set_ylim(0.0,ax.get_ylim()[1]*1.1)
    plt.savefig(plotname, dpi=400)
    plt.close()


def histogram2D(plotname, obsx, obsy, w,  xrange=None, yrange=None,  xlabel='x',  ylabel='y', title="Dark News", nbins=20, logx=False, logy=False):
    
    fsize = 11
    
    fig, ax = std_fig(ax_form = [0.15,0.15,0.78,0.74])

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

    plt.legend(loc="upper left", frameon=False, fontsize=fsize)
    ax.set_xlabel(xlabel,fontsize=fsize)
    ax.set_ylabel(ylabel,fontsize=fsize)

    plt.savefig(plotname)
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
    
    # number of events
    sample_size = len(df.index)

    # four momenta
    pN   = df['P_decay_N_parent']
    pnu  = df['P_decay_N_daughter']
    pZ   = df['P_decay_ell_minus'] + df['P_decay_ell_plus']
    plm  = df['P_decay_ell_minus']
    plp  = df['P_decay_ell_plus']
    pHad = df['P_recoil']

    # weights
    w     = df['w_rate']
    w_pel = df['w_rate'][pel]
    w_coh = df['w_rate'][coherent]

    # variables
    df['E_N']   = pN['t']
    df['E_Z']   = pZ['t']
    df['E_lp']  = plp['t']
    df['E_lm']  = plm['t']
    df['E_tot'] = plm['t'] + plp['t']
    df['E_asy'] = (df['E_lp'] - df['E_lm'])/(df['E_lp'] + df['E_lm'])
    df['E_Had'] = pHad['t']

    df['M_had'] = fv.df_inv_mass(pHad, pHad)
    df['Q2'] = -(2*df['M_had']**2-2*df['E_Had']*df['M_had'])
    
    df['costheta_N']   = fv.df_cos_azimuthal(pN) 
    df['costheta_nu']  = fv.df_cos_azimuthal(pnu) 
    df['costheta_Had'] = fv.df_cos_azimuthal(pHad) 
    df['inv_mass']     = fv.df_inv_mass(plm+plp, plm+plp)
    
    df['costheta_sum'] = fv.df_cos_azimuthal(plm+plp)
    df['costheta_lp'] = fv.df_cos_azimuthal(plp)
    df['costheta_lm'] = fv.df_cos_azimuthal(plm)

    df['costheta_sum_had'] = fv.df_cos_opening_angle(plm+plp, pHad)
    df['theta_sum_had'] = np.arccos(df['costheta_sum_had'])*180/np.pi
    
    df['theta_sum'] = np.arccos(df['costheta_sum'])*180/np.pi
    df['theta_lp'] = np.arccos(df['costheta_lp'])*180/np.pi
    df['theta_lm'] = np.arccos(df['costheta_lm'])*180/np.pi
    df['theta_nu'] = np.arccos(df['costheta_nu'])*180/np.pi

    df['Delta_costheta'] = fv.df_cos_opening_angle(plm,plp)
    df['Delta_theta'] = np.arccos(df['Delta_costheta'])*180/np.pi

    df['theta_proton'] = np.arccos(df['costheta_Had'][pel])*180/np.pi
    df['theta_nucleus'] = np.arccos(df['costheta_Had'][coherent])*180/np.pi
    
    df['T_proton'] = (df['E_Had'] - df['M_had'])[pel]
    df['T_nucleus'] = (df['E_Had'] - df['M_had'])[coherent]
    
    minus_lead = (plm['t'] >= plp['t'])
    plus_lead  = (plp['t'] > plm['t'])

    df['E_subleading'] = np.minimum(plm['t'], plp['t'])
    df['E_leading'] = np.maximum(plm['t'], plp['t'])

    df['theta_subleading'] = df['theta_lp']*plus_lead + df['theta_lm']*minus_lead
    df['theta_leading'] = df['theta_lp']*(~plus_lead) + df['theta_lm']*(~minus_lead)

    # CCQE neutrino energy
    df['E_nu_reco'] = const.m_proton * (plp['t'] + plm['t']) / ( const.m_proton - (plp['t'] + plm['t'])*(1.0 - (df['costheta_lm']*plm['t'] + df['costheta_lp'] * plp['t'])/(plp['t'] + plm['t'])  ))




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
    
    histogram1D(PATH+"/1D_T_proton.pdf", df['T_proton'][pel]*1e3, w_pel, 0.0, 2.0, r"$T_{\rm p^+}$ (MeV)", 'el proton only', 50, **args)
    histogram1D(PATH+"/1D_theta_proton.pdf", df['theta_proton'][pel]*180.0/np.pi, w_pel, 0.0, 180, r"$\theta_{p^+}$ ($^\circ$)", 'el proton only', 50, **args)
    histogram1D(PATH+"/1D_T_nucleus.pdf", df['T_nucleus'][coherent]*1e3, w_coh, 0.0, 3, r"$T_{\rm Nucleus}$ (MeV)", 'coh nucleus only', 50, **args)
    histogram1D(PATH+"/1D_theta_nucleus.pdf", df['theta_nucleus'][coherent]*180.0/np.pi, w_coh, 0.0, 180, r"$\theta_{\rm Nucleus}$ ($^\circ$)", 'coh nucleus only', 50, **args)

    # energies
    histogram1D(PATH+"/1D_E_lp.pdf", df['E_lp'], w, 0.0, 2.0, r"$E_{\ell^+}$ GeV", title, 100, **args)
    histogram1D(PATH+"/1D_E_lm.pdf", df['E_lm'], w, 0.0, 2.0, r"$E_{\ell^-}$ GeV", title, 100, **args)
    histogram1D(PATH+"/1D_E_tot.pdf", df['E_tot'], w, 0.0, 2.0, r"$E_{\ell^-}+E_{\ell^+}$ GeV", title, 100, **args)

    histogram1D(PATH+"/1D_E_nu_truth.pdf", df['P_projectile','t'], w, 0.0, 2.0, r"$E_\nu^{\rm truth}/$GeV", title, 20, **args)
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



      
    return 0 



