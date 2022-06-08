import os 
import numpy as np
from scipy.interpolate import splprep, splev
from pathlib import Path

from DarkNews import const
from DarkNews import fourvec as fv

import matplotlib

import matplotlib.pyplot as plt
from matplotlib import rc, rcParams
from matplotlib.pyplot import *
from matplotlib.pyplot import cm

###########################
fsize=12
fsize_annotate=10

std_figsize = (1.2*3.7,1.3*2.3617)
std_axes_form  = [0.16,0.16,0.81,0.76]

# standard figure  
def std_fig(ax_form=std_axes_form, 
            figsize=std_figsize,
            rasterized=False):

    rcparams={'axes.labelsize':fsize,'xtick.labelsize':fsize,'ytick.labelsize':fsize,\
                    'figure.figsize':std_figsize, 
                    'legend.frameon': False,
                    'legend.loc': 'best'  }
    plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}\usepackage{amssymb}'
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    rc('text', usetex=True)
    rcParams.update(rcparams)
    matplotlib.rcParams['hatch.linewidth'] = 0.3
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes(ax_form, rasterized=rasterized)
    ax.patch.set_alpha(0.0)

    return fig, ax

# standard saving function
def std_savefig(fig, path, dpi=400, **kwargs):
    fig.savefig(path, dpi = dpi, **kwargs)

def sci_notation(num, decimal_digits=1, precision=None, exponent=None):
    """
    Returns a string representation of the scientific
    notation of the given number formatted for use with
    LaTeX or Mathtext, with specified number of significant
    decimal digits and precision (number of decimal digits
    to show). The exponent to be used can also be specified
    explicitly.
    """
    if num != 0:
        if exponent is None:
            exponent = int(np.floor(np.log10(abs(num))))
        coeff = round(num / float(10**exponent), decimal_digits)
        if precision is None:
            precision = decimal_digits

        return r"${0:.{2}f}\cdot10^{{{1:d}}}$".format(coeff, exponent, precision)
    else:
        return r"0"

def get_hist(ax):
    n,bins = [],[]
    for rect in ax.patches:
        ((x0, y0), (x1, y1)) = rect.get_bbox().get_points()
        n.append(y1-y0)
        bins.append(x0) # left edge of each bin
    bins.append(x1) # also get right edge of last bin

    return n,bins



def histogram1D(plotname, obs, w, XLABEL, TITLE, nbins, regime=None, colors=None, legends=None, rasterized = True, TMIN=None, TMAX=None,):

    fsize = 10
    fig = plt.figure()
    ax = fig.add_axes(std_axes_form, rasterized=rasterized)
    ax.patch.set_alpha(0.0)

    if not TMIN:
        TMIN = np.min(obs)
    if not TMAX:
        TMAX = np.max(obs)

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
    ax.set_ylabel(r"events",fontsize=fsize)

    if TMIN != TMAX:
        ax.set_xlim(TMIN,TMAX)
    elif TMIN == TMAX and TMIN!=0.0:
        ax.set_xlim(TMIN*0.5, TMIN*2)
    
    ax.set_ylim(0.0,ax.get_ylim()[1]*1.1)
    std_savefig(fig, plotname, dpi=400)
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

    ax.set_xlabel(xlabel,fontsize=fsize)
    ax.set_ylabel(ylabel,fontsize=fsize)
    std_savefig(fig, plotname, dpi=400)
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
    
    histogram2D(PATH/"2D_EN_Etot.pdf", df['E_N'], df['E_tot'], w,
                                xrange=[0.0, 2.0],
                                yrange=[0.0, 2.0],
                                xlabel=r"$E_{N}$ (GeV)", 
                                ylabel=r"$E_{\ell^-}+E_{\ell^+}$ (GeV)",
                                **args_2d)

    histogram2D(PATH/"2D_Ep_Em.pdf", df['E_lm'], df['E_lp'],w,
                                xrange=[0.0, 2.0],
                                yrange=[0.0, 2.0],
                                xlabel=r"$E_{\ell^-}$ (GeV)", 
                                ylabel=r"$E_{\ell^+}$ (GeV)",
                                **args_2d)

    histogram2D(PATH/"2D_dtheta_Etot.pdf", df['Delta_theta'], df['E_tot'], w, \
                                              xrange=[0.0, 90],
                                              yrange=[0.0, 2.0],
                                              xlabel=r"$\Delta \theta_{\ell \ell}$ ($^\circ$)", 
                                              ylabel=r"$E_{\ell^+}+E_{\ell^-}$ (GeV)",
                                              **args_2d)

    histogram2D(PATH/"2D_Easyabs_Etot.pdf", np.abs(df['E_asy']), df['Delta_costheta'], w,
                                xrange=[0.0, 1.0],
                                yrange=[0.0, 90.0],
                                xlabel=r"$|E_{\rm asy}|$", 
                                ylabel=r"$\Delta \theta_{\ell \ell}$ ($^\circ$)",
                                **args_2d)

    histogram2D(PATH/"2D_Easyabs_Etot.pdf", np.abs(df['E_asy']), df['E_tot'], w,
                                xrange=[0.0, 1.0],
                                yrange=[0.0, 2.0],
                                xlabel=r"$|E_{\rm asy}|$", 
                                ylabel=r"$E_{\ell^+}+E_{\ell^-}$ (GeV)",
                                **args_2d)

    histogram2D(PATH/"2D_Easyabs_Etot.pdf", np.abs(df['E_asy']), df['E_tot'], w,
                                xrange=[0.0, 1.0],
                                yrange=[0.0, 2.0],
                                xlabel=r"$|E_{\rm asy}|$", 
                                ylabel=r"$E_{\ell^+}+E_{\ell^-}$ (GeV)",
                                **args_2d)

    histogram2D(PATH/"2D_Ehad_Etot.pdf", df['T_proton'][pel]*1e3, df['E_tot'][pel], w_pel,
                                xrange=[0.0, 1000],
                                yrange=[0.0, 2.0],
                                xlabel=r"$T_{\rm proton}$ (MeV)", 
                                ylabel=r'$E_{\ell^+} + E_{\ell^-}$ (GeV)', 
                                title=title +' proton-elastic only', nbins=n2D)


    histogram2D(PATH/"2D_thetaLead_dtheta.pdf", df['theta_subleading'], df['theta_leading'], w,
                                                xrange=[0.0, 40.0],
                                                yrange=[0.0, 40.0],
                                                xlabel=r"$\theta_{\nu_\mu \ell_{\rm lead}}$ ($^\circ$)", 
                                                ylabel=r'$\Delta \theta$ ($^\circ$)', 
                                                **args_2d)

    #################### HISTOGRAMS 1D ####################################################    
    # momentum exchange
    histogram1D(PATH/"1D_Q.pdf", np.sqrt(df['Q2']), w, r"$Q/$GeV", title, 50, **args)
    histogram1D(PATH/"1D_Q2.pdf", df['Q2'], w, r"$Q^2/$GeV$^2$", title, 50, **args)
    histogram1D(PATH/"1D_T_proton.pdf", df['T_proton'][pel]*1e3, w_pel, r"$T_{\rm p^+}$ (MeV)", 'el proton only', 50, **args)
    histogram1D(PATH/"1D_theta_proton.pdf", df['theta_proton'][pel], w_pel, r"$\theta_{p^+}$ ($^\circ$)", 'el proton only', 50, **args)
    histogram1D(PATH/"1D_T_nucleus.pdf", df['T_nucleus'][coherent]*1e3, w_coh, r"$T_{\rm Nucleus}$ (MeV)", 'coh nucleus only', 50, **args)
    histogram1D(PATH/"1D_theta_nucleus.pdf", df['theta_nucleus'][coherent], w_coh, r"$\theta_{\rm Nucleus}$ ($^\circ$)", 'coh nucleus only', 50, **args)
    histogram1D(PATH/"1D_E_lp.pdf", df['E_lp'], w, r"$E_{\ell^+}$ GeV", title, 100, **args)
    histogram1D(PATH/"1D_E_lm.pdf", df['E_lm'], w, r"$E_{\ell^-}$ GeV", title, 100, **args)
    histogram1D(PATH/"1D_E_tot.pdf", df['E_tot'], w, r"$E_{\ell^-}+E_{\ell^+}$ GeV", title, 100, **args)
    histogram1D(PATH/"1D_E_nu_truth.pdf", df['P_projectile','0'], w, r"$E_\nu^{\rm truth}/$GeV", title, 20, **args)
    histogram1D(PATH/"1D_E_nu_QEreco.pdf", df['E_nu_reco'], w, r"$E_\nu^{\rm QE-reco}/$GeV", title, 20, **args)
    histogram1D(PATH/"1D_E_N.pdf", df['E_N'], w, r"$E_N/$GeV", title, 20, **args)
    histogram1D(PATH/"1D_E_leading.pdf", df['E_leading'], w, r"$E_{\rm leading}$ GeV", title, 100, **args)
    histogram1D(PATH/"1D_E_subleading.pdf", df['E_subleading'], w, r"$E_{\rm subleading}$ GeV", title, 100, **args)
    histogram1D(PATH/"1D_costN.pdf", df['costheta_N'], w, r"$\cos(\theta_{\nu_\mu N})$", title, 20, **args)
    histogram1D(PATH/"1D_cost_sum.pdf", df['costheta_sum'], w, r"$\cos(\theta_{(ee)\nu_\mu})$", title, 20, **args)
    histogram1D(PATH/"1D_cost_sum_had.pdf", df['costheta_sum_had'], w, r"$\cos(\theta_{(ee) {\rm hadron}})$", title, 20, **args)
    histogram1D(PATH/"1D_cost_nu.pdf", df['costheta_nu'], w, r"$\cos(\theta_{\nu_\mu \nu_{\rm out}})$", title, 40, **args)
    histogram1D(PATH/"1D_theta_nu.pdf", df['theta_nu'], w, r"$\theta_{\nu_\mu \nu_{\rm out}}$", title, 40, **args)
    histogram1D(PATH/"1D_cost_lp.pdf", df['costheta_lp'],  w, r"$\cos(\theta_{\nu_\mu \ell^+})$", title, 40, **args)
    histogram1D(PATH/"1D_cost_lm.pdf", df['costheta_lm'], w, r"$\cos(\theta_{\nu_\mu \ell^-})$", title, 40, **args)
    histogram1D(PATH/"1D_theta_lp.pdf", df['theta_lp'], w, r"$\theta_{\nu_\mu \ell^+}$", title, 40, **args)
    histogram1D(PATH/"1D_theta_lm.pdf", df['theta_lm'], w, r"$\theta_{\nu_\mu \ell^-}$", title, 40, **args)
    histogram1D(PATH/"1D_theta_lead.pdf", df['theta_leading'], w, r"$\theta_{\nu_\mu \ell_{\rm lead}}$ ($^\circ$)", title, 40, **args)
    histogram1D(PATH/"1D_theta_sublead.pdf", df['theta_subleading'], w, r"$\theta_{\nu_\mu \ell_{\rm sublead}}$ ($^\circ$)", title, 40, **args)
    histogram1D(PATH/"1D_deltacos.pdf", df['Delta_costheta'], w, r"$\cos(\theta_{\ell^+ \ell^-})$", title, 40, **args)
    histogram1D(PATH/"1D_deltatheta.pdf", df['Delta_theta'], w, r"$\theta_{\ell^+ \ell^-}$", title, 40, **args)
    histogram1D(PATH/"1D_invmass.pdf", df['inv_mass'], w, r"$m_{\ell^+ \ell^-}$ [GeV]", title, 50, **args)
    histogram1D(PATH/"1D_asym.pdf", df['E_asy'], w, r"$(E_{\ell^+}-E_{\ell^-})$/($E_{\ell^+}+E_{\ell^-}$)", title, 20, **args)
    histogram1D(PATH/"1D_asym_abs.pdf", np.abs(df['E_asy']), w, r"$|E_{\ell^+}-E_{\ell^-}|$/($E_{\ell^+}+E_{\ell^-}$)", title, 20, **args)



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


