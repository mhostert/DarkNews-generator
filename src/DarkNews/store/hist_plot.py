import numpy as np
import scipy 
import matplotlib.pyplot as plt
from matplotlib import rc, rcParams
from matplotlib.pyplot import *
from matplotlib.legend_handler import HandlerLine2D

from . import const 

#CYTHON
import pyximport
pyximport.install(
    language_level=3,
    pyimport=False,
    )
from . import Cfourvec as Cfv

def batch_plot_signalMB(df_gen, PATH, title='Dark News', NEVENTS=1):
	

	obs = analysis.compute_MB_spectrum(df_gen)

	I = np.sum(obs['w'])

	#################### HISTOGRAMS 1D ####################################################
	# histogram1D(PATH+"/1D_Evis.pdf", [obs['Evis'], obs['w'], I], 0.0, 2.0, r"$E_{\rm vis}$", title, 10, regime=obs['regime'])
	# histogram1D(PATH+"/1D_costheta.pdf", [np.cos(obs['theta_beam']*np.pi/180.0), obs['w'], I], -1.0, 1.0, r"$\cos\theta$", title, 10, regime=obs['regime'])
	
	#################### HISTOGRAMS 1D ####################################################
	histogram1D_data(PATH+"/1D_Enu_data.pdf", [obs['Enu'], obs['w'], I], 0.2, 1.5, r"$E_{\rm \nu}/$GeV", title, 10, regime=obs['regime'], 
		varplot='Enu', normalization=NEVENTS*obs['eff'])
	histogram1D_data(PATH+"/1D_Evis_data.pdf", [obs['Evis'], obs['w'], I], 0.1, 1.250, r"$E_{\rm vis}/$GeV", title, 10, regime=obs['regime'], 
		varplot='Evis', normalization=NEVENTS*obs['eff'])
	histogram1D_data(PATH+"/1D_costheta_data.pdf", [np.cos(obs['theta_beam']*np.pi/180.0), obs['w'], I], -1.0, 1.0, r"$\cos\theta$", title, 10, regime=obs['regime'], 
		varplot='angle', normalization=NEVENTS*obs['eff'])

	###################### HISTOGRAM 2D ##################################################
	n2D = 40

	# histogram2D(PATH+"/2D_Evis_ctheta.pdf", [obs['Evis'], obs['w'], I*obs['eff']],\
	# 										[np.cos(obs['theta_beam']*np.pi/180.0),obs['w'],I*obs['eff']],\
	# 										0.0, 2.0,\
	# 										-1.0,1.0,\
	# 										r"$\theta_{\nu_\mu \ell_{\rm lead}}$ ($^\circ$)", r'$\Delta \theta$ ($^\circ$)', title, n2D)
	
	return 0 

def histogram1D_data(plotname, DATA, TMIN, TMAX,  XLABEL, TITLE, nbins, regime=None,varplot='angle', normalization = 1.0):
	
	fsize = 10
	
	x1 = DATA[0]
	w1 = DATA[1]
	I1 = DATA[2]
	
	rc('text', usetex=True)
	plt.rcParams['text.latex.preamble'] = [
	r'\usepackage{amsmath}',
	r'\usepackage{amssymb}']
	params={'axes.labelsize':fsize,'xtick.labelsize':fsize,'ytick.labelsize':fsize,\
				'figure.figsize':(1.2*3.7,2.3617)	}
	rc('font',**{'family':'serif', 'serif': ['computer modern roman']})
	plt.rcParams['hatch.linewidth'] = 0.3
	rcParams.update(params)

	label4=r"coh $N_5\to N_4$"
	label3=r"incoh $N_5\to N_4$"
	label2=r"coh $N_6\to N_4$"
	label1=r"incoh $N_6\to N_4$"
	ALPHA = 0.8


	color4='dodgerblue'
	color3='dodgerblue'
	color2='violet'
	color1='violet'

	if varplot=='Evis':
		axes_form  = [0.14,0.17,0.82,0.74]

		fig = plt.figure()
		ax = fig.add_axes(axes_form)

		# # miniboone nu data
		# Enu_binc, _ = np.loadtxt("digitized/miniboone/Evis_nubar_data.dat", unpack=True)
		# _, data_MB_enu_nue = np.loadtxt("digitized/miniboone/Evis_nu_data.dat", unpack=True)
		# _, data_MB_enu_nue_bkg = np.loadtxt("digitized/miniboone/Evis_nu_data_bkg.dat", unpack=True)
		# _, data_MB_enu_nue_errorlow = np.loadtxt("digitized/miniboone/Evis_nu_data_lowererror.dat", unpack=True)
		# _, data_MB_enu_nue_errorup = np.loadtxt("digitized/miniboone/Evis_nu_data_uppererror.dat", unpack=True)
		# binw_enu = 0.1*Enu_binc/Enu_binc
		# bin_e = np.append(0.1, binw_enu/2.0 + Enu_binc)
		# nbins=np.size(Enu_binc)

		# data_plot(ax,\
		# 			Enu_binc,
		# 			binw_enu, 
		# 			(data_MB_enu_nue-data_MB_enu_nue_bkg),
		# 			-(data_MB_enu_nue_errorlow-data_MB_enu_nue), 
		# 			(data_MB_enu_nue_errorup-data_MB_enu_nue))


		# miniboone nu data
		Enu_binc, data_MB_enu_nue = np.loadtxt("digitized/miniboone_2020/Evis/data_Evis.dat", unpack=True)
		_, data_MB_enu_nue_bkg = np.loadtxt("digitized/miniboone_2020/Evis/bkg_Evis.dat", unpack=True)
		Enu_binc *= 1e-3
		binw_enu = 0.05*Enu_binc/Enu_binc
		bin_e = np.append(0.1, binw_enu/2.0 + Enu_binc)
		nbins=np.size(Enu_binc)

		data_plot(ax,\
					Enu_binc,
					binw_enu, 
					(data_MB_enu_nue-data_MB_enu_nue_bkg),
					(np.sqrt(data_MB_enu_nue)), 
					(np.sqrt(data_MB_enu_nue)))



		if (np.size(regime)!=None):
			hist4 = np.histogram(x1[regime==const.COHLH], weights=w1[regime==const.COHLH], bins=bin_e, density = False, range = (TMIN,TMAX) )
			hist3 = np.histogram(x1[regime==const.DIFLH], weights=w1[regime==const.DIFLH], bins=bin_e, density = False, range = (TMIN,TMAX) )
			hist2 = np.histogram(x1[regime==const.COHRH], weights=w1[regime==const.COHRH], bins=bin_e, density = False, range = (TMIN,TMAX) )
			hist1 = np.histogram(x1[regime==const.DIFRH], weights=w1[regime==const.DIFRH], bins=bin_e, density = False, range = (TMIN,TMAX) )

			ans0 = hist1[1][:nbins]
			norm=np.sum(hist1[0]+hist2[0]+hist3[0]+hist4[0])/normalization
			print('NORMALIZATION:',norm)
			full = (hist1[0]+hist2[0]+hist3[0]+hist4[0])/norm
			
			ax.step(np.append(ans0,10e10), 
				np.append(full, 0.0), 
				where='post',
				c='black', lw = 0.5,rasterized=True)

			ax.bar( hist4[1][:nbins],hist4[0]/norm,bottom =(hist1[0]+hist2[0]+hist3[0])/norm, width=binw_enu, #label=label4+' (%i events)'%(round(np.sum(hist4[0]/norm))),\
				facecolor='white', lw=0.1, zorder=-1, ec=color4, align='edge', alpha=ALPHA,hatch='/////////////')
			
			ax.bar( hist3[1][:nbins],hist3[0]/norm,bottom =(hist1[0]+hist2[0])/norm, width=binw_enu, #label=label3+' (%i events)'%(round(np.sum(hist3[0]/norm))),\
				facecolor=color3, lw=0.1, zorder=-1, ec=color3, align='edge', alpha=ALPHA)

			ax.bar( hist2[1][:nbins],hist2[0]/norm,bottom =hist1[0]/norm, width=binw_enu, #label=label2+' (%.2f events)'%((np.sum(hist2[0]/norm))),\
				facecolor='white', lw=0.1, zorder=-1, ec=color2, align='edge', alpha=ALPHA,hatch='/////////////')
		
			ax.bar( hist1[1][:nbins],hist1[0]/norm, width=binw_enu, #label=label1+' (%.2f events)'%((np.sum(hist1[0]/norm))),\
				facecolor=color1, lw=0.1, zorder=-1, ec=color1, align='edge', alpha=ALPHA)



		else:
			hist1 = np.histogram(x1, weights=w1, bins=nbins, density = False, range = (TMIN,TMAX) )

			ans0 = hist1[1][:nbins]
			ans1 = hist1[0]/np.sum(hist1[0])#/(ans0[1]-ans0[0])
			
			ax.bar(ans0,ans1, ans0[1]-ans0[0], label=r"PDF",\
					ec=None, fc='indigo', alpha=0.4, align='edge', lw = 0.0)	

			ax.step(np.append(ans0,10e10), np.append(ans1, 0.0), where='post',
					c='indigo', lw = 1.0)

		ax.set_ylabel(r"Excess events",fontsize=fsize)
		ax.set_title(TITLE, fontsize=0.8*fsize)

	elif varplot=='Enu':
		axes_form  = [0.14,0.22,0.82,0.76]

		fig = plt.figure()
		ax = fig.add_axes(axes_form)

		# miniboone nu data
		Enu_binc, data_MB_enu_nue = np.loadtxt("digitized/miniboone/Enu_excess_nue.dat", unpack=True)
		Enu_binc, data_MB_enu_nue_errorlow = np.loadtxt("digitized/miniboone/Enu_excess_nue_lowererror.dat", unpack=True)
		Enu_binc, data_MB_enu_nue_errorup = np.loadtxt("digitized/miniboone/Enu_excess_nue_uppererror.dat", unpack=True)
		binw_enu = np.array([0.1,0.075,0.1,0.075,0.125,0.125,0.15,0.15,0.2,0.2])
		bin_e = np.array([0.2,0.3,0.375,0.475,0.550,0.675,0.8,0.95,1.1,1.3,1.5])
		data_MB_enu_nue *= 	binw_enu*1e3
		data_MB_enu_nue_errorlow *= binw_enu*1e3
		data_MB_enu_nue_errorup *= binw_enu*1e3
		units = 1e3
		data_plot(ax,\
					Enu_binc,
					binw_enu, 
					data_MB_enu_nue/binw_enu/units,
					-(data_MB_enu_nue_errorlow-data_MB_enu_nue)/binw_enu/units, 
					(data_MB_enu_nue_errorup-data_MB_enu_nue)/binw_enu/units)



		if (np.size(regime)!=None):
			hist4 = np.histogram(x1[regime==const.COHLH], weights=w1[regime==const.COHLH], bins=bin_e, density = False, range = (TMIN,TMAX) )
			hist3 = np.histogram(x1[regime==const.DIFLH], weights=w1[regime==const.DIFLH], bins=bin_e, density = False, range = (TMIN,TMAX) )
			hist2 = np.histogram(x1[regime==const.COHRH], weights=w1[regime==const.COHRH], bins=bin_e, density = False, range = (TMIN,TMAX) )
			hist1 = np.histogram(x1[regime==const.DIFRH], weights=w1[regime==const.DIFRH], bins=bin_e, density = False, range = (TMIN,TMAX) )

			ans0 = hist1[1][:nbins]
			norm=np.sum(hist1[0]+hist2[0]+hist3[0]+hist4[0])/normalization*binw_enu*units
			print('NORMALIZATION:',norm)
			full = (hist1[0]+hist2[0]+hist3[0]+hist4[0])/norm
			

			ax.bar( hist4[1][:nbins],hist4[0]/norm,bottom =(hist1[0]+hist2[0]+hist3[0])/norm, width=binw_enu, label=label4,\
				facecolor='white', lw=0.1, zorder=-1, ec=color4, align='edge', alpha=ALPHA,hatch='/////////////')
			
			ax.bar( hist3[1][:nbins],hist3[0]/norm,bottom =(hist1[0]+hist2[0])/norm, width=binw_enu, label=label3,\
				facecolor=color3, lw=0.1, zorder=-1, ec=color3, align='edge', alpha=ALPHA)

			ax.bar( hist2[1][:nbins],hist2[0]/norm,bottom =hist1[0]/norm, width=binw_enu, label=label2,\
				facecolor='white', lw=0.1, zorder=-1, ec=color2, align='edge', alpha=ALPHA,hatch='/////////////')
		
			ax.bar( hist1[1][:nbins],hist1[0]/norm, width=binw_enu, label=label1,\
				facecolor=color1, lw=0.1, zorder=-1, ec=color1, align='edge', alpha=ALPHA)


			ax.step(np.append(ans0,10e10), 
				np.append(full, 0.0), 
				where='post',
				c='black', lw = 0.5)
		else:
			hist1 = np.histogram(x1, weights=w1, bins=nbins, density = False, range = (TMIN,TMAX) )

			ans0 = hist1[1][:nbins]
			ans1 = hist1[0]/np.sum(hist1[0])#/(ans0[1]-ans0[0])
			
			ax.bar(ans0,ans1, ans0[1]-ans0[0], label=r"PDF",\
					ec=None, fc='indigo', alpha=0.4, align='edge', lw = 0.0)	

			ax.step(np.append(ans0,10e10), np.append(ans1, 0.0), where='post',
					c='indigo', lw = 1.0)


		ax.set_ylabel(r"Excess events/MeV",fontsize=fsize)
		ax.set_title(TITLE, fontsize=0.8*fsize)

	elif varplot=='angle':
		axes_form  = [0.14,0.17,0.82,0.74]

		fig = plt.figure()
		ax = fig.add_axes(axes_form)
	

			# data_plot(ax,\
		# 			cost_binc,
		# 			binw_cost, 
		# 			(data_MB_cost_nue-data_MB_cost_nue_bkg),
		# 			-(data_MB_cost_nue_errorlow-data_MB_cost_nue), 
		# 			(data_MB_cost_nue_errorup-data_MB_cost_nue))
	
		# miniboone nu data
		cost_binc, data_MB_cost_nue = np.loadtxt("digitized/miniboone_2020/cos_Theta/data_cosTheta.dat", unpack=True)
		_, data_MB_cost_nue_bkg = np.loadtxt("digitized/miniboone_2020/cos_Theta/bkg_cosTheta.dat", unpack=True)
		nbins = np.size(cost_binc)
		binw_cost = np.ones(nbins)*0.1
		bincost_e = np.linspace(-1,1,21)

		data_plot(ax,\
					cost_binc,
					binw_cost, 
					(data_MB_cost_nue-data_MB_cost_nue_bkg),
					np.sqrt(data_MB_cost_nue), 
					np.sqrt(data_MB_cost_nue))
	
		if (np.size(regime)!=None):
			hist4 = np.histogram(x1[regime==const.COHLH], weights=w1[regime==const.COHLH], bins=bincost_e, density = False, range = (TMIN,TMAX) )
			hist3 = np.histogram(x1[regime==const.DIFLH], weights=w1[regime==const.DIFLH], bins=bincost_e, density = False, range = (TMIN,TMAX) )
			hist2 = np.histogram(x1[regime==const.COHRH], weights=w1[regime==const.COHRH], bins=bincost_e, density = False, range = (TMIN,TMAX) )
			hist1 = np.histogram(x1[regime==const.DIFRH], weights=w1[regime==const.DIFRH], bins=bincost_e, density = False, range = (TMIN,TMAX) )

			ans0 = hist1[1][:nbins]
			norm=np.sum(hist1[0]+hist2[0]+hist3[0]+hist4[0])/normalization
			print('NORMALIZATION:',norm)
			full = (hist1[0]+hist2[0]+hist3[0]+hist4[0])/norm
			 
			ax.bar( hist4[1][:nbins],hist4[0]/norm,bottom =(hist1[0]+hist2[0]+hist3[0])/norm, width=binw_cost, label=label4+' (%i events)'%(round(np.sum(hist4[0]/norm))),\
				facecolor='white', lw=0.1, zorder=-1, ec=color4, align='edge', alpha=ALPHA,hatch='/////////////')
			
			ax.bar( hist3[1][:nbins],hist3[0]/norm,bottom =(hist1[0]+hist2[0])/norm, width=binw_cost, label=label3+' (%i events)'%(round(np.sum(hist3[0]/norm))),\
				facecolor=color3, lw=0.1, zorder=-1, ec=color3, align='edge', alpha=ALPHA)

			ax.bar( hist2[1][:nbins],hist2[0]/norm,bottom =hist1[0]/norm, width=binw_cost, label=label2+' (%.2g events)'%((np.sum(hist2[0]/norm))),\
				facecolor='white', lw=0.1, zorder=-1, ec=color2, align='edge', alpha=ALPHA,hatch='/////////////')
		
			ax.bar( hist1[1][:nbins],hist1[0]/norm, width=binw_cost, label=label1+' (%.2g events)'%((np.sum(hist1[0]/norm))),\
				facecolor=color1, lw=0.1, zorder=-1, ec=color1, align='edge', alpha=ALPHA)



			ax.step(np.append(ans0,10e10), 
				np.append(full, 0.0), 
				where='post',
				c='black', lw = 0.5)
			
		else:
			hist1 = np.histogram(x1, weights=w1, bins=nbins, density = False, range = (TMIN,TMAX) )

			ans0 = hist1[1][:nbins]
			ans1 = hist1[0]/np.sum(hist1[0])#/(ans0[1]-ans0[0])
			
			ax.bar(ans0,ans1, ans0[1]-ans0[0], label=r"PDF",\
					ec=None, fc='indigo', alpha=0.4, align='edge', lw = 0.0)	

			ax.step(np.append(ans0,10e10), np.append(ans1, 0.0), where='post',
					c='indigo', lw = 1.0)


		# # miniboone nu data
		# cost_binc, data_MB_cost_nue = np.loadtxt("digitized/miniboone/costheta_nu_data.dat", unpack=True)
		# cost_binc, data_MB_cost_nue_errorlow = np.loadtxt("digitized/miniboone/costheta_nu_data_lowererror.dat", unpack=True)
		# cost_binc, data_MB_cost_nue_errorup = np.loadtxt("digitized/miniboone/costheta_nu_data_uppererror.dat", unpack=True)
		# cost_binc, data_MB_cost_nue_bkg = np.loadtxt("digitized/miniboone/costheta_nu_data_bkg.dat", unpack=True)
		# binw_cost = np.ones(np.size(cost_binc))*0.2
		# bincost_e = np.array([-1.0,-0.8,-0.6,-0.4,-0.2,0.0,0.2,0.4,0.6,0.8,1.0])




		ax.set_ylabel(r"Excess events",fontsize=fsize)
		ax.set_title(TITLE, fontsize=0.8*fsize)

	else:
		print('Error! No plot type specified.')



	plt.legend(loc="best", frameon=False, fontsize=0.8*fsize)
	ax.set_xlabel(XLABEL,fontsize=fsize)

	ax.set_xlim(TMIN,TMAX)
	# ax.set_yscale('log')
	ax.set_ylim(-20,ax.get_ylim()[1]*1.1)
	# plt.show()
	plt.savefig(plotname,dpi=400)
	plt.close()

def data_plot(ax, X, BINW, DATA, ERRORLOW, ERRORUP):
	ax.errorbar(X, DATA, yerr= np.array([ERRORLOW,ERRORUP]), xerr = BINW/2.0, \
							marker="o", markeredgewidth=0.5, capsize=1.0,markerfacecolor="black",\
							markeredgecolor="black",ms=2, color='black', lw = 0.0, elinewidth=0.8, zorder=10)


def histogram1D(plotname, DATA, TMIN, TMAX,  XLABEL, TITLE, nbins, regime=None):
	
	fsize = 11
	
	x1 = DATA[0]
	w1 = DATA[1]
	I1 = DATA[2]
	
	rc('text', usetex=True)
	params={'axes.labelsize':fsize,'xtick.labelsize':fsize,'ytick.labelsize':fsize,\
					'figure.figsize':(1.2*3.7,1.4*2.3617)	}
	rc('font',**{'family':'serif', 'serif': ['computer modern roman']})
	rcParams.update(params)
	axes_form  = [0.15,0.15,0.78,0.74]

	fig = plt.figure()
	ax = fig.add_axes(axes_form)

	if (np.size(regime)!=None):
		hist4 = np.histogram(x1[regime==const.COHLH], weights=w1[regime==const.COHLH], bins=nbins, density = False, range = (TMIN,TMAX) )
		hist3 = np.histogram(x1[regime==const.DIFLH], weights=w1[regime==const.DIFLH], bins=nbins, density = False, range = (TMIN,TMAX) )
		hist2 = np.histogram(x1[regime==const.COHRH], weights=w1[regime==const.COHRH], bins=nbins, density = False, range = (TMIN,TMAX) )
		hist1 = np.histogram(x1[regime==const.DIFRH], weights=w1[regime==const.DIFRH], bins=nbins, density = False, range = (TMIN,TMAX) )

		ans0 = hist1[1][:nbins]
		norm=np.sum(hist1[0]+hist2[0]+hist3[0]+hist4[0])
		print('NORMALIZATION:',norm)
		full = (hist1[0]+hist2[0]+hist3[0]+hist4[0])/norm
		

		ax.bar( hist1[1][:nbins],hist1[0]/norm, width=ans0[1]-ans0[0], label=r"$p^+$ el $\nu_6$",\
			ec=None, fc='dodgerblue', alpha=0.4, align='edge', lw = 0.0)	

		ax.bar( hist2[1][:nbins],hist2[0]/norm,bottom =hist1[0]/norm, width=ans0[1]-ans0[0], label=r"coh $\nu_6$",\
			ec=None, fc='firebrick', alpha=0.4, align='edge', lw = 0.0)	

		ax.bar( hist3[1][:nbins],hist3[0]/norm,bottom =(hist1[0]+hist2[0])/norm, width=ans0[1]-ans0[0], label=r"$p^+$ el $\nu_5$",\
			ec=None, fc='purple', alpha=0.4, align='edge', lw = 0.0)	

		ax.bar( hist4[1][:nbins],hist4[0]/norm,bottom =(hist1[0]+hist2[0]+hist3[0])/norm, width=ans0[1]-ans0[0], label=r"coh $\nu_5$",\
			ec=None, fc='green', alpha=0.4, align='edge', lw = 0.0)	
		
		ax.step(np.append(ans0,10e10), 
			np.append(full, 0.0), 
			where='post',
			c='black', lw = 0.3)
	else:
		hist1 = np.histogram(x1, weights=w1, bins=nbins, density = False, range = (TMIN,TMAX) )

		ans0 = hist1[1][:nbins]
		ans1 = hist1[0]/np.sum(hist1[0])#/(ans0[1]-ans0[0])
		
		ax.bar(ans0,ans1, ans0[1]-ans0[0], label=r"PDF",\
				ec=None, fc='indigo', alpha=0.4, align='edge', lw = 0.0)	

		ax.step(np.append(ans0,10e10), np.append(ans1, 0.0), where='post',
				c='indigo', lw = 1.0)

	ax.set_title(TITLE, fontsize=fsize)

	plt.legend(loc="upper left", frameon=False, fontsize=fsize)
	ax.set_xlabel(XLABEL,fontsize=fsize)
	ax.set_ylabel(r"PDF",fontsize=fsize)

	ax.set_xlim(TMIN,TMAX)
	# ax.set_yscale('log')
	ax.set_ylim(0.0,ax.get_ylim()[1]*1.1)
	# plt.show()
	plt.savefig(plotname)
	plt.close()

def histogram2D(plotname, DATACOHX, DATACOHY, XMIN, XMAX, YMIN, YMAX,  XLABEL,  YLABEL, TITLE, NBINS):
	
	fsize = 11
	
	x1 = DATACOHX[0]
	y1 = DATACOHY[0]
	w1 = DATACOHY[1]
	I1 = DATACOHX[2]

	rc('text', usetex=True)
	params={'axes.labelsize':fsize,'xtick.labelsize':fsize,'ytick.labelsize':fsize,\
					'figure.figsize':(1.2*3.7,1.4*2.3617)	}
	rc('font',**{'family':'serif', 'serif': ['computer modern roman']})
	rcParams.update(params)
	axes_form  = [0.15,0.15,0.78,0.74]

	fig = plt.figure()
	ax = fig.add_axes(axes_form)

	bar = ax.hist2d(x1,y1, bins=NBINS, weights=w1, range=[[XMIN,XMAX],[YMIN,YMAX]],cmap="Blues",density=True)

	ax.set_title(TITLE, fontsize=fsize)
	cbar_R = fig.colorbar(bar[3],ax=ax)
	cbar_R.ax.set_ylabel(r'a.u.', rotation=90)

	plt.legend(loc="upper left", frameon=False, fontsize=fsize)
	ax.set_xlabel(XLABEL,fontsize=fsize)
	ax.set_ylabel(YLABEL,fontsize=fsize)

	# ax.set_xlim(XMIN,XMAX)
	# ax.set_ylim(YMIN,YMAX)

	# plt.show()
	plt.savefig(plotname)
	plt.close()

def histogram2DLOG(plotname, DATACOHX, DATACOHY, XMIN, XMAX, YMIN, YMAX,  XLABEL,  YLABEL, TITLE, NBINS):
	
	fsize = 11
	
	x1 = DATACOHX[0]
	y1 = DATACOHY[0]
	w1 = DATACOHY[1]
	I1 = DATACOHX[2]

	rc('text', usetex=True)
	params={'axes.labelsize':fsize,'xtick.labelsize':fsize,'ytick.labelsize':fsize,\
					'figure.figsize':(1.2*3.7,1.4*2.3617)	}
	rc('font',**{'family':'serif', 'serif': ['computer modern roman']})
	rcParams.update(params)
	axes_form  = [0.15,0.15,0.78,0.74]

	fig = plt.figure()
	ax = fig.add_axes(axes_form)
			
	x1 = np.log10(x1)
	y1 = np.log10(y1)


	bar = ax.hist2d(x1,y1, bins=NBINS, weights=w1, range=[[np.log10(XMIN),np.log10(XMAX)],[np.log10(YMIN),np.log10(YMAX)]],cmap="Blues",density=True)
	# hist[1][:nbins] = 10**hist[1][:nbins]

	ax.set_title(TITLE, fontsize=fsize)
	cbar_R = fig.colorbar(bar[3],ax=ax)
	cbar_R.ax.set_ylabel(r'a.u.', rotation=90)

	plt.legend(loc="upper left", frameon=False, fontsize=fsize)
	ax.set_xlabel(XLABEL,fontsize=fsize)
	ax.set_ylabel(YLABEL,fontsize=fsize)

	# ax.set_xlim(XMIN,XMAX)
	# ax.set_ylim(YMIN,YMAX)

	# ax.set_xscale("log")
	# ax.set_yscale("log")
	# plt.show()
	plt.savefig(plotname)
	plt.close()



def batch_plot(df_gen, PATH, title='Dark News'):
	
	pN   = df_gen['P3']
	pnu   = df_gen['P2_decay']
	pZ   = df_gen['P3_decay']+df_gen['P4_decay']
	plm  = df_gen['P3_decay']
	plp  = df_gen['P4_decay']
	pHad = df_gen['P4']
	w = df_gen['w']
	I = df_gen['I']
	regime = df_gen['flags']

	sample_size = np.shape(plp)[0]

	########################## PROCESS FOR DISTRIBUTIONS ##################################################

	costhetaN = pN[:,3]/np.sqrt( Cfv.dot3(pN,pN) )
	costhetanu = pnu[:,3]/np.sqrt( Cfv.dot3(pnu,pnu) )
	costhetaHad = pHad[:,3]/np.sqrt( Cfv.dot3(pHad,pHad) )
	invmass = np.sqrt( Cfv.dot4(plm + plp, plm + plp) )
	EN   = pN[:,0] 
	EZ = pZ[:,0]
	Elp  = plp[:,0]
	Elm  = plm[:,0]
	EHad = pHad[:,0]

	Mhad = np.sqrt( Cfv.dot4(pHad, pHad) )
	Q2 = -(2*Mhad*Mhad-2*EHad*Mhad)
	Q = np.sqrt(Q2)


	MHad = Cfv.dot4(pHad,pHad)
	costheta_sum = (plm+plp)[:,3]/np.sqrt( Cfv.dot3(plm+plp,plm+plp) )
	costhetalp = plp[:,3]/np.sqrt( Cfv.dot3(plp,plp) )

	costhetalm = plm[:,3]/np.sqrt( Cfv.dot3(plm,plm) )
	Delta_costheta = Cfv.dot3(plm,plp)/np.sqrt(Cfv.dot3(plm,plm))/np.sqrt(Cfv.dot3(plp,plp))


	costhetaLeading = []
	costhetaSubLeading = []
	for i in range(sample_size):
		if Elp[i] > Elm[i]:
			costhetaLeading.append(costhetalp[i])
			costhetaSubLeading.append(costhetalm[i])
		else:
			costhetaLeading.append(costhetalm[i])
			costhetaSubLeading.append(costhetalp[i])
	costhetaLeading = np.array(costhetaLeading)
	costhetaSubLeading = np.array(costhetaSubLeading)


	Enu = const.mproton * (Elp + Elm) / ( const.mproton - (Elp + Elm)*(1.0 - (costhetalm *Elm + costhetalp * Elp)/(Elp + Elm)  ))


	#################### HISTOGRAMS 1D ####################################################
	# histogram1D_regimes(PATH+"/1D_Q.pdf", [Q, w, I], 0.0, 1., r"$Q/$GeV", title, 10)
	# histogram1D(PATH+"/1D_costN.pdf", [costhetaN, w, I], -1.0, 1.0, r"$\cos(\theta_{\nu_\mu N})$", title, 10)

	histogram1D(PATH+"/1D_Q.pdf", [Q, w, I], 0.0, 1., r"$Q/$GeV", title, 10, regime=regime)
	histogram1D(PATH+"/1D_Q2.pdf", [Q2, w, I], 0.0, 1.5, r"$Q^2/$GeV$^2$", title, 10, regime=regime)
	histogram1D(PATH+"/1D_Enu_real.pdf", [EZ, w, I], 0.0, 2.0, r"$E_\nu^{\rm real}/$GeV", title, 10, regime=regime)
	histogram1D(PATH+"/1D_Enu.pdf", [Enu, w, I], 0.0, 2.0, r"$E_\nu^{\rm fake}/$GeV", title, 10, regime=regime)
	histogram1D(PATH+"/1D_EN.pdf", [EN, w, I], 0.0, 2.0, r"$E_N/$GeV", title, 10, regime=regime)
	histogram1D(PATH+"/1D_costN.pdf", [costhetaN, w, I], -1.0, 1.0, r"$\cos(\theta_{\nu_\mu N})$", title, 10, regime=regime)
	histogram1D(PATH+"/1D_cost_sum.pdf", [costheta_sum, w, I], -1.0, 1.0, r"$\cos(\theta_{(ee)\nu_\mu})$", title, 10, regime=regime)
	histogram1D(PATH+"/1D_costnu.pdf", [costhetanu, w, I], -1.0, 1.0, r"$\cos(\theta_{\nu_\mu \nu_{\rm out}})$", title, 40, regime=regime)
	histogram1D(PATH+"/1D_thetanu.pdf", [np.arccos(costhetanu)*180.0/np.pi, w, I], 0.0, 180.0, r"$\theta_{\nu_\mu \nu_{\rm out}}$", title, 40, regime=regime)

	histogram1D(PATH+"/1D_costlp.pdf", [costhetalp, w, I],  -1.0, 1.0, r"$\cos(\theta_{\nu_\mu \ell^+})$", title, 40, regime=regime)
	histogram1D(PATH+"/1D_costlm.pdf", [costhetalm, w, I], -1.0, 1.0, r"$\cos(\theta_{\nu_\mu \ell^-})$", title, 40, regime=regime)

	histogram1D(PATH+"/1D_thetalp.pdf", [np.arccos(costhetalp)*180.0/np.pi, w, I],  0.0, 180.0, r"$\theta_{\nu_\mu \ell^+}$", title, 40, regime=regime)
	histogram1D(PATH+"/1D_thetalm.pdf", [np.arccos(costhetalm)*180.0/np.pi, w, I],  0.0, 180.0, r"$\theta_{\nu_\mu \ell^-}$", title, 40, regime=regime)

	histogram1D(PATH+"/1D_theta_lead.pdf", [np.arccos(costhetaLeading)*180.0/np.pi, w, I],  0.0, 180.0, r"$\theta_{\nu_\mu \ell_{\rm lead}}$ ($^\circ$)", title, 40, regime=regime)
	histogram1D(PATH+"/1D_theta_sublead.pdf", [np.arccos(costhetaSubLeading)*180.0/np.pi, w, I],  0.0, 180.0, r"$\theta_{\nu_\mu \ell_{\rm sublead}}$ ($^\circ$)", title, 40, regime=regime)

	histogram1D(PATH+"/1D_Elp.pdf", [Elp, w, I], 0.0, 2.0, r"$E_{\ell^+}$ GeV", title, 100, regime=regime)
	histogram1D(PATH+"/1D_Elm.pdf", [Elm, w, I], 0.0, 2.0, r"$E_{\ell^-}$ GeV", title, 100, regime=regime)

	histogram1D(PATH+"/1D_Etot.pdf", [Elm+Elp, w, I], 0.0, 2.0, r"$E_{\ell^-}+E_{\ell^+}$ GeV", title, 100, regime=regime)

	histogram1D(PATH+"/1D_deltacos.pdf", [Delta_costheta, w, I],  -1.0, 1.0, r"$\cos(\theta_{\ell^+ \ell^-})$", title, 40, regime=regime)
	histogram1D(PATH+"/1D_deltatheta.pdf", [np.arccos(Delta_costheta)*180/np.pi, w, I],  0, 180.0, r"$\theta_{\ell^+ \ell^-}$", title, 40, regime=regime)

	histogram1D(PATH+"/1D_invmass.pdf", [invmass, w, I], 0.0, np.max(invmass), r"$m_{\ell^+ \ell^-}$ [GeV]", title, 50, regime=regime)


	histogram1D(PATH+"/1D_asym_em.pdf", [Elm/(Elm+Elp), w, I], 0.0, 1.0, r"$E_{\ell^-}$/($E_{\ell^+}+E_{\ell^-}$)", title, 20, regime=regime)
	histogram1D(PATH+"/1D_asym_ep.pdf", [Elp/(Elm+Elp), w, I], 0.0, 1.0, r"$E_{\ell^+}$/($E_{\ell^+}+E_{\ell^-}$)", title, 20, regime=regime)
	histogram1D(PATH+"/1D_asym.pdf", [(Elp-Elm)/(Elm+Elp), w, I], -1.0, 1.0, r"$(E_{\ell^+}-E_{\ell^-})$/($E_{\ell^+}+E_{\ell^-}$)", title, 20, regime=regime)

	histogram1D(PATH+"/1D_Ehad_proton.pdf", [(EHad[regime==1]-MHad[regime==1])*1e3, w[regime==1], I], 0.0, 2.0, r"$T_{\rm p^+}$ (MeV)", 'el proton only', 50, regime=regime[regime==1])
	histogram1D(PATH+"/1D_theta_proton.pdf", [np.arccos(costhetaHad[regime==1])*180.0/np.pi, w[regime==1], I], 0.0, 180, r"$\theta_{p^+}$ ($^\circ$)", 'el proton only', 50, regime=regime[regime==1])
	histogram1D(PATH+"/1D_Ehad_nucleus.pdf", [(EHad[regime==0]-MHad[regime==0])*1e3, w[regime==0], I], 0.0, 3, r"$T_{\rm Nucleus}$ (MeV)", 'coh nucleus only', 50, regime=regime[regime==0])
	histogram1D(PATH+"/1D_theta_nucleus.pdf", [np.arccos(costhetaHad[regime==0])*180.0/np.pi, w[regime==0], I], 0.0, 180, r"$\theta_{\rm Nucleus}$ ($^\circ$)", 'coh nucleus only', 50, regime=regime[regime==0])

	###################### HISTOGRAM 2D ##################################################
	n2D = 40

	histogram2D(PATH+"/2D_EN_Etot.pdf", [EN, w, I], \
																[Elm+Elp, w, I], \
																0.0, 2.0, 0.0, 2.0,
																r"$E_{N}$ (GeV)", r"$E_{\ell^-}+E_{\ell^+}$ (GeV)", title, n2D)

	histogram2D(PATH+"/2D_Ep_Em.pdf", [Elm, w, I], \
																[Elp, w, I], \
																0.03, 1.0, 0.03, 1.0,
																r"$E_{\ell^-}$ (GeV)", r"$E_{\ell^+}$ (GeV)", title, n2D)

	histogram2D(PATH+"/2D_dtheta_Etot.pdf", [np.arccos(Delta_costheta)*180/np.pi, w, I], \
																[Elp+Elm, w, I], \
																0.0, 90, 0.0, 1.0,
																r"$\Delta \theta_{\ell \ell}$ ($^\circ$)", r"$E_{\ell^+}+E_{\ell^-}$ (GeV)", title, n2D)

	histogram2D(PATH+"/2D_asymmetry.pdf", [np.abs(Elp-Elm)/(Elp+Elm), w, I], \
																[Elp+Elm, w, I], \
																0.0, 1.0, 0.0, 2.0,
																r"$|E_{\ell^+}-E_{\ell^-}|$/($E_{\ell^+}+E_{\ell^-}$)", r"$E_{\ell^+}+E_{\ell^-}$ (GeV)", title, n2D)

	histogram2D(PATH+"/2D_Ehad_Etot.pdf", [(EHad[regime==1]-const.mproton)*1e3, w[regime==1], I],\
																[Elp[regime==1]+Elm[regime==1],w[regime==1],I],\
																0.0, 500, 0.0, 1.0,\
																r"$T_{\rm proton}$ (MeV)", r'$E_{\ell^+} + E_{\ell^-}$ (GeV)', 'el proton only', n2D)


	histogram2D(PATH+"/2D_Easy_dtheta.pdf", [np.abs(Elp-Elm)/(Elp+Elm), w, I],\
																[np.arccos(Delta_costheta)*180/np.pi,w,I],\
																0.0, 1.0, 0.0, 90.0,\
																r"$|E_{\ell^+}-E_{\ell^-}|$/($E_{\ell^+}+E_{\ell^-}$)", r'$\Delta \theta$ ($^\circ$)', title, n2D)
	
	histogram2D(PATH+"/2D_thetaLead_dtheta.pdf", [np.arccos(costhetaLeading)*180.0/np.pi, w, I],\
																[np.arccos(Delta_costheta)*180/np.pi,w,I],\
																0.0, 40.0, 0.0, 40.0,\
																r"$\theta_{\nu_\mu \ell_{\rm lead}}$ ($^\circ$)", r'$\Delta \theta$ ($^\circ$)', title, n2D)
	
	return 0 


