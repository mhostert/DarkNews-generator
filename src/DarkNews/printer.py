import os
import sys
import numpy as np
import pandas as pd
import pickle 
import dill 

from DarkNews import logger

from collections import defaultdict
from functools import partial

from . import const
from . import pdg
from . import fourvec as fv
from . import model

from DarkNews.decayer import decay_position
#CYTHON
import pyximport
pyximport.install(
    language_level=3,
    pyimport=False,)

from DarkNews.geom import geometry_muboone, geometry_zero_origin, geometry_muboone
from DarkNews.Cfourvec import dot4

def print_events_to_pandas(PATH_data, df_gen, bsm_model):

	# Decay events
	t_decay, x_decay, y_decay, z_decay = decay_position(df_gen['P_decay_N_parent'], l_decay_proper_cm=0.0)

	###############################################
	# SAVE ALL EVENTS AS A PANDAS DATAFRAME
	columns = [['P_projectile', 'P_decay_N_parent', 'P_target', 'P_recoil', 
					'P_decay_ell_minus', 'P_decay_ell_plus','P_decay_N_daughter', 'decay_displacement'], ['0','1','2','3']]
	columns_index = pd.MultiIndex.from_product(columns)
	
	aux_data = [
			df_gen['P_projectile'][:, 0],
			df_gen['P_projectile'][:, 1],
			df_gen['P_projectile'][:, 2],
			df_gen['P_projectile'][:, 3],

			df_gen['P_decay_N_parent'][:, 0],
			df_gen['P_decay_N_parent'][:, 1],
			df_gen['P_decay_N_parent'][:, 2],
			df_gen['P_decay_N_parent'][:, 3],

			df_gen['P_target'][:, 0],
			df_gen['P_target'][:, 1],
			df_gen['P_target'][:, 2],
			df_gen['P_target'][:, 3],

			df_gen['P_recoil'][:, 0],
			df_gen['P_recoil'][:, 1],
			df_gen['P_recoil'][:, 2],
			df_gen['P_recoil'][:, 3],

			df_gen['P_decay_ell_minus'][:, 0],
			df_gen['P_decay_ell_minus'][:, 1],
			df_gen['P_decay_ell_minus'][:, 2],
			df_gen['P_decay_ell_minus'][:, 3],

			df_gen['P_decay_ell_plus'][:, 0],
			df_gen['P_decay_ell_plus'][:, 1],
			df_gen['P_decay_ell_plus'][:, 2],
			df_gen['P_decay_ell_plus'][:, 3],
			
			df_gen['P_decay_N_daughter'][:, 0],
			df_gen['P_decay_N_daughter'][:, 1],
			df_gen['P_decay_N_daughter'][:, 2],
			df_gen['P_decay_N_daughter'][:, 3],

			t_decay,
			x_decay,
			y_decay,
			z_decay,]
	

	aux_df = pd.DataFrame(np.stack(aux_data, axis=-1), columns=columns_index)

	# differential weights
	for column in df_gen:
		if ("w_" in column):
			aux_df[column, ''] = df_gen[column]
		elif ("I_" in column):
			aux_df.attrs[column] = df_gen[column]

	aux_df['target', ''] = df_gen['target']
	aux_df['target_pdgid', ''] = df_gen['target_pdgid']
	aux_df['scattering_regime', ''] = df_gen['scattering_regime']
	aux_df['helicity', ''] = df_gen['helicity']
	aux_df['underlying_process', ''] = df_gen['underlying_process']

	# Create target Directory if it doesn't exist
	if not os.path.exists(PATH_data):
	    os.makedirs(PATH_data)
	if PATH_data[-1] != '/':
		PATH_data += '/'
	out_file_name = PATH_data+f"pandas_df.pckl"


	# saving the experiment class
	aux_df.attrs['experiment'] = df_gen['experiment']

	# saving the bsm_model class
	aux_df.attrs['bsm_model'] = df_gen['bsm_model']


	
	# pickles DarkNews classes with support for lambda functions
	dill.dump(aux_df, open(out_file_name, 'wb'))
	
	# aux_df.to_pickle(out_file_name)
	# pickle.dump(aux_df, open(out_file_name, 'wb'	))
	
def print_events_to_ndarray(PATH_data, df_gen, bsm_model):

	# Decay events
	t_decay, x_decay, y_decay, z_decay = decay_position(df_gen['P_decay_N_parent'], l_decay_proper_cm=0.0) 
	df_gen['t_decay']=t_decay 
	df_gen['x_decay']=x_decay 
	df_gen['y_decay']=y_decay 
	df_gen['z_decay']=z_decay 

	# Create target Directory if it doesn't exist 
	if not os.path.exists(PATH_data): 
	    os.makedirs(PATH_data) 
	if PATH_data[-1] != '/':
		PATH_data += '/'
	out_file_name = PATH_data+f"numpy_ndarray"

	###############################################
	# SAVE ALL EVENTS AS A NUMPY BINARY
	np.save(out_file_name, df_gen, allow_pickle=True)

#######
# not relevant anymore. 
def print_unweighted_events_to_HEPEVT(df_gen, bsm_model, unweigh=False, geom=geometry_muboone, max_events=np.inf):
	
	# sample size (# of events)
	tot_generated_events = np.shape(df_gen['w_event_rate'])[0]
	AllEntries = np.array(range(tot_generated_events))
	
	# the event rate weight
	w = df_gen['w_event_rate']

	if unweigh:
		logger.info(f"Unweighing vegas events, and saving {max_events} HEPEVT events to file.")
		if max_events < np.size(w):
			# Accept/reject method -- samples distributed according to their weights
			AccEntries = np.random.choice(AllEntries, size=max_events, replace=True, p=(np.abs(w))/np.sum(w))
	else:
		logger.info("Printing weighted HEPEVT events.")
		if max_events<np.inf:
			logger.info(f"Saving {max_events} weighted HEPEVT events to file.")
			AccEntries = np.random.choice(AllEntries, size=max_events, replace=False)
		else:
			logger.info(f"Saving {len(AllEntries)} weighted HEPEVT events to file.")
			AccEntries = AllEntries

	# get scattering positions
	t_scatter, x_scatter, y_scatter, z_scatter = geom(len(AccEntries))
	
	# decay events
	t_decay,x_decay,y_decay,z_decay = decay_position(df_gen['P_decay_N_parent'], l_decay_proper_cm=0.0)

	###############################################
	# SAVE ALL EVENTS AS A HEPEVT .dat file
	hepevt_file_name = df_gen['DATA_PATH']+f"HEPevt.dat"
	
	# Open file in write mode
	f = open(hepevt_file_name,"w+") 
	
	f.write("%i\n"%max_events)
	
	# loop over events
	for i, i_df in zip(range(len(AccEntries)), AccEntries):
		
		# no particles & event id
		if unweigh:
			f.write("%i 7\n" % i)
		else:
			f.write("%i 7 %g\n" % (i, df_gen['w_event_rate'][i_df]))

		# scattering inital states
		f.write("0 %i 0 0 0 0 %g %g %g %g %g %g %g %g %g\n"%(pdg.numu.pdgid, *df_gen['P_projectile'][i_df][1:], df_gen['P_projectile'][i_df][0], 0.0, x_scatter[i], y_scatter[i], z_scatter[i], t_scatter[i]))
		f.write("0 %i 0 0 0 0 %g %g %g %g %g %g %g %g %g\n"%(df_gen['target_pdgid'][i_df], *df_gen['P_target'][i_df][1:], df_gen['P_recoil'][i_df][0], fv.mass(df_gen['P_target'][i_df]), x_scatter[i],y_scatter[i],z_scatter[i],t_scatter[i]))

		# scatter final products
		f.write("0 %i 0 0 0 0 %g %g %g %g %g %g %g %g %g\n"%(pdg.neutrino4.pdgid, *df_gen['P_decay_N_parent'][i_df][1:], df_gen['P_decay_N_parent'][i_df][0], fv.mass(df_gen['P_decay_N_parent'][i_df]), x_scatter[i], y_scatter[i], z_scatter[i], t_scatter[i]))
		f.write("0 %i 0 0 0 0 %g %g %g %g %g %g %g %g %g\n"%(df_gen['target_pdgid'][i_df], *df_gen['P_recoil'][i_df][1:], df_gen['P_recoil'][i_df][0], fv.mass(df_gen['P_recoil'][i]), x_scatter[i],y_scatter[i],z_scatter[i],t_scatter[i]))

		# decay final products
		f.write("0 %i 0 0 0 0 %g %g %g %g %g %g %g %g %g\n"%(pdg.nulight.pdgid, *df_gen['P_decay_N_daughter'][i_df][1:], df_gen['P_decay_N_daughter'][i_df][0], fv.mass(df_gen['P_decay_N_daughter'][i_df])
			, x_decay[i], y_decay[i], z_decay[i], t_decay[i]))				
		f.write("1 %i 0 0 0 0 %g %g %g %g %g %g %g %g %g\n"%(pdg.electron.pdgid, *df_gen['P_decay_ell_minus'][i_df][1:], df_gen['P_decay_ell_minus'][i_df][0], const.m_e, x_decay[i], y_decay[i], z_decay[i],t_decay[i]))
		f.write("1 %i 0 0 0 0 %g %g %g %g %g %g %g %g %g\n"%(pdg.positron.pdgid, *df_gen['P_decay_ell_plus'][i_df][1:], df_gen['P_decay_ell_plus'][i_df][0], const.m_e, x_decay[i], y_decay[i], z_decay[i],t_decay[i]))

	f.close()
