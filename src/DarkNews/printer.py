import os
import numpy as np
import dill 

from particle import literals as lp

from . import logger, prettyprinter
from . import const
from . import pdg
from . import fourvec as fv
from . import Cfourvec as Cfv


def unweigh_events(df_gen, nevents, prob_col = 'w_event_rate', **kwargs):
	'''
		Unweigh events in dataframe down to "nevents" using accept-reject method with the weights in "prob_col" of dataframe.
			
		Fails if nevents is smaller than the total number of unweighted events.

		kwargs passed to numpy's random.choice.
	'''
	logger.info(f"Unweighing events down to {nevents} entries.")
	prob = df_gen[prob_col]/np.sum(df_gen[prob_col])
	if (prob < 0).any():
		logger.error(f"ERROR! Probabily for unweighing contains negative values! Bad weights? {prob_col} < 0.")
	if (prob == 0).any():
		logger.warning(f"WARNING! Discarding zero-valued weights for unweighing. Total of {sum(prob == 0)} of {len(prob)} zero entries for {prob_col}.")
		AccEntries = np.random.choice(df_gen.index[prob>0], size=nevents, p=prob[prob>0], *kwargs)
	else:
		AccEntries = np.random.choice(df_gen.index, size=nevents, p=prob, *kwargs)

	# extract selected entries
	return df_gen.filter(AccEntries, axis=0).reset_index()



def print_events_to_pandas(PATH_data, df_gen):
	
	# Create target Directory if it doesn't exist
	if not os.path.exists(PATH_data):
		os.makedirs(PATH_data)
	if PATH_data[-1] != '/':
		PATH_data += '/'
	out_file_name = PATH_data+f"pandas_df.pckl"
	
	# pickles DarkNews classes with support for lambda functions
	dill.dump(df_gen, open(out_file_name, 'wb'))
	prettyprinter.info(f"* Events in pandas dataframe saved to file successfully:\n\n{out_file_name}")
	# aux_df.to_pickle(out_file_name)
	# pickle.dump(aux_df, open(out_file_name, 'wb'	))
	return df_gen

def print_in_order(x):
    return ' '.join(f'{t:.8g}' for t in list(x))

def print_unweighted_events_to_HEPEVT(df_gen, unweigh=False, max_events=100):
	'''
		Print events to HEPevt format.

			The file start with the total number of events:
				
				'tot_events_to_print'

			On a new line, each event starts with a brief description of the event:
				
				'event_number number_of_particles (event_weight if it exists)'

			On a new line, a new particle and its properties are added. Using string concatenation, 
			we print each particle as follows:
			
			(	
				f'0 '				  	# ignored = 0 or tracked = 1
				f' {i}'				  	# particle PDG number
				f' 0 0 0 0' 		  	# ????? (parentage)
				f' {*list([px,py,pz])}'	# particle px py pz momenta
				f' {E}' 				# particle energy
				f' {m}'				  	# particle mass
				f' {*list([x,y,z])}' 	# spatial x y z coords
				f' {t}' 				# time coord
				'\n'
			)

			The last two steps repeat until the EOF.

	'''

	# Unweigh events down to max_events?
	if unweigh:
		df_gen = unweigh_events(df_gen, nevents = max_events)

	# sample size (# of events)
	tot_events_to_print = len(df_gen.index)

	if not 'pos_scatt' in df_gen.columns:
		logger.debug("DEBUG: enforcing pos_scatt = 0")
		df_gen['pos_scatt', '0'] = np.zeros((tot_events_to_print, 0))
		df_gen['pos_scatt', '1'] = np.zeros((tot_events_to_print, 0))
		df_gen['pos_scatt', '2'] = np.zeros((tot_events_to_print, 0))
		df_gen['pos_scatt', '3'] = np.zeros((tot_events_to_print, 0))

	if not 'pos_decay' in df_gen.columns:
		logger.debug("DEBUG: enforcing pos_decay = pos_scatt")
		df_gen['pos_decay', '0'] = df_gen['pos_scatt', '0']
		df_gen['pos_decay', '1'] = df_gen['pos_scatt', '1']
		df_gen['pos_decay', '2'] = df_gen['pos_scatt', '2']
		df_gen['pos_decay', '3'] = df_gen['pos_scatt', '3']


	# pre-computing some variables

	mass_projectile = Cfv.mass(df_gen['P_projectile'].to_numpy())
	mass_target = Cfv.mass(df_gen['P_target'].to_numpy())
	mass_decay_N_parent = Cfv.mass(df_gen['P_decay_N_parent'].to_numpy())
	mass_recoil = Cfv.mass(df_gen['P_recoil'].to_numpy())
	mass_decay_N_daughter = Cfv.mass(df_gen['P_decay_N_daughter'].to_numpy())

	pvec_projectile = df_gen['P_projectile'][['1','2','3']].to_numpy()
	pvec_target = df_gen['P_target'][['1','2','3']].to_numpy()
	pvec_decay_N_parent = df_gen['P_decay_N_parent'][['1','2','3']].to_numpy()
	pvec_recoil = df_gen['P_recoil'][['1','2','3']].to_numpy()
	pvec_decay_N_daughter = df_gen['P_decay_N_daughter'][['1','2','3']].to_numpy()
	pvec_decay_ell_minus = df_gen['P_decay_ell_minus'][['1','2','3']].to_numpy()
	pvec_decay_ell_plus = df_gen['P_decay_ell_plus'][['1','2','3']].to_numpy()


	pvec_pos_decay = df_gen['pos_decay'][['1','2','3']].to_numpy()
	pvec_pos_scatt = df_gen['pos_scatt'][['1','2','3']].to_numpy()


	# HEPevt file name
	hepevt_file_name = f"{df_gen.attrs['data_path']}HEPevt.dat"
	f = open(hepevt_file_name,"w+") 
	
	# print total number of events
	f.write(f"{tot_events_to_print}\n")

	# print(df_gen['P_projectile'][['1','2','3']].to_numpy()[0])
	# loop over events
	for i in df_gen.index:
		
		# no particles & event id
		if unweigh:
			f.write(f"{i} 7 {df_gen['w_event_rate'][i]:.8g}\n")
		else:
			f.write(f"{i} 7\n")

		# scattering inital states
		f.write((	# Projectile
					f"0 "
					f" {int(lp.nu_mu.pdgid)}"
					f" 0 0 0 0"
					f" {print_in_order(pvec_projectile[i])}"
					f" {df_gen['P_projectile','0'][i]:.8g}"
					f" {mass_projectile[i]:.8g}"
					f" {print_in_order(pvec_pos_scatt[i])}"
					f" {df_gen['pos_scatt','0'][i]:.8g}"
					"\n"
					))

		f.write((	# Target
					f"0 "
					f" {int(df_gen['target_pdgid'][i])}"
					f" 0 0 0 0"
					f" {print_in_order(pvec_target[i])}"
					f" {df_gen['P_recoil','0'][i]:.8g}"
					f" {mass_target[i]:.8g}"
					f" {print_in_order(pvec_pos_scatt[i])}"
					f" {df_gen['pos_scatt','0'][i]:.8g}"
					"\n"
					))

		# scatter final products
		f.write((	# HNL produced
					f"0 "
					f" {int(pdg.neutrino5.pdgid)}"
					f" 0 0 0 0"
					f" {print_in_order(pvec_decay_N_parent[i])}"
					f" {df_gen['P_decay_N_parent','0'][i]:.8g}"
					f" {mass_decay_N_parent[i]:.8g}"
					f" {print_in_order(pvec_pos_scatt[i])}"
					f" {df_gen['pos_scatt','0'][i]:.8g}"
					"\n"
					))

		f.write((	# recoiled target
					f"0 "
					f" {int(df_gen['target_pdgid'][i])}"
					f" 0 0 0 0"
					f" {print_in_order(pvec_recoil[i])}"
					f" {df_gen['P_recoil','0'][i]:.8g}"
					f" {mass_recoil[i]:.8g}"
					f" {print_in_order(pvec_pos_scatt[i])}"
					f" {df_gen['pos_scatt','0'][i]:.8g}"
					'\n'
					))

		# decay final products
		f.write((	# daughter neutrino/HNL
					f"0 "
					f" {int(pdg.nulight.pdgid)}"
					f" 0 0 0 0"
					f" {print_in_order(pvec_decay_N_daughter[i])}"
					f" {df_gen['P_decay_N_daughter','0'][i]:.8g}"
					f" {mass_decay_N_daughter[i]:.8g}"
					f" {print_in_order(pvec_pos_decay[i])}"
					f" {df_gen['pos_decay','0'][i]:.8g}"
					'\n'
					))

		f.write((	# electron
					f"1 "
					f" {int(lp.e_minus.pdgid)}"
					f" 0 0 0 0"
					f" {print_in_order(pvec_decay_ell_minus[i])}"
					f" {df_gen['P_decay_ell_minus','0'][i]:.8g}"
					f" {const.m_e:.8g}"
					f" {print_in_order(pvec_pos_decay[i])}"
					f" {df_gen['pos_decay','0'][i]:.8g}"
					"\n"
					))

		f.write((	# positron
					f"1 "
					f" {int(lp.e_plus.pdgid)}"
					f" 0 0 0 0"
					f" {print_in_order(pvec_decay_ell_plus[i])}"
					f" {df_gen['P_decay_ell_plus','0'][i]:.8g}"
					f" {const.m_e:.8g}"
					f" {print_in_order(pvec_pos_decay[i])}"
					f" {df_gen['pos_decay','0'][i]:.8g}"
					"\n"
					))

	f.close()

	prettyprinter.info(f"* HEPevt events saved to file successfully:\n\n{hepevt_file_name}")