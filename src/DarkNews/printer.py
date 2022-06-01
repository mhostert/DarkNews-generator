import os
import pandas as pd
import numpy as np
import dill 
from pathlib import Path
from particle import literals as lp

from . import logger, prettyprinter
from . import const
from . import pdg
from . import Cfourvec as Cfv

import pyarrow.parquet as pq
import pyarrow as pa


def print_in_order(x):
    return ' '.join(f'{t:.8g}' for t in list(x))


class Printer:

	def __init__(self, df_gen, data_path=None, sparse=False, print_to_float32=False):
		"""
		Main printer of DarkNews. Can print events contained in the pandas dataframe to files of various types.

		Args:
			df_gen (pd.DataFrame): dataframe with the generated events.
			
			data_path (str, optional): path to be used to save the event files. Defaults to the "data_path" attribute of df_gen

			sparse (bool, optional): if True, save only the neutrino energy, charged lepton or photon momenta, and weights to save storage space.
									Not supported for HEPevt.
									Defaults to False.
			
			print_to_float32 (bool, optional): If true downgrade floats to float32 to save storage space. Only relevant when sparse is True.
												Defaults to False.

		"""
		
		# main DataFrame
		self.df_gen = df_gen
		
		if data_path:
			self.data_path = data_path
		else:
			self.data_path = self.df_gen.attrs['data_path']

		# file name and path (without extension)
		self.out_file_name = self.create_dir()

		self._sparse = sparse
		if self._sparse:
			self.sparse = self._sparse
		
		self._print_to_float32 = print_to_float32
		if self._print_to_float32:
			self.print_to_float32 = self._print_to_float32
		
	@property
	def sparse(self):
		return self._sparse
	@sparse.setter
	def sparse(self, new_value):
		self._sparse = new_value
		if self._sparse:
			self.df_sparse = self.get_sparse_df(self.df_gen)
	
	@property
	def print_to_float32(self):
		return self._print_to_float32
	@print_to_float32.setter
	def print_to_float32(self, new_value):
		self._print_to_float32 = new_value
		if self._print_to_float32:
			if self.sparse:
				self.df_sparse = self.df_sparse.apply(pd.to_numeric, downcast='float')
			else:
				raise ValueError("Can only downgrade dataframe to float32 in sparse mode.")

	# Create target directory if it doesn't exist
	def create_dir(self):
		if not os.path.exists(self.data_path):
			os.makedirs(self.data_path)
		return Path(self.data_path)

	# Keep only Enu, charged leptons, photons, and weights
	def get_sparse_df(self, df_gen):
		# keep neutrino energy
		keep_cols = ['P_projectile']
		for col in df_gen.columns.levels[0]:
			if '_ell_' in col or '_gamma' in col or 'w_decay' in col or 'w_event' in col or 'N_parent' in col:
				keep_cols.append(col)
		return df_gen[keep_cols].drop([('P_projectile','1'),('P_projectile','2'),('P_projectile','3')], axis=1)
			

	def print_events_to_ndarray(self, **kwargs):
		""" 
			Print to numpy array file (.npy) 
		
		"""
		kwargs['allow_pickle'] = kwargs.get("allow_pickle", False)
	
		if self.sparse:
			if self.print_to_float32:
				self.array_gen = self.df_sparse.to_numpy(dtype=np.float32)
			else:
				self.array_gen = self.df_sparse.to_numpy(dtype=np.float64)
			cols = [f'{v[0]}_{v[1]}' if v[1] else f'{v[0]}' for v in self.df_sparse.columns.values]
		else:
			# convert to numeric values
			self.df_gen.loc[self.df_gen['helicity']=='conserving', 'helicity'] = +1
			self.df_gen.loc[self.df_gen['helicity']=='flipping', 'helicity'] = -1
			# remove non-numeric entries
			self.df_for_numpy = self.df_gen.drop(['underlying_process','target','scattering_regime'],axis=1, level=0) 
			cols = [f'{v[0]}_{v[1]}' if v[1] else f'{v[0]}' for v in self.df_for_numpy.columns.values]
			self.array_gen = self.df_for_numpy.to_numpy(dtype=np.float64)

		np.save(f'{self.out_file_name}ndarray.npy', self.array_gen, **kwargs)
		prettyprinter.info(f"Events in numpy array saved to file successfully:\n{self.out_file_name}")
		return self.array_gen


	def print_events_to_parquet(self, **kwargs):
		""" 
			Print to pandas DataFrame to parquet file using pyarrow (.parquet) 

			This format cannot save df.attrs to file.

		"""
		# kwargs['engine']=kwargs.get('engine','pyarrow')
		
		if self.sparse:
			pq.write_table(pa.Table.from_pandas(self.df_sparse), f"{self.out_file_name}pandas_df.parquet", **kwargs)
			# self.df_sparse.to_parquet(f"{self.out_file_name}pandas_df.parquet", **kwargs)
			prettyprinter.info(f"Events in sparse pandas dataframe saved to parquet file successfully:\n{self.out_file_name}")
			return self.df_sparse
		else:
			pq.write_table(pa.Table.from_pandas(self.df_gen), f"{self.out_file_name}pandas_df.parquet", **kwargs)
			# self.df_gen.to_parquet(f"{self.out_file_name}pandas_df.parquet", **kwargs)
			prettyprinter.info(f"Events in pandas dataframe saved to parquet file successfully:\n{self.out_file_name}")
			return self.df_gen

	def print_events_to_pandas(self, **kwargs):
		""" 
			Print to pandas DataFrame pickle file (.pckl)

			This is the only format that allows to save df.attrs to file.
			Using Dill to serialize the Model, Detector, and NuclearTarget classes to file.

		"""
		if self.sparse:
			dill.dump(self.df_sparse, open(f'{self.out_file_name}pandas_df.pckl', 'wb'), **kwargs)
			prettyprinter.info(f"Events in sparse pandas dataframe saved to file successfully:\n{self.out_file_name}")
			return self.df_sparse
		else:
			# pickles DarkNews classes with support for lambda functions
			dill.dump(self.df_gen, open(f'{self.out_file_name}pandas_df.pckl', 'wb'), **kwargs)
			prettyprinter.info(f"Events in pandas dataframe saved to file successfully:\n{self.out_file_name}")
			return self.df_gen




	def get_unweighted_events(self, nevents, prob_col = 'w_event_rate', **kwargs):
		'''
			Unweigh events in dataframe down to "nevents" using accept-reject method with the weights in "prob_col" of dataframe.
				
			Fails if nevents is smaller than the total number of unweighted events.

			kwargs passed to numpy's random.choice.
		'''
		logger.info(f"Unweighing events down to {nevents} entries.")
		prob = self.df_gen[prob_col]/np.sum(self.df_gen[prob_col])
		if (prob < 0).any():
			logger.error(f"ERROR! Probabily for unweighing contains negative values! Bad weights? {prob_col} < 0.")
		if (prob == 0).any():
			logger.warning(f"WARNING! Discarding zero-valued weights for unweighing. Total of {sum(prob == 0)} of {len(prob)} zero entries for {prob_col}.")
			AccEntries = np.random.choice(self.df_gen.index[prob>0], size=nevents, p=prob[prob>0], *kwargs)
		else:
			AccEntries = np.random.choice(self.df_gen.index, size=nevents, p=prob, *kwargs)

		# extract selected entries
		return self.df_gen.filter(AccEntries, axis=0).reset_index()





	def print_events_to_HEPEVT(self, unweigh=False, max_events=100, sparse=False, decay_product='e+e-'):
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
			df_gen = self.get_unweighted_events(self.df_gen, nevents = max_events)
		else:
			df_gen = self.df_gen

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

		# string to be saved to file
		hepevt_string = ''
		hepevt_string += f"{tot_events_to_print}\n"

		projectile_flavor = int(lp.nu_mu.pdgid)
		if decay_product == 'e+e-':
			id_lepton_minus = int(lp.e_minus.pdgid)
			id_lepton_plus = int(lp.e_plus.pdgid)
		elif decay_product == 'mu+mu-':
			id_lepton_minus = int(lp.mu_minus.pdgid)
			id_lepton_plus = int(lp.mu_plus.pdgid)
		else:
			id_lepton_minus = int(lp.e_minus.pdgid)
			id_lepton_plus = int(lp.e_plus.pdgid)
			logger.warning(f'Decay product {decay_product} not recognized, assuming it to be e+e-.')

		lines=[]
		# loop over events
		for i in df_gen.index:
			
			# no particles & event id
			if unweigh:
				lines.append(f"{i} 7\n")
			else:
				lines.append(f"{i} 7 {df_gen['w_event_rate',''].to_numpy()[i]:.8g}\n")

			# scattering inital states
			lines.append((	# Projectile
						f"0 "
						f" {projectile_flavor}"
						f" 0 0 0 0"
						f" {print_in_order(pvec_projectile[i])}"
						f" {df_gen['P_projectile','0'].to_numpy()[i]:.8g}"
						f" {mass_projectile[i]:.8g}"
						f" {print_in_order(pvec_pos_scatt[i])}"
						f" {df_gen['pos_scatt','0'].to_numpy()[i]:.8g}"
						"\n"
						))
						
			lines.append((	# Target
						f"0 "
						f" {int(df_gen['target_pdgid',''].to_numpy()[i])}"
						f" 0 0 0 0"
						f" {print_in_order(pvec_target[i])}"
						f" {df_gen['P_recoil','0'].to_numpy()[i]:.8g}"
						f" {mass_target[i]:.8g}"
						f" {print_in_order(pvec_pos_scatt[i])}"
						f" {df_gen['pos_scatt','0'].to_numpy()[i]:.8g}"
						"\n"
						))

			# scatter final products
			lines.append((	# HNL produced
						f"0 "
						f" {int(pdg.neutrino5.pdgid)}"
						f" 0 0 0 0"
						f" {print_in_order(pvec_decay_N_parent[i])}"
						f" {df_gen['P_decay_N_parent','0'].to_numpy()[i]:.8g}"
						f" {mass_decay_N_parent[i]:.8g}"
						f" {print_in_order(pvec_pos_scatt[i])}"
						f" {df_gen['pos_scatt','0'].to_numpy()[i]:.8g}"
						"\n"
						))

			lines.append((	# recoiled target
						f"0 "
						f" {int(df_gen['target_pdgid',''].to_numpy()[i])}"
						f" 0 0 0 0"
						f" {print_in_order(pvec_recoil[i])}"
						f" {df_gen['P_recoil','0'].to_numpy()[i]:.8g}"
						f" {mass_recoil[i]:.8g}"
						f" {print_in_order(pvec_pos_scatt[i])}"
						f" {df_gen['pos_scatt','0'].to_numpy()[i]:.8g}"
						'\n'
						))

			# decay final products
			lines.append((	# daughter neutrino/HNL
						f"0 "
						f" {int(pdg.nulight.pdgid)}"
						f" 0 0 0 0"
						f" {print_in_order(pvec_decay_N_daughter[i])}"
						f" {df_gen['P_decay_N_daughter','0'].to_numpy()[i]:.8g}"
						f" {mass_decay_N_daughter[i]:.8g}"
						f" {print_in_order(pvec_pos_decay[i])}"
						f" {df_gen['pos_decay','0'].to_numpy()[i]:.8g}"
						'\n'
						))

			lines.append((	# electron
						f"1 "
						f" {id_lepton_minus}"
						f" 0 0 0 0"
						f" {print_in_order(pvec_decay_ell_minus[i])}"
						f" {df_gen['P_decay_ell_minus','0'].to_numpy()[i]:.8g}"
						f" {const.m_e:.8g}"
						f" {print_in_order(pvec_pos_decay[i])}"
						f" {df_gen['pos_decay','0'].to_numpy()[i]:.8g}"
						"\n"
						))

			lines.append((	# positron
						f"1 "
						f" {id_lepton_plus}"
						f" 0 0 0 0"
						f" {print_in_order(pvec_decay_ell_plus[i])}"
						f" {df_gen['P_decay_ell_plus','0'].to_numpy()[i]:.8g}"
						f" {const.m_e:.8g}"
						f" {print_in_order(pvec_pos_decay[i])}"
						f" {df_gen['pos_decay','0'].to_numpy()[i]:.8g}"
						"\n"
						))

		
		# HEPevt file name
		hepevt_file_name = f"{df_gen.attrs['data_path']}HEPevt.dat"
		f = open(hepevt_file_name,"w+") 
		# print events
		f.write(hepevt_string.join(lines))
		f.close()

		prettyprinter.info(f"HEPevt events saved to file successfully:\n{hepevt_file_name}")


