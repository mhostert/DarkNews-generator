import os
import sys
import numpy as np
import pandas as pd

from . import const
from . import pdg
from dark_news.decayer import decay_position
#CYTHON
import pyximport
pyximport.install(
    language_level=3,
    pyimport=False,
    )
from . import Cfourvec as Cfv

def print_events_to_pandas(PATH_data, df_gen, TOT_EVENTS, bsm_model, l_decay_proper=0.0):
	# events
	pN   = df_gen['P3']
	pnu   = df_gen['P2_decay']
	pZ   = df_gen['P3_decay']+df_gen['P4_decay']
	plm  = df_gen['P3_decay']
	plp  = df_gen['P4_decay']
	pHad = df_gen['P4']
	w = df_gen['w']
	I = df_gen['I']
	regime = df_gen['flags']

	t_decay, x_decay, y_decay, z_decay = decay_position(pN, l_decay_proper)

	size = np.shape(plm)[0]

	# Create target Directory if it doesn't exist
	if not os.path.exists(PATH_data):
	    os.makedirs(PATH_data)
	if PATH_data[-1] != '/':
		PATH_data += '/'
	out_file_name = PATH_data+f"MC_m4_{bsm_model.m4:.8g}_mzprime_{bsm_model.mzprime:.8g}.pckl"
	###############################################
	# SAVE ALL EVENTS AS A PANDAS DATAFRAME
	columns = [['plm', 'plp', 'pnu', 'pHad', 'decay_point'], ['t', 'x', 'y', 'z']]
	columns_index = pd.MultiIndex.from_product(columns)
	aux_data = [plm[:, 0],
			plm[:, 1],
			plm[:, 2],
			plm[:, 3],
			plp[:, 0],
			plp[:, 1],
			plp[:, 2],
			plp[:, 3],
			pnu[:, 0],
			pnu[:, 1],
			pnu[:, 2],
			pnu[:, 3],
			pHad[:, 0],
			pHad[:, 1],
			pHad[:, 2],
			pHad[:, 3],
			t_decay,
			x_decay,
			y_decay,
			z_decay,]
	
	aux_df = pd.DataFrame(np.stack(aux_data, axis=-1), columns=columns_index)
	aux_df.loc[:, 'weight'] = w
	aux_df.loc[:, 'regime'] = regime

	print(out_file_name)
	aux_df.to_pickle(out_file_name)


def print_unweighted_events_to_HEPEVT(PATH_data, df_gen, TOT_EVENTS, bsm_model, l_decay_proper=0.0):

	# events
	pN   = df_gen['P3']
	pnu   = df_gen['P2_decay']
	pZ   = df_gen['P3_decay']+df_gen['P4_decay']
	plm  = df_gen['P3_decay']
	plp  = df_gen['P4_decay']
	pHad = df_gen['P4']
	w = df_gen['w']
	I = df_gen['I']
	regime = df_gen['flags']

	t_decay, x_decay, y_decay, z_decay = decay_position(pN, l_decay_proper)

	# Accept/reject method -- samples distributed according to their weights
	AllEntries = np.array(range(np.shape(plm)[0]))
	AccEntries = np.random.choice(AllEntries, size=TOT_EVENTS, replace=True, p=w/np.sum(w))
	pN, plp, plm, pnu, pHad, w, regime  = pN[AccEntries], plp[AccEntries], plm[AccEntries], pnu[AccEntries], pHad[AccEntries], w[AccEntries], regime[AccEntries]

	size = np.shape(plm)[0]

	# decay the heavy nu
	M4 = np.sqrt(Cfv.dot4(pN,pN))
	mzprime = np.sqrt(Cfv.dot4(pZ,pZ))
	Mhad = np.sqrt(Cfv.dot4(pHad,pHad))
	gammabeta_inv = M4/(np.sqrt(pN[:,0]**2 -  M4*M4 ))
	######################
	# *PROPER* decay length -- BY HAND AT THE MOMENT!
	ctau = l_decay_proper
	######################
	d_decay = np.random.exponential(scale=ctau/gammabeta_inv)*1e2 # centimeters

	###############################################
	# SAVE ALL EVENTS AS A HEPEVT .dat file
	hepevt_file_name = PATH_data+"MC_m4_"+format(bsm_model.m4,'.8g')+"_mzprime_"+format(bsm_model.mzprime,'.8g')+".dat"
	# Open file in write mode
	f = open(hepevt_file_name,"w+") 
	
	# f.write("%i\n",TOT_EVENTS)
	# loop over events
	for i in range(TOT_EVENTS):
		f.write("%i 4\n" % i)
		f.write("1 %i 0 0 0 0 %f %f %f %f %f %f %f %f %f\n"%(pdg.electron,plm[i][1], plm[i][2], plm[i][3], plm[i][0], const.m_e, x_decay[i], y_decay[i], z_decay[i],t_decay[i]))
		f.write("1 %i 0 0 0 0 %f %f %f %f %f %f %f %f %f\n"%(pdg.positron,plp[i][1], plp[i][2], plp[i][3], plp[i][0], const.m_e, x_decay[i], y_decay[i], z_decay[i],t_decay[i]))	
		f.write("2 %i 0 0 0 0 %f %f %f %f %f %f %f %f %f\n"%(pdg.numu,pnu[i][1], pnu[i][2], pnu[i][3], pnu[i][0], const.m_e, x_decay[i], y_decay[i], z_decay[i], t_decay[i]))
		if (regime[i] == const.COHRH or regime[i] == const.COHLH):
			f.write("1 %i 0 0 0 0 %f %f %f %f %f %f %f %f %f\n"%(pdg.Argon40,pHad[i][1], pHad[i][2], pHad[i][3], pHad[i][0], Mhad[i],x_decay[i],y_decay[i],z_decay[i],t_decay[i]))
		elif (regime[i] == const.DIFRH or regime[i] == const.DIFLH):
			f.write("1 %i 0 0 0 0 %f %f %f %f %f %f %f %f %f\n"%(pdg.proton,pHad[i][1], pHad[i][2], pHad[i][3], pHad[i][0], Mhad[i],x_decay[i],y_decay[i],z_decay[i],t_decay[i]))
		else:
			print('Error! Cannot find regime of event ', i)
	f.close()