import numpy as np
import pandas as pd
import vegas as vg

from DarkNews import logger, prettyprinter

from collections import defaultdict
from functools import partial

from . import model 
from . import integrands

from . import const
from . import pdg
from . import geom

NINT = 10
NEVAL = 1000
NINT_warmup = 10
NEVAL_warmup = 1000


class MC_events:
    """ MC events generated with importance sampling (vegas)

    Correct weights are computed from cross-section, decay width, 
    and experimental considerations

    Args:
        experiment:             instance of Detector class
        target:                 scattering target class 
        scattering_regime:      regime for scattering process, e.g. coherent, p-el, (future: RES, DIS) 
        nu_projectile:          incoming neutrino flavor
        nu_upscattered:         intermediate dark NU in upscattering process
        nu_outgoing:            outgoing neutrino flavor
        decay_product:          visible decay products in the detector
        helicity:               helicity of the up-scattered neutrino
    """

    def __init__(self,
                experiment, 
                **kwargs):

        # default parameters
        scope ={
            'nu_projectile': pdg.numu,
            'nu_upscattered': pdg.neutrino4,
            'nu_outgoing': pdg.nulight,
            'scattering_regime': 'coherent',
            'decay_product': ['e+e-'],
            'helicity': 'conserving'
            }

        scope.update(kwargs)
        self.scope = scope
        self.experiment = experiment

        # set target properties for this scattering regime
        self.nuclear_target = scope['nuclear_target']
        if scope['scattering_regime'] == 'coherent':
            self.target = self.nuclear_target
        elif scope['scattering_regime'] == 'p-el':
            self.target = self.nuclear_target.get_constituent_nucleon('proton')
        elif scope['scattering_regime'] == 'n-el':
            self.target = self.nuclear_target.get_constituent_nucleon('neutron')
        else:
            logger.error(f"Scattering regime {scope['scattering_regime']} not supported.")

        self.MA = self.target.mass 

        # identifiers for particles in the process 
        self.nu_projectile = scope['nu_projectile']
        self.nu_upscattered = scope['nu_upscattered']
        self.nu_outgoing = scope['nu_outgoing']
        self.helicity = scope['helicity']

        if scope['decay_product'] == 'e+e-':
            self.decay_product = pdg.electron
            # process being considered
            DECAY_PRODUCTS = f"{self.decay_product.invert().name} + {self.decay_product.name}"
            self.decays_to_dilepton = True
            self.decays_to_singlephoton = False

        elif scope['decay_product'] == 'mu+mu-':
            self.decay_product = pdg.muon
            # process being considered
            DECAY_PRODUCTS = f"{self.decay_product.invert().name} + {self.decay_product.name}"
            self.decays_to_dilepton = True
            self.decays_to_singlephoton = False

        elif scope['decay_product'] == 'photon':
            self.decay_product = pdg.photon
            DECAY_PRODUCTS = f"{self.decay_product.name}"
            self.decays_to_dilepton = False
            self.decays_to_singlephoton = True

        else:
            logger.error(f"Error! Could not find decay product: {scope['decay_product']}")
            raise ValueError

        # process being considered
        self.underl_process_name = f'{self.nu_projectile.name} + {self.target.name} -> {self.nu_upscattered.name} +  {self.target.name} -> {self.nu_outgoing.name} + {DECAY_PRODUCTS} + {self.target.name}'
    


    def set_theory_params(self, bsm_model):
        """ 
            Sets the theory parameters for the current process

            Also defines upscattering and decay objects

        """
        self.bsm_model = bsm_model
        # scope for upscattering process
        self.ups_case = model.UpscatteringProcess(nu_projectile=self.nu_projectile, 
                                                nu_upscattered=self.nu_upscattered,
                                                target=self.target,
                                                helicity=self.helicity,
                                                TheoryModel=bsm_model)
        if self.decays_to_dilepton:
            # scope for decay process
            self.decay_case = model.FermionDileptonDecay(nu_parent=self.nu_upscattered,
                                                    nu_daughter=self.nu_outgoing,
                                                    final_lepton1 = self.decay_product, 
                                                    final_lepton2 = self.decay_product,
                                                    h_parent = self.ups_case.h_upscattered,
                                                    TheoryModel=bsm_model)
        elif self.decays_to_singlephoton:
            # scope for decay process
            self.decay_case = model.FermionSinglePhotonDecay(nu_parent=self.nu_upscattered,
                                                    nu_daughter=self.nu_outgoing,
                                                    h_parent = self.ups_case.h_upscattered,
                                                    TheoryModel=bsm_model)
        else:
            logger.error("Error! Could not determine what type of decay class to use.")
            raise ValueError

        self.Ethreshold = self.ups_case.m_ups**2 / 2.0 / self.MA + self.ups_case.m_ups

        if self.Ethreshold > self.experiment.ERANGE[-1]:
            logger.error(f"Particle {self.nu_upscattered.name} is too heavy to be produced in the energy range of experiment {self.experiment.NAME}")
            raise ValueError


        if (self.ups_case.m_ups != self.decay_case.m_parent):
            logger.error(f"Error! Mass of HNL produced in neutrino scattering, m_ups = {self.ups_case.m_upscattered} GeV, different from that of parent HNL, m_parent = { self.decay_case.m_parent} GeV.")
            raise ValueError


    def get_total_xsec(self, Enu):
        """ 
            Returns the total upscattering xsec for a fixed neutrino energy

        """

        DIM = 1
        self.Enu = Enu

        # below threshold
        if Enu < (self.Ethreshold):
            return 0.0

        batch_f = integrands.UpscatteringXsec(dim=DIM, Enu=self.Enu, MC_case=self)
        integ = vg.Integrator(DIM*[[0.0, 1.0]]) # unit hypercube
        
        integrals = run_vegas(batch_f, integ, NINT=NINT, NEVAL=NEVAL, NINT_warmup=NINT_warmup, NEVAL_warmup=NEVAL_warmup)
        logger.debug(f"Main VEGAS run completed.")

        #############
        # integrated xsec coverted to cm^2
        tot_xsec = integrals['diff_xsec'].mean*const.attobarn_to_cm2*batch_f.norm['diff_xsec']
  
        # How many constituent targets inside scattering regime? 
        if self.scope['scattering_regime'] == 'coherent':
            target_multiplicity = 1
        elif self.scope['scattering_regime'] == 'p-el':
            target_multiplicity = self.nuclear_target.Z
        elif self.scope['scattering_regime'] == 'n-el':
            target_multiplicity = self.nuclear_target.N
        else:
            logger.error(f"Scattering regime {self.scope['scattering_regime']} not supported.")

        logger.debug(f"Total cross section calculated.")
        return tot_xsec*target_multiplicity


    def get_MC_events(self):
        """ 
            Returns MC events from importance sampling
            
            Handles the off and on shell cases separately

            After sampling the integrands, we normalize the integral by
            the integral of the decay rates, which is also estimated by VEGAS
            in the multi-component integrand. 

            The first integrand entry is the one VEGAS uses to optimize the importance sampling.
        """

        prettyprinter.info(f"{self.underl_process_name}")
        logger.info(f"Helicity {self.helicity} upscattering.")
        #########################################3
        # Some experimental definitions
        #self.exp = experiment  # NO NEED TO STORE THIS
        self.flux = self.experiment.neutrino_flux(self.nu_projectile)
        self.EMIN = max(self.experiment.ERANGE[0], 1.05 * self.Ethreshold)
        self.EMAX = self.experiment.ERANGE[1]


        if self.decays_to_dilepton:

            if self.decay_case.on_shell:
                DIM = 3
                logger.info(f"{self.nu_upscattered.name} decays via on-shell Z'.")
            elif self.decay_case.off_shell:
                DIM = 6
                logger.info(f"{self.nu_upscattered.name} decays via off-shell Z'.")
        elif self.decays_to_singlephoton:
            DIM = 3
            logger.info(f"{self.nu_upscattered.name} decays via TMM.")
        else:
            logger.error(f"ERROR! Could not find decay process.")

        integrand_type = integrands.UpscatteringHNLDecay
        

        #########################################################################
        # BATCH SAMPLE INTEGRAND OF INTEREST
        batch_f = integrand_type(dim=DIM, Emin=self.EMIN, Emax=self.EMAX, MC_case=self)
        integ = vg.Integrator(DIM*[[0.0, 1.0]]) # unit hypercube
        
        result = run_vegas(batch_f, integ, NINT=NINT, NEVAL=NEVAL, NINT_warmup=NINT_warmup, NEVAL_warmup=NEVAL_warmup)
        logger.debug(f"Main VEGAS run completed.")

        logger.debug(f"Vegas results for the integrals: {result.summary()}")
        ##########################################################################


        #########################################################################
        # GET THE INTEGRATION SAMPLES and translate to physical variables in MC events
        samples, weights = get_samples(integ, batch_f)
        tot_nevents = len(weights['diff_event_rate'])

        logger.debug(f"Normalization factors in MC: {batch_f.norm}.")
        logger.debug(f"Vegas results for diff_event_rate: {np.sum(weights['diff_event_rate'])}")
        logger.debug(f"Vegas results for diff_flux_avg_xsec: {np.sum(weights['diff_flux_avg_xsec'])}")

        four_momenta = integrands.get_momenta_from_vegas_samples(samples, MC_case=self)
        ##########################################################################

        ##########################################################################
        # SAVE ALL EVENTS AS A PANDAS DATAFRAME
        particles = list(four_momenta.keys())
        columns = particles + ['pos_scatt']
        indexing = [columns, ['0','1','2','3']]
        columns_index = pd.MultiIndex.from_product(indexing)
        
        # create auxiliary data list
        aux_data = []
        # create pandas dataframe
        for p in particles:
            for component in range(4):
                aux_data.append(four_momenta[p][:,component])
        # scattering position
        aux_data.append(np.zeros((tot_nevents, )))
        aux_data.append(np.zeros((tot_nevents, )))
        aux_data.append(np.zeros((tot_nevents, )))
        aux_data.append(np.zeros((tot_nevents, )))
        
        df_gen = pd.DataFrame(np.stack(aux_data, axis=-1), columns=columns_index)
              
        # differential weights
        for column in df_gen:
            if ("w_" in column):
                df_gen[column, ''] = df_gen[column]

        # Normalize weights and total integral with decay rates and set units to nus*cm^2/POT
        decay_rates = 1
        for decay_step in (k for k in batch_f.int_dic.keys() if 'decay_rate' in k):
            logger.debug(f"Vegas results for {decay_step}: {np.sum(weights[decay_step])}")
            
            # saving decay weights and integrals
            df_gen[f'w_{decay_step}'.replace('diff_','')] = weights[decay_step] * batch_f.norm[decay_step]
            
            # combining all decay rates into one factor
            decay_rates *= np.sum(df_gen[f'w_{decay_step}'.replace('diff_','')]) 

        # How many constituent targets inside scattering regime? 
        if self.scope['scattering_regime'] == 'coherent':
            target_multiplicity = 1
        elif self.scope['scattering_regime'] == 'p-el':
            target_multiplicity = self.nuclear_target.Z
        elif self.scope['scattering_regime'] == 'n-el':
            target_multiplicity = self.nuclear_target.N
        else:
            logger.error(f"Scattering regime {self.scope['scattering_regime']} not supported.")

        # Normalize to total exposure
        exposure = self.experiment.NUMBER_OF_TARGETS[self.nuclear_target.name]*self.experiment.POTS

        # differential rate weights
        df_gen['w_event_rate'] = weights['diff_event_rate']*const.attobarn_to_cm2/decay_rates*target_multiplicity*exposure*batch_f.norm['diff_event_rate']

        # flux averaged xsec weights (neglecting kinematics of decay)
        df_gen['w_flux_avg_xsec'] = weights['diff_flux_avg_xsec']*const.attobarn_to_cm2*target_multiplicity*exposure*batch_f.norm['diff_flux_avg_xsec']

        df_gen['target']       = np.full(tot_nevents, self.target.name)
        df_gen['target_pdgid'] = np.full(tot_nevents, self.target.pdgid)

        df_gen['scattering_regime']  = np.full(tot_nevents, self.scope['scattering_regime'])
        df_gen['helicity']           = np.full(tot_nevents, self.helicity)
        df_gen['underlying_process'] = np.full(tot_nevents, self.underl_process_name)

        # saving the experiment class
        df_gen.attrs['experiment'] = self.experiment

        # saving the bsm_model class
        df_gen.attrs['bsm_model'] = self.bsm_model



        ##########################################################################
        # PROPAGATE PARENT PARTICLE
        
        self.experiment.set_geometry()
        self.experiment.place_scatters(df_gen)

        geom.place_decay(df_gen, 'P_decay_N_parent', l_decay_proper_cm=0.0, label='pos_decay')

        ##########################################################################

        # print final result
        logger.info(f"Predicted ({np.sum(df_gen['w_event_rate']):.3g} +/- {np.sqrt(np.sum(df_gen['w_event_rate']**2)):.3g}) events.\n")

        return df_gen


def run_MC(bsm_model, experiment, **kwargs):
    """ Create MC_events objects and run the MC computations

    Args:
        bsm_model:              physics parameters
        experiment:             instance of Detector object
        **kwargs:
            FLAVORS (list):         input flavors 
            UPSCATTERED_NUS (list): dark NU in upscattering process
            OUTFOING_NUS (list):    output flavors 
            SCATTERING_REGIMES (list):    regimes of scattering process (coherent, p-el, n-el)
            INCLUDE_HC (bool):      flag to include helicity conserving case
            INCLUDE_HF (bool):      flag to include helicity flipping case
            NO_COH (bool):          flag to skip coherent case
            NO_PELASTIC (bool):     flag to skip proton elastic case
            DECAY_PRODUCTS (list): decay processes to include
    """
    
    # Default values
    scope = {
    'FLAVORS': [pdg.numu],
    'UPSCATTERED_NUS': [pdg.neutrino4],
    'OUTGOING_NUS': [pdg.nulight],
    'SCATTERING_REGIMES': ['coherent','p-el'],
    'INCLUDE_HC': True,
    'INCLUDE_HF': True,
    'NO_COH': False,
    'NO_PELASTIC': False,
    'DECAY_PRODUCTS': ['e+e-']
    }
    scope.update(kwargs)

    # create instances of all MC cases of interest
    logger.debug(f"Creating instances of MC cases:")
    gen_cases = []
    # neutrino flavor initiating scattering
    for flavor in scope['FLAVORS']:
        # neutrino produced in upscattering
        for nu_upscattered in scope['UPSCATTERED_NUS']:
            #neutrino produced in the subsequent decay
            for nu_outgoing in scope['OUTGOING_NUS']:
                # final state to consider in decay process
                for decay_product in scope['DECAY_PRODUCTS']:
                    # skip cases with obviously forbidden decay 
                    if np.abs(nu_outgoing.pdgid) >= np.abs(nu_upscattered.pdgid):
                        continue
                    # material on which upscattering happened
                    for nuclear_target in experiment.NUCLEAR_TARGETS:
                        # scattering regime to use
                        for scattering_regime in scope['SCATTERING_REGIMES']:
                            # skip disallowed regimes
                            if (
                                    ( (scattering_regime in ['n-el']) and (nuclear_target.N < 1)) # no neutrons
                                    |
                                    ( (scattering_regime in ['coherent']) and (not nuclear_target.is_nucleus)) # coherent = p-el for hydrogen
                                ):
                                continue 
                            elif ( (scattering_regime in ['coherent'] and scope['NO_COH'])
                                 | 
                                    (scattering_regime in ['p-el'] and scope['NO_PELASTIC'])
                                ):
                                continue
                            else:
                                # bundle arguments of MC_events here
                                args = {'nuclear_target' : nuclear_target,
                                        'scattering_regime' : scattering_regime,
                                        'nu_projectile' : flavor, 
                                        'nu_upscattered' : nu_upscattered,
                                        'nu_outgoing' : nu_outgoing, 
                                        'decay_product' : decay_product,
                                        }

                                if scope['INCLUDE_HC']:  # helicity conserving scattering
                                    gen_cases.append(MC_events(experiment, **args, helicity = 'conserving'))

                                if scope['INCLUDE_HF']:  # helicity flipping scattering
                                    gen_cases.append(MC_events(experiment, **args, helicity = 'flipping'))

                                logger.debug(f"Created an MC instance of {gen_cases[-1].underl_process_name}.")

    
    logger.debug(f"Now running the generator for each instance.")
    # Set theory params and run generation of events
    gen_cases[0].set_theory_params(bsm_model)
    gen_cases_dfs = gen_cases[0].get_MC_events()
    for mc in gen_cases[1:]:
        mc.set_theory_params(bsm_model)
        merge_MC_output(gen_cases_dfs, mc.get_MC_events())
    
    prettyprinter.info(f"* Generation successful\n\nTotal events predicted:\n({np.sum(gen_cases_dfs['w_event_rate']):.3g} +/- {np.sqrt(np.sum(gen_cases_dfs['w_event_rate']**2)):.3g}) events.")

    return gen_cases_dfs


# merge all generation cases into one dictionary
def merge_MC_output(df1,df2):
    
    logger.debug(f"Appending {df2.underlying_process[0]}")
    df = pd.concat([df1, df2], axis = 0).reset_index()    
    
    return df


def get_samples(integ, batch_integrand):
    '''    
        Accesses integration samples for a single iteration as in vegas/_vegas.pyx
        
        Args:  
            integ:              Vegas integrator object initialized by the user.
            batch_integrand:    Vegas batch_integrand object created by the user. These are defined in integrands.py

    '''
    unit_samples = batch_integrand.dim*[[]]
    weights = defaultdict(partial(np.ndarray,0))

    for x, y, wgt in integ.random_batch(yield_y=True, fcn=batch_integrand):

        # compute integrand on samples including jacobian factors
        if integ.uses_jac:
            fx = batch_integrand(x, jac=integ.map.jac1d(y))
        else:
            fx = batch_integrand(x, jac=None)

        # weights
        for fx_i in fx.keys():
            if np.any(np.isnan(fx[fx_i])):
                raise ValueError(f'integrand {fx_i} evaluates to nan')
            weights[fx_i] = np.append(weights[fx_i], wgt*fx[fx_i])
        
        # MC samples in unit hypercube
        for i in range(batch_integrand.dim):
            unit_samples[i] = np.append(unit_samples[i], x[:,i])

    return np.array(unit_samples), weights

def run_vegas(batch_f, integ, NINT=10, NEVAL=1000, NINT_warmup=10, NEVAL_warmup=1000):
        
        # warm up the MC, adapting to the integrand
        integ(batch_f, nitn=NINT_warmup, neval=NEVAL_warmup, uses_jac=True)
        logger.debug(f"VEGAS warm-up completed.")

        # sample again, now saving result and turning off further adaption
        return integ(batch_f,  nitn = NINT, neval = NEVAL, uses_jac=True, adapt=False)
