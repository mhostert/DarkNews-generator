import numpy as np
import vegas as vg

from DarkNews import logger, prettyprinter


from collections import defaultdict
from functools import partial

#CYTHON
import pyximport
pyximport.install(
    language_level=3,
    pyimport=False,
    )

from . import decay_rates
from . import model 
from . import integrands

from . import const
from . import pdg
from . import detector

NINT = 10
NEVAL = 1000
NINT_warmup = 10
NEVAL_warmup = 1000


class MC_events:
    """ MC events generated with importance sampling (vegas)

    Correct weights are computed from cross-section, decay width, 
    and experimental considerations

    Args:
        experiment:     instance of Detector class
        target:                 scattering target class 
        scattering_regime:      regime for scattering process, e.g. coherent, p-el, (future: RES, DIS) 
        nu_projectile:          incoming neutrino flavor
        nu_upscattered:         intermediate dark NU in upscattering process
        nu_outgoing:            outgoing neutrino flavor
        final_lepton:           final lepton detected in detector
        helicity:          helicity of the up-scattered neutrino
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
            'final_lepton': pdg.electron,
            'helicity': 'conserving' # helicity
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
        self.final_lepton = scope['final_lepton']
        self.nu_projectile = scope['nu_projectile']
        self.nu_upscattered = scope['nu_upscattered']
        self.nu_outgoing = scope['nu_outgoing']
        self.helicity = scope['helicity']

        # process being considered
        self.underl_process_name = f'{self.nu_projectile.name} + {self.target.name} -> {self.nu_upscattered.name} +  {self.target.name} -> {self.nu_outgoing.name} + {self.final_lepton.invert().name} + {self.final_lepton.name} + {self.target.name}'

    def set_theory_params(self, params):
        """ 
            Sets the theory parameters for the current process

            Also defines upscattering and decay objects

        """
        # Theory parameters
        self.params = params
        
        # scope for upscattering process
        self.ups_case = model.UpscatteringProcess(nu_projectile=self.nu_projectile, 
                                                nu_upscattered=self.nu_upscattered,
                                                target=self.target,
                                                helicity=self.helicity,
                                                TheoryModel=params)
        # scope for upscattering process
        self.decay_case = model.FermionDecayProcess(nu_parent=self.nu_upscattered,
                                                nu_daughter=self.nu_outgoing,
                                                final_lepton1 = self.final_lepton, 
                                                final_lepton2 = self.final_lepton,
                                                h_parent = self.ups_case.h_upscattered,
                                                TheoryModel=params)
        
        self.Ethreshold = self.ups_case.m_ups**2 / 2.0 / self.MA + self.ups_case.m_ups

        if self.Ethreshold > self.experiment.ERANGE[-1]:
            logger.error(f"Particle {self.nu_upscattered.name} is too heavy to be produced in the energy range of experiment {self.experiment.NAME}")
            raise ValueError


        if (self.ups_case.m_ups != self.decay_case.m_parent):
            logger.error(f"Error! Mass of HNL produced in neutrino scattering, m_ups = {ups_case.m_upscattered} GeV, different from that of parent HNL, m_parent = { decay_case.m_parent} GeV.")
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
            logger.error(f"Scattering regime {scope['scattering_regime']} not supported.")

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

        logger.info(f"Generating helicity {self.helicity} upscattering events for:\n\t{self.underl_process_name}\n")
        #########################################3
        # Some experimental definitions
        #self.exp = experiment  # NO NEED TO STORE THIS
        self.flux = self.experiment.get_flux_func(flavor=self.nu_projectile)
        self.EMIN = max(self.experiment.ERANGE[0], 1.05 * self.Ethreshold)
        self.EMAX = self.experiment.ERANGE[1]


        if self.decay_case.on_shell:
            DIM = 3
            logger.info(f"decaying {self.nu_upscattered.name} using on-shell mediator.")
        else:  
            DIM = 6
            logger.info(f"decaying {self.nu_upscattered.name} using off-shell mediator.")
        
        integrand_type = integrands.UpscatteringHNLDecay
        

        #########################################################################
        # BATCH SAMPLE INTEGRAND OF INTEREST
        batch_f = integrand_type(dim=DIM, Emin=self.EMIN, Emax=self.EMAX, MC_case=self)
        integ = vg.Integrator(DIM*[[0.0, 1.0]]) # unit hypercube
        
        result = run_vegas(batch_f, integ, NINT=NINT, NEVAL=NEVAL, NINT_warmup=NINT_warmup, NEVAL_warmup=NEVAL_warmup)
        logger.debug(f"Main VEGAS run completed.")

        integrals = dict(result)
        logger.debug(f"Vegas results for the integrals: {result.summary()}")
        ##########################################################################


        #########################################################################
        # GET THE INTEGRATION SAMPLES and translate to physical variables in MC events
        samples, weights = get_samples(DIM, integ, batch_f)
        logger.debug(f"Vegas results for diff_event_rate: {np.sum(weights['diff_event_rate'])}")
        logger.debug(f"Vegas results for diff_flux_avg_xsec: {np.sum(weights['diff_flux_avg_xsec'])}")

        if self.decay_case.on_shell:
            # decay_rates = result['diff_gammaN'].mean * result['gammaMediator'].mean*batch_f.NORM_NDECAY*batch_f.NORM_MDECAY
            get_four_momenta = integrands.get_four_momenta_from_vsamples_onshell
        else:   
            # decay_rates = result['diff_gammaN'].mean * batch_f.NORM_NDECAY
            get_four_momenta = integrands.get_four_momenta_from_vsamples_offshell
        
        events = get_four_momenta(samples, MC_case=self)
        ##########################################################################

        

        ##########################################################################
        # Normalize weights and total integral with decay rates and set units to nus*cm^2/POT
        decay_rates = 1
        for decay_step in (k for k in batch_f.int_dic.keys() if 'decay_rate' in k):
            logger.debug(f"Vegas results for {decay_step}: {np.sum(weights[decay_step])}")
            
            # combining all decay rates into one factor
            decay_rates *= integrals[decay_step].mean * batch_f.norm[decay_step]
            
            # saving decay weights and integrals
            events[f'w_{decay_step}'.replace('diff_','')] = weights[decay_step] * batch_f.norm[decay_step]
            events[f'I_{decay_step}'.replace('diff_','')] = integrals[decay_step].mean * batch_f.norm[decay_step]


        # How many constituent targets inside scattering regime? 
        if self.scope['scattering_regime'] == 'coherent':
            target_multiplicity = 1
        elif self.scope['scattering_regime'] == 'p-el':
            target_multiplicity = self.nuclear_target.Z
        elif self.scope['scattering_regime'] == 'n-el':
            target_multiplicity = self.nuclear_target.N
        else:
            logger.error(f"Scattering regime {scope['scattering_regime']} not supported.")

        # Normalize to total exposure
        exposure = self.experiment.NUMBER_OF_TARGETS[self.nuclear_target.name]*self.experiment.POTS


        # differential rate weights
        events['w_event_rate'] = weights['diff_event_rate']*const.attobarn_to_cm2/decay_rates*target_multiplicity*exposure*batch_f.norm['diff_event_rate']

        # flux averaged xsec weights (neglecting kinematics of decay)
        events['w_flux_avg_xsec'] = weights['diff_flux_avg_xsec']*const.attobarn_to_cm2*target_multiplicity*exposure*batch_f.norm['diff_flux_avg_xsec']

        events['target'] = np.full(np.size(events['w_event_rate']), self.target.name)
        events['target_pdgid'] = np.full(np.size(events['w_event_rate']), self.target.pdgid)

        regime = self.scope['scattering_regime']

        events['scattering_regime'] = np.full(np.size(events['w_event_rate']), regime)
        events['helicity'] = np.full(np.size(events['w_event_rate']), self.helicity)
        events['underlying_process'] = np.full(np.size(events['w_event_rate']), self.underl_process_name)
        logger.debug(f"Inspecting dataframe\nkeys of events dictionary = {events.keys()}.")

        return events


#############################3
# THIS FUNCTION NEEDS SOME OPTIMIZING... currently setting event flags by hand.
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
            INCLUDE_HC (bool):      flag to include helicity conserving terms
            INCLUDE_HF (bool):      flag to include helicity flipping terms
            INCLUDE_COH (bool):     flag to include coherent terms
            INCLUDE_PELASTIC (bool):flag to include proton elastic terms
    """
    
    # Default values
    scope = {
    'FLAVORS': [pdg.numu],
    'UPSCATTERED_NUS': [pdg.neutrino4],
    'OUTGOING_NUS': [pdg.nulight],
    'SCATTERING_REGIMES': ['coherent','p-el'],
    'INCLUDE_HC': True,
    'INCLUDE_HF': True,
    'INCLUDE_COH': True,
    'INCLUDE_PELASTIC': True}
    scope.update(kwargs)

    # create instances of all MC cases of interest
    gen_cases = []
    # neutrino flavor initiating scattering
    for flavor in scope['FLAVORS']:
        # neutrino produced in upscattering
        for nu_upscattered in scope['UPSCATTERED_NUS']:
            #neutrino produced in the subsequent decay
            for nu_outgoing in scope['OUTGOING_NUS']:
                # skip cases with obvious forbidden decay 
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

                        # bundle arguments of MC_events here
                        args = {'nuclear_target' : nuclear_target,
                                'scattering_regime' : scattering_regime,
                                'nu_projectile' : flavor, 
                                'nu_upscattered' : nu_upscattered,
                                'nu_outgoing' : nu_outgoing, 
                                'final_lepton' : pdg.electron,
                                }

                        if scope['INCLUDE_HC']:  # helicity conserving scattering
                            gen_cases.append(MC_events(experiment, **args, helicity = 'conserving'))

                        if scope['INCLUDE_HF']:  # helicity flipping scattering
                            gen_cases.append(MC_events(experiment, **args, helicity = 'flipping'))

    gen_cases_events=[]
    # Set theory params and run generation of events
    for mc in gen_cases:
        mc.set_theory_params(bsm_model)
        gen_cases_events.append(mc.get_MC_events())
    
    # Combine all cases into one object
    all_events = merge_MC_output(gen_cases_events)

    all_events['bsm_model'] = bsm_model
    all_events['experiment'] = experiment


    return all_events


# merge all generation cases into one dictionary
def merge_MC_output(cases):
    
    logger.debug(f"\n\nMerging MC events for {np.shape(cases)[0]} processes.")
    # merged dic
    dic ={}
    # initialize with first case
    for x in cases[0]:
        dic[x] = cases[0][x]
    
    # append all subsequent ones
    for i in range(1,np.shape(cases)[0]):
        for x in cases[0]:
            logger.debug(f"Merging {x} in case {i}.")
            
            # merge event kinematics and weights
            if hasattr(cases[i][x], "__len__"):
                dic[x] = np.array( np.append(dic[x], cases[i][x], axis=0) )
        
            # merge integrals
            elif np.isscalar(cases[i][x]):
        
                # correct over-counting of total decay rate 
                if "decay_rate" in x:
                    dic[x] += cases[i][x]/np.size(cases)
                else:
                    dic[x] += cases[i][x]
            else:
                logger.error(f"Could not merge variable {x}, in {cases[i][x]}.")
                raise ValueError

    logger.debug(f"MC outputs merged with {len(cases)} cases.")
    
    return dic



# get samples from VEGAS integration and their respective weights
def get_samples(DIM, integ, batch_f):
           
        unit_samples = [[] for i in range(DIM)]
        weights = defaultdict(partial(np.ndarray,0))

        for x, wgt in integ.random_batch():
            
            # weights
            for regime in batch_f(x).keys():
                weights[regime] = np.append(weights[regime], wgt*(batch_f(x)[regime]))
            
            # MC samples in unit hypercube
            for i in range(DIM):
                unit_samples[i] = np.concatenate((unit_samples[i],x[:,i]))

        return np.array(unit_samples), weights

def run_vegas(batch_f, integ, NINT=10, NEVAL=1000, NINT_warmup=10, NEVAL_warmup=1000):
        # warm up the MC
        integ(batch_f, nitn=NINT_warmup, neval=NEVAL_warmup)
        logger.debug(f"VEGAS warm-up completed.")

        # sample again, now saving result
        return integ(batch_f,  nitn = NINT, neval = NEVAL)

