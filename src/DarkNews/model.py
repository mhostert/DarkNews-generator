import numpy as np
import vegas as vg
import math

from DarkNews import logger, prettyprinter

from particle import literals as lp

from DarkNews import const 
from DarkNews import pdg
from DarkNews import integrands
from DarkNews import decay_rates as dr
from DarkNews import amplitudes as amps
from DarkNews import phase_space as ps

from DarkNews import MC

def create_3portal_HNL_model(**kwargs):

    bsm_model = ThreePortalModel()

    if 'gD' in kwargs:
        bsm_model.gD = 1.0
    elif 'alphaD' in kwargs:
        bsm_model.gD = np.sqrt(4*np.pi*kwargs['alphaD'])
    
    
    if 'epsilon' in kwargs:
        bsm_model.epsilon = kwargs['epsilon']
    elif 'epsilon2' in kwargs:
        bsm_model.epsilon = np.sqrt(kwargs['epsilon2'])
    elif 'chi' in kwargs:
        bsm_model.epsilon = kwargs['chi']*const.cw
    elif 'alpha_epsilon2' in kwargs:
        bsm_model.epsilon = np.sqrt(kwargs['alpha_epsilon2']/const.alphaQED)
    else:
        bsm_model.epsilon = 1e-2
    
    # neutrino mixing
    bsm_model.Umu4  = np.sqrt(1.5e-6*7/4)

    # masses
    bsm_model.m4 =  0.140
    bsm_model.mzprime = 1.25
    bsm_model.HNLtype = "dirac"

    # update the attributes of the model with user-defined parameters
    bsm_model.__dict__.update(kwargs)

    # lock-in parameters and compute interaction vertices
    bsm_model.set_vertices()

    return bsm_model


def create_generic_HNL_model(**kwargs):

    bsm_model = GenericHNLModel()

    ## Default choices
    # vector couplings
    bsm_model.d_mu4=np.sqrt(1.5e-6*7/4)
    bsm_model.duV=const.eQED*1e-2*2/3
    bsm_model.ddV=-const.eQED*1e-2*1/3
    bsm_model.deV=const.eQED*1e-2

    # masses
    bsm_model.m4 =  0.140
    bsm_model.mzprime = 1.25
    bsm_model.HNLtype = "dirac"

    # update the attributes of the model with user-defined parameters
    bsm_model.__dict__.update(kwargs)

    # lock-in parameters and compute interaction vertices
    bsm_model.set_vertices()

    return bsm_model


class UpscatteringProcess:
    ''' 
        Describes the process of upscattering with arbitrary vertices and masses
    
    '''

    def __init__(self, nu_projectile, nu_upscattered, nuclear_target, scattering_regime, TheoryModel, helicity):

        self.nuclear_target = nuclear_target
        self.scattering_regime = scattering_regime
        if self.scattering_regime == 'coherent':
            self.target = self.nuclear_target
        elif self.scattering_regime == 'p-el':
            self.target = self.nuclear_target.get_constituent_nucleon('proton')
        elif self.scattering_regime == 'n-el':
            self.target = self.nuclear_target.get_constituent_nucleon('neutron')
        elif self.scattering_regime == 'DIS':
            self.target = self.nuclear_target.get_constituent_quarks()
        else:
            logger.error(f"Scattering regime {scattering_regime} not supported.")

        # How many constituent targets inside scattering regime? 
        if self.scattering_regime == 'coherent':
            self.target_multiplicity = 1
        elif self.scattering_regime == 'p-el':
            self.target_multiplicity = self.nuclear_target.Z
        elif self.scattering_regime == 'n-el':
            self.target_multiplicity = self.nuclear_target.N
        else:
            logger.error(f"Scattering regime {self.scattering_regime} not supported.")

        self.nu_projectile = nu_projectile
        self.nu_upscattered = nu_upscattered
        self.TheoryModel = TheoryModel
        self.helicity = helicity

        self.MA = self.target.mass
        self.mzprime = TheoryModel.mzprime
        self.mhprime = TheoryModel.mhprime
        self.m_ups = self.nu_upscattered.mass

        if self.helicity == 'conserving':
            self.h_upscattered = -1
        elif self.helicity == 'flipping':
            self.h_upscattered = +1
        else:
            logger.error(f"Error! Could not find helicity case {self.helicity}")
        

        self.Cij=TheoryModel.c_aj[pdg.get_lepton_index(nu_projectile), pdg.get_HNL_index(nu_upscattered)]
        self.Cji=self.Cij
        self.Vij=TheoryModel.d_aj[pdg.get_lepton_index(nu_projectile), pdg.get_HNL_index(nu_upscattered)]
        self.Vji=self.Vij
        self.Sij=TheoryModel.s_aj[pdg.get_lepton_index(nu_projectile), pdg.get_HNL_index(nu_upscattered)]
        self.Sji=self.Sij
        self.mu_tr=TheoryModel.t_aj[pdg.get_lepton_index(nu_projectile), pdg.get_HNL_index(nu_upscattered)]

        ###############
        # Hadronic vertices
        if self.target.is_nucleus:
            self.Chad = const.gweak/2.0/const.cw*np.abs((1.0-4.0*const.s2w)*self.target.Z-self.target.N)
            self.Vhad = const.eQED*TheoryModel.epsilon*self.target.Z
            self.Shad = TheoryModel.cSproton*self.target.Z + TheoryModel.cSneutron*self.target.N
        elif self.target.is_proton:
            self.Chad = TheoryModel.cVproton
            self.Vhad = TheoryModel.dVproton
            self.Shad = TheoryModel.cSproton
        elif self.target.is_neutron:
            self.Chad = TheoryModel.cVneutron
            self.Vhad = TheoryModel.dVneutron
            self.Shad = TheoryModel.cSneutron
        # mass mixed vertex
        self.Cprimehad = self.Chad*TheoryModel.epsilonZ

        # Neutrino energy threshold
        self.Ethreshold = self.m_ups**2 / 2.0 / self.MA + self.m_ups

        # vectorize total cross section calculator using vegas integration
        self.vectorized_total_xsec = np.vectorize(self.scalar_total_xsec, excluded=['self','diagram','NINT','NEVAL','NINT_warmup','NEVAL_warmup'])

        self.calculable_diagrams = find_calculable_diagrams(TheoryModel)

    def scalar_total_xsec(self, Enu, diagram='total', NINT=MC.NINT, NEVAL=MC.NEVAL, NINT_warmup=MC.NINT_warmup, NEVAL_warmup=MC.NEVAL_warmup):
        # below threshold
        if Enu < (self.Ethreshold):
            return 0.0
        else:
            DIM = 1
            batch_f = integrands.UpscatteringXsec(dim=DIM, Enu=Enu, ups_case=self, diagram=diagram)
            integ   = vg.Integrator(DIM*[[0.0, 1.0]]) # unit hypercube
            
            integrals = MC.run_vegas(batch_f, integ, adapt_to_errors=True,
                                        NINT=NINT, 
                                        NEVAL=NEVAL, 
                                        NINT_warmup=NINT_warmup, 
                                        NEVAL_warmup=NEVAL_warmup)
            logger.debug(f"Main VEGAS run completed.")
            
            return integrals['diff_xsec'].mean*batch_f.norm['diff_xsec']

    def total_xsec(self, Enu, diagrams=['total'], NINT=MC.NINT, NEVAL=MC.NEVAL, NINT_warmup=MC.NINT_warmup, NEVAL_warmup=MC.NEVAL_warmup):
        """ 
            Returns the total upscattering xsec for a fixed neutrino energy in cm^2
        """
        self.Enu = Enu
        all_xsecs=0.0
        for diagram in diagrams:
            if diagram in self.calculable_diagrams or diagram=='total':
                tot_xsec = self.vectorized_total_xsec(Enu, diagram=diagram, NINT=NINT, NEVAL=NEVAL, NINT_warmup=NINT_warmup, NEVAL_warmup=NEVAL_warmup)
            else:
                logger.warning(f'Warning: Diagram not found. Either not implemented or misspelled. Setting tot xsec it to zero: {diagram}')
                tot_xsec = 0.0*Enu
            
            #############
            # integrated xsec coverted to cm^2
            all_xsecs += tot_xsec*const.attobarn_to_cm2*self.target_multiplicity
            logger.debug(f"Total cross section for {diagram} calculated.")

        return all_xsecs

    def diff_xsec_Q2(self, Enu, Q2, diagrams=['total']):
        """ 
            Returns the differential upscattering xsec for a fixed neutrino energy in cm^2
        """
        s = Enu*self.MA*2+self.MA**2
        physical =  ((Q2 > ps.upscattering_Q2min(Enu, self.m_ups, self.MA)) & (Q2 < ps.upscattering_Q2max(Enu, self.m_ups, self.MA)))
        diff_xsecs=amps.upscattering_dxsec_dQ2([s,-Q2,0.0], process=self, diagrams=diagrams)
        if type(diff_xsecs) is dict:
            return {key: diff_xsecs[key]*physical for key in diff_xsecs.keys()}
        else:
            return diff_xsecs*physical*const.attobarn_to_cm2*self.target_multiplicity



class FermionDileptonDecay:

    def __init__(self, nu_parent, nu_daughter, final_lepton1, final_lepton2, TheoryModel, h_parent=-1):

        self.TheoryModel = TheoryModel
        self.HNLtype = TheoryModel.HNLtype
        self.h_parent = h_parent
        
        # particle masses
        self.mzprime = TheoryModel.mzprime
        self.mhprime = TheoryModel.mhprime
        self.mm = final_lepton1.mass*const.MeV_to_GeV 
        self.mp = final_lepton2.mass*const.MeV_to_GeV 

        if nu_daughter == pdg.nulight:
            self.Cih = np.sqrt(np.sum(np.abs(TheoryModel.c_aj[const.inds_active,pdg.get_HNL_index(nu_parent)])**2))
            self.Dih = np.sqrt(np.sum(np.abs(TheoryModel.d_aj[const.inds_active,pdg.get_HNL_index(nu_parent)])**2))
            self.Sih = np.sqrt(np.sum(np.abs(TheoryModel.s_aj[const.inds_active,pdg.get_HNL_index(nu_parent)])**2))
            self.Tih = np.sqrt(np.sum(np.abs(TheoryModel.t_aj[const.inds_active,pdg.get_HNL_index(nu_parent)])**2))
        else:
            self.Cih = TheoryModel.c_aj[pdg.get_HNL_index(nu_daughter),pdg.get_HNL_index(nu_parent)]
            self.Dih = TheoryModel.d_aj[pdg.get_HNL_index(nu_daughter),pdg.get_HNL_index(nu_parent)]
            self.Sih = TheoryModel.s_aj[pdg.get_HNL_index(nu_daughter),pdg.get_HNL_index(nu_parent)]
            self.Tih = TheoryModel.t_aj[pdg.get_HNL_index(nu_daughter),pdg.get_HNL_index(nu_parent)]


        if nu_parent == pdg.neutrino4:
            self.m_parent = TheoryModel.m4
        elif nu_parent == pdg.neutrino5:
            self.m_parent = TheoryModel.m5
        elif nu_parent == pdg.neutrino6:
            self.m_parent = TheoryModel.m6
        else:
            self.m_parent = 0.0


        if nu_daughter == pdg.neutrino4:
            self.m_daughter = TheoryModel.m4
        elif nu_daughter == pdg.neutrino5:
            self.m_daughter = TheoryModel.m5
        elif nu_daughter == pdg.neutrino6:
            self.m_daughter = TheoryModel.m6
        else:
            self.m_daughter = 0.0


        # check if CC is allowed 
        # CC_mixing1 = LNC, CC_mixing2 = LNV channel.
        if pdg.in_same_doublet(nu_daughter,final_lepton1):
            self.CC_mixing1 = TheoryModel.Ulep[pdg.get_lepton_index(final_lepton1), pdg.get_HNL_index(nu_parent)]
            self.CC_mixing2 = TheoryModel.Ulep[pdg.get_lepton_index(final_lepton2), pdg.get_HNL_index(nu_parent)]
        else:
            self.CC_mixing1 = 0
            self.CC_mixing2 = 0

        ## Minus sign important for interference!
        self.CC_mixing2 *= -1

        ## Is the mediator on shell?
        self.on_shell = (self.m_parent - self.m_daughter - self.mm - self.mp > TheoryModel.mzprime)
        self.off_shell = not self.on_shell
        ## does it have transition magnetic moment?
        self.TMM = TheoryModel.is_TMM

class FermionSinglePhotonDecay:

    def __init__(self, nu_parent, nu_daughter, TheoryModel, h_parent=-1):

        self.TheoryModel = TheoryModel
        self.HNLtype = TheoryModel.HNLtype
        self.h_parent = h_parent

        # mass of the HNLs
        if nu_daughter == pdg.neutrino4:
            self.m_daughter = TheoryModel.m4
        elif nu_daughter == pdg.neutrino5:
            self.m_daughter = TheoryModel.m5
        elif nu_daughter == pdg.neutrino6:
            self.m_daughter = TheoryModel.m6
        else:
            self.m_daughter = 0.0

        if nu_parent == pdg.neutrino4:
            self.m_parent = TheoryModel.m4
        elif nu_parent == pdg.neutrino5:
            self.m_parent = TheoryModel.m5
        elif nu_parent == pdg.neutrino6:
            self.m_parent = TheoryModel.m6
        else:
            self.m_parent = 0.0

        # transition magnetic moment parameter
        if nu_daughter == pdg.nulight:
            # |T| = sqrt(|T_ei|^2 + |T_mui|^2 + |T_taui|^2)
            self.Tih = np.sqrt(np.sum(np.abs(self.TheoryModel.t_aj[const.inds_active,pdg.get_HNL_index(nu_parent)])**2))
            self.m_daughter = 0.0
        else:
            self.Tih = self.TheoryModel.t_aj[pdg.get_HNL_index(nu_daughter),pdg.get_HNL_index(nu_parent)]


class HNLModel:

    def __init__(self, model_file=None, name='my_model'):
        """ Parent HNL model class for models with HNLs + additional new physics

        Args:
            model_file (string, optional): The filename of the model file to load model parameters. Defaults to None.
            name (str, optional): the desired name of the model. Defaults to 'my_model'.
        """
        self.model_file = model_file
        self.name = name
        
        # Particle Masses 
        self.m4			= None
        self.m5			= None
        self.m6			= None
        
        self.mzprime    = None
        self.mhprime    = None

        # Initialize spectrum
        self.nu_spectrum = [lp.nu_e, lp.nu_mu, lp.nu_tau]
        
        # scalar couplings
        self.s_e4 = 0.0
        self.s_e5 = 0.0
        self.s_e6 = 0.0
        self.s_mu4 = 0.0
        self.s_mu5 = 0.0
        self.s_mu6 = 0.0
        self.s_tau4 = 0.0
        self.s_tau5 = 0.0
        self.s_tau6 = 0.0
        self.s_44 = 0.0
        self.s_45 = 0.0
        self.s_46 = 0.0
        self.s_55 = 0.0
        self.s_56 = 0.0
        self.s_66 = 0.0

        # TMM is always set in a "model-independent" way
        # TMM in GeV^-1
        self.mu_tr_e4 = 0.0
        self.mu_tr_e5 = 0.0
        self.mu_tr_e6 = 0.0
        self.mu_tr_mu4 = 0.0
        self.mu_tr_mu5 = 0.0
        self.mu_tr_mu6 = 0.0
        self.mu_tr_tau4 = 0.0
        self.mu_tr_tau5 = 0.0
        self.mu_tr_tau6 = 0.0
        self.mu_tr_44 = 0.0
        self.mu_tr_45 = 0.0
        self.mu_tr_46 = 0.0
        self.mu_tr_55 = 0.0
        self.mu_tr_56 = 0.0
        self.mu_tr_66 = 0.0

        # Initilize nucleon couplings. These will be filled with the quark combination, which is what is actually set by the user
        self.cVproton = None
        self.cAproton = None
        self.cVneutron = None
        self.cAneutron = None
        self.dVproton = None
        self.dAproton = None
        self.dVneutron = None
        self.dAneutron = None
        self.cSproton = None
        self.cSneutron = None
        self.cPproton = None
        self.cPneutron = None

    def initialize_spectrum(self):
        
        self._spectrum = ""
        self.hnl_masses = np.empty(0)
        if self.m4:
            self.hnl_masses = np.append(self.m4,self.hnl_masses)
            self.neutrino4 = pdg.neutrino4
            self.neutrino4.mass = self.m4
        if self.m5:
            self.hnl_masses = np.append(self.m5,self.hnl_masses)
            self.neutrino5 = pdg.neutrino5
            self.neutrino5.mass = self.m5
        if self.m6:
            self.hnl_masses = np.append(self.m6,self.hnl_masses)
            self.neutrino6 = pdg.neutrino6
            self.neutrino6.mass = self.m6

        # define new HNL particles and pass the masses in MeV units (default units for Particle...)
        for i in range(len(self.hnl_masses)):
            
            # PDGID  =  59(particle spin code: 0-scalar 1-fermion 2-vector)(generation number)
            # GeV units in particle module!
            hnl = pdg.new_particle(name=f'N{4+i}', pdgid=5914+i, latex_name=f'N_{{{4+i}}}', mass=self.hnl_masses[i])
            setattr(self, f'neutrino{4+i}', hnl)
            self.nu_spectrum.append(getattr(self,  f'neutrino{4+i}'))
            
        self.HNL_spectrum = self.nu_spectrum[3:]
        self.n_nus = len(self.nu_spectrum)
        self.n_HNLs = len(self.HNL_spectrum)
        self._spectrum += f"\n\t{self.n_HNLs} {self.HNLtype} heavy neutrino(s)."




class GenericHNLModel(HNLModel):
    def __init__(self, model_file=None, name='my_model'):
        super().__init__(model_file, name)

        # Z boson couplings
        self.c_e4 = 0.0
        self.c_e5 = 0.0
        self.c_e6 = 0.0
        self.c_mu4 = 0.0
        self.c_mu5 = 0.0
        self.c_mu6 = 0.0
        self.c_tau4 = 0.0
        self.c_tau5 = 0.0
        self.c_tau6 = 0.0
        self.c_44 = 0.0
        self.c_45 = 0.0
        self.c_46 = 0.0
        self.c_55 = 0.0
        self.c_56 = 0.0
        self.c_66 = 0.0

        # vector couplings
        self.d_e4 = 0.0
        self.d_e5 = 0.0
        self.d_e6 = 0.0
        self.d_mu4 = 0.0
        self.d_mu5 = 0.0
        self.d_mu6 = 0.0
        self.d_tau4 = 0.0
        self.d_tau5 = 0.0
        self.d_tau6 = 0.0
        self.d_44 = 0.0
        self.d_45 = 0.0
        self.d_46 = 0.0
        self.d_55 = 0.0
        self.d_56 = 0.0
        self.d_66 = 0.0

        ########################
        # Charge particle couplings

        self.ceV = 0.0
        self.ceA = 0.0
        self.cuV = 0.0
        self.cuA = 0.0
        self.cdV = 0.0
        self.cdA = 0.0

        self.deV = 0.0
        self.deA = 0.0
        self.duV = 0.0
        self.duA = 0.0
        self.ddV = 0.0
        self.ddA = 0.0

        self.cSe = 0.0
        self.cSu = 0.0
        self.cSd = 0.0

        self.cPe = 0.0
        self.cPu = 0.0
        self.cPd = 0.0

    def set_vertices(self):

        # initialize spectrum of HNLs
        self.initialize_spectrum()

        ####################################################
        # SM Z boson couplings
        self.c_aj = np.array([\
                                [const.gweak/2/const.cw,0,0, self.c_e4,   self.c_e5,    self.c_e6],
                                [0,const.gweak/2/const.cw,0, self.c_mu4,  self.c_mu5,   self.c_mu6],
                                [0,0,const.gweak/2/const.cw, self.c_tau4, self.c_tau5,  self.c_tau6],
                                [0,0,0, self.c_44,   self.c_45,    self.c_46],
                                [0,0,0, self.c_45,   self.c_55,    self.c_56],
                                [0,0,0, self.c_46,   self.c_56,    self.c_66],
                                ])
        self.has_Zboson_coupling = np.any(self.c_aj[3:,:] != 0)
        if self.has_Zboson_coupling:
            self._spectrum += f"\n\t{np.sum(self.c_aj[3:,:]!=0)} non-zero Z boson coupling(s) beyond the SM."
        
        ####################################################
        # Z' vector couplings
        self.d_aj = np.array([\
                                [0,0,0, self.d_e4,   self.d_e5,    self.d_e6],
                                [0,0,0, self.d_mu4,  self.d_mu5,   self.d_mu6],
                                [0,0,0, self.d_tau4, self.d_tau5,  self.d_tau6],
                                [0,0,0, self.d_44,   self.d_45,    self.d_46],
                                [0,0,0, self.d_45,   self.d_55,    self.d_56],
                                [0,0,0, self.d_46,   self.d_56,    self.d_66],
                                ])
        self.has_vector_coupling = np.any(self.d_aj != 0)
        if self.has_vector_coupling:
            # dark photon 
            self.zprime = pdg.new_particle(name='zprime', pdgid=5921, latex_name='Z^\prime')
            self._spectrum += f"\n\t{np.sum(self.d_aj!=0)} non-zero Z'-neutrino coupling(s)."

        ####################################################
        # h' scalar couplings
        self.s_aj = np.array([\
                                [0,0,0, self.s_e4,   self.s_e5,    self.s_e6],
                                [0,0,0, self.s_mu4,  self.s_mu5,   self.s_mu6],
                                [0,0,0, self.s_tau4, self.s_tau5,  self.s_tau6],
                                [0,0,0, self.s_44,   self.s_45,    self.s_46],
                                [0,0,0, self.s_45,   self.s_55,    self.s_56],
                                [0,0,0, self.s_46,   self.s_56,    self.s_66],
                                ])
        self.has_scalar_coupling = np.any(self.s_aj != 0)
        if self.has_scalar_coupling:
            # dark scalar 
            self.hprime = pdg.new_particle(name='hprime', pdgid=5901, latex_name='h^\prime')
            self._spectrum += f"\n\t{np.sum(self.s_aj!=0)} non-zero h'-neutrino coupling(s)."
        
        ####################################################
        # create the transition mag moment scope
        self.t_aj = np.array([\
                                [0,0,0, self.mu_tr_e4,   self.mu_tr_e5,    self.mu_tr_e6],
                                [0,0,0, self.mu_tr_mu4,  self.mu_tr_mu5,   self.mu_tr_mu6],
                                [0,0,0, self.mu_tr_tau4, self.mu_tr_tau5,  self.mu_tr_tau6],
                                [0,0,0, self.mu_tr_44,   self.mu_tr_45,    self.mu_tr_46],
                                [0,0,0, self.mu_tr_45,   self.mu_tr_55,    self.mu_tr_56],
                                [0,0,0, self.mu_tr_46,   self.mu_tr_56,    self.mu_tr_66],
                                ])
        self.is_TMM = np.any(self.t_aj != 0)
        if self.is_TMM:
            self._spectrum += f"\n\t{np.sum(self.t_aj!=0)} non-zero transition magnetic moment(s)."

        prettyprinter.info(f"Model:{self._spectrum}")


        ####################################################
        # Nucleon couplings
        # n.b. lepton vertices already defined
        # for TMM, we already know it has to be (e*charge)

        if not self.cVproton:
            self.cVproton = 2*self.cuV +self.cdV
        if not self.cAproton:
            self.cAproton = 2*self.cuA + self.cdA
        if not self.cVneutron:
            self.cVneutron = 2*self.cdV + self.cuV
        if not self.cAneutron:
            self.cAneutron = 2*self.cdA + self.cuA

        if not self.dVproton:
            self.dVproton = 2*self.duV +self.ddV
        if not self.dAproton:
            self.dAproton = 2*self.duA + self.ddA
        if not self.dVneutron:
            self.dVneutron = 2*self.ddV + self.duV
        if not self.dAneutron:
            self.dAneutron = 2*self.ddA + self.duA

        if not self.cSproton:
            self.cSproton = 2*self.cSu +self.cSd
        if not self.cSneutron:
            self.cSneutron = 2*self.cSd + self.cSu
        if not self.cPproton:
            self.cPproton = 2*self.cPu +self.cPd
        if not self.cPneutron:
            self.cPneutron = 2*self.cPd + self.cPu




class ThreePortalModel(HNLModel):
    
    def __init__(self, model_file=None):

        super().__init__(model_file)

        self.name = 'Untitled'

        self.Ue4		= 0.0
        self.Umu4		= 0.0
        self.Utau4		= 0.0

        self.Ue5		= 0.0
        self.Umu5		= 0.0
        self.Utau5		= 0.0

        self.Ue6		= 0.0
        self.Umu6		= 0.0
        self.Utau6		= 0.0

        self.UD4		= 1.0
        self.UD5		= 1.0
        self.UD6		= 1.0

        # Z'
        self.gD         = 1.0
        self.epsilon    = 1.0 # kinetic mixing
        self.epsilonZ   = 0.0 # mass mixing
        
        # h'
        self.theta   = 0.0 # higgs mixing


    def set_vertices(self):
        """ 
            set all other variables starting from base members
        
        """
        
        # initialize spectrum
        self.initialize_spectrum()

        # create the vector mediator scope
        self.is_kinetically_mixed = (self.epsilon != 0)
        self.is_mass_mixed = (self.epsilonZ != 0)
        if self.is_kinetically_mixed or self.is_mass_mixed:
            # dark photon 
            self.zprime = pdg.new_particle(name='zprime', pdgid=5921, latex_name='Z^\prime')
            self._spectrum+="\n\tkinetically mixed Z'"

        # create the scalar mediator scope
        self.is_scalar_mixed = (self.theta != 0)
        if self.is_scalar_mixed:
            # dark scalar 
            self.hprime = pdg.new_particle(name='hprime', pdgid=5901, latex_name='h^\prime')
            self._spectrum+="\n\thiggs mixed h'"

        # create the scalar couplings
        self.s_aj = np.array([\
                                [0,0,0, self.s_e4,   self.s_e5,    self.s_e6],
                                [0,0,0, self.s_mu4,  self.s_mu5,   self.s_mu6],
                                [0,0,0, self.s_tau4, self.s_tau5,  self.s_tau6],
                                [0,0,0, self.s_44,   self.s_45,    self.s_46],
                                [0,0,0, self.s_45,   self.s_55,    self.s_56],
                                [0,0,0, self.s_46,   self.s_56,    self.s_66],
                                ])
        self.is_scalar_mixed = np.any(self.s_aj != 0)
        if self.is_scalar_mixed:
            self._spectrum += f"\n\t{np.sum(self.s_aj!=0)} non-zero scalar-neutrino coupling(s)."

        # create the transition mag moment scope
        self.t_aj = np.array([\
                                [0,0,0, self.mu_tr_e4,   self.mu_tr_e5,    self.mu_tr_e6],
                                [0,0,0, self.mu_tr_mu4,  self.mu_tr_mu5,   self.mu_tr_mu6],
                                [0,0,0, self.mu_tr_tau4, self.mu_tr_tau5,  self.mu_tr_tau6],
                                [0,0,0, self.mu_tr_44,   self.mu_tr_45,    self.mu_tr_46],
                                [0,0,0, self.mu_tr_45,   self.mu_tr_55,    self.mu_tr_56],
                                [0,0,0, self.mu_tr_46,   self.mu_tr_56,    self.mu_tr_66],
                                ])
        self.is_TMM = np.any(self.t_aj != 0)
        if self.is_TMM:
            self._spectrum += f"\n\t{np.sum(self.t_aj!=0)} non-zero transition magnetic moment(s)."

        prettyprinter.info(f"Model:{self._spectrum}")

        ####################################################
        # CHARGED FERMION VERTICES 
        # all the following is true to leading order in chi
        
        # Kinetic mixing with photon
        # self.epsilon = const.cw * self.chi
        self.chi = self.epsilon/const.cw 
    
        self.tanchi = math.tan(self.chi)
        self.sinof2chi  = 2*self.tanchi/(1.0+self.tanchi**2)
        self.cosof2chi  = (1.0 - self.tanchi**2)/(1.0+self.tanchi**2)
        self.s2chi = (1.0 - self.cosof2chi)/2.0
        self.c2chi = 1 - self.s2chi

        entry_22 = self.c2chi - const.s2w*self.s2chi - (self.mzprime/const.m_Z)**2 
        self.tanof2beta = const.sw *  self.sinof2chi / (entry_22)
        self.beta=const.sw*self.chi
        self.sinof2beta = const.sw * self.sinof2chi /np.sqrt(entry_22**2 + self.sinof2chi**2*const.s2w)
        self.cosof2beta = entry_22 /np.sqrt(entry_22**2 + self.sinof2chi**2*const.s2w)

        ######################
        if self.tanof2beta != 0:
            self.tbeta = self.sinof2beta/(1+self.cosof2beta)
        else:
            self.tbeta = 0.0
        ######################

        self.sbeta = math.sqrt( (1 - self.cosof2beta)/2.0)*np.sign(self.sinof2beta) # FIX ME -- works only for |beta| < pi/2 
        self.cbeta = math.sqrt( (1 + self.cosof2beta)/2.0) # FIX ME -- works only for |beta| < pi/2 

        # some abbreviations
        self._weak_vertex = const.gweak / const.cw / 2.
        self._gschi = self.gD * self.sbeta
        # dark couplings acquired by Z boson 
        self._g_weak_correction = (self.cbeta + self.tanchi*const.sw*self.sbeta)
        self._g_dark_correction = (self.cbeta*self.tanchi*const.sw - self.sbeta)
        

        # Charged leptons
        self.ceV = self._weak_vertex*(self.cbeta*(2*const.s2w - 0.5) + 3.0/2.0*self.sbeta*const.sw*self.tanchi)
        self.ceA = self._weak_vertex*(-(self.cbeta + self.sbeta*const.sw*self.tanchi)/2.0)
        # self.ceV = weak_vertex*(const.gweak/(2*const.cw) * (2*const.s2w - 0.5))
        # self.ceA = weak_vertex*(const.gweak/(2*const.cw) * (-1.0/2.0))

        # quarks
        self.cuV = self._weak_vertex*(self.cbeta*(0.5 - 4*const.s2w/3.0) - 5.0/6.0*self.sbeta*const.sw*self.tanchi)
        self.cuA = self._weak_vertex*((self.cbeta + self.sbeta*const.sw*self.tanchi)/2.0)

        self.cdV = self._weak_vertex*(self.cbeta*(-0.5 + 2*const.s2w/3.0) + 1.0/6.0*self.sbeta*const.sw*self.tanchi)
        self.cdA = self._weak_vertex*(-(self.cbeta + self.sbeta*const.sw*self.tanchi)/2.0)


        # if not self.minimal:
        self.deV = self._weak_vertex*(3.0/2.0 * self.cbeta * const.sw * self.tanchi - self.sbeta*(-0.5 + 2*const.s2w))
        self.deA = self._weak_vertex*((self.sbeta - self.cbeta * const.sw * self.tanchi)/2.0)
        # self.deV = const.gweak/(2*const.cw) * 2*const.sw*const.cw**2*self.chi
        # self.deA = const.gweak/(2*const.cw) * 0

        self.duV = -self._weak_vertex*(-5.0/6.0*self.cbeta*const.sw*self.tanchi - self.sbeta*(0.5 - 4.0/3.0*const.s2w) )
        self.duA = self._weak_vertex*((-self.sbeta + self.cbeta*const.sw*self.tanchi)/2.0)

        self.ddV = -self._weak_vertex*(-self.sbeta*(-0.5 + 2/3.0*const.s2w) + 1.0/6.0*self.cbeta*const.sw*self.tanchi)
        self.ddA = self._weak_vertex*((self.sbeta - self.cbeta*const.sw*self.tanchi)/2.0)

        self.dVproton = 2*self.duV +self.ddV
        self.dAproton = 2*self.duA + self.ddA
        self.dVneutron = 2*self.ddV + self.duV
        self.dAneutron = 2*self.ddA + self.duA

        self.cVproton = 2*self.cuV +self.cdV
        self.cAproton = 2*self.cuA + self.cdA
        self.cVneutron = 2*self.cdV + self.cuV
        self.cAneutron = 2*self.cdA + self.cuA
    

        ####################################################
        # NEUTRAL FERMION VERTICES 
        self.Ue = [1,0,0,self.Ue4, self.Ue5, self.Ue6]
        self.Umu = [0,1,0,self.Umu4, self.Umu5, self.Umu6]
        self.Utau = [0,0,1,self.Utau4, self.Utau5, self.Utau6]
        self.Udark = [0,0,0,self.UD4, self.UD5, self.UD6]

        ##### FIX-ME -- expand to arbitrary number of dark flavors.
        self.n_dark_HNLs= 1#self.n_HNLs
        # list of dark flavors 
        self.inds_dark = range(const.ind_tau+1,3+self.n_dark_HNLs)

        # Mixing matrices
        # if PMNS, use dtype=complex
        self.Ulep = np.diag(np.full_like(self.Ue,1))
        # self.Ulep = np.diag(np.full(self.n_nus,1,dtype=complex))
        # self.Ulep[:3,:3] = const.PMNS # massless light neutrinos
        
        # loop over HNL indices
        for i in range(3,self.n_HNLs+3):
            self.Ulep[const.ind_e,   i] = self.Ue[i]
            self.Ulep[i,const.ind_e] = self.Ue[i]

            self.Ulep[const.ind_mu,  i] = self.Umu[i]
            self.Ulep[i,const.ind_mu] = self.Umu[i]

            self.Ulep[const.ind_tau, i] = self.Utau[i]
            self.Ulep[i,const.ind_tau] = self.Utau[i]

            self.Ulep[self.inds_dark,   i] = self.Udark[i]/self.n_dark_HNLs
            self.Ulep[i,   self.inds_dark] = self.Udark[i]/self.n_dark_HNLs
            
        self.Ua = self.Ulep[const.inds_active,:]
        self.Uactive_heavy = self.Ulep[const.inds_active,3:]

        self.UD = self.Ulep[self.inds_dark,:]
        self.UD_heavy = self.Ulep[self.inds_dark,3:]

        ### Matrix
        # (Ua^dagger . Ua)_ij = Uei Uej + Umui Umuj + Utaui Utauj
        self.C_weak = np.dot(self.Ua.conjugate().T,self.Ua)
        # (UDi^dagger . UD)_ij
        self.D_dark = np.dot(self.UD.conjugate().T,self.UD)


        ### Vectors
        # ( |Ua4|^2, |Ua5|^2, |Ua6|^2, ...)
        self.UactiveUactive_diag = np.diagonal(np.dot(self.Uactive_heavy.conjugate(),self.Uactive_heavy.T))
        # ( |UD4|^2, |UD5|^2, |UD6|^2, ...)
        self.UDUD_diag = np.diagonal(np.dot(self.UD_heavy.conjugate(),self.UD_heavy.T))
        # (Ua4* UD4, Ua5* UD5, Ua6* UD6,..)
        self.UactiveUD_diag = np.diagonal(np.dot(self.Uactive_heavy.conjugate(),self.UD_heavy.T))


        # ( |Ue4|^2 +  |Ue5|^2 + ... , |Umu4|^2 + |Umu5|^2 + ..., |Utau4|^2 + |Utau5|^2 + ...)
        self.UahUah_mass_summed = np.sum(np.dot(self.Uactive_heavy.conjugate(),self.Uactive_heavy.T), axis=0)
        # (Ue4* UD4 + Ue5* UD5+... , Umu4* Umu4 + Umu5* Umu5+..., Utau4* Utau4 + Utau5* Utau5+...)
        self.UDhUDh_mass_summed = np.sum(np.dot(self.UD_heavy.conjugate(),self.UD_heavy.T), axis=0)
        # (Ue4* UD4 + Ue5* UD5+... , Umu4* Umu4 + Umu5* Umu5+..., Utau4* Utau4 + Utau5* Utau5+...)
        self.UahUDh_mass_summed = np.sum(np.dot(self.Uactive_heavy.conjugate(),self.UD_heavy.T), axis=0)

        ### Numbers
        # |U_a4|^2 + |U_{a5}|^2 + ...
        self.A_heavy_sum = np.sum(self.UactiveUactive_diag)
        # |U_D4|^2 + |U_{D5}|^2 + ...
        self.D_heavy_sum = np.sum(self.UDUD_diag)
        # U_A4* U_D4 + U_A5* U_D5 + ..
        self.AD_heavy_sum = np.sum(self.UactiveUD_diag)

        self.A4 = self.Ue4**2 + self.Umu4**2 + self.Utau4**2
        self.A5 = self.Ue5**2 + self.Umu5**2 + self.Utau5**2
        self.A6 = self.Ue6**2 + self.Umu6**2 + self.Utau6**2
        
        self.D4 = self.UD4**2 # self.UD4#(1.0 - self.A4 - self.A5)/(1.0+self.R)
        self.D5 = self.UD5**2 # self.UD5#(1.0 - self.A4 - self.A5)/(1.0+1.0/self.R)
        self.D6 = self.UD6**2 # self.UD6#(1.0 - self.A4 - self.A5)/(1.0+1.0/self.R)

        # NEUTRAL LEPTON SECTOR VERTICES
        self.c_ij = self._weak_vertex * self.C_weak * self._g_weak_correction\
                                + self.D_dark * self._gschi
        
        self.c_aj = self._weak_vertex * np.dot(np.diag(1-self.UahUah_mass_summed),self.Uactive_heavy) * self._g_weak_correction\
                                + np.dot(np.diag(-self.UahUDh_mass_summed),self.UD_heavy) * self._gschi

        self.d_ij = self._weak_vertex * self.C_weak * self._g_dark_correction\
                                + self.D_dark * self.gD
        self.d_aj = self._weak_vertex * np.dot(np.diag(1-self.UahUah_mass_summed),self.Uactive_heavy) * self._g_dark_correction\
                                + np.dot(np.diag(-self.UahUDh_mass_summed),self.UD_heavy) * self.gD

        # make it 3 x n_nus
        self.c_aj = np.hstack((np.diag([1,1,1]),self.c_aj))
        self.d_aj = np.hstack((np.diag([1,1,1]),self.d_aj))

        self.dlight = 0.0


        #########################
        # Scalar couplings
        self.tantheta = np.tan(self.theta)
        self.sintheta = np.sin(self.theta)
        self.costheta = np.cos(self.theta)
        
        # light quark couplings determine higgs coupling to nucleon 
        # see e.g. arxiv.org/abs/1306.4710

        self.sigma_l = 0.058 # GeV
        self.sigma_0 = 0.055 # GeV
        z = 1.49 # isospin breaking parameter
        y = 1 - self.sigma_0/self.sigma_l
        _prefactor = 1/(const.m_u + const.m_d)*self.sigma_l/const.m_avg
        self.fu = _prefactor*const.m_u * (2*z + y*(1 - z))/(1 + z)
        self.fd = _prefactor*const.m_d * (2 - y*(1 - z))/(1+z)
        self.fs = _prefactor*const.m_s * y

        self.fN_higgs = 2/9 + 7/9* ( self.fu + self.fd + self.fs )
        self.c_nucleon_higgs = self.fN_higgs * const.m_avg/const.vev_EW
        
        self.cSnucleon = self.sintheta*self.c_nucleon_higgs
        # isospin
        self.cSproton  = self.cSnucleon
        self.cSneutron = self.cSnucleon
        self.ceS = self.costheta*const.m_e/const.vev_EW/np.sqrt(2)
        self.deS = self.sintheta*const.m_e/const.vev_EW/np.sqrt(2)


    def compute_rates(self):
        
            ## FIX ME -- GENERALIZE TO N5 and N6
            ##################
            # Neutrino 4
            mh = self.m4
            rates = {}
            neutrinos = [lp.nu_e, lp.nu_mu, lp.nu_tau]

            # channels with 3 neutrinos in final state
            rates['nu_nu_nu'] = 0.0
            for nu_a in neutrinos:
                rates['nu_nu_nu'] += dr.nui_nuj_nuk_nuk(self, const.N4, nu_a)

            # channels with 1 neutrino in final states
            rates['nu_gamma'] = 0
            rates['nu_e_e'] = 0
            rates['nu_mu_mu'] = 0
            rates['nu_e_mu'] = 0
            rates['nu_pi'] = 0 
            rates['nu_eta'] = 0
            rates['e_pi'] = 0
            rates['e_K'] = 0
            rates['mu_pi'] = 0
            rates['mu_K'] = 0

            for nu_a in neutrinos:          # nu gamma 
                rates['nu_gamma'] += dr.nui_nuj_gamma(self, const.N4, nu_a)
                # dileptons -- already contains the Delta L = 2 channel
                if mh > 2*lp.e_minus.mass/1e3:
                    rates['nu_e_e'] += dr.nui_nuj_ell1_ell2(self, const.N4, nu_a, lp.e_minus, lp.e_plus)
                if mh > lp.e_minus.mass/1e3 + lp.mu_minus.mass/1e3:
                    rates['nu_e_mu'] += dr.nui_nuj_ell1_ell2(self, const.N4, nu_a, lp.e_minus, lp.mu_plus)
                    rates['nu_e_mu'] += dr.nui_nuj_ell1_ell2(self, const.N4, nu_a, lp.mu_minus, lp.e_plus)
                if mh > 2*lp.mu_minus.mass/1e3:
                    rates['nu_mu_mu'] += dr.nui_nuj_ell1_ell2(self, const.N4, nu_a, lp.mu_minus, lp.mu_plus)
                # pseudoscalar -- neutral current
                if mh > lp.pi_0.mass/1e3:
                    rates['nu_pi'] += dr.nui_nu_P(self, const.N4, nu_a, lp.pi_0)
                if mh > lp.eta.mass/1e3:
                    rates['nu_eta'] += dr.nui_nu_P(self, const.N4, nu_a, lp.eta)
                

            # CC-only channels  
            # pseudoscalar -- factor of 2 for delta L=2 channel 
            if mh > lp.e_minus.mass/1e3+lp.pi_plus.mass/1e3:
                rates['e_pi'] = dr.nui_l_P(self, const.N4, lp.e_minus, lp.pi_plus)
            if mh > lp.e_minus.mass/1e3+lp.K_plus.mass/1e3:
                rates['e_K'] = dr.nui_l_P(self, const.N4, lp.e_minus, lp.K_plus)
            
            # pseudoscalar -- already contain the Delta L = 2 channel
            if mh > lp.mu_minus.mass/1e3+lp.pi_plus.mass/1e3:
                rates['mu_pi'] = dr.nui_l_P(self, const.N4, lp.mu_minus, lp.pi_plus)
            if mh > lp.mu_minus.mass/1e3+lp.K_plus.mass/1e3:
                rates['mu_K'] = dr.nui_l_P(self, const.N4, lp.mu_minus, lp.K_plus)
        
            self.rates = rates          

            # total decay rate
            self.rate_total = sum(self.rates.values())

            # total decay rate
            self.lifetime = const.get_decay_rate_in_s(self.rate_total)
            self.ctau0 = const.get_decay_rate_in_cm(self.rate_total)

            # Branchin ratios
            brs = {}
            for channel in self.rates.keys():
                brs[channel] = self.rates[channel]/self.rate_total
            self.brs = brs


class HNLparticle():

    def __init__(self, this_hnl, bsm_model):

        self.this_hnl = this_hnl
        self.bsm_model = bsm_model
        
        # Dirac or Majorana
        self.HNLtype    = bsm_model.HNLtype

        self.mHNL = bsm_model.masses[get_HNL_index(this_hnl)]


    def _setup_rates(self):

        self.rates = {}
        self.daughter_neutrinos = self.bsm_model.nu_spectrum[:pdg.get_HNL_index(self.this_hnl)]


        # Setting all decay BRs to 0
        for nu_i in self.daughter_neutrinos:
            for nu_j in self.daughter_neutrinos:
                for nu_k in self.daughter_neutrinos:
                    self.rates[f'{nu_i.name}_{nu_j.name}_{nu_k.name}'] = 0
            self.rates[f'{nu_i.name}_{lp.gamma.name}']                     = 0
            self.rates[f'{nu_i.name}_{lp.e_plus.name}_{lp.e_minus.name}']  = 0
            self.rates[f'{nu_i.name}_{lp.mu_plus.name}_{lp.e_minus.name}'] = 0
            self.rates[f'{nu_i.name}_{lp.e_plus.name}_{lp.mu_minus.name}'] = 0
            self.rates[f'{nu_i.name}_{mu_plus.name}_{mu_minus.name}']      = 0
            self.rates[f'{nu_i.name}_{lp.pi_0.name}']                      = 0
            self.rates[f'{nu_i.name}_{lp.eta.name}']                       = 0
            self.rates[f'{lp.e_minus.name}_{lp.pi_plus.name}']             = 0
            self.rates[f'{lp.e_plus.name}_{lp.pi_minus.name}']             = 0
            self.rates[f'{lp.e_minus.name}_{lp.K_plus.name}']              = 0
            self.rates[f'{lp.e_plus.name}_{lp.K_minus.name}']              = 0
            self.rates[f'{lp.mu_minus.name}_{lp.pi_plus.name}']            = 0
            self.rates[f'{lp.mu_plus.name}_{lp.pi_minus.name}']            = 0
            self.rates[f'{lp.mu_minus.name}_{lp.K_plus.name}']             = 0
            self.rates[f'{lp.mu_plus.name}_{lp.K_minus.name}']             = 0
       

    def compute_rates(self):
        

        #################
        mh = self.mHNL

        # channels with 3 neutrinos in final state
        for nu_i in self.daughter_neutrinos:
            for nu_j in self.daughter_neutrinos:
                for nu_k in self.daughter_neutrinos:
                    self.rates[f'{nu_i.name}_{nu_j.name}_{nu_k.name}'] += dr.new_nuh_nui_nuj_nuk(
                                                                    self.bsm_model, 
                                                                    initial_neutrino=self.particle,
                                                                    final_neutrinoi=nu_a,
                                                                    final_neutrinoj=nu_b,
                                                                    final_neutrinok=nu_c)
        for nu_i in self.daughter_neutrinos:          
            # nu gamma 
            self.rates[f'{nu_i.name}_{lp.gamma.name}'] += dr.nui_nuj_gamma(self.bsm_model, self.this_hnl, nu_i)
            # dileptons -- already contains the Delta L = 2 channel
            # e+e-
            if mh > 2*lp.e_minus.mass/1e3:
                self.rates[f'{nu_i.name}_{lp.e_plus.name}_{lp.e_minus.name}'] += dr.nui_nuj_ell1_ell2(self.bsm_model, self.this_hnl, nu_i, lp.e_minus, lp.e_plus)
            # e+ mu- and # e- mu+
            if mh > lp.e_minus.mass/1e3 + lp.mu_minus.mass/1e3:
                self.rates[f'{nu_i.name}_{lp.mu_plus.name}_{lp.e_minus.name}'] += dr.nui_nuj_ell1_ell2(self.bsm_model, self.this_hnl, nu_i, lp.e_minus, lp.mu_plus)
                self.rates[f'{nu_i.name}_{lp.e_plus.name}_{lp.mu_minus.name}'] += dr.nui_nuj_ell1_ell2(self.bsm_model, self.this_hnl, nu_i, lp.mu_minus, lp.e_plus)
            # mu+ mu- 
            if mh > 2*lp.mu_minus.mass/1e3:
                self.rates[f'{nu_i.name}_{mu_plus.name}_{mu_minus.name}'] += dr.nui_nuj_ell1_ell2(self.bsm_model, self.this_hnl, nu_i, lp.mu_minus, lp.mu_plus)
            # pseudoscalar -- neutral current
            if mh > lp.pi_0.mass/1e3:
                self.rates[f'{nu_i.name}_{lp.pi_0.name}'] += dr.nui_nu_P(self.bsm_model, self.this_hnl, nu_i, lp.pi_0)
            if mh > lp.eta.mass/1e3:
                self.rates[f'{nu_i.name}_{lp.eta.name}'] += dr.nui_nu_P(self.bsm_model, self.this_hnl, nu_i, lp.eta)


        # CC-only channels  
        # pseudoscalar 
        if mh > lp.e_minus.mass/1e3+lp.pi_plus.mass/1e3:
            self.rates[f'{lp.e_minus.name}_{lp.pi_plus.name}'] = nui_l_P(self.bsm_model, self.this_hnl, lp.e_minus, lp.pi_plus)
            self.rates[f'{lp.e_plus.name}_{lp.pi_minus.name}'] = nui_l_P(self.bsm_model, self.this_hnl, lp.e_plus, lp.pi_minus)
        if mh > lp.e_minus.mass/1e3+lp.K_plus.mass/1e3:
            self.rates[f'{lp.e_minus.name}_{lp.K_plus.name}'] = nui_l_P(self.bsm_model, self.this_hnl, lp.e_minus, lp.K_plus)
            self.rates[f'{lp.e_plus.name}_{lp.K_minus.name}'] = nui_l_P(self.bsm_model, self.this_hnl, lp.e_plus, lp.K_minus)
        
        # pseudoscalar
        if mh > lp.mu_minus.mass/1e3+lp.pi_plus.mass/1e3:
            self.rates[f'{lp.mu_minus.name}_{lp.pi_plus.name}'] = nui_l_P(self.bsm_model, self.this_hnl, lp.mu_minus, lp.pi_plus)
            self.rates[f'{lp.mu_plus.name}_{lp.pi_minus.name}'] = nui_l_P(self.bsm_model, self.this_hnl, lp.mu_plus, lp.pi_minus)
        if mh > lp.mu_minus.mass/1e3+lp.K_plus.mass/1e3:
            self.rates[f'{lp.mu_minus.name}_{lp.K_plus.name}'] = nui_l_P(self.bsm_model, self.this_hnl, lp.mu_minus, lp.K_plus)
            self.rates[f'{lp.mu_plus.name}_{lp.K_minus.name}'] = nui_l_P(self.bsm_model, self.this_hnl, lp.mu_plus, lp.K_minus)
       

        # total decay rate
        self.rate_total = sum(self.rates.values())

        # total decay rate
        self.lifetime = const.get_decay_rate_in_s(self.rate_total)
        self.ctau0 = const.get_decay_rate_in_cm(self.rate_total)

        # branching ratios
        brs = {}
        for channel in self.rates.keys():
            brs[channel] = self.rates[channel]/self.rate_total
        self.brs = brs


def find_calculable_diagrams(bsm_model):
    """ 
    Args:
        bsm_model (DarkNews.model.Model): main BSM model class of DarkNews

    Returns:
        list: with all non-zero upscattering diagrams to be computed in this model.
    """

    calculable_diagrams = []
    calculable_diagrams.append('NC_SQR')
    if bsm_model.is_kinetically_mixed: 
        calculable_diagrams.append('KinMix_SQR')
        calculable_diagrams.append('KinMix_NC_inter')
    if bsm_model.is_mass_mixed: 
        calculable_diagrams.append('MassMix_SQR')
        calculable_diagrams.append('MassMix_NC_inter')
        if bsm_model.is_kinetically_mixed: 
            calculable_diagrams.append('KinMix_MassMix_inter')
    if bsm_model.is_TMM: 
        calculable_diagrams.append('TMM_SQR')
    if bsm_model.is_scalar_mixed: 
        calculable_diagrams.append('Scalar_SQR')
        calculable_diagrams.append('Scalar_NC_inter')
        if bsm_model.is_kinetically_mixed: 
            calculable_diagrams.append('Scalar_KinMix_inter')
        if bsm_model.is_mass_mixed: 
            calculable_diagrams.append('Scalar_MassMix_inter')
    return calculable_diagrams