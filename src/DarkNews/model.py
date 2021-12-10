import numpy as np
import math

from DarkNews import logger

from particle import Particle
from particle import literals as lp

from . import const 
from . import decay_rates as dr
from . import pdg


def create_model(args):

    bsm_model = Model()
    if args.gD:
        bsm_model.gD = args.gD
    elif args.alphaD:
        bsm_model.gD = np.sqrt(4*np.pi*args.alphaD)
    
    
    if args.epsilon:
        bsm_model.epsilon = args.epsilon
    elif args.epsilon2:
        bsm_model.epsilon2 = np.sqrt(args.epsilon2)
    elif args.chi:
        bsm_model.epsilon = args.chi*const.cw
    elif args.alpha_epsilon2:
        bsm_model.epsilon = np.sqrt(args.alpha_epsilon2/const.alphaQED)
    

    bsm_model.Ue4   = args.ue4
    bsm_model.Umu4  = args.umu4
    bsm_model.Utau4 = args.utau4
    
    bsm_model.Ue5   = args.ue5
    bsm_model.Umu5  = args.umu5
    bsm_model.Utau5 = args.utau5
    
    bsm_model.Ue6   = args.ue6
    bsm_model.Umu6  = args.umu6
    bsm_model.Utau6 = args.utau6
    
    bsm_model.UD4 = args.ud4
    bsm_model.UD5 = args.ud5
    bsm_model.UD6 = args.ud6
    
    bsm_model.m4 = args.m4
    bsm_model.m5 = args.m5
    bsm_model.m6 = args.m6
    
    bsm_model.mzprime = args.mzprime
    bsm_model.HNLtype = args.D_or_M

    bsm_model.set_high_level_variables()

    return bsm_model


class UpscatteringProcess:
    ''' 
        Describes the process of upscattering with arbitrary vertices and masses
    
    '''

    def __init__(self, nu_projectile, nu_upscattered, target, TheoryModel, helicity):

        self.MAJ = int(TheoryModel.HNLtype == 'majorana')

        self.nu_projectile = nu_projectile
        self.nu_upscattered = nu_upscattered
        self.target = target
        self.TheoryModel = TheoryModel
        self.helicity = helicity

        if self.helicity == 'conserving':
            self.h_upscattered = -1
        elif self.helicity == 'flipping':
            self.h_upscattered = +1
        else:
            logger.error(f"Error! Could not find helicity case {self.helicity}")
        
        h = self.h_upscattered

        self.mzprime = TheoryModel.mzprime

        ###############
        # leptonic vertices for upscattering 
        if nu_upscattered==pdg.neutrino4:
            self.m_ups = TheoryModel.m4
            self.Cij=TheoryModel.cmu4
            self.Cji=self.Cij
            self.Vij=TheoryModel.dmu4
            self.Vji=self.Vij
        
        elif nu_upscattered==pdg.neutrino5:
            self.m_ups = TheoryModel.m5
            self.Cij=TheoryModel.cmu5
            self.Cji=self.Cij
            self.Vij=TheoryModel.dmu5
            self.Vji=self.Vij
        
        elif nu_upscattered==pdg.neutrino6:
            self.m_ups = TheoryModel.m6
            self.Cij=TheoryModel.cmu6
            self.Cji=self.Cij
            self.Vij=TheoryModel.dmu6
            self.Vji=self.Vij
        else:
            logger.error(f"Error! Could not find particle produced in upscattering: {nu_upscattered}.")


        # Hadronic vertices
        if target.is_nucleus:
            self.Chad = const.gweak/2.0/const.cw*np.abs((1.0-4.0*const.s2w)*target.Z-target.N)
            self.Vhad = const.eQED*TheoryModel.epsilon*target.Z
        elif target.is_proton:
            self.Chad = TheoryModel.cVproton
            self.Vhad = TheoryModel.dVproton
        elif target.is_neutron:
            self.Chad = TheoryModel.cVneutron
            self.Vhad = TheoryModel.dVneutron

        # mass mixed vertex
        self.Cprimehad = self.Chad*TheoryModel.epsilonZ

        self.MA = target.mass

class FermionDecayProcess:

    def __init__(self, nu_parent, nu_daughter, final_lepton1, final_lepton2, TheoryModel, h_parent=-1):

        params = TheoryModel
        self.TheoryModel = TheoryModel
        self.HNLtype = params.HNLtype
        self.h_parent = h_parent

        if nu_parent==pdg.neutrino4:
            
            self.m_parent = params.m4
            if nu_daughter==pdg.nue:
                self.Cih = params.ce4
                self.Dih = params.de4
                self.m_daughter = 0.0
            elif nu_daughter==pdg.numu:
                self.Cih = params.cmu4
                self.Dih = params.dmu4
                self.m_daughter = 0.0
            elif nu_daughter==pdg.nutau:
                self.Cih = params.ctau4
                self.Dih = params.dtau4
                self.m_daughter = 0.0
            elif nu_daughter==pdg.nulight:
                self.Cih = params.clight4
                self.Dih = params.dlight4
                self.m_daughter = 0.0
            elif nu_daughter==pdg.neutrino4:
                logger.error('ERROR! (nu4 -> nu4 l l) is kinematically not allowed!')
        
        elif nu_parent==pdg.neutrino5:
            
            self.m_parent = params.m5
            if nu_daughter==pdg.nue:
                self.Cih = params.ce5
                self.Dih = params.de5
                self.m_daughter = 0.0
            elif nu_daughter==pdg.numu:
                self.Cih = params.cmu5
                self.Dih = params.dmu5
                self.m_daughter = 0.0
            elif nu_daughter==pdg.nutau:
                self.Cih = params.ctau5
                self.Dih = params.dtau5
                self.m_daughter = 0.0
            elif nu_daughter==pdg.nulight:
                self.Cih = params.clight5
                self.Dih = params.dlight5
                self.m_daughter = 0.0
            elif nu_daughter==pdg.neutrino4:
                self.Cih = params.c45
                self.Dih = params.d45
                self.m_daughter = params.m4

        elif nu_parent == pdg.neutrino6:

            self.m_parent = params.m6
            if nu_daughter==pdg.nue:
                self.Cih = params.ce6
                self.Dih = params.de6
                self.m_daughter = 0.0
            elif nu_daughter==pdg.numu:
                self.Cih = params.cmu6
                self.Dih = params.dmu6
                self.m_daughter = 0.0
            elif nu_daughter==pdg.nutau:
                self.Cih = params.ctau6
                self.Dih = params.dtau6
                self.m_daughter = 0.0
            elif nu_daughter==pdg.nulight:
                self.Cih = params.clight6
                self.Dih = params.dlight
                self.m_daughter = 0.0
            elif nu_daughter==pdg.neutrino4:
                self.Cih = params.c46
                self.Dih = params.d46
                self.m_daughter = params.m4
            elif nu_daughter==pdg.neutrino5:
                self.Cih = params.c56
                self.Dih = params.d56
                self.m_daughter = params.m5
        else:
            logger.error(f"Error! Could not find parent particle {nu_parent}.")


        ################################################
        # DECIDE ON CHARGED LEPTON
        if pdg.in_same_doublet(nu_daughter,final_lepton1):
            # Mixing required for CC N-like
            if (final_lepton1==pdg.tau):
                self.CC_mixing1 = params.Utau4
            elif(final_lepton1==pdg.muon):
                self.CC_mixing1 = params.Umu4
            elif(final_lepton1==pdg.electron):
                self.CC_mixing1 = params.Ue4
            else:
                logger.warning("WARNING: Unable to set CC mixing parameter for decay. Assuming 0.")
                self.CC_mixing1 = 0

            # Mixing required for CC Nbar-like
            if (final_lepton2==pdg.tau):
                self.CC_mixing2 = params.Utau4
            elif(final_lepton2==pdg.muon):
                self.CC_mixing2 = params.Umu4
            elif(final_lepton2==pdg.electron):
                self.CC_mixing2 = params.Ue4
            else:
                logger.warning("WARNING: Unable to set CC mixing parameter for decay. Assuming 0.")
                self.CC_mixing2 = 0
        else:
            self.CC_mixing1 = 0
            self.CC_mixing2 = 0
        

        ##############################
        # CHARGED LEPTON MASSES 

        self.mm = final_lepton1.mass*1e-3 # PDG from MeV to GeV
        self.mp = final_lepton2.mass*1e-3 # PDG from MeV to GeV

        #######################################
        ### WATCH OUT FOR THE MINUS SIGN HERE -- IMPORTANT FOR INTERFERENCE
        ## Put required mixings in CCflags
        self.CC_mixing2 *= -1


        ## Is the mediator on shell?
        self.on_shell = (self.m_parent - self.m_daughter > params.mzprime)

        self.mzprime = TheoryModel.mzprime


class Model:
    """ Define a BSM model by setting parameters
    """

    def __init__(self, model_file=None):


        if not model_file:

            self.gD         = 1.0
            self.epsilon	= 1.0
            self.epsilonZ   = 0.0


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

            self.m4			= 1e10
            self.m5			= 1e10
            self.m6			= 1e10

            self.mzprime	= 1.0
            self.HNLtype	= 'majorana'

        else: 
            # self.number_of_HNLs = len(HNLspectrum)

            # # HNL masses
            # self.mN = np.array(HNL_spectrum)

            # # Zprime vertices
            # self.V_alphaN =  np.zeros((3,self.number_of_HNLs)) # (e,mu,tau) (V_alphaN) (N1,N2,N3)^T

            # # hprime vertices
            # self.S_alphaN =  np.zeros((3,self.number_of_HNLs)) # (e,mu,tau) (S_alphaN) (N1,N2,N3)^T

            # # transition magnetic moment
            # self.TMM_alphaN = np.zeros((3,self.number_of_HNLs)) # (e,mu,tau) (TMM) (N1,N2,N3)^T
            pass

    def set_high_level_variables(self):
        """ set all other variables starting from base members
        """

        # Scope of the model
        self.is_kinetically_mixed = (self.epsilon != 0)
        self.is_mass_mixed = (self.epsilonZ != 0)
        
        # self.has_Zprime = (np.sum(self.V_alphaN) != 0)
        # self.has_hprime = (np.sum(self.S_alphaN) != 0)
        # self.has_TMM = (np.sum(self.TMM_alphaN) != 0)

        self.Ue1    = math.sqrt(1. - self.Ue4**2 - self.Ue5**2)
        self.Umu1   = math.sqrt(1. - self.Umu4**2 - self.Umu5**2)
        self.Utau1  = math.sqrt(1. - self.Utau4**2 - self.Utau5**2)
        self.UD1    = math.sqrt(self.Ue4**2 + self.Umu4**2 + self.Utau4**2)

        # self.R = self.m4 / self.m5

        # same as UactiveXSQR
        self.A4 = self.Ue4**2 + self.Umu4**2 + self.Utau4**2
        self.A5 = self.Ue5**2 + self.Umu5**2 + self.Utau5**2
        self.A6 = self.Ue6**2 + self.Umu6**2 + self.Utau6**2
        
        self.D4 = self.UD4**2 #self.UD4#(1.0 - self.A4 - self.A5)/(1.0+self.R)
        self.D5 = self.UD5**2 #self.UD5#(1.0 - self.A4 - self.A5)/(1.0+1.0/self.R)
        self.D6 = self.UD6**2 #self.UD6#(1.0 - self.A4 - self.A5)/(1.0+1.0/self.R)

        #########
        # FIX ME -- this expression depends on a large cancellation -- can we make more stable?
        self.C45SQR = self.A5 * (1. - self.A5) - self.D5 * (1. - self.D4 - self.D5)
        self.D45SQR = self.D4 * self.D5

        # self.UD4 = (1.0 - self.A4 - self.A5)/(1.0+self.R)
        # self.UD5 = (1.0 - self.A4 - self.A5)/(1.0+1.0/self.R)

        # potentially not used
        self.alphaD = self.gD**2 / 4. / math.pi
        self.chi = self.epsilon/const.cw


        ########################################################
        # all the following is true to leading order in chi


        # NEUTRAL LEPTON SECTOR VERTICES
        
        gschi = self.gD * const.sw * self.chi # constant repeated over and over

        self.epsilon = const.cw * self.chi
        self.weak_vertex = const.gweak / const.cw / 2.
        self.dlight = 0.0

        # UmuiUdi for i 1 to 3
        self.UeiUDi = -(self.UD4 * self.Ue4 + self.UD5 * self.Ue5 + self.UD6 * self.Ue6)
        self.UmuiUDi = -(self.UD4 * self.Umu4 + self.UD5 * self.Umu5 + self.UD6 * self.Umu6)
        self.UtauiUDi = -(self.UD4 * self.Utau4 + self.UD5 * self.Utau5 + self.UD6 * self.Utau6)

        # Neutrino couplings ## CHECK THE SIGN IN THE SECOND TERM????
        self.ce6 = self.weak_vertex * self.Ue6 + self.UD6 * self.UeiUDi * gschi 
        self.cmu6 = self.weak_vertex * self.Umu6 + self.UD6 * self.UmuiUDi * gschi
        self.ctau6 = self.weak_vertex * self.Utau6 + self.UD6 * self.UtauiUDi * gschi

        self.de6 = self.UD6 * self.gD * self.UeiUDi
        self.dmu6 = self.UD6 * self.gD * self.UmuiUDi
        self.dtau6 = self.UD6 * self.gD * self.UtauiUDi

        self.ce5 = self.weak_vertex * self.Ue5 + self.UD5 * self.UeiUDi * gschi
        self.cmu5 = self.weak_vertex * self.Umu5 + self.UD5 * self.UmuiUDi * gschi
        self.ctau5 = self.weak_vertex * self.Utau5 + self.UD5 * self.UtauiUDi * gschi

        self.de5 = self.UD5 * self.gD * self.UeiUDi
        self.dmu5 = self.UD5 * self.gD * self.UmuiUDi
        self.dtau5 = self.UD5 * self.gD * self.UtauiUDi

        self.ce4 = self.weak_vertex * self.Ue4 + self.UD4 * self.UeiUDi * gschi
        self.cmu4 = self.weak_vertex * self.Umu4 + self.UD4 * self.UmuiUDi * gschi
        self.ctau4 = self.weak_vertex * self.Utau4 + self.UD4 * self.UtauiUDi * gschi

        self.de4 = self.UD4 * self.gD * self.UeiUDi
        self.dmu4 = self.UD4 * self.gD * self.UmuiUDi
        self.dtau4 = self.UD4 * self.gD * self.UtauiUDi

        self.weak_vertex4 = math.sqrt(self.ce4**2 + self.cmu4**2 + self.ctau4**2)
        self.dlight4 = math.sqrt(self.de4**2 + self.dmu4**2 + self.dtau4**2)

        self.weak_vertex5 = math.sqrt(self.ce5**2 + self.cmu5**2 + self.ctau5**2)
        self.dlight5 = math.sqrt(self.de5**2 + self.dmu5**2 + self.dtau5**2)

        self.weak_vertex6 = math.sqrt(self.ce6**2 + self.cmu6**2 + self.ctau6**2)
        self.dlight6 = math.sqrt(self.de6**2 + self.dmu6**2 + self.dtau6**2)

        # combinations
        self.c46 = self.weak_vertex * math.sqrt(self.A4 * self.A6) + self.UD6 * self.UD4 * gschi
        self.c45 = self.weak_vertex * math.sqrt(self.A4 * self.A5) + self.UD5 * self.UD4 * gschi
        self.c44 = self.weak_vertex * math.sqrt(self.A4 * self.A4) + self.UD4 * self.UD4 * gschi
        self.c55 = self.weak_vertex * math.sqrt(self.A5 * self.A5) + self.UD5 * self.UD5 * gschi
        self.c56 = self.weak_vertex * math.sqrt(self.A5 * self.A6) + self.UD6 * self.UD5 * gschi
        self.c66 = self.weak_vertex * math.sqrt(self.A6 * self.A6) + self.UD6 * self.UD6 * gschi

        self.d56 = self.UD6 * self.UD5 * self.gD
        self.d46 = self.UD6 * self.UD4 * self.gD
        self.d45 = self.UD5 * self.UD4 * self.gD
        self.d44 = self.UD4 * self.UD4 * self.gD
        self.d55 = self.UD5 * self.UD5 * self.gD
        self.d66 = self.UD6 * self.UD6 * self.gD

        ########################################################
        # all the following is true to leading order in chi

        # Kinetic mixing
        ##############
        s2w = const.s2w
        sw  = const.sw
        cw  = const.cw

        tanchi = math.tan(self.chi)
        sinof2chi  = 2*tanchi/(1.0+tanchi*tanchi)
        cosof2chi  = (1.0 - tanchi*tanchi)/(1.0+tanchi*tanchi)
        s2chi = (1.0 - cosof2chi)/2.0
        c2chi = 1 - s2chi

        entry_22 = c2chi - s2w*s2chi - (self.mzprime/const.m_Z)**2 
        self.tanof2beta = sw *  sinof2chi / (entry_22)
        self.sinof2beta = sw * sinof2chi /np.sqrt(entry_22**2 + sinof2chi**2*s2w)
        self.cosof2beta = entry_22 /np.sqrt(entry_22**2 + sinof2chi**2*s2w)

        ######################
        if self.tanof2beta != 0:
            self.tbeta = self.sinof2beta/(1+self.cosof2beta)
        else:
            self.tbeta = 0.0
        ######################

        self.sbeta = math.sqrt( (1 - self.cosof2beta)/2.0) # FIX ME -- works only for very beta < 2pi -- prob okay
        self.cbeta = math.sqrt( (1 + self.cosof2beta)/2.0) # FIX ME -- works only for very beta < 2pi -- prob okay

        weak_vertex = const.gweak/(2*const.cw) ####### IS THERE AN ADDITIONAL FACTOR OF 2??
        # Charged leptons
        self.ceV = weak_vertex*(self.cbeta*(2*const.s2w - 0.5) + 3.0/2.0*self.sbeta*const.sw*tanchi)
        self.ceA = weak_vertex*(-(self.cbeta + self.sbeta*const.sw*tanchi)/2.0)
        # self.ceV = weak_vertex*(const.gweak/(2*const.cw) * (2*const.s2w - 0.5))
        # self.ceA = weak_vertex*(const.gweak/(2*const.cw) * (-1.0/2.0))

        # quarks
        self.cuV = weak_vertex*(self.cbeta*(0.5 - 4*const.s2w/3.0) - 5.0/6.0*self.sbeta*const.sw*tanchi)
        self.cuA = weak_vertex*((self.cbeta + self.sbeta*const.sw*tanchi)/2.0)

        self.cdV = weak_vertex*(self.cbeta*(-0.5 + 2*const.s2w/3.0) + 1.0/6.0*self.sbeta*const.sw*tanchi)
        self.cdA = weak_vertex*(-(self.cbeta + self.sbeta*const.sw*tanchi)/2.0)


        # if not self.minimal:
        self.deV = weak_vertex*(3.0/2.0 * self.cbeta * sw * tanchi - self.sbeta*(-0.5 + 2*const.s2w))
        self.deA = weak_vertex*((self.sbeta - self.cbeta * const.sw * tanchi)/2.0)
        # self.deV = const.gweak/(2*const.cw) * 2*const.sw*const.cw**2*self.chi
        # self.deA = const.gweak/(2*const.cw) * 0

        self.duV = -weak_vertex*(-5.0/6.0*self.cbeta*const.sw*tanchi - self.sbeta*(0.5 - 4.0/3.0*const.s2w) )
        self.duA = weak_vertex*((-self.sbeta + self.cbeta*const.sw*tanchi)/2.0)

        self.ddV = -weak_vertex*(-self.sbeta*(-0.5 + 2/3.0*const.s2w) + 1.0/6.0*self.cbeta*const.sw*tanchi)
        self.ddA = weak_vertex*((self.sbeta - self.cbeta*const.sw*tanchi)/2.0)

        self.dVproton = 2*self.duV +self.ddV
        self.dAproton = 2*self.duA + self.ddA
        self.dVneutron = 2*self.ddV + self.duV
        self.dAneutron = 2*self.ddA + self.duA

        self.cVproton = 2*self.cuV +self.cdV
        self.cAproton = 2*self.cuA + self.cdA
        self.cVneutron = 2*self.cdV + self.cuV
        self.cAneutron = 2*self.cdA + self.cuA



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
            self.lifetime = get_decay_rate_in_s(self.rate_total)
            self.ctau0 = get_decay_rate_in_cm(self.rate_total)

            # Branchin ratios
            brs = {}
            for channel in self.rates.keys():
                brs[channel] = self.rates[channel]/self.rate_total
            self.brs = brs


