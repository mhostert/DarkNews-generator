import numpy as np
import math

from DarkNews import logger, prettyprinter

from particle import literals as lp

from DarkNews import const
from DarkNews import pdg
from DarkNews import decay_rates as dr


def create_3portal_HNL_model(**kwargs):

    bsm_model = ThreePortalModel()

    # update the attributes of the model with user-defined parameters
    bsm_model.__dict__.update(kwargs)

    # lock-in parameters and compute interaction vertices
    bsm_model.set_vertices()

    return bsm_model


def create_generic_HNL_model(**kwargs):

    bsm_model = GenericHNLModel()

    # update the attributes of the model with user-defined parameters
    bsm_model.__dict__.update(kwargs)

    # lock-in parameters and compute interaction vertices
    bsm_model.set_vertices()

    return bsm_model


class HNLModel:
    def __init__(self, model_file=None, name="my_model"):
        """Parent HNL model class for models with HNLs + additional new physics

        Args:
            model_file (string, optional): The filename of the model file to load model parameters. Defaults to None.
            name (str, optional): the desired name of the model. Defaults to 'my_model'.
        """
        self.model_file = model_file
        self.name = name

        # Particle Masses
        self.m4 = None
        self.m5 = None
        self.m6 = None
        self.HNLtype = None

        self.mzprime = None
        self.mhprime = None

        # scalar couplings
        self.s_e4 = None
        self.s_e5 = None
        self.s_e6 = None
        self.s_mu4 = None
        self.s_mu5 = None
        self.s_mu6 = None
        self.s_tau4 = None
        self.s_tau5 = None
        self.s_tau6 = None
        self.s_44 = None
        self.s_45 = None
        self.s_46 = None
        self.s_55 = None
        self.s_56 = None
        self.s_66 = None

        # TMM is always set in a "model-independent" way
        # TMM in GeV^-1
        self.mu_tr_e4 = None
        self.mu_tr_e5 = None
        self.mu_tr_e6 = None
        self.mu_tr_mu4 = None
        self.mu_tr_mu5 = None
        self.mu_tr_mu6 = None
        self.mu_tr_tau4 = None
        self.mu_tr_tau5 = None
        self.mu_tr_tau6 = None
        self.mu_tr_44 = None
        self.mu_tr_45 = None
        self.mu_tr_46 = None
        self.mu_tr_55 = None
        self.mu_tr_56 = None
        self.mu_tr_66 = None

        # Initilize nucleon couplings. These will be filled with the quark combination, which is what is actually set by the user
        self.cprotonV = None
        self.cprotonA = None
        self.cneutronV = None
        self.cneutronA = None
        self.dprotonV = None
        self.dprotonA = None
        self.dneutronV = None
        self.dneutronA = None
        self.dSproton = None
        self.dSneutron = None
        self.dPproton = None
        self.dPneutron = None

    def _initialize_spectrum(self):

        # Initialize spectrum
        self.nu_spectrum = [lp.nu_e, lp.nu_mu, lp.nu_tau]
        self._spectrum = ""
        self.hnl_masses = np.empty(0)

        if self.m4 is not None:
            self.hnl_masses = np.append(self.m4, self.hnl_masses)
            self.neutrino4 = pdg.neutrino4
            self.neutrino4.mass = self.m4
        if self.m5 is not None:
            self.hnl_masses = np.append(self.m5, self.hnl_masses)
            self.neutrino5 = pdg.neutrino5
            self.neutrino5.mass = self.m5
        if self.m6 is not None:
            self.hnl_masses = np.append(self.m6, self.hnl_masses)
            self.neutrino6 = pdg.neutrino6
            self.neutrino6.mass = self.m6

        # define new HNL particles and pass the masses in MeV units (default units for Particle...)
        for i in range(len(self.hnl_masses)):

            # PDGID  =  59(particle spin code: 0-scalar 1-fermion 2-vector)(generation number)
            # GeV units in particle module!
            hnl = pdg.new_particle(
                name=f"N{4+i}",
                pdgid=5914 + i,
                latex_name=f"N_{{{4+i}}}",
                mass=self.hnl_masses[i],
            )
            setattr(self, f"neutrino{4+i}", hnl)
            self.nu_spectrum.append(getattr(self, f"neutrino{4+i}"))

        self.HNL_spectrum = self.nu_spectrum[3:]
        self.n_nus = len(self.nu_spectrum)
        self.n_HNLs = len(self.HNL_spectrum)
        self._spectrum += f"\n\t{self.n_HNLs} {self.HNLtype} heavy neutrino(s)."

    def _update_spectrum(self):

        # mass mixing between Z' and Z
        # NOT YET FUNCTIONAL
        self.is_mass_mixed = False  # (self.epsilonZ != 0.0)

        self.has_Zboson_coupling = np.any(self.c_aj[3:, :] != 0)
        if self.has_Zboson_coupling:
            self._spectrum += f"\n\t{np.sum(self.c_aj[3:,:]!=0)} non-zero Z boson coupling(s) beyond the SM."

        self.has_vector_coupling = np.any(self.d_aj != 0)
        if self.has_vector_coupling:
            # dark photon
            self.zprime = pdg.new_particle(
                name="zprime",
                pdgid=5921,
                mass=self.mzprime * 1e3,
                latex_name="Z^\prime",
            )
            self._spectrum += (
                f"\n\t{np.sum(self.d_aj!=0)} non-zero Z'-neutrino coupling(s)."
            )

        self.has_scalar_coupling = np.any(self.s_aj != 0)
        if self.has_scalar_coupling:
            # dark scalar
            self.hprime = pdg.new_particle(
                name="hprime",
                pdgid=5901,
                mass=self.mhprime * 1e3,
                latex_name="h^\prime",
            )
            self._spectrum += (
                f"\n\t{np.sum(self.s_aj!=0)} non-zero h'-neutrino coupling(s)."
            )

        self.has_TMM = np.any(self.t_aj != 0)
        if self.has_TMM:
            self._spectrum += (
                f"\n\t{np.sum(self.t_aj!=0)} non-zero transition magnetic moment(s)."
            )


class GenericHNLModel(HNLModel):
    def __init__(self, model_file=None, name="my_model"):
        super().__init__(model_file, name)

        # Z boson couplings
        self.c_e4 = None
        self.c_e5 = None
        self.c_e6 = None
        self.c_mu4 = None
        self.c_mu5 = None
        self.c_mu6 = None
        self.c_tau4 = None
        self.c_tau5 = None
        self.c_tau6 = None
        self.c_44 = None
        self.c_45 = None
        self.c_46 = None
        self.c_55 = None
        self.c_56 = None
        self.c_66 = None

        # vector couplings
        self.d_e4 = None
        self.d_e5 = None
        self.d_e6 = None
        self.d_mu4 = None
        self.d_mu5 = None
        self.d_mu6 = None
        self.d_tau4 = None
        self.d_tau5 = None
        self.d_tau6 = None
        self.d_44 = None
        self.d_45 = None
        self.d_46 = None
        self.d_55 = None
        self.d_56 = None
        self.d_66 = None

        ########################
        # Charge particle couplings

        self.ceV = None
        self.ceA = None
        self.cuV = None
        self.cuA = None
        self.cdV = None
        self.cdA = None

        self.deV = None
        self.deA = None
        self.duV = None
        self.duA = None
        self.ddV = None
        self.ddA = None

        self.deS = None
        self.deP = None
        self.duS = None
        self.duP = None
        self.ddS = None
        self.ddP = None

    def set_vertices(self):

        # initialize spectrum of HNLs
        self._initialize_spectrum()

        ####################################################
        # SM Z boson couplings
        self.c_aj = np.array(
            [
                [const.gweak / 2 / const.cw, 0, 0, self.c_e4, self.c_e5, self.c_e6],
                [0, const.gweak / 2 / const.cw, 0, self.c_mu4, self.c_mu5, self.c_mu6],
                [
                    0,
                    0,
                    const.gweak / 2 / const.cw,
                    self.c_tau4,
                    self.c_tau5,
                    self.c_tau6,
                ],
                [0, 0, 0, self.c_44, self.c_45, self.c_46],
                [0, 0, 0, self.c_45, self.c_55, self.c_56],
                [0, 0, 0, self.c_46, self.c_56, self.c_66],
            ]
        )

        ####################################################
        # Z' vector couplings
        self.d_aj = np.array(
            [
                [0, 0, 0, self.d_e4, self.d_e5, self.d_e6],
                [0, 0, 0, self.d_mu4, self.d_mu5, self.d_mu6],
                [0, 0, 0, self.d_tau4, self.d_tau5, self.d_tau6],
                [0, 0, 0, self.d_44, self.d_45, self.d_46],
                [0, 0, 0, self.d_45, self.d_55, self.d_56],
                [0, 0, 0, self.d_46, self.d_56, self.d_66],
            ]
        )

        ####################################################
        # h' scalar couplings
        self.s_aj = np.array(
            [
                [0, 0, 0, self.s_e4, self.s_e5, self.s_e6],
                [0, 0, 0, self.s_mu4, self.s_mu5, self.s_mu6],
                [0, 0, 0, self.s_tau4, self.s_tau5, self.s_tau6],
                [0, 0, 0, self.s_44, self.s_45, self.s_46],
                [0, 0, 0, self.s_45, self.s_55, self.s_56],
                [0, 0, 0, self.s_46, self.s_56, self.s_66],
            ]
        )

        ####################################################
        # create the transition mag moment scope
        self.t_aj = np.array(
            [
                [0, 0, 0, self.mu_tr_e4, self.mu_tr_e5, self.mu_tr_e6],
                [0, 0, 0, self.mu_tr_mu4, self.mu_tr_mu5, self.mu_tr_mu6],
                [0, 0, 0, self.mu_tr_tau4, self.mu_tr_tau5, self.mu_tr_tau6],
                [0, 0, 0, self.mu_tr_44, self.mu_tr_45, self.mu_tr_46],
                [0, 0, 0, self.mu_tr_45, self.mu_tr_55, self.mu_tr_56],
                [0, 0, 0, self.mu_tr_46, self.mu_tr_56, self.mu_tr_66],
            ]
        )

        prettyprinter.info(f"Model:{self._spectrum}")

        ####################################################
        # Nucleon couplings
        # n.b. lepton vertices already defined
        # for TMM, we already know it has to be (e*charge)

        if self.cprotonV is None:
            self.cprotonV = 2 * self.cuV + self.cdV
        if self.cprotonA is None:
            self.cprotonA = 2 * self.cuA + self.cdA
        if self.cneutronV is None:
            self.cneutronV = 2 * self.cdV + self.cuV
        if self.cneutronA is None:
            self.cneutronA = 2 * self.cdA + self.cuA

        if self.dprotonV is None:
            self.dprotonV = 2 * self.duV + self.ddV
        if self.dprotonA is None:
            self.dprotonA = 2 * self.duA + self.ddA
        if self.dneutronV is None:
            self.dneutronV = 2 * self.ddV + self.duV
        if self.dneutronA is None:
            self.dneutronA = 2 * self.ddA + self.duA

        if self.dprotonS is None:
            self.dprotonS = 2 * self.duS + self.ddS
        if self.dneutronS is None:
            self.dneutronS = 2 * self.ddS + self.duS
        if self.dprotonP is None:
            self.dprotonP = 2 * self.duP + self.ddP
        if self.dneutronP is None:
            self.dneutronP = 2 * self.ddP + self.duP

        self._update_spectrum()


class ThreePortalModel(HNLModel):
    def __init__(self, model_file=None):

        super().__init__(model_file)

        self.name = "Untitled"

        self.Ue4 = 0.0
        self.Umu4 = 0.0
        self.Utau4 = 0.0

        self.Ue5 = 0.0
        self.Umu5 = 0.0
        self.Utau5 = 0.0

        self.Ue6 = 0.0
        self.Umu6 = 0.0
        self.Utau6 = 0.0

        self.UD4 = 1.0
        self.UD5 = 1.0
        self.UD6 = 1.0

        # Z'
        self.gD = None
        self.epsilon = None  # kinetic mixing
        self.epsilonZ = 0.0  # mass mixing

        # h'
        self.theta = 0.0  # higgs mixing

    def set_vertices(self):
        """
        set all other variables starting from base members

        """

        # initialize spectrum
        self._initialize_spectrum()

        # create the scalar couplings
        self.s_aj = np.array(
            [
                [0, 0, 0, self.s_e4, self.s_e5, self.s_e6],
                [0, 0, 0, self.s_mu4, self.s_mu5, self.s_mu6],
                [0, 0, 0, self.s_tau4, self.s_tau5, self.s_tau6],
                [0, 0, 0, self.s_44, self.s_45, self.s_46],
                [0, 0, 0, self.s_45, self.s_55, self.s_56],
                [0, 0, 0, self.s_46, self.s_56, self.s_66],
            ]
        )

        # create the transition mag moment scope
        self.t_aj = np.array(
            [
                [0, 0, 0, self.mu_tr_e4, self.mu_tr_e5, self.mu_tr_e6],
                [0, 0, 0, self.mu_tr_mu4, self.mu_tr_mu5, self.mu_tr_mu6],
                [0, 0, 0, self.mu_tr_tau4, self.mu_tr_tau5, self.mu_tr_tau6],
                [0, 0, 0, self.mu_tr_44, self.mu_tr_45, self.mu_tr_46],
                [0, 0, 0, self.mu_tr_45, self.mu_tr_55, self.mu_tr_56],
                [0, 0, 0, self.mu_tr_46, self.mu_tr_56, self.mu_tr_66],
            ]
        )

        prettyprinter.info(f"Model:{self._spectrum}")

        #### Assign the correct value of kinetic mixing and gD given user input

        #### Kinetic mixing
        if self.chi is not None:
            self.epsilon = self.epsilon * const.cw
        elif self.epsilon2 is not None:
            self.epsilon = np.sqrt(self.epsilon2)
            self.chi = self.epsilon / const.cw
        elif self.alpha_epsilon2 is not None:
            self.epsilon = np.sqrt(self.alpha_epsilon2 / const.alphaQED)
            self.chi = self.epsilon / const.cw
        elif self.epsilon is not None:
            self.chi = self.epsilon / const.cw

        # dark coupling
        if self.alphaD is not None:
            self.gD = np.sqrt(4 * np.pi * self.alphaD)
        elif self.gD is not None:
            pass

        #### CHARGED FERMION VERTICES
        # all the following is true to leading order in chi
        self.tanchi = math.tan(self.chi)
        self.sinof2chi = 2 * self.tanchi / (1.0 + self.tanchi**2)
        self.cosof2chi = (1.0 - self.tanchi**2) / (1.0 + self.tanchi**2)
        self.s2chi = (1.0 - self.cosof2chi) / 2.0
        self.c2chi = 1 - self.s2chi

        entry_22 = self.c2chi - const.s2w * self.s2chi - (self.mzprime / const.m_Z) ** 2
        self.tanof2beta = const.sw * self.sinof2chi / (entry_22)
        self.beta = const.sw * self.chi
        self.sinof2beta = (
            const.sw
            * self.sinof2chi
            / np.sqrt(entry_22**2 + self.sinof2chi**2 * const.s2w)
        )
        self.cosof2beta = entry_22 / np.sqrt(
            entry_22**2 + self.sinof2chi**2 * const.s2w
        )

        ######################
        if self.tanof2beta != 0:
            self.tbeta = self.sinof2beta / (1 + self.cosof2beta)
        else:
            self.tbeta = 0.0
        ######################

        self.sbeta = math.sqrt((1 - self.cosof2beta) / 2.0) * np.sign(
            self.sinof2beta
        )  # FIX ME -- works only for |beta| < pi/2
        self.cbeta = math.sqrt(
            (1 + self.cosof2beta) / 2.0
        )  # FIX ME -- works only for |beta| < pi/2

        # some abbreviations
        self._weak_vertex = const.gweak / const.cw / 2.0
        self._gschi = self.gD * self.sbeta
        # dark couplings acquired by Z boson
        self._g_weak_correction = self.cbeta + self.tanchi * const.sw * self.sbeta
        self._g_dark_correction = self.cbeta * self.tanchi * const.sw - self.sbeta

        # Charged leptons
        self.ceV = self._weak_vertex * (
            self.cbeta * (2 * const.s2w - 0.5)
            + 3.0 / 2.0 * self.sbeta * const.sw * self.tanchi
        )
        self.ceA = self._weak_vertex * (
            -(self.cbeta + self.sbeta * const.sw * self.tanchi) / 2.0
        )
        # self.ceV = weak_vertex*(const.gweak/(2*const.cw) * (2*const.s2w - 0.5))
        # self.ceA = weak_vertex*(const.gweak/(2*const.cw) * (-1.0/2.0))

        # quarks
        self.cuV = self._weak_vertex * (
            self.cbeta * (0.5 - 4 * const.s2w / 3.0)
            - 5.0 / 6.0 * self.sbeta * const.sw * self.tanchi
        )
        self.cuA = self._weak_vertex * (
            (self.cbeta + self.sbeta * const.sw * self.tanchi) / 2.0
        )

        self.cdV = self._weak_vertex * (
            self.cbeta * (-0.5 + 2 * const.s2w / 3.0)
            + 1.0 / 6.0 * self.sbeta * const.sw * self.tanchi
        )
        self.cdA = self._weak_vertex * (
            -(self.cbeta + self.sbeta * const.sw * self.tanchi) / 2.0
        )

        self.deV = self._weak_vertex * (
            3.0 / 2.0 * self.cbeta * const.sw * self.tanchi
            - self.sbeta * (-0.5 + 2 * const.s2w)
        )
        self.deA = self._weak_vertex * (
            (self.sbeta - self.cbeta * const.sw * self.tanchi) / 2.0
        )
        # self.deV = const.gweak/(2*const.cw) * 2*const.sw*const.cw**2*self.chi
        # self.deA = const.gweak/(2*const.cw) * 0

        self.duV = -self._weak_vertex * (
            -5.0 / 6.0 * self.cbeta * const.sw * self.tanchi
            - self.sbeta * (0.5 - 4.0 / 3.0 * const.s2w)
        )
        self.duA = self._weak_vertex * (
            (-self.sbeta + self.cbeta * const.sw * self.tanchi) / 2.0
        )

        self.ddV = -self._weak_vertex * (
            -self.sbeta * (-0.5 + 2 / 3.0 * const.s2w)
            + 1.0 / 6.0 * self.cbeta * const.sw * self.tanchi
        )
        self.ddA = self._weak_vertex * (
            (self.sbeta - self.cbeta * const.sw * self.tanchi) / 2.0
        )

        self.dprotonV = 2 * self.duV + self.ddV
        self.dprotonA = 2 * self.duA + self.ddA
        self.dneutronV = 2 * self.ddV + self.duV
        self.dneutronA = 2 * self.ddA + self.duA

        self.cprotonV = 2 * self.cuV + self.cdV
        self.cprotonA = 2 * self.cuA + self.cdA
        self.cneutronV = 2 * self.cdV + self.cuV
        self.cneutronA = 2 * self.cdA + self.cuA

        ####################################################
        # NEUTRAL FERMION VERTICES
        self.Ue = [1, 0, 0, self.Ue4, self.Ue5, self.Ue6]
        self.Umu = [0, 1, 0, self.Umu4, self.Umu5, self.Umu6]
        self.Utau = [0, 0, 1, self.Utau4, self.Utau5, self.Utau6]
        self.Udark = [0, 0, 0, self.UD4, self.UD5, self.UD6]

        ##### FIX-ME -- expand to arbitrary number of dark flavors?
        self.n_dark_HNLs = 1  # self.n_HNLs
        # list of dark flavors
        self.inds_dark = range(const.ind_tau + 1, 3 + self.n_dark_HNLs)

        # Mixing matrices
        # if PMNS, use dtype=complex
        self.Ulep = np.diag(np.full_like(self.Ue, 1))
        # self.Ulep = np.diag(np.full(self.n_nus,1,dtype=complex))
        # self.Ulep[:3,:3] = const.UPMNS # massless light neutrinos

        # loop over HNL indices
        for i in range(3, self.n_HNLs + 3):
            self.Ulep[const.ind_e, i] = self.Ue[i]
            self.Ulep[i, const.ind_e] = self.Ue[i]

            self.Ulep[const.ind_mu, i] = self.Umu[i]
            self.Ulep[i, const.ind_mu] = self.Umu[i]

            self.Ulep[const.ind_tau, i] = self.Utau[i]
            self.Ulep[i, const.ind_tau] = self.Utau[i]

            self.Ulep[self.inds_dark, i] = self.Udark[i] / self.n_dark_HNLs
            self.Ulep[i, self.inds_dark] = self.Udark[i] / self.n_dark_HNLs

        self.Ua = self.Ulep[const.inds_active, :]
        self.Uactive_heavy = self.Ulep[const.inds_active, 3:]

        self.UD = self.Ulep[self.inds_dark, :]
        self.UD_heavy = self.Ulep[self.inds_dark, 3:]

        ### Matrix
        # (Ua^dagger . Ua)_ij = Uei Uej + Umui Umuj + Utaui Utauj
        self.C_weak = np.dot(self.Ua.conjugate().T, self.Ua)
        # (UDi^dagger . UD)_ij
        self.D_dark = np.dot(self.UD.conjugate().T, self.UD)

        ### Vectors
        # ( |Ua4|^2, |Ua5|^2, |Ua6|^2, ...)
        self.UactiveUactive_diag = np.diagonal(
            np.dot(self.Uactive_heavy.conjugate(), self.Uactive_heavy.T)
        )
        # ( |UD4|^2, |UD5|^2, |UD6|^2, ...)
        self.UDUD_diag = np.diagonal(np.dot(self.UD_heavy.conjugate(), self.UD_heavy.T))
        # (Ua4* UD4, Ua5* UD5, Ua6* UD6,..)
        self.UactiveUD_diag = np.diagonal(
            np.dot(self.Uactive_heavy.conjugate(), self.UD_heavy.T)
        )

        # ( |Ue4|^2 +  |Ue5|^2 + ... , |Umu4|^2 + |Umu5|^2 + ..., |Utau4|^2 + |Utau5|^2 + ...)
        self.UahUah_mass_summed = np.sum(
            np.dot(self.Uactive_heavy.conjugate(), self.Uactive_heavy.T), axis=0
        )
        # (Ue4* UD4 + Ue5* UD5+... , Umu4* Umu4 + Umu5* Umu5+..., Utau4* Utau4 + Utau5* Utau5+...)
        self.UDhUDh_mass_summed = np.sum(
            np.dot(self.UD_heavy.conjugate(), self.UD_heavy.T), axis=0
        )
        # (Ue4* UD4 + Ue5* UD5+... , Umu4* Umu4 + Umu5* Umu5+..., Utau4* Utau4 + Utau5* Utau5+...)
        self.UahUDh_mass_summed = np.sum(
            np.dot(self.Uactive_heavy.conjugate(), self.UD_heavy.T), axis=0
        )

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

        self.D4 = self.UD4**2  # self.UD4#(1.0 - self.A4 - self.A5)/(1.0+self.R)
        self.D5 = self.UD5**2  # self.UD5#(1.0 - self.A4 - self.A5)/(1.0+1.0/self.R)
        self.D6 = self.UD6**2  # self.UD6#(1.0 - self.A4 - self.A5)/(1.0+1.0/self.R)

        # NEUTRAL LEPTON SECTOR VERTICES
        self.c_ij = (
            self._weak_vertex * self.C_weak * self._g_weak_correction
            + self.D_dark * self._gschi
        )

        self.c_aj = (
            self._weak_vertex
            * np.dot(np.diag(1 - self.UahUah_mass_summed), self.Uactive_heavy)
            * self._g_weak_correction
            + np.dot(np.diag(-self.UahUDh_mass_summed), self.UD_heavy) * self._gschi
        )

        self.d_ij = (
            self._weak_vertex * self.C_weak * self._g_dark_correction
            + self.D_dark * self.gD
        )
        self.d_aj = (
            self._weak_vertex
            * np.dot(np.diag(1 - self.UahUah_mass_summed), self.Uactive_heavy)
            * self._g_dark_correction
            + np.dot(np.diag(-self.UahUDh_mass_summed), self.UD_heavy) * self.gD
        )

        # make it 3 x n_nus
        self.c_aj = np.hstack((np.diag([1, 1, 1]), self.c_aj))
        self.d_aj = np.hstack((np.diag([1, 1, 1]), self.d_aj))

        # make n_nus x n_nus
        self.c_aj = np.vstack((self.c_aj, self.c_ij[3:]))
        self.d_aj = np.vstack((self.d_aj, self.d_ij[3:]))
        self.dlight = 0.0

        #########################
        # Scalar couplings
        self.tantheta = np.tan(self.theta)
        self.sintheta = np.sin(self.theta)
        self.costheta = np.cos(self.theta)

        #### light quark couplings determine higgs coupling to nucleon
        # see e.g. arxiv.org/abs/1306.4710
        self.sigma_l = 0.058  # GeV
        self.sigma_0 = 0.055  # GeV
        z = 1.49  # isospin breaking parameter
        y = 1 - self.sigma_0 / self.sigma_l
        _prefactor = 1 / (const.m_u + const.m_d) * self.sigma_l / const.m_avg
        self.fu = _prefactor * const.m_u * (2 * z + y * (1 - z)) / (1 + z)
        self.fd = _prefactor * const.m_d * (2 - y * (1 - z)) / (1 + z)
        self.fs = _prefactor * const.m_s * y

        self.fN_higgs = 2 / 9 + 7 / 9 * (self.fu + self.fd + self.fs)
        self.c_nucleon_higgs = self.fN_higgs * const.m_avg / const.vev_EW

        self.cnucleonS = self.sintheta * self.c_nucleon_higgs
        # isospin
        self.dprotonS = self.cnucleonS
        self.dneutronS = self.cnucleonS
        self.ceS = self.costheta * const.m_e / const.vev_EW / np.sqrt(2)
        self.deS = self.sintheta * const.m_e / const.vev_EW / np.sqrt(2)

        # no pseudo-scalar coupling
        self.deP = 0.0

        self._update_spectrum()

    # def compute_rates(self):

    #         ## FIX ME -- GENERALIZE TO N5 and N6
    #         ##################
    #         # Neutrino 4
    #         mh = self.m4
    #         rates = {}
    #         neutrinos = [lp.nu_e, lp.nu_mu, lp.nu_tau]

    #         # channels with 3 neutrinos in final state
    #         rates['nu_nu_nu'] = 0.0
    #         for nu_a in neutrinos:
    #             rates['nu_nu_nu'] += dr.nui_nuj_nuk_nuk(self, const.N4, nu_a)

    #         # channels with 1 neutrino in final states
    #         rates['nu_gamma'] = 0
    #         rates['nu_e_e'] = 0
    #         rates['nu_mu_mu'] = 0
    #         rates['nu_e_mu'] = 0
    #         rates['nu_pi'] = 0
    #         rates['nu_eta'] = 0
    #         rates['e_pi'] = 0
    #         rates['e_K'] = 0
    #         rates['mu_pi'] = 0
    #         rates['mu_K'] = 0

    #         for nu_a in neutrinos:          # nu gamma
    #             rates['nu_gamma'] += dr.nui_nuj_gamma(self, const.N4, nu_a)
    #             # dileptons -- already contains the Delta L = 2 channel
    #             if mh > 2*lp.e_minus.mass/1e3:
    #                 rates['nu_e_e'] += dr.nui_nuj_ell1_ell2(self, const.N4, nu_a, lp.e_minus, lp.e_plus)
    #             if mh > lp.e_minus.mass/1e3 + lp.mu_minus.mass/1e3:
    #                 rates['nu_e_mu'] += dr.nui_nuj_ell1_ell2(self, const.N4, nu_a, lp.e_minus, lp.mu_plus)
    #                 rates['nu_e_mu'] += dr.nui_nuj_ell1_ell2(self, const.N4, nu_a, lp.mu_minus, lp.e_plus)
    #             if mh > 2*lp.mu_minus.mass/1e3:
    #                 rates['nu_mu_mu'] += dr.nui_nuj_ell1_ell2(self, const.N4, nu_a, lp.mu_minus, lp.mu_plus)
    #             # pseudoscalar -- neutral current
    #             if mh > lp.pi_0.mass/1e3:
    #                 rates['nu_pi'] += dr.nui_nu_P(self, const.N4, nu_a, lp.pi_0)
    #             if mh > lp.eta.mass/1e3:
    #                 rates['nu_eta'] += dr.nui_nu_P(self, const.N4, nu_a, lp.eta)

    #         # CC-only channels
    #         # pseudoscalar -- factor of 2 for delta L=2 channel
    #         if mh > lp.e_minus.mass/1e3+lp.pi_plus.mass/1e3:
    #             rates['e_pi'] = dr.nui_l_P(self, const.N4, lp.e_minus, lp.pi_plus)
    #         if mh > lp.e_minus.mass/1e3+lp.K_plus.mass/1e3:
    #             rates['e_K'] = dr.nui_l_P(self, const.N4, lp.e_minus, lp.K_plus)

    #         # pseudoscalar -- already contain the Delta L = 2 channel
    #         if mh > lp.mu_minus.mass/1e3+lp.pi_plus.mass/1e3:
    #             rates['mu_pi'] = dr.nui_l_P(self, const.N4, lp.mu_minus, lp.pi_plus)
    #         if mh > lp.mu_minus.mass/1e3+lp.K_plus.mass/1e3:
    #             rates['mu_K'] = dr.nui_l_P(self, const.N4, lp.mu_minus, lp.K_plus)

    #         self.rates = rates

    #         # total decay rate
    #         self.rate_total = sum(self.rates.values())

    #         # total decay rate
    #         self.lifetime = const.get_decay_rate_in_s(self.rate_total)
    #         self.ctau0 = const.get_decay_rate_in_cm(self.rate_total)

    #         # Branchin ratios
    #         brs = {}
    #         for channel in self.rates.keys():
    #             brs[channel] = self.rates[channel]/self.rate_total
    #         self.brs = brs


# class HNLparticle():

#     def __init__(self, this_hnl, bsm_model):

#         self.this_hnl = this_hnl
#         self.bsm_model = bsm_model

#         # Dirac or Majorana
#         self.HNLtype    = bsm_model.HNLtype

#         self.mHNL = bsm_model.masses[get_HNL_index(this_hnl)]


#     def _setup_rates(self):

#         self.rates = {}
#         self.daughter_neutrinos = self.bsm_model.nu_spectrum[:pdg.get_HNL_index(self.this_hnl)]


#         # Setting all decay BRs to 0
#         for nu_i in self.daughter_neutrinos:
#             for nu_j in self.daughter_neutrinos:
#                 for nu_k in self.daughter_neutrinos:
#                     self.rates[f'{nu_i.name}_{nu_j.name}_{nu_k.name}'] = 0
#             self.rates[f'{nu_i.name}_{lp.gamma.name}']                     = 0
#             self.rates[f'{nu_i.name}_{lp.e_plus.name}_{lp.e_minus.name}']  = 0
#             self.rates[f'{nu_i.name}_{lp.mu_plus.name}_{lp.e_minus.name}'] = 0
#             self.rates[f'{nu_i.name}_{lp.e_plus.name}_{lp.mu_minus.name}'] = 0
#             self.rates[f'{nu_i.name}_{mu_plus.name}_{mu_minus.name}']      = 0
#             self.rates[f'{nu_i.name}_{lp.pi_0.name}']                      = 0
#             self.rates[f'{nu_i.name}_{lp.eta.name}']                       = 0
#             self.rates[f'{lp.e_minus.name}_{lp.pi_plus.name}']             = 0
#             self.rates[f'{lp.e_plus.name}_{lp.pi_minus.name}']             = 0
#             self.rates[f'{lp.e_minus.name}_{lp.K_plus.name}']              = 0
#             self.rates[f'{lp.e_plus.name}_{lp.K_minus.name}']              = 0
#             self.rates[f'{lp.mu_minus.name}_{lp.pi_plus.name}']            = 0
#             self.rates[f'{lp.mu_plus.name}_{lp.pi_minus.name}']            = 0
#             self.rates[f'{lp.mu_minus.name}_{lp.K_plus.name}']             = 0
#             self.rates[f'{lp.mu_plus.name}_{lp.K_minus.name}']             = 0


#     def compute_rates(self):


#         #################
#         mh = self.mHNL

#         # channels with 3 neutrinos in final state
#         for nu_i in self.daughter_neutrinos:
#             for nu_j in self.daughter_neutrinos:
#                 for nu_k in self.daughter_neutrinos:
#                     self.rates[f'{nu_i.name}_{nu_j.name}_{nu_k.name}'] += dr.new_nuh_nui_nuj_nuk(
#                                                                     self.bsm_model,
#                                                                     initial_neutrino=self.particle,
#                                                                     final_neutrinoi=nu_a,
#                                                                     final_neutrinoj=nu_b,
#                                                                     final_neutrinok=nu_c)
#         for nu_i in self.daughter_neutrinos:
#             # nu gamma
#             self.rates[f'{nu_i.name}_{lp.gamma.name}'] += dr.nui_nuj_gamma(self.bsm_model, self.this_hnl, nu_i)
#             # dileptons -- already contains the Delta L = 2 channel
#             # e+e-
#             if mh > 2*lp.e_minus.mass/1e3:
#                 self.rates[f'{nu_i.name}_{lp.e_plus.name}_{lp.e_minus.name}'] += dr.nui_nuj_ell1_ell2(self.bsm_model, self.this_hnl, nu_i, lp.e_minus, lp.e_plus)
#             # e+ mu- and # e- mu+
#             if mh > lp.e_minus.mass/1e3 + lp.mu_minus.mass/1e3:
#                 self.rates[f'{nu_i.name}_{lp.mu_plus.name}_{lp.e_minus.name}'] += dr.nui_nuj_ell1_ell2(self.bsm_model, self.this_hnl, nu_i, lp.e_minus, lp.mu_plus)
#                 self.rates[f'{nu_i.name}_{lp.e_plus.name}_{lp.mu_minus.name}'] += dr.nui_nuj_ell1_ell2(self.bsm_model, self.this_hnl, nu_i, lp.mu_minus, lp.e_plus)
#             # mu+ mu-
#             if mh > 2*lp.mu_minus.mass/1e3:
#                 self.rates[f'{nu_i.name}_{mu_plus.name}_{mu_minus.name}'] += dr.nui_nuj_ell1_ell2(self.bsm_model, self.this_hnl, nu_i, lp.mu_minus, lp.mu_plus)
#             # pseudoscalar -- neutral current
#             if mh > lp.pi_0.mass/1e3:
#                 self.rates[f'{nu_i.name}_{lp.pi_0.name}'] += dr.nui_nu_P(self.bsm_model, self.this_hnl, nu_i, lp.pi_0)
#             if mh > lp.eta.mass/1e3:
#                 self.rates[f'{nu_i.name}_{lp.eta.name}'] += dr.nui_nu_P(self.bsm_model, self.this_hnl, nu_i, lp.eta)


#         # CC-only channels
#         # pseudoscalar
#         if mh > lp.e_minus.mass/1e3+lp.pi_plus.mass/1e3:
#             self.rates[f'{lp.e_minus.name}_{lp.pi_plus.name}'] = nui_l_P(self.bsm_model, self.this_hnl, lp.e_minus, lp.pi_plus)
#             self.rates[f'{lp.e_plus.name}_{lp.pi_minus.name}'] = nui_l_P(self.bsm_model, self.this_hnl, lp.e_plus, lp.pi_minus)
#         if mh > lp.e_minus.mass/1e3+lp.K_plus.mass/1e3:
#             self.rates[f'{lp.e_minus.name}_{lp.K_plus.name}'] = nui_l_P(self.bsm_model, self.this_hnl, lp.e_minus, lp.K_plus)
#             self.rates[f'{lp.e_plus.name}_{lp.K_minus.name}'] = nui_l_P(self.bsm_model, self.this_hnl, lp.e_plus, lp.K_minus)

#         # pseudoscalar
#         if mh > lp.mu_minus.mass/1e3+lp.pi_plus.mass/1e3:
#             self.rates[f'{lp.mu_minus.name}_{lp.pi_plus.name}'] = nui_l_P(self.bsm_model, self.this_hnl, lp.mu_minus, lp.pi_plus)
#             self.rates[f'{lp.mu_plus.name}_{lp.pi_minus.name}'] = nui_l_P(self.bsm_model, self.this_hnl, lp.mu_plus, lp.pi_minus)
#         if mh > lp.mu_minus.mass/1e3+lp.K_plus.mass/1e3:
#             self.rates[f'{lp.mu_minus.name}_{lp.K_plus.name}'] = nui_l_P(self.bsm_model, self.this_hnl, lp.mu_minus, lp.K_plus)
#             self.rates[f'{lp.mu_plus.name}_{lp.K_minus.name}'] = nui_l_P(self.bsm_model, self.this_hnl, lp.mu_plus, lp.K_minus)


#         # total decay rate
#         self.rate_total = sum(self.rates.values())

#         # total decay rate
#         self.lifetime = const.get_decay_rate_in_s(self.rate_total)
#         self.ctau0 = const.get_decay_rate_in_cm(self.rate_total)

#         # branching ratios
#         brs = {}
#         for channel in self.rates.keys():
#             brs[channel] = self.rates[channel]/self.rate_total
#         self.brs = brs
