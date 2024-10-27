import numpy as np
import vegas as vg
import json

from DarkNews import const
from DarkNews import pdg
from DarkNews import integrands
from DarkNews import MC
from DarkNews import model
from DarkNews import amplitudes as amps
from DarkNews import phase_space as ps
from DarkNews import decay_rates as dr

from . import Cfourvec as Cfv

import logging

logger = logging.getLogger("logger." + __name__)


class UpscatteringProcess:
    def __init__(self, nu_projectile, nu_upscattered, nuclear_target, scattering_regime, TheoryModel, helicity):
        """
        A class to describe the process of neutrino upscattering, which involves a neutrino scattering off a target and gaining energy in the process.
        This class supports various scattering regimes (coherent, proton elastic, and neutron elastic), and allows for the calculation of total and differential cross sections for these processes.

        Attributes:
            nuclear_target (object): The nuclear target involved in the scattering process.
            scattering_regime (str): The regime of scattering, e.g., 'coherent', 'p-el' (proton elastic), 'n-el' (neutron elastic).
            target (object): The actual target of the scattering, which could be the whole nucleus, a constituent nucleon, or constituent quarks, depending on the scattering regime.
            target_multiplicity (int): The number of targets involved in the scattering process, relevant for calculating cross sections.
            nu_projectile (object): The incoming neutrino involved in the upscattering process.
            nu_upscattered (object): The upscattered neutrino resulting from the scattering process.
            TheoryModel (object): The theoretical model used to describe the interactions in the upscattering process.
            helicity (str): The helicity configuration of the upscattering process, either 'conserving' or 'flipping'.
            MA (float): The mass of the target involved in the scattering.
            mzprime (float): The mass of the Z' boson in the theory model, if applicable.
            mhprime (float): The mass of the H' boson in the theory model, if applicable.
            m_ups (float): The mass of the upscattered neutrino.
            Cij, Cji, Vij, Vji, Sij, Sji, Tij, Tji (float): Coupling constants for the interaction vertices involved in the upscattering process.
            Chad, Vhad, Shad (float): Hadronic coupling constants for the interaction vertices.
            Cprimehad (float): Mass-mixed vertex coupling constant for the hadronic interaction.
            Ethreshold (float): The minimum energy threshold for the upscattering process to occur.
            vectorized_total_xsec (function): A vectorized function to calculate the total cross section for the upscattering process.
            calculable_diagrams (list): A list of diagrams that can be calculated for the upscattering process.

        Methods:
            __init__(self, nu_projectile, nu_upscattered, nuclear_target, scattering_regime, TheoryModel, helicity): Initializes the upscattering process with specified parameters.
            scalar_total_xsec(self, Enu, diagram="total", NINT=MC.NINT, NEVAL=MC.NEVAL, NINT_warmup=MC.NINT_warmup, NEVAL_warmup=MC.NEVAL_warmup, savefile_xsec=None, savefile_norm=None): Calculates the scalar total cross section for a given neutrino energy and diagram.
            total_xsec(self, Enu, diagrams=["total"], NINT=MC.NINT, NEVAL=MC.NEVAL, NINT_warmup=MC.NINT_warmup, NEVAL_warmup=MC.NEVAL_warmup, seed=None, savestr=None): Calculates the total cross section for the upscattering process for a fixed neutrino energy.
            diff_xsec_Q2(self, Enu, Q2, diagrams=["total"]): Calculates the differential cross section for the upscattering process as a function of the squared momentum transfer Q2.
        """

        self.nuclear_target = nuclear_target
        self.scattering_regime = scattering_regime
        if self.scattering_regime == "coherent":
            self.target = self.nuclear_target
        elif self.scattering_regime == "p-el":
            self.target = self.nuclear_target.get_constituent_nucleon("proton")
        elif self.scattering_regime == "n-el":
            self.target = self.nuclear_target.get_constituent_nucleon("neutron")
        elif self.scattering_regime == "DIS":
            self.target = self.nuclear_target.get_constituent_quarks()
        else:
            logger.error(f"Scattering regime {scattering_regime} not supported.")

        # How many constituent targets inside scattering regime?
        if self.scattering_regime == "coherent":
            self.target_multiplicity = 1
        elif self.scattering_regime == "p-el":
            self.target_multiplicity = self.nuclear_target.Z
        elif self.scattering_regime == "n-el":
            self.target_multiplicity = self.nuclear_target.N
        elif self.scattering_regime == "DIS":
            self.target_multiplicity = self.nuclear_target.N * self.target.in_neutron + self.nuclear_target.Z * self.target.in_proton
        else:
            logger.error(f"Scattering regime {self.scattering_regime} not supported.")

        self.nu_projectile = nu_projectile
        self.nu_upscattered = nu_upscattered
        self.TheoryModel = TheoryModel
        self.helicity = helicity

        self.MA = self.target.mass
        self.mzprime = TheoryModel.mzprime if TheoryModel.mzprime is not None else 1e10
        self.mhprime = TheoryModel.mhprime if TheoryModel.mhprime is not None else 1e10
        self.m_ups = self.nu_upscattered.mass

        if self.helicity == "conserving":
            self.h_upscattered = -1
        elif self.helicity == "flipping":
            self.h_upscattered = +1
        else:
            logger.error(f"Error! Could not find helicity case {self.helicity}")

        self.Cij = TheoryModel.c_aj[pdg.get_lepton_index(nu_projectile), pdg.get_HNL_index(nu_upscattered)]
        self.Cji = self.Cij
        self.Vij = TheoryModel.d_aj[pdg.get_lepton_index(nu_projectile), pdg.get_HNL_index(nu_upscattered)]
        self.Vji = self.Vij
        self.Sij = TheoryModel.s_aj[pdg.get_lepton_index(nu_projectile), pdg.get_HNL_index(nu_upscattered)]
        self.Sji = self.Sij
        self.Tij = TheoryModel.t_aj[pdg.get_lepton_index(nu_projectile), pdg.get_HNL_index(nu_upscattered)]
        self.Tji = self.Tij

        ###############
        # Hadronic vertices
        if self.target.is_nucleus:
            self.Chad = TheoryModel.cprotonV * self.target.Z + TheoryModel.cneutronV * self.target.N
            self.Vhad = TheoryModel.dprotonV * self.target.Z + TheoryModel.dneutronV * self.target.N
            self.Shad = TheoryModel.dprotonS * self.target.Z + TheoryModel.dneutronS * self.target.N
        elif self.target.is_proton:
            self.Chad = TheoryModel.cprotonV
            self.Vhad = TheoryModel.dprotonV
            self.Shad = TheoryModel.dprotonS
        elif self.target.is_neutron:
            self.Chad = TheoryModel.cneutronV
            self.Vhad = TheoryModel.dneutronV
            self.Shad = TheoryModel.dneutronS

        # If three portal model, set mass-mixed vertex
        if isinstance(TheoryModel, model.ThreePortalModel):
            # mass mixed vertex
            self.Cprimehad = self.Chad * TheoryModel.epsilonZ
        else:
            self.Cprimehad = 0.0

        # Neutrino energy threshold
        self.Ethreshold = self.m_ups**2 / 2.0 / self.MA + self.m_ups

        # vectorize total cross section calculator using vegas integration
        self.vectorized_total_xsec = np.vectorize(
            self.scalar_total_xsec, excluded=["self", "diagram", "NINT", "NEVAL", "NINT_warmup", "NEVAL_warmup", "savefile_xsec", "savefile_norm"]
        )

        self.calculable_diagrams = find_calculable_diagrams(TheoryModel)

    def scalar_total_xsec(
        self,
        Enu,
        diagram="total",
        NINT=MC.NINT,
        NEVAL=MC.NEVAL,
        NINT_warmup=MC.NINT_warmup,
        NEVAL_warmup=MC.NEVAL_warmup,
        savefile_xsec=None,
        savefile_norm=None,
    ):
        # below threshold
        if Enu < (self.Ethreshold):
            return 0.0
        else:
            DIM = 1
            batch_f = integrands.UpscatteringXsec(dim=DIM, Enu=Enu, ups_case=self, diagram=diagram)
            integ = vg.Integrator(DIM * [[0.0, 1.0]])  # unit hypercube

            if savefile_norm is not None:
                # Save normalization information
                with open(savefile_norm, "w") as f:
                    json.dump(batch_f.norm, f)
            integrals = MC.run_vegas(
                batch_f, integ, adapt_to_errors=True, NINT=NINT, NEVAL=NEVAL, NINT_warmup=NINT_warmup, NEVAL_warmup=NEVAL_warmup, savestr=savefile_xsec
            )
            logger.debug("Main VEGAS run completed.")

            return integrals["diff_xsec"].mean * batch_f.norm["diff_xsec"]

    def total_xsec(
        self, Enu, diagrams=["total"], NINT=MC.NINT, NEVAL=MC.NEVAL, NINT_warmup=MC.NINT_warmup, NEVAL_warmup=MC.NEVAL_warmup, seed=None, savestr=None
    ):
        """
        Returns the total upscattering xsec for a fixed neutrino energy in cm^2
        """

        if seed is not None:
            np.random.seed(seed)

        self.Enu = Enu
        all_xsecs = 0.0
        for diagram in diagrams:
            if diagram in self.calculable_diagrams or diagram == "total":
                tot_xsec = self.vectorized_total_xsec(
                    Enu, diagram=diagram, NINT=NINT, NEVAL=NEVAL, NINT_warmup=NINT_warmup, NEVAL_warmup=NEVAL_warmup, savefile_xsec=savestr
                )
            else:
                logger.warning(f"Warning: Diagram not found. Either not implemented or misspelled. Setting tot xsec it to zero: {diagram}")
                tot_xsec = 0.0 * Enu

            #############
            # integrated xsec coverted to cm^2
            all_xsecs += tot_xsec * self.target_multiplicity
            logger.debug(f"Total cross section for {diagram} calculated.")

        return all_xsecs

    def diff_xsec_Q2(self, Enu, Q2, diagrams=["total"]):
        """
        Returns the differential upscattering xsec for a fixed neutrino energy in cm^2
        """
        s = Enu * self.MA * 2 + self.MA**2
        physical = (Q2 > ps.upscattering_Q2min(Enu, self.m_ups, self.MA)) & (Q2 < ps.upscattering_Q2max(Enu, self.m_ups, self.MA))
        u = 2 * self.MA**2 + self.m_ups - s + Q2
        diff_xsecs = amps.upscattering_dxsec_dQ2([s, -Q2, u], process=self, diagrams=diagrams)
        if type(diff_xsecs) is dict:
            return {key: diff_xsecs[key] * physical for key in diff_xsecs.keys()}
        else:
            return diff_xsecs * physical * self.target_multiplicity


class FermionDileptonDecay:
    def __init__(self, nu_parent, nu_daughter, final_lepton1, final_lepton2, TheoryModel, h_parent=-1):

        self.TheoryModel = TheoryModel
        self.HNLtype = TheoryModel.HNLtype
        self.h_parent = h_parent

        self.nu_parent = nu_parent
        self.nu_daughter = nu_daughter
        self.secondaries = [final_lepton1, final_lepton2]

        # particle masses
        self.mzprime = TheoryModel.mzprime if TheoryModel.mzprime is not None else 1e10
        self.mhprime = TheoryModel.mhprime if TheoryModel.mhprime is not None else 1e10
        self.mm = final_lepton1.mass * const.MeV_to_GeV
        self.mp = final_lepton2.mass * const.MeV_to_GeV

        # Neutral lepton vertices
        if nu_daughter == pdg.nulight:
            self.Cih = np.sqrt(np.sum(np.abs(TheoryModel.c_aj[const.inds_active, pdg.get_HNL_index(nu_parent)]) ** 2))
            self.Dih = np.sqrt(np.sum(np.abs(TheoryModel.d_aj[const.inds_active, pdg.get_HNL_index(nu_parent)]) ** 2))
            self.Sih = np.sqrt(np.sum(np.abs(TheoryModel.s_aj[const.inds_active, pdg.get_HNL_index(nu_parent)]) ** 2))
            self.Tih = np.sqrt(np.sum(np.abs(TheoryModel.t_aj[const.inds_active, pdg.get_HNL_index(nu_parent)]) ** 2))
        else:
            self.Cih = TheoryModel.c_aj[pdg.get_HNL_index(nu_daughter), pdg.get_HNL_index(nu_parent)]
            self.Dih = TheoryModel.d_aj[pdg.get_HNL_index(nu_daughter), pdg.get_HNL_index(nu_parent)]
            self.Sih = TheoryModel.s_aj[pdg.get_HNL_index(nu_daughter), pdg.get_HNL_index(nu_parent)]
            self.Tih = TheoryModel.t_aj[pdg.get_HNL_index(nu_daughter), pdg.get_HNL_index(nu_parent)]

        # Charged lepton vertices
        self.Cv = TheoryModel.ceV
        self.Ca = TheoryModel.ceA
        self.Dv = TheoryModel.deV
        self.Da = TheoryModel.deA
        self.Ds = TheoryModel.deS
        self.Dp = TheoryModel.deP

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
        if pdg.in_same_doublet(nu_daughter, final_lepton1):
            self.CC_mixing1 = TheoryModel.Ulep[pdg.get_lepton_index(final_lepton1), pdg.get_HNL_index(nu_parent)]
            self.CC_mixing2 = TheoryModel.Ulep[pdg.get_lepton_index(final_lepton2), pdg.get_HNL_index(nu_parent)]
        else:
            self.CC_mixing1 = 0
            self.CC_mixing2 = 0
        # Minus sign important for interference!
        self.CC_mixing2 *= -1

        if self.m_parent - self.m_daughter - self.mm - self.mp < 0:
            logger.error(f"Error! Final states are above the mass of parent particle: mass excess = {self.m_parent - self.m_daughter - self.mm - self.mp}.")
            raise ValueError("Energy not conserved.")

        # Is the mediator on shell?
        self.vector_on_shell = (
            (TheoryModel.mzprime is not None) and (self.m_parent - self.m_daughter > TheoryModel.mzprime) and (TheoryModel.mzprime > self.mm + self.mp)
        )
        self.vector_off_shell = not self.vector_on_shell

        self.scalar_on_shell = (
            (TheoryModel.mhprime is not None) and (self.m_parent - self.m_daughter > TheoryModel.mhprime) and (TheoryModel.mhprime > self.mm + self.mp)
        )
        self.scalar_off_shell = not self.scalar_on_shell

        # does it have transition magnetic moment?
        self.TMM = TheoryModel.has_TMM

    def SamplePS(
        self,
        NINT=MC.NINT,
        NEVAL=MC.NEVAL,
        NINT_warmup=MC.NINT_warmup,
        NEVAL_warmup=MC.NEVAL_warmup,
        NINT_sample=1,
        NEVAL_sample=10_000,
        savefile_norm=None,
        savefile_dec=None,
        existing_integrator=None,
    ):
        """
        Samples the phase space of the differential decay width in the rest frame of the HNL
        """
        if self.vector_on_shell and self.scalar_on_shell:
            logger.error("Vector and scalar simultaneously on shell is not implemented.")
            raise NotImplementedError("Feature not implemented.")
        elif (self.vector_off_shell and self.scalar_on_shell) or (self.vector_on_shell and self.scalar_off_shell):
            DIM = 1
        elif self.vector_off_shell and self.scalar_off_shell:
            DIM = 4
        batch_f = integrands.HNLDecay(dim=DIM, dec_case=self)
        if existing_integrator is None:
            # need to define a new integrator
            integ = vg.Integrator(DIM * [[0.0, 1.0]])  # unit hypercube

            if savefile_norm is not None:
                # Save normalization information
                with open(savefile_norm, "w") as f:
                    json.dump(batch_f.norm, f)
            # run the integrator
            MC.run_vegas(batch_f, integ, adapt_to_errors=True, NINT=NINT, NEVAL=NEVAL, NINT_warmup=NINT_warmup, NEVAL_warmup=NEVAL_warmup, savestr=savefile_dec)
            logger.debug("Main VEGAS run completed for decay sampler.")
            # Run one more time without adaptation to fix integration points to sample
            # Save the resulting integrator to a pickle file
            existing_integrator = integ
        existing_integrator(batch_f, adapt=False, nitn=NINT_sample, neval=NEVAL_sample)
        return MC.get_samples(existing_integrator, batch_f)

    def total_width(self, NINT=MC.NINT, NEVAL=MC.NEVAL, NINT_warmup=MC.NINT_warmup, NEVAL_warmup=MC.NEVAL_warmup, savefile_norm=None, savefile_dec=None):
        if self.vector_on_shell and self.scalar_on_shell:
            logger.error("Vector and scalar simultaneously on shell is not implemented.")
            raise NotImplementedError("Feature not implemented.")
        elif self.vector_on_shell and self.scalar_off_shell:
            return dr.gamma_Ni_to_Nj_V(vertex_ij=self.Dih, mi=self.m_parent, mj=self.m_daughter, mV=self.mzprime, HNLtype=self.HNLtype)
        # * dr.gamma_V_to_ell_ell(vertex=self.TheoryModel.deV, mV=self.mzprime, m_ell=self.mm)
        elif self.vector_off_shell and self.scalar_on_shell:
            return dr.gamma_Ni_to_Nj_S(vertex_ij=self.Sih, mi=self.m_parent, mj=self.m_daughter, mS=self.mhprime, HNLtype=self.HNLtype)
            # * dr.gamma_S_to_ell_ell(vertex=self.TheoryModel.deS, mS=self.mhprime, m_ell=self.mm)
        elif self.vector_off_shell and self.scalar_off_shell:

            # We need to integraate the differential cross section
            batch_f = integrands.HNLDecay(dim=4, dec_case=self)

            integ = vg.Integrator(4 * [[0.0, 1.0]])  # unit hypercube
            if savefile_norm is not None:
                # Save normalization information
                with open(savefile_norm, "w") as f:
                    json.dump(batch_f.norm, f)
            # run the integrator
            integrals = MC.run_vegas(batch_f, integ, adapt_to_errors=True, NINT=NINT, NEVAL=NEVAL, NINT_warmup=NINT_warmup, NEVAL_warmup=NEVAL_warmup)
            logger.debug("Main VEGAS run completed for decay total width calculation.")
            return integrals["diff_decay_rate_0"].mean * batch_f.norm["diff_decay_rate_0"]

    def differential_width(self, momenta):
        PN_LAB, Plepminus_LAB, Plepplus_LAB, Pnu_LAB = momenta
        # Calculate kinematics of HNL
        CosThetaPNLab = Cfv.get_cosTheta(PN_LAB)
        PhiPNLab = np.arctan2(PN_LAB.T[2], PN_LAB.T[1])
        pN_LAB = np.sqrt(PN_LAB.T[0] ** 2 - self.m_parent**2)
        if self.vector_on_shell and self.scalar_on_shell:
            logger.error("Vector and scalar simultaneously on shell is not implemented.")
            raise NotImplementedError("Feature not implemented.")
        elif self.vector_on_shell and self.scalar_off_shell:
            # Find vector boson four momentum in lab frame
            PV_LAB = Plepminus_LAB + Plepplus_LAB
            # Boost vector boson to the HNL rest frame
            PV_CM = Cfv.T(PV_LAB, -pN_LAB / PN_LAB.T[0], np.arccos(CosThetaPNLab), PhiPNLab)
            # CosTheta of vector boson in HNL rest frame
            PS = Cfv.get_cosTheta(PV_CM)
            return dr.diff_gamma_Ni_to_Nj_V(
                cost=PS, vertex_ij=self.Dih, mi=self.m_parent, mj=self.m_daughter, mV=self.mzprime, HNLtype=self.HNLtype, h=self.h_parent
            ) * dr.gamma_V_to_ell_ell(vertex=self.TheoryModel.deV, mV=self.mzprime, m_ell=self.mm)
        elif self.vector_off_shell and self.scalar_on_shell:
            # Find scalar boson four momentum in lab frame
            PS_LAB = Plepminus_LAB + Plepplus_LAB
            # Boost scalar boson to the HNL rest frame
            PS_CM = Cfv.T(PS_LAB, -pN_LAB / PN_LAB.T[0], np.arccos(CosThetaPNLab), PhiPNLab)
            # CosTheta of vector boson in HNL rest frame
            PS = Cfv.get_cosTheta(PS_CM)
            return dr.diff_gamma_Ni_to_Nj_S(
                cost=PS, vertex_ij=self.Sih, mi=self.m_parent, mj=self.m_daughter, mS=self.mhprime, HNLtype=self.HNLtype, h=self.h_parent
            ) * dr.gamma_S_to_ell_ell(vertex=self.TheoryModel.deS, mS=self.mhprime, m_ell=self.mm)
        elif self.vector_off_shell and self.scalar_off_shell:
            # Ni (k1) --> ell-(k2)  ell+(k3)  Nj(k4)

            # t = m23^2
            t = Cfv.dot4(Plepminus_LAB, Plepplus_LAB)

            # u = m24^2
            u = Cfv.dot4(Plepminus_LAB, Pnu_LAB)

            # c3 = cosine of polar angle of k3
            # Boost ell+ to HNL rest frame
            Plepplus_CM = Cfv.T(Plepplus_LAB, -pN_LAB / PN_LAB.T[0], np.arccos(CosThetaPNLab), PhiPNLab)
            c3 = Cfv.get_cosTheta(Plepplus_CM)

            # phi34 = azimuthal angle of k4 wrt k3
            # Boost Nj to HNL rest frame
            Pnu_CM = Cfv.T(Pnu_LAB, -pN_LAB / PN_LAB.T[0], np.arccos(CosThetaPNLab), PhiPNLab)
            PhiPnuCM = np.arctan2(Pnu_CM.T[2], Pnu_CM.T[1])
            PhiPlepplusCM = np.arctan2(Plepplus_CM.T[2], Plepplus_CM.T[1])
            phi34 = PhiPnuCM - PhiPlepplusCM

            m1 = self.m_parent
            m2 = self.mm
            m3 = self.mp
            m4 = self.m_daughter
            masses = np.array([m1, m2, m3, m4])

            v = np.sum(masses**2) - u - t

            return dr.diff_gamma_Ni_to_Nj_ell_ell([t, u, v, c3, phi34], self)


class FermionSinglePhotonDecay:
    def __init__(self, nu_parent, nu_daughter, TheoryModel, h_parent=-1):

        self.TheoryModel = TheoryModel
        self.HNLtype = TheoryModel.HNLtype
        self.h_parent = h_parent

        self.nu_parent = nu_parent
        self.nu_daughter = nu_daughter
        self.secondaries = [pdg.photon]

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
            self.Tih = np.sqrt(np.sum(np.abs(self.TheoryModel.t_aj[const.inds_active, pdg.get_HNL_index(nu_parent)]) ** 2))
            self.m_daughter = 0.0
        else:
            self.Tih = self.TheoryModel.t_aj[pdg.get_HNL_index(nu_daughter), pdg.get_HNL_index(nu_parent)]

    def SamplePS(
        self,
        NINT=MC.NINT,
        NEVAL=MC.NEVAL,
        NINT_warmup=MC.NINT_warmup,
        NEVAL_warmup=MC.NEVAL_warmup,
        NINT_sample=1,
        NEVAL_sample=10_000,
        savefile_norm=None,
        savefile_dec=None,
        existing_integrator=None,
    ):
        """
        Samples the phase space of the differential decay width in the rest frame of the HNL
        """
        DIM = 1
        batch_f = integrands.HNLDecay(dim=DIM, dec_case=self)
        if existing_integrator is None:
            # need to define a new integrator
            integ = vg.Integrator(DIM * [[0.0, 1.0]])  # unit hypercube

            if savefile_norm is not None:
                # Save normalization information
                with open(savefile_norm, "w") as f:
                    json.dump(batch_f.norm, f)
            # run the integrator
            MC.run_vegas(batch_f, integ, adapt_to_errors=True, NINT=NINT, NEVAL=NEVAL, NINT_warmup=NINT_warmup, NEVAL_warmup=NEVAL_warmup)
            logger.debug("Main VEGAS run completed for decay sampler.")
            # Run one more time without adaptation to fix integration points to sample
            # Save the resulting integrator to a pickle file
            integ(batch_f, adapt=False, nitn=NINT_sample, neval=NEVAL_sample, saveall=savefile_dec)
            existing_integrator = integ
        return MC.get_samples(existing_integrator, batch_f)

    def total_width(self):
        return dr.gamma_Ni_to_Nj_gamma(vertex_ij=self.Tih, mi=self.m_parent, mj=self.m_daughter, HNLtype=self.HNLtype)

    def differential_width(self, momenta):
        PN_LAB, Pgamma_LAB = momenta
        # Calculate kinematics of HNL
        CosThetaPNLab = Cfv.get_cosTheta(PN_LAB)
        PhiPNLab = np.arctan2(PN_LAB.T[2], PN_LAB.T[1])
        pN_LAB = np.sqrt(PN_LAB.T[0] ** 2 - self.m_parent**2)
        # Boost gamma to the HNL rest frame
        Pgamma_CM = Cfv.T(Pgamma_LAB, -pN_LAB / PN_LAB.T[0], np.arccos(CosThetaPNLab), PhiPNLab)
        # PN_CM = Cfv.T(PN_LAB, -pN_LAB/PN_LAB.T[0], np.arccos(CosThetaPNLab), PhiPNLab)
        # CosTheta of gamma in HNL rest frame
        PS = Cfv.get_cosTheta(Pgamma_CM)
        return dr.diff_gamma_Ni_to_Nj_gamma(cost=PS, vertex_ij=self.Tih, mi=self.m_parent, mj=self.m_daughter, HNLtype=self.HNLtype, h=self.h_parent)


def find_calculable_diagrams(bsm_model):
    """
    Args:
        bsm_model (DarkNews.model.Model): main BSM model class of DarkNews

    Returns:
        list: with all non-zero upscattering diagrams to be computed in this model.
    """

    calculable_diagrams = []
    calculable_diagrams.append("NC_SQR")
    if bsm_model.has_vector_coupling:
        calculable_diagrams.append("KinMix_SQR")
        calculable_diagrams.append("KinMix_NC_inter")
    if bsm_model.is_mass_mixed:
        calculable_diagrams.append("MassMix_SQR")
        calculable_diagrams.append("MassMix_NC_inter")
        if bsm_model.has_vector_coupling:
            calculable_diagrams.append("KinMix_MassMix_inter")
    if bsm_model.has_TMM:
        calculable_diagrams.append("TMM_SQR")
    if bsm_model.has_scalar_coupling:
        calculable_diagrams.append("Scalar_SQR")
        calculable_diagrams.append("Scalar_NC_inter")
        if bsm_model.has_vector_coupling:
            calculable_diagrams.append("Scalar_KinMix_inter")
        if bsm_model.is_mass_mixed:
            calculable_diagrams.append("Scalar_MassMix_inter")
    return calculable_diagrams
