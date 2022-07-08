import numpy as np
import vegas as vg

from DarkNews import logger, prettyprinter

from DarkNews import const
from DarkNews import pdg
from DarkNews import integrands
from DarkNews import MC
from DarkNews import model
from DarkNews import amplitudes as amps
from DarkNews import phase_space as ps


class UpscatteringProcess:
    """ 
        Describes the process of upscattering with arbitrary vertices and masses
    
    """

    def __init__(self, nu_projectile, nu_upscattered, nuclear_target, scattering_regime, TheoryModel, helicity):

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
            # self.Chad = const.gweak/4.0/const.cw*np.abs((1.0-4.0*const.s2w)*self.target.Z-self.target.N)
            # self.Vhad = const.eQED*TheoryModel.epsilon*self.target.Z
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
        self.Ethreshold = self.m_ups ** 2 / 2.0 / self.MA + self.m_ups

        # vectorize total cross section calculator using vegas integration
        self.vectorized_total_xsec = np.vectorize(self.scalar_total_xsec, excluded=["self", "diagram", "NINT", "NEVAL", "NINT_warmup", "NEVAL_warmup"])

        self.calculable_diagrams = find_calculable_diagrams(TheoryModel)

    def scalar_total_xsec(self, Enu, diagram="total", NINT=MC.NINT, NEVAL=MC.NEVAL, NINT_warmup=MC.NINT_warmup, NEVAL_warmup=MC.NEVAL_warmup):
        # below threshold
        if Enu < (self.Ethreshold):
            return 0.0
        else:
            DIM = 1
            batch_f = integrands.UpscatteringXsec(dim=DIM, Enu=Enu, ups_case=self, diagram=diagram)
            integ = vg.Integrator(DIM * [[0.0, 1.0]])  # unit hypercube

            integrals = MC.run_vegas(batch_f, integ, adapt_to_errors=True, NINT=NINT, NEVAL=NEVAL, NINT_warmup=NINT_warmup, NEVAL_warmup=NEVAL_warmup)
            logger.debug(f"Main VEGAS run completed.")

            return integrals["diff_xsec"].mean * batch_f.norm["diff_xsec"]

    def total_xsec(self, Enu, diagrams=["total"], NINT=MC.NINT, NEVAL=MC.NEVAL, NINT_warmup=MC.NINT_warmup, NEVAL_warmup=MC.NEVAL_warmup):
        """ 
            Returns the total upscattering xsec for a fixed neutrino energy in cm^2
        """
        self.Enu = Enu
        all_xsecs = 0.0
        for diagram in diagrams:
            if diagram in self.calculable_diagrams or diagram == "total":
                tot_xsec = self.vectorized_total_xsec(Enu, diagram=diagram, NINT=NINT, NEVAL=NEVAL, NINT_warmup=NINT_warmup, NEVAL_warmup=NEVAL_warmup)
            else:
                logger.warning(f"Warning: Diagram not found. Either not implemented or misspelled. Setting tot xsec it to zero: {diagram}")
                tot_xsec = 0.0 * Enu

            #############
            # integrated xsec coverted to cm^2
            all_xsecs += tot_xsec * const.attobarn_to_cm2 * self.target_multiplicity
            logger.debug(f"Total cross section for {diagram} calculated.")

        return all_xsecs

    def diff_xsec_Q2(self, Enu, Q2, diagrams=["total"]):
        """ 
            Returns the differential upscattering xsec for a fixed neutrino energy in cm^2
        """
        s = Enu * self.MA * 2 + self.MA ** 2
        physical = (Q2 > ps.upscattering_Q2min(Enu, self.m_ups, self.MA)) & (Q2 < ps.upscattering_Q2max(Enu, self.m_ups, self.MA))
        diff_xsecs = amps.upscattering_dxsec_dQ2([s, -Q2, 0.0], process=self, diagrams=diagrams)
        if type(diff_xsecs) is dict:
            return {key: diff_xsecs[key] * physical for key in diff_xsecs.keys()}
        else:
            return diff_xsecs * physical * const.attobarn_to_cm2 * self.target_multiplicity


class FermionDileptonDecay:
    def __init__(self, nu_parent, nu_daughter, final_lepton1, final_lepton2, TheoryModel, h_parent=-1):

        self.TheoryModel = TheoryModel
        self.HNLtype = TheoryModel.HNLtype
        self.h_parent = h_parent

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
        ## Minus sign important for interference!
        self.CC_mixing2 *= -1

        if self.m_parent - self.m_daughter - self.mm - self.mp < 0:
            logger.error(f"Error! Final states have an excess in mass of {self.m_parent - self.m_daughter - self.mm - self.mp} on top of parent particle.")
            raise ValueError("Energy not conserved.")

        ## Is the mediator on shell?
        self.on_shell = (self.m_parent - self.m_daughter > TheoryModel.mzprime) and (TheoryModel.mzprime > self.mm + self.mp)
        self.off_shell = not self.on_shell
        ## does it have transition magnetic moment?
        self.TMM = TheoryModel.has_TMM


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
            self.Tih = np.sqrt(np.sum(np.abs(self.TheoryModel.t_aj[const.inds_active, pdg.get_HNL_index(nu_parent)]) ** 2))
            self.m_daughter = 0.0
        else:
            self.Tih = self.TheoryModel.t_aj[pdg.get_HNL_index(nu_daughter), pdg.get_HNL_index(nu_parent)]


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
