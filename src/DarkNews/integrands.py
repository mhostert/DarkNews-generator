import numpy as np
import vegas as vg
from collections import OrderedDict

from DarkNews import Cfourvec as Cfv
from DarkNews import phase_space
from DarkNews import decay_rates as dr
from DarkNews import amplitudes as amps
from DarkNews import processes as proc

import logging

logger = logging.getLogger("logger." + __name__)
prettyprinter = logging.getLogger("prettyprinter." + __name__)


class UpscatteringXsec(vg.BatchIntegrand):
    def __init__(self, dim, Enu, ups_case, diagram="total"):
        """
        Vegas integrand for diff cross section for upscattering

        Args:
                dim (int): integration dimensions
                Enu (float): neutrino energy to be considered
                ups_case (DarkNews.processes.UpscatteringProcess): the upscattering class of DarkNews
                diagram (str, optional): _description_. Defaults to 'total'.

        Raises:
                ValueError: if cannot find diagrams to be computed
        """
        self.dim = dim
        self.Enu = Enu
        self.ups_case = ups_case
        self.diagram = diagram
        if not isinstance(self.diagram, str):
            logger.error(f"ERROR. Cannot calculate total cross section for more than one diagram at a time. Passed diagram={self.diagram}.")
            raise ValueError

        # Enforce Q2 range
        if self.ups_case.scattering_regime == "coherent":
            self.QMIN = 0  # GeV
            self.QMAX = 2  # GeV
        elif self.ups_case.scattering_regime in ["p-el", "n-el"]:
            self.QMIN = 0  # GeV
            self.QMAX = 5  # GeV
        elif self.ups_case.scattering_regime in ["DIS"]:
            self.QMIN = 2  # GeV
            self.QMAX = np.inf  # GeV
        else:
            self.QMIN = 0  # GeV
            self.QMAX = np.inf

        # Find the normalization factor
        self.norm = {}
        self.norm["diff_xsec"] = 1
        # normalize integrand with an initial throw
        _throw = self.__call__(np.random.rand(self.dim, 20000), np.ones((self.dim, 20000)))
        for key, val in _throw.items():
            self.norm[key] = np.mean(val)
            # cannot normalize zero integrand
            if self.norm[key] == 0.0:
                logger.warning("Warning: mean of integrand is zero. Vegas may break.")
                self.norm[key] = 1

    def __call__(self, x, jac):

        ups_case = self.ups_case

        ##############################################
        # Upscattering Kinematics
        Enu = self.Enu
        M = ups_case.target.mass

        Q2lmin = np.log(phase_space.upscattering_Q2min(Enu, ups_case.m_ups, M))
        Q2lmax = np.log(np.minimum(phase_space.upscattering_Q2max(Enu, ups_case.m_ups, M), self.QMAX**2))

        Q2l = (Q2lmax - Q2lmin) * x[:, 0] + Q2lmin
        Q2 = np.exp(Q2l)

        s = M**2 + 2 * Enu * M  # massless projectile
        t = -Q2
        u = 2 * M**2 + ups_case.m_ups**2 - s - t  # massless projectile

        ##############################################
        # Upscattering amplitude squared (spin summed -- not averaged)
        diff_xsec = amps.upscattering_dxsec_dQ2([s, t, u], self.ups_case, diagrams=[self.diagram])
        if type(diff_xsec) is dict:
            diff_xsec = np.sum([diff_xsec[diagram] for diagram in diff_xsec.keys()])
        # hypercube jacobian (vegas hypercube --> physical limits) transformation
        hypercube_jacobian = (Q2lmax - Q2lmin) * np.exp(Q2l)
        diff_xsec *= hypercube_jacobian

        ##############################################
        # return all differential quantities of interest
        self.int_dic = OrderedDict()
        self.int_dic["diff_xsec"] = diff_xsec
        ##############################################
        # normalization
        self.int_dic["diff_xsec"] /= self.norm["diff_xsec"]

        return self.int_dic


class HNLDecay(vg.BatchIntegrand):
    def __init__(self, dim, dec_case, diagram="total"):
        """
        Vegas integrand for diff decay width of HNL decay.

        Args:
                dim (int): integration dimensions
                dec_case (DarkNews.processes.DecayProcess): the decay class of DarkNews
                diagram (str, optional): _description_. Defaults to 'total'.

        Raises:
                ValueError: if cannot find diagrams to be computed
        """
        self.dim = dim
        self.decay_case = dec_case
        self.diagram = diagram
        if not isinstance(self.diagram, str):
            logger.error(f"ERROR. Cannot calculate total decay width for more than one diagram at a time. Passed diagram={self.diagram}.")
            raise ValueError

        self.norm = {}
        if isinstance(self.decay_case, proc.FermionDileptonDecay):
            if (self.decay_case.scalar_on_shell and self.decay_case.vector_off_shell) or (self.decay_case.scalar_off_shell and self.decay_case.vector_on_shell):
                self.norm["diff_decay_rate_0"] = 1
                self.norm["diff_decay_rate_1"] = 1
            elif self.decay_case.scalar_off_shell and self.decay_case.vector_off_shell:
                self.norm["diff_decay_rate_0"] = 1
            else:
                raise NotImplementedError("Both mediators on shell not yet implemented.")
        elif isinstance(self.decay_case, proc.FermionSinglePhotonDecay):
            self.norm["diff_decay_rate_0"] = 1
        else:
            logger.error("ERROR: Could not determine decay process in vegas integral.")
            raise ValueError

        # normalize integrand with an initial throw
        logger.debug("Throwing an initial 10000 random points to find the normalization")
        _throw = self.__call__(np.random.rand(self.dim, 10_000), np.ones((self.dim, 10_000)))
        logger.debug("Throwing successful")
        for key, val in _throw.items():
            self.norm[key] = np.mean(val)
            # cannot normalize zero integrand
            if self.norm[key] == 0:
                self.norm[key] = 1

    def __call__(self, x, jac):

        self.int_dic = OrderedDict()

        m_parent = self.decay_case.m_parent
        m_daughter = self.decay_case.m_daughter

        i_var = 0

        ##############################################
        if isinstance(self.decay_case, proc.FermionDileptonDecay):

            if self.decay_case.vector_on_shell and self.decay_case.scalar_on_shell:
                logger.error("Vector and scalar simultaneously on shell is not implemented.")
                raise NotImplementedError("Feature not implemented.")

            elif self.decay_case.vector_on_shell and self.decay_case.scalar_off_shell:
                # decay nu_parent -> nu_daughter mediator
                # angle between nu_daughter and z axis
                cost = -1.0 + (2.0) * x[:, i_var]
                i_var += 1

                self.int_dic["diff_decay_rate_0"] = dr.diff_gamma_Ni_to_Nj_V(
                    cost=cost,
                    vertex_ij=self.decay_case.Dih,
                    mi=m_parent,
                    mj=m_daughter,
                    mV=self.decay_case.mzprime,
                    HNLtype=self.decay_case.HNLtype,
                    h=self.decay_case.h_parent,
                )
                self.int_dic["diff_decay_rate_0"] *= 2  # hypercube jacobian

                # mediator decay M --> ell+ ell-
                self.int_dic["diff_decay_rate_1"] = dr.gamma_V_to_ell_ell(
                    vertex=np.sqrt(self.decay_case.TheoryModel.deV**2 + self.decay_case.TheoryModel.deA**2),
                    mV=self.decay_case.mzprime,
                    m_ell=self.decay_case.mm,
                ) * np.full_like(self.int_dic["diff_decay_rate_0"], 1.0)

            elif self.decay_case.vector_off_shell and self.decay_case.scalar_on_shell:
                # decay nu_parent -> nu_daughter mediator
                # angle between nu_daughter and z axis
                cost = -1.0 + (2.0) * x[:, i_var]
                i_var += 1

                self.int_dic["diff_decay_rate_0"] = dr.diff_gamma_Ni_to_Nj_S(
                    cost=cost,
                    vertex_ij=self.decay_case.Sih,
                    mi=m_parent,
                    mj=m_daughter,
                    mS=self.decay_case.mhprime,
                    HNLtype=self.decay_case.HNLtype,
                    h=self.decay_case.h_parent,
                )
                self.int_dic["diff_decay_rate_0"] *= 2  # hypercube jacobian

                ##############################################
                # mediator decay M --> ell+ ell-
                self.int_dic["diff_decay_rate_1"] = dr.gamma_S_to_ell_ell(
                    vertex=self.decay_case.TheoryModel.deS, mS=self.decay_case.mhprime, m_ell=self.decay_case.mm
                ) * np.full_like(self.int_dic["diff_decay_rate_0"], 1.0)

            elif self.decay_case.vector_off_shell and self.decay_case.scalar_off_shell:
                ##############################################
                # decay nu_parent -> nu_daughter ell+ ell-

                m1 = self.decay_case.m_parent
                m2 = self.decay_case.mm
                m3 = self.decay_case.mp
                m4 = self.decay_case.m_daughter
                masses = np.array([m1, m2, m3, m4])

                # limits
                tmax = phase_space.three_body_tmax(*masses)
                tmin = phase_space.three_body_tmin(*masses)
                t = (tmax - tmin) * x[:, i_var] + tmin
                i_var += 1

                umax = phase_space.three_body_umax(*masses, t)
                umin = phase_space.three_body_umin(*masses, t)
                u = (umax - umin) * x[:, i_var] + umin
                i_var += 1

                v = np.sum(masses**2) - u - t

                c3 = (2.0) * x[:, i_var] - 1.0
                i_var += 1
                phi34 = (2.0 * np.pi) * x[:, i_var]
                i_var += 1

                dgamma = dr.diff_gamma_Ni_to_Nj_ell_ell([t, u, v, c3, phi34], self.decay_case)

                # hypercube jacobian (vegas hypercube --> physical limits) transformation
                dgamma *= tmax - tmin
                dgamma *= umax - umin
                dgamma *= 2  # c3
                dgamma *= 2 * np.pi  # phi34
                self.int_dic["diff_decay_rate_0"] = dgamma
            else:
                logger.error("Could not find decay integrand.")
                raise ValueError("Integrand not found for this model.")

        elif isinstance(self.decay_case, proc.FermionSinglePhotonDecay):
            ##############################################
            # decay nu_parent -> nu_daughter gamma

            # angle between nu_daughter and z axis
            cost = -1.0 + (2.0) * x[:, i_var]
            i_var += 1

            self.int_dic["diff_decay_rate_0"] = dr.diff_gamma_Ni_to_Nj_gamma(
                cost=cost,
                vertex_ij=self.decay_case.Tih,
                mi=m_parent,
                mj=m_daughter,
                HNLtype=self.decay_case.HNLtype,
                h=self.decay_case.h_parent,
            )

            # hypercube jacobian (vegas hypercube --> physical limits) transformation
            self.int_dic["diff_decay_rate_0"] *= 2

        else:
            logger.error("Could not find decay integrand.")
            raise ValueError("Integrand not found for this model.")

        ##############################################
        # storing normalization factor that guarantees that integrands are O(1) numbers
        # normalize integrands to be O(1)
        for k in self.norm.keys():
            self.int_dic[k] /= self.norm[k]
        logger.debug(f"Normalization factors in integrand: {self.norm}.")

        # return all differential quantities of interest
        return self.int_dic


class UpscatteringHNLDecay(vg.BatchIntegrand):
    def __init__(self, dim, Emin, Emax, MC_case):
        """
        Vegas integrand for the process of upscattering with subsequent decays.
        Scattering diff xsec and diff decay rates are integrated simultaneously.

        Args:
                dim (int): _description_
                Emin (float): min neutrino energy to integrate flux
                Emax (float): max neutrino energy to integrate flux
                MC_case (DarkNews.MC.MC_events): the main Monte-Carlo class of DarkNews

        Raises:
                ValueError: if cannot find what decay process to consider
        """
        self.dim = dim
        self.Emax = Emax
        self.Emin = Emin
        self.MC_case = MC_case

        self.norm = {}
        self.norm["diff_event_rate"] = 1
        self.norm["diff_flux_avg_xsec"] = 1
        if self.MC_case.decays_to_dilepton:
            if (self.MC_case.decay_case.scalar_on_shell and self.MC_case.decay_case.vector_off_shell) or (
                self.MC_case.decay_case.scalar_off_shell and self.MC_case.decay_case.vector_on_shell
            ):
                self.norm["diff_decay_rate_0"] = 1
                self.norm["diff_decay_rate_1"] = 1
            elif self.MC_case.decay_case.scalar_off_shell and self.MC_case.decay_case.vector_off_shell:
                self.norm["diff_decay_rate_0"] = 1
            else:
                raise NotImplementedError("Both mediators on shell not yet implemented.")
        elif self.MC_case.decays_to_singlephoton:
            self.norm["diff_decay_rate_0"] = 1
        else:
            logger.error("ERROR: Could not determine decay process in vegas integral.")
            raise ValueError

        # normalize integrand with an initial throw
        logger.debug("Throwing an initial 10000 random points to find the normalization")
        _throw = self.__call__(np.random.rand(self.dim, 10_000), np.ones((self.dim, 10_000)))
        logger.debug("Throwing successful")
        for key, val in _throw.items():
            self.norm[key] = np.mean(val)
            # cannot normalize zero integrand
            if self.norm[key] == 0:
                self.norm[key] = 1

    def __call__(self, x, jac):

        self.int_dic = OrderedDict()

        ups_case = self.MC_case.ups_case
        decay_case = self.MC_case.decay_case

        M = ups_case.target.mass
        m_parent = ups_case.m_ups
        m_daughter = decay_case.m_daughter

        i_var = 0
        # neutrino energy
        Enu = (self.Emax - self.Emin) * x[:, i_var] + self.Emin
        i_var += 1

        ##############################################
        # Upscattering differential cross section (spin averaged)

        Q2lmin = np.log(phase_space.upscattering_Q2min(Enu, m_parent, M))
        Q2lmax = np.log(phase_space.upscattering_Q2max(Enu, m_parent, M))

        Q2l = (Q2lmax - Q2lmin) * x[:, i_var] + Q2lmin
        i_var += 1

        Q2 = np.exp(Q2l)
        s_scatt = M**2 + 2 * Enu * M  # massless projectile
        t_scatt = -Q2
        u_scatt = 2 * M**2 + m_parent**2 - s_scatt + Q2  # massless projectile

        diff_xsec = amps.upscattering_dxsec_dQ2([s_scatt, t_scatt, u_scatt], ups_case)

        # hypercube jacobian (vegas hypercube --> physical limits) transformation
        hypercube_jacobian = (Q2lmax - Q2lmin) * np.exp(Q2l) * (self.Emax - self.Emin)
        diff_xsec *= hypercube_jacobian

        self.int_dic["diff_event_rate"] = diff_xsec * self.MC_case.flux(Enu)
        self.int_dic["diff_flux_avg_xsec"] = diff_xsec * self.MC_case.flux(Enu)

        ##############################################
        if self.MC_case.decays_to_dilepton:

            if decay_case.vector_on_shell and decay_case.scalar_on_shell:
                logger.error("Vector and scalar simultaneously on shell is not implemented.")
                raise NotImplementedError("Feature not implemented.")

            elif decay_case.vector_on_shell and decay_case.scalar_off_shell:
                # decay nu_parent -> nu_daughter mediator
                # angle between nu_daughter and z axis
                cost = -1.0 + (2.0) * x[:, i_var]
                i_var += 1

                self.int_dic["diff_decay_rate_0"] = dr.diff_gamma_Ni_to_Nj_V(
                    cost=cost,
                    vertex_ij=decay_case.Dih,
                    mi=m_parent,
                    mj=m_daughter,
                    mV=decay_case.mzprime,
                    HNLtype=decay_case.HNLtype,
                    h=decay_case.h_parent,
                )
                self.int_dic["diff_decay_rate_0"] *= 2  # hypercube jacobian

                # mediator decay M --> ell+ ell-
                self.int_dic["diff_decay_rate_1"] = dr.gamma_V_to_ell_ell(
                    vertex=np.sqrt(decay_case.TheoryModel.deV**2 + decay_case.TheoryModel.deA**2),
                    mV=decay_case.mzprime,
                    m_ell=decay_case.mm,
                ) * np.full_like(self.int_dic["diff_decay_rate_0"], 1.0)

            elif decay_case.vector_off_shell and decay_case.scalar_on_shell:
                # decay nu_parent -> nu_daughter mediator
                # angle between nu_daughter and z axis
                cost = -1.0 + (2.0) * x[:, i_var]
                i_var += 1

                self.int_dic["diff_decay_rate_0"] = dr.diff_gamma_Ni_to_Nj_S(
                    cost=cost,
                    vertex_ij=decay_case.Sih,
                    mi=m_parent,
                    mj=m_daughter,
                    mS=decay_case.mhprime,
                    HNLtype=decay_case.HNLtype,
                    h=decay_case.h_parent,
                )
                self.int_dic["diff_decay_rate_0"] *= 2  # hypercube jacobian

                ##############################################
                # mediator decay M --> ell+ ell-
                self.int_dic["diff_decay_rate_1"] = dr.gamma_S_to_ell_ell(
                    vertex=decay_case.TheoryModel.deS, mS=decay_case.mhprime, m_ell=decay_case.mm
                ) * np.full_like(self.int_dic["diff_decay_rate_0"], 1.0)

            elif decay_case.vector_off_shell and decay_case.scalar_off_shell:
                ##############################################
                # decay nu_parent -> nu_daughter ell+ ell-

                m1 = decay_case.m_parent
                m2 = decay_case.mm
                m3 = decay_case.mp
                m4 = decay_case.m_daughter
                masses = np.array([m1, m2, m3, m4])

                # limits
                tmax = phase_space.three_body_tmax(*masses)
                tmin = phase_space.three_body_tmin(*masses)
                t = (tmax - tmin) * x[:, i_var] + tmin
                i_var += 1

                umax = phase_space.three_body_umax(*masses, t)
                umin = phase_space.three_body_umin(*masses, t)
                u = (umax - umin) * x[:, i_var] + umin
                i_var += 1

                v = np.sum(masses**2) - u - t

                c3 = (2.0) * x[:, i_var] - 1.0
                i_var += 1
                phi34 = (2.0 * np.pi) * x[:, i_var]
                i_var += 1

                dgamma = dr.diff_gamma_Ni_to_Nj_ell_ell([t, u, v, c3, phi34], decay_case)

                # hypercube jacobian (vegas hypercube --> physical limits) transformation
                dgamma *= tmax - tmin
                dgamma *= umax - umin
                dgamma *= 2  # c3
                dgamma *= 2 * np.pi  # phi34
                self.int_dic["diff_decay_rate_0"] = dgamma
            else:
                logger.error("Could not find decay integrand.")
                raise ValueError("Integrand not found for this model.")

        elif self.MC_case.decays_to_singlephoton:
            ##############################################
            # decay nu_parent -> nu_daughter gamma

            # angle between nu_daughter and z axis
            cost = -1.0 + (2.0) * x[:, i_var]
            i_var += 1

            self.int_dic["diff_decay_rate_0"] = dr.diff_gamma_Ni_to_Nj_gamma(
                cost=cost,
                vertex_ij=decay_case.Tih,
                mi=m_parent,
                mj=m_daughter,
                HNLtype=decay_case.HNLtype,
                h=decay_case.h_parent,
            )

            # hypercube jacobian (vegas hypercube --> physical limits) transformation
            self.int_dic["diff_decay_rate_0"] *= 2

        else:
            logger.error("Could not find decay integrand.")
            raise ValueError("Integrand not found for this model.")

        ##############################################
        # storing normalization factor that guarantees that integrands are O(1) numbers
        # loop over decay processes
        for decay_step in (k for k in self.int_dic.keys() if "decay_rate" in k):
            # multiply differential event rate by dGamma_i/dPS
            self.int_dic["diff_event_rate"] *= self.int_dic[decay_step]

        # normalize integrands to be O(1)
        for k in self.norm.keys():
            self.int_dic[k] /= self.norm[k]
        logger.debug(f"Normalization factors in integrand: {self.norm}.")

        ##############################################
        # flat direction jacobian (undoing the vegas adaptation in flat directions of factors of the full integrands)
        # loop over decay processes
        for decay_step in (k for k in self.int_dic.keys() if "decay_rate" in k):
            self.int_dic[decay_step] /= jac[:, 0] * jac[:, 1]
        decay_directions = range(2, self.dim)
        for d in decay_directions:
            self.int_dic["diff_flux_avg_xsec"] /= jac[:, d]

        # return all differential quantities of interest
        return self.int_dic


def get_momenta_from_vegas_samples(vsamples, MC_case):
    """
    Construct the four momenta of all particles in the upscattering+decay process from the
    vegas weights.

    Args:
            vsamples (np.ndarray): integration samples obtained from vegas
                            as hypercube coordinates. Always in the interval [0,1].

            MC_case (DarkNews.MC.MC_events): the main Monte-Carlo class of DarkNews

    Returns:
            dict: each key corresponds to a set of four momenta for a given particle involved,
                    so the values are 2D np.ndarrays with each row a different event and each column a different
                    four momentum component. Contains also the weights.
    """

    four_momenta = {}

    ########################
    # upscattering
    # Ni(k1) target(k2) -->  Nj(k3) target(k4)

    mh = MC_case.ups_case.m_ups
    MA = MC_case.ups_case.MA

    # energy of projectile
    Eprojectile = (MC_case.EMAX - MC_case.EMIN) * vsamples[0] + MC_case.EMIN
    scatter_samples = {"Eprojectile": Eprojectile, "unit_Q2": vsamples[1]}
    masses_scatter = {
        "m1": 0.0,  # nu_projectile
        "m2": MA,  # target
        "m3": mh,  # nu_upscattered
        "m4": MA,  # final target
    }

    P1LAB, P2LAB, P3LAB, P4LAB = phase_space.two_to_two_scatter(scatter_samples, **masses_scatter, rng=MC_case.rng)

    # N boost parameters
    boost_scattered_N = {
        "EP_LAB": P3LAB.T[0],
        "costP_LAB": Cfv.get_cosTheta(P3LAB),
        "phiP_LAB": np.arctan2(P3LAB.T[2], P3LAB.T[1]),
    }

    four_momenta["P_projectile"] = P1LAB
    four_momenta["P_target"] = P2LAB
    four_momenta["P_recoil"] = P4LAB

    #######################
    # decay processes

    if MC_case.decays_to_dilepton:

        mf = MC_case.decay_case.m_daughter
        mm = MC_case.decay_case.mm
        mp = MC_case.decay_case.mm

        if MC_case.decay_case.vector_on_shell or MC_case.decay_case.scalar_on_shell:

            if MC_case.decay_case.vector_on_shell and MC_case.decay_case.scalar_off_shell:
                m_mediator = MC_case.decay_case.mzprime
            elif MC_case.decay_case.vector_off_shell and MC_case.decay_case.scalar_on_shell:
                m_mediator = MC_case.decay_case.mhprime
            else:
                raise NotImplementedError("Both mediators on-shell is not yet implemented.")

            ########################
            # HNL decay
            N_decay_samples = {"unit_cost": np.array(vsamples[2])}

            # Ni (k1) --> Nj (k2)  Z' (k3)
            masses_decay = {
                "m1": mh,  # Ni
                "m2": mf,  # Nj
                "m3": m_mediator,  # Z'
            }
            # Phnl, Phnl_daughter, Pz'
            P1LAB_decay, P2LAB_decay, P3LAB_decay = phase_space.two_body_decay(N_decay_samples, boost=boost_scattered_N, **masses_decay, rng=MC_case.rng)

            # Z' boost parameters
            boost_Z = {
                "EP_LAB": P3LAB_decay.T[0],
                "costP_LAB": Cfv.get_cosTheta(P3LAB_decay),
                "phiP_LAB": np.arctan2(P3LAB_decay.T[2], P3LAB_decay.T[1]),
            }

            ########################
            # Z' decay

            Z_decay_samples = {}  # all uniform
            # Z'(k1) --> ell- (k2)  ell+ (k3)
            masses_decay = {
                "m1": m_mediator,  # Ni
                "m2": mp,  # \ell+
                "m3": mm,  # \ell-
            }
            # PZ', pe-, pe+
            P1LAB_decayZ, P2LAB_decayZ, P3LAB_decayZ = phase_space.two_body_decay(Z_decay_samples, boost=boost_Z, **masses_decay, rng=MC_case.rng)

            four_momenta["P_decay_N_parent"] = P1LAB_decay
            four_momenta["P_decay_N_daughter"] = P2LAB_decay
            four_momenta["P_decay_ell_minus"] = P2LAB_decayZ
            four_momenta["P_decay_ell_plus"] = P3LAB_decayZ

        elif MC_case.decay_case.vector_off_shell and MC_case.decay_case.scalar_off_shell:

            ########################
            # HNL decay
            N_decay_samples = {
                "unit_t": vsamples[2],
                "unit_u": vsamples[3],
                "unit_c3": vsamples[4],
                "unit_phi34": vsamples[5],
            }

            # Ni (k1) --> ell-(k2)  ell+(k3)  Nj(k4)
            masses_decay = {
                "m1": mh,  # Ni
                "m2": mm,  # ell-
                "m3": mp,  # ell+
                "m4": mf,
            }  # Nj
            # Phnl, pe-, pe+, pnu
            (
                P1LAB_decay,
                P2LAB_decay,
                P3LAB_decay,
                P4LAB_decay,
            ) = phase_space.three_body_decay(N_decay_samples, boost=boost_scattered_N, **masses_decay, rng=MC_case.rng)

            four_momenta["P_decay_N_parent"] = P1LAB_decay
            four_momenta["P_decay_ell_minus"] = P2LAB_decay
            four_momenta["P_decay_ell_plus"] = P3LAB_decay
            four_momenta["P_decay_N_daughter"] = P4LAB_decay

    elif MC_case.decays_to_singlephoton:

        mf = MC_case.decay_case.m_daughter

        ########################
        # HNL decay
        N_decay_samples = {"unit_cost": np.array(vsamples[2])}

        # Ni (k1) --> Nj (k2)  gamma (k3)
        masses_decay = {
            "m1": mh,  # Ni
            "m2": mf,  # Nj
            "m3": 0.0,  # gamma
        }
        # Phnl, Phnl', Pgamma
        P1LAB_decay, P2LAB_decay, P3LAB_decay = phase_space.two_body_decay(N_decay_samples, boost=boost_scattered_N, **masses_decay, rng=MC_case.rng)

        four_momenta["P_decay_N_parent"] = P1LAB_decay
        four_momenta["P_decay_N_daughter"] = P2LAB_decay
        four_momenta["P_decay_photon"] = P3LAB_decay

    return four_momenta


def get_decay_momenta_from_vegas_samples(vsamples, decay_case, PN_LAB):
    """
    Construct the four momenta of all final state particles in the decay process from the
    vegas weights.

    Args:
            vsamples (np.ndarray): integration samples obtained from vegas
                            as hypercube coordinates. Always in the interval [0,1].

            decay_case (DarkNews.processes.dec_case): the decay class of DarkNews

            PN_LAB (np.ndarray): four-momentum of the upscattered N in the lab frame: [E, pX, pY, pZ]

    Returns:
            dict: each key corresponds to a set of four momenta for a given particle involved,
                    so the values are 2D np.ndarrays with each row a different event and each column a different
                    four momentum component. Contains also the weights.
    """

    four_momenta = {}

    # N boost parameters
    boost_scattered_N = {
        "EP_LAB": PN_LAB.T[0],
        "costP_LAB": Cfv.get_cosTheta(PN_LAB),
        "phiP_LAB": np.arctan2(PN_LAB.T[2], PN_LAB.T[1]),
    }

    #######################
    # DECAY PROCESSES

    if isinstance(decay_case, proc.FermionDileptonDecay):

        mh = decay_case.m_parent
        mf = decay_case.m_daughter
        mm = decay_case.mm
        mp = decay_case.mm

        if decay_case.vector_on_shell or decay_case.scalar_on_shell:

            if decay_case.vector_on_shell and decay_case.scalar_off_shell:
                m_mediator = decay_case.mzprime
            elif decay_case.vector_off_shell and decay_case.scalar_on_shell:
                m_mediator = decay_case.mhprime
            else:
                raise NotImplementedError("Both mediators on-shell is not yet implemented.")

            ########################
            # HNL decay
            N_decay_samples = {"unit_cost": np.array(vsamples[0])}
            # Ni (k1) --> Nj (k2)  Z' (k3)
            masses_decay = {
                "m1": mh,  # Ni
                "m2": mf,  # Nj
                "m3": m_mediator,  # Z'
            }
            # Phnl, Phnl_daughter, Pz'
            P1LAB_decay, P2LAB_decay, P3LAB_decay = phase_space.two_body_decay(N_decay_samples, boost=boost_scattered_N, **masses_decay)

            # Z' boost parameters
            boost_Z = {
                "EP_LAB": P3LAB_decay.T[0],
                "costP_LAB": Cfv.get_cosTheta(P3LAB_decay),
                "phiP_LAB": np.arctan2(P3LAB_decay.T[2], P3LAB_decay.T[1]),
            }

            ########################
            # Z' decay
            Z_decay_samples = {}  # all uniform
            # Z'(k1) --> ell- (k2)  ell+ (k3)
            masses_decay = {
                "m1": m_mediator,  # Ni
                "m2": mp,  # \ell+
                "m3": mm,  # \ell-
            }
            # PZ', pe-, pe+
            P1LAB_decayZ, P2LAB_decayZ, P3LAB_decayZ = phase_space.two_body_decay(Z_decay_samples, boost=boost_Z, **masses_decay)

            four_momenta["P_decay_N_parent"] = P1LAB_decay
            four_momenta["P_decay_N_daughter"] = P2LAB_decay
            four_momenta["P_decay_ell_minus"] = P2LAB_decayZ
            four_momenta["P_decay_ell_plus"] = P3LAB_decayZ

        elif decay_case.vector_off_shell and decay_case.scalar_off_shell:

            ########################
            # HNL decay
            N_decay_samples = {
                "unit_t": vsamples[0],
                "unit_u": vsamples[1],
                "unit_c3": vsamples[2],
                "unit_phi34": vsamples[3],
            }

            # Ni (k1) --> ell-(k2)  ell+(k3)  Nj(k4)
            masses_decay = {
                "m1": mh,  # Ni
                "m2": mm,  # ell-
                "m3": mp,  # ell+
                "m4": mf,
            }  # Nj
            # Phnl, pe-, pe+, pnu
            (
                P1LAB_decay,
                P2LAB_decay,
                P3LAB_decay,
                P4LAB_decay,
            ) = phase_space.three_body_decay(N_decay_samples, boost=boost_scattered_N, **masses_decay)

            four_momenta["P_decay_N_parent"] = P1LAB_decay
            four_momenta["P_decay_ell_minus"] = P2LAB_decay
            four_momenta["P_decay_ell_plus"] = P3LAB_decay
            four_momenta["P_decay_N_daughter"] = P4LAB_decay

    elif isinstance(decay_case, proc.FermionSinglePhotonDecay):

        mh = decay_case.m_parent
        mf = decay_case.m_daughter

        ########################
        # HNL decay
        N_decay_samples = {"unit_cost": np.array(vsamples[0])}
        # Ni (k1) --> Nj (k2)  gamma (k3)
        masses_decay = {
            "m1": mh,  # Ni
            "m2": mf,  # Nj
            "m3": 0.0,  # gamma
        }
        # Phnl, Phnl', Pgamma
        P1LAB_decay, P2LAB_decay, P3LAB_decay = phase_space.two_body_decay(N_decay_samples, boost=boost_scattered_N, **masses_decay)

        four_momenta["P_decay_N_parent"] = P1LAB_decay
        four_momenta["P_decay_N_daughter"] = P2LAB_decay
        four_momenta["P_decay_photon"] = P3LAB_decay

    return four_momenta
