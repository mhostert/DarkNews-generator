import numpy as np
import pandas as pd
import vegas as vg

from DarkNews import logger, prettyprinter

from collections import defaultdict
from functools import partial

from DarkNews import processes
from DarkNews import integrands
from DarkNews import const
from DarkNews import pdg
from DarkNews import geom

NINT = 10
NEVAL = 1000
NINT_warmup = 10
NEVAL_warmup = 1000


class MC_events:
    """MC events generated with importance sampling (vegas)

    Correct weights are computed from cross-section, decay width,
    and experimental considerations

    Args:
        experiment:             instance of Detector class
        target:                 scattering target class
        scattering_regime:      regime for scattering process, choices: ['coherent', 'p-el', 'n-el', 'DIS'] (future: RES)
        nu_projectile:          incoming neutrino flavor
        nu_upscattered:         intermediate dark NU in upscattering process
        nu_outgoing:            outgoing neutrino flavor
        decay_product:          visible decay products in the detector
        helicity:               helicity of the up-scattered neutrino
    """

    def __init__(self, experiment, bsm_model, **kwargs):

        # default parameters
        scope = {
            "nu_projectile": pdg.numu,
            "nu_upscattered": pdg.neutrino4,
            "nu_outgoing": pdg.nulight,
            "scattering_regime": "coherent",
            "decay_product": ["e+e-"],
            "helicity": "conserving",
            "enforce_prompt": False,
        }

        scope.update(kwargs)
        self.scope = scope
        self.experiment = experiment

        # set target properties for this scattering regime
        if "nuclear_target" in scope:
            self.nuclear_target = scope["nuclear_target"]
        else:
            self.nuclear_target = self.experiment.NUCLEAR_TARGETS[0]
            logger.warning("No target passed to MC_events, using first entry in experiment class instead.")

        # identifiers for particles in the process
        self.nu_projectile = scope["nu_projectile"]
        self.nu_upscattered = scope["nu_upscattered"]
        self.nu_outgoing = scope["nu_outgoing"]
        self.helicity = scope["helicity"]

        if scope["decay_product"] == "e+e-":
            self.decay_product = pdg.electron
            # process being considered
            DECAY_PRODUCTS = f"{self.decay_product.invert().name} {self.decay_product.name}"
            self.decays_to_dilepton = True
            self.decays_to_singlephoton = False

        elif scope["decay_product"] == "mu+mu-":
            self.decay_product = pdg.muon
            # process being considered
            DECAY_PRODUCTS = f"{self.decay_product.invert().name} {self.decay_product.name}"
            self.decays_to_dilepton = True
            self.decays_to_singlephoton = False

        elif scope["decay_product"] == "photon":
            self.decay_product = pdg.photon
            DECAY_PRODUCTS = f"{self.decay_product.name}"
            self.decays_to_dilepton = False
            self.decays_to_singlephoton = True

        else:
            logger.error(f"Error! Could not find decay product: {scope['decay_product']}")
            raise ValueError

        self.bsm_model = bsm_model
        # scope for upscattering process
        self.ups_case = processes.UpscatteringProcess(
            nu_projectile=self.nu_projectile,
            nu_upscattered=self.nu_upscattered,
            nuclear_target=self.nuclear_target,
            scattering_regime=self.scope["scattering_regime"],
            helicity=self.helicity,
            TheoryModel=bsm_model,
        )

        if self.decays_to_dilepton:
            # scope for decay process
            self.decay_case = processes.FermionDileptonDecay(
                nu_parent=self.nu_upscattered,
                nu_daughter=self.nu_outgoing,
                final_lepton1=self.decay_product,
                final_lepton2=self.decay_product,
                h_parent=self.ups_case.h_upscattered,
                TheoryModel=bsm_model,
            )
        elif self.decays_to_singlephoton:
            # scope for decay process
            self.decay_case = processes.FermionSinglePhotonDecay(
                nu_parent=self.nu_upscattered, nu_daughter=self.nu_outgoing, h_parent=self.ups_case.h_upscattered, TheoryModel=bsm_model,
            )
        else:
            logger.error("Error! Could not determine what type of decay class to use.")
            raise ValueError

        if self.ups_case.Ethreshold > self.experiment.ERANGE[-1]:
            logger.error(f"Particle {self.nu_upscattered.name} is too heavy to be produced in the energy range of experiment {self.experiment.NAME}")
            raise ValueError

        if self.ups_case.m_ups != self.decay_case.m_parent:
            logger.error(
                f"Error! Mass of HNL produced in neutrino scattering m_ups = {self.ups_case.m_upscattered} GeV, \
                    different from that of parent HNL, m_parent = {self.decay_case.m_parent} GeV."
            )
            raise ValueError

        # process being considered
        self.underl_process_name = \
            f"{self.nu_projectile.name} {self.ups_case.target.name} --> \
{self.nu_upscattered.name}  {self.ups_case.target.name} --> \
{self.nu_outgoing.name} {DECAY_PRODUCTS} {self.ups_case.target.name}"

    def get_total_xsec(self, Enu):
        """
        Returns the total upscattering xsec for a fixed neutrino energy

        """

        DIM = 1
        self.Enu = Enu

        # below threshold
        if Enu < (self.ups_case.Ethreshold):
            return 0.0

        batch_f = integrands.UpscatteringXsec(dim=DIM, Enu=self.Enu, MC_case=self)
        integ = vg.Integrator(DIM * [[0.0, 1.0]])  # unit hypercube

        integrals = run_vegas(batch_f, integ, NINT=NINT, NEVAL=NEVAL, NINT_warmup=NINT_warmup, NEVAL_warmup=NEVAL_warmup,)
        logger.debug("Main VEGAS run completed.")

        #############
        # integrated xsec coverted to cm^2
        tot_xsec = integrals["diff_xsec"].mean * const.attobarn_to_cm2 * batch_f.norm["diff_xsec"]

        # How many constituent targets inside scattering regime?
        if self.scope["scattering_regime"] == "coherent":
            target_multiplicity = 1
        elif self.scope["scattering_regime"] == "p-el":
            target_multiplicity = self.nuclear_target.Z
        elif self.scope["scattering_regime"] == "n-el":
            target_multiplicity = self.nuclear_target.N
        elif self.scope["scattering_regime"] == "DIS":
            target_multiplicity = 1
        else:
            logger.error(f"Scattering regime {self.scope['scattering_regime']} not supported.")

        logger.debug("Total cross section calculated.")
        return tot_xsec * target_multiplicity

    def get_MC_events(self):
        """
        Returns MC events from importance sampling

        Handles the off and on shell cases separately

        After sampling the integrands, we normalize the integral by
        the integral of the decay rates, which is also estimated by VEGAS
        in the multi-component integrand.

        The first integrand entry is the one VEGAS uses to optimize the importance sampling.
        """

        logger.info(f"{self.underl_process_name}")
        logger.info(f"Helicity {self.helicity} upscattering.")

        # ###############################
        # Some experimental definitions
        self.flux = self.experiment.neutrino_flux(self.nu_projectile)
        self.EMIN = max(self.experiment.ERANGE[0], 1.05 * self.ups_case.Ethreshold)
        self.EMAX = self.experiment.ERANGE[1]

        if self.decays_to_dilepton:

            if self.decay_case.on_shell:
                DIM = 3
                logger.info(f"{self.nu_upscattered.name} decays via on-shell Z'.")
            elif self.decay_case.off_shell:
                DIM = 6
                logger.info(f"{self.nu_upscattered.name} three-body decays.")
        elif self.decays_to_singlephoton:
            DIM = 3
            logger.info(f"{self.nu_upscattered.name} decays via TMM.")
        else:
            logger.error("ERROR! Could not determine decay process.")
            raise ValueError("Could not determine decay process.")
        integrand_type = integrands.UpscatteringHNLDecay

        #########################################################################
        # BATCH SAMPLE INTEGRAND OF INTEREST
        logger.debug(f"Running VEGAS for DIM={DIM}")
        batch_f = integrand_type(dim=DIM, Emin=self.EMIN, Emax=self.EMAX, MC_case=self)
        integ = vg.Integrator(DIM * [[0.0, 1.0]])  # unit hypercube
        result = run_vegas(batch_f, integ, NINT=NINT, NEVAL=NEVAL, NINT_warmup=NINT_warmup, NEVAL_warmup=NEVAL_warmup,)
        logger.debug("Main VEGAS run completed.")

        logger.debug(f"Vegas results for the integrals: {result.summary()}")
        ##########################################################################

        #########################################################################
        # GET THE INTEGRATION SAMPLES and translate to physical variables in MC events
        samples, weights = get_samples(integ, batch_f)
        tot_nevents = len(weights["diff_event_rate"])

        logger.debug(f"Normalization factors in MC: {batch_f.norm}.")
        logger.debug(f"Vegas results for diff_event_rate: {weights['diff_event_rate'].sum()}")
        logger.debug(f"Vegas results for diff_flux_avg_xsec: {weights['diff_flux_avg_xsec'].sum()}")

        four_momenta = integrands.get_momenta_from_vegas_samples(samples, MC_case=self)
        ##########################################################################

        ##########################################################################
        # SAVE ALL EVENTS AS A PANDAS DATAFRAME
        particles = list(four_momenta.keys())
        columns = particles + ["pos_scatt"]
        indexing = [columns, ["0", "1", "2", "3"]]
        columns_index = pd.MultiIndex.from_product(indexing)

        # create auxiliary data list
        aux_data = []
        # create pandas dataframe
        for p in particles:
            for component in range(4):
                aux_data.append(four_momenta[p][:, component])
        # scattering position
        aux_data.append(np.zeros((tot_nevents,)))
        aux_data.append(np.zeros((tot_nevents,)))
        aux_data.append(np.zeros((tot_nevents,)))
        aux_data.append(np.zeros((tot_nevents,)))

        df_gen = pd.DataFrame(np.stack(aux_data, axis=-1), columns=columns_index)

        # differential weights
        for column in df_gen:
            if "w_" in column:
                df_gen[column, ""] = df_gen[column]

        # Normalize weights and total integral with decay rates and set units to nus*cm^2/POT
        decay_rates = 1
        for decay_step in (k for k in batch_f.int_dic.keys() if "decay_rate" in k):
            logger.debug(f"Vegas results for {decay_step}: {weights[decay_step].sum()}")

            # saving decay weights and integrals
            df_gen[f"w_{decay_step}".replace("diff_", "")] = weights[decay_step] * batch_f.norm[decay_step]

            # combining all decay rates into one factor
            decay_rates *= df_gen[f"w_{decay_step}".replace("diff_", "")].sum()

        # How many constituent targets inside scattering regime?
        if self.scope["scattering_regime"] == "coherent":
            target_multiplicity = 1
        elif self.scope["scattering_regime"] == "p-el":
            target_multiplicity = self.nuclear_target.Z
        elif self.scope["scattering_regime"] == "n-el":
            target_multiplicity = self.nuclear_target.N
        elif self.scope["scattering_regime"] == "DIS":
            target_multiplicity = 1
        else:
            logger.error(f"Scattering regime {self.scope['scattering_regime']} not supported.")

        # Normalize to total exposure
        exposure = self.experiment.NUMBER_OF_TARGETS[self.nuclear_target.name] * self.experiment.POTS

        # differential rate weights
        df_gen["w_event_rate"] = (
            weights["diff_event_rate"] * const.attobarn_to_cm2 / decay_rates * target_multiplicity * exposure * batch_f.norm["diff_event_rate"]
        )

        # flux averaged xsec weights (neglecting kinematics of decay)
        df_gen["w_flux_avg_xsec"] = weights["diff_flux_avg_xsec"] * const.attobarn_to_cm2 * target_multiplicity * exposure * batch_f.norm["diff_flux_avg_xsec"]

        df_gen.insert(
            loc=len(df_gen.columns), column="target", value=np.full(tot_nevents, self.ups_case.target.name),
        )
        df_gen["target"] = df_gen["target"].astype("string")
        df_gen.insert(
            loc=len(df_gen.columns), column="target_pdgid", value=self.ups_case.target.pdgid,
        )
        df_gen["target_pdgid"] = df_gen["target_pdgid"].astype("int")
        df_gen.insert(
            loc=len(df_gen.columns), column="scattering_regime", value=np.full(tot_nevents, self.ups_case.scattering_regime),
        )
        df_gen["scattering_regime"] = df_gen["scattering_regime"].astype("string")
        df_gen.insert(
            loc=len(df_gen.columns), column="helicity", value=np.full(tot_nevents, self.helicity),
        )
        df_gen["helicity"] = df_gen["helicity"].astype("string")
        df_gen.insert(
            loc=len(df_gen.columns), column="underlying_process", value=np.full(tot_nevents, self.underl_process_name),
        )
        df_gen["underlying_process"] = df_gen["underlying_process"].astype("string")

        # Helicity of incoming neutrino
        if self.nu_projectile.pdgid < 0:
            df_gen.insert(
                loc=len(df_gen.columns), column="h_projectile", value=np.full(tot_nevents, +1),
            )
        elif self.nu_projectile.pdgid > 0:
            df_gen.insert(
                loc=len(df_gen.columns), column="h_projectile", value=np.full(tot_nevents, -1),
            )

        # Helicity of outgoing HNL
        if self.helicity == "conserving":
            df_gen["h_parent", ""] = df_gen["h_projectile"]
        elif self.helicity == "flipping":
            df_gen["h_parent", ""] = -df_gen["h_projectile"]

        # saving the experiment class
        df_gen.attrs["experiment"] = self.experiment

        # saving the bsm_model class
        df_gen.attrs["model"] = self.bsm_model

        # saving the lifetime of the parent (upscattered) HNL
        df_gen.attrs[f"{self.nu_upscattered.name}_ctau0"] = const.get_decay_rate_in_cm(df_gen["w_decay_rate_0"].sum())

        # #########################################################################
        # PROPAGATE PARENT PARTICLE

        self.experiment.set_geometry()
        self.experiment.place_scatters(df_gen)

        if self.scope["enforce_prompt"]:
            geom.place_decay(df_gen, "P_decay_N_parent", l_decay_proper_cm=0.0, label="pos_decay")
        else:
            # decay only the first mother (typically the HNL produced)
            logger.info(f"Parent {self.ups_case.nu_upscattered.name} proper decay length: {df_gen.attrs[f'{self.nu_upscattered.name}_ctau0']:.3E} cm.\n")
            geom.place_decay(
                df_gen, "P_decay_N_parent", l_decay_proper_cm=df_gen.attrs[f"{self.nu_upscattered.name}_ctau0"], label="pos_decay",
            )

        # print final result
        logger.info(f"Predicted ({df_gen['w_event_rate'].sum():.3g} +/- {np.sqrt((df_gen['w_event_rate']**2).sum()):.3g}) events.\n")

        return df_gen


# merge all generation cases into one dictionary
def get_merged_MC_output(df1, df2):
    """
    take two pandas dataframes with events and combine them.
    Resetting index to go from (0,n_1+n_2) where n_i is the number of events in dfi
    """
    logger.debug(f"Appending {df2.underlying_process[0]}")
    df = pd.concat([df1, df2], axis=0).reset_index(drop=True)

    # for older versions of pandas, concat does not keep the attributes
    #  -- if they disappear, force first dataframe.
    if not df.attrs:
        logger.debug("DEBUG: Forcing the storage of the df.attrs using the first dataframe. This is done automatically for newer versions of pandas.")
        df.attrs = df1.attrs

    # Now we merge lifetimes
    for i in range(4, 7):
        this_ctau0 = f"N{i}_ctau0"
        if this_ctau0 in df1.attrs.keys() and this_ctau0 in df2.attrs.keys():
            # take the average
            df.attrs[this_ctau0] = 0.5 * (df1.attrs[this_ctau0] + df2.attrs[this_ctau0])

    return df


def get_samples(integ, batch_integrand, return_jac=False):
    """_summary_

    Args:
        integ (vegas.Integrator): vegas integrator object initialized by the user.

        batch_integrand (vegas.BatchIntegrand): vegas batch_integrand object created by the user.
        These are defined in integrands.py

        return_jac (bool, optional): if True, returns the jacobian of the integrand as well. Defaults to False.

    Raises:
        ValueError: if the integrand evaluates to nan

    Returns:
        tuple of np.ndarrays:
    """

    unit_samples = batch_integrand.dim * [[]]
    weights = defaultdict(partial(np.ndarray, 0))

    for x, y, wgt in integ.random_batch(yield_y=True, fcn=batch_integrand):

        # compute integrand on samples including jacobian factors
        if integ.uses_jac:
            jac = integ.map.jac1d(y)
        else:
            jac = None

        fx = batch_integrand(x, jac=jac)
        # weights
        for fx_i in fx.keys():
            if np.any(np.isnan(fx[fx_i])):
                raise ValueError(f"integrand {fx_i} evaluates to nan")
            weights[fx_i] = np.append(weights[fx_i], wgt * fx[fx_i])

        # MC samples in unit hypercube
        for i in range(batch_integrand.dim):
            unit_samples[i] = np.append(unit_samples[i], x[:, i])

    if return_jac:
        return np.array(unit_samples), weights, jac
    else:
        return np.array(unit_samples), weights


def run_vegas(batch_f, integ, NINT=10, NEVAL=1000, NINT_warmup=10, NEVAL_warmup=1000, **kwargs):
    """
    Function that calls vegas evaluations. This function defines the vegas parameters used by
    DarkNews throughout.

    Args:
        integ (vegas.Integrator): vegas integrator object initialized by the user.

        batch_integrand (vegas.BatchIntegrand): vegas batch_integrand object created by the user.

        NINT (int, optional): number of iterations for vegas. Defaults to 10.
        NEVAL (int, optional): number of evaluations in an iteration for vegas to 1000.
        NINT_warmup (int, optional): same as above, but for warmup run. Defaults to 10.
        NEVAL_warmup (int, optional): same as above, but for warmup run. Defaults to 1000.

    Returns:
        integ (vegas.Integrator): with the evaluated integrals.
    """

    # warm up the MC, adapting to the integrand
    integ(batch_f, nitn=NINT_warmup, neval=NEVAL_warmup, uses_jac=True, **kwargs)
    logger.debug("VEGAS warm-up completed.")

    # sample again, now saving result and turning off further adaption
    return integ(batch_f, nitn=NINT, neval=NEVAL, uses_jac=True, **kwargs)
