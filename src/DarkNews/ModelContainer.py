import logging
import logging.handlers
import sys
import numpy as np
from particle import literals as lp

# Dark Neutrino and MC stuff
import DarkNews as dn
logger = logging.getLogger("logger." + __name__)
prettyprinter = logging.getLogger("prettyprinter." + __name__)
from DarkNews.AssignmentParser import AssignmentParser
from DarkNews.nuclear_tools import NuclearTarget

GENERATOR_ARGS = [
    # scope
    "decay_product",
    "experiment",
    "nopelastic",
    "nocoh",
    "include_nelastic",
    "noHC",
    "noHF",
    "nu_flavors",
    "sample_geometry",
    "make_summary_plots",
    "enforce_prompt",
    "nuclear_targets",
    #
    # generator
    "loglevel",
    "verbose",
    "logfile",
    "neval",
    "nint",
    "neval_warmup",
    "nint_warmup",
    "seed",
    #
    # output
    "pandas",
    "parquet",
    "numpy",
    "hepevt",
    "hepevt_legacy",
    "hepmc2",
    "hepmc3",
    "hep_unweight",
    "unweighted_hep_events",
    "sparse",
    "print_to_float32",
    "path",
]

COMMON_MODEL_ARGS = [
    "name",
    "m4",
    "m5",
    "m6",
    "mzprime",
    "mhprime",
    "HNLtype",
    "mu_tr_e4",
    "mu_tr_e5",
    "mu_tr_e6",
    "mu_tr_mu4",
    "mu_tr_mu5",
    "mu_tr_mu6",
    "mu_tr_tau4",
    "mu_tr_tau5",
    "mu_tr_tau6",
    "mu_tr_44",
    "mu_tr_45",
    "mu_tr_46",
    "mu_tr_55",
    "mu_tr_56",
    "s_e4",
    "s_e5",
    "s_e6",
    "s_mu4",
    "s_mu5",
    "s_mu6",
    "s_tau4",
    "s_tau5",
    "s_tau6",
    "s_44",
    "s_45",
    "s_46",
    "s_55",
    "s_56",
    "s_66",
]

THREE_PORTAL_ARGS = [
    "gD",
    "epsilon",
    "epsilonZ",
    "alphaD",
    "epsilon2",
    "chi",
    "alpha_epsilon2",
    "theta",
    "Ue4",
    "Ue5",
    "Ue6",
    "Umu4",
    "Umu5",
    "Umu6",
    "Utau4",
    "Utau5",
    "Utau6",
    "UD4",
    "UD5",
    "UD6",
]

GENERIC_MODEL_ARGS = [
    "c_e4",
    "c_e5",
    "c_e6",
    "c_mu4",
    "c_mu5",
    "c_mu6",
    "c_tau4",
    "c_tau5",
    "c_tau6",
    "c_44",
    "c_45",
    "c_46",
    "c_55",
    "c_56",
    "c_66",
    "d_e4",
    "d_e5",
    "d_e6",
    "d_mu4",
    "d_mu5",
    "d_mu6",
    "d_tau4",
    "d_tau5",
    "d_tau6",
    "d_44",
    "d_45",
    "d_46",
    "d_55",
    "d_56",
    "d_66",
    "ceV",
    "ceA",
    "cuV",
    "cuA",
    "cdV",
    "cdA",
    "deV",
    "deA",
    "duV",
    "duA",
    "ddV",
    "ddA",
    "deS",
    "deP",
    "duS",
    "duP",
    "ddS",
    "ddP",
    "cprotonV",
    "cneutronV",
    "cprotonA",
    "cneutronA",
    "dprotonV",
    "dneutronV",
    "dprotonA",
    "dneutronA",
    "dprotonS",
    "dneutronS",
    "dprotonP",
    "dneutronP",
]


class ModelContainer:

    banner = r"""   ______           _        _   _                     
   |  _  \         | |      | \ | |                    
   | | | |__ _ _ __| | __   |  \| | _____      _____   
   | | | / _  | ___| |/ /   | .   |/ _ \ \ /\ / / __|  
   | |/ / (_| | |  |   <    | |\  |  __/\ V  V /\__ \  
   |___/ \__,_|_|  |_|\_\   \_| \_/\___| \_/\_/ |___/  """

    # handle parameters that can assume only certain values
    _choices = {
        "HNLtype": ["dirac", "majorana"],
        "decay_product": ["e+e-", "mu+mu-", "photon"],
        "nu_flavors": ["nu_e", "nu_mu", "nu_tau", "nu_e_bar", "nu_mu_bar", "nu_tau_bar"],
        "sparse": [0, 1, 2, 3, 4],
    }

    def __init__(self, param_file=None, **kwargs):
        """ModelContainer creates a set of upscattering and decay processes given user-defined model parameters.

            It instantiates an object to that contains the upscattering and decay processes for the user-defined settings.
            There are different ways to accomplish this:
                - file: read the file in input, parsing it looking for the different variables' values;
                - other arguments: variables' values.
            If a variable is not provided, it sets the default.
            at the end it looks inside the kwargs (so kwargs overwrite file definitions).

        Args:
            param_file (str, optional): path to model file. Defaults to None, which uses the parameters set at run-time.
            **kwargs: list of user-defined parameters

        Raises:
            AttributeError: when unused parameters are specified (raise only a
                warning if unused parameters are specified within the model file).
            FileNotFoundError: when param_file is specified, but the path is invalid.
            ValueError: when model, experiment, or generation parameters are not well defined.
        """

        # Choose what model to initialize
        captured_3portal_args = set(THREE_PORTAL_ARGS).intersection(kwargs.keys())
        captured_generic_args = set(GENERIC_MODEL_ARGS).intersection(kwargs.keys())
        if len(captured_3portal_args) >= 0 and len(captured_generic_args) == 0:
            logger.info("Initializing the three-portal model.")
            self._model_parameters = COMMON_MODEL_ARGS + THREE_PORTAL_ARGS
            self._model_class = dn.model.ThreePortalModel
        elif len(captured_3portal_args) == 0 and len(captured_generic_args) > 0:
            logger.info("Initializing a generic model.")
            self._model_parameters = COMMON_MODEL_ARGS + GENERIC_MODEL_ARGS
            self._model_class = dn.model.GenericHNLModel
        elif len(captured_3portal_args) > 0 and len(captured_generic_args) > 0:
            logger.error(
                f"Generic Model parameters {captured_generic_args} set at the same time as {captured_3portal_args}. Cannot mix generic and three-portal model arguments together."
            )
            raise ValueError("Two types of parameters were found. You cannot mix Generic Model parameters with three-portal Model parameters.")
        else:
            raise ValueError(f"Could not determine what model type to use with kwargs.keys = {kwargs.keys()}")

        self._parameters = GENERATOR_ARGS + self._model_parameters

        # Scope parameters
        self.name = None
        self.nu_flavors = ["nu_mu"]
        self.decay_product = "e+e-"
        self.nuclear_targets = []
        self.nopelastic = False
        self.include_nelastic = False
        self.nocoh = False
        self.noHC = False
        self.noHF = False
        self.sample_geometry = False
        self.make_summary_plots = False
        self.enforce_prompt = False

        # Generator parameters
        self.loglevel = "INFO"
        self.verbose = False
        self.logfile = None
        self.neval = int(1e4)
        self.nint = 20
        self.neval_warmup = int(1e3)
        self.nint_warmup = 10
        self.seed = None

        ####################################################
        # Loading parameters

        # the argument dictionaries (will contain valid arguments extracted from **kwargs)
        self.model_args_dict = {}

        # load file if not None
        if param_file is not None:
            self._load_file(param_file)

        # load constructor parameters
        self._load_parameters(raise_errors=True, **kwargs)

        ####################################################
        # Set up loggers
        self.prettyprinter = prettyprinter
        self.configure_logger(
            logger=logger,
            loglevel=self.loglevel,
            prettyprinter=self.prettyprinter,
            verbose=self.verbose,
            logfile=self.logfile,
        )
        prettyprinter.info(self.banner)

        ####################################################
        # Choose the model to be used in this generation
        self.bsm_model = self._model_class(**self.model_args_dict)

        ####################################################
        # MC evaluations and iterations
        dn.MC.NEVAL_warmup = self.neval_warmup
        dn.MC.NINT_warmup = self.nint_warmup
        dn.MC.NEVAL = self.neval
        dn.MC.NINT = self.nint

        # get the initial projectiles
        self.projectiles = [getattr(lp, nu) for nu in self.nu_flavors]

        # decide which helicities to use
        self.helicities = []
        if not self.noHC:
            self.helicities.append("conserving")
        if not self.noHF:
            self.helicities.append("flipping")

        ####################################################
        ## Determine scope of upscattering given the heavy nu spectrum
        _mass_strings = [f"{m}_{getattr(self.bsm_model, m)}_" for m in ["m6", "m5", "m4"] if getattr(self.bsm_model, m) is not None]
        # 3+1
        if len(_mass_strings) == 1:
            self.upscattered_nus = [dn.pdg.neutrino4]
            self.outgoing_nus = [dn.pdg.nulight]
        # 3+2
        elif len(_mass_strings) == 2:
            ## FIXING 3+2 process chain to be nualpha --> N5 --> N4
            self.upscattered_nus = [dn.pdg.neutrino5]
            self.outgoing_nus = [dn.pdg.neutrino4]
        # 3+3
        elif len(_mass_strings) == 3:
            self.upscattered_nus = [
                dn.pdg.neutrino4,
                dn.pdg.neutrino5,
                dn.pdg.neutrino6,
            ]
            self.outgoing_nus = [dn.pdg.nulight, dn.pdg.neutrino4, dn.pdg.neutrino5]
        else:
            logger.error("Error! Mass spectrum not allowed (m4,m5,m6) = ({self.bsm_model.m4:.4g},{self.bsm_model.m5:.4g},{self.bsm_model.m6:.4g}) GeV.")
            raise ValueError("Could not find a heavy neutrino spectrum from user input.")

        ####################################################
        # Create all model cases
        self._create_all_model_cases()

        # end __init__

    def _load_file(self, file):
        parser = AssignmentParser({})
        try:
            parser.parse_file(file=file, comments="#")
        except FileNotFoundError:
            logger.error(f"Error! File '{file}' not found.")
            raise
        # store variables
        self._load_parameters(raise_errors=False, **parser.parameters)

    def _load_parameters(self, raise_errors=True, **kwargs):
        # start from the list of parameters available
        for parameter in self._parameters:
            # take the value from the kwargs, if not provided, go to next parameter
            try:
                value = kwargs.pop(parameter)
            except KeyError:
                continue

            # check the value within the choices
            # if parameter in self._choices.keys() and value not in self._choices[parameter]:
            if parameter in self._choices.keys() and [
                *parameter,
                *self._choices[parameter],
            ] == set([*parameter, *self._choices[parameter]]):
                raise ValueError(
                    f"Parameter '{parameter}', invalid choice: {value}, (choose among " + ", ".join([f"{el}" for el in self._choices[parameter]]) + ")"
                )

            # set the parameter
            setattr(self, parameter, value)

            # save the parameters that pertain to the model classes in a dict
            if parameter in self._model_parameters:
                self.model_args_dict[parameter] = value

        # at the end, if kwargs is not empty, that would mean some parameters were unused, i.e. they are spelled wrong or do not exist: raise AttributeError
        if len(kwargs) != 0:
            if raise_errors:
                raise AttributeError("Parameters " + ", ".join(kwargs.keys()) + " were unused. Either not supported or misspelled.")
            else:
                logger.warning("Parameters " + ", ".join(kwargs.keys()) + " will not be used.")

    def _create_all_model_cases(self, **kwargs):
        """Create ups_case and decay_case objects

        Args:
            **kwargs:
                FLAVORS (list):             input flavors for projectiles ['nu_e','nu_mu','nu_tau','nu_e_bar','nu_mu_bar','nu_tau_bar']
                UPSCATTERED_NUS (list):     dark NU in upscattering process
                OUTFOING_NUS (list):        output flavors
                SCATTERING_REGIMES (list):  regimes of scattering process (coherent, p-el, n-el)
                INCLUDE_HC (bool):          flag to include helicity conserving case
                INCLUDE_HF (bool):          flag to include helicity flipping case
                NO_COH (bool):              flag to skip coherent case
                NO_PELASTIC (bool):         flag to skip proton elastic case
                INCLUDE_NELASTIC (bool):         flag to skip neutron elastic case
                DECAY_PRODUCTS (list):      decay processes to include
        """

        # Default values
        scope = {
            "NO_COH": self.nocoh,
            "NO_PELASTIC": self.nopelastic,
            "INCLUDE_NELASTIC": self.include_nelastic,
            "HELICITIES": self.helicities,
            "FLAVORS": self.projectiles,
            "UPSCATTERED_NUS": self.upscattered_nus,
            "OUTGOING_NUS": self.outgoing_nus,
            "DECAY_PRODUCTS": [self.decay_product],
            "SCATTERING_REGIMES": ["coherent", "p-el", "n-el"],
            "NUCLEAR_TARGETS": [NuclearTarget(_t) for _t in self.nuclear_targets],
        }
        # override default with kwargs
        scope.update(kwargs)

        if len(scope["SCATTERING_REGIMES"]) == 0:
            logger.error("No scattering regime found -- please specify at least one.")
            raise ValueError
        if len(scope["DECAY_PRODUCTS"]) == 0:
            logger.error("No visible decay products found -- please specify at least one final state (e+e-, mu+mu- or photon).")
            raise ValueError
        if len(scope["HELICITIES"]) == 0:
            logger.error("No helicity structure was allowed -- please allow at least one type: HC or HF.")
            raise ValueError
        if len(scope["FLAVORS"]) == 0:
            logger.error("No projectile neutrino flavors specified -- please specify at least one.")
            raise ValueError

        # create instances of all MC cases of interest
        logger.debug("Creating instances of MC cases:")
        self.ups_cases = {}
        self.dec_cases = {}
        # neutrino flavor initiating scattering
        for flavor in scope["FLAVORS"]:
            # neutrino produced in upscattering
            for nu_upscattered in scope["UPSCATTERED_NUS"]:
                # neutrino produced in the subsequent decay
                for nu_outgoing in scope["OUTGOING_NUS"]:
                    # final state to consider in decay process
                    for decay_product in scope["DECAY_PRODUCTS"]:
                        # skip cases with obviously forbidden decay
                        if np.abs(nu_outgoing.pdgid) >= np.abs(nu_upscattered.pdgid):
                            continue
                        # material on which upscattering happened
                        for nuclear_target in scope["NUCLEAR_TARGETS"]:
                            # scattering regime to use
                            for scattering_regime in scope["SCATTERING_REGIMES"]:

                                # skip disallowed regimes
                                if ((scattering_regime in ["n-el"]) and (nuclear_target.N < 1)) or (  # no neutrons
                                    (scattering_regime in ["coherent"]) and (not nuclear_target.is_nucleus)
                                ):  # coherent = p-el for hydrogen
                                    continue
                                elif (
                                    (scattering_regime in ["coherent"] and scope["NO_COH"])
                                    or (scattering_regime in ["p-el"] and scope["NO_PELASTIC"])
                                    or (scattering_regime in ["n-el"])
                                    and (not scope["INCLUDE_NELASTIC"])  # do not include n-el
                                ):
                                    continue
                                else:
                                    for helicity in scope["HELICITIES"]:

                                        # Upscattering process
                                        ups_args = {
                                            "nuclear_target": nuclear_target,
                                            "scattering_regime": scattering_regime,
                                            "nu_projectile": flavor,
                                            "nu_upscattered": nu_upscattered,
                                            "helicity": helicity,
                                        }
                                        ups_key = tuple(ups_args.values())
                                        ups_case = dn.processes.UpscatteringProcess(
                                            nu_projectile=ups_args["nu_projectile"],
                                            nu_upscattered=ups_args["nu_upscattered"],
                                            nuclear_target=ups_args["nuclear_target"],
                                            scattering_regime=ups_args["scattering_regime"],
                                            helicity=ups_args["helicity"],
                                            TheoryModel=self.bsm_model,
                                        )
                                        if ups_key not in self.ups_cases.keys():
                                            self.ups_cases[ups_key] = ups_case

                                        # Decay process
                                        decay_args = {
                                            "nu_parent": nu_upscattered,
                                            "nu_daughter": nu_outgoing,
                                            "h_parent": ups_case.h_upscattered,
                                            "decay_product": decay_product,
                                        }
                                        dec_key = tuple(decay_args.values())
                                        if decay_product == "e+e-":
                                            decay_pdg = dn.pdg.electron
                                            decays_to_dilepton = True
                                            decays_to_singlephoton = False

                                        elif decay_product == "mu+mu-":
                                            decay_pdg = dn.pdg.muon
                                            decays_to_dilepton = True
                                            decays_to_singlephoton = False

                                        elif decay_product == "photon":
                                            decay_pdg = dn.pdg.photon
                                            decays_to_dilepton = False
                                            decays_to_singlephoton = True

                                        if decays_to_dilepton:
                                            decay_case = dn.processes.FermionDileptonDecay(
                                                nu_parent=decay_args["nu_parent"],
                                                nu_daughter=decay_args["nu_daughter"],
                                                final_lepton1=decay_pdg,
                                                final_lepton2=-decay_pdg,
                                                h_parent=decay_args["h_parent"],
                                                TheoryModel=self.bsm_model,
                                            )
                                        elif decays_to_singlephoton:
                                            decay_case = dn.processes.FermionSinglePhotonDecay(
                                                nu_parent=decay_args["nu_parent"],
                                                nu_daughter=decay_args["nu_daughter"],
                                                h_parent=decay_args["h_parent"],
                                                TheoryModel=self.bsm_model,
                                            )
                                        else:
                                            logger.error("Error! Could not determine what type of decay class to use.")
                                            raise ValueError
                                        if dec_key not in self.dec_cases.keys():
                                            self.dec_cases[dec_key] = decay_case

        return self.ups_cases, self.dec_cases

    def configure_logger(
        self,
        logger,
        loglevel=logging.INFO,
        prettyprinter=None,
        logfile=None,
        verbose=False,
    ):
        """
        Configure the DarkNews logger

            logger -->

            prettyprint --> secondary logger for pretty-print messages.

        Args:
            logger (logging.Logger): main DarkNews logger to be configured.
                                    It handles all debug, info, warning, and error messages

            loglevel (int, optional): what logging level to use.
                                    Can be logging.(DEBUG, INFO, WARNING, or ERROR). Defaults to logging.INFO.

            prettyprinter (logging.Logger, optional): if passed, configures this logger for the prettyprint.
                                                    Cannot override the main logger levelCannot override the main logger level.
                                                    Defaults to None.

            logfile (str, optional): path to file where to log the output. Defaults to None.

            verbose (bool, optional): If true, keep date and time in the logger format. Defaults to False.

        Raises:
            ValueError: _description_
        """

        loglevel = loglevel.upper()
        _numeric_level = getattr(logging, loglevel, None)
        if not isinstance(_numeric_level, int):
            raise ValueError("Invalid log level: %s" % self.loglevel)
        logger.setLevel(_numeric_level)

        if logfile:
            # log to files with max 1 MB with up to 4 files of backup
            handler = logging.handlers.RotatingFileHandler(f"{logfile}", maxBytes=1000000, backupCount=4)

        else:
            # stdout only
            handler = logging.StreamHandler(stream=sys.stdout)
            if prettyprinter:
                pretty_handler = logging.StreamHandler(stream=sys.stdout)
                pretty_handler.setLevel(loglevel)
                delimiter = "---------------------------------------------------------"
                pretty_handler.setFormatter(logging.Formatter(delimiter + "\n%(message)s\n"))
                # update pretty printer
                if prettyprinter.hasHandlers():
                    prettyprinter.handlers.clear()
                prettyprinter.addHandler(pretty_handler)

        handler.setLevel(loglevel)
        if verbose:
            handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s - %(levelname)s - %(name)s:\n\t%(message)s\n",
                    datefmt="%H:%M:%S",
                )
            )
        else:
            handler.setFormatter(logging.Formatter("%(message)s"))

        if logger.hasHandlers():
            logger.handlers.clear()
        logger.addHandler(handler)
