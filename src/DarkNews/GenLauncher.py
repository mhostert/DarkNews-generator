import logging
import logging.handlers
import sys
import numpy as np
import os
from pathlib import Path
from particle import literals as lp

# Dark Neutrino and MC stuff
import DarkNews as dn
from DarkNews import logger, prettyprinter
from DarkNews.AssignmentParser import AssignmentParser

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


class GenLauncher:

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
        "nu_flavors": ["nu_e", "nu_mu", "nu_tau", "nu_e_bar", "nu_mu_bar", "nu_tau_bar",],
    }


    def __init__(self, param_file=None, **kwargs):
        """GenLauncher launches the generator with user-defined model parameters, experimental configs, and generation settings.

            It instantiates an object to that can run the generator for the user-defined settings.
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
        self.experiment = "miniboone_fhc"
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
        
        # Output parameters
        self.pandas = True
        self.parquet = False
        self.numpy = False
        self.hepevt = False
        self.hepevt_legacy = False
        self.hepmc2 = False
        self.hepmc3 = False
        self.hep_unweight = False
        self.unweighted_hep_events = 100
        self.sparse = False
        self.print_to_float32 = False
        self.path = "."

        ####################################################
        # Loading parameters

        # the argument dictionaries (will contain valid arguments extracted from **kwargs)
        self.model_args_dict = {}

        # load file if not None
        if param_file is not None:
            self._load_file(param_file)

        # load constructor parameters
        self._load_parameters(raise_errors=True, **kwargs)

        if self.parquet and not dn.HAS_PYARROW:
            logger.error("Error: pyarrow is not installed.")
            raise ModuleNotFoundError("pyarrow is not installed.")

        ####################################################
        # Set up loggers
        self.prettyprinter = prettyprinter
        self.configure_logger(
            logger=logger, loglevel=self.loglevel, prettyprinter=self.prettyprinter, verbose=self.verbose, logfile=self.logfile,
        )
        prettyprinter.info(self.banner)

        ####################################################
        # Choose the model to be used in this generation
        self.bsm_model = self._model_class(**self.model_args_dict)

        ####################################################
        # Choose experiment and scope of simulation
        self.experiment = dn.detector.Detector(self.experiment)

        ####################################################
        # MC evaluations and iterations
        dn.MC.NEVAL_warmup = self.neval_warmup
        dn.MC.NINT_warmup = self.nint_warmup
        dn.MC.NEVAL = self.neval
        dn.MC.NINT = self.nint

        # get the initial projectiles
        self.projectiles = [getattr(lp, nu) for nu in self.nu_flavors]

        ####################################################
        # Default data path based on model and experimental definitioons

        # set the path of the experiment name (needed in the case of custom experiment path)
        # it uses the name of the detector object (don't use the path otherwise it could include
        # the full directory tree inside the data_path if the file picked is not in the same
        # directory)
        exp_path_part = os.path.basename(str(self.experiment)).rsplit(".", maxsplit=1)[0]

        _boson_string = ""
        if self.bsm_model.mzprime is not None:
            _boson_string += f"_mzprime_{self.bsm_model.mzprime:.4g}"
        if self.bsm_model.mhprime is not None:
            _boson_string += f"_mhprime_{self.bsm_model.mhprime:.4g}"

        # 3+1
        if self.bsm_model.m4 is not None and self.bsm_model.m5 is None and self.bsm_model.m6 is None:
            self.upscattered_nus = [dn.pdg.neutrino4]
            self.outgoing_nus = [dn.pdg.nulight]
            self.data_path = Path(f"{self.path}/data/{exp_path_part}/3plus1/m4_{self.bsm_model.m4:.4g}{_boson_string}_{self.bsm_model.HNLtype}/")

        # 3+2
        elif self.bsm_model.m4 is not None and self.bsm_model.m5 is not None and self.bsm_model.m6 is None:
            ## FIXING 3+2 process chain to be nualpha --> N5 --> N4
            self.upscattered_nus = [dn.pdg.neutrino5]
            self.outgoing_nus = [dn.pdg.neutrino4]
            self.data_path = Path(
                f"{self.path}/data/{exp_path_part}/3plus2/m5_{self.bsm_model.m5:.4g}_m4_{self.bsm_model.m4:.4g}{_boson_string}_{self.bsm_model.HNLtype}/"
            )

        # 3+3
        elif self.bsm_model.m4 is not None and self.bsm_model.m5 is not None and self.bsm_model.m6 is not None:
            self.upscattered_nus = [
                dn.pdg.neutrino4,
                dn.pdg.neutrino5,
                dn.pdg.neutrino6,
            ]
            self.outgoing_nus = [dn.pdg.nulight, dn.pdg.neutrino4, dn.pdg.neutrino5]
            self.data_path = Path(
                f"{self.path}/data/{exp_path_part}/3plus3/m6_{self.bsm_model.m6:.4g}_m5_{self.bsm_model.m5:.4g}_m4_{self.bsm_model.m4:.4g}{_boson_string}_{self.bsm_model.HNLtype}/"
            )

        else:
            logger.error("Error! Mass spectrum not allowed (m4,m5,m6) = ({self.bsm_model.m4:.4g},{self.bsm_model.m5:.4g},{self.bsm_model.m6:.4g}) GeV.")
            raise ValueError("Could not find a heavy neutrino spectrum from user input.")

        ####################################################
        # Miscellaneous checks 
        if self.hep_unweight:
            logger.warning(
                f"Unweighted events requested. This feature requires a large number of weighted events with respect to the requested number of hep-formatted events. Currently: n_unweighted/n_eval = {self.unweighted_hep_events/self.neval*100}%."
            )
        

        ####################################################
        # Create all MC cases
        self._create_all_MC_cases()

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
            if parameter in self._choices.keys() and [*parameter, *self._choices[parameter],] == set([*parameter, *self._choices[parameter]]):
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


    def _create_all_MC_cases(self, **kwargs):
        """Create MC_events objects and run the MC computations

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
            "INCLUDE_HC": not self.noHC,
            "INCLUDE_HF": not self.noHF,
            "FLAVORS": self.projectiles,
            "UPSCATTERED_NUS": self.upscattered_nus,
            "OUTGOING_NUS": self.outgoing_nus,
            "DECAY_PRODUCTS": [self.decay_product],
            "SCATTERING_REGIMES": ["coherent", "p-el"],  # , 'n-el'],
        }
        # override default with kwargs
        scope.update(kwargs)

        if not scope["SCATTERING_REGIMES"]:
            logger.error("No scattering regime found -- please specify at least one.")
            raise ValueError
        if not scope["DECAY_PRODUCTS"]:
            logger.error("No visible decay products found -- please specify at least one final state (e+e-, mu+mu- or photon).")
            raise ValueError
        if not scope["INCLUDE_HC"] and not scope["INCLUDE_HF"]:
            logger.error("No helicity structure was allowed -- please allow at least one type: HC or HF.")
            raise ValueError
        if not scope["FLAVORS"]:
            logger.error("No projectile neutrino flavors specified -- please specify at least one.")
            raise ValueError

        # create instances of all MC cases of interest
        logger.debug("Creating instances of MC cases:")
        self.gen_cases = []
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
                        for nuclear_target in self.experiment.NUCLEAR_TARGETS:
                            # scattering regime to use
                            for scattering_regime in scope["SCATTERING_REGIMES"]:
                                # skip disallowed regimes
                                if ((scattering_regime in ["n-el"]) and (nuclear_target.N < 1)) or (  # no neutrons
                                    (scattering_regime in ["coherent"]) and (not nuclear_target.is_nucleus)
                                ):  # coherent = p-el for hydrogen
                                    continue
                                elif (
                                    (scattering_regime in ["coherent"] and scope["NO_COH"])\
                                    or (scattering_regime in ["p-el"] and scope["NO_PELASTIC"])\
                                    or (scattering_regime in ["n-el"])\
                                    and (not scope["INCLUDE_NELASTIC"])  # do not include n-el
                                ):
                                    continue
                                else:
                                    # bundle arguments of MC_events here
                                    args = {
                                        "nuclear_target": nuclear_target,
                                        "scattering_regime": scattering_regime,
                                        "nu_projectile": flavor,
                                        "nu_upscattered": nu_upscattered,
                                        "nu_outgoing": nu_outgoing,
                                        "decay_product": decay_product,
                                        "enforce_prompt": self.enforce_prompt,
                                    }

                                    if scope["INCLUDE_HC"]:  # helicity conserving scattering
                                        mc_case = dn.MC.MC_events(self.experiment, bsm_model=self.bsm_model, **args, helicity="conserving",)
                                        self.gen_cases.append(mc_case)

                                    if scope["INCLUDE_HF"]:  # helicity flipping scattering
                                        mc_case = dn.MC.MC_events(self.experiment, bsm_model=self.bsm_model, **args, helicity="flipping",)
                                        self.gen_cases.append(mc_case)
                                    
                                    logger.debug(f"Created an MC instance of {self.gen_cases[-1].underl_process_name}.")

        return self.gen_cases

    def _scramble_df(self):
        self.df = self.df.sample(frac=1, axis=0).reset_index(drop=True)

    def _drop_zero_weight_samples(self):
        zero_entries = self.df["w_event_rate"] == 0
        if zero_entries.sum() / len(self.df.index) > 0.01:
            logger.warning(
                f"Warning: number of entries with w_event_rate = 0 surpasses 1% of number of samples. Found: {zero_entries.sum()/len(self.df.index)}."
            )
        self.df = self.df.drop(self.df[zero_entries].index).reset_index(drop=True)

    def run(self, loglevel=None, verbose=None, logfile=None, overwrite_path=None):
        """
        Run GenLauncher generation of events

        Args:
            loglevel (int, optional): what logging level to use. Can be logging.(DEBUG, INFO, WARNING, or ERROR). Defaults to logging.INFO.

            verbose (bool, optional): If true, keep date and time in the logger format. Defaults to False.

            logfile (str, optional): path to file where to log the output. Defaults to None.

            overwrite_path (str, optional): new path to save the data, it overwrites the default.

        Returns:
            pd.DataFrame: the final pandas dataframe with all the events
        """

        ####################################################
        args = {"loglevel": loglevel, "verbose": verbose, "logfile": logfile}
        for attr in args.keys():
            if args[attr] is not None:
                setattr(self, attr, args[attr])

        ############
        # temporarily overwrite path
        if overwrite_path:
            old_path = self.data_path
            self.data_path = Path(overwrite_path + "/")

        ############
        # create directory tree
        try:
            os.makedirs(self.data_path)
        except OSError:
            logger.warning("Directory tree for this run already exists. Overriding it.")

        ############
        # superseed original logger configuration
        self.configure_logger(
            logger=logger, prettyprinter=self.prettyprinter, loglevel=self.loglevel, verbose=self.verbose, logfile=self.logfile,
        )

        ######################################
        # run generator
        logger.debug("Now running the generator for each instance.")
        # Set theory params and run generation of events

        prettyprinter.info("Generating Events using the neutrino-nucleus upscattering engine")
        # numpy set used by vegas
        if self.seed:
            np.random.seed(self.seed)

        self.df = self.gen_cases[0].get_MC_events()
        for mc in self.gen_cases[1:]:
            self.df = dn.MC.get_merged_MC_output(self.df, mc.get_MC_events())

        # scramble events for minimum bias
        self._drop_zero_weight_samples()
        # eliminate events with zero weights. If too many, raise warning.
        self._scramble_df()

        #################################################
        # Save attrs
        self.df.attrs["data_path"] = self.data_path

        prettyprinter.info(
            f"Generation successful\n\nTotal events predicted:\n({np.sum(self.df['w_event_rate']):.3g} +/- {np.sqrt(np.sum(self.df['w_event_rate']**2)):.3g}) events."
        )

        ############################################################################
        # Print events to file
        self.dn_printer = dn.printer.Printer(self.df, sparse=self.sparse, print_to_float32=self.print_to_float32, decay_product=self.decay_product,)
        if self.pandas:
            self.dn_printer.print_events_to_pandas()
        if self.parquet:
            self.dn_printer.print_events_to_parquet()
        if self.numpy:
            self.dn_printer.print_events_to_ndarray()
        if self.hepevt_legacy:
            self.dn_printer.print_events_to_hepevt_legacy(hep_unweight=self.hep_unweight, unweighted_hep_events=self.unweighted_hep_events)
        if self.hepmc2:
            self.dn_printer.print_events_to_hepmc2(hep_unweight=self.hep_unweight, unweighted_hep_events=self.unweighted_hep_events)
        if self.hepmc3:
            self.dn_printer.print_events_to_hepmc3(hep_unweight=self.hep_unweight, unweighted_hep_events=self.unweighted_hep_events)
        if self.hepevt:
            self.dn_printer.print_events_to_hepevt(hep_unweight=self.hep_unweight, unweighted_hep_events=self.unweighted_hep_events)

        #############################################################################
        # Make summary plots?
        if self.make_summary_plots:
            logger.info("Making summary plots of the kinematics of the process...")
            try:
                import matplotlib
            except ImportError as e:
                logger.warning("Warning! Could not find matplotlib -- stopping the making of summary plots.")
            else:
                self.path_to_summary_plots = Path(self.data_path) / "summary_plots/"
                dn.plot_tools.batch_plot(self.df, self.path_to_summary_plots, title=rf"{self.name}")
            logger.info(f"Plots saved in {self.path_to_summary_plots}.")

        # restore overwritten path
        if overwrite_path:
            self.data_path = old_path

        return self.df

    def configure_logger(
        self, logger, loglevel=logging.INFO, prettyprinter=None, logfile=None, verbose=False,
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
            handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(name)s:\n\t%(message)s\n", datefmt="%H:%M:%S",))
        else:
            handler.setFormatter(logging.Formatter("%(message)s"))

        if logger.hasHandlers():
            logger.handlers.clear()
        logger.addHandler(handler)
