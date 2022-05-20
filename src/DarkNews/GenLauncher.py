import logging
import sys
import numpy as np
import os

# Dark Neutrino and MC stuff
import DarkNews as dn
from DarkNews import logger, prettyprinter
from DarkNews.AssignmentParser import AssignmentParser

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
        "decay_product": ["e+e-", "mu+mu-", "photon"]
    }
    # parameters names list
    _parameters = [
        "gD", "alphaD", "epsilon", "epsilon2", "chi", "alpha_epsilon2", 
        "Ue4", "Ue5", "Ue6", "Umu4", "Umu5", "Umu6", "Utau4", "Utau5", "Utau6", 
        "UD4", "UD5", "UD6", "m4", "m5", "m6", "mzprime", "HNLtype", 
        "mu_tr_e4", "mu_tr_e5", "mu_tr_e6", "mu_tr_mu4", "mu_tr_mu5", "mu_tr_mu6", 
        "mu_tr_tau4", "mu_tr_tau5", "mu_tr_tau6", "mu_tr_44", "mu_tr_45", "mu_tr_46", "mu_tr_55", "mu_tr_56",  
        "s_e4", "s_e5", "s_e6", "s_mu4", "s_mu5", "s_mu6", "s_tau4", "s_tau5", "s_tau6", 
        "s_44", "s_45", "s_46", "s_55", "s_56", "s_66", "mhprime","theta",
        "decay_product", "exp", "nopelastic", "nocoh", "noHC", "noHF", 
        "loglevel", "verbose", "logfile", "neval", "nint", "neval_warmup", "nint_warmup", 
        "pandas", "parquet", "numpy", "hepevt", "hepevt_unweigh", "hepevt_events", 
        "sparse", "print_to_float32", "sample_geometry", "summary_plots", "path"
    ]

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
            ValueError: when model, exp, or generation parameters are not well defined.
        """

        # set defaults
        self.gD = 1.0
        self.alphaD = None
        self.epsilon = 1e-2
        self.epsilon2 = None
        self.chi = None
        self.alpha_epsilon2 = None
        self.Ue4 = 0.
        self.Ue5 = 0.
        self.Ue6 = 0.
        self.Umu4 = np.sqrt(1.5e-6*7/4)
        self.Umu5 = np.sqrt(11.5e-6)
        self.Umu6 = 0.
        self.Utau4 = 0.
        self.Utau5 = 0.
        self.Utau6 = 0.
        self.UD4 = 1.0
        self.UD5 = 1.0
        self.UD6 = 1.0
        self.m4 = 0.140
        self.m5 = None
        self.m6 = None
        self.mzprime = 1.25
        self.HNLtype = "majorana"
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
        self.theta = 0.0
        self.mhprime = 1e10
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
        self.decay_product = "e+e-"
        self.exp = "miniboone_fhc"
        self.nopelastic = False
        self.include_nelastic = False
        self.nocoh = False
        self.noHC = False
        self.noHF = False
        self.loglevel = "INFO"
        self.verbose = False
        self.logfile = None
        self.neval = int(1e4)
        self.nint = 20
        self.neval_warmup = int(1e3)
        self.nint_warmup = 10
        self.pandas = True
        self.parquet = False
        self.numpy = False
        self.hepevt = False
        self.hepevt_unweigh = False
        self.hepevt_events = 100
        self.sparse = False
        self.print_to_float32 = False
        self.sample_geometry = False
        self.summary_plots = True
        self.path = "."

        # load file if not None
        if param_file is not None:
            self._load_file(param_file)

        # load constructor parameters
        self._load_parameters(raise_errors=True, **kwargs)

        # build args_dict to pass to various methods
        args_dict = {}
        for parameter in self._parameters:
            args_dict[parameter] = getattr(self, parameter)

        #########################
        # Set up loggers
        self.prettyprinter = prettyprinter
        self.configure_logger(
            logger=logger,
            loglevel=self.loglevel,
            prettyprinter=self.prettyprinter,
            verbose=self.verbose,
            logfile=self.logfile
        )

        prettyprinter.info(self.banner)

        #########################
        # Set BSM parameters
        self.bsm_model = dn.model.create_model(**args_dict)

        ####################################################
        # Choose experiment and scope of simulation
        self.experiment = dn.detector.Detector(self.exp)
    
        ##########################
        # MC evaluations and iterations
        dn.MC.NEVAL_warmup = self.neval_warmup
        dn.MC.NINT_warmup = self.nint_warmup
        dn.MC.NEVAL = self.neval
        dn.MC.NINT  = self.nint

        ####################################################
        # Set the model to use

        # 3+1
        if (self.bsm_model.m4 and not self.bsm_model.m5 and not self.bsm_model.m6) :
            self.upscattered_nus = [dn.pdg.neutrino4]
            self.outgoing_nus =[dn.pdg.nulight]
            self.data_path = f'{self.path}/data/{self.exp}/3plus1/m4_{self.bsm_model.m4:.4g}_mzprime_{self.bsm_model.mzprime:.4g}_{self.bsm_model.HNLtype}/'

        # 3+2
        elif (self.bsm_model.m4 and self.bsm_model.m5 and not self.bsm_model.m6):
            ## FIXING 3+2 process chain to be numu --> N5 --> N4
            self.upscattered_nus = [dn.pdg.neutrino5]
            self.outgoing_nus =[dn.pdg.neutrino4]
            # upscattered_nus = [dn.pdg.neutrino4,dn.pdg.neutrino5]
            # outgoing_nus =[dn.pdg.numu,dn.pdg.neutrino4]
            self.data_path = f'{self.path}/data/{self.exp}/3plus2/m5_{self.bsm_model.m5:.4g}_m4_{self.bsm_model.m4:.4g}_mzprime_{self.bsm_model.mzprime:.4g}_{self.bsm_model.HNLtype}/'

        # 3+3
        elif (self.bsm_model.m4 and self.bsm_model.m5 and self.bsm_model.m6):
            self.upscattered_nus = [dn.pdg.neutrino4,dn.pdg.neutrino5,dn.pdg.neutrino6]
            self.outgoing_nus =[dn.pdg.nulight,dn.pdg.neutrino4,dn.pdg.neutrino5]
            self.data_path = f'{self.path}/data/{self.exp}/3plus3/m6_{self.bsm_model.m6:.4g}_m5_{self.bsm_model.m5:.4g}_m4_{self.bsm_model.m4:.4g}_mzprime_{self.bsm_model.mzprime:.4g}_{self.bsm_model.HNLtype}/'

        else:
            logger.error('ERROR! Mass spectrum not allowed.')
            raise ValueError 

        # create directory tree
        try:
            os.makedirs(self.data_path)
        except OSError:
            logger.warning("Note that the directory tree for this run already exists.")

        ####################################################
        # Create all MC cases
        self._create_all_MC_cases()

    def _load_file(self, file):
        parser = AssignmentParser({})
        try:
            parser.parse_file(file=file, comments="#")
        except FileNotFoundError:
            print(f"File '{file}' not found.")
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
            if parameter in self._choices.keys() and value not in self._choices[parameter]:
                raise ValueError(f"Parameter '{parameter}', invalid choice: {value}, (choose among " + ", ".join([f"{el}" for el in self._choices[parameter]]) + ")")
            # set the parameter
            setattr(self, parameter, value)
        # at the end, if kwargs is not empty, that would mean some parameters were unused, i.e. they are spelled wrong or do not exist: raise AttributeError
        if len(kwargs) != 0:
            if raise_errors:
                raise AttributeError("Parameters " + ", ".join(kwargs.keys()) + " were unused. Either not supported or spelled wrong.")
            else:
                logger.warning("Parameters " + ", ".join(kwargs.keys()) + " will not be used.")

    def _create_all_MC_cases(self, **kwargs):
        """ Create MC_events objects and run the MC computations

        Args:
            **kwargs:
                FLAVORS (list):             input flavors 
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
        scope = {  'NO_COH': self.nocoh,
                    'NO_PELASTIC': self.nopelastic,
                    'INCLUDE_NELASTIC': self.include_nelastic,
                    'INCLUDE_HC': not self.noHC,
                    'INCLUDE_HF': not self.noHF,
                    'FLAVORS': [dn.pdg.numu],
                    'UPSCATTERED_NUS': self.upscattered_nus,
                    'OUTGOING_NUS': self.outgoing_nus,
                    'DECAY_PRODUCTS': [self.decay_product],
                    'SCATTERING_REGIMES': ['coherent', 'p-el'],#, 'n-el'],
                }
        # override default with kwargs
        scope.update(kwargs)

        if not scope['SCATTERING_REGIMES']:
            logger.error('No scattering regime found -- please specify at least one.')
            raise ValueError 
        if not scope['DECAY_PRODUCTS']:
            logger.error('No visible decay products found -- please specify at least one final state (e+e-, mu+mu- or photon).')
            raise ValueError 
        if not scope['INCLUDE_HC'] and not scope['INCLUDE_HF']:
            logger.error('No helicity structure was allowed -- please allow at least one type: HC or HF.')
            raise ValueError 
        if not scope['FLAVORS']:
            logger.error('No projectile neutrino flavors specified -- please specify at least one.')
            raise ValueError 


        # create instances of all MC cases of interest
        logger.debug(f"Creating instances of MC cases:")
        self.gen_cases = []
        # neutrino flavor initiating scattering
        for flavor in scope['FLAVORS']:
            # neutrino produced in upscattering
            for nu_upscattered in scope['UPSCATTERED_NUS']:
                #neutrino produced in the subsequent decay
                for nu_outgoing in scope['OUTGOING_NUS']:
                    # final state to consider in decay process
                    for decay_product in scope['DECAY_PRODUCTS']:
                        # skip cases with obviously forbidden decay 
                        if np.abs(nu_outgoing.pdgid) >= np.abs(nu_upscattered.pdgid):
                            continue
                        # material on which upscattering happened
                        for nuclear_target in self.experiment.NUCLEAR_TARGETS:
                            # scattering regime to use
                            for scattering_regime in scope['SCATTERING_REGIMES']:
                                # skip disallowed regimes
                                if (
                                        ( (scattering_regime in ['n-el']) and (nuclear_target.N < 1)) # no neutrons
                                        or
                                        ( (scattering_regime in ['coherent']) and (not nuclear_target.is_nucleus)) # coherent = p-el for hydrogen
                                    ):
                                    continue 
                                elif ( (scattering_regime in ['coherent'] and scope['NO_COH'])
                                    or 
                                        (scattering_regime in ['p-el'] and scope['NO_PELASTIC'])
                                    or
                                        (scattering_regime in ['n-el']) and (not scope['INCLUDE_NELASTIC']) # do not include n-el
                                    ):
                                    continue
                                else:
                                    # bundle arguments of MC_events here
                                    args = {'nuclear_target' : nuclear_target,
                                            'scattering_regime' : scattering_regime,
                                            'nu_projectile' : flavor, 
                                            'nu_upscattered' : nu_upscattered,
                                            'nu_outgoing' : nu_outgoing, 
                                            'decay_product' : decay_product,
                                            }

                                    if scope['INCLUDE_HC']:  # helicity conserving scattering
                                        self.gen_cases.append(dn.MC.MC_events(self.experiment, **args, helicity = 'conserving'))

                                    if scope['INCLUDE_HF']:  # helicity flipping scattering
                                        self.gen_cases.append(dn.MC.MC_events(self.experiment, **args, helicity = 'flipping'))

                                    logger.debug(f"Created an MC instance of {self.gen_cases[-1].underl_process_name}.")
                                    
        
        # Assign new physics parameters
        for g in self.gen_cases:
            g.set_theory_params(self.bsm_model)
            
        return self.gen_cases


    def run(self, loglevel=None, verbose=None, logfile=None):
        """
        Run GenLauncher generation of events

        Args:            
            loglevel (int, optional): what logging level to use. Can be logging.(DEBUG, INFO, WARNING, or ERROR). Defaults to logging.INFO.
            
            prettyprinter (logging.Logger, optional): if passed, configures this logger for the prettyprint. Defaults to None.
            
            logfile (str, optional): path to file where to log the output. Defaults to None.
            
            verbose (bool, optional): If true, keep date and time in the logger format. Defaults to False.
        
        Returns:
            pd.DataFrame: the final pandas dataframe with all the events
        """

        ####################################################
        args = {"loglevel": loglevel, "verbose": verbose, "logfile": logfile}
        for attr in args.keys():
            if args[attr] is not None:
                setattr(self, attr, args[attr])

        ############
        # supersede original logger configuration 
        self.configure_logger( 
            logger = logger,
            prettyprinter = self.prettyprinter,
            loglevel = self.loglevel, 
            verbose = self.verbose,
            logfile = self.logfile
        )

        ######################################
        # run generator
        logger.debug(f"Now running the generator for each instance.")
        # Set theory params and run generation of events
        
        prettyprinter.info(f"Generating Events using the neutrino-nucleus upscattering engine")
        self.df = self.gen_cases[0].get_MC_events()
        for mc in self.gen_cases[1:]:
            self.df = dn.MC.get_merged_MC_output(self.df, mc.get_MC_events())

        #################################################
        # Save attrs
        self.df.attrs['data_path'] = self.data_path

        prettyprinter.info(f"Generation successful\n\nTotal events predicted:\n({np.sum(self.df['w_event_rate']):.3g} +/- {np.sqrt(np.sum(self.df['w_event_rate']**2)):.3g}) events.")

        ############################################################################
        # Print events to file
        self.dn_printer = dn.printer.Printer(self.df, sparse=self.sparse, print_to_float32=self.print_to_float32)
        if self.pandas:
            self.dn_printer.print_events_to_pandas()
        if self.parquet:
            self.dn_printer.print_events_to_parquet()
        if self.numpy:
            self.dn_printer.print_events_to_ndarray()
        if self.hepevt:
            self.dn_printer.print_events_to_HEPEVT(unweigh= self.hepevt_unweigh, 
                                                    max_events=self.hepevt_events,
                                                    decay_product=self.decay_product)
        
        return self.df




    def configure_logger(self, logger, loglevel=logging.INFO, prettyprinter = None, logfile = None, verbose=False):
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
            raise ValueError('Invalid log level: %s' % self.loglevel)  
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
                delimiter = '---------------------------------------------------------'
                pretty_handler.setFormatter(logging.Formatter(delimiter+'\n%(message)s\n'))
                # update pretty printer 
                if (prettyprinter.hasHandlers()):
                    prettyprinter.handlers.clear()
                prettyprinter.addHandler(pretty_handler)

        handler.setLevel(loglevel)
        if verbose:
            handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(name)s:\n\t%(message)s\n', datefmt='%H:%M:%S'))
        else:
            handler.setFormatter(logging.Formatter('%(message)s'))
        
        if (logger.hasHandlers()):
            logger.handlers.clear()    
        logger.addHandler(handler)
