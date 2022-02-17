import logging
import numpy as np
import math
import os.path

# Dark Neutrino and MC stuff
import DarkNews as dn
from DarkNews.const import Q, ConfigureLogger
from DarkNews import logger, prettyprinter
from DarkNews.AssignmentParser import AssignmentParser, ParseException

class GenLauncher:

    banner = r"""|   ______           _        _   _                     |
|   |  _  \         | |      | \ | |                    |
|   | | | |__ _ _ __| | __   |  \| | _____      _____   |
|   | | | / _  | ___| |/ /   | .   |/ _ \ \ /\ / / __|  |
|   | |/ / (_| | |  |   <    | |\  |  __/\ V  V /\__ \  |
|   |___/ \__,_|_|  |_|\_\   \_| \_/\___| \_/\_/ |___/  |"""

    def __init__(self, param_file=None, **kwargs):
        '''
            Instantiate an object to make the runs, allowing the user to set the parameters' value.
            There are different ways to accomplish this:
                - file: read the file in input, parsing it looking for the different variables' values;
                - kwargs: variables' values specified as a list of keyword arguments.
            It first set the default value for each parameter, then it sets values according to the file,
            at the end it looks inside the kwargs (so kwargs overwrite file definitions).
        '''
        # set defaults
        self.mzprime = 1.25
        self.m4 = 0.140
        self.m5 = None
        self.m6 = None
        self.D_or_M = "majorana"
        self.ue4 = 0.0
        self.ue5 = 0.0
        self.ue6 = 0.0
        self.umu4 = np.sqrt(1.5e-6 * 7/4)
        self.umu5 = np.sqrt(11.5e-6)
        self.umu6 = np.sqrt(0.0)
        self.utau4 = 0
        self.utau5 = 0
        self.utau6 = 0
        self.ud4 = 1.0
        self.ud5 = 1.0
        self.ud6 = 1.0
        self.gD = 1.0
        self.epsilon = 1e-2
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
        self.decay_products="e+e-"
        self.exp = "miniboone_fhc"
        self.nopelastic = False
        self.nocoh = False
        self.noHC = False
        self.noHF = False
        self.log = "INFO"
        self.verbose = False
        self.logfile = None
        self.neval = int(1e4)
        self.nint = 20
        self.neval_warmup = int(1e3)
        self.nint_warmup = 10
        self.hepevt_events = 100
        self.pandas = True
        self.numpy = False
        self.hepevt = False
        self.hepevt_unweigh = False
        self.hepevt_events = 100
        self.sample_geometry = False
        self.summary_plots = True
        self.path = "."

        # load file if not None
        if param_file and isinstance(param_file, str):
            try:
                self._load_file(param_file)
            except FileNotFoundError:
                print(f"File '{param_file}' not found.")
                raise

        # look into kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)

    def _load_file(self, file):
        # read file
        with open(file, "r") as f:
            lines = f.readlines()
        # create parser
        parser = AssignmentParser(parameters={})
        for i, line in enumerate(lines):
            partition = line.partition("#")[0]
            if partition.strip() == "":
                continue
            try:
                parser.parse_string(partition, parseAll=True)
                parser.evaluate_stack()
            except ParseException as pe:
                print(partition, f"Failed parse (line {i+1}):", str(pe))
            except AssignmentParser.ParsingError as e:
                print(partition, f"Failed evaluation (line {i+1}):", str(e))
        # store variables
        for k, v in parser.parameters.items():
            setattr(self, k, v)

    def run(self, log="INFO", verbose=None, logfile=None, path="."):
        args = {"log": log, "verbose": verbose, "logfile": logfile, "path": path}
        for attr in args.keys():
            if args[attr] is not None:
                setattr(self, attr, args[attr])

        numeric_level = getattr(logging, log, None)
        if not isinstance(numeric_level, int):
            raise ValueError('Invalid log level: %s' % log)  
        ConfigureLogger(logger, level=numeric_level, prettyprinter=prettyprinter, verbose=verbose, logfile=logfile)

        ######################################
        # run generator
        prettyprinter.info(self.banner)

        ##########################
        # path
        path = self.path

        ##########################
        # MC evaluations and iterations
        dn.MC.NEVAL_warmup = self.neval_warmup
        dn.MC.NINT_warmup = self.nint_warmup
        dn.MC.NEVAL = self.neval
        dn.MC.NINT  = self.nint

        #########################
        # Set BSM parameters
        bsm_model = dn.model.create_model(self)


        ####################################################
        # Set the model to use

        threeplusone = (self.m4 and not self.m5 and not self.m6) 
        threeplustwo = (self.m4 and self.m5 and not self.m6) 
        threeplusthree = (self.m4 and self.m5 and self.m6)

        if threeplusone:
            upscattered_nus = [dn.pdg.neutrino4]
            outgoing_nus =[dn.pdg.nulight]

            ### NAMING 
            ## HEPEVT Event file name
            PATH_data = f'data/{self.exp}/3plus1/m4_{self.m4:.4g}_mzprime_{self.mzprime:.4g}_{self.D_or_M}/'
            PATH = f'plots/{self.exp}/3plus1/m4_{self.m4:.4g}_mzprime_{self.mzprime:.4g}_{self.D_or_M}/'
            
            # title for plots
            power = int(math.log10(self.umu4**2))-1
            title = r"$m_{4} = \,$"+str(round(self.m4,4))+r" GeV, $M_{Z^\prime} = \,$"+str(round(self.mzprime,4))+r" GeV, $|U_{D4}|^2=%1.1g$, $|U_{\mu 4}|^2=%1.1f \times 10^{-%i}$"%(self.ud4**2,self.umu4**2/10**(power),-power)
        

        elif threeplustwo:
            ## FIXING 3+2 process chain to be numu --> N5 --> N4
            upscattered_nus = [dn.pdg.neutrino5]
            outgoing_nus =[dn.pdg.neutrino4]
            # upscattered_nus = [dn.pdg.neutrino4,dn.pdg.neutrino5]
            # outgoing_nus =[dn.pdg.numu,dn.pdg.neutrino4]
        
            PATH_data = f'data/{self.exp}/3plus2/m5_{self.m5:.4g}_m4_{self.m4:.4g}_mzprime_{self.mzprime:.4g}_{bsm_model.HNLtype}/'
            PATH = f'plots/{self.exp}/3plus2/m5_{self.m5:.4g}_m4_{self.m4:.4g}_mzprime_{self.mzprime:.4g}_{bsm_model.HNLtype}/'
            title = fr"$m_{4} = \,${round(self.m4,4)}GeV, $m_{5} = \,${round(self.m5,4)}GeV, \
                        $M_{{Z^\prime}} = \,${round(self.mzprime,4)} GeV, \
                        $|U_{{D4}}|^2={dn.const.sci_notation(self.ud4**2)}$, $|U_{{\mu 4}}|^2={dn.const.sci_notation(self.umu4**2)}$, \
                        $|U_{{D5}}|^2={dn.const.sci_notation(self.ud5**2)}$, $|U_{{\mu 5}}|^2={dn.const.sci_notation(self.umu5**2)}$"   



        elif threeplusthree:
            upscattered_nus = [dn.pdg.neutrino4,dn.pdg.neutrino5,dn.pdg.neutrino6]
            outgoing_nus =[dn.pdg.nulight,dn.pdg.neutrino4,dn.pdg.neutrino5]
            
            PATH_data = f'data/{self.exp}/3plus3/m6_{self.m6:.4g}_m5_{self.m5:.4g}_m4_{self.m4:.4g}_mzprime_{self.mzprime:.4g}/'
            PATH = f'plots/{self.exp}/3plus3/m6_{self.m6:.4g}_m5_{self.m5:.4g}_m4_{self.m4:.4g}_mzprime_{self.mzprime:.4g}/'
            
            # title for plots
            title = fr"$m_{4} = \,${round(self.m4,4)}GeV, $m_{5} = \,${round(self.m5,4)}GeV, $m_{6} = \,${round(self.m6,4)}GeV,\
                        $M_{{Z^\prime}} = \,${round(self.mzprime,4)} GeV, \
                        $|U_{{D4}}|^2={dn.const.sci_notation(self.ud4**2)}$, $|U_{{\mu 4}}|^2={dn.const.sci_notation(self.umu4**2)}$, \
                        $|U_{{D5}}|^2={dn.const.sci_notation(self.ud5**2)}$, $|U_{{\mu 5}}|^2={dn.const.sci_notation(self.umu5**2)}$, \
                        $|U_{{D6}}|^2={dn.const.sci_notation(self.ud6**2)}$, $|U_{{\mu 6}}|^2={dn.const.sci_notation(self.umu6**2)}$"


        else:
            logger.error('ERROR! Mass spectrum not allowed.')
            raise ValueError 


        ####################################################
        # Choose experiment and scope of simulation
        myexp = dn.detector.Detector(self.exp)

        kwargs = {  'NO_COH': self.nocoh,
                    'NO_PELASTIC': self.nopelastic,
                    'INCLUDE_HC': ~self.noHC,
                    'INCLUDE_HF': ~self.noHF,
                    'FLAVORS': [dn.pdg.numu],
                    'UPSCATTERED_NUS': upscattered_nus,
                    'OUTGOING_NUS': outgoing_nus,
                    'DECAY_PRODUCTS': [self.decay_products],
                }

        ####################################################
        # Run MC and get events
        df_gen = dn.MC.run_MC(bsm_model, myexp, **kwargs)

        ####################################################
        # Paths
        PATH_data = os.path.join(path, PATH_data)
        PATH = os.path.join(path, PATH)
        
        df_gen.attrs['DATA_PATH'] = PATH_data
        df_gen.attrs['PLOTS_PATH'] = PATH
        df_gen.attrs['PLOTS_TITLE'] = title

        ############################################################################
        # Print events to file -- currently in data/exp/m4____mzprime____.dat 
        self.df = df_gen
        if self.numpy:
            dn.printer.print_events_to_ndarray(PATH_data, self.df)
        if self.pandas:
            dn.printer.print_events_to_pandas(PATH_data, self.df)
        if self.hepevt:
            dn.printer.print_unweighted_events_to_HEPEVT(self.df, unweigh= self.hepevt_unweigh, max_events=self.hepevt_events)
        
        return self.df