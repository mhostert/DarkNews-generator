import logging
import numpy as np
import math
import os.path

# Dark Neutrino and MC stuff
import DarkNews as dn
from DarkNews import const
from DarkNews.const import ConfigureLogger
from DarkNews import logger, prettyprinter

class GenLauncher:

    banner = r"""
#########################################################
#   ______           _        _   _                     #
#   |  _  \         | |      | \ | |                    #
#   | | | |__ _ _ __| | __   |  \| | _____      _____   #
#   | | | / _  | ___| |/ /   | .   |/ _ \ \ /\ / / __|  #
#   | |/ / (_| | |  |   <    | |\  |  __/\ V  V /\__ \  #
#   |___/ \__,_|_|  |_|\_\   \_| \_/\___| \_/\_/ |___/  #
#                                                       #
#########################################################
        """
    LOG_LEVEL = "INFO"
    VERBOSE = False
    NB_CORES = 1

    def __init__(self, **kwargs):
        self.mzprime = 1.25
        self.m4 = 0.140
        self.m5 = None
        self.m6 = None
        self.D_or_M = "majorana"
        self.ue4 = 0.0
        self.ue5 = 0.0
        self.ue6 = 0.0
        self.umu4 = math.sqrt(1.5e-6 * 7/4)
        self.umu5 = math.sqrt(11.5e-6)
        self.umu6 = math.sqrt(0.0)
        self.utau4 = 0
        self.utau5 = 0
        self.utau6 = 0
        self.ud4 = 1.0
        self.ud5 =  1.0
        self.ud6 =  1.0
        self.gD = 1.0
        self.log = "INFO"
        self.verbose = False
        self.nb_cores = 1
        self.neval = int(1e4)
        self.nint = 20
        self.neval_warmup = int(1e3)
        self.nint_warmup = 10
        self.hepevt_events = 100
        self.path = ""

    def run(self, nb_cores=None, log_level=None, verbose=None, logfile=None):
        args = {"nb_cores": nb_cores, "log_level": log_level, "verbose": verbose, "logfile": logfile}
        for attr in ["nb_cores", "log_level", "verbose", "logfile"]:
            if args[attr] is not None:
                setattr(self, attr, args[attr])

        numeric_level = getattr(logging, log_level, None)
        if not isinstance(numeric_level, int):
            raise ValueError('Invalid log level: %s' % log_level)  
        ConfigureLogger(logger, level=numeric_level, prettyprinter = prettyprinter, verbose=verbose)

        ######################################
        # run generator

        prettyprinter.info(self.banner)

        ##########################
        # path
        path = self.path
        use_default_path = True if path == "" else False

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
            logger.info(f'Theory model used: 3+1 {self.D_or_M} HNL model\n\n')
            MODEL = dn.const.THREEPLUSONE
            upscattered_nus = [dn.pdg.neutrino4]
            outgoing_nus =[dn.pdg.numu]

            ### NAMING 
            ## HEPEVT Event file name
            PATH_data = f'data/{self.exp}/3plus1/m4_{self.m4:.4g}_mzprime_{self.mzprime:.4g}_{self.D_or_M}/'
            PATH = f'plots/{self.exp}/3plus1/m4_{self.m4:.4g}_mzprime_{self.mzprime:.4g}_{self.D_or_M}/'
            
            # title for plots
            power = int(math.log10(self.umu4**2))-1
            title = r"$m_{4} = \,$"+str(round(self.m4,4))+r" GeV, $M_{Z^\prime} = \,$"+str(round(self.mzprime,4))+r" GeV, $|U_{D4}|^2=%1.1g$, $|U_{\mu 4}|^2=%1.1f \times 10^{-%i}$"%(self.ud4**2,self.umu4**2/10**(power),-power)
        

        elif threeplustwo:
            logger.info(f'Theory model used: 3+2 {self.D_or_M} HNL model\n\n')
            MODEL = dn.const.THREEPLUSTWO
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
            logger.info(f'Theory model used: 3+3 {self.D_or_M} HNL model\n\n')
            MODEL = dn.const.THREEPLUSTHREE
            upscattered_nus = [dn.pdg.neutrino4,dn.pdg.neutrino5,dn.pdg.neutrino6]
            outgoing_nus =[dn.pdg.numu,dn.pdg.neutrino4,dn.pdg.neutrino5]
            
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


        kwargs = {  'INCLUDE_COH': ~self.nocoh,
                    'INCLUDE_PELASTIC': ~self.nopelastic,
                    'INCLUDE_HC': ~self.noHC,
                    'INCLUDE_HF': ~self.noHF,
                    'FLAVORS': [dn.pdg.numu],
                    'UPSCATTERED_NUS': upscattered_nus,
                    'OUTGOING_NUS': outgoing_nus,
                }

        ####################################################
        # Run MC and get events
        df_gen = dn.MC.run_MC(bsm_model, myexp, **kwargs)


        ####################################################
        # create directory if it does not exist: skip this if you're in a grid run
        if not use_default_path:
            logger.info(path)
            PATH_data = os.path.join(path, "data", os.path.split(PATH_data.rstrip('/'))[-1])
            PATH      = os.path.join(path, "plots", os.path.split(PATH.rstrip('/'))[-1])
            for p in [os.path.join(path, "data"), os.path.join(path, "plots")]:
                if not os.path.exists(p):
                    os.mkdir(p)
        
        df_gen['DATA_PATH'] = PATH_data
        df_gen['PLOTS_PATH'] = PATH
        df_gen['PLOTS_TITLE'] = title

        ############################################################################
        # Print events to file -- currently in data/exp/m4____mzprime____.dat 
        if self.numpy:
            dn.printer.print_events_to_ndarray(PATH_data, df_gen, bsm_model)
        if self.pandas:
            dn.printer.print_events_to_pandas(PATH_data, df_gen, bsm_model)
        if self.hepevt:
            dn.printer.print_unweighted_events_to_HEPEVT(df_gen, bsm_model, unweigh= self.hepevt_unweigh, TOT_EVENTS=self.hepevt_events)
        logger.info(f"Outputs saved in {PATH_data}")