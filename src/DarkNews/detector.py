import importlib
import numpy as np
from scipy import interpolate
from pathlib import Path
import os.path
import json
local_dir = Path(__file__).parent

from DarkNews import logger

from .const import *
from . import pdg
from . import nuclear_tools


class NuclearTarget:
    """ for scattering on a nuclear target 
    Args:
        params      ??
        name: name of the target. Can be either "electron" or the name of the element (e.g. C12, Pb208).
    """

    def __init__(self, name):
        self.name = name
        
        #####################################        
        # free electron
        if name == "electron":
            self.is_hadron  = False
            self.is_nucleus = False
            self.is_proton  = False
            self.is_neutron = False            
            self.is_free_nucleon = False

            self.mass=m_e
            self.charge=1
            self.Z=0
            self.N=0
            self.A=0
            self.pdgid=11
        #####################################
        # hadronic *nuclear* target
        else:

            # Using the global dictionary of elements defined in nuclear_tools
            # Set all items as attributes of the class
            for k, v in nuclear_tools.elements_dic[name].items():
                setattr(self, k, v)
            self.mass = self.nuclear_mass
            self.charge = self.Z 

            self.is_hadron  = True
            self.is_nucleus  = (self.A > 1)
            self.is_proton  = (self.Z == 1 and self.A == 1)
            self.is_neutron = (self.N == 1 and self.A == 1)

            self.is_nucleon = (self.is_neutron or self.is_proton)
            self.is_free_nucleon = (self.A == 1)
            self.is_bound_nucleon = False

            if self.is_nucleus:
                # no hyperons and always ground state
                self.pdgid = int(f'100{self.Z:03d}{self.A:03d}0')
            elif self.is_neutron:
                self.pdgid = pdg.neutron.pdgid
            elif self.is_proton:
                self.pdgid = pdg.proton.pdgid
            else:
                logger.error(f"Error. Could not find the PDG ID of {self.name}.")
                raise ValueError

            if self.is_neutron and self.is_free_nucleon:
                logger.error(f"Error. Target {self.name} is a free neutron.")
                raise ValueError
        
            self.tau3 = self.Z*2 - 1  # isospin +1 proton / -1 neutron 
            

            nuclear_tools.assign_form_factors(self)


    # hadronic *constituent* target
    def get_constituent_nucleon(self, name):
        return self.BoundNucleon(self, name)

 
    class BoundNucleon():    
        """ for scattering on bound nucleon in the nuclear target 
        
        Inner Class

        Args:
            nucleus: nucleus of which this particle is bound into (always the outer class)
            name: 'proton' or 'neutron'
        """
        def __init__(self, nucleus, name):
           
            # if not nucleus.is_nucleus:
            #     print(f"Error! Sattering target {nucleus.name} is not a nucleus with bound {name}.")
            #     raise ValueError 

            self.nucleus = nucleus
            self.A = int(name=='proton' or name == 'neutron')
            self.Z = int(name=='proton')
            self.N = int(name=='neutron')
            self.charge = self.Z
            self.mass = m_proton
            self.name = f'{name}_in_{nucleus.name}'

            self.is_hadron = True
            self.is_nucleus = False
            self.is_neutron = (self.N == 1 and self.A == 1)
            self.is_proton = (self.Z == 1 and self.A == 1)

            self.is_nucleon = (self.is_neutron or self.is_proton)
            self.is_false_nucleon = False
            self.is_bound_nucleon = True

            if self.is_neutron:
                self.pdgid = pdg.neutron.pdgid
            elif self.is_proton:
                self.pdgid = pdg.proton.pdgid
            else:
                logger.error(f"Error. Could not find the PDG ID of {self.name}.")
                raise ValueError


            self.tau3 = self.Z*2 - 1  # isospin +1 proton / -1 neutron 

            nuclear_tools.assign_form_factors(self)



class Detector():
    """ 

    Detector is a collection of necessary variables for cross-section and gamma
    calculations, .e.g energy range, target type, weight, and POTs for exposure.
    It provides the `get_flux_func` which creates a spline interpolation of the
    input neutrino flux.

    Args:
        exp_module (name):  name of experiment corresponding to a python module
                            in the path `dark_news/detector/exp_module.py`.
                            the module should contain parameters specific for
                            that experiment if it is user defined.
    """
    PATH_CONFIG_FILES = os.path.join(local_dir, "detectors")

    def __init__(self, experiment_name):
        file_path = os.path.join(self.PATH_CONFIG_FILES, experiment_name.lower() + ".json")
        try:
            with open(file_path, 'r') as f:
                read_file = json.load(f)
                self.NAME            = read_file['name']
                self.FLUXFILE        = read_file['fluxfile']
                self.FLUX_NORM       = read_file['flux_norm']
                self.ERANGE          = read_file['erange']
                #self.EMAX           = read_file['erange[1]']
                # Detector targets
                self.NUCLEAR_TARGETS = [NuclearTarget(target) for target in read_file['nuclear_targets']]
                self.FIDUCIAL_MASS   = read_file['fiducial_mass']
                self.POTS            = read_file['POTs']
        except FileNotFoundError:
            raise FileNotFoundError("The experiment configuration file '{}.json' does not exist.".format(experiment_name.lower()))
        except KeyError as err:
            raise KeyError("No field '{}' specified in the the experiment configuration file '{}.json'.".format(err.args[0], experiment_name.lower()))

        # total number of targets
        self.NUMBER_OF_TARGETS = {}
        for fid_mass_fraction, target in zip(read_file['fiducial_mass_fraction_per_target'], self.NUCLEAR_TARGETS):
            self.NUMBER_OF_TARGETS[f'{target.name}'] = self.FIDUCIAL_MASS*fid_mass_fraction/target.A * NAvo 

    # this one is too specific, need to make it more generic
    # and force the user to provide fluxes normalized in the right format!
    def get_flux_func(self, flavor = pdg.numu):
        """ Return flux interpolating function using scipy interp1d

        Args:
            flavour (pdg.flavour) : neutrino flavour for required flux

        """
        data = np.genfromtxt(f'{local_dir}/{self.FLUXFILE}',unpack=True)
        E = data[0]
        if flavor==pdg.numu:
            nf = data[2]
        elif flavor==pdg.numubar:
            nf = data[5]
        elif flavor==pdg.nue:
            nf = data[1]
        elif flavor==pdg.nuebar:
            nf = data[4]
        else:
            logger.error("ERROR! Neutrino flavor {flavor.name} not supported.")
        # raise ValueError(f"Unknown file \"{self.FLUXFILE}\"")

        flux = interpolate.interp1d(E, nf*self.FLUX_NORM, fill_value=0.0, bounds_error=False)
        return flux
