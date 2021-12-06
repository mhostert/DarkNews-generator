import importlib
import numpy as np
from scipy import interpolate
from pathlib import Path
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
            self.pdgid = 11
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


    def __init__(self, exp_module):
        try: # relative module
            det = importlib.import_module('.' + exp_module, 'DarkNews.detectors')
        except ModuleNotFoundError:
            raise ImportError(f"Cannot import module \"{exp_module}\"")
        #    pass
        #try: # absolute module
        #    det = importlib.import_module(exp_module)
        #except ModuleNotFoundError:

        self.NAME = det.name
        self.FLUXFILE  = det.fluxfile
        self.FLUX_NORM = det.flux_norm
        self.ERANGE = det.erange
        #self.EMAX = det.erange[1]

        # Detector targets
        self.NUCLEAR_TARGETS = [NuclearTarget(target) for target in det.nuclear_targets]
                
        # total number of targets
        self.NUMBER_OF_TARGETS = {}
        for fid_mass, target in zip(det.fiducial_mass_per_target, self.NUCLEAR_TARGETS):
            self.NUMBER_OF_TARGETS[f'{target.name}'] = fid_mass/target.A * NAvo 

        self.POTS = det.POTs

    # this one is too specific, need to make it more generic
    # and force the user to provide fluxes normalized in the right format!
    def get_flux_func(self, flavor = pdg.numu):
        """ Return flux interpolating function using scipy interp1d

        Args:
            flavour (pdg.flavour) : neutrino flavour for required flux

        TODO: unifying fluxes inputs
        """
        if (self.FLUXFILE == "fluxes/MiniBooNE_nu_mode_flux.dat" and (flavor==pdg.numu or flavor==pdg.numubar)):
            Elo, Ehi, numu, numub, nue, nueb = np.loadtxt(f'{local_dir}/{self.FLUXFILE}', unpack=True)
            E = (Ehi+Elo)/2.0
            if flavor==pdg.numu:
                nf = numu
            if flavor==pdg.numubar:
                nf = numub
        elif (self.FLUXFILE == "fluxes/MINERVA_ME_numu_flux.dat" and flavor==pdg.numu):
            E, nf = np.loadtxt(f'{local_dir}/{self.FLUXFILE}', unpack=True)
        elif (self.FLUXFILE == "fluxes/MINERVA_LE_numu_flux.dat" and flavor==pdg.numu):
            E, nf = np.loadtxt(f'{local_dir}/{self.FLUXFILE}', unpack=True)
        elif (self.FLUXFILE == "fluxes/CHARMII.dat" and flavor==pdg.numu):
            E, nf = np.loadtxt(f'{local_dir}/{self.FLUXFILE}', unpack=True)
        elif (self.FLUXFILE=="fluxes/T2Kflux2016/t2kflux_2016_nd280_minus250kA.txt" or 
                self.FLUXFILE=="fluxes/T2Kflux2016/t2kflux_2016_nd280_plus250kA.txt"):
            data = np.genfromtxt(f'{local_dir}/{self.FLUXFILE}',unpack=True,skip_header=3)
            E = (data[1]+data[2])/2
            if flavor==pdg.numu:
                nf = data[3]
            elif flavor==pdg.numubar:
                nf = data[4]
            elif flavor==pdg.nue:
                nf = data[5]
            elif flavor==pdg.nuebar:
                nf = data[6]
        else:
            raise ValueError(f"Unknown file \"{self.FLUXFILE}\"")

        flux = interpolate.interp1d(E, nf*self.FLUX_NORM, fill_value=0.0, bounds_error=False)
        return flux