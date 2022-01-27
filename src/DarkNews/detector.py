import numpy as np
from scipy import interpolate
from particle import literals as lp
from pathlib import Path
import os.path
import json
local_dir = Path(__file__).parent

from . import logger, prettyprinter
from . import pdg
from . import geom
from . import const
from .nuclear_tools import NuclearTarget


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
        with open(file_path, 'r') as f:
            read_file = json.load(f)
        try:
            self.NAME            = read_file['name']
            self.FLUXFILE        = read_file['fluxfile']
            self.FLUX_NORM       = read_file['flux_norm']
            self.ERANGE          = read_file['erange']

            # Detector targets
            self.NUCLEAR_TARGETS = [NuclearTarget(target) for target in read_file['nuclear_targets']]
            self.FIDUCIAL_MASS   = read_file['fiducial_mass']
            self.POTS            = read_file['POTs']
            
            # total number of targets
            self.NUMBER_OF_TARGETS = {}
            for fid_mass_fraction, target in zip(read_file['fiducial_mass_fraction_per_target'], self.NUCLEAR_TARGETS):
                self.NUMBER_OF_TARGETS[f'{target.name}'] = self.FIDUCIAL_MASS*fid_mass_fraction*const.t_to_GeV/(target.mass)

            # load neutrino fluxes
            _enu, *_fluxes = np.genfromtxt(f'{local_dir}/{self.FLUXFILE}',unpack=True)
            self.FLUX_FUNCTIONS = 6*[[]]
            for i in range(len(_fluxes)):
                self.FLUX_FUNCTIONS[i] = interpolate.interp1d(_enu, _fluxes[i]*self.FLUX_NORM, fill_value=0.0, bounds_error=False)


            prettyprinter.info(f"Experiment: \n\t{self.NAME}\n\tfluxfile loaded: {self.FLUXFILE}\n\tPOT: {self.POTS}\n\tnuclear targets: {[n.name for n in self.NUCLEAR_TARGETS]}\n\tfiducial mass: {[self.FIDUCIAL_MASS*frac for frac in read_file['fiducial_mass_fraction_per_target']]} tonnes")

        except FileNotFoundError:
            raise FileNotFoundError("The experiment configuration file '{}.json' does not exist.".format(experiment_name.lower()))
        except KeyError as err:
            if err.args[0] in ["name", "fluxfile", "flux_norm", "erange", "nuclear_targets", "fiducial_mass", "fiducial_mass_fraction_per_target", "POTs"]:
                # check that the error comes from reading the file and not elsewhere
                raise KeyError("No field '{}' specified in the the experiment configuration file '{}.json'.".format(err.args[0], experiment_name.lower()))
            raise

    def neutrino_flux(self, projectile):
        _flux_index = pdg.get_doublet(projectile) + 3*pdg.is_antiparticle(projectile)
        return self.FLUX_FUNCTIONS[_flux_index]

    def set_geometry(self):
        if 'microboone' in self.NAME.lower():
            self.place_scatters = geom.microboone_geometry
        elif 'miniboone' in self.NAME.lower():
            self.place_scatters = geom.miniboone_geometry
        else:
            self.place_scatters = geom.point_geometry