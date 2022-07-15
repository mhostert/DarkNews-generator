import numpy as np
from scipy import interpolate
from pathlib import Path
import os.path
import os
try:
    import importlib.resources as resources
except ImportError:
    # Try backported to PY<37 `importlib_resources`.
    import importlib_resources as resources


from . import logger, prettyprinter
from . import pdg
from . import geom
from . import const
from .nuclear_tools import NuclearTarget
from .AssignmentParser import AssignmentParser


class Detector:
    """
    Detector is a collection of necessary variables for cross-section and event rate
    calculations, e.g., energy range, target type, weight, and POTs for exposure.

    It provides the `get_flux_func` which creates a spline interpolation of the
    input neutrino flux.

    Args:
        exp_module (name):  name of experiment corresponding to a .txt file
                            in the path `DarkNews/include/detector/exp_module.txt`.
                            the file should contain parameters specific for
                            that experiment if it is user defined.

    Raises:
        FileNotFoundError: if no detector file was found
        KeyError: if a required field is not specified in the detector file

    """
    KEYWORDS = {
        "dune_nd_fhc": "dune_nd_fhc.txt",
        "dune_nd_rhc": "dune_nd_rhc.txt",
        "microboone": "microboone.txt",
        "minerva_le_fhc": "minerva_le_fhc.txt",
        "minerva_me_fhc": "minerva_me_fhc.txt",
        "minerva_me_rhc": "minerva_me_fhc.txt",
        "miniboone_fhc": "miniboone_fhc.txt",
        "miniboone_rhc": "miniboone_rhc.txt",
        "minos_le_fhc": "minos_le_fhc.txt",
        "nd280_fhc": "nd280_fhc.txt",
        "nova_le_fhc": "nova_le_fhc.txt",
        "fasernu": "fasernu.txt",
        "nutev_fhc": "nutev_fhc.txt",
    }

    def __init__(self, experiment):
        parser = AssignmentParser({})
        DET_MODULE = "DarkNews.include.detectors"
        kwargs = {"encoding": "utf8"}
        try:
            # experiment is initially interpreted as a path to a local file
            experiment_file = experiment
            parser.parse_file(file=experiment_file, comments="#")
        except (OSError, IOError, FileNotFoundError) as err:
            # if no file is found, then it is interpreted as a keyword for a pre-defined experiment
            if experiment in self.KEYWORDS:
                with resources.open_text(DET_MODULE, self.KEYWORDS[experiment], **kwargs) as f:
                    parser.parse_file(file=f, comments="#")
            else:
                raise err

        params = parser.parameters
        try:
            self.NAME = params["name"]
            self.FLUXFILE = params["fluxfile"]
            self.FLUX_NORM = params["flux_norm"]
            self.ERANGE = params["erange"]

            # Detector targets
            self.NUCLEAR_TARGETS = [NuclearTarget(target) for target in params["nuclear_targets"]]
            self.POTS = params["POTs"]
            self.FIDUCIAL_MASS_PER_TARGET = params["fiducial_mass_per_target"]

        except KeyError as err:
            if err.args[0] in [
                "name",
                "fluxfile",
                "flux_norm",
                "erange",
                "nuclear_targets",
                "fiducial_mass_per_target",
                "POTs",
            ]:
                # check that the error comes from reading the file and not elsewhere
                raise KeyError(f"No field '{err.args[0]}' specified in the the experiment configuration file '{experiment}'.".format(err.args[0], experiment))
            raise err

        # total number of targets
        self.NUMBER_OF_TARGETS = {}
        for fid_mass, target in zip(self.FIDUCIAL_MASS_PER_TARGET, self.NUCLEAR_TARGETS):
            self.NUMBER_OF_TARGETS[f"{target.name}"] = fid_mass * const.t_to_GeV / (target.mass)

        # load neutrino fluxes: first try with path relative to experiment file, if error try with path from original config files
        try:
            exp_dir = os.path.dirname(experiment_file)
            _enu, *_fluxes = np.genfromtxt(f"{exp_dir}/{self.FLUXFILE}", unpack=True)
        except (OSError, FileNotFoundError, TypeError):
            try:
                file = resources.open_text("DarkNews.include.fluxes", self.FLUXFILE)
                _enu, *_fluxes = np.genfromtxt(file, unpack=True)
            except FileNotFoundError:
                raise FileNotFoundError(f"Fluxes file {self.FLUXFILE} not found in current experiment file path nor in config file path.")
            
        self.FLUX_FUNCTIONS = 6 * [[]]
        for i in range(len(_fluxes)):
            self.FLUX_FUNCTIONS[i] = interpolate.interp1d(_enu, _fluxes[i] * self.FLUX_NORM, fill_value=0.0, bounds_error=False)

        prettyprinter.info(
            f"""Experiment:
\t{self.NAME}
\tfluxfile loaded: {self.FLUXFILE}
\tPOT: {self.POTS}
\tnuclear targets: {[n.name for n in self.NUCLEAR_TARGETS]}
\tfiducial mass: {self.FIDUCIAL_MASS_PER_TARGET} tonnes"""
        )

    def neutrino_flux(self, projectile):
        _flux_index = pdg.get_doublet(projectile) + 3 * pdg.is_antiparticle(projectile)
        return self.FLUX_FUNCTIONS[_flux_index]

    def set_geometry(self):
        if "microboone" in self.NAME.lower():
            self.place_scatters = geom.microboone_geometry
        elif "miniboone" in self.NAME.lower():
            self.place_scatters = geom.miniboone_geometry
        else:
            self.place_scatters = geom.point_geometry

    def __str__(self):
        return self.NAME.replace(" ", "_").lower()