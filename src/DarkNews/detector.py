import numpy as np
from scipy import interpolate
import os.path
import os

try:
    from importlib.resources import files
except ImportError:
    from importlib_resources import files

import logging

logger = logging.getLogger("logger." + __name__)
prettyprinter = logging.getLogger("prettyprinter." + __name__)

from DarkNews import pdg
from DarkNews import geom
from DarkNews import const
from DarkNews.nuclear_tools import NuclearTarget
from DarkNews.AssignmentParser import AssignmentParser


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
        "sbnd": "sbnd.txt",
        "sbnd_dirt": "sbnd_dirt.txt",
        "sbnd_dirt_cone": "sbnd_dirt_cone.txt",
        "microboone": "microboone.txt",
        "microboone_tpc": "microboone_tpc.txt",
        "microboone_dirt": "microboone_dirt.txt",
        "minerva_le_fhc": "minerva_le_fhc.txt",
        "minerva_me_fhc": "minerva_me_fhc.txt",
        "minerva_me_rhc": "minerva_me_fhc.txt",
        "miniboone_fhc": "miniboone_fhc.txt",
        "miniboone_fhc_dirt": "miniboone_fhc_dirt.txt",
        "icarus": "icarus.txt",
        "icarus_dirt": "icarus_dirt.txt",
        "miniboone_rhc": "miniboone_rhc.txt",
        "miniboone_rhc_dirt": "miniboone_rhc_dirt.txt",
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
                with files(DET_MODULE).joinpath(self.KEYWORDS[experiment]).open(**kwargs) as f:
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
                file = files("DarkNews.include.fluxes").joinpath(self.FLUXFILE).open()
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

    # For experiment comparison: experiments are equivalent if main defintions are the same
    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return (
            (self.NAME == other.NAME)
            and (self.FLUXFILE == other.FLUXFILE)
            and (self.FLUX_NORM == other.FLUX_NORM)
            and (self.ERANGE == other.ERANGE)
            and (self.NUCLEAR_TARGETS == other.NUCLEAR_TARGETS)
            and (self.POTS == other.POTS)
            and (self.FIDUCIAL_MASS_PER_TARGET == other.FIDUCIAL_MASS_PER_TARGET)
        )

    def neutrino_flux(self, projectile):
        _flux_index = pdg.get_doublet(projectile) + 3 * pdg.is_antiparticle(projectile)
        return self.FLUX_FUNCTIONS[_flux_index]

    def set_geometry(self):
        geometries = {}
        geometries["microboone_dirt"] = geom.microboone_dirt_geometry
        geometries["sbnd_dirt"] = geom.sbnd_dirt_geometry
        geometries["sbnd_dirt_cone"] = geom.sbnd_dirt_cone_geometry
        geometries["icarus_dirt"] = geom.icarus_dirt_geometry
        geometries["miniboone_fhc_dirt"] = geom.miniboone_dirt_geometry
        geometries["miniboone_rhc_dirt"] = geom.miniboone_dirt_geometry
        geometries["microboone"] = geom.microboone_geometry
        geometries["microboone_tpc"] = geom.microboone_tpc_geometry
        geometries["sbnd"] = geom.sbnd_geometry
        geometries["icarus"] = geom.icarus_geometry
        geometries["miniboone_fhc"] = geom.miniboone_geometry
        geometries["miniboone_rhc"] = geom.miniboone_geometry

        if self.NAME.lower() in geometries.keys():
            self.place_scatters = geometries[self.NAME.lower()]
        else:
            logger.info(f"Experimental geometry for {self.NAME} not implemented, assuming scattering at (0,0,0,0)")
            self.place_scatters = geom.point_geometry

    def __str__(self):
        return self.NAME.replace(" ", "_").lower()
