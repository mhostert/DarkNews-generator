__version__ = "0.4.8"

import sys

"""
    Initializing loggers
"""
import logging
import logging.handlers


def configure_loggers(loglevel="WARNING", logfile=None, verbose=False):
    """
    Configure the DarkNews loggers:

    1) logger (logging.Logger): main DarkNews logger to be configured. It handles all debug, info, warning, and error messages

    2) prettyprinter (logging.Logger): for pretty printing INFO messages. It is used to print the progress of the generation.

    Args:

        loglevel (str, optional): what logging level to use.
                                Can be logging.(DEBUG, INFO, WARNING, or ERROR). Defaults to logging.INFO.

        logfile (str, optional): path to file where to log the output. Defaults to None.

        verbose (bool, optional): If true, keep date and time in the logger format. Defaults to False.

    Raises:
        ValueError: _description_
    """

    # Access or create the loggers
    logger = logging.getLogger("logger." + __name__)
    prettyprinter = logging.getLogger("prettyprinter." + __name__)

    loglevel = loglevel.upper()
    _numeric_level = getattr(logging, loglevel, None)
    if not isinstance(_numeric_level, int):
        raise ValueError("Invalid log level: %s" % loglevel)

    logger.setLevel(_numeric_level)
    prettyprinter.setLevel(_numeric_level)

    # Create handlers
    if logfile is not None:
        # log to files with max 1 MB with up to 4 files of backup
        handler = logging.handlers.RotatingFileHandler(f"{logfile}", maxBytes=1000000, backupCount=4)
    else:
        # stdout only
        handler = logging.StreamHandler(stream=sys.stdout)

        delimiter = "---------------------------------------------------------"
        pretty_formatter = logging.Formatter(delimiter + "\n%(message)s\n")

        pretty_handler = logging.StreamHandler(stream=sys.stdout)
        pretty_handler.setFormatter(pretty_formatter)
        pretty_handler.setLevel(_numeric_level)

        # update pretty printer
        if prettyprinter.hasHandlers():
            prettyprinter.handlers.clear()
        prettyprinter.addHandler(pretty_handler)

    if verbose:
        main_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s:\n\t%(message)s\n", datefmt="%H:%M:%S")
    else:
        main_formatter = logging.Formatter("%(message)s")

    handler.setFormatter(main_formatter)
    handler.setLevel(_numeric_level)
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.addHandler(handler)


configure_loggers()

"""
    Optional imports
"""

# Check if user has pyarrow installed -- if not, no parquet output is available
try:
    import pyarrow.parquet as pq
    import pyarrow as pa

    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False

# Check if user has pyhepmc3 installed -- if not, no HepMC output is available
try:
    import pyhepmc as hep

    HAS_PYHEPMC3 = True
except ImportError:
    HAS_PYHEPMC3 = False


"""
    Making it easier to import modules
"""
from DarkNews import pdg
from DarkNews import const
from DarkNews import fourvec
from DarkNews import phase_space
from DarkNews import parsing_tools

# Experimental setups
from DarkNews import detector
from DarkNews import nuclear_tools

# Physics modules
from DarkNews import decay_rates
from DarkNews import processes
from DarkNews import model

# Monte Carlo modules
from DarkNews import MC

# for output of MC
from DarkNews import printer
from DarkNews import geom

from DarkNews import plot_tools


"""
    And now making it easier to import the main DarkNews classes.
    It allows DarkNews.XXXX instead of DarkNews.YYYY.XXXX
"""
# Definition modules
from DarkNews.GenLauncher import GenLauncher
from DarkNews.AssignmentParser import AssignmentParser
from DarkNews.processes import UpscatteringProcess
from DarkNews.processes import FermionDileptonDecay
from DarkNews.processes import FermionSinglePhotonDecay
from DarkNews.nuclear_tools import NuclearTarget
from DarkNews.detector import Detector
from DarkNews.geom import Chisel
