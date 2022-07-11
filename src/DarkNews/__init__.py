__version__ = "0.1.1"

import sys
from pathlib import Path

"""
    Initializing loggers
"""
import logging

# for debug and error handling
logger = logging.getLogger(__name__ + ".logger")

# for pretty printing
prettyprinter = logging.getLogger(__name__ + ".pretty_printer")
prettyprinter.setLevel(logging.INFO)
handler = logging.StreamHandler(stream=sys.stdout)
handler.setLevel(logging.INFO)
if not prettyprinter.hasHandlers():
    prettyprinter.addHandler(handler)
logger.propagate = False
prettyprinter.propagate = False


"""
    These definition modules make import of main DarkNews classes easier.
    Essentially, it allows DarkNews.XXXX instead of DarkNews.YYYY.XXXX
"""
# Definition modules
from DarkNews.GenLauncher import GenLauncher

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

from DarkNews.processes import UpscatteringProcess
from DarkNews.processes import FermionDileptonDecay
from DarkNews.processes import FermionSinglePhotonDecay

# Monte Carlo modules
from DarkNews import MC

# for output of MC
from DarkNews import printer
from DarkNews import geom

from DarkNews import plot_tools
