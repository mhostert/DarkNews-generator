import sys 
from pathlib import Path
local_dir = Path(__file__).parent

import logging
# for debug and error handling
logger = logging.getLogger(__name__+'.logger')

# for pretty printing
prettyprinter = logging.getLogger(__name__+'.pretty_printer')
prettyprinter.setLevel(logging.INFO)
handler = logging.StreamHandler(stream=sys.stdout)
handler.setLevel(logging.INFO)
if not prettyprinter.hasHandlers():
    prettyprinter.addHandler(handler)
logger.propagate = False
prettyprinter.propagate = False

'''
    These definition modules make import of main DarkNews classes easier.
    Essentially, it allows DarkNews.XXXX instead of DarkNews.YYYY.XXXX
'''
# Definition modules
from DarkNews.GenLauncher import GenLauncher



from DarkNews import pdg
from DarkNews import const
from DarkNews import fourvec
from DarkNews import phase_space

# Experimental setups
from DarkNews import detector

# Physics modules
from DarkNews import decay_rates
from DarkNews import model

from DarkNews.model import UpscatteringProcess 
from DarkNews.model import FermionDileptonDecay 
from DarkNews.model import FermionSinglePhotonDecay 

# Monte Carlo modules
from DarkNews import MC

# for output of MC 
from DarkNews import printer
from DarkNews import geom

from DarkNews import plot_tools
