import numpy
import os
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
handler.setFormatter(logging.Formatter(''))
prettyprinter.addHandler(handler)


#CYTHON -- MAC OS X FIX --  https://github.com/cython/cython/issues/1725
import pyximport
numpy_path = numpy.get_include()
os.environ['CFLAGS'] = "-I" + numpy_path
pyximport.install(
	language_level=3,
    pyimport=False,
    setup_args={'include_dirs': numpy.get_include()}
    )
from . import Cfourvec as Cfv



# Definition modules
from DarkNews import pdg
from DarkNews import const
from DarkNews import fourvec
from DarkNews import phase_space

# Experimental setups
from DarkNews import detector

# Monte Carlo modules
from DarkNews import MC

# Physics modules
from DarkNews import model
from DarkNews import decay_rates
from DarkNews import xsecs

# for output of MC 
from DarkNews import printer
from DarkNews import decayer
from DarkNews import geom

