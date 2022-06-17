import numpy
import os

#CYTHON -- MAC OS X FIX -- https://github.com/cython/cython/issues/1725
import pyximport
numpy_path = numpy.get_include()
os.environ['CFLAGS'] = "-I" + numpy_path
pyximport.install(
	language_level=3,
    pyimport=False,
    setup_args={'include_dirs': numpy.get_include()}
    )

from pathlib import Path
local_dir = Path(__file__).parent

import logging

# for debug and error handling
toy_logger = logging.getLogger(__name__)
toy_logger.setLevel(logging.INFO)


# Handling four vectors
from ToyAnalysis import fourvec # python only
from ToyAnalysis import Cfourvec as Cfv # cython

# Analysis and plotting modules 
from ToyAnalysis import analysis
from ToyAnalysis import analysis_decay
from ToyAnalysis import cuts
from ToyAnalysis import plot_tools


PATH_TO_DATA_RELEASE = f'{local_dir}/data/'