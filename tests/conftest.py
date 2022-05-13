import pytest
import numpy as np
from particle import literals as lp
from DarkNews import const
from DarkNews.GenLauncher import GenLauncher



# generate a specific set of events to be tested in a session
@pytest.fixture(scope='session')
def SM_gen():
    gen = GenLauncher(gD=0.0, Umu4=1e-3, epsilon=0.0, m4=0.001)
    return gen.run()


@pytest.fixture(scope='session')
def light_DP_gen():
    # Test 1 -- saving files correctly and checking they are all the same 
    ud4_def = 1.0
    alphaD = 0.25
    gD_def = np.sqrt(alphaD*4*np.pi)
    umu4_def = np.sqrt(9e-7)
    ud4 = 1.
    epsilon_def = np.sqrt(2e-10/const.alphaQED)
    gen = GenLauncher(mzprime=0.03, m4=0.420, epsilon=epsilon_def, Umu4=umu4_def, UD4=ud4_def, gD=gD_def, 
                        neval=1000, HNLtype="dirac", exp="miniboone_fhc", loglevel='INFO',
                        parquet=True, numpy=True, hepevt=True, sparse=True, print_to_float32=True)

    return gen.run(loglevel="INFO")