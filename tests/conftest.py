import pytest
import numpy as np
from particle import literals as lp
from DarkNews import const
from DarkNews.GenLauncher import GenLauncher

MODEL_KWARGS = {'HNLtype': 'dirac', 'UD4': 1.0, 'alphaD': 0.25, 'Umu4': np.sqrt(9e-7), 'epsilon': np.sqrt(2e-10/const.alphaQED)}


# generate a specific set of events to be tested in a session
@pytest.fixture(scope='session')
def SM_gen():
    gen = GenLauncher(gD=0.0, Umu4=1e-3, epsilon=0.0, m4=0.01, loglevel='ERROR', neval=1000, seed=42)
    return gen.run()


@pytest.fixture(scope='session')
def light_DP_gen_all_outputs():

    gen = GenLauncher(mzprime=0.03, m4=0.420, neval=1000, exp="miniboone_fhc", loglevel='ERROR',
                        parquet=True, numpy=True, hepevt=True, sparse=True, print_to_float32=True, **MODEL_KWARGS)

    return gen.run()


@pytest.fixture(scope='session')
def gen_simplest_benchmarks():

    gen = GenLauncher(mzprime=0.03, m4=0.420, neval=1000, exp="miniboone_fhc", loglevel='ERROR', seed=42, **MODEL_KWARGS)
    df_light = gen.run()

    gen = GenLauncher(mzprime=1.25, m4=0.150, neval=1000, exp="miniboone_fhc", loglevel='ERROR', seed=42, **MODEL_KWARGS)
    df_heavy = gen.run()

    gen = GenLauncher(mu_tr_mu4=1e-6, m4=0.150, epsilon=0.0, gD=0.0, Umu4=0.0,
                    neval=1000, exp="miniboone_fhc", loglevel='ERROR', seed=42)
    df_TMM = gen.run()

    return df_light, df_heavy, df_TMM


@pytest.fixture(scope='session')
def gen_other_final_states():

    gen = GenLauncher(mzprime=0.3, m4=0.5, decay_product='mu+mu-', neval=1000, exp="miniboone_fhc", loglevel='ERROR', seed=42, **MODEL_KWARGS)
    df_light = gen.run()

    gen = GenLauncher(mzprime=1.25, m4=0.5, decay_product='mu+mu-', neval=1000, exp="miniboone_fhc", loglevel='ERROR', seed=42, **MODEL_KWARGS)
    df_heavy = gen.run()

    gen = GenLauncher(mu_tr_mu4=1e-6, m4=0.5, epsilon=0.0, gD=0.0, decay_product='mu+mu-', Umu4=0.0,
                        neval=1000, HNLtype="dirac", exp="miniboone_fhc", loglevel='ERROR', seed=42)
    df_TMM_mumu = gen.run()

    gen = GenLauncher(mu_tr_mu4=1e-6, m4=0.5, epsilon=0.0, gD=0.0, decay_product='photon', Umu4=0.0,
                        neval=1000, HNLtype="dirac", exp="miniboone_fhc", loglevel='ERROR', seed=42)
    df_TMM_photon = gen.run()

    return df_light, df_heavy, df_TMM_mumu, df_TMM_photon



