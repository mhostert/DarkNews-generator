import os
import pytest
import random
import numpy as np
import pandas as pd
from particle import literals as lp
from DarkNews import const
from DarkNews.GenLauncher import GenLauncher

MODEL_KWARGS = {"HNLtype": "dirac", "UD4": 1.0, "alphaD": 0.25, "Umu4": np.sqrt(9e-7), 
                "epsilon": np.sqrt(2e-10 / const.alphaQED)}

GENERIC_MODEL_KWARGS = {"HNLtype": "dirac", "deV": 1, "duV": 1, "ddV": 1,
                    "d_mu4": 1, "d_mu5": 1, "d_mu6": 1,
                    "d_45": 1, "d_56": 1, "d_46": 1}

@pytest.fixture(scope="session")
def set_seeds():
    seed = 42
    random.seed(seed)
    # os.environ('PYTHONHASHSEED') = str(seed)


# generate a specific set of events to be tested in a session
@pytest.fixture(scope="session")
def SM_gen():
    gen = GenLauncher(gD=0.0, Umu4=1e-3, epsilon=0.0, m4=0.01, loglevel="ERROR", neval=1000, seed=42)
    return gen.run()


# # use command line and make summary plots
# @pytest.fixture(scope='session')
# def gen_SM_from_script():
#     os.system("dn_gen --gD=0.0 --Umu4=1e-3 --epsilon=0.0 --m4=0.01 --loglevel='ERROR' --neval=1000 --seed=42 --path=./test_generation --make_summary_plots")
#     df_p = pd.read_pickle('test_generation/pandas_df.pckl')
#     return df_p


@pytest.fixture(scope="session")
def light_DP_gen_all_outputs():

    gen = GenLauncher(
        mzprime=0.03,
        m4=0.420,
        neval=1000,
        exp="miniboone_fhc",
        loglevel="ERROR",
        seed=42,
        parquet=True,
        numpy=True,
        hepevt=True,
        hepevt_legacy=True,
        hepmc2=True,
        hepmc3=True,
        **MODEL_KWARGS
    )
    return gen.run()


@pytest.fixture(scope="session")
def generic_model_gen():

    gen = GenLauncher(
        mzprime=0.03,
        m4=0.100,
        m5=0.200,
        m6=0.300,
        neval=1000,
        exp="miniboone_fhc",
        loglevel="ERROR",
        seed=42,
        parquet=True,
        numpy=True,
        hepevt=True,
        hepevt_legacy=True,
        hepmc2=True,
        hepmc3=True,
        hep_unweight=True,
        hep_unweight_events=100,
        **GENERIC_MODEL_KWARGS
    )
    return gen.run()



@pytest.fixture(scope="session")
def light_DP_gen_all_outputs_sparse():

    gen = GenLauncher(
        mzprime=0.03,
        m4=0.425,
        neval=1000,
        exp="miniboone_fhc",
        loglevel="ERROR",
        seed=42,
        parquet=True,
        numpy=True,
        hepevt=True,
        hepevt_legacy=True,
        hepmc2=True,
        hepmc3=True,
        sparse=True,
        print_to_float32=True,
        **MODEL_KWARGS
    )
    return gen.run()


@pytest.fixture(scope="session")
def gen_simplest_benchmarks():

    gen = GenLauncher(mzprime=0.03, m4=0.420, neval=1000, exp="miniboone_fhc", loglevel="ERROR", seed=42, **MODEL_KWARGS)
    df_light = gen.run()

    gen = GenLauncher(mzprime=1.25, m4=0.150, neval=1000, exp="miniboone_fhc", loglevel="ERROR", seed=42, **MODEL_KWARGS)
    df_heavy = gen.run()

    gen = GenLauncher(mu_tr_mu4=1e-6, m4=0.150, epsilon=0.0, gD=0.0, Umu4=0.0, neval=1000, exp="miniboone_fhc", loglevel="ERROR", seed=42)
    df_TMM = gen.run()

    return df_light, df_heavy, df_TMM


@pytest.fixture(scope="session")
def gen_other_final_states():

    gen = GenLauncher(mzprime=0.3, m4=0.5, decay_product="mu+mu-", neval=1000, exp="miniboone_fhc", loglevel="ERROR", seed=42, **MODEL_KWARGS)
    df_light = gen.run()

    gen = GenLauncher(mzprime=1.25, m4=0.5, decay_product="mu+mu-", neval=1000, exp="miniboone_fhc", loglevel="ERROR", seed=42, **MODEL_KWARGS)
    df_heavy = gen.run()

    gen = GenLauncher(
        mu_tr_mu4=1e-6,
        m4=0.5,
        epsilon=0.0,
        gD=0.0,
        decay_product="mu+mu-",
        Umu4=0.0,
        neval=1000,
        HNLtype="dirac",
        exp="miniboone_fhc",
        loglevel="ERROR",
        seed=42,
    )
    df_TMM_mumu = gen.run()

    gen = GenLauncher(
        mu_tr_mu4=1e-6,
        m4=0.5,
        epsilon=0.0,
        gD=0.0,
        decay_product="photon",
        Umu4=0.0,
        neval=1000,
        HNLtype="dirac",
        exp="miniboone_fhc",
        loglevel="ERROR",
        seed=42,
    )
    df_TMM_photon = gen.run()

    return df_light, df_heavy, df_TMM_mumu, df_TMM_photon
