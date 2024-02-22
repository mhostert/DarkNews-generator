import pytest
import random
import numpy as np

from DarkNews import const
from DarkNews.GenLauncher import GenLauncher

MODEL_KWARGS = {"HNLtype": "dirac", "UD4": 1.0, "alphaD": 0.25, "Umu4": np.sqrt(9e-7), "epsilon": np.sqrt(2e-10 / const.alphaQED)}

GENERIC_MODEL_KWARGS = {"HNLtype": "dirac", "deV": 1, "duV": 1, "ddV": 1, "d_mu4": 1, "d_mu5": 1, "d_mu6": 1, "d_45": 1, "d_56": 1, "d_46": 1}

MOST_GENERIC_MODEL_KWARGS = {
    "HNLtype": "majorana",
    # Turn on all lepton couplings
    "deV": 1e-3,
    "duV": 1e-3,
    "ddV": 1e-3,
    "deA": 1e-3,
    "duA": 1e-3,
    "ddA": 1e-3,
    "deS": 1e-3,
    "duS": 1e-3,
    "ddS": 1e-3,
    "ceV": 1e-3,
    "cuV": 1e-3,
    "cdV": 1e-3,
    "ceA": 1e-3,
    "cuA": 1e-3,
    "cdA": 1e-3,
    # all neutral lepton couplings
    "c_mu4": 1e-3,
    "c_mu5": 1e-3,
    "c_mu6": 1e-3,
    "d_mu4": 1e-3,
    "d_mu5": 1e-3,
    "d_mu6": 1e-3,
    "s_mu4": 1e-3,
    "s_mu5": 1e-3,
    "s_mu6": 1e-3,
    "d_45": 1,
    "d_56": 1,
    "d_46": 1,
    "c_45": 1,
    "c_56": 1,
    "c_46": 1,
    "s_45": 1,
    "s_56": 1,
    "s_46": 1,
    "mu_tr_mu4": 2e-8,
    "mu_tr_mu5": 2e-8,
    "mu_tr_mu6": 2e-8,
    "mu_tr_45": 2,
    "mu_tr_56": 2,
    "mu_tr_46": 2,
}


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


@pytest.fixture(scope="session")
def portal_vs_simplified():
    EPSILON = 3.4e-4
    common_kwargs = {"loglevel": "ERROR", "HNLtype": "majorana", "neval": 1e4, "m5": 0.15, "m4": 0.1, "mzprime": 1.25}

    kwargs_1 = {"d_mu5": 1e-3, "d_45": 1 / 2, "dprotonV": EPSILON * const.eQED, "deV": EPSILON * const.eQED}
    kwargs_2 = {"UD4": 1 / np.sqrt(2), "UD5": 1 / np.sqrt(2), "Umu5": 1e-3, "Umu4": 1e-3, "gD": 1, "epsilon": EPSILON}

    gen_1 = GenLauncher(experiment="miniboone_fhc", **kwargs_1, **common_kwargs)
    gen_2 = GenLauncher(experiment="miniboone_fhc", **kwargs_2, **common_kwargs)

    return gen_1.run(), gen_2.run()


@pytest.fixture(scope="session")
def light_DP_gen_all_outputs():
    gen = GenLauncher(
        mzprime=0.03,
        m4=0.430,
        neval=1000,
        experiment="miniboone_fhc",
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
        experiment="miniboone_fhc",
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
        experiment="miniboone_fhc",
        loglevel="ERROR",
        seed=42,
        parquet=True,
        numpy=True,
        hepevt=True,
        hepevt_legacy=True,
        hepmc2=True,
        hepmc3=True,
        sparse=4,
        print_to_float32=True,
        **MODEL_KWARGS
    )
    return gen.run()


@pytest.fixture(scope="session")
def gen_simplest_benchmarks():
    gen = GenLauncher(mzprime=0.03, m4=0.420, neval=1000, experiment="miniboone_fhc", loglevel="ERROR", seed=42, **MODEL_KWARGS)
    df_light = gen.run()

    gen = GenLauncher(mzprime=1.25, m4=0.150, neval=1000, experiment="miniboone_fhc", loglevel="ERROR", seed=42, **MODEL_KWARGS)
    df_heavy = gen.run()

    gen = GenLauncher(mu_tr_mu4=2e-6, m4=0.150, epsilon=0.0, gD=0.0, Umu4=0.0, neval=1000, experiment="miniboone_fhc", loglevel="ERROR", seed=42)
    df_TMM = gen.run()

    return df_light, df_heavy, df_TMM


@pytest.fixture(scope="session")
def gen_other_final_states():
    gen = GenLauncher(mzprime=0.3, m4=0.5, decay_product="mu+mu-", neval=1000, experiment="miniboone_fhc", loglevel="ERROR", seed=42, **MODEL_KWARGS)
    df_light = gen.run()

    gen = GenLauncher(mzprime=1.25, m4=0.5, decay_product="mu+mu-", neval=1000, experiment="miniboone_fhc", loglevel="ERROR", seed=42, **MODEL_KWARGS)
    df_heavy = gen.run()

    gen = GenLauncher(
        mu_tr_mu4=2e-6,
        m4=0.5,
        epsilon=0.0,
        gD=0.0,
        decay_product="mu+mu-",
        Umu4=0.0,
        neval=1000,
        HNLtype="dirac",
        experiment="miniboone_fhc",
        loglevel="ERROR",
        seed=42,
    )
    df_TMM_mumu = gen.run()

    gen = GenLauncher(
        mu_tr_mu4=2e-6,
        m4=0.5,
        epsilon=0.0,
        gD=0.0,
        decay_product="photon",
        Umu4=0.0,
        neval=1000,
        HNLtype="dirac",
        experiment="miniboone_fhc",
        loglevel="ERROR",
        seed=42,
    )
    df_TMM_photon = gen.run()

    return df_light, df_heavy, df_TMM_mumu, df_TMM_photon


@pytest.fixture(scope="session")
def gen_most_generic_model():
    light_gen = GenLauncher(
        mzprime=1.25,
        mhprime=0.06,
        m4=0.100,
        m5=0.200,
        m6=0.300,
        neval=1000,
        experiment="miniboone_fhc",
        include_nelastic=True,
        loglevel="ERROR",
        seed=42,
        pandas=False,
        **MOST_GENERIC_MODEL_KWARGS
    )
    heavy_gen = GenLauncher(
        mzprime=1.0,
        mhprime=2.0,
        m4=0.100,
        m5=0.200,
        m6=0.300,
        neval=1000,
        experiment="miniboone_fhc",
        include_nelastic=True,
        loglevel="ERROR",
        seed=42,
        pandas=False,
        **MOST_GENERIC_MODEL_KWARGS
    )

    photon_gen = GenLauncher(
        mzprime=1.0,
        mhprime=2.0,
        m4=0.100,
        m5=0.200,
        m6=0.300,
        neval=1000,
        decay_product="photon",
        experiment="miniboone_fhc",
        include_nelastic=True,
        loglevel="ERROR",
        seed=42,
        pandas=False,
        **MOST_GENERIC_MODEL_KWARGS
    )

    return light_gen.run(), heavy_gen.run(), photon_gen.run()


@pytest.fixture(scope="session")
def gen_dirt_cases():
    gen = GenLauncher(mzprime=1.25, m4=0.5, neval=1000, experiment="sbnd_dirt", loglevel="ERROR", seed=42, **MODEL_KWARGS)
    df_1 = gen.run()

    gen = GenLauncher(mzprime=1.25, m4=0.5, neval=1000, experiment="microboone_dirt", loglevel="ERROR", seed=42, **MODEL_KWARGS)
    df_2 = gen.run()

    gen = GenLauncher(mzprime=1.25, m4=0.5, neval=1000, experiment="icarus_dirt", loglevel="ERROR", seed=42, **MODEL_KWARGS)
    df_3 = gen.run()

    gen = GenLauncher(mzprime=1.25, m4=0.5, neval=1000, experiment="miniboone_fhc_dirt", loglevel="ERROR", seed=42, **MODEL_KWARGS)
    df_4 = gen.run()

    return df_1, df_2, df_3, df_4
