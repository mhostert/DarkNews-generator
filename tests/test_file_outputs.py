#!/usr/bin/env python3

import pytest
import os
from pathlib import Path

import pandas as pd
import numpy as np
import pyhepmc as hep

import DarkNews as dn


def test_output(light_DP_gen_all_outputs, light_DP_gen_all_outputs_sparse):
    """Test all output formats of DarkNews"""

    for df in [light_DP_gen_all_outputs, light_DP_gen_all_outputs_sparse]:

        # check the attributes of the output pandas dataframe:
        assert "experiment" in df.attrs.keys(), "Could not find attribute 'experiment' in the DataFrame attrs"
        assert "model" in df.attrs.keys(), "Could not find attribute 'model' in the DataFrame attrs"
        assert "data_path" in df.attrs.keys(), "Could not find attribute 'data_path' in the DataFrame attrs"

        # check filename exists
        gen_path = df.attrs["data_path"]

        assert gen_path
        assert os.path.isdir(gen_path)
        file_formats = ["pandas_df.pckl", "pandas_df.parquet", "ndarray.npy", "HEPevt.dat"]
        for format in file_formats:
            assert os.path.isfile(Path(f"{gen_path}/{format}")), "Cannot find the generated datafile in the data_path attribute"
            assert os.access(Path(f"{gen_path}/{format}"), os.R_OK), "Cannot read the generated datafile in the data_path attribute"

        # loading information
        df_std = pd.read_pickle(Path(f"{gen_path}/pandas_df.pckl"))
        df_pq = pd.read_parquet(Path(f"{gen_path}/pandas_df.parquet"), engine="pyarrow")
        nda = np.load(Path(f"{gen_path}/ndarray.npy"))

        for df_pandas in [df_std, df_pq]:

            # helicity only present in sparse format
            if "helicity" in df_pandas.columns.levels[0]:
                # test that numpy array and dataframe formats save the same information
                df_pandas = df_pandas.replace(to_replace="conserving", value="+1")
                df_pandas = df_pandas.replace(to_replace="flipping", value="-1")

            # remove non-numeric entries
            drop_list = ["underlying_process", "target", "scattering_regime"]
            if not set(drop_list).isdisjoint(df_pandas.columns.levels[0]):
                df_for_numpy = df_pandas.drop([col for col in drop_list if col in df_pandas.columns.levels[0]], axis=1, level=0).to_numpy(dtype=np.float64)
            else:
                df_for_numpy = df_pandas.to_numpy(dtype=np.float64)
            assert (df_for_numpy[nda != 0] / nda[nda != 0] != 1).sum() == 0, "pandas dataframe and numpy array seem to contain different data."

        oss_hepevt = Path(f"{gen_path}/HEPevt.dat").__str__()
        oss_hepmc2 = Path(f"{gen_path}/hep_ascii.hepmc2").__str__()
        oss_hepmc3 = Path(f"{gen_path}/hep_ascii.hepmc3").__str__()

        with hep.ReaderHEPEVT(oss_hepevt) as f_hepevt, hep.ReaderAsciiHepMC2(oss_hepmc2) as f_hepmc2, hep.ReaderAscii(oss_hepmc3) as f_hepmc3:
            # test three cases
            for i in range(0, 3):
                evtHEPEVT = hep.GenEvent()
                f_hepevt.read_event(evtHEPEVT)

                evtHEPMC2 = hep.GenEvent()
                f_hepmc2.read_event(evtHEPMC2)

                evtHEPMC3 = hep.GenEvent()
                f_hepmc3.read_event(evtHEPMC3)

                # event number comparison
                assert evtHEPEVT.event_number == evtHEPMC2.event_number
                assert evtHEPEVT.event_number == evtHEPMC3.event_number

                assert evtHEPMC2.momentum_unit == evtHEPMC3.momentum_unit
                assert evtHEPMC2.length_unit == evtHEPMC3.length_unit
                assert evtHEPMC2.particles == evtHEPMC3.particles
                assert evtHEPMC2.vertices == evtHEPMC3.vertices
                assert evtHEPMC2.weight_names == evtHEPMC3.weight_names
                assert evtHEPMC2.weights == evtHEPMC3.weights

                for j in range(len(evtHEPMC3.weight_names)):
                    wn = evtHEPMC3.weight_names[j]
                    assert (
                        df[wn, ""][i] == evtHEPMC3.weights[j]
                    ), f'weight of type "{wn}" is {df[wn,""][i]} for dataframe and {evtHEPMC3.weights[j]} for HepMC3. They should be the same.'


def close_enough(x, y, tol=1e-3):
    return (x - y) / y < tol


def test_MB_rates_of_BPs(
    SM_gen, gen_simplest_benchmarks, gen_other_final_states,
):

    import platform
    # Ubuntu is giving me a hard time with seeded random numbers on GitHub tests
    if "Darwin" in platform.system():

        #######
        expect = 0.038374333394344776
        assert close_enough(SM_gen.w_event_rate.sum(), expect), "seeded SM generation has changed!"

        #######
        df_light, df_heavy, df_TMM = gen_simplest_benchmarks
        # check seeded generation
        expect = 13504.557608335577
        assert close_enough(df_light.w_event_rate.sum(), expect), "seeded light dark photon has changed!"

        # check seeded generation
        expect = 6.384879373980034
        assert close_enough(df_heavy.w_event_rate.sum(), expect), "seeded heavy dark photon has changed!"

        # check seeded generation
        expect = 52283.95934407082
        assert close_enough(df_TMM.w_event_rate.sum(), expect), "seeded heavy dark photon has changed!"

        #######
        df_light, df_heavy, df_TMM_mumu, df_TMM_photon = gen_other_final_states
        # check seeded generation
        expect = 205.54654591769042
        assert close_enough(df_light.w_event_rate.sum(), expect), "seeded light dark photon to muons has changed!"

        # check seeded generation
        expect = 2.490401387114275
        assert close_enough(df_heavy.w_event_rate.sum(), expect), "seeded heavy dark photon to muons has changed!"

        # check seeded generation
        expect = 3426.3242951081183
        assert close_enough(df_TMM_mumu.w_event_rate.sum(), expect), "seeded light dark photon to muons has changed!"
        assert "P_decay_ell_plus" in df_TMM_mumu.columns
        assert df_TMM_mumu["P_decay_ell_plus", "0"].min() > dn.const.m_mu, "Mu+ energy smaller than its mass? Not generating for muons?"
        assert df_TMM_mumu["P_decay_ell_minus", "0"].min() > dn.const.m_mu, "Mu- energy smaller than its mass? Not generating for muons?"

        # check seeded generation
        expect = 3446.785418216368
        assert close_enough(df_TMM_photon.w_event_rate.sum(), expect), "seeded heavy dark photon to muons has changed!"
        assert "P_decay_photon" in df_TMM_photon.columns
