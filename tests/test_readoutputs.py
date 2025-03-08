#!/usr/bin/env python3

import os
from pathlib import Path

import pandas as pd
import numpy as np

from DarkNews import HAS_PYHEPMC3

if HAS_PYHEPMC3:
    import pyhepmc.io as io


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
            assert np.allclose(df_for_numpy, nda, rtol=1e-05, equal_nan=True), "pandas dataframe and numpy array seem to contain different data."

        oss_hepevt = Path(f"{gen_path}/HEPevt.dat").__str__()
        oss_hepmc2 = Path(f"{gen_path}/hep_ascii.hepmc2").__str__()
        oss_hepmc3 = Path(f"{gen_path}/hep_ascii.hepmc3").__str__()

        if HAS_PYHEPMC3:
            with io.ReaderHEPEVT(oss_hepevt) as r_hepevt, io.ReaderAsciiHepMC2(oss_hepmc2) as r_hepmc2, io.ReaderAscii(oss_hepmc3) as r_hepmc3:
                # test three cases
                for i in range(0, 3):
                    evtHEPEVT = r_hepevt.read()
                    evtHEPMC2 = r_hepmc2.read()
                    evtHEPMC3 = r_hepmc3.read()

                    # event number comparison
                    assert evtHEPEVT.event_number == evtHEPMC2.event_number, f"{evtHEPEVT} \n\n {evtHEPMC2}"
                    assert evtHEPEVT.event_number == evtHEPMC3.event_number, f"{evtHEPEVT} \n\n {evtHEPMC3}"

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
