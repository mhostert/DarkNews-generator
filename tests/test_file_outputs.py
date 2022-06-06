#!/usr/bin/env python3

# import unittest
import pytest
import os
from pathlib import Path

import pandas as pd
import numpy as np
import DarkNews as dn


def test_output(light_DP_gen_all_outputs):
    
    df = light_DP_gen_all_outputs
    # check filename exists
    gen_path = df.attrs['data_path']
    
    assert gen_path
    assert os.path.isdir(gen_path)
    file_formats = ['pandas_df.pckl','pandas_df.parquet','ndarray.npy','HEPevt.dat']
    for format in file_formats:
        assert os.path.isfile(Path(f'{gen_path}/{format}'))
        assert os.access(Path(f'{gen_path}/{format}'), os.R_OK)

    # loading information
    df_std = pd.read_pickle(Path(f"{gen_path}/pandas_df.pckl"))
    df_pq = pd.read_parquet(Path(f"{gen_path}/pandas_df.parquet"), engine='pyarrow')
    nda = np.load(Path(f"{gen_path}/ndarray.npy"))

    # test that numpy array and dataframe formats save the same information
    assert (df_std.to_numpy()[nda!=0]/nda[nda!=0]!=1).sum() == 0 
    assert (df_pq.to_numpy()[nda!=0]/nda[nda!=0]!=1).sum() == 0


def test_MB_rates_of_BPs(gen_all_benchmarks):
    df_light,df_heavy = gen_all_benchmarks
    # check seeded generation
    expec = 12903.909075535918
    assert (np.abs(np.sum(df_light['w_event_rate']) - expec)/expec < 0.01)

    # check seeded generation
    expec = 4.965717632634995
    assert (np.abs(np.sum(df_heavy['w_event_rate']) - expec)/expec < 0.01)
