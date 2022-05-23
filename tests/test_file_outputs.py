#!/usr/bin/env python3

# import unittest
import pytest
import os
import pandas as pd
import numpy as np
import DarkNews as dn


def test_output(light_DP_gen):
    
    df = light_DP_gen
    # check filename exists
    gen_path = df.attrs['data_path']
    
    assert gen_path
    assert os.path.isdir(gen_path)
    file_formats = ['pandas_df.pckl','pandas_df.parquet','ndarray.npy','HEPevt.dat']
    for format in file_formats:
        assert os.path.isfile(f'{gen_path}{format}')
        assert os.access(f'{gen_path}{format}', os.R_OK)

    # loading information
    df_std = pd.read_pickle(f"{gen_path}pandas_df.pckl")
    df_pq = pd.read_parquet(f"{gen_path}pandas_df.parquet", engine='pyarrow')
    nda = np.load(f"{gen_path}ndarray.npy")

    # test that numpy array and dataframe formats save the same information
    assert (df_std.to_numpy()[nda!=0]/nda[nda!=0]!=1).sum() == 0 
    assert (df_pq.to_numpy()[nda!=0]/nda[nda!=0]!=1).sum() == 0





def test_MB_rates_of_BPs(light_DP_gen):
    df = light_DP_gen
    assert np.abs(np.sum(df.w_event_rate) - np.sum(df.w_event_rate)) <1e-4
