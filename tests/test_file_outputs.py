#!/usr/bin/env python3

# import unittest
import pytest
import os
import pandas as pd
import numpy as np
import DarkNews as dn


def test_output_files(light_DP_gen):
    
    df = light_DP_gen
    # check filename exists
    filename = df.attrs['data_path']
    assert filename
    assert os.path.isfile(filename)
    assert os.access(filename, os.R_OK)

    # loading information
    df_std = pd.read_pickle(f"{filename}pandas_df.pckl")
    assert df_std
    df_pq = pd.read_parquet(f"{filename}pandas_df.parquet", engine='pyarrow')
    assert df_pq
    nda = np.load(f"{filename}ndarray.npy")
    assert nda


    # test that numpy array and dataframe formats save the same information
    assert (df_std.to_numpy()[nda!=0]/nda[nda!=0]!=1).sum() == 0 
    assert (df_pq.to_numpy()[nda!=0]/nda[nda!=0]!=1).sum() == 0





def test_MB_rates_of_BPs(light_DP_gen):
    
    df = light_DP_gen

    assert (df_std.to_numpy()[nda!=0]/nda[nda!=0]!=1).sum() == 0 
    assert (df_pq.to_numpy()[nda!=0]/nda[nda!=0]!=1).sum() == 0

