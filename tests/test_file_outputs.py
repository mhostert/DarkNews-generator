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


def close_enough(x,y, tol=1e-3):
    return (x - y)/y < tol


def test_MB_rates_of_BPs(SM_gen,
                        gen_simplest_benchmarks,
                        gen_other_final_states,
                        ):
    
    #######
    expect = 0.038374333394344776
    assert close_enough(SM_gen.w_event_rate.sum(), expect), 'seeded SM generation has changed!'
    
    #######
    df_light,df_heavy,df_TMM = gen_simplest_benchmarks
    # check seeded generation
    expect = 13504.557608335577
    assert close_enough(df_light.w_event_rate, expect, 'seeded light dark photon has changed!'

    # check seeded generation
    expect = 6.384879373980034
    assert close_enough(df_heavy.w_event_rate, expect, 'seeded heavy dark photon has changed!'

    # check seeded generation
    expect = 6.384879373980034
    assert close_enough(df_TMM.w_event_rate, expect, 'seeded heavy dark photon has changed!'

    #######
    df_light, df_heavy, df_TMM_mumu, df_TMM_photon = gen_other_final_states
    # check seeded generation
    expect = 13504.557608335577
    assert close_enough(df_light.w_event_rate, expect, 'seeded light dark photon has changed!'

    # check seeded generation
    expect = 6.384879373980034
    assert close_enough(df_heavy.w_event_rate, expect, 'seeded heavy dark photon has changed!'

    # check seeded generation
    expect = 6.384879373980034
    assert close_enough(df_TMM.w_event_rate, expect, 'seeded heavy dark photon has changed!'
