#!/usr/bin/env python3

import pytest
import os
import sys

from DarkNews import scripts


def test_dn_gen():
    """test_scripts check that dn scripts run and produce the output desired"""
    # dark photon
    os.system('dn_gen --path="./test_script_v1" --HNLtype="majorana" --make_summary_plots')

    filename = "test_script_v1/data/miniboone_fhc/3plus1/m4_0.15_mzprime_1.25_majorana/pandas_df.pckl"
    assert os.path.exists(filename), f"Could not find dn_gen output in {filename}"
    plots_path = "test_script_v1/data/miniboone_fhc/3plus1/m4_0.15_mzprime_1.25_majorana/summary_plots"
    assert os.path.exists(plots_path), f"Could not find summary plots in {plots_path}"

    # now for TMM
    os.system('dn_gen --path="./test_script_v2" --HNLtype="majorana" --decay_product="photon" --mu_tr_mu4=1e-6 --make_summary_plots')

    filename = "test_script_v2/data/miniboone_fhc/3plus1/m4_0.15_mzprime_1.25_mu_tr_mu4_1e-06_majorana/pandas_df.pckl"
    assert os.path.exists(filename), f"Could not find dn_gen output in {filename}"
    plots_path = "test_script_v2/data/miniboone_fhc/3plus1/m4_0.15_mzprime_1.25_mu_tr_mu4_1e-06_majorana/summary_plots"
    assert os.path.exists(plots_path), f"Could not find summary plots in {plots_path}"


def test_python_call_to_dngen():
    sys.argv = [sys.argv[0]]
    scripts.dn_gen()


@pytest.mark.skip(reason="Still need to find a way to set up git for venvs")
def test_examples_download():
    os.system("dn_get_examples")
    assert os.path.exists("DarkNews-examples"), "Could not find DarkNews-examples/ after running dn_get_examples"
