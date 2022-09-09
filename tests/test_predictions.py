#!/usr/bin/env python3

import pytest
import platform
import DarkNews as dn

def close_enough(x, y, tol=1e-3):
    return (x - y) / y < tol

@pytest.mark.skipif("Linux" in platform.system(),
                    reason="Linux appears to misbehave with seeded random numbers on GitHub actions")
def test_MB_rates_of_BPs(
    SM_gen, gen_simplest_benchmarks, gen_other_final_states,
):

    #######
    expect = 0.038383306113781296
    assert close_enough(SM_gen.w_event_rate.sum(), expect), "seeded SM generation has changed!"

    #######
    df_light, df_heavy, df_TMM = gen_simplest_benchmarks
    # check seeded generation
    expect = 13349.332011155739
    assert close_enough(df_light.w_event_rate.sum(), expect), "seeded light dark photon has changed!"

    # check seeded generation
    expect = 5.502976676512979
    assert close_enough(df_heavy.w_event_rate.sum(), expect), "seeded heavy dark photon has changed!"

    # check seeded generation
    expect = 52283.95934429322
    assert close_enough(df_TMM.w_event_rate.sum(), expect), "seeded heavy dark photon has changed!"

    #######
    df_light, df_heavy, df_TMM_mumu, df_TMM_photon = gen_other_final_states
    # check seeded generation
    expect = 201.95690347250735
    assert close_enough(df_light.w_event_rate.sum(), expect), "seeded light dark photon to muons has changed!"

    # check seeded generation
    expect = 2.3240714690039566
    assert close_enough(df_heavy.w_event_rate.sum(), expect), "seeded heavy dark photon to muons has changed!"

    # check seeded generation
    expect = 3426.3242951080792
    assert close_enough(df_TMM_mumu.w_event_rate.sum(), expect), "seeded light dark photon to muons has changed!"
    assert "P_decay_ell_plus" in df_TMM_mumu.columns
    assert df_TMM_mumu["P_decay_ell_plus", "0"].min() > dn.const.m_mu, "Mu+ energy smaller than its mass? Not generating for muons?"
    assert df_TMM_mumu["P_decay_ell_minus", "0"].min() > dn.const.m_mu, "Mu- energy smaller than its mass? Not generating for muons?"

    # check seeded generation
    expect = 3446.7854182166047
    assert close_enough(df_TMM_photon.w_event_rate.sum(), expect), "seeded heavy dark photon to muons has changed!"
    assert "P_decay_photon" in df_TMM_photon.columns

def test_portal_vs_simplified(portal_vs_simplified):
    df_1, df_2 = portal_vs_simplified

    # simplified model approach generates similar output to 3 portal model
    assert df_1.w_event_rate.sum()/df_2.w_event_rate.sum() < 1.5
    assert df_1.w_decay_rate_0.sum()/df_2.w_decay_rate_0.sum() < 1.5


def test_geometries(SM_gen):

    df = SM_gen

    dn.geom.microboone_geometry(df)
    dn.geom.sbnd_geometry(df)
    dn.geom.icarus_geometry(df)
    dn.geom.miniboone_dirt_geometry(df)
    dn.geom.microboone_dirt_geometry(df)
    dn.geom.icarus_dirt_geometry(df)
    dn.geom.microboone_tpc_geometry(df)
    dn.geom.sbnd_dirt_geometry(df)
    dn.geom.miniboone_geometry(df)
    dn.geom.point_geometry(df)
