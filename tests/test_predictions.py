#!/usr/bin/env python3

import pytest
import platform
import DarkNews as dn

import numpy as np

def close_enough(x, y, tol=1e-3):
    return abs(x - y) / y < tol

@pytest.mark.skipif("Linux" in platform.system(),
                    reason="Linux appears to misbehave with seeded random numbers on GitHub actions")
def test_MB_rates_of_BPs(
    SM_gen, 
    gen_simplest_benchmarks,
    gen_other_final_states,
    gen_most_generic_model,
    gen_dirt_cases
):

    #######
    expect = 0.038363091541911774
    assert close_enough(SM_gen.w_event_rate.sum(), expect), "seeded SM generation has changed!"

    #######
    df_light, df_heavy, df_TMM = gen_simplest_benchmarks
    # check seeded generation
    expect = 13180.463971429017
    assert close_enough(df_light.w_event_rate.sum(), expect), "seeded light dark photon has changed!"

    # check seeded generation
    expect = 5.516596808396571
    assert close_enough(df_heavy.w_event_rate.sum(), expect), "seeded heavy dark photon has changed!"

    # check seeded generation
    expect = 52620.678472114305
    assert close_enough(df_TMM.w_event_rate.sum(), expect), "seeded heavy dark photon has changed!"

    #######
    df_light, df_heavy, df_TMM_mumu, df_TMM_photon = gen_other_final_states
    # check seeded generation
    expect = 202.14258392685633
    assert close_enough(df_light.w_event_rate.sum(), expect), "seeded light dark photon to muons has changed!"

    # check seeded generation
    expect = 2.3258916609020366
    assert close_enough(df_heavy.w_event_rate.sum(), expect), "seeded heavy dark photon to muons has changed!"

    # check seeded generation
    expect = 3458.8436785760227
    assert close_enough(df_TMM_mumu.w_event_rate.sum(), expect), "seeded light dark photon to muons has changed!"
    assert "P_decay_ell_plus" in df_TMM_mumu.columns
    assert df_TMM_mumu["P_decay_ell_plus", "0"].min() > dn.const.m_mu, "Mu+ energy smaller than its mass? Not generating for muons?"
    assert df_TMM_mumu["P_decay_ell_minus", "0"].min() > dn.const.m_mu, "Mu- energy smaller than its mass? Not generating for muons?"

    # check seeded generation
    expect = 3411.9684811051043
    assert close_enough(df_TMM_photon.w_event_rate.sum(), expect), "seeded heavy dark photon to muons has changed!"
    assert "P_decay_photon" in df_TMM_photon.columns

    #######
    df_light, df_heavy, df_photon = gen_most_generic_model
    # check seeded generation
    expect = 23222307.376793236
    assert close_enough(df_light.w_event_rate.sum(), expect), "seeded light dark photon to muons has changed!"

    # check seeded generation
    expect = 178964.05762603838
    assert close_enough(df_heavy.w_event_rate.sum(), expect), "seeded heavy most-generic model has changed!"

    # check seeded generation
    expect = 179580.3899866729
    assert close_enough(df_photon.w_event_rate.sum(), expect), "seeded heavy most-generic model has changed!"

    #######
    df_1, df_2, df_3, df_4 = gen_dirt_cases
    # check seeded generation
    expect = 77791.02852083213
    assert close_enough(df_1.w_event_rate.sum(), expect), "seeded light dark photon to muons has changed!"

    # check seeded generation
    expect = 204.64030497933084
    assert close_enough(df_2.w_event_rate.sum(), expect), "seeded heavy dark photon to muons has changed!"

    # check seeded generation
    expect = 2849.3695799331485
    assert close_enough(df_3.w_event_rate.sum(), expect), "seeded light dark photon to muons has changed!"

    # check seeded generation
    expect = 3648.2531261721588
    assert close_enough(df_4.w_event_rate.sum(), expect), "seeded heavy dark photon to muons has changed!"


def test_portal_vs_simplified(portal_vs_simplified):
    df_1, df_2 = portal_vs_simplified

    # simplified model approach generates similar output to 3 portal model
    assert df_1.w_event_rate.sum()/df_2.w_event_rate.sum() < 1.5
    assert df_1.w_decay_rate_0.sum()/df_2.w_decay_rate_0.sum() < 1.5


def test_xsecs():

    # targets
    C12    = dn.detector.NuclearTarget("C12")

    # create models
    dipole   = dn.model.ThreePortalModel(name='dipole', m4 = 0.200, mu_tr_mu4=1e-6)
    vector_h   = dn.model.ThreePortalModel(name='heavy vector', m4 = 0.200, epsilon=1e-3, Umu4=1e-3, mzprime=1.25)
    vector_l   = dn.model.ThreePortalModel(name='light vector', m4 = 0.200, epsilon=1e-3, Umu4=1e-3, mzprime=0.03)
    scalar_h   = dn.model.ThreePortalModel(name='heavy scalar', m4 = 0.200, s_mu4=1e-3, Umu4=0.0, theta=1e-3, mhprime=1.25, epsilon=0.0, gD=0.0)
    scalar_l   = dn.model.ThreePortalModel(name='light scalar', m4 = 0.200, s_mu4=1e-3, Umu4=0.0, theta=1e-3, mhprime=0.03, epsilon=0.0, gD=0.0)

    # dipole interaction flips helicity while dark photon conserves it -- the other helicity channel can be computed, but will be subdominant.
    common_kwargs = {'nu_projectile': dn.pdg.numu, 'scattering_regime': 'coherent', 'nuclear_target': C12}
    calculator_dipole   = dn.UpscatteringProcess(TheoryModel = dipole, nu_upscattered=dipole.neutrino4, helicity = 'flipping', **common_kwargs)  
    calculator_vector_h  = dn.UpscatteringProcess(TheoryModel = vector_h, nu_upscattered=vector_h.neutrino4, helicity = 'conserving',  **common_kwargs) 
    calculator_vector_l   = dn.UpscatteringProcess(TheoryModel = vector_l, nu_upscattered=vector_l.neutrino4, helicity = 'conserving', **common_kwargs) 
    calculator_scalar_h = dn.UpscatteringProcess(TheoryModel = scalar_h, nu_upscattered=scalar_h.neutrino4, helicity = 'flipping', **common_kwargs) 
    calculator_scalar_l = dn.UpscatteringProcess(TheoryModel = scalar_l, nu_upscattered=scalar_l.neutrino4, helicity = 'flipping', **common_kwargs) 

    common_kwargs['scattering_regime'] = 'p-el'
    calculator_dipole_pel   = dn.UpscatteringProcess(TheoryModel = dipole, nu_upscattered=dipole.neutrino4, helicity = 'flipping', **common_kwargs) 
    calculator_vector_h_pel  = dn.UpscatteringProcess(TheoryModel = vector_h, nu_upscattered=vector_h.neutrino4, helicity = 'conserving',  **common_kwargs)
    calculator_vector_l_pel   = dn.UpscatteringProcess(TheoryModel = vector_l, nu_upscattered=vector_l.neutrino4, helicity = 'conserving', **common_kwargs)
    calculator_scalar_h_pel = dn.UpscatteringProcess(TheoryModel = scalar_h, nu_upscattered=scalar_h.neutrino4, helicity = 'flipping', **common_kwargs)
    calculator_scalar_l_pel = dn.UpscatteringProcess(TheoryModel = scalar_l, nu_upscattered=scalar_l.neutrino4, helicity = 'flipping', **common_kwargs)


    # dipole interaction flips helicity while dark photon conserves it -- the other helicity channel can be computed, but will be subdominant.
    common_kwargs = {'nu_projectile': dn.pdg.numu, 'scattering_regime': 'coherent', 'nuclear_target': C12}
    calculator_dipole_subd   = dn.UpscatteringProcess(TheoryModel = dipole, nu_upscattered=dipole.neutrino4, helicity = 'conserving', **common_kwargs)  
    calculator_vector_h_subd  = dn.UpscatteringProcess(TheoryModel = vector_h, nu_upscattered=vector_h.neutrino4, helicity = 'flipping',  **common_kwargs) 
    calculator_vector_l_subd   = dn.UpscatteringProcess(TheoryModel = vector_l, nu_upscattered=vector_l.neutrino4, helicity = 'flipping', **common_kwargs) 
    calculator_scalar_h_subd = dn.UpscatteringProcess(TheoryModel = scalar_h, nu_upscattered=scalar_h.neutrino4, helicity = 'conserving', **common_kwargs) 
    calculator_scalar_l_subd = dn.UpscatteringProcess(TheoryModel = scalar_l, nu_upscattered=scalar_l.neutrino4, helicity = 'conserving', **common_kwargs) 

    common_kwargs['scattering_regime'] = 'p-el'
    calculator_dipole_pel_subd   = dn.UpscatteringProcess(TheoryModel = dipole, nu_upscattered=dipole.neutrino4, helicity = 'conserving', **common_kwargs) 
    calculator_vector_h_pel_subd  = dn.UpscatteringProcess(TheoryModel = vector_h, nu_upscattered=vector_h.neutrino4, helicity = 'flipping',  **common_kwargs)
    calculator_vector_l_pel_subd   = dn.UpscatteringProcess(TheoryModel = vector_l, nu_upscattered=vector_l.neutrino4, helicity = 'flipping', **common_kwargs)
    calculator_scalar_h_pel_subd = dn.UpscatteringProcess(TheoryModel = scalar_h, nu_upscattered=scalar_h.neutrino4, helicity = 'conserving', **common_kwargs)
    calculator_scalar_l_pel_subd = dn.UpscatteringProcess(TheoryModel = scalar_l, nu_upscattered=scalar_l.neutrino4, helicity = 'conserving', **common_kwargs)

    xs_kwargs = {'Enu': np.linspace(0.1,10,5), 'NEVAL': 1000, 'NINT': 10, 'seed': 42}
    # dominant
    dipole_xs = (calculator_dipole.total_xsec(**xs_kwargs), calculator_dipole_pel.total_xsec(**xs_kwargs))
    vector_h_xs = (calculator_vector_h.total_xsec(**xs_kwargs), calculator_vector_h_pel.total_xsec(**xs_kwargs))
    vector_l_xs = (calculator_vector_l.total_xsec(**xs_kwargs), calculator_vector_l_pel.total_xsec(**xs_kwargs))
    scalar_h_xs = (calculator_scalar_h.total_xsec(**xs_kwargs), calculator_scalar_h_pel.total_xsec(**xs_kwargs))
    scalar_l_xs = (calculator_scalar_l.total_xsec(**xs_kwargs), calculator_scalar_l_pel.total_xsec(**xs_kwargs))

    # sub-dominant
    dipole_xs_subd = (calculator_dipole_subd.total_xsec(**xs_kwargs), calculator_dipole_pel_subd.total_xsec(**xs_kwargs))
    vector_h_xs_subd = (calculator_vector_h_subd.total_xsec(**xs_kwargs), calculator_vector_h_pel_subd.total_xsec(**xs_kwargs))
    vector_l_xs_subd = (calculator_vector_l_subd.total_xsec(**xs_kwargs), calculator_vector_l_pel_subd.total_xsec(**xs_kwargs))
    scalar_h_xs_subd = (calculator_scalar_h_subd.total_xsec(**xs_kwargs), calculator_scalar_h_pel_subd.total_xsec(**xs_kwargs))
    scalar_l_xs_subd = (calculator_scalar_l_subd.total_xsec(**xs_kwargs), calculator_scalar_l_pel_subd.total_xsec(**xs_kwargs))

    kwargs = {'tol': 1e-3}

    assert close_enough( np.sum(  dipole_xs[0] +   dipole_xs_subd[0] +   dipole_xs[1] +   dipole_xs_subd[1])*1e38, 1.167635698988556, **kwargs), "seeded dipole_xs_subd prediction changed."
    assert close_enough( np.sum(vector_l_xs[0] + vector_l_xs_subd[0] + vector_l_xs[1] + vector_l_xs_subd[1])*1e38, 47.71350400042774, **kwargs), "seeded vector_l_xs_subd prediction changed."
    assert close_enough( np.sum(scalar_l_xs[0] + scalar_l_xs_subd[0] + scalar_l_xs[1] + scalar_l_xs_subd[1])*1e38, 1.292457796553301e-06, **kwargs), "seeded scalar_l_xs_subd prediction changed."
    assert close_enough( np.sum(scalar_l_xs[0] + scalar_l_xs_subd[0] + scalar_l_xs[1] + scalar_l_xs_subd[1])*1e38, 1.292457796553301e-06, **kwargs), "seeded scalar_l_xs_subd prediction changed."
    assert close_enough( np.sum(vector_h_xs[0] + vector_h_xs_subd[0] + vector_h_xs[1] + vector_h_xs_subd[1])*1e38, 0.0009700678537285885, **kwargs), "seeded vector_h_xs_subd prediction changed."
    assert close_enough( np.sum(scalar_h_xs[0] + scalar_h_xs_subd[0] + scalar_h_xs[1] + scalar_h_xs_subd[1])*1e38, 5.916783033781255e-11, **kwargs), "seeded scalar_h_xs_subd prediction changed."




def test_geometries(SM_gen):

    df = SM_gen

    dn.geom.microboone_geometry(df)
    dn.geom.sbnd_geometry(df)
    dn.geom.icarus_geometry(df)
    dn.geom.miniboone_dirt_geometry(df)
    dn.geom.microboone_dirt_geometry(df)
    dn.geom.microboone_tpc_geometry(df)
    dn.geom.icarus_dirt_geometry(df)
    dn.geom.microboone_tpc_geometry(df)
    dn.geom.sbnd_dirt_geometry(df)
    dn.geom.miniboone_geometry(df)
    dn.geom.point_geometry(df)
