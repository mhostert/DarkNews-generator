#!/usr/bin/env python3

import pytest
import platform
import numpy as np

import DarkNews as dn

from .helpers import assert_all, soft_compare, soft_assert, close_enough


def is_macos_14():
    if platform.system() == "Darwin":
        vers = platform.mac_ver()[0]
        if vers.startswith("14."):
            return True
    return False


@pytest.mark.skipif(not is_macos_14(), reason="This test runs only on macOS 14.")
def test_MB_benchmarks(gen_simplest_benchmarks):
    df_light, df_heavy, df_TMM = gen_simplest_benchmarks

    with assert_all() as assertions:
        # check seeded generation
        expect = 13215.295377586925
        assertions.append(soft_compare(df_light.w_event_rate.sum(), expect, "seeded light dark photon has changed!"))

        # check seeded generation
        expect = 5.574120950173484
        assertions.append(soft_compare(df_heavy.w_event_rate.sum(), expect, "seeded heavy dark photon has changed!"))

        # check seeded generation
        expect = 83495.38110215841
        assertions.append(soft_compare(df_TMM.w_event_rate.sum(), expect, "seeded TMM has changed!"))


@pytest.mark.skipif(not is_macos_14(), reason="This test runs only on macOS 14.")
def test_MB_SM(SM_gen):
    expect = 0.03666960244288936
    soft_compare(SM_gen.w_event_rate.sum(), expect, "seeded SM generation has changed!")


@pytest.mark.skipif(not is_macos_14(), reason="This test runs only on macOS 14.")
def test_MB_other_final_states(gen_other_final_states):
    df_light, df_heavy, df_TMM_mumu, df_TMM_photon = gen_other_final_states

    with assert_all() as assertions:
        # check seeded generation
        expect = 188.64454603928297
        assertions.append(soft_compare(df_light.w_event_rate.sum(), expect, "seeded light dark photon to muons has changed!"))

        # check seeded generation
        expect = 2.4974196930861705
        assertions.append(soft_compare(df_heavy.w_event_rate.sum(), expect, "seeded heavy dark photon to muons has changed!"))

        # check seeded generation
        expect = 3621.413325452752
        assertions.append(soft_compare(df_TMM_mumu.w_event_rate.sum(), expect, "seeded TMM to muons has changed!"))
        assertions.append(soft_assert(("P_decay_ell_plus" in df_TMM_mumu.columns), "Could not find ell+ in the decay products!"))
        assertions.append(
            soft_assert(df_TMM_mumu["P_decay_ell_plus", "0"].min() > dn.const.m_mu, "Mu+ energy smaller than its mass? Not generating for muons?")
        )
        assertions.append(
            soft_assert(df_TMM_mumu["P_decay_ell_minus", "0"].min() > dn.const.m_mu, "Mu- energy smaller than its mass? Not generating for muons?")
        )

        # check seeded generation
        expect = 3471.176387572824
        assertions.append(soft_compare(df_TMM_photon.w_event_rate.sum(), expect, "seeded heavy dark photon to muons has changed!"))
        assertions.append(soft_assert(("P_decay_photon" in df_TMM_photon.columns), "Could not find photon in the decay products!"))


@pytest.mark.skipif(not is_macos_14(), reason="This test runs only on macOS 14.")
def test_MB_generic_model(gen_most_generic_model):
    df_light, df_heavy, df_photon = gen_most_generic_model

    with assert_all() as assertions:
        # check seeded generation
        expect = 25570154.745263256
        assertions.append(soft_compare(df_light.w_event_rate.sum(), expect, "seeded light dark photon to muons has changed!"))

        # check seeded generation
        expect = 192334.36051129547
        assertions.append(soft_compare(df_heavy.w_event_rate.sum(), expect, "seeded heavy most-generic model has changed!"))

        # check seeded generation
        expect = 220759.7604344437
        assertions.append(soft_compare(df_photon.w_event_rate.sum(), expect, "seeded heavy most-generic model has changed!"))


@pytest.mark.skipif(not is_macos_14(), reason="This test runs only on macOS 14.")
def test_MB_dirt(gen_dirt_cases):
    df_1, df_2, df_3, df_4 = gen_dirt_cases

    with assert_all() as assertions:
        # check seeded generation
        expect = 76212.61530482752
        assertions.append(soft_compare(df_1.w_event_rate.sum(), expect, "seeded sbnd dirt generation has changed!"))

        # check seeded generation
        expect = 34.6264453233758
        assertions.append(soft_compare(df_2.w_event_rate.sum(), expect, "seeded microboone dirt generation has changed!"))

        # check seeded generation
        expect = 290.55091582348734
        assertions.append(soft_compare(df_3.w_event_rate.sum(), expect, "seeded icarus dirt generation has changed!"))

        # check seeded generation
        expect = 398.08986326841875
        assertions.append(soft_compare(df_4.w_event_rate.sum(), expect, "seeded miniboone dirt generation has changed!"))


def test_portal_vs_simplified(portal_vs_simplified):
    df_1, df_2 = portal_vs_simplified

    # simplified model approach generates similar output to 3 portal model
    with assert_all() as assertions:
        assertions.append(soft_assert((df_1.w_event_rate.sum() / df_2.w_event_rate.sum() < 1.5), "3 portal and general models do not agree."))
        assertions.append(soft_assert((df_1.w_decay_rate_0.sum() / df_2.w_decay_rate_0.sum() < 1.5), "3 portal and general models do not agree."))


def test_xsecs():
    # targets
    C12 = dn.detector.NuclearTarget("C12")

    # create models
    dipole = dn.model.ThreePortalModel(name="dipole", m4=0.200, mu_tr_mu4=2e-6)
    vector_h = dn.model.ThreePortalModel(name="heavy vector", m4=0.200, epsilon=1e-3, Umu4=1e-3, mzprime=1.25)
    vector_l = dn.model.ThreePortalModel(name="light vector", m4=0.200, epsilon=1e-3, Umu4=1e-3, mzprime=0.03)
    scalar_h = dn.model.ThreePortalModel(name="heavy scalar", m4=0.200, s_mu4=1e-3, Umu4=0.0, theta=1e-3, mhprime=1.25, epsilon=0.0, gD=0.0)
    scalar_l = dn.model.ThreePortalModel(name="light scalar", m4=0.200, s_mu4=1e-3, Umu4=0.0, theta=1e-3, mhprime=0.03, epsilon=0.0, gD=0.0)

    # dipole interaction flips helicity while dark photon conserves it -- the other helicity channel can be computed, but will be subdominant.
    common_kwargs = {"nu_projectile": dn.pdg.numu, "scattering_regime": "coherent", "nuclear_target": C12}
    calculator_dipole = dn.UpscatteringProcess(TheoryModel=dipole, nu_upscattered=dipole.neutrino4, helicity="flipping", **common_kwargs)
    calculator_vector_h = dn.UpscatteringProcess(TheoryModel=vector_h, nu_upscattered=vector_h.neutrino4, helicity="conserving", **common_kwargs)
    calculator_vector_l = dn.UpscatteringProcess(TheoryModel=vector_l, nu_upscattered=vector_l.neutrino4, helicity="conserving", **common_kwargs)
    calculator_scalar_h = dn.UpscatteringProcess(TheoryModel=scalar_h, nu_upscattered=scalar_h.neutrino4, helicity="flipping", **common_kwargs)
    calculator_scalar_l = dn.UpscatteringProcess(TheoryModel=scalar_l, nu_upscattered=scalar_l.neutrino4, helicity="flipping", **common_kwargs)

    common_kwargs["scattering_regime"] = "p-el"
    calculator_dipole_pel = dn.UpscatteringProcess(TheoryModel=dipole, nu_upscattered=dipole.neutrino4, helicity="flipping", **common_kwargs)
    calculator_vector_h_pel = dn.UpscatteringProcess(TheoryModel=vector_h, nu_upscattered=vector_h.neutrino4, helicity="conserving", **common_kwargs)
    calculator_vector_l_pel = dn.UpscatteringProcess(TheoryModel=vector_l, nu_upscattered=vector_l.neutrino4, helicity="conserving", **common_kwargs)
    calculator_scalar_h_pel = dn.UpscatteringProcess(TheoryModel=scalar_h, nu_upscattered=scalar_h.neutrino4, helicity="flipping", **common_kwargs)
    calculator_scalar_l_pel = dn.UpscatteringProcess(TheoryModel=scalar_l, nu_upscattered=scalar_l.neutrino4, helicity="flipping", **common_kwargs)

    # dipole interaction flips helicity while dark photon conserves it -- the other helicity channel can be computed, but will be subdominant.
    common_kwargs = {"nu_projectile": dn.pdg.numu, "scattering_regime": "coherent", "nuclear_target": C12}
    calculator_dipole_subd = dn.UpscatteringProcess(TheoryModel=dipole, nu_upscattered=dipole.neutrino4, helicity="conserving", **common_kwargs)
    calculator_vector_h_subd = dn.UpscatteringProcess(TheoryModel=vector_h, nu_upscattered=vector_h.neutrino4, helicity="flipping", **common_kwargs)
    calculator_vector_l_subd = dn.UpscatteringProcess(TheoryModel=vector_l, nu_upscattered=vector_l.neutrino4, helicity="flipping", **common_kwargs)
    calculator_scalar_h_subd = dn.UpscatteringProcess(TheoryModel=scalar_h, nu_upscattered=scalar_h.neutrino4, helicity="conserving", **common_kwargs)
    calculator_scalar_l_subd = dn.UpscatteringProcess(TheoryModel=scalar_l, nu_upscattered=scalar_l.neutrino4, helicity="conserving", **common_kwargs)

    common_kwargs["scattering_regime"] = "p-el"
    calculator_dipole_pel_subd = dn.UpscatteringProcess(TheoryModel=dipole, nu_upscattered=dipole.neutrino4, helicity="conserving", **common_kwargs)
    calculator_vector_h_pel_subd = dn.UpscatteringProcess(TheoryModel=vector_h, nu_upscattered=vector_h.neutrino4, helicity="flipping", **common_kwargs)
    calculator_vector_l_pel_subd = dn.UpscatteringProcess(TheoryModel=vector_l, nu_upscattered=vector_l.neutrino4, helicity="flipping", **common_kwargs)
    calculator_scalar_h_pel_subd = dn.UpscatteringProcess(TheoryModel=scalar_h, nu_upscattered=scalar_h.neutrino4, helicity="conserving", **common_kwargs)
    calculator_scalar_l_pel_subd = dn.UpscatteringProcess(TheoryModel=scalar_l, nu_upscattered=scalar_l.neutrino4, helicity="conserving", **common_kwargs)

    xs_kwargs = {"Enu": np.linspace(0.1, 10, 5), "NEVAL": 1000, "NINT": 10, "seed": 42}
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

    kwargs = {"tol": 1e-3}
    if is_macos_14():
        with assert_all() as assertions:
            assertions.append(
                soft_assert(
                    close_enough(np.sum(dipole_xs[0] + dipole_xs_subd[0] + dipole_xs[1] + dipole_xs_subd[1]) * 1e38, 1.167635698988556, **kwargs),
                    "seeded dipole_xs_subd prediction changed.",
                )
            )
            assertions.append(
                soft_assert(
                    close_enough(np.sum(vector_l_xs[0] + vector_l_xs_subd[0] + vector_l_xs[1] + vector_l_xs_subd[1]) * 1e38, 47.71350400042774, **kwargs),
                    "seeded vector_l_xs_subd prediction changed.",
                )
            )
            assertions.append(
                soft_assert(
                    close_enough(np.sum(scalar_l_xs[0] + scalar_l_xs_subd[0] + scalar_l_xs[1] + scalar_l_xs_subd[1]) * 1e38, 1.292457796553301e-06, **kwargs),
                    "seeded scalar_l_xs_subd prediction changed.",
                )
            )
            assertions.append(
                soft_assert(
                    close_enough(np.sum(scalar_l_xs[0] + scalar_l_xs_subd[0] + scalar_l_xs[1] + scalar_l_xs_subd[1]) * 1e38, 1.292457796553301e-06, **kwargs),
                    "seeded scalar_l_xs_subd prediction changed.",
                )
            )
            assertions.append(
                soft_assert(
                    close_enough(np.sum(vector_h_xs[0] + vector_h_xs_subd[0] + vector_h_xs[1] + vector_h_xs_subd[1]) * 1e38, 0.0009700678537285885, **kwargs),
                    "seeded vector_h_xs_subd prediction changed.",
                )
            )
            assertions.append(
                soft_assert(
                    close_enough(np.sum(scalar_h_xs[0] + scalar_h_xs_subd[0] + scalar_h_xs[1] + scalar_h_xs_subd[1]) * 1e38, 5.916783033781255e-11, **kwargs),
                    "seeded scalar_h_xs_subd prediction changed.",
                )
            )


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
