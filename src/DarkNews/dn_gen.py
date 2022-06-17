#!/usr/bin/env python3

import argparse
from DarkNews.GenLauncher import GenLauncher

def dn_gen():
    DEFAULTS = GenLauncher(loglevel='error')
    # Default case implements the 3+2 model of Ballett et al, Phys. Rev. D 101, 115025 (2020) (https://arxiv.org/abs/1903.07589)

    # --------------
    # User specified
    # particle masses
    # use argument_default=argparse.SUPPRESS so no defaults attributes are instantiated in the final Namespace
    parser = argparse.ArgumentParser(description="Generate upscattering events", formatter_class=argparse.ArgumentDefaultsHelpFormatter, argument_default=argparse.SUPPRESS)

    ##### file containing the parameters
    parser.add_argument("--param-file", type=str, help="file containing parameters definitions")

    ##### dark sector spectrum
    parser.add_argument("--mzprime", type=float, help="Z' mass")
    parser.add_argument("--m4", type=float, help="mass of the fourth neutrino")
    parser.add_argument("--m5", type=float, help="mass of the fifth neutrino")
    parser.add_argument("--m6", type=float, help="mass of the sixth neutrino")

    parser.add_argument("--HNLtype", help="HNLtype: dirac or majorana", choices=DEFAULTS._choices['HNLtype'])

    # neutral lepton mixing
    parser.add_argument("--Ue4", type=float, help="Ue4")
    parser.add_argument("--Ue5", type=float, help="Ue5")
    parser.add_argument("--Ue6", type=float, help="Ue6")

    parser.add_argument("--Umu4", type=float, help="Umu4")
    parser.add_argument("--Umu5", type=float, help="Umu5")
    parser.add_argument("--Umu6", type=float, help="Umu6")

    parser.add_argument("--Utau4", type=float, help="Utau4")
    parser.add_argument("--Utau5", type=float, help="Utau5")
    parser.add_argument("--Utau6", type=float, help="Utau6")

    parser.add_argument("--UD4", type=float, help="UD4")
    parser.add_argument("--UD5", type=float, help="UD5")
    parser.add_argument("--UD6", type=float, help="UD6")

    # dark coupling choices
    parser.add_argument("--gD", type=float, help="U(1)_d dark coupling")
    parser.add_argument("--alphaD", type=float, help="U(1)_d  alpha_dark = (g_dark^2 /4 pi)")

    # kinetic mixing options
    parser.add_argument("--epsilon", type=float, help="epsilon")
    parser.add_argument("--epsilon2", type=float, help="epsilon^2")
    parser.add_argument("--alpha_epsilon2", type=float, help="alpha_QED*epsilon^2")
    parser.add_argument("--chi", type=float, help="chi")

    # TMM in GeV^-1
    parser.add_argument("--mu_tr_e4", type=float, help="TMM mu_tr_e4")
    parser.add_argument("--mu_tr_e5", type=float, help="TMM mu_tr_e5")
    parser.add_argument("--mu_tr_e6", type=float, help="TMM mu_tr_e6")

    parser.add_argument("--mu_tr_mu4", type=float, help="TMM mu_tr_mu4")
    parser.add_argument("--mu_tr_mu5", type=float, help="TMM mu_tr_mu5")
    parser.add_argument("--mu_tr_mu6", type=float, help="TMM mu_tr_mu6")

    parser.add_argument("--mu_tr_tau4", type=float, help="TMM mu_tr_tau4")
    parser.add_argument("--mu_tr_tau5", type=float, help="TMM mu_tr_tau5")
    parser.add_argument("--mu_tr_tau6", type=float, help="TMM mu_tr_tau6")

    parser.add_argument("--mu_tr_44", type=float, help="TMM mu_tr_44")
    parser.add_argument("--mu_tr_45", type=float, help="TMM mu_tr_45")
    parser.add_argument("--mu_tr_46", type=float, help="TMM mu_tr_46")

    parser.add_argument("--mu_tr_55", type=float, help="TMM mu_tr_55")
    parser.add_argument("--mu_tr_56", type=float, help="TMM mu_tr_56")

    parser.add_argument("--mu_tr_66", type=float, help="TMM mu_tr_66")

    # Scalar vertices
    parser.add_argument("--s_e4", type=float, help="scalar vertex s_e4")
    parser.add_argument("--s_e5", type=float, help="scalar vertex s_e5")
    parser.add_argument("--s_e6", type=float, help="scalar vertex s_e6")

    parser.add_argument("--s_mu4", type=float, help="scalar vertex s_mu4")
    parser.add_argument("--s_mu5", type=float, help="scalar vertex s_mu5")
    parser.add_argument("--s_mu6", type=float, help="scalar vertex s_mu6")

    parser.add_argument("--s_tau4", type=float, help="scalar vertex s_tau4")
    parser.add_argument("--s_tau5", type=float, help="scalar vertex s_tau5")
    parser.add_argument("--s_tau6", type=float, help="scalar vertex s_tau6")

    parser.add_argument("--s_44", type=float, help="scalar vertex s_44")
    parser.add_argument("--s_45", type=float, help="scalar vertex s_45")
    parser.add_argument("--s_46", type=float, help="scalar vertex s_46")

    parser.add_argument("--s_55", type=float, help="scalar vertex s_55")
    parser.add_argument("--s_56", type=float, help="scalar vertex s_56")

    parser.add_argument("--s_66", type=float, help="scalar vertex s_66")


    # visible final states in HNL decay
    parser.add_argument("--decay_product", help="decay process of interest", choices=DEFAULTS._choices['decay_product'])

    # experiments    
    parser.add_argument("--exp", type=str.lower, help="experiment file path or keyword")
                                                                    
    # scattering types
    parser.add_argument("--nopelastic", help="do not generate proton elastic events", action="store_true")
    parser.add_argument("--nocoh", help="do not generate coherent events", action="store_true")
    parser.add_argument("--include_nelastic", help="generate neutron elastic events", action="store_true")

    parser.add_argument("--noHC", help="do not include helicity conserving events", action="store_true")
    parser.add_argument("--noHF", help="do not include helicity flipping events", action="store_true")


    ###########
    # run related arguments
    parser.add_argument("--loglevel", help="Logging level")
    parser.add_argument("--verbose", help="Verbose for logging", action="store_true")
    parser.add_argument("--logfile", help="Path to logfile. If not set, use std output.")

    # Vegas parameters 
    parser.add_argument("--neval", type=int, help="number of evaluations of integrand")
    parser.add_argument("--nint", type=int, help="number of adaptive iterations")
    parser.add_argument("--neval_warmup", type=int, help="number of evaluations of integrand in warmup")
    parser.add_argument("--nint_warmup", type=int, help="number of adaptive iterations in warmup")

    # program options
    parser.add_argument("--pandas", help="If true, prints pandas dataframe to .pckl files", action="store_true")
    parser.add_argument("--parquet", help="If true, prints pandas dataframe to .parquet files. Loses metadata in attrs.", action="store_true")
    parser.add_argument("--numpy", help="If true, prints events as ndarrays in a .npy file", action="store_true")
    parser.add_argument("--hepevt", help="If true, unweigh events and print them in HEPEVT-formatted text files", action="store_true")
    parser.add_argument("--hepevt_unweigh", help="unweigh events when printing in HEPEVT format (needs large statistics)", action="store_true")
    parser.add_argument("--unweighed_hepevt_events", type=int, help="number of unweighed events to accept in HEPEVT format. Has to be much smaller than neval for unweigh procedure to work.")

    parser.add_argument("--sparse", help="drop all information in the event except for visible particle momenta, neutrino energy, and weights.", action="store_true")
    parser.add_argument("--print_to_float32", help="Use float32 instead of default float64 when printing output to save storage space.", action="store_true")

    parser.add_argument("--enforce_prompt", help="forces the particles to decay promptly", action="store_true")
    parser.add_argument("--make_summary_plots", help="generate summary plots of kinematics", action="store_true")
    parser.add_argument("--path", help="path where to save run's outputs")
    parser.add_argument("--seed", type=int, help="numpy seed to be used by vegas.")

    kwargs = vars(parser.parse_args())

    gen_object = GenLauncher(**kwargs)
    gen_object.run(
        loglevel=kwargs.get("loglevel", gen_object.loglevel),
        verbose=kwargs.get("verbose", gen_object.verbose),
        logfile=kwargs.get("logfile", gen_object.logfile),
    )

if __name__ == "__main__":
    dn_gen()