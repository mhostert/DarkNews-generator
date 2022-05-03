#!/usr/bin/env python3

import argparse
from DarkNews.GenLauncher import GenLauncher

def dn_gen():
    DEFAULTS = GenLauncher(log='error')
    # Default case implements the 3+2 model of Ballett et al, Phys. Rev. D 101, 115025 (2020) (https://arxiv.org/abs/1903.07589)

    # --------------
    # User specified
    # particle masses
    # use argument_default=argparse.SUPPRESS so no defaults attributes are instantiated in the final Namespace
    parser = argparse.ArgumentParser(description="Generate dark nu events", formatter_class=argparse.ArgumentDefaultsHelpFormatter, argument_default=argparse.SUPPRESS)

    ##### file containing the parameters
    parser.add_argument("--param-file", type=str, help="file containing parameters definitions")

    ##### dark sector spectrum
    parser.add_argument("--mzprime", type=float, help="Z' mass")
    parser.add_argument("--m4", type=float, help="mass of the fourth neutrino")
    parser.add_argument("--m5", type=float, help="mass of the fifth neutrino")
    parser.add_argument("--m6", type=float, help="mass of the sixth neutrino")

    parser.add_argument("--HNLtype", help="HNLtype: dirac or majorana", choices=DEFAULTS._choices['HNLtype'])

    # neutral lepton mixing
    parser.add_argument("--ue4", type=float, help="Ue4")
    parser.add_argument("--ue5", type=float, help="Ue5")
    parser.add_argument("--ue6", type=float, help="Ue6")

    parser.add_argument("--umu4", type=float, help="Umu4")
    parser.add_argument("--umu5", type=float, help="Umu5")
    parser.add_argument("--umu6", type=float, help="Umu6")

    parser.add_argument("--utau4", type=float, help="Utau4")
    parser.add_argument("--utau5", type=float, help="Utau5")
    parser.add_argument("--utau6", type=float, help="Utau6")

    parser.add_argument("--ud4", type=float, help="UD4")
    parser.add_argument("--ud5", type=float, help="UD5")
    parser.add_argument("--ud6", type=float, help="UD6")

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

    parser.add_argument("--mu_tr_44", type=float, help="TMM mu_tr_tau4")
    parser.add_argument("--mu_tr_45", type=float, help="TMM mu_tr_tau5")
    parser.add_argument("--mu_tr_46", type=float, help="TMM mu_tr_tau6")

    parser.add_argument("--mu_tr_55", type=float, help="TMM mu_tr_tau5")
    parser.add_argument("--mu_tr_56", type=float, help="TMM mu_tr_tau6")

    parser.add_argument("--mu_tr_66", type=float, help="TMM mu_tr_tau6")

    # visible final states in HNL decay
    parser.add_argument("--decay_products", help="decay process of interest", choices=DEFAULTS._choices['decay_products'])

    # experiments    
    parser.add_argument("--exp", type=str.lower, help="experiment file path or keyword")
                                                                    
    # scattering types
    parser.add_argument("--nopelastic", help="do not generate proton elastic events", action="store_true")
    parser.add_argument("--nocoh", help="do not generate coherent events", action="store_true")

    parser.add_argument("--noHC", help="do not include helicity conserving events", action="store_true")
    parser.add_argument("--noHF", help="do not include helicity flipping events", action="store_true")


    ###########
    # run related arguments
    parser.add_argument("--log", help="Logging level")
    parser.add_argument("--verbose", help="Verbose for logging", action="store_true")
    parser.add_argument("--logfile", help="Path to logfile. If not set, use std output.")

    # Vegas parameters 
    parser.add_argument("--neval", type=int, help="number of evaluations of integrand")
    parser.add_argument("--nint", type=int, help="number of adaptive iterations")
    parser.add_argument("--neval_warmup", type=int, help="number of evaluations of integrand in warmup")
    parser.add_argument("--nint_warmup", type=int, help="number of adaptive iterations in warmup")

    # program options
    parser.add_argument("--pandas", help="If true, prints events in .npy files", action="store_true")
    parser.add_argument("--numpy", help="If true, prints events in .npy files", action="store_true")
    parser.add_argument("--hepevt", help="If true, unweigh events and print them in HEPEVT-formatted text files", action="store_true")
    parser.add_argument("--hepevt_unweigh", help="unweigh events when printing in HEPEVT format (needs large statistics)", action="store_true")
    parser.add_argument("--hepevt_events", type=int, help="number of events to accept in HEPEVT format")

    parser.add_argument("--summary_plots", help="generate summary plots of kinematics", action="store_false")
    parser.add_argument("--path", help="path where to save run's outputs")

    kwargs = vars(parser.parse_args())

    gen_object = GenLauncher(**kwargs)
    gen_object.run(
        log=kwargs.get("log", DEFAULTS.log),
        verbose=kwargs.get("verbose", DEFAULTS.verbose),
        logfile=kwargs.get("logfile", DEFAULTS.logfile),
        path=kwargs.get("path", DEFAULTS.path)
    )

if __name__ == "__main__":
    dn_gen()