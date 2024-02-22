#!/usr/bin/env python3

import argparse
from DarkNews.GenLauncher import GenLauncher
import DarkNews.parsing_tools as pt
import os
import subprocess
import shutil
from pathlib import Path


def dn_gen():
    DEFAULTS = GenLauncher(loglevel="ERROR")

    # --------------
    # use argument_default=argparse.SUPPRESS so no defaults attributes are instantiated in the final Namespace
    parser = argparse.ArgumentParser(
        description="Generate upscattering events", formatter_class=argparse.ArgumentDefaultsHelpFormatter, argument_default=argparse.SUPPRESS
    )

    # BSM parameters common to all upscattering
    pt.add_common_bsm_arguments(parser, DEFAULTS)
    # Three portal model arguments
    pt.add_three_portal_arguments(parser, DEFAULTS)
    # Generic model arguments
    pt.add_generic_bsm_arguments(parser, DEFAULTS)
    # scope of the physical process of interest
    pt.add_scope_arguments(parser, DEFAULTS)
    # Monte Carlo arguments
    pt.add_mc_arguments(parser, DEFAULTS)

    kwargs = vars(parser.parse_args())

    gen_object = GenLauncher(**kwargs)
    gen_object.run(
        loglevel=kwargs.get("loglevel", gen_object.loglevel),
        verbose=kwargs.get("verbose", gen_object.verbose),
        logfile=kwargs.get("logfile", gen_object.logfile),
    )


def dn_get_examples():

    REPO_NAME = "DarkNews-generator"
    REPO_URL = f"https://github.com/mhostert/{REPO_NAME}.git"
    EXAMPLES_FOLDER = Path("examples")
    DESTINATION_FOLDER = Path("../DarkNews-examples")

    download_repo_cmd = f"git clone --depth 1 --filter=blob:none --sparse {REPO_URL}"
    checkout_cmd = f"git sparse-checkout set {EXAMPLES_FOLDER}"

    print("Using git clone method...")
    with subprocess.Popen([download_repo_cmd], stdout=subprocess.PIPE, shell=True) as proc:
        print(download_repo_cmd)
        print(proc.stdout.read())
    print("git clone successful...")

    os.chdir(f"{Path(REPO_NAME)}")

    with subprocess.Popen([checkout_cmd], stdout=subprocess.PIPE, shell=True) as proc:
        print(checkout_cmd)
        print(proc.stdout.read())
    print("git sparse-checkout successful...")

    os.rename(f"./{EXAMPLES_FOLDER}", DESTINATION_FOLDER)

    os.chdir(Path("../"))
    shutil.rmtree(REPO_NAME)

    print(f"Successfully downloaded examples folder in current directory: {DESTINATION_FOLDER}")


if __name__ == "__main__":
    dn_gen()
