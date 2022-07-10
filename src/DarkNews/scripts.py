#!/usr/bin/env python3

import argparse
from DarkNews.GenLauncher import GenLauncher
import DarkNews.parsing_tools as pt


def dn_gen():
    DEFAULTS = GenLauncher(loglevel="error")

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


def dn_download_examples():
    import subprocess
    
    user = "jckantor"
    repo = "cbe-virtual-laboratory"
    src_dir = "src"
    pyfile = "hello_world.py"

    url = f"https://raw.githubusercontent.com/{user}/{repo}/main/{src_dir}/{pyfile}"

    result = subprocess.run(["wget", "--no-cache", "--backups=1", url], stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    print(result.stderr.decode("utf-8"))



if __name__ == "__main__":
    dn_gen()
