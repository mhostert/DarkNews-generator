<h1 align="center"> DarkNews </h1> <br>

<!-- <img align="left" src="https://github.com/mhostert/DarkNews-generator/blob/main/logo.png" width="180" title="DarkNews-logo"> -->
<img align="left" src="assets/logo_2.svg" width="180" title="DarkNews-logo">
DarkNews is an event generator for new physics processes at accelerator neutrino experiments based on Vegas. It simulates neutrino upscattering to heavy neutrinos as well as heavy neutrino decays to dileptons via neutrino, vector, and transition magnetic moment portals.


![Tests](https://github.com/mhostert/DarkNews-generator/actions/workflows/tests.yml/badge.svg)
[![InspireHEP](https://img.shields.io/badge/InspireHEP-Abdullahi:xxx2022-dodgerblue.svg)](https://arxiv.org/abs/2202.xxxxx)
<!-- [![Coverage Badge](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/mhostert/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/raw/DarkNews-generator__heads_main.json)] -->
<!-- <br>[![License: MIT](https://img.shields.io/badge/License-MIT-deeppink.svg)](https://opensource.org/licenses/MIT) -->
<!-- [![arXiv](https://img.shields.io/badge/arXiv-2202.xxxxxx%20-violet.svg)](https://arxiv.org/abs/2202.xxxxx) -->


<br>

**Please note this generator is currently under development, so please expect frequent updates.**


## Table of Contents

- [Introduction](#introduction)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Usage](#usage)
    - [Command line functionality](#command-line-functionality)
    - [Scripting functionality](#scripting-functionality)
    - [List of parameters](#list-of-parameters)
    - [Specify parameters via a file](#specify-parameters-via-a-file)
    - [The experiments](#the-experiments)
    - [Generated events dataframe](#generated-events-dataframe)

## Introduction

DarkNews uses Vegas to generate weighted Monte Carlo samples of scattering and decay processes. Differential observables are implemented using analytical expressions with arbitrary interaction vertices, which are then specified at run-time based on the available models and the user's parameter choices. Processes involving heavy neutrinos N are calculated including contributions from the Standard Model Z and W bosons, as well as from a kinetically-mixed dark photon (Z'), and neutrino-N transition magnetic moments. DarkNews also computes the partial decay widths of heavy neutrinos for models with up to 3 heavy neutrinos. 

Experiments as well as models are implemented on a case-by-case basis. The necessary ingredients to simulate upscattering or decay-in-flight rates are the active-neutrino flux, detector material, and geometry.

The full information of the event genration is saved to a pandas dataframes, but the user may also choose to print events to numpy ndarrays, as well as to HEPevt-formatted text files. 

## Dependencies

Required dependencies:
* [Python](http://www.python.org/) 3.6.1 or above
* [NumPy](http://www.numpy.org/)

The following dependencies (if missing) will be automatically installed during the main installation of the package:
* [pandas](https://pandas.pydata.org/) 1.0 or above
* [Cython](https://cython.org/)
<!-- * [Requests](https://docs.python-requests.org/en/master/index.html) -->
* [vegas](https://pypi.org/project/vegas/) 5.1.1 or above
* [Particle](https://pypi.org/project/particle/)
* [dill](https://pypi.org/project/dill/)

## Installation

*Currently set for local `pip` installation*  
To install the package, download the release for the stable version (or clone the repository for the development version).
Save everything in a directory.
Then enter in the main folder and run:
```sh
python3 -m pip install -e .
```

The package will be installed locally in editable mode.  
The command will take care of installing any missing dependencies. It is necessary to have Python 3.6.1.

Alternatively, if you experience any problems with pip, you can resort to a local manual installation. In the main folder of the repo, run
```sh
python3 setup.py install
```
or
```sh
python3 setup.py develop
```
to install it in developer mode (similar to editable mode above).

## Usage

The main usage of DarkNews is covered in depth in the notebook `Example_0_start_here.ipynb` in the `examples` folder.

It is possible to run the generator in two ways.
In both cases, the generated dataset is saved into a directory tree which is created by default in the same folder the generator is run.  
The directory tree has the following form:
```
<path>/data/<exp>/<model_name>/<relevant_masses>_<HNLtype>/
```
where:
* `<path>`: is the value of the `path` argument (or option), default to `./`
* `<exp>`: is the value of the `exp` argument (or option), default set to `miniboone_fhc`
* `<model_name>`: is the name of the chosen model according to the values of the chosen parameters; it can be `3plus1`, `3plus2`, `3plus3`
* `<relevant_masses>`: it is a string made of strings of the kind `"<parameter>_<mass>"` separated by underscores, where `<parameter>` is the name of a mass parameter among `mzprime`, `m4`, `m5`, `m6`; while `<mass>` is a the value, formatted as float, of `<parameter>`
* `<HNLtype>`: is the value of the `HNLtype` argument (or option), default set to `majorana`

### Command line functionality

It is possible to run the generator through the script `bin/dn_gen`, passing the parameters as options.
```sh
dn_gen --mzprime=1.25 --m4=0.140 --neval=1000 --HNLtype=dirac --loglevel=INFO
```
Run `dn_gen --help` to inspect the meaning of each parameter.

### Scripting functionality

It is possible to run the generator by creating an instance of the `DarkNews.GenLauncher.GenLauncher` class and calling its `run` method.
```python
from DarkNews.GenLauncher import GenLauncher
gen_object = GenLauncher(mzprime=1.25, m4=0.140, neval=1000, HNLtype="dirac")
gen_object.run(loglevel="INFO")
```
The parameters are passed directly while instantiating the `GenLauncher` object.
Some parameters (`loglevel`, `verbose`, `logfile`, `path`) related to the run itself can be passed also within the call of the `run()` method.

### List of parameters

This is an exhaustive list of the parameters that can be passed to the program.
They can be passed in the command line mode by prepending `--` to the name.
This summary can also be accessed by running
```sh
dn_gen --help
```
The first column is the name of the parameter, the second is the type or the list of allowed values, the third is a brief explanation and the fourth is the default.
Parameters marked as *internal* can not be specified as they are automatically computed on the basis of other parameters.

#### Physics parameters

##### Dark sector spectrum

|<!-- -->|<!-- -->|<!-- -->|<!-- -->|
|:------------|:-----------------------:|:----------------------------|-------------:|
| **mzprime** | `float`                 | Mass of Z'                  | 1.25         |
| **m4**      | `float`                 | Mass of the fourth neutrino | 0.14         |
| **m5**      | `float`                 | Mass of the fifth neutrino  | `None`       |
| **m6**      | `float`                 | Mass of the sixth neutrino  | `None`       |
| **HNLtype**  | `["dirac", "majorana"]` | Dirac or majorana           | `"majorana"` |

##### Mixings

|<!-- -->|<!-- -->|<!-- -->|<!-- -->|
|:----------|:--------:|:----|----------:|
| **ue4**   | `float`  |     | 0.0       |
| **ue5**   | `float`  |     | 0.0       |
| **ue6**   | `float`  |     | 0.0       |
| **umu4**  | `float`  |     | 0.0016202 |
| **umu5**  | `float`  |     | 0.0033912 |
| **umu6**  | `float`  |     | 0.0       |
| **utau4** | `float`  |     | 0.0       |
| **utau5** | `float`  |     | 0.0       |
| **utau6** | `float`  |     | 0.0       |
| **ud4**   | `float`  |     | 1.0       |
| **ud5**   | `float`  |     | 1.0       |
| **ud6**   | `float`  |     | 1.0       |

##### Couplings

|<!-- -->|<!-- -->|<!-- -->|<!-- -->|
|:-------------------|:-------:|:-------------------------------------------------------------------------------|-----------:|
| **gD**             | `float` | U(1)<sub>d</sub> dark coupling g<sub>D</sub>                                   | 1.0        | 
| **alphaD**         | `float` | U(1)<sub>d</sub> &alpha;<sub>dark</sub> = (g<sub>D</sub><sup>2</sup> / 4 &pi;) | *internal* |
| **epsilon**        | `float` | &epsilon;                                                                      | 0.01       |
| **epsilon2**       | `float` | &epsilon;<sup>2</sup>                                                          | *internal* |
| **alpha_epsilon2** | `float` | &alpha;<sub>QED</sub> &sdot; &epsilon;<sup>2</sup>                             | *internal* |
| **chi**            | `float` | Kinetic mixing parameter                                                       | `None`     |

##### Transition magnetic moment
|<!-- -->|<!-- -->|<!-- -->|<!-- -->|
|:---------------|:--------:|:---------------|----:|
| **mu_tr_e4**       | `float` | TMM mu_tr_e4   | 0.0 |
| **mu_tr_e5**       | `float` | TMM mu_tr_e5   | 0.0 |
| **mu_tr_e6**       | `float` | TMM mu_tr_e6   | 0.0 |
| **mu_tr_mu4**      | `float` | TMM mu_tr_mu4  | 0.0 |
| **mu_tr_mu5**      | `float` | TMM mu_tr_mu5  | 0.0 |
| **mu_tr_mu6**      | `float` | TMM mu_tr_mu6  | 0.0 |
| **mu_tr_tau4**     | `float` | TMM mu_tr_tau4 | 0.0 |
| **mu_tr_tau5**     | `float` | TMM mu_tr_tau5 | 0.0 |
| **mu_tr_tau6**     | `float` | TMM mu_tr_tau6 | 0.0 |
| **mu_tr_44**       | `float` | TMM mu_tr_44   | 0.0 |
| **mu_tr_45**       | `float` | TMM mu_tr_45   | 0.0 |
| **mu_tr_46**       | `float` | TMM mu_tr_46   | 0.0 |
| **mu_tr_55**       | `float` | TMM mu_tr_54   | 0.0 |
| **mu_tr_56**       | `float` | TMM mu_tr_55   | 0.0 |
| **mu_tr_56**       | `float` | TMM mu_tr_66   | 0.0 |
| **decay_products** | `["e+e-","mu+mu-","photon"]` | Decay process of interest | "e+e-" |

##### Experiment

|<!-- -->|<!-- -->|<!-- -->|<!-- -->|
|:--------|:--------:|:-----------------------------------------------------------------|------------------:|
| **exp** | `string` | The experiment to consider: see [this section](#the-experiments) | `"miniboone_fhc"` |

##### Monte-Carlo options

|<!-- -->|<!-- -->|<!-- -->|<!-- -->|
|:---------------|:------:|:------------------------------------------|--------:|
| **nopelastic** | `bool` | Do not generate proton elastic events     | `False` | 
| **nocoh**      | `bool` | Do not generate coherent events           | `False` | 
| **noHC**       | `bool` | Do not include helicity conserving events | `False` | 
| **noHF**       | `bool` | Do not include helicity flipping events   | `False` | 

#### Code behavior options

##### Verbose

|<!-- -->|<!-- -->|<!-- -->|<!-- -->|
|:------------|:---------------------------------------:|:---------------------------------------------|---------:|
| **loglevel**     | `["INFO", "WARNING", "ERROR", "DEBUG"]` | Logging level                                | `"INFO"` | 
| **verbose** | `bool`                                  | Verbose for logging                          | `False`  | 
| **logfile** | `string`                                | Path to log file; if not set, use std output | `None`   | 

##### `vegas` integration arguments

|<!-- -->|<!-- -->|<!-- -->|<!-- -->|
|:-----------------|:-----:|:---------------------------------------------|------:|
| **neval**        | `int` | Number of evaluations of integrand           | 10000 | 
| **nint**         | `int` | Number of adaptive iterations                | 20    | 
| **neval_warmup** | `int` | Number of evaluations of integrand in warmup | 1000  | 
| **nint_warmup**  | `int` | Number of adaptive iterations in warmup      | 10    | 

##### Output formats

|<!-- -->|<!-- -->|<!-- -->|<!-- -->|
|:------------------|:--------:|:-----------------------------------------------------------------------|--------:|
| **pandas**        | `bool`   | Save `pandas.DataFrame` as `.pckl` file                                | `True`  | 
| **parquet**        | `bool`   | Save `pandas.DataFrame` as `.parquet` file (engine=pyarrow)                                | `False`  | 
| **numpy**         | `bool`   | Save events in `.npy` files                                            | `False` | 
| **hepevt**        | `bool`   | Save events in HEPEVT-formatted text files                             | `False` | 
| **hepvt_unweigh** | `bool`   | Unweigh events when printing in HEPEVT format (needs large statistics) | `False` | 
| **hepvt_events**  | `int`    | Number of events to accept in HEPEVT format                            | 100     | 
| **path**          | `string` | Path where to save run's outputs                                       | `"./"`  | 

| **sparse**          | `bool` | if True, save only the neutrino energy, charged lepton or photon momenta, and weights. Not supported for HEPevt.                                       | `False`  | 

### Specify parameters via a file

It is possible to specify the parameters through a file, instead of writing each one as an option for the command line functionality or as an argument of `GenLauncher` instance.
The parameter file should be provided as the argument `param_file` of `GenLauncher` or via the option `--param-file` of the command line interface.

A template file [`template_parameters_file.txt`](examples/template_parameters_file.txt) can be found in [in the `examples` directory](examples/).
In the following there are the main rules to specify the parameters:
* every line should be in the form of an assignment statement
```rst
<variable_name> = <variable_value>
```
* comments can be specified with with `"#"`
* the different parameters can be specified with:
    * string: by encapsulating each string with double or single quotes `"<string>"` or `'<string>'` are equivalent, the escape character is `\` (backslash).
    * booleans: by writing `True` or `False` (it is case insensitive)
    * mathematical expression (which will results in `float` or `int` numbers): see section below
    * lists: by encapsulating them into square brackets, separating each element with a comma; elements can be string, numbers, mathematical expressions or all of them together.
* When using mathematical expression, the following rules should be applied:
    * numbers can be specified as usual: `5` is integer, `5.0` is float (but `5.` will result in an error), `5e10` is the float number 5*10^10
    * `+`, `-`, `*`, `/` are the usual mathematical operators;
    * `^` is used to make powers
    * it is possible to use round brackets `(` and `)`
    * `e` (case-insensitive, isolated: not inside float numbers) is understood as python `math.e = 2.718281828`
    * `pi` (case-insensitive) is understood as `math.pi = 3.1415926535`
    * `sin(<expr>)`, `cos(<expr>)`, `tan(<expr>)` are the usual trigonometric functions
    * `exp(<expr>)` is the usual exponentiation
    * `abs(<expr>)` is the absolute value
    * `sgn(<expr>) = -1` if `<expr> < -1e-100`, `+1` if `<expr> > 1e-100`, `0` otherwise
    * `trunc(<expr>)` returns the truncated float `<expr>` to integer
    * `round(<expr>)` is the integer part of the float number `<expr>`
    * `sum(<list>)` will sum each element of the list, returning a float number
    * any other string is intended as a variable which must have been previously defined (the file is scanned from top to bottom)
    * in general it is possible to define any kind of variable, independently on those that will be actually used by the program, following the usual conventions for the variable name (use only letters, digits and underscores). Moreover, it's not possible to name variables after program-defined names, as for example the name of the functions.

#### Example 1
The following lines
```rst
hbar = 6.582119569e-25 # GeV s
c = 299792458.0 # m s^-1
```
will define two variables, named `hbar` and `c` with their values.

#### Example 2
It is possible to write
```rst
a_certain_constant = hbar * c
```
to define a variable named `a_certain_constant` with the value of the product between the pre-defined `hbar` and `c` variables from the example above.

#### Example 3
It is possible to write any kind of possible expression, for example
```rst
a_variable = c^2 * 3.2e-4 / sin(PI/7) + 12 * exp( -2 * abs(hbar) )
```
obtaining a new variable `a_variable` with the value of 66285419633555.3

#### Example 4
The line
```rst
path = "my_directory/projects/this_project"
```
defines the `path` variable, stored as the string `"my_directory/projects/this_project"`.

#### Example 5
The following lines are defining booleans (they are case insensitive), used to set the various switches:
```rst
pandas = True
numpy = false
```

### The experiments

The experiment to use can be specified in two ways through the `exp` argument (or `--exp` option accordingly):
1. specifying a keyword for a pre-defined experiment among:
    * DUNE FHC ND (`"dune_nd_fhc"`)
    * DUNE RHC ND (`"dune_nd_rhc"`)
    * MicroBooNE (`"microboone"`)
    * MINERVA FHC LE (`"minerva_le_fhc"`)
    * MINERVA FHC ME (`"minerva_me_fhc"`)
    * MiniBooNE FHC (`"miniboone_fhc"`)
    * NUMI FHC ME (`"minos_le_fhc"`)
    * NUMI FHC LE (`"minos_me_fhc"`)
    * ND280 FHC (`"nd280_fhc"`)
    * NOva FHC LE (`"nova_le_fhc"`)
    * FASERnu (`"fasernu"`)
1. specifying the file path of an experiment file: every file should be specified using the same rules as for the parameters file, listed in [the previous section](#specify-parameters-via-a-file).
A template file [`template_custom_experiment.txt`](examples/template_custom_experiment.txt) can be found in [in the `examples` directory](examples/).
The following parameters must be present (in general it is possible to specify any number of parameters, but only the ones below would be relevant).

|<!-- -->|<!-- -->|<!-- -->|
|:-----------------------------|:-----------------:|:-----------------------------------------------------------------------------------------------------------|
| **name**                     | `string`          | Name of the experiment (your are free to use capital letters, when needed)                                 |
| **fluxfile**                 | `string`          | Path of the fluxes file with respect to the experiment file directory                                      |
| **flux_norm**                | `float`           | Flux normalization factor: **all fluxes should be normalized so that the units are nus/cm&sup2;/GeV/POT**  |
| **erange**                   | list of `float`   | Neutrino energy range `[<min>, <max>]` in GeV                                                              |
| **nuclear_targets**          | list of `string`  | Detector materials in the form of `"<element_name><mass_number>"` (e.g. `"Ar40"`)                          |
| **fiducial_mass**            | `float`           | Fiducial mass in tons                                                                                      |
| **fiducial_mass_per_target** | list of `float`   | Fiducial mass for each target in the same order as the `nuclear_targets` parameter                         |
| **POTs**                     | `float`           | Protons on target                                                                                          |

### Generated events dataframe

If the argument `pandas = True` (as it is by default), a dataframe containing the generated events is saved in the directory tree.

The dataframe is multi-indexed, where the second column is empty, then there is only the first index.  
The process described is:

neutrino (P\_projectile) + Nucleus (P\_target) &#8594; HNL\_parent (P\_decay\_N\_parent) + Recoiled Nucleus (P\_recoil)

Followed by:

HNL\_parent (P\_decay\_N\_parent) &#8594; HNL/nu\_daughter (P\_decay\_N\_daughter) + e<sup>-</sup> (P\_decay\_ell\_plus) + e<sup>-</sup> (P\_decay\_ell\_minus)

|<!-- -->|<!-- -->|<!-- -->|<!-- -->|
|:--------------------------|:--------:|:--------:|:-----------------------------------|
| **P\_projectile**         | **t**    | `float`  | 4-momenta of beam neutrino |
| <!-- -->                  | **x**    | `float`  | <!-- --> |
| <!-- -->                  | **y**    | `float`  | <!-- --> |
| <!-- -->                  | **z**    | `float`  | <!-- --> |
| **P\_decay\_N\_parent**   | **t**    | `float`  | 4-momenta of HNL\_parent |
| <!-- -->                  | **x**    | `float`  | <!-- --> |
| <!-- -->                  | **y**    | `float`  | <!-- --> |
| <!-- -->                  | **z**    | `float`  | <!-- --> |
| **P\_target**             | **t**    | `float`  | 4-momenta of nucleus |
| <!-- -->                  | **x**    | `float`  | <!-- --> |
| <!-- -->                  | **y**    | `float`  | <!-- --> |
| <!-- -->                  | **z**    | `float`  | <!-- --> |
| **P\_recoil**             | **t**    | `float`  | 4-momenta of recoiled nucleus |
| <!-- -->                  | **x**    | `float`  | <!-- --> |
| <!-- -->                  | **y**    | `float`  | <!-- --> |
| <!-- -->                  | **z**    | `float`  | <!-- --> |
| **P\_decay\_photon**  | **t**    | `float`  | 4-momenta of the photon (if exists)|
| <!-- -->                  | **x**    | `float`  | <!-- --> |
| <!-- -->                  | **y**    | `float`  | <!-- --> |
| <!-- -->                  | **z**    | `float`  | <!-- --> |
| **P\_decay\_ell\_minus**  | **t**    | `float`  | 4-momenta of e- (if exists)|
| <!-- -->                  | **x**    | `float`  | <!-- --> |
| <!-- -->                  | **y**    | `float`  | <!-- --> |
| <!-- -->                  | **z**    | `float`  | <!-- --> |
| **P\_decay\_ell\_plus**   | **t**    | `float`  | 4-momenta of e+ (if exists)|
| <!-- -->                  | **x**    | `float`  | <!-- --> |
| <!-- -->                  | **y**    | `float`  | <!-- --> |
| <!-- -->                  | **z**    | `float`  | <!-- --> |
| **P\_decay\_N\_daughter** | **t**    | `float`  | 4-momenta of HNL\_daughter / nu\_daughter |
| <!-- -->                  | **x**    | `float`  | <!-- --> |
| <!-- -->                  | **y**    | `float`  | <!-- --> |
| <!-- -->                  | **z**    | `float`  | <!-- --> |
| **decay\_displacement**   | **t**    | `float`  | Distance travelled by N\_parent |
| <!-- -->                  | **x**    | `float`  | <!-- --> |
| <!-- -->                  | **y**    | `float`  | <!-- --> |
| <!-- -->                  | **z**    | `float`  | <!-- --> |
| **w\_decay\_rate\_0**     | <!-- --> | `float`  | Weight of the decay rate of primary unstable particle: &Sigma;<sub>i</sub> w<sub>i</sub> = &Gamma;<sub>N</sub> |
| **w\_decay\_rate\_1**     | <!-- --> | `float`  | Weight of the decay rate of secondary unstable particle: &Sigma;<sub>i</sub> w<sub>i</sub> = &Gamma;<sub>X</sub> |
| **w\_event\_rate**        | <!-- --> | `float`  | Weight for the event rate: &Sigma;<sub>i</sub> w<sub>i</sub> = event rate |
| **w\_flux\_avg\_xsec**    | <!-- --> | `float`  | Weight of the flux averaged cross section: &Sigma;<sub>i</sub> w<sub>i</sub> = int(sigma &sdot; flux) &sdot; exposure |
| **target**                | <!-- --> | *object* | Target object, it will typically be a nucleus |
| **target\_pdgid**         | <!-- --> | `int`    | PDG id of the target |
| **scattering\_regime**    | <!-- --> | *object* | Regime can be coherent or p-elastic |
| **helicity**              | <!-- --> | *object* | Helicity process: can be flipping or conserving; flipping is suppressed |
| **underlying\_process**   | <!-- --> | *object* | String of the underlying process, e.g, "nu(mu) + proton_in_C12 -> N4 +  proton_in_C12 -> nu(mu) + e+ + e- + proton_in_C12" |


### The event generator engine

DarkNews relies on vegas to integrate and sample differential cross sections and decay rates. 
The main point of contact with vegas is through the Integrator class, which will receive the DarkNews integrands
containing the differential rates.

By default, Vegas uses numpy's random number generator, which in turn uses the Mersenne Twister pseudo-random number generator. It is possible to set a seed for numpy's random number generator using numpy.seed().
The reproducibility of the Vegas samples and integral are only guaranteed for the same series and number of calls to numpy.random, which effectively means, the same number of neval, nint, as well as neval_warmup, nint_warmup.
