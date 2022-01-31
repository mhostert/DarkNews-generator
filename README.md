<h1 align="center"> Dark News </h1> <br>

![alt text](https://github.com/mhostert/DarkNews-generator/logo.png?raw=True "DarkNews logo")

<!-- ```
    ###############################################################
    #    ______           _        _   _                          #
    #    |  _  \         | |      | \ | |                         #
    #    | | | |__ _ _ __| | __   |  \| | _____      _____        #
    #    | | | / _  | ___| |/ /   | .   |/ _ \ \ /\ / / __|       #
    #    | |/ / (_| | |  |   <    | |\  |  __/\ V  V /\__ \       #
    #    |___/ \__,_|_|  |_|\_\   \_| \_/\___| \_/\_/ |___/       #
    #                                                             #
    ###############################################################
```
---
-->

*Here place the relevant badges*

## Table of Contents

- [Introduction](#introduction)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Usage](#usage)
    - [Command line functionality](#command-line-functionality)
    - [Scripting functionality](#scripting-functionality)
    - [List of parameters](#list-of-parameters)
    - [The experiments](#the-experiments)
    - [Generated events dataframe](#generated-events-dataframe)

## Introduction
DarkNews is an event generator for dark neutrino events (in progress).
*write introduction on what's the project aim, its capabilities*

## Dependencies

Required dependencies:
* [Python](http://www.python.org/) 3.6.1 or above
* [NumPy](http://www.numpy.org/)

The following dependencies (if missing) will be automatically installed during the main installation of the package:
* [pandas](https://pandas.pydata.org/) 1.0 or above
* [Cython](https://cython.org/)
* [Requests](https://docs.python-requests.org/en/master/index.html)
* [vegas](https://pypi.org/project/vegas/) 5.1.1 or above
* [Scikit-HEP](https://scikit-hep.org/)
* [Particle](https://pypi.org/project/particle/)
* [dill](https://pypi.org/project/dill/)

## Installation

*Currently set for local `pip` installation*  
To install the package, download the release for the stable version (or clone the repository for the development version).
Save everything in a directory.
Then enter in the main folder and run:
```sh
python3 -m pip install -e . --ignore-installed certifi
```

The package will be installed locally in editable mode.  
The command will take care of installing any missing dependencies.
In any case, it is necessary to have Python 3.6.1 with at least NumPy installed prior to run it.

## Usage

The main usage of DarkNews is covered in depth in the notebook `Example_0_start_here.ipynb` in the `examples` folder.

It is possible to run the generator in two ways.
In both cases, the generated dataset is saved into a directory tree which is created by default in the same folder the generator is run.  
The directory tree has the following form:
```
<path>/data/<exp>/<model_name>/<relevant_masses>_<D_or_M>/
```
where:
* `<path>`: is the value of the `path` argument (or option), default to `./`
* `<exp>`: is the value of the `exp` argument (or option), default set to `miniboone_fhc`
* `<model_name>`: is the name of the chosen model according to the values of the chosen parameters; it can be `3plus1`, `3plus2`, `3plus3`
* `<relevant_masses>`: it is a string made of strings of the kind `"<parameter>_<mass>"` separated by underscores, where `<parameter>` is the name of a mass parameter among `mzprime`, `m4`, `m5`, `m6`; while `<mass>` is a the value, formatted as float, of `<parameter>`
* `<D_or_M>`: is the value of the `D_or_M` argument (or option), default set to `majorana`

### Command line functionality

It is possible to run the generator through the script `bin/dn_gen`, passing the parameters as options.
```sh
dn_gen --mzprime=1.25 --m4=0.140 --neval=1000 --D_or_M=dirac --log=INFO
```
Run `dn_gen --help` to inspect the meaning of each parameter.

### Scripting functionality

It is possible to run the generator by creating an instance of the `DarkNews.GenLauncher.GenLauncher` class and calling its `run` method.
```python
from DarkNews.GenLauncher import GenLauncher
gen_object = GenLauncher(mzprime=1.25, m4=0.140, neval=1000, D_or_M="dirac")
gen_object.run(log="INFO")
```
The parameters are passed directly while instantiating the `GenLauncher` object.
Some parameters (`log`, `verbose`, `logfile`, `path`) related to the run itself can be passed also within the call of the `run()` method.

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
| **D_or_M**  | `["dirac", "majorana"]` | Dirac or majorana           | `"majorana"` |

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
| **log**     | `["INFO", "WARNING", "ERROR", "DEBUG"]` | Logging level                                | `"INFO"` | 
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
| **numpy**         | `bool`   | Save events in `.npy` files                                            | `False` | 
| **hepevt**        | `bool`   | Save events in HEPEVT-formatted text files                             | `False` | 
| **hepvt_unweigh** | `bool`   | Unweigh events when printing in HEPEVT format (needs large statistics) | `False` | 
| **hepvt_events**  | `int`    | Number of events to accept in HEPEVT format                            | 100     | 
| **path**          | `string` | Path where to save run's outputs                                       | `"./"`  | 

### The experiments

The experiments to which pass the generated events are saved as `.json` files, contained in the `src/DarkNews/detectors/` folder.
Currently, the following experiments are defined:
* DUNE FHC ND (`dune_nd_fhc.json`)
* DUNE RHC ND (`dune_nd_rhc.json`)
* MicroBooNE (`microboone.json`)
* MINERVA FHC LE (`minerva_le_fhc.json`)
* MINERVA FHC ME (`minerva_me_fhc.json`)
* MiniBooNE FHC (`miniboone_fhc.json`)
* NUMI FHC ME (`minos_le_fhc.json`)
* NUMI FHC LE (`minos_me_fhc.json`)
* ND280 FHC (`nd280_fhc.json`)
* NOva FHC (`nova_le_fhc.json`)

The experiment can be specified with the filename (without the extension) in the `exp` argument.

It is possible to add user-defined experiments, by creating similar files in the same directory.
A template `user.json` is already present in that folder, in order to explain the various parameters.

|<!-- -->|<!-- -->|<!-- -->|
|:--------------------------------------|:-----------------:|:-----------------------------------------------------------------------------------------------------------|
| **name**                              | `string`          | Name of the experiment                                |
| **fluxfile**                          | `string`          | Path of the fluxes file with respect to the parent directory, it will have the form `fluxes/flux_file.dat` |
| **flux_norm**                         | `float`           | Flux normalization factor: **all fluxes should be normalized so that the units are nus/cm&sup2;/GeV/POT**      |
| **erange**                            | list of `float`   | Neutrino energy range in GeV                                                                               |
| **nuclear_targets**                   | list of `strings` | Detector materials in the form of `"<element_name><mass_number>"` (e.g. `"Ar40"`)                          |
| **fiducial_mass**                     | `float`           | Fiducial mass in tons                                                                                      |
| **fiducial_mass_fraction_per_target** | list of `float`   | Fiducial mass fraction for each target in order, the total sum should be 1                                 |
| **POTs**                              | `float`           | Protons on target                                                                                          |

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
| **P\_decay\_ell\_minus**  | **t**    | `float`  | 4-momenta of e- |
| <!-- -->                  | **x**    | `float`  | <!-- --> |
| <!-- -->                  | **y**    | `float`  | <!-- --> |
| <!-- -->                  | **z**    | `float`  | <!-- --> |
| **P\_decay\_ell\_plus**   | **t**    | `float`  | 4-momenta of e+ |
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
| **w\_decay\_rate\_0**     | <!-- --> | `float`  | Weight of the decay rate: &Sigma;<sub>i</sub> w<sub>i</sub> = &Gamma;<sub>N</sub> |
| **I\_decay\_rate\_0**     | <!-- --> | `float`  | Total rate &Gamma;<sub>N</sub> |
| **w\_event\_rate**        | <!-- --> | `float`  | Weight for the event rate: &Sigma;<sub>i</sub> w<sub>i</sub> = event rate |
| **I\_event\_rate**        | <!-- --> | `float`  | Total event rate |
| **w\_flux\_avg\_xsec**    | <!-- --> | `float`  | Weight of the flux averaged cross section: &Sigma;<sub>i</sub> w<sub>i</sub> = int(sigma &sdot; flux) &sdot; exposure |
| **I\_flux\_avg\_xsec**    | <!-- --> | `float`  | int(sigma &sdot; flux) &sdot; exposure |
| **target**                | <!-- --> | *object* | Target object, it will typically be a nucleus |
| **target\_pdgid**         | <!-- --> | `int`    | PDG id of the target |
| **scattering\_regime**    | <!-- --> | *object* | Regime can be coherent or p-elastic |
| **helicity**              | <!-- --> | *object* | Helicity process: can be flipping or conserving; flipping is suppressed |
| **underlying\_process**   | <!-- --> | *object* | String of the underlying process, e.g, "nu(mu) + proton_in_C12 -> N4 +  proton_in_C12 -> nu(mu) + e+ + e- + proton_in_C12" |
