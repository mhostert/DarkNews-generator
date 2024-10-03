<h1 align="center"> DarkNews </h1> <br>

<img align="left" src="https://raw.githubusercontent.com/mhostert/DarkNews-generator/main/src/DarkNews/include/assets/logo.svg" width="180" title="DarkNews-logo">
DarkNews is an event generator for new physics processes at accelerator neutrino experiments that simulates neutrino upscattering to heavy neutral leptons and their subsequent decays to single photons and di-lepton pairs.

![Tests](https://github.com/mhostert/DarkNews-generator/actions/workflows/tests.yml/badge.svg) [![CodeCov](https://codecov.io/gh/mhostert/DarkNews-generator/graph/badge.svg?branch=master)](https://codecov.io/gh/mhostert/DarkNews-generator/?branch=master) [![InspireHEP](https://img.shields.io/badge/InspireHEP-Abdullahi:2207.04137-dodgerblue.svg)](https://arxiv.org/abs/2207.04137)
<!-- <br>[![License: MIT](https://img.shields.io/badge/License-MIT-deeppink.svg)](https://opensource.org/licenses/MIT) -->

<br>

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

---

## Introduction

DarkNews uses Vegas to generate weighted Monte Carlo samples of scattering and decay processes. Differential observables are implemented using analytical expressions with arbitrary interaction vertices, which are then specified at run-time based on the available models and the user's parameter choices. Processes involving heavy neutrinos N are calculated including contributions from the Standard Model Z and W bosons, as well as from a kinetically-mixed dark photon (Z'), and neutrino-N transition magnetic moments. DarkNews also computes the partial decay widths of heavy neutrinos for models with up to 3 heavy neutrinos.

Experiments as well as models are implemented on a case-by-case basis. The necessary ingredients to simulate upscattering or decay-in-flight rates are the active-neutrino flux, detector material, and geometry.

The full information of the event genration is saved to a pandas dataframes, but the user may also choose to print events to numpy ndarrays, as well as to HEPevt-formatted text files.

If you experience any problems or bugs, either open a new issue or contact <mhostert@pitp.ca>.

## Dependencies

Required dependencies:

- [Python](http://www.python.org/) 3.8 or above
- [NumPy](http://www.numpy.org/)

The following dependencies (if missing) will be automatically installed during the main installation of the package:

- [scipy](https://scipy.org/)
- [pandas](https://pandas.pydata.org/) 1.0 or above
- [Cython](https://cython.org/)
- [vegas](https://pypi.org/project/vegas/) 5.1.1 or above
- [Particle](https://pypi.org/project/particle/)
- [pyparsing](https://github.com/pyparsing/pyparsing/)
- [dill](https://pypi.org/project/dill/)
- [matplotlib](https://matplotlib.org/)

Additional optional dependencies for extras:

DarkNews[parquet]
- [pyarrow](https://arrow.apache.org/docs/python/index.html)

DarkNews[pyhepmc]
- [pyhepmc](https://github.com/scikit-hep/pyhepmc)

---

## Installation

DarkNews is available on PyPI so from your python3.8+ (virtual environment or otherwise), run

```sh
python3 -m pip install DarkNews
```

or if your pip version is already set to your preferred python version, simply ```pip install DarkNews```. This should install all dependencies for you. Installing DarkNews on a virtual environment (using `conda` for instance) can avoid several pitfalls, including issues with the `Cython` extension.

**troubleshooting**: If you have any problems, try creating a brand new (conda or pyenv) environment, install the latest version of ```pip```, then pip install numpy first, and only then try to install pip install DarkNews.

If you experience any issues installing pyhepmc-ng, try installing numpy first, and then install pyhepmc-ng directly from Git using the following command: `pip install git+https://git@github.com/scikit-hep/pyhepmc@master`.
Then pip install DarkNews.

If your installation is successful, you should be able to

- import the module from your python scripts or notebook with `import DarkNews`
- run the command line tool `dn_gen` to generate events on the terminal.

##### Editable mode / Developing

To make changes to the package or contribute, you can clone the latest repository from GitHub, and from the main folder of the cloned directory, run:

```sh
python3 -m pip install -e .
```

The package will be installed locally in editable mode.  Any changes to your local files in the repo will be automatically propagated to your DarkNews installation (except setup configurations). You may want to use Jupyter notebooks with `auto reload`.

If you experience any problems with pip, you can resort to a local manual installation. After downloading the repository, from the main folder, you can run

```sh
python3 setup.py develop
```

to install it in developer mode (similar to editable mode above).

##### Extras

If you would like to output events to `.parquet` files, you can install the following ```pip install DarkNews[parquet]``` or ```pip install "DarkNews[parquet]"```.

Similarly, to output events to any `hepmc2/3` or `hepevt` formtats, you can install both extras via ```pip install DarkNews[parquet,pyhepmc]``` or ```pip install "DarkNews[parquet,pyhepmc]"```.

---

## Usage

A lot of information on the usage of the generator is provided by the example Jupyter notebooks in the repository.

You can download example notebooks from the repository's folder `examples/`, or simply run

- ```dn_get_examples```
to download the examples folder in the current directory (requires git version >=2.19).

The main usage of DarkNews is covered in depth in the notebook **`Example_0_start_here.ipynb`**.
We encourage you to read through the cells of the notebook.

#### Command line functionality

It is possible to run the generator through the script `bin/dn_gen`, passing the parameters as options.

```sh
dn_gen --mzprime=1.25 --m4=0.140 --neval=1000 --HNLtype=dirac --loglevel=INFO
```

Run `dn_gen --help` to inspect the meaning of each parameter.

#### Scripting functionality

It is possible to run the generator by creating an instance of the `DarkNews.GenLauncher.GenLauncher` class and calling its `run` method.

```python
from DarkNews import GenLauncher
gen_object = GenLauncher(mzprime=1.25, m4=0.140, neval=1000, HNLtype="dirac")
gen_object.run(loglevel="INFO")
```

The parameters are passed directly while instantiating the `GenLauncher` object.
Some parameters (`loglevel`, `verbose`, `logfile`, `path`, `overwrite_path`) related to the run itself can be passed also within the call of the `run()` method.

---

## Output

### Generated events dataframe

If the argument `pandas = True` (as it is by default), a dataframe containing the generated events is saved in the directory tree.

The dataframe is multi-indexed, where the second column is empty, then there is only the first index.  
All dataframes contain the following process:

$\nu$ (`P_projectile`) + $\mathcal{H}$ (`P_target`) $\to$ $N_{\rm i}$ (`P_decay_N_parent`) + $\mathcal{H}$ (`P_recoil`)

which can be followed by a decay into di-leptons if `decay_product=e+e-` or `decay_product=mu+mu-`:

$N_i$ (`P_decay_N_parent`) $\to$ $N_j$(`P_decay_N_daughter`) + $\ell^+$ (`P_decay_ell_plus`) + $\ell^-$ (`P_decay_ell_minus`)

where $\ell = \{e, \mu\}$, or if `decay_product='photon'`:

$N_i$ (`P_decay_N_parent`) $\to$ $N_j$(`P_decay_N_daughter`) + $\gamma$ (`P_decay_photon`). Only one of the above decays is allowed per generation.

Here follows a complete description of the pandas dataframe:

| **Column**            | **Subcolumn** |**type**  | **description**|
|:--------------------------|:--------:|:--------:|:-----------------------------------|
| **P\_projectile**         | 0, 1, 2, 3  | `float`  | 4-momenta of beam neutrino |
| **P\_decay\_N\_parent**   | 0, 1, 2, 3  | `float`  | 4-momenta of HNL\_parent |
| **P\_target**             | 0, 1, 2, 3  | `float`  | 4-momenta of nucleus |
| **P\_recoil**             | 0, 1, 2, 3  | `float`  | 4-momenta of recoiled nucleus |
| **P\_decay\_photon**      | 0, 1, 2, 3  | `float`  | 4-momenta of photon (if it exists)|
| **P\_decay\_ell\_minus**  | 0, 1, 2, 3  | `float`  | 4-momenta of e- (if it exists)|
| **P\_decay\_ell\_plus**   | 0, 1, 2, 3  | `float`  | 4-momenta of e+ (if it exists)|
| **P\_decay\_N\_daughter** | 0, 1, 2, 3  | `float`  | 4-momenta of HNL\_daughter / nu\_daughter |
| **pos_scatt**             | 0, 1, 2, 3  | `float`  | upscattering position|
| **pos_decay**             | 0, 1, 2, 3  | `float`  | decay position of primary particle (N\_parent) -- no secondary decay position is saved. |
| **w\_decay\_rate\_0**     | <!-- --> | `float`  | Weight of the decay rate of primary unstable particle: &Sigma;<sub>i</sub> w<sub>i</sub> = &Gamma;<sub>N</sub> |
| **w\_decay\_rate\_1**     | <!-- --> | `float`  | Weight of the decay rate of secondary unstable particle: &Sigma;<sub>i</sub> w<sub>i</sub> = &Gamma;<sub>X</sub> |
| **w\_event\_rate**        | <!-- --> | `float`  | Weight for the event rate: &Sigma;<sub>i</sub> w<sub>i</sub> = event rate |
| **w\_flux\_avg\_xsec**    | <!-- --> | `float`  | Weight of the flux averaged cross section: &Sigma;<sub>i</sub> w<sub>i</sub> = int(sigma &sdot; flux) &sdot; exposure |
| **target**                | <!-- --> | *string* | Name of the target object, it will typically be a nucleus |
| **target\_pdgid**         | <!-- --> | `int`    | PDG id of the target |
| **scattering\_regime**    | <!-- --> | *string* | Regime can be coherent or p-elastic |
| **helicity**              | <!-- --> | *string* | Helicity process: can be flipping or conserving; flipping is suppressed |
| **underlying\_process**   | <!-- --> | *string* | String of the underlying process, e.g, "nu(mu) + proton_in_C12 -> N4 +  proton_in_C12 -> nu(mu) + e+ + e- + proton_in_C12" |

#### Additional Attributes

The pandas DataFrame also contains several useful information in `attrs`. They can be accessed via

```python
df.attrs['foo']
```

and are specific to the generation run. The attributes are detailed below:

| **Attrs**            | **type**  | **description**|
|:--------------------------|:--------:|:-----------------------------------|
| **experiment**         | `DarkNews.detector.Detector()`  | The experiment class of DarkNews containing all information on the experimental setup, including neutrino fluxes, targets, exposure, and geometry (if implemented). |
| **model**   | `DarkNews.model.HNLModel()`  | The model class of DarkNews with all the couplings and vertices calculated from the user input. |
| **data_path**   | *string*  | The path used to save the generation outputs. |
| **N{i}_ctau0**   | *float*  | The proper decay length of the i-th HNL in *centimeters* used in the generation of events, with `i`={4,5,6}. |

---

## Input

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

##### Experiment

|<!-- -->|<!-- -->|<!-- -->|<!-- -->|
|:--------|:--------:|:-----------------------------------------------------------------|------------------:|
| **experiment** | `string` | The experiment to consider: see [this section](#the-experiments) | `"miniboone_fhc"` |

##### Monte-Carlo scope

|<!-- -->|<!-- -->|<!-- -->|<!-- -->|
|:---------------|:------:|:------------------------------------------|--------:|
| **nopelastic** | `bool` | Do not generate proton elastic events     | `False` |
| **nocoh**      | `bool` | Do not generate coherent events           | `False` |
| **noHC**       | `bool` | Do not include helicity conserving events | `False` |
| **noHF**       | `bool` | Do not include helicity flipping events   | `False` |
| **decay_products** | `["e+e-","mu+mu-","photon"]` | Decay process of interest | "e+e-" |
| **enforce_prompt** | `bool` | Force particles to decay promptly | `False` |
| **nu_flavors**     | `["nu_e","nu_mu","nu_tau","nu_e_bar","nu_mu_bar","nu_tau_bar"]` | Projectile neutrino | `["nu_mu"]` |
| **sample_geometry**     | `sample_geometry` | Whether to sample the detector geometry using DarkNews. If False or a geometry function is not found, the upscattering position is assumed to be (0,0,0,0). | `True` |

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
| **seed**         | `int` | numpy random number generator seed used in vegas      | None    |

##### Output formats

|<!-- -->|<!-- -->|<!-- -->|<!-- -->|
|:------------------|:--------:|:-----------------------------------------------------------------------|--------:|
| **pandas**        | `bool`   | Save `pandas.DataFrame` as `.pckl` file                                | `True`  |
| **parquet**       | `bool`   | Save `pandas.DataFrame` as `.parquet` file (engine=pyarrow)            | `False`  |
| **numpy**         | `bool`   | Save events in `.npy` files                                            | `False` |
| **hepevt**        | `bool`   | If true, print events to HEPEVT-formatted text files (does not save event weights) | `False` |
| **hepevt_legacy**        | `bool`   | If true, print events to a legacy HEPEVT format (saving weights next to the number of particle in the event and without linebreaks in particle entries) | `False` |
| **hepmc2** | `bool`   | If true, prints events to HepMC2 format. | `False` |
| **hepmc3** | `bool`   | If true, prints events to HepMC3 format. | `False` |
| **hep_unweight** | `bool`   | Unweigh events when printing in HEPEVT format (needs large statistics) | `False` |
| **unweighted_hep_events** | `int`   | number of unweighted events to accept in any of the standard HEP formats. Has to be much smaller than neval for unweight procedure to work. | 100 |
| **sparse**        | `int`   | Specify the level of sparseness of the internal dataframe and output. Not supported for HEPevt. Allowed values are 0--3, where: </br> `0`: keep all information, including event-by-event descriptions; </br> `1`: keep all particle momenta, scattering and decay positions, and all weights; </br> `2`: keep only neutrino energy (or 4-momentum in HEPMC/EVT), visible decay products and unstable particle momenta, scattering and decay positions, and all weights; </br> `3`: keep only neutrino energy (or 4-momentum in HEPMC/EVT), visible decay products and unstable particle momenta, and all weights; </br> `4`: keep only neutrino energy (or 4-momentum in HEPMC/EVT), unstable particle momenta, visible decay products momenta, and w_event_rate. </br> Metadata is always kept if output is pickled. | `0`  |
| **path**          | `string` | Path where to save run's outputs                                       | `"./"`  |
| **make_summary_plots** | `bool` | if True, generates summary plots of kinematics in the `path` | `False`  |

### Specify parameters via a file

It is possible to specify the parameters through a file, instead of writing each one as an option for the command line functionality or as an argument of `GenLauncher` instance.
The parameter file should be provided as the argument `param_file` of `GenLauncher` or via the option `--param-file` of the command line interface.

A template file [`template_parameters_file.txt`](examples/template_parameters_file.txt) can be found in [in the `examples` directory](examples/).
In the following there are the main rules to specify the parameters:

- every line should be in the form of an assignment statement

```rst
<variable_name> = <variable_value>
```

- comments can be specified with with `"#"`

- the different parameters can be specified with:
  - string: by encapsulating each string with double or single quotes `"<string>"` or `'<string>'` are equivalent, the escape character is `\` (backslash).
  - booleans: by writing `True` or `False` (it is case insensitive)
  - mathematical expression (which will results in `float` or `int` numbers): see section below
  - lists: by encapsulating them into square brackets, separating each element with a comma; elements can be string, numbers, mathematical expressions or all of them together.
- When using mathematical expression, the following rules should be applied:
  - numbers can be specified as usual: `5` is integer, `5.0` is float (but `5.` will result in an error), `5e10` is the float number 5*10^10
  - `+`, `-`, `*`, `/` are the usual mathematical operators;
  - `^` is used to make powers (do not use `**`);
  - it is possible to use round brackets `(` and `)`
  - `e` (case-insensitive, isolated: not inside float numbers) is understood as python `math.e = 2.718281828`
  - `pi` (case-insensitive) is understood as `math.pi = 3.1415926535`
  - `sin(<expr>)`, `cos(<expr>)`, `tan(<expr>)` are the usual trigonometric functions
  - `exp(<expr>)` is the usual exponentiation
  - `abs(<expr>)` is the absolute value
  - `sgn(<expr>) = -1` if `<expr> < -1e-100`, `+1` if `<expr> > 1e-100`, `0` otherwise
  - `trunc(<expr>)` returns the truncated float `<expr>` to integer
  - `round(<expr>)` is the integer part of the float number `<expr>`
  - `sum(<list>)` will sum each element of the list, returning a float number
  - any other string is intended as a variable which must have been previously defined (the file is scanned from top to bottom)
  - in general it is possible to define any kind of variable, independently on those that will be actually used by the program, following the usual conventions for the variable name (use only letters, digits and underscores). Moreover, it's not possible to name variables after program-defined names, as for example the name of the functions.

##### Example 1

The following lines

```rst
hbar = 6.582119569e-25 # GeV s
c = 299792458.0 # m s^-1
```

will define two variables, named `hbar` and `c` with their values.

##### Example 2

It is possible to write

```rst
a_certain_constant = hbar * c
```

to define a variable named `a_certain_constant` with the value of the product between the pre-defined `hbar` and `c` variables from the example above.

##### Example 3

It is possible to write any kind of possible expression, for example

```rst
a_variable = c^2 * 3.2e-4 / sin(PI/7) + 12 * exp( -2 * abs(hbar) )
```

obtaining a new variable `a_variable` with the value of 66285419633555.3

##### Example 4

The line

```rst
path = "my_directory/projects/this_project"
```

defines the `path` variable, stored as the string `"my_directory/projects/this_project"`.

##### Example 5

The following lines are defining booleans (they are case insensitive), used to set the various switches:

```rst
pandas = True
numpy = false
```

### The experiments

The experiment to use can be specified in two ways through the `experiment` argument (or `--experiment` option accordingly):

1. specifying a keyword for a pre-defined experiment among:
    - DUNE near detector FHC (`"dune_nd_fhc"`)
    - DUNE near detector RHC (`"dune_nd_rhc"`)
    - SBND detector (`"sbnd"`)
    - SBND dirt-cylinder (`"sbnd_dirt"`)
    - SBND dirt-cone (`"sbnd_dirt_cone"`)
    - MicroBooNE detector (`"microboone"`)
    - MicroBooNE detector TPC volume only (`"microboone_tpc"`)
    - MicroBooNE dirt-cone (`"microboone_dirt"`)
    - MINERvA detector low-energy NuMI FHC (`"minerva_le_fhc"`)
    - MINERvA detector medium-energy NuMI FHC (`"minerva_me_fhc"`)
    - MINERvA detector medium-energy NuMI RHC (`"minerva_me_rhc"`)
    - MiniBooNE detector FHC (`"miniboone_fhc"`)
    - MiniBooNE dirt-cone FHC (`"miniboone_fhc_dirt"`)
    - ICARUS detector (`"icarus"`)
    - ICARUS dirt-cone (`"icarus_dirt"`)
    - MiniBooNE detector RHC (`"miniboone_rhc"`)
    - MiniBooNE dirt-cone RHC (`"miniboone_rhc_dirt"`)
    - MINOS low-energy NuMI FHC (`"minos_le_fhc"`)
    - T2K ND280 detector FHC (`"nd280_fhc"`)
    - NOVA low-energy NuMI FHC (`"nova_le_fhc"`)
    - FASERnu detector (`"fasernu"`)
    - NuTeV FHC (`"nutev_fhc"`)


2. specifying the file path of an experiment file: every file should be specified using the same rules as for the parameters file, listed in [the previous section](#specify-parameters-via-a-file).
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

#### Detector geometries

The only geometries currently implemented in DarkNews are:

- MiniBooNE -- a sphere with origin (0,0,0,0) and radius 574.6 cm.
- SBND -- a box with sides 4m x 4m x 5m.
- MicroBooNE -- the geometry of the cryostat. This is just a junction of a tube with two spherical caps.

Times of interactions are always set to 0, and any additional delay due to the N lifetime is taken in to account. All other experiments spit out events with spatial coordinates (0,0,0).

### Additional information on the generator engine

DarkNews relies on vegas to integrate and sample differential cross sections and decay rates.
The main point of contact with vegas is through the ```vegas.Integrator``` class, which will receive the DarkNews integrands (e.g. ```DarkNews.integrands.UpscatteringHNLDecay()```), whose ```__call__()``` method will compute the differential rates.

It is possible to set a seed for numpy's random number with the ```--seed``` argument, which accepts integer values. This integer seed is then passed to ```numpy.random.default_rng()```, which will then be used by ```vegas``` as its random number generator. By default, vegas uses numpy's random number generator ``numpy.random.random()```, which is based on the Mersenne Twister pseudo-random number generator method.

The reproducibility of the generator output (i.e., same vegas samples) is only guaranteed for the same parameters and number of integrand evaluations, which effectively means that the user has to specify the same scope, model parameters, as well as the same number of neval, nint, neval_warmup and nint_warmup.
