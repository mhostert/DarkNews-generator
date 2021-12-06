```
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

Event generator for dark neutrino events -- in progress

## SETUP

Requirements:
* numpy
* scipy
* cython
* [VEGAS](https://pypi.org/project/vegas/)
* [Particle](https://github.com/scikit-hep/particle)




## USAGE

To generate N HEPevt events with a 4-th neutrino mass of `m_4` and a Zprime mass of `m_zprime` you can simply run

`python3 dark_gen.py --mlight 0.42 --mzprime 0.03 --nevents N --noplot --exp microboone`

This will create a file "data/uboone/3plus1/m4_0.42_mzprime_0.03/MC_m4_0.42_mzprime_0.03.dat" for m_zprime=0.03 GeV and m_4 = 0.420 GeV, and will contain N events in HEPevt format.

## Options
  -h, --help            show this help message and exit


---
### Physics args
##### dark sector spectrum

    --D_or_M {dirac,majorana}
                        D_or_M: dirac or majorana

    --mzprime MZPRIME     Z' mass
    --m4 M4               mass of the fourth neutrino
    --m5 M5               mass of the fifth neutrino
    --m6 M6               mass of the sixth neutrino

##### mixings
    --ue4 UE4             Ue4
    --ue5 UE5             Ue5
    --ue6 UE6             Ue6
    --umu4 UMU4           Umu4
    --umu5 UMU5           Umu5
    --umu6 UMU6           Umu6
    --utau4 UTAU4         Utau4
    --utau5 UTAU5         Utau5
    --utau6 UTAU6         Utau6
    --ud4 UD4             UD4
    --ud5 UD5             UD5
    --ud6 UD6             UD6

##### couplings
    --gD GD               U(1)_d dark coupling
    --alphaD ALPHAD       U(1)_d alpha_dark = (g_dark^2 /4 pi)
    --epsilon EPSILON     epsilon^2
    --epsilon2 EPSILON2   epsilon^2
    --alpha_epsilon2 ALPHA_EPSILON2
                        alpha_QED*epsilon^2
    --chi CHI             chi


##### experiment
    --exp {minerva_le,minerva_me,miniboone,microboone}
                        experiment

##### monte-carlo scope
    --nopelastic          do not generate proton elastic events
    --nocoh               do not generate coherent events

    --includeHF           include helicity flipping events


---
## Code behavior args
##### verbose
    --log LOG             Logging level
    --verbose             Verbose for logging
    --logfile LOGFILE     Path to logfile. If not set, use std output.

##### vegas integration arguments
    --neval NEVAL         number of evaluations of integrand
    --nint NINT           number of adaptive iterations
    --neval_warmup NEVAL_WARMUP
                        number of evaluations of integrand in warmup
    --nint_warmup NINT_WARMUP
                        number of adaptive iterations in warmup

##### output format options
    --pandas              If true, prints events in .npy files
    --numpy               If true, prints events in .npy files
    --hepevet             If true, unweigh events and print them in HEPEVT-formatted text files
    --hepevt_events HEPEVT_EVENTS
                        number of events to accept in HEPEVT format

    --summary_plots       generate summary plots of kinematics

    --path PATH           path where to save run's outputs
