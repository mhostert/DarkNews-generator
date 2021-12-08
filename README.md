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

Event generator for dark neutrino events (in progress).

## **SETUP**

Currently set for local pip installation. From the main folder,

`python3 -m pip install -e .`

Requirements:
* numpy
* scipy
* cython
* [vegas](https://pypi.org/project/vegas/)
* [scikit-hep](https://github.com/scikit-hep/)
* [Particle](https://github.com/scikit-hep/particle)

---
## **USAGE**

To generate 100 HEPevt events with a 4-th neutrino mass of `m_4` and a Zprime mass of `m_zprime` you can simply run

`dn_gen --mzprime=1.25 --m4=0.140 --neval=10000 --D_or_M=dirac --log=INFO --hepevt --hepevt_events=100 --exp=microboone `

By default, the generator prints all events to a pandas dataframe:

1) a pandas dataframe file (by default)
> data/microboone/3plus1/m4_0.14_mzprime_1.25/pandas_df.pckl

but the `--hepevet` flag will also print a HEPEVT file with the specified number of events in `hepevt_events`:

2) a HEPEVT file
> data/microboone/3plus1/m4_0.14_mzprime_1.25/HEPevt.dat


To print unweighted events, use `--unweigh`. In this case, the number of `neval >> hepevt_events` in order for the accept-reject method to be successful.

***
## **OPTIONS**
    -h, --help            show this help message and exit

### Physics args

#### dark sector spectrum
    --mzprime MZPRIME     Z' mass (default: 1.25)
    --m4 M4               mass of the fourth neutrino (default: 0.14)
    --m5 M5               mass of the fifth neutrino (default: None)
    --m6 M6               mass of the sixth neutrino (default: None)
    --D_or_M {dirac,majorana}
                    D_or_M: dirac or majorana (default: majorana)

#### mixings
    --ue4 UE4             Ue4 (default: 0.0)
    --ue5 UE5             Ue5 (default: 0.0)
    --ue6 UE6             Ue6 (default: 0.0)

    --umu4 UMU4           Umu4 (default: 0.0016201851746019652)
    --umu5 UMU5           Umu5 (default: 0.003391164991562634)
    --umu6 UMU6           Umu6 (default: 0.0)

    --utau4 UTAU4         Utau4 (default: 0)
    --utau5 UTAU5         Utau5 (default: 0)
    --utau6 UTAU6         Utau6 (default: 0)

    --ud4 UD4             UD4 (default: 1.0)
    --ud5 UD5             UD5 (default: 1.0)
    --ud6 UD6             UD6 (default: 1.0)

#### couplings
    --gD GD               U(1)_d dark coupling (default: 1.0)
    --alphaD ALPHAD       U(1)_d alpha_dark = (g_dark^2 /4 pi) (default: None)

    --epsilon EPSILON     epsilon^2 (default: 0.01)
    --epsilon2 EPSILON2   epsilon^2 (default: None)

    --alpha_epsilon2 ALPHA_EPSILON2 alpha_QED*epsilon^2 (default: None)
    --chi CHI             chi (default: None)


#### experiment
    --exp {miniboone_fhc,microboone,minerva_le_fhc,minerva_me_fhc,minos_le_fhc,minos_me_fhc,nova_le_fhc,nd280_fhc} experiment (default: miniboone_fhc)

#### monte-carlo scope
    --nopelastic          do not generate proton elastic events (default: False)
    --nocoh               do not generate coherent events (default: False)

    --noHC                do not include helicity conserving events (default: False)
    --noHF                do not include helicity flipping events (default: False)


---
### Code behavior args

#### verbose
    --log {ERROR,WARNING,INFO,DEBUG}  Logging level (default: INFO)
    --verbose             Verbose for logging (default: False)
    --logfile LOGFILE     Path to logfile. If not set, use std output. (default: None)

#### vegas integration arguments
    --neval NEVAL         number of evaluations of integrand (default: 10000)
    --nint NINT           number of adaptive iterations (default: 20)

    --neval_warmup NEVAL_WARMUP number of evaluations of integrand in warmup (default: 1000)
    --nint_warmup NINT_WARMUP number of adaptive iterations in warmup (default: 10)

#### output format options
    --pandas              If true, prints events in .npy files (default: True)

    --numpy               If true, prints events in .npy files (default: False)

    --hepevt              If true, unweigh events and print them in HEPEVT-formatted text files (default: False)
    --hepevt_unweigh      unweigh events when printing in HEPEVT format (needs large statistics) (default: False)
    --hepevt_events HEPEVT_EVENTS number of events to accept in HEPEVT format (default: 100)

    --path PATH           path where to save run's outputs (default: )
