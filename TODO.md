# Progress Summary

See also the project page: https://github.com/mhostert/nu2HNLscatt/projects/1

## open issues

1. in accordance to [PEP8](https://pep8.org/#naming-conventions), class names should be Capitalized and module names (e.g. dark_news) should not contain underscores. Let's rename it to *blank*
	a) Constants still not following guide -- it seems overkill to define so many constants with UPPERCASE format.
	b) We may want to use the sci-kit particle package for constants, but then units are MeV... can we choose units? Or define new particles??

2.  Expressions in model.py of `ce4`, `cmu4`, `ctau4`,`ce5`, `cmu5`, `ctau5`,`ce6`, `5mu6`, `ctau6`,`de4`, `dmu4`, `dtau4`,`de5`, `dmu5`, `dtau5`,`de6`, `dmu6`, `dtau6` do not seem consistent with one another.

	a) yes, need to check that. Asli was looking into it some time ago. We do not currently use them, but we will want something like that in the future.

3.	Decay module needs to be standardized. Currently it looks too complicated and is all channels are hard-coded. We need to be 	cleverer about the number of HNLs and allowed channels. Creat a dictionary which we can loop over to add new channels?
	
	a) I have coded something like that as part of the model definition, but needs a lot of improvement. Do we want to compute potentially expensive decay rates every time we load a model class? I think it is okay.

4. Currently computing the total decay integral several times, and then dividing by the number of cases. This seems prone to errors. What happens when one of the cases has a different decay chain? Need to find a better way to merge the final decay rate integrals.


## new features implemented

* Matheus) I added some nuclear physics functions and data. nuclear_tools stores all.
* Matheus) Integrands now call external functions for computing the particle decays.
* Matheus) New implementation of phase space. Completely general functions in phase_space.
* Matheus) removed a lot of redundant definitions in integrands.

* Matheus) going back to using separate integrands for coh and p-elastic instead of a vector integrand. VEGAS optimizes for a single componenet (the first) of the vector integrand. So if we pass coh first, it will do a poor job of the p-elastic cross section. Situation is a bit different for the decay rates as they depend on a set of different variables.

* Matheus) Using PDG codes everywhere now. Still having to convert MeV to GeV in every in every use of particle.mass
* Matheus) Definition of new particles now done within the Particle module

* Matheus) Nuclear targets are now the main scattering target. Scattering in bound nucleons now supported via an inner class of NuclearTarget. It may be tricky to implement other regimes this way, like RES, DIS...



# To-do stuff

### Matheus

* implement antineutrino scattering (flip sign of axial-vector leptonic vertices)

### Asli

* verify high level parameters in model.py?