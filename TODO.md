# Progress Summary

<!-- See also the project page: https://github.com/mhostert/DarkNews-generator/ -->

## open issues

1. in accordance to [PEP8](https://pep8.org/#naming-conventions), class names should be Capitalized and module names (e.g. dark_news) should not contain underscores. Let's rename it to *blank*
	a) Constants still not following guide -- it seems overkill to define so many constants with UPPERCASE format.
	b) We may want to use the sci-kit particle package for constants, but then units are MeV... can we choose units? Or define new particles?? 
		* THere is also a scikit-hep package for units... maybe we can use that.

2. Do we want to compute potentially expensive decay rates every time we load a model class? If yes, we would handle the BR's as well in the code. This is interesting because of the Ni -> N N N decay computation.

3. Currently computing the total decay integral several times, and then dividing by the number of cases. This seems prone to errors. What happens when one of the cases has a different decay chain? Need to find a better way to merge the final decay rate integrals.

4. We have some support for mass mixing and dipole scattering, but the model files do not know about these parameters. We need to include support for that.

## new features implemented

* Matheus) Same integrand function for light and heavy Z'.

# To-do stuff

### Matheus

* implement antineutrino scattering (flip sign of axial-vector leptonic vertices)

* propagate the lifetime parameters to geometry module.
