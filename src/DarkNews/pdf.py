"""Parton distribution function (PDF) interface for the DIS scattering regime.

DarkNews does not ship a PDF set. This module defines a small backend-agnostic
interface so a real set (via LHAPDF, a bundled grid, etc.) can be plugged in, plus
the electromagnetic structure-function combination used by the dipole-portal DIS
cross section.

Design goals:
  * keep the physics code (amplitudes.py) independent of *which* PDF set is used;
  * make the missing-PDF case fail loudly with an actionable message rather than
    silently returning wrong numbers.

Quark flavor convention follows the PDG/LHAPDF parton id:
    1 = d, 2 = u, 3 = s, 4 = c, 5 = b   (negative ids are antiquarks, 21 = gluon)
"""

import logging
from abc import ABC, abstractmethod

import numpy as np

logger = logging.getLogger("logger." + __name__)

# Squared electric charges of the quarks, keyed by |PDG id|.
QUARK_CHARGE2 = {
    1: 1.0 / 9.0,  # d
    2: 4.0 / 9.0,  # u
    3: 1.0 / 9.0,  # s
    4: 4.0 / 9.0,  # c
    5: 1.0 / 9.0,  # b
}

# Active quark flavors summed over in the structure function (add top only if a
# set/scale ever warrants it -- irrelevant at neutrino-telescope energies).
DIS_FLAVORS = (1, 2, 3, 4, 5)


class PDFSet(ABC):
    """Backend-agnostic PDF interface.

    Concrete subclasses only need to implement :meth:`xfxQ2`, returning
    ``x * f(x, Q2)`` for a given parton id, exactly as LHAPDF does. Everything
    else (isospin rotation to the neutron, the EM structure function) is built on
    top of that single method.
    """

    @abstractmethod
    def xfxQ2(self, pid, x, Q2):
        """Return x*f(x, Q2) for parton ``pid``. x, Q2 may be arrays."""
        raise NotImplementedError

    def fxQ2(self, pid, x, Q2):
        """Return the bare density f(x, Q2) = xfxQ2 / x."""
        return self.xfxQ2(pid, x, Q2) / x


class LHAPDFSet(PDFSet):
    """Adapter around an LHAPDF set (optional dependency).

    Example:
        >>> pdf = LHAPDFSet("CT18NNLO")          # requires lhapdf + the set installed
    """

    def __init__(self, name, member=0):
        try:
            import lhapdf
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "LHAPDFSet requires the 'lhapdf' Python bindings and the requested PDF set. "
                "Install LHAPDF (https://lhapdf.hepforge.org/) and the set, or supply a "
                "CallablePDF / custom PDFSet instead."
            ) from exc
        self.name = name
        self._pdf = lhapdf.mkPDF(name, member)

    def xfxQ2(self, pid, x, Q2):
        # LHAPDF's Python API is scalar; vectorize over array inputs.
        xfxQ2 = np.vectorize(self._pdf.xfxQ2, otypes=[float])
        return xfxQ2(int(pid), x, Q2)


class CallablePDF(PDFSet):
    """Wrap any callable ``f(pid, x, Q2) -> x*f(x, Q2)`` as a PDFSet."""

    def __init__(self, func):
        self._func = func

    def xfxQ2(self, pid, x, Q2):
        return self._func(pid, x, Q2)


class UnavailablePDF(PDFSet):
    """Default placeholder used when no PDF set has been provided.

    Any attempt to evaluate it raises, so a DIS calculation cannot silently run
    with a missing/zero PDF. Replace it with an :class:`LHAPDFSet` (or any
    :class:`PDFSet`) via ``GenLauncher(..., dis_pdf=...)`` / the process API.
    """

    def xfxQ2(self, pid, x, Q2):
        raise NotImplementedError(
            "No PDF set configured for the DIS regime. Provide one, e.g. LHAPDFSet('CT18NNLO'), via the `dis_pdf` argument. DarkNews does not bundle PDF sets."
        )


def _neutron_from_proton(pid):
    """Isospin map p<->n: u<->d (and ubar<->dbar); other flavors unchanged."""
    if abs(pid) == 1:
        return 2 * np.sign(pid) if pid != 0 else 2
    if abs(pid) == 2:
        return 1 * np.sign(pid) if pid != 0 else 1
    return pid


def em_structure_function(pdf, x, Q2, Z, N):
    """Electromagnetic structure combination summed over a nucleus.

    Returns  sum_q e_q^2 * [ Z (q_p + qbar_p) + N (q_n + qbar_n) ](x, Q2),

    i.e. the F2-like quark+antiquark charge-weighted sum entering the dipole-portal
    DIS cross section (paper: sum_i e_i^2 q_i(x)). The neutron distributions are
    obtained from the proton set by isospin (u<->d). Vectorized over x, Q2.

    Args:
        pdf (PDFSet): proton PDF set.
        x, Q2 (array): Bjorken x and momentum transfer squared [GeV^2].
        Z, N (int): proton and neutron numbers of the target nucleus.
    """
    x = np.asarray(x, dtype=float)
    Q2 = np.asarray(Q2, dtype=float)
    total = np.zeros(np.broadcast(x, Q2).shape)
    for q in DIS_FLAVORS:
        e2 = QUARK_CHARGE2[q]
        # proton: quark q and antiquark -q
        fp = pdf.fxQ2(q, x, Q2) + pdf.fxQ2(-q, x, Q2)
        # neutron via isospin rotation of the proton set
        fn = pdf.fxQ2(_neutron_from_proton(q), x, Q2) + pdf.fxQ2(_neutron_from_proton(-q), x, Q2)
        total = total + e2 * (Z * fp + N * fn)
    return total
