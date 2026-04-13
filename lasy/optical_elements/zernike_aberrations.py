import numpy as np
from scipy.constants import c

from .optical_element import OpticalElement
from ..utils.zernike import zernike

class ZernikePhase(OpticalElement):
    r"""
    Class for a parabolic mirror.

    More precisely, the amplitude multiplier corresponds to:

    .. math::

        T(\boldsymbol{x}_\perp,\omega) = \exp(-i\omega (x^2+y^2)/2cf)

    where
    :math:`\boldsymbol{x}_\perp` is the transverse coordinate (orthogonal
    to the propagation direction). The other parameters in this formula
    are defined below.

    Parameters
    ----------
    f : float (in meter)
        The focal length of the parabolic mirror.
    """

    def __init__(self, j, amp, R_max):
        self.j = j
        self.amp = amp
        self.pupilCoords =(0,0, R_max)

    def amplitude_multiplier(self, x, y, omega):
        """
        Return the amplitude multiplier.

        Parameters
        ----------
        x, y, omega : ndarrays of floats
            Define points on which to evaluate the multiplier.
            These arrays need to all have the same shape.

        Returns
        -------
        multiplier : ndarray of complex numbers
            Contains the value of the multiplier at the specified points.
            This array has the same shape as the array omega.
        """
        rho = np.sqrt(x[:,:,0]**2+y[:,:,0]**2)/self.pupilCoords[-1]
        self.z_base = zernike(x[:,:,0], y[:,:,0], self.pupilCoords, self.j)*(rho<=1)


        return np.exp(-1j *self.z_base[:,:,np.newaxis]*self.amp)