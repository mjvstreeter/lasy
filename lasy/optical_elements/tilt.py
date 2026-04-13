import numpy as np
from scipy.constants import c
from scipy.fft import fft, ifft, fftshift

from .optical_element import OpticalElement


class FixedTilt(OpticalElement):
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

    def __init__(self, x_angle, y_angle):
        self.x_angle = x_angle
        self.y_angle = y_angle

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
        return np.exp(-1j * omega * (x*self.x_angle + y*self.y_angle) / c)



class AngularDispersion(OpticalElement):
    r"""
    Class for adding angular dispersion

    More precisely, the amplitude multiplier corresponds to:

    .. math::

        T(\boldsymbol{x}_\perp,\omega) = \exp(-i\omega/c (alpha_x (omega-omega_0)x +alpha_y (omega-omega_0)y))

    where
    :math:`\boldsymbol{x}_\perp` is the transverse coordinate (orthogonal
    to the propagation direction). The other parameters in this formula
    are defined below.

    Parameters
    ----------
    alpha_x : float (in radians)
        The angular dispersion in x
    alpha_y : float (in radians)
        The angular dispersion in y
    omega_0 : float (in fs^-1)
        The central angular frequency for the dispersion
    """

    def __init__(self, alpha_x, alpha_y, omega_0):
        self.alpha_x = alpha_x
        self.alpha_y = alpha_y
        self.omega_0 = omega_0

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
        self.x_angle = self.alpha_x*(omega-self.omega_0)
        self.y_angle = self.alpha_y*(omega-self.omega_0)
        
        return np.exp(-1j * omega * (x*self.x_angle + y*self.y_angle) / c)


class SpectralShear(OpticalElement):
    r"""
    Class for adding spectral shear

    More precisely, the amplitude multiplier corresponds to:

    .. math::

        T(\boldsymbol{x}_\perp,\omega) = \exp(-i\omega/c (alpha_x (omega-omega_0)x +alpha_y (omega-omega_0)y))

    where
    :math:`\boldsymbol{x}_\perp` is the transverse coordinate (orthogonal
    to the propagation direction). The other parameters in this formula
    are defined below.

    Parameters
    ----------
    alpha_x : float (in radians)
        The angular dispersion in x
    alpha_y : float (in radians)
        The angular dispersion in y
    omega_0 : float (in fs^-1)
        The central angular frequency for the dispersion
    """

    def __init__(self, shear_m_per_s,):
        self.shear_m_per_s = shear_m_per_s

    def amplitude_multiplier(self, x, y, omega, field):
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
        

        Nx = field.shape[0]
        dx = x[1,0,0]-x[0,0,0]
        xs = (fftshift(np.arange(Nx)-Nx/2)*2*np.pi/Nx).reshape(-1,1,1)
        ys = (omega-omega[0,0,0])*self.shear_m_per_s/dx
        shift_mat = np.exp( 1j* xs*ys)
        field_shift = ifft(shift_mat*fft(field,axis=0),axis=0)
        
        return field_shift
