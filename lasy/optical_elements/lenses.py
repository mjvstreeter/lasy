import numpy as np
from scipy.constants import c

from .optical_element import OpticalElement


def sellmeier(l,B,C):
    n2 = 1.0
    for n in range(3):
        n2+= B[n]*(l*1e6)**2/((l*1e6)**2-C[n])
    return np.sqrt(n2)
    
def calc_ref_ind(l,glass='none',n0 = 1.5):
    '''
    l is lambda in m
    glass is a type of glass which is hopefully in the list
    n0 = is the nominal refractive index if you don't want a physical glass

    returns refractive index
    '''
    n = 1.0*n0
    if 'bk7' in glass.lower():
        B = [1.03961212,0.231792344,1.01046945] 
        C = [0.00600069867,0.0200179144,103.560653]
        n = sellmeier(l,B,C)
    elif 'sf10' in glass.lower():
        B = [1.62153902,0.256287842,1.64447552] 
        C = [0.0122241457,0.0595736775,147.468793]
        n = sellmeier(l,B,C)
    elif 'uvfs' in glass.lower():
        B = [0.6961663,0.4079426,0.8974794] 
        C = [0.0684043**2, 0.1162414**2, 9.896161**2]
        n = sellmeier(l,B,C)
    elif 'air' in glass.lower():
        B = [0.05792105, 0.00167917] 
        C = [238.0185, 57.362]
        n = 1
        for ind in range(2):
            n += B[ind]**2/(C[ind]-(l*1e6)**(-2))
    
    elif 'none' in glass.lower():
        n = n0*1.0
    else:
        print(f"glass not found - returning constant {n:1.06f}")
    return n


class Lens(OpticalElement):
    r"""
    Class for a chromatic lens, with a varying refractive index with the wavelength.
    It takes into account the thickness as well

    Parameters
    ----------
    R1 : float
        ROC of the first surface (>0 if convex)
    R2 : float
        ROC of the second surface (>0 if concave)
    d : float
        Thickness of the lens
    n_func : function
        Function that returns the refractive index given the formula from refractiveindexinfo.com 
        e.g. for Fused Silica: 
        nFS = lambda x: (1+0.6961663/(1-(0.0684043/x)**2)+0.4079426/(1-(0.1162414/x)**2)+0.8974794/(1-(9.896161/x)**2))**.5 
    """
    def __init__(self, R1, R2, d, glass,n0=1.5):
        self.R1 = R1
        self.R2 = R2
        self.d = d
        self.glass = glass
        self.n0 = n0

    def amplitude_multiplier(self, x, y, omega):
        """
        Return the amplitude multiplier.

        Parameters
        ----------
        x, y, omega : ndarrays of floats
            Define points on which to evaluate the multiplier. Must have the same shape.

        Returns
        -------
        multiplier : ndarray of complex numbers
            Contains the value of the multiplier at the specified points
        """
        self.lam = 2 * np.pi * c / omega  # Wavelength in m
        # n = self.n_func(lam)
        self.n = calc_ref_ind(self.lam,self.glass,n0=self.n0)
        self.f = 1 / ((self.n - 1) * (1 / self.R1 - 1 / self.R2 + (self.n - 1) * self.d / (self.n * self.R1 * self.R2)))
        
        return np.exp(-1j * omega * (x**2 + y**2) / (2 * c * self.f))