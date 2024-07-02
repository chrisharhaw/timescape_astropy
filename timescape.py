# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 14:29:50 2024

@author: zgl12 + cha227 
"""


import numpy as np
from math import exp, floor, log, pi, sqrt
from scipy import integrate,optimize
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, root_scalar

from astropy import units as u
import astropy.constants as const

############################################################################################################
# Unit Conversions and Constants

_H0units_to_invs = (u.km / (u.s * u.Mpc)).to(1.0 / u.s)
_sec_to_Gyr = u.s.to(u.Gyr)
# const in critical density in cgs units (g cm^-3)
_critdens_const = (3 / (8 * pi * const.G)).cgs.value
# angle conversions
_radian_in_arcsec = (1 * u.rad).to(u.arcsec)
_radian_in_arcmin = (1 * u.rad).to(u.arcmin)
# Radiation parameter over c^2 in cgs (g cm^-3 K^-4)
_a_B_c2 = (4 * const.sigma_sb / const.c**3).cgs.value
# Boltzmann constant in eV / K
_kB_evK = const.k_B.to(u.eV / u.K)


############################################################################################################

class timescape:
    def __init__(self, fv0 = 0.695, H0 = 61.7, H0_type = 'dressed'):
        '''
        Parameters
        ----------
        fv0 : Float
            Void Fraction at present time.
        
        H0 : Float
            Dressed Hubble Parameter at present time.
        '''

        #Void Fraction at Present Time
        self.fv0 = fv0 

        if H0_type.lower() == 'bare':
            self.H0_bare = H0
            self.H0_dressed = (4*self.fv0**2 + self.fv0 + 4) * self.H0_bare / (2 * (2 + self.fv0))
        elif H0_type.lower() == 'dressed':
            self.H0_dressed = H0
            self.H0_bare = (2 * (2 + self.fv0) * self.H0_dressed) / (4*self.fv0**2 + self.fv0 + 4)
        else:
            raise ValueError("H0_type must be either 'bare' or 'dressed'")
        

        #Present Bare Density Parameters
        self.Om0_bare = self.Om_bare(0)
        self.Ok0_bare = self.Ok_bare(0)
        self.OQ0_bare = self.OQ_bare(0)

        #Present Dressed Density Parameters
        self.Om0_dressed = self.Om0_bare * self._lapse_function(0)**3
        self.Ok0_dressed = self.Ok0_bare * self._lapse_function(0)**3
        self.OQ0_dressed = self.OQ0_bare * self._lapse_function(0)**3
        


        self.b = self.b() # b parameter
        self.t0 = (2.0 /3) + (self.fv0 / 3.0)
        

    #Present Void Fraction in terms of Dressed Matter Density
    def dressed_matter_to_void_fraction(self):
        self.fv0 = 0.5 * ( np.sqrt(9-8*self.Om0_dressed) - 1 )

    def z1(self, t):
        '''
        Parameters
        ----------
        t : Time in billion years.

        Returns
        -------
        Float
            The cosmological redshift function for the time. 
        '''
        fv = self.fv(t)
        return (2+fv)*fv**(1/3) / ( 3*t* self.H0_bare*self.fv0**(1/3) ) # Eq. 38 Average observational quantities in the timescape cosmology

    # Define the texer function
    def tex(self, z):
        '''
        Parameters
        ----------
        z : redshift

        Returns
        -------
        Float
            Time 
        '''
        # Define the function to find the root
        def func(t):
            return self.z1(t) - z - 1
        #   Use root_scalar to find the root
        result = root_scalar(func, bracket=[0.0001, 1])  # Adjust the bracket if needed
        return result.root
    
    def yin(self,yy):
        yin = ((2 * yy) + (self.b / 6) * (np.log(((yy + self.b) ** 2) / ((yy ** 2) + (self.b ** 2) - yy * self.b)))
            + ((self.b / (3 ** 0.5)) * np.arctan((2 * yy - self.b) / ((3 ** 0.5) * self.b))))
        return yin
    

    def fv(self, t):
        '''
        Parameters
        ----------
        t : time in billion years.

        Returns
        -------
        Float
            Tracker soln as function of time.
        '''
        
        x = 3 * self.fv0 * t * self.H0_bare
        return x /(x + (1.-self.fv0)*(2.+self.fv0)) # Eq. B2 Average observational quantities in the timescape cosmology

    def H_bare(self, z):
        '''
        Parameters
        ----------
        z : redshift

        Returns
        -------
        Float
            Bare Hubble Parameter.
        '''
        
        t = self.tex(z) # Time
        return ( 2 + self.fv(t) ) / ( 3*t ) # Eq. B6 Average observational quantities in the timescape cosmology
    
    def b(self):
        '''
        Returns
        -------
        Float
            b parameter.
            Eq.39 Average observational quantities in the timescape cosmology
        '''
        return (2. * (1. - self.fv0)*(2. + self.fv0)) / (9. * self.fv0 * self.H0_bare) 
    
    def F(self, z):
        '''
        Parameters
        ----------
        z : redshift

        Returns
        -------
        Float
            F function 
            Eqn.41 Average observational quantities in the timescape cosmology
        '''
        t = self.tex(z)
        term1 = 2.*t**(1/3.)
        term2 = (self.b**(1/3.) /6.)*np.log( (t**(1/3.) + self.b**(1/3.))**2 / (-self.b**(1/3.)*t**(1/3.) + self.b**(2/3.) + t**(2/3.)) )
        term3 = (self.b**(1/3.)/np.sqrt(3)) * np.arctan( (2*t**(1/3.) - self.b**(1/3.)) / (np.sqrt(3) * self.b**(1/3.)) )
        return term1 + term2 + term3

    def angular_diameter_distance(self, z, z1=0): #arrange these distacnes so users can call either
        '''
        Parameters
        ----------
        z : Array of floats
            Redshift.

        Returns
        -------
        Float
            Angular Diameter Distance.
        '''
        t = self.tex(z)
        distance = const.c.to('km/s').value * t**(2/3.) * (self.F(z1) - self.F(z)) 
        
        return distance * u.Mpc
    
    def angular_diameter_distance_z1z2(self, z1, z2):
        '''
        Parameters
        ----------
        z1: redshift of lens
        z2: redshift of source

        Returns
        -------
        Float
            Angular Diameter Distance between z1 and z2.
        '''
        
        return self.angular_diameter_distance(z1, z2)
    
    def transverse_comoving_distance(self, z):
        '''
        Parameters
        ----------
        z : Array of floats
            Redshift.

        Returns
        -------
        Transverse Comoving Distance: Float
            Angular Diameter Distance between z1 and z2.
        '''
        
        return self.angular_diameter_distance(z) * (1+z)
    
    def luminosity_distance(self, z):
        '''
        Parameters
        ----------
        z : Array of floats
            Redshift.

        Returns
        -------
        Luminosity Distance: Float
            Luminosity Distance.
        '''
        
        return self.angular_diameter_distance(z) * (1+z)**2


   
        
    


if __name__ == '__main__':
    H0 = 61.7 # dressed H0 value
    fv0 = 0.73 # Void Fraction at present time
    ts = timescape(fv0=fv0, H0=H0) # Initialise TS class
    

    print("test distance = ", ts.angular_diameter_distance(0.5))

 
   
####Check which H parameter is found from CMB data and whether it is dressed or bare Hubble parameter
####Might need to redo code to take bare parameter as input and then calculate dressed parameter