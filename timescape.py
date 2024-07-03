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
            self.H0_bare = H0 * u.km / (u.s * u.Mpc)
            self.H0_dressed = (4*self.fv0**2 + self.fv0 + 4) * self.H0_bare / (2 * (2 + self.fv0)) #* u.km / (u.s * u.Mpc)
        elif H0_type.lower() == 'dressed':
            self.H0_dressed = H0 * u.km / (u.s * u.Mpc)
            self.H0_bare = (2 * (2 + self.fv0) * self.H0_dressed) / (4*self.fv0**2 + self.fv0 + 4) #* u.km / (u.s * u.Mpc)
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
        

        self.b = (2. * (1. - self.fv0)*(2. + self.fv0)) / (9. * self.fv0 * self.H0_bare.value)  # b parameter
        self.t0 = (2.0 + self.fv0) / (3.0 * self.H0_bare.value * _H0units_to_invs / _sec_to_Gyr ) * u.Gyr
        

    #Energy densities for dressed and bare parameters
    def Om_bare(self, z):
        t = self._tex(z)
        print(t)
        return 4* (1 - self.fv(t)) / (2 + self.fv(t))**2 # Eq. B3 Average observational quantities in the timescape cosmology

    def Ok_bare(self, z):
        return 9 * self.fv(self._tex(z)) / (2 + self.fv(self._tex(z)))**2 # Eq. B4 Average observational quantities in the timescape cosmology
    
    def OQ_bare(self, z):
        return -self.fv(self._tex(z)) * (1 - self.fv(self._tex(z)))  / (2 + self.fv(self._tex(z)))**2 # Eq. B5 Average observational quantities in the timescape cosmology
    
    def Om_dressed(self, z):
        return self.Om_bare(z) * self._lapse_function(z)**3
    
    def Ok_dressed(self, z):
        return self.Ok_bare(z) * self._lapse_function(z)**3

    def OQ_dressed(self, z):
        return self.OQ_bare(z) * self._lapse_function(z)**3

    def _lapse_function(self, z):
        return 0.5*(2 + self.fv(self._tex(z))) # Eq. B7 Average observational quantities in the timescape cosmology

      
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
      return (2+fv)*fv**(1/3) / ( 3*t* self.H0_bare.value*self.fv0**(1/3) ) # Eq. 38 Average observational quantities in the timescape cosmology

    # Define the _texer function
    def _tex(self, z):
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
    
    def wall_time(self, z):
        '''
        Parameters
        ----------
        z : redshift

        Returns
        -------
        Float
            Gives the age of the universe at a given redshift for a wall observer.
        '''
        t = self._tex(z) / _H0units_to_invs * _sec_to_Gyr
        term1 = 2.0 *t /3.0
        term2 = 4.0 * self.Om0_dressed/(27.0 * self.fv0 * self.H0_bare.value * _H0units_to_invs / _sec_to_Gyr)
        term3 = log(1.0 + (9.0*self.fv0 * self.H0_bare.value * _H0units_to_invs * t / _sec_to_Gyr) / (4.0 * self.Om0_dressed))
        tau = term1 + term2 * term3
        return tau * u.Gyr
    
    def volume_average_time(self, z):
        '''
        Parameters
        ----------
        z : redshift

        Returns
        -------
        Float
            Gives the volume average age of the universe at a given redshift.
        '''
        t = self._tex(z) / _H0units_to_invs * _sec_to_Gyr
        return t
    
    def _yin(self,yy):
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
        
        x = 3 * self.fv0 * t * self.H0_bare.value
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
        
        t = self._tex(z) # Time
        return ( 2 + self.fv(t) ) / ( 3*t ) # Eq. B6 Average observational quantities in the timescape cosmology
    

    def H_dressed(self, z):
        '''
        Parameters
        ----------
        z : Array of floats
               CMB Redshift.
 
        Returns
        -------
        Float
            Dressed Hubble Parameter.
        '''
       
        t = self._tex(z) # Time
        return ( 4*self.fv_t(t)**2 + self.fv_t(t) +4 ) / ( 6*t ) # Eq. B8 Average observational quantities in the timescape cosmology
    

    def q_bare(self, z):
        '''
        Parameters
        ----------
        z : Array of floats
            CMB Redshift.

        Returns
        -------
        Float
            Bare Deceleration Parameter.
        '''

        return 2 * (1 - self.fv(self._tex(z)))**2 / (2 + self.fv(self._tex(z)))**2 # Eq. 58 ArXiv: 1311.3787
    
    def q_dressed(self, z):
        '''
        Parameters
        ----------
        z : Array of floats
            CMB Redshift.

        Returns
        -------
        Float
            Dressed Deceleration Parameter.
        ''' 
        t = self._tex(z)
        numerator = - (1 - self.fv(t)) * (8*self.fv(t)**3 + 39*self.fv(t)**2 - 12*self.fv(t) - 8)
        denominator = (4 + 4*self.fv(t)**2 + self.fv(t))**2
        return numerator/denominator

    def scale_factor_bare(self,z):
        '''
        Parameters
        ----------
        z : redshift  
        
        Returns
        -------
            Bare Scale Factor (setting a_dressed(0) = 1).
        '''
        t = self._tex(z)
        prefactor = self._lapse_function(0) * (3*self.H0_bare.value*t)**(2/3) / (2 + self.fv0)
        term1 = (3*self.fv0*self.H0_bare.value*t  +  (1-self.fv0)*(2+self.fv0))**(1/3)
        a_bare = prefactor * term1
        return a_bare
    
    def scale_factor_dressed(self,z):
        '''
        Parameters
        ----------
        z : redshift
        
        Returns
        -------
            Dressed Scale Factor (setting a_dressed(0) = 1).
        '''
        a_dressed = (self.scale_factor_bare(z) * self._lapse_function(0)) / (self.scale_factor_bare(0) * self._lapse_function(z))
        return a_dressed

    def _F(self, z):

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
        t = self._tex(z)
        term1 = 2.*t**(1/3.)
        term2 = (self.b**(1/3.) /6.)*np.log( (t**(1/3.) + self.b**(1/3.))**2 / (-self.b**(1/3.)*t**(1/3.) + self.b**(2/3.) + t**(2/3.)) )
        term3 = (self.b**(1/3.)/np.sqrt(3)) * np.arctan( (2*t**(1/3.) - self.b**(1/3.)) / (np.sqrt(3) * self.b**(1/3.)) )
        return term1 + term2 + term3

    def angular_diameter_distance(self, z, z_2=0): #arrange these distacnes so users can call either
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
        t = self._tex(z)
        distance = const.c.to('km/s').value * t**(2/3.) * (self._F(z_2) - self._F(z)) 
        
        return distance * u.Mpc
    
    def angular_diameter_distance_z1z2(self, z_1, z_2):
        '''
        Parameters
        ----------
        z_1: redshift of lens
        z_2: redshift of source

        Returns
        -------
        Float
            Angular Diameter Distance between z_1 and z_2.
        '''
        
        return self.angular_diameter_distance(z_1, z_2)
    
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
    fv0 = 0.695 # Void Fraction at present time
    ts = timescape(fv0=fv0, H0=H0) # Initialise TS class
    

    print("test distance = ", ts.angular_diameter_distance(0.5))
    print("volume age average age of the universe = ", ts.t0)
    print("Wall age average age of the universe = ", ts.wall_time(0))
  

 
