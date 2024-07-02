# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 14:29:50 2024

@author: zgl12 + cha227
"""

import sys
from astropy import units as u
import numpy as np
from scipy import integrate,optimize
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, root_scalar



class timescape:
    
    def __init__(self, fv0, H0):

        self.fv0 = fv0 # Void Fraction
        self.H0 = H0 # Dressed Hubble Parameter
        self.c = 2.99792458e5 # Speed of Light km s^-1
        self.Hbar0 = (2 * (2 + self.fv0) * self.H0) / (4*self.fv0**2 + self.fv0 + 4) #bare Hubble parameter
        self.b = self.b() # b parameter
        self.t0 = (2.0 /3) + (self.fv0 / 3.0)
        
        km_per_Mpc = 3.2407792896664E-20 # Ratio for converting km/Mpc
        s_to_billion_years = 1/ (1e9 * 365.25 * 24 * 60 * 60) # Seconds to Billion Years
        self.adjustment = km_per_Mpc / s_to_billion_years # Adjustment Factor to make units [Byr^{-1}]
        self.H0a = H0 * self.adjustment # Adjusted H0 [Byr^{-1}]
        self.Hbar0a = self.Hbar0 * self.adjustment # Adjusted Hbar0 [Byr^{-1}]

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
        return (2+fv)*fv**(1/3) / ( 3*t* self.Hbar0*self.fv0**(1/3) ) # Eq. 38 Average observational quantities in the timescape cosmology

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
        
        x = 3 * self.fv0 * t * self.Hbar0 
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
        '''
        return (2. * (1. - self.fv0)*(2. + self.fv0)) / (9. * self.fv0 * self.Hbar0) #Eq.39 Average observational quantities in the timescape cosmology
    
    def F(self, z):
        '''
        Parameters
        ----------
        z : redshift

        Returns
        -------
        Float
            F function.

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
        distance = self.c * t**(2/3.) * (self.F(z1) - self.F(z))
        
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
        
        t = self.tex(z2)
        distance = self.c * t**(2/3.) * (self.F(z1) - self.F(z2))
        return distance * u.Mpc
    

if __name__ == '__main__':
    H0 = 61.7 # dressed H0 value
    fv0 = 0.695 # Void Fraction at present time
    ts = timescape(fv0=fv0, H0=H0) # Initialise TS class
    

    print("test distance = ", ts.angular_diameter_distance(0.5))

 
   
####Check which H parameter is found from CMB data and whether it is dressed or bare Hubble parameter
####Might need to redo code to take bare parameter as input and then calculate dressed parameter