import numpy as np
from scipy import optimize

import PhenomA as pa
import LISA as li

""" Cosmological values """
H0      = 69.6      # Hubble parameter today
Omega_m = 0.286     # density parameter of matter

""" Constants """
C       = 299792458.         # m/s
YEAR    = 3.15581497632e7    # sec
TSUN    = 4.92549232189886339689643862e-6 # mass of sun in seconds (G=C=1)
MPC     = 3.08568025e22/C    # mega-Parsec in seconds
pi      = np.pi

TOBS_MAX = 4*YEAR # Maximum observation period (LISA's nominal mission lifetime)

def get_Dl(z, Omega_m, H0):
    """ calculate luminosity distance in geometric units """
    # see http://arxiv.org/pdf/1111.6396v1.pdf
    x0 = (1. - Omega_m)/Omega_m
    xZ = x0/(1. + z)**3

    Phi0  = (1. + 1.320*x0 + 0.4415*x0**2  + 0.02656*x0**3)
    Phi0 /= (1. + 1.392*x0 + 0.5121*x0**2  + 0.03944*x0**3)
    PhiZ  = (1. + 1.320*xZ + 0.4415*xZ**2  + 0.02656*xZ**3)
    PhiZ /= (1. + 1.392*xZ + 0.5121*xZ**2  + 0.03944*xZ**3)
    
    return 2.*C/H0*(1.0e-3*MPC)*(1. + z)/np.sqrt(Omega_m)*(Phi0 - PhiZ/np.sqrt(1. + z))


def get_z(z, Dl, Omega_m, H0):
    """ calculate redishift uisng root finder """
    
    return get_Dl(z, Omega_m, H0) - Dl

class Binary():
    """ 
    Binary Class
    -------------------------------------------
    Inputs:
        Specify source-frame masses: m1, m2
        Specify a distance parameter: z, Dl (redshift, luminosity distance IN SECONDS)
        Specify an initial condition parameter: T_merge, f_start
                    (note that an upper limit of 4 years will be set on the 
                     observation period)
    
    Methods:
        CalcStrain: Calculate the characteristic strain of the binary. If (the optional
                    arguments) sky angles are provided use the stataionary phase approximation
                    signal generator, else use PhenomA amplitude exclusively
                    
        CalcSNR: Calculate the SNR averaged over polarization, inclination,
                  and sky angles. Theta, phi (spherical polar) are optional arguments
                  allowing the user to calculate the SNR at a specific sky location
                  averaged over only polarization and inclination angles

        PlotStrain: Plot the characteristic strain
    
    """
    
    def __init__(self, m1, m2, z=None, Dl=None, T_merge=None, f_start=None):
        # source-frame component masses
        self.m1 = m1
        self.m2 = m2
        
        # Store distance parameters
        if (Dl == None): # convert redshift into luminosity distance
            self.z = z # TODO: check that one of these is provided
            self.Dl = get_Dl(self.z, Omega_m, H0) # Dl returned in seconds (i.e. G=c=1, geometric units)
            print("Redshift provided. \n\tLuminosity Distance........... {} Mpc".format(self.Dl/MPC))

        else: # convert luminosity distance to redshift
            self.Dl = Dl # TODO: check that one of these is provided
            self.z = optimize.root(get_z, 1., args=(self.Dl, Omega_m, H0)).x[0]
            
        # adjust source-frame masses to detector-frame masses
        self.m1 *= 1. + z 
        self.m2 *= 1. + z
        
        # calculate relevant mass parameters
        self.M   = self.m1 + self.m2 # total mass
        self.eta = (self.m1 + self.m2)/self.M**2 # symmetric mass ratio
        
        # Obtain the frequency limits of the signal
        self.f_cut = pa.get_freq(self.M, self.eta, "cut") # PhenomA cut-off frequency i.e. frequency upper bound
    
        if (self.T_merge == None):
            self.f_start = f_start
            self.T_merge = (pa.dPsieff_df(self.f_start, self.M, self.eta, 0.0) \
                          - pa.dPsieff_df(self.f_cut,   self.M, self.eta, 0.0))/(2*pi)
            if (self.T_merge > Tobs_MAX): # Verify that 4 year observation period is not breached
                self.T_merg = Tobs_MAX
                # solve for the corresponding f_start
        else: # 
            self.T_merge = T_merge
            if (T_merge > Tobs_MAX):
                Raise(ValueError, "T_merge exceeds maximum allowed observation period: {} years".format(Tobs_MAX/YEAR))
        
        
        
        
        
        
        