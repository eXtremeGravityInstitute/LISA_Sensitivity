import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

import PhenomA as pa

# constants
fm     = 3.168753575e-8   # LISA modulation frequency
YEAR   = 3.15581497632e7  # year in seconds
AU     = 1.49597870660e11 # Astronomical unit (meters)
Clight = 299792458.       # speed of light (m/s)


##########################################################
################# Noise Curve Methods ####################
##########################################################

def LoadTransfer(self, file_name):
    """ 
    Load the data file containing the numerically calculate transfer function
    (sky and polarization averaged)
    """
    
    try:    # try to read in the data file
        transfer_data = np.genfromtxt(file_name) # read in the data
        
    except: # If file isn't successfully read in, use approximate transfer function
        print("Warning: Could not find transfer function file!")
        print("         \tApproximation will be used...")
        self.FLAG_R_APPROX = True
        return
        
    f = transfer_data[:,0]*self.fstar        # convert to frequency
    R = transfer_data[:,1]*self.NC           # response gets improved by more data channels

    # create an interpolation function; attach to LISA object
    self.R_INTERP = interpolate.splrep(f, R, s=0)
    self.FLAG_R_APPROX = False
    
    return
    
def Pn(self, f):
    """
    Caclulate the Strain Power Spectral Density
    """
    
    # single-link optical metrology noise (Hz^{-1}), Equation (10)
    P_oms = (1.5e-11)**2*(1. + (2.0e-3/f)**4) 
    
    # single test mass acceleration noise, Equation (11)
    P_acc = (3.0e-15)**2*(1. + (0.4e-3/f)**2)*(1. + (f/(8.0e-3))**4) 
    
    # total noise in Michelson-style LISA data channel, Equation (12)
    Pn = (P_oms + 2.*(1. + np.cos(f/self.fstar)**2)*P_acc/(2.*np.pi*f)**4)/self.Larm**2
    
    return Pn
    
def SnC(self, f):
    """
    Get an estimation of the galactic binary confusion noise are available for
        Tobs = {0.5 yr, 1 yr, 2 yr, 4yr}
    Enter Tobs as a year or fraction of a year
    """
    Tobs = self.Tobs 
    NC   = self.NC

    # Fix the parameters of the confusion noise fit
    if (Tobs < .75*YEAR):
        est = 1
    elif (0.75*YEAR < Tobs and Tobs < 1.5*YEAR):
        est = 2
    elif (1.5*YEAR < Tobs and Tobs < 3.0*YEAR):   
        est = 3
    else:
        est = 4
            
    if (est==1):
        alpha  = 0.133
        beta   = 243.
        kappa  = 482.
        gamma  = 917.
        f_knee = 2.58e-3  
    elif (est==2):
        alpha  = 0.171
        beta   = 292.
        kappa  = 1020.
        gamma  = 1680.
        f_knee = 2.15e-3 
    elif (est==3):
        alpha  = 0.165
        beta   = 299.
        kappa  = 611.
        gamma  = 1340.
        f_knee = 1.73e-3  
    else:
        alpha  = 0.138
        beta   = -221.
        kappa  = 521.
        gamma  = 1680.
        f_knee = 1.13e-3 
    
    A = 1.8e-44/NC
    
    Sc  = 1. + np.tanh(gamma*(f_knee - f))
    Sc *= np.exp(-f**alpha + beta*f*np.sin(kappa*f))
    Sc *= A*f**(-7./3.)
    
    return Sc
    
def Sn(self, f):
    """ Calculate the sensitivity curve """

    if (self.FLAG_R_APPROX == False): # if sensitivity curve file is provided use it
        R = interpolate.splev(f, self.R_INTERP, der=0)
    else:
        R = 3./20./(1. + 6./10.*(f/self.fstar)**2)*self.NC
        
    Sn = self.Pn(f)/R + self.SnC(f)

    return Sn
	
def Pn_WC(self, f):
    """ Calculate Power Spectral Density with confusion (WC) noise estimate """

    if (self.FLAG_R_APPROX == False):
        R = interpolate.splev(f, self.R_INTERP, der=0)
    else:
        R = 3./20./(1. + 6./10.*(f/self.fstar)**2)*self.NC
        
    PnC = self.Pn(f) + self.SnC(f)*R

    return PnC
    
##########################################################
################# LISA's Orbit Methods ###################
##########################################################
 
def SC_Orbits(self, t):
    """ Calculate the analytic (leading order in eccentricity) LISA orbits """

    N = len(t)
    kappa  = 0.0 # initial phase of LISA orbits
    Lambda = 0.0 # initial phase of spacecraft in their quasi-triangle configuration

    alpha = (2.*np.pi*fm*t + kappa).reshape((1,N)) 
    sa = np.sin(alpha) 
    ca = np.cos(alpha)

    beta = (np.array([0.0, 2.*np.pi/3., 4.*np.pi/3.]) + Lambda).reshape((3,1))
    sb = np.sin(beta)
    cb = np.cos(beta) # (S/C, len(t))
    
    x = np.zeros((3, 3, N)) # dim, S/C, time

    x[0] = AU*ca + AU*self.ecc*(sa*ca*sb - (1. + sa*sa)*cb)
    x[1] = AU*sa + AU*self.ecc*(sa*ca*cb - (1. + ca*ca)*sb)
    x[2] = -np.sqrt(3.)*AU*self.ecc*(ca*cb + sa*sb)

    return x
    
def SC_Seps(self, t, x):
    """ Calculate S/C unit-separation vectors """

    N = len(t)

    rij = np.zeros((3,3,3,N))

    rij[:,0,1,:] = x[:,1,:] - x[:,0,:]
    rij[:,1,0,:] = -rij[:,0,1,:]

    rij[:,0,2,:] = x[:,2,:] - x[:,0,:]
    rij[:,2,0,:] = -rij[:,0,2,:]

    rij[:,1,2,:] = x[:,2,:] - x[:,1,:]
    rij[:,2,1,:] = -rij[:,1,2,:]

    return rij/self.Larm
   

class LISA():
    """ 
    LISA class
    -----------------------
    Handles LISA's orbit and detector noise quantities
    
    Methods:
        LoadTranfer - read in, and store, transfer function data file
        SC_Orbit    - return calculate spacecraft (S/C) positions
        SC_Seps     - return unit-separation vectors between LISA S/C
        Pn          - return LISA's strain power spectral density
        Pn_WC       - return LISA's strain power spectral density with confusion noise estimate
        SnC         - return confusion noise estimate
        Sn          - return LISA's sensitivity curve
    """
    
    def __init__(self, Tobs=4*YEAR, Larm=2.5e9, NC=2, transfer_file='R.txt'):
        """
        Tobs - LISA observation period (4 years is nominal mission lifetime)
        Larm = 2.5e9 LISA's arm length, current design arm length, 
                            constant to 1st order in eccentricity
        NC - Number of data channels
        """
        self.Tobs = Tobs
        self.Larm = Larm 
        self.NC   = NC 
        
        self.ecc   = self.Larm/(2*np.sqrt(3.)*AU)  # to maintain quasi-equilateral triangle configuration
        self.fstar = Clight/(2*np.pi*self.Larm) # transfer frequency, design value ~ 19.1 mHz
        
        self.LoadTransfer(transfer_file) # load the transfer function

    # Methods
    LoadTransfer = LoadTransfer 
    
    Pn  = Pn
    Pn_WC = Pn_WC
    Sn  = Sn
    SnC = SnC
    
    SC_Orbits = SC_Orbits
    SC_Seps   = SC_Seps
    
    
def PlotSensitivityCurve(f, Sn, figure_file=None):
    """ 
    Plot the characteristic strain the sensitivity curve 
    
    If figure_file is provided, the figure will be saved
    """
    
    fig, ax = plt.subplots(1, figsize=(8,6))
    plt.tight_layout()

    ax.set_xlabel(r'f [Hz]', fontsize=20, labelpad=10)
    ax.set_ylabel(r'Characteristic Strain', fontsize=20, labelpad=10)
    ax.tick_params(axis='both', which='major', labelsize=20)
    
    ax.set_xlim(1.0e-5, 1.0e0)
    ax.set_ylim(3.0e-22, 1.0e-15)
    
    ax.loglog(f, np.sqrt(f*Sn)) # plot the characteristic strain
    
    plt.show()
    
    if (figure_file != None):
        plt.savefig(figure_file)
        
    return
    
    

    
    





    
    
    
    
    