import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

""" PhenomA coefficeints """

a0 = 2.9740e-1
b0 = 4.4810e-2
c0 = 9.5560e-2

a1 = 5.9411e-1
b1 = 8.9794e-2
c1 = 1.9111e-1

a2 = 5.0801e-1
b2 = 7.7515e-2
c2 = 2.2369e-2

a3 = 8.4845e-1
b3 = 1.2848e-1
c3 = 2.7299e-1

""" Constants """
C       = 299792458.         # m/s
YEAR    = 3.15581497632e7 # sec
TSUN    = 4.92549232189886339689643862e-6 # mass of sun in seconds (G=C=1)
H0      = 69.6           # Hubble's parameter today
Omega_m = 0.286  # density parameter of dust matter
MPC     = 3.08568025e22 # mega-Parsec in meters

""" LISA arm length """
L = 2.5e9 # meters

""" Transfer Frequency """
f_star = C/(2.*np.pi*L)

""" Observation Period """
Tobs = 4.*YEAR

""" Number of Michelson Data Channels """
NC = 2

def get_Sc_est(f, Tobs):
    """
    Get an estimation of the galactic binary confusion noise are available for
        Tobs = {0.5 yr, 1 yr, 2 yr, 4yr}
    Enter Tobs as a year or fraction of a year
    """
    # Fix the parameters of the confusion noise fit
    if (Tobs == 0.5*YEAR):
        est = 1
    elif (Tobs == 1.0*YEAR):
        est = 2
    elif (Tobs == 2.0*YEAR):
        est = 3
    elif (Tobs == 4.0*YEAR):
        est = 4

    # else find the closest observation period estimation
    else:
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
    
def get_Pn(f):
    """
    Get the Power Spectral Density
    """
    
    # single-link optical metrology noise (Hz^{-1}), Equation (10)
    P_oms = (1.5e-11)**2*(1. + (2.0e-3/f)**4) 
    
    # single test mass acceleration noise, Equation (11)
    P_acc = (3.0e-15)**2*(1. + (0.4e-3/f)**2)*(1. + (f/(8.0e-3))**4) 
    
    # total noise in Michelson-style LISA data channel, Equation (12)
    Pn = (P_oms + 2.*(1. + np.cos(f/f_star)**2)*P_acc/(2.*np.pi*f)**4)/L**2
    
    return Pn
    
    
def get_Sn_approx(f):
    """
    Get the noise curve approximation, Equation (1) of ``LISA Sensitivity'' -Neil Cornish
    """
    
    # Sky and polarization averaged signal response of the detector, Equation (9)
    Ra = 3./20./(1. + 6./10.*(f/f_star)**2)*NC
    
    # strain spectral density, Equation (2)
    Sn = get_Pn(f)/Ra
    
    return Sn
    
  
def get_A(f, M, eta, Mc, Dl):
    
    f0 = (a0*eta**2 + b0*eta + c0)/(np.pi*M) # merger frequency
    f1 = (a1*eta**2 + b1*eta + c1)/(np.pi*M) # ringdown frequency
    f2 = (a2*eta**2 + b2*eta + c2)/(np.pi*M) # decay-width of ringdown
    f3 = (a3*eta**2 + b3*eta + c3)/(np.pi*M) # cut-off frequency
    
    A = np.sqrt(5./24.)*Mc**(5./6.)*f0**(-7./6.)/np.pi**(2./3)/(Dl/C)

    if (f < f0):
        A *= (f/f0)**(-7./6.)
    
    elif (f0 <= f and f < f1):
        A *= (f/f0)**(-2./3.)
    
    elif (f1 <= f and f < f3):
        w = 0.5*np.pi*f2*(f0/f1)**(2./3.)
        A *= w*f2/((f - f1)**2 + 0.25*f2**2)/(2.*np.pi)
        
    else:
        A *= 0.
    
    return A

def get_Dl(z):
    """ calculate luminosity distance in geometrized units """
    # see http://arxiv.org/pdf/1111.6396v1.pdf
    x0 = (1. - Omega_m)/Omega_m
    xZ = x0/(1. + z)**3

    Phi0  = (1. + 1.320*x0 + 0.4415*x0**2  + 0.02656*x0**3)
    Phi0 /= (1. + 1.392*x0 + 0.5121*x0**2  + 0.03944*x0**3)
    PhiZ  = (1. + 1.320*xZ + 0.4415*xZ**2  + 0.02656*xZ**3)
    PhiZ /= (1. + 1.392*xZ + 0.5121*xZ**2  + 0.03944*xZ**3)
    
    return 2.*C/H0*(1.0e-3*MPC)*(1. + z)/np.sqrt(Omega_m)*(Phi0 - PhiZ/np.sqrt(1. + z))

def get_z(z, Dl):
    """ calculate redishift uisng root finder """
    return get_Dl(z) - Dl
    
    
    
    
    
    
    
    
    
    

    
    
    
    
    
    
    
    
    
    
    