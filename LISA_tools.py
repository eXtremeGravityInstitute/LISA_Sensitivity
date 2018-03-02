"""
Tools...
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

"""
PhenomA coefficeints:
Coefficients from Table 2 in ``LISA Sensitivity'' - Neil Cornish & Travis Robson
"""

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
YEAR    = 3.15581497632e7    # sec
TSUN    = 4.92549232189886339689643862e-6 # mass of sun in seconds (G=C=1)
MPC     = 3.08568025e22/C    # mega-Parsec in seconds


def get_Sn(constants):
	# constants = np.arrays([H0, Omega_m, L, f_star, Tobs, NC])
	f_star = constants[3]
	L      = constants[2]
	NC     = constants[5]
	Tobs   = constants[4]
	
	transfer_data = np.genfromtxt('R.txt')
	f = transfer_data[:,0]*f_star # convert to frequency
	R = transfer_data[:,1]*NC     # response gets improved by more data channels
	
	Sn = get_Pn(f, f_star, L)/R + get_Sc_est(f, Tobs, NC)
	
	return f, Sn

def get_Sc_est(f, Tobs, NC):
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
    
def get_Pn(f, f_star, L):
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
    
    
def get_Sn_approx(f, f_star, L, NC):
    """
    Get the noise curve approximation, Equation (1) of ``LISA Sensitivity'' - Neil Cornish & Travis Robson
    """
    
    # Sky and polarization averaged signal response of the detector, Equation (9)
    Ra = 3./20./(1. + 6./10.*(f/f_star)**2)*NC
    
    # strain spectral density, Equation (2)
    Sn = get_Pn(f, f_star, L)/Ra
    
    return Sn
    

def get_A(f, M, eta, Mc, Dl):
	"""
	PhenomA: Binary Waveform
	--------------
	Section 2 of ``LISA Sensitivity'' - Neil Cornish & Travis Robson
	"""

	f0 = (a0*eta**2 + b0*eta + c0)/(np.pi*M) # merger frequency
	f1 = (a1*eta**2 + b1*eta + c1)/(np.pi*M) # ringdown frequency
	f2 = (a2*eta**2 + b2*eta + c2)/(np.pi*M) # decay-width of ringdown
	f3 = (a3*eta**2 + b3*eta + c3)/(np.pi*M) # cut-off frequency

	A = np.sqrt(5./24.)*Mc**(5./6.)*f0**(-7./6.)/np.pi**(2./3)/Dl

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

def get_Dl(z, Omega_m, H0):
    """ calculate luminosity distance in geometrized units """
    # see http://arxiv.org/pdf/1111.6396v1.pdf
    x0 = (1. - Omega_m)/Omega_m
    xZ = x0/(1. + z)**3

    Phi0  = (1. + 1.320*x0 + 0.4415*x0**2  + 0.02656*x0**3)
    Phi0 /= (1. + 1.392*x0 + 0.5121*x0**2  + 0.03944*x0**3)
    PhiZ  = (1. + 1.320*xZ + 0.4415*xZ**2  + 0.02656*xZ**3)
    PhiZ /= (1. + 1.392*xZ + 0.5121*xZ**2  + 0.03944*xZ**3)
    
    return 2.*C/H0*(1.0e-3*MPC)*(1. + z)/np.sqrt(Omega_m)*(Phi0 - PhiZ/np.sqrt(1. + z))
   # return 2./H0*(1.0e-3*MPC)*(1. + z)/np.sqrt(Omega_m)*(Phi0 - PhiZ/np.sqrt(1. + z))


def get_z(z, Dl, Omega_m, H0):
    """ calculate redishift uisng root finder """
    return get_Dl(z, Omega_m, H0) - Dl
    
def get_h_char_track(f, f_start, f_end, M, eta, M_chirp, Dl, constants):

	# constants = np.arrays([H0, Omega_m, L, f_star, Tobs, NC])
	f_star = constants[3]
	L      = constants[2]
	NC     = constants[5]
	Tobs   = constants[4]
	
	A_arr   = np.zeros(len(f))
	h_c_arr = np.zeros(len(f))

	arg_start = np.where(f<=f_start)[0][-1]
	if (f_end > 1.): # i.e. off the graph
		arg_end = np.len(f)-1
	else:
		arg_end = np.where(f>=f_end)[0][0]

	for i in range(arg_start, arg_end):
		A_arr[i]   = get_A(f[i], M, eta, M_chirp, Dl) 
		h_c_arr[i] = f[i]*A_arr[i]*np.sqrt(16./5.) 
		
		
	SNR = 0.
	f, Sn = get_Sn(constants)
	for i in range(arg_start, arg_end):
		freq   = 0.5*(f[i] + f[i-1])
		Sn_est = 0.5*(1./Sn[i] + 1./Sn[i-1])
		SNR += 16./5.*freq*get_A(freq, M, eta, M_chirp, Dl)**2*Sn_est*(np.log(f[i]) - np.log(f[i-1]))
	SNR = np.sqrt(SNR)

	return h_c_arr, SNR
    
    
def get_h_char_point(f_start, f_end, M, eta, M_chirp, Dl, constants):

	# constants = np.arrays([H0, Omega_m, L, f_star, Tobs, NC])
	f_star = constants[3]
	L      = constants[2]
	NC     = constants[5]
	Tobs   = constants[4]
	
	h_c = np.sqrt(16./5.*(f_end-f_start)*f_start)*get_A(f_start, M, eta, M_chirp, Dl)

	Sn_est = get_Sn_approx(f_start, f_star, L, NC) + get_Sc_est(f_start, Tobs, NC)
	SNR = 8.*np.sqrt(Tobs/5.)*M_chirp**(5./3.)*(np.pi*f_start)**(2./3.)/Dl/np.sqrt(Sn_est)
        
	return h_c, SNR
    
"""
Calculate the Characteristic strain of the source

Inputs:
    m1 - component mass 1, SOURCE FRAME!
    m2 - component mass 2, SOURCE FRAME!
    
    Initial condition options (Specify one!)
    --------------------------
    T_merger - time to merger for source
    f_start  - start frequency for source
    
    Distance options (Specify one!)
    --------------------------
    D_lum - Luminosity distance
    z     - redshift
"""

def calculate_plot_source(m1, m2, constants, Dl=None, z=None, T_merger=None, f_start=None):
	"""
	Determine the appropriate way to plot the source, calculate its characteristic strain
	and print the correpsonding SNR.
	"""
	# constants = np.arrays([H0, Omega_m, L, f_star, Tobs, NC])
	Omega_m = constants[1]
	H0      = constants[0]
	Tobs    = constants[4]
	NC      = constants[5]
	L       = constants[2]
	f_star  = constants[3]

	f, Sn = get_Sn(constants)

	""" Sort out the luminosity distance and redshift of the source """
	if (Dl==None): # Z was specified, then we must calculate Dl
		Dl = get_Dl(z, Omega_m, H0)
	elif(z==None):
		z = optimize.root(get_z, 1., args=(Dl, Omega_m, H0)).x[0]

	""" Calculate relevant mass parameters """
	m1 *= (1. + z) # redshift the source frame masses
	m2 *= (1. + z)
	M = m1 + m2                           # total mass
	M_chirp = (m1*m2)**(3./5.)/M**(1./5.) # chirp mass
	eta = (m1*m2)/M**2                    # symmetric mass ratio   

	""" Calculate PhenomA cut-off frequency """
	f3 = (a3*eta**2 + b3*eta + c3)/(np.pi*M) 

	if (f_start==None): # T_merger was specified
		f_start = (5.*M_chirp/T_merger)**(3./8.)/(8.*np.pi*M_chirp)
	else: # f_start was specified, calculate time to merger for circular binary
		T_merger = 5.*M_chirp/(8.*np.pi*f_start*M_chirp)**(8./3.)

	""" Determine the ending frequency of this source """
	if (T_merger > Tobs):
		f_end = (5.*M_chirp/(np.abs(Tobs-T_merger)))**(3./8.)/(8.*np.pi*M_chirp)
	elif (T_merger <= Tobs):
		f_end = f3

	""" Plot the results """
	plt.figure(figsize=(8,6))
	plt.rcParams['text.usetex'] = True
	plt.ion()
	plt.rc('text', usetex=True)
	plt.rc('font', family='calibri')
	plt.xlabel(r'$f ~[Hz]$', fontsize=20, labelpad=10)
	plt.ylabel(r'Characteristic Strain$', fontsize=20, labelpad=10)
	plt.tick_params(axis='both', which='major', labelsize=20)

	# How much log bandwidth does the source span
	d_log_f = np.log(f_end/f_start)

	if (d_log_f > 0.5): # plot a track
		h_c_arr, SNR = get_h_char_track(f, f_start, f_end, M, eta, M_chirp, Dl, constants)

		plt.loglog(f, h_c_arr) 
		title = 'Track SNR: ' + str(SNR)
		plt.title(title, fontsize=20)

		out_file = 'char_signal_strain.dat'
		np.savetxt(out_file,(np.vstack((f, h_c_arr)).T), delimiter=' ')

	else: # track is too short, plot a point
		h_c, SNR = get_h_char_point(f_start, f_end, M, eta, M_chirp, Dl, constants)

		plt.loglog(f_start, h_c, 'r.')
		title = 'Point SNR: ' + str(SNR)
		plt.title(title, fontsize=20)

		out_file = 'char_signal_strain.dat'
		np.savetxt(out_file,(np.vstack((f_start, h_c)).T), delimiter=' ')

	plt.loglog(f, np.sqrt(f*Sn)) # plot the characteristic strain of noise
	#%config InlineBackend.figure_format = 'retina'
	plt.xlim(1.0e-5, 1.0e0)
	plt.ylim(1.0e-22, 1.0e-15)
	plt.tight_layout()
	plt.tick_params(labelsize=20)

	plt.show()

	return None
    
    
    
    
    
    