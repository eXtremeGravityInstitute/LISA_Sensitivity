import numpy as np

TSUN = 4.9169e-6

# PhenomA, frequency coefficients
a = np.array([2.9740e-1, 5.9411e-1, 5.0801e-1, 8.4845e-1])
b = np.array([4.4810e-2, 8.9794e-2, 7.7515e-2, 1.2848e-1])
c = np.array([9.5560e-2, 1.9111e-1, 2.2369e-2, 2.7299e-1])

# PhenomA, phase coefficients
x = np.array([1.7516e-1, 0., -5.1571e1, 6.5866e2, -3.9031e3, 0., -2.4874e4, 2.5196e4])
y = np.array([7.9483e-2, 0., -1.7595e1, 1.7803e2, -7.7493e2, 0., -1.4892e3, 3.3970e2])
z = np.array([-7.2390e-2, 0., 1.3253e1, -1.5972e2, 8.8195e2, 0., 4.4588e3, -3.9573e3])


def Lorentzian(f, f_ring, sigma):
    """ """ 
    return sigma/(2*np.pi)/( (f-f_ring)**2 + 0.25*sigma**2 )
    
    
def get_freq(M, eta, name):
    """ """
    if (name == "merg"):
       idx = 0
    elif (name == "ring"):
        idx = 1
    elif (name == "sigma"):
        idx = 2
    elif (name == "cut"):
        idx = 3
        
    result = a[idx]*eta**2 + b[idx]*eta + c[idx]
    
    return result/(np.pi*M)
    
def Aeff(f, M, eta, Dl):
    """ """
    
    # generate phenomA frequency parameters
    f_merg = get_freq(M, eta, "merg")
    f_ring = get_freq(M, eta, "ring")
    sigma  = get_freq(M, eta, "sigma")
    f_cut  = get_freq(M, eta, "cut")
    
    # break up frequency array into pieces
    mask1 = (f<f_merg)
    mask2 = (f>=f_merg) & (f<f_ring)
    mask3 = (f>=f_ring) & (f<f_cut)
    
    C = M**(5./6)/Dl/np.pi**(2./3.)/f_merg**(7./6)*np.sqrt(5.*eta/24)
    w = 0.5*np.pi*sigma*(f_ring/f_merg)**(-2./3)
    
    A = np.zeros(len(f))
    
    A[mask1] = C*(f[mask1]/f_merg)**(-7./6)
    A[mask2] = C*(f[mask2]/f_merg)**(-2./3)
    A[mask3] = C*w*Lorentzian(f[mask3], f_ring, sigma)
    
    return A
    
def Psieff(f, M, eta, t0, phi0):
    """ """
    
    result = 0.0
    
    for i in range(8):
        result += (x[i]*eta**2 + y[i]*eta + z[i])*(np.pi*M*f)**((i-5.)/3.)
        
    return 2*np.pi*f*t0 + phi0 + result/eta
    
def dPsieff_df(f, M, eta, t0):
    """ """
    
    result = 0.0
    
    for i in range(8):
        result += ((i-5.)/3.)*(x[i]*eta**2 + y[i]*eta + z[i])*(np.pi*M)**((i-5.)/3.)*f**((i-5.)/3. - 1)
    
    return 2*np.pi*t0 + result/eta
    
    
    
    
    
    
    
    
