import numpy as np
from numpyro import distributions as dist
from eryn.prior import uniform_dist
from scipy.stats import Uniform

@property
def logpdf(self):
     return self.log_prob

dist.Uniform.logpdf = logpdf

def powerlaw_prior():
        '''
        Prior function for an isotropic stochastic backgound analysis.

        Parameters
        -----------

        theta   : float
            A list or numpy array containing samples from a unit cube.

        Returns
        ---------

        theta   :   float
            theta with each element rescaled. The elements are  interpreted as alpha and log(Omega0)

        '''
        # Unpack: Theta is defined in the unit cube
        # Transform to actual priors
        #alpha = dist.Uniform(-5, 5)
        alpha = uniform_dist(-5, 5)
        #log_omega0 = dist.Uniform(-26, -14)
        log_omega0 = uniform_dist(-26, -14)

        return [alpha, log_omega0]
        
def fixedpowerlaw_prior():


    '''
    Prior function for a power law with fixed slope.
    
    Parameters
    -----------

    theta   : float
        A list or numpy array containing samples from a unit cube.

    Returns
    ---------

    theta   :   float
        theta with each element rescaled. The elements are  interpreted as alpha and log(Omega0)

    '''


    # Unpack: Theta is defined in the unit cube
    # Transform to actual priors
    #log_omega0  = -26*theta[0] + 12
    #log_omega0 = dist.Uniform(-26, -14)
    log_omega0 = uniform_dist(-26, -14)

    return [log_omega0]

def instr_noise_prior():


    '''
    Prior function for only instrumental noise

    Parameters
    -----------

    theta   : float
        A list or numpy array containing samples from a unit cube.

    Returns
    ---------

    theta   :   float
        theta with each element rescaled. The elements are  interpreted as alpha, omega_ref, Np and Na

    '''


    # Unpack: Theta is defined in the unit cube
    #log_Np, log_Na = theta

    # Transform to actual priors
    #log_Np = -5*log_Np - 39
    #log_Np = dist.Uniform(-44, -39)
    log_Np = uniform_dist(-44, -39)
    #log_Na = -5*log_Na - 46
    log_Na = uniform_dist(-51, -46)

    return [log_Np, log_Na]
