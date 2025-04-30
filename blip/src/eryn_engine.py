import numpy as np
from eryn.ensemble import EnsembleSampler
from eryn.prior import ProbDistContainer

#@staticmethod
def build_prior_container(lisaobj):
    """
    utility function to build eryn prior container
    """
    submodels = lisaobj.Model.submodels

    prior_dict = dict()
    ct = 0
    for sm in submodels:
        for p in submodels[sm].prior:
            prior_dict[ct] = p
            ct += 1
    
    prior = ProbDistContainer(prior_dict)

    return prior

class eryn_engine():

    '''
    Class for interfacing with dynesty sampler. This method also contains the
    priors definition for all models written to work with the dynesty sampler.
    '''

    @classmethod
    def define_engine(cls, lisaobj, nwalkers,):
        """
        Class method to initialise the eryn EnsembleSampler.

        Parameters
        ----------
        lisaobj :
            LISA data object
        
        nwalkers : int
            number of ensemble walkers

        Returns
        -------
        ensemble
            The initialised eryn EnsembleSampler
        
        parameters : dict
            A dictionary of model parameters
        
        init_samples
            An array of initial coordinates for each ensemble walker
        """

        model = lisaobj.Model

        if 'seed' in model.params.keys():
            seed = model.params['seed']
        else:
            seed = None
        rng = np.random.default_rng(seed=seed)
        
        priors = build_prior_container(lisaobj)

        init_samples = priors.rvs(size=(nwalkers,))

        parameters = model.parameters['all']

        ensemble = EnsembleSampler(
            nwalkers=nwalkers,
            ndims=model.Npar,
            log_like_fn=model.likelihood,
            priors=priors,
        )

        return ensemble, parameters, init_samples

    @staticmethod
    def run_engine(engine, nsteps, init_samples, burn=100, thin_by=5,
                   progress=True):
        '''
        Run the eryn ensemble sampler
        '''
        out = engine.run_mcmc(init_samples, nsteps, burn=burn,
                              progress=progress, thin_by=thin_by)

        samples = engine.get_chain()['model_0'].reshape(-1, )

        return samples
