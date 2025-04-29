import numpy as np
from eryn.ensemble import EnsembleSampler
from eryn.prior import ProbDistContainer

#@staticmethod
def build_prior_container(lisaobj):
    submodels = lisaobj.Model.submodels

    prior_dict = dict()
    ct = 0
    for sm in submodels:
        for jj in range(len(submodels[sm].prior)):
            prior_dict[ct] = submodels[sm].prior[jj]
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

        model = lisaobj.Model

        if 'seed' in model.params.keys():
            seed = model.params['seed']
        else:
            seed = None
        rng = np.random.default_rng(seed=seed)

        Npar = model.Npar
        
        priors = build_prior_container(lisaobj)

        ## get initial samples on the unit cube
        #init_samples = np.array([rng.uniform(0, 1, nwalkers)
        #                         for ii in range(Npar)]).T
        init_samples = np.array([priors.priors_in[p].rvs() for p in priors.priors_in]).T
        
        ensemble = EnsembleSampler(
            nwalkers=nwalkers,
            ndims=lisaobj.Model.Npar,
            log_like_fn=lisaobj.Model.likelihood,
            priors=priors,
        )

        return ensemble, init_samples

    @staticmethod
    def run_engine(engine, nsteps, init_samples, burn=100, thin_by=5,
                   progress=True):
        '''
        
        '''
        out = engine.run_mcmc(init_samples, nsteps, burn=burn,
                              progress=progress, thin_by=thin_by)

        samples = engine.get_chain()['model_0'].reshape(-1, engine.ndim)

        return samples
