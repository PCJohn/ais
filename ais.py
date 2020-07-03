import numpy as np
import tensorflow as tf
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt


class AIS():
    def __init__(self, D, num_iter, parallel_rounds=10):
        self.D = D
        self.num_iter = num_iter
        self.parallel_rounds = parallel_rounds
        self.transition_steps = 10

        self.p_0_mean = np.zeros(self.D)
        self.p_0_cov = np.eye(self.D)
        self.p_0 = multivariate_normal(self.p_0_mean,self.p_0_cov).pdf

        #self.p_0_Z = 1./(np.sqrt((2*np.pi)**self.D * np.linalg.det(self.p_0_cov)))
        #print('---',self.p_0_Z)

    def sample_p_0(self):
        return np.random.multivariate_normal(self.p_0_mean,self.p_0_cov,size=self.parallel_rounds)

    def p_j(self, x, target, beta):
        return self.p_0(x)**(1-beta) * target(x)**beta

    def partition(self, target):
        all_beta = np.linspace(0,1,self.num_iter)
        x = self.sample_p_0()
        w = np.ones(self.parallel_rounds)
        for i in range(self.num_iter-1):
            for _ in range(self.transition_steps):
                x_ = x + np.random.normal(size=x.shape)
                p_x_ = self.p_j(x_, target, all_beta[i])
                p_x = self.p_j(x, target, all_beta[i])
                
                ind = np.random.random(size=self.parallel_rounds) < (p_x_ / p_x)
                x[ind] = x_[ind]
            w = w * (self.p_j(x, target, all_beta[i+1]) / self.p_j(x, target, all_beta[i]))
        import pdb; pdb.set_trace();
        return np.mean(w)

    def mean(self,target):
        pass


def target(x):
    d = multivariate_normal([5,5],[[1,0],[0,1]])
    K = 50
    return K * d.pdf(x)

if __name__ == '__main__':
    
    n_rounds = 2000
    n_iter = 20
    D = 2
    ais = AIS(D, n_iter, parallel_rounds=n_rounds)

    Z = ais.partition(target)
    print('>>>',Z)

    '''
    for t in range(n_samples):
        # Sample initial point from q(x)
        x = p_n.rvs()
        w = 1

        for n in range(1, len(betas)):
            # Transition
            x = T(x, lambda x: f_j(x, betas[n]), n_steps=5)

            # Compute weight in log space (log-sum):
            # w *= f_{n-1}(x_{n-1}) / f_n(x_{n-1})
            w += np.log(f_j(x, betas[n])) - np.log(f_j(x, betas[n-1]))

        samples[t] = x
        weights[t] = np.exp(w)  # Transform back using exp


    # Compute expectation
    a = 1/np.sum(weights) * np.sum(weights * samples)
    '''


