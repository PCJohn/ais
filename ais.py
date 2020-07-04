import numpy as np
from scipy.stats import multivariate_normal


class AIS():
    def __init__(self, D, num_iter, parallel_rounds=10):
        self.D = D                              # dimension
        self.num_iter = num_iter                # num iterations
        self.parallel_rounds = parallel_rounds  # number of parallel runs
        self.transition_steps = 10              # number of steps for MCMC
        
        self.w = np.ones(self.parallel_rounds)  # importance weights
        self.traj = []                          # sampled trajectories

        # proposal -- centered, unti gaussian
        self.p_0_mean = np.zeros(self.D)
        self.p_0_cov = np.eye(self.D)
        self.p_0 = multivariate_normal(self.p_0_mean,self.p_0_cov).pdf
        self.p_0_Z = 1./(np.sqrt((2*np.pi)**self.D * np.linalg.det(self.p_0_cov)))

    def sample_p_0(self):
        return np.random.multivariate_normal(self.p_0_mean,self.p_0_cov,size=self.parallel_rounds)

    def p_j(self, x, target, beta):
        return self.p_0(x)**(1-beta) * target(x)**beta

    def compute_w(self, target):
        self.traj = []
        all_beta = np.linspace(0,1,self.num_iter)
        x = self.sample_p_0()
        for i in range(self.num_iter-1):
            for _ in range(self.transition_steps):
                self.traj.append(x.copy())
                x_ = x + np.random.normal(size=x.shape)
                p_x_ = self.p_j(x_, target, all_beta[i])
                p_x = self.p_j(x, target, all_beta[i])
                
                ind = np.random.random(size=self.parallel_rounds) < (p_x_ / p_x)
                x[ind] = x_[ind]
            self.w = self.w * (self.p_j(x, target, all_beta[i+1]) / self.p_j(x, target, all_beta[i]))

    def partition(self):
        return np.mean(self.w) * self.p_0_Z

    def mean(self):
        return np.matmul(self.w[np.newaxis,:],self.traj[-1]) / self.w.sum()
        

