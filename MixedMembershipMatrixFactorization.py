from __future__ import division
import numpy as np
import cPickle as pickle

import pylab
import matplotlib.pyplot as plt

from numpy import exp
from numpy.linalg import pinv
from numpy.random import multivariate_normal
from numpy.random import normal
from numpy.random import dirichlet
from numpy.random import multinomial

from joblib import Parallel, delayed

class M3F_TIB:
    def __init__(self, X, N, M, W_0, v_0, D, lambda_0, sigmaSqd_0, sigmaSqd, c_0, d_0, chi_0, K_U, K_V, alpha, mu_0, n_iter=100):
	self.X = X #X:= data is a dictionary because the data is sparse
        self.N = N
        self.M = M
        self.n_iter = n_iter
        self.D = D

        W_0_shape = W_0.shape
        assert W_0_shape[0] == W_0_shape[1], 'W_0 is not a square matrix'
        assert W_0_shape[0] == d, 'W_0 is not a square matrix of dimensionality d'
        assert v_0 > d-1

        # take inputs
        self.alpha = alpha
        self.lambda_0 = lambda_0
        self.W_0 = W_0
        self.v_0 = v_0
        self.sigmaSqd_0 = sigmaSqd_0
        self.sigmaSqd = sigmaSqd
        self.c_0 = c_0
        self.d_0 = d_0
        self.chi_0 = chi_0 # fixed global bias
        # number of topics
        self.K_U = K_U
        self.K_V = K_V
        self.mu_0 = mu_0
        # biases
        self.c = np.zeros((self.N, self.K_U))
        self.d = np.zeros((self.M, self.K_V))
        # assignments
        self.z_U = np.zeros((self.N, self.M))
        self.z_V = np.zeros((self.N, self.M))
        # probability distribution of topics 
        self.theta_U = np.repeat(np.zeros(self.K_U, self.N)).reshape((self.N, self.K_U))
        self.theta_K = np.repeat(np.zeros(self.K_V, self.N)).reshape((self.M, self.K_U))

	def dictionary_index_mapping(self, d):
	    I_U, I_V = {}, {}
	    for i,j in d.keys():
		I_U[i].append(j)	
		I_V[j].append(i)
	    self.I_U = I_U
	    self.I_V = I_V

        def initial_sample(self):
            '''
            draw random sample from model prior
            initialize latent variables to model means
            Didn't do this!! initialize static factors to MAP estimates trained using stochastic gardient descent
            set remaining variables to model means
            '''        
            # posterior v's
            self.v_N = self.v_0 + self.N
            self.v_M = self.v_0 + self.M
            # precision matrices 
            self.lambda_U = self.v_0 * self.W_0
            self.lambda_V = self.v_0 * self.W_0
            # mean vectors 
            self.mu_U = self.mu_0
            self.mu_V = self.mu_0

            self.c = self.c + self.c_0
            self.d = self.d + self.d_0

            self.sample_u = self.mu_0
            self.sample_v = self.mu_0

            self.U = np.repeat(self.sample_u, self.N).reshape((self.N, self.D))
            self.V = np.repeat(self.sample_v, self.M).reshape((self.D, self.M))


        def sample_hyperparameters(self):
            for t in xrange(self.n_iter):
                u = self.U - self.sample_u
                v = self.V - self.sample_v
                m_u = self.mu_0 - self.sample_u
                m_v = self.mu_0 - self.sample_v
                u_sample_cov = np.outer(u, u)
                v_sample_cov = np.outer(v, v)
                W_u = pinv(pinv(self.W_0) + u_sample_cov + self.lambda_0*self.N/(self.lambda_0+self.N)*np.outer(m_u, m_u))
                W_v = pinv(pinv(self.W_0) + v_sample_cov + self.lambda_0*self.M/(self.lambda_0+self.M)*np.outer(m_v, m_v))
            # sample hyperparameters
                # Wishart precision matrices
                self.precision_U = Wishart(W_u, self.v_N).sample()
                self.precision_V = Wishart(W_v, self.v_M).sample()
                # Gaussian means
                self.mean_U = multivariate_normal((self.lambda_0*self.mu_0 + self.N*self.sample_u)/(self.lambda_0 + self.N), pinv(self.v_N*self.precision_U))
                self.mean_U = multivariate_normal((self.lambda_0*self.mu_0 + self.M*self.sample_v)/(self.lambda_0 + self.M), pinv(self.v_M*self.precision_V))


        def sample_topics(self):
            for i in xrange(self.N):    
                for k in xrange(self.K_U):
                    z_sum = 0 
                    mean_sum = 0
                    for j in self.I_U[i]:
                        z_sum += self.z_U[i,j] 
                        resid = X[(i,j)] - self.chi_0 - self.d[j,k] - np.dot(self.U[i,:], self.V[:,j])
                        mean_sum += self.z_U[i,j]*resid
                    std = 1./(1./self.sigmaSqd_0 + z_sum/self.Sqd)
                    mean = (self.c_0/self.sigmaSqd_0 + mean_sum/self.Sqd)*std
                    self.c[i,k] = normal(mean, std)

            for j in xrange(self.M):
                for k in xrange(self.K_V):
                    z_sum = 0
                    mean_sum = 0 
                    for i in self.I_V[j]:
                        z_sum += self.z_V[i,j]
                        resid = X[(i,j)] - self.chi_0 - self.c[i,k] - np.dot(self.U[i,:], self.V[:,j])
                        mean_sum += self.z_V[i,j]*resid
                    std = 1./(1./self.sigmaSqd_0 + z_sum/self.Sqd)
                    mean = (self.c_0/self.sigmaSqd_0 + mean_sum/self.Sqd)*std
                    self.d[j,k] = normal(mean, std)


        def sample_user_parameters(self):
            for i in xrange(self.N):
                resid_sum = np.zeros(self.D)
                outer_product_sum = np.zeros((self.D, self.D))
                for j in I_U[i]:
                    v = self.V[:,j]
                    outer_product_sum += np.outer(v, v)
                    resid = X[(i,j)] - self.chi_0 - self.c[i, self.z_U[i,j]] - self.d[i, self.z_U[i,j]]
                    resid_sum += v*resid
                lambda_U_star = self.precision_U + outer_product_sum/self.sigmaSqd
                self.U[i,:] = multivariate_normal(pinv(lambda_U_star)*(self.precision_U*self.mu_U + resid_sum/self.sigmaSqd), pinv(lambda_U_star))


        def sample_item_parameters(self):
            for j in xrange(self.M):
                resid_sum = np.zeros(self.D)
                outer_product_sum = np.zeros((self.D, self.D))
                for j in self.I_V[i]:
                    u = self.V[i,:]
                    outer_product_sum += np.outer(u, u)
                    resid = X[(i,j)] - self.chi_0 - self.c[i, self.z_V[i,j]] - self.d[self.z_V[i,j], j]
                    resid_sum += u*resid
                lambda_V_star = self.precision_V + outer_product_sum/self.sigmaSqd
                self.V[:,j] = multivariate_normal(pinv(lambda_V_star)*(self.precision_V*self.mu_V + resid_sum/self.sigmaSqd), pinv(lambda_V_star))


        def sample_topic_parameters(self):
	    # user topic assignment
            for i in xrange(self.N):
                z_sum = 0
                for j in self.I_U[i]:
                    z_sum += self.z_U[i,j]
                self.theta_U[i] = dirichlet(self.alpha/self.K_U + z_sum)
	    # item topic assigment
            for j in xrange(self.M):
                z_sum = 0
                for i in I_V[j]: 
                    z_sum += self.z_V[i,j]
                self.theta_V[j] = dirichlet(self.alpha/self.K_V + z_sum)

        def sample_topic_assignments(self):
            # user topics
            for i in xrange(self.N):
                for j in I_U[i]:
                    theta_U_star = np.zeros(self.K_U)
                    for k_idx, k in enumerate(self.z_U[i,:]):
                        theta_U_star[k_idx] = self.theta_U[i,j]*exp(-(X[(i,j)] - self.chi_0 - self.c_[i, self.z_V[i,j]] - self.d[k_idx, j] - np.dot(self.U_[i,:], self.V_[:,j]))**2/(2*self.sigmaSqd))
                    self.z_U[i,j] = multinomial(1, theta_U_star/sum(theta_U_star))

            # item topics
            for j in xrange(self.M):
                for i in self.I_V[j]:
                    theta_V_star = np.zeros(self.K_V)
                    for k_idx, k in enumerate(self.z_V[:,j]):
                        theta_V_star[k_idx] = self.theta_V[i,j]*exp(-(X[(i,j)] - self.chi_0 - self.c_[i, k_idx] - self.d[self.z_U[i,j], j] - np.dot(self.U_[i,:], self.V_[:,j]))**2/(2*self.sigmaSqd))
                    self.z_V[i,j] = multinomial(1, theta_V_star/sum(theta_V_star))


        def Gibbs(self):
            self.sample_hyperparameters()
            self.sample_topics()
            self.sample_user_parameters()
            self.sample_item_parameters()
            self.sample_topic_parameters()
            self.sample_topic_assignments()
            
        def main(self):
            self.initial_sample()
            for i in xrange(self.n_iters):
                self.Gibbs()
            
        
        def compute_ratings(self, i, j):
            r_ij = normal(self.chi_O + self.c[i,self.z_V[i,j]] + self.d[self.z_U[i,j], j] + np.dot(self.U[i,:], self.V[:,j]), self.sigmaSqd)
            return r_ij
