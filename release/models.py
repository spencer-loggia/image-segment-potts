"""
Keep model implementations in here.

This file is where you will write most of your code!
"""

import numpy as np
import math

def initialize_variational_parameters(num_rows_of_image, num_cols_of_image, K):
    """ Helper function to initialize variational distributions before each E-step.
    Args:
                num_rows_of_image: Integer representing the number of rows in the image
                num_cols_of_image: Integer representing the number of columns in the image
                K: The number of latent states in the MRF
    Returns:
                q: 3-dimensional numpy matrix with shape [num_rows_of_image, num_cols_of_image, K]
     """
    q = np.random.random((num_rows_of_image, num_cols_of_image, K))
    for row_num in range(num_rows_of_image):
        for col_num in range(num_cols_of_image):
            q[row_num, col_num, :] = q[row_num, col_num, :]/sum(q[row_num, col_num, :])
    return q

def initialize_theta_parameters(K):
    """ Helper function to initialize theta before begining of EM.
    Args:
                K: The number of latent states in the MRF
    Returns:
                mu: A numpy vector of dimension [K] representing the mean for each of the K classes
                sigma: A numpy vector of dimension [K] representing the standard deviation for each of the K classes
    """
    mu = np.zeros(K)
    sigma = np.zeros(K) + 10
    for k in range(K):
        mu[k] = np.random.randint(10,240)
    return mu, sigma


class MRF(object):
    def __init__(self, J, K, n_em_iter, n_vi_iter):
        self.J = J
        self.K = K
        self.n_em_iter = n_em_iter
        self.n_vi_iter = n_vi_iter

    def get_neighbors(self, X, row, col):
        neighbors = []
        neighbors.append((row-1, col))
        neighbors.append((row, col+1))
        neighbors.append((row+1, col))
        neighbors.append((row, col+1))
        return neighbors

    def gauss(self, d, mu, sigma):
        return (1/(sigma*np.sqrt(2*math.pi))) * np.exp(-1 * 1/2 * ((d - mu) / sigma)**2)

    def KL(self, xs, neighbors, mu, sigma, q, k) -> int:
        neighbors = neighbors.T
        cur_liklihood = self.gauss(xs, mu[k], sigma[k])
        temp = np.array(q)
        temp.transpose(2,1,0)
        sum_over_neighbors = self.J * np.sum(q[k][neighbors[1], neighbors[0]])

        sum_over_allk = 0
        for i in range(self.K):
            norm = self.gauss(xs, mu[i], sigma[i])
            qsum = self.J * np.sum(q[i][neighbors[1], neighbors[0]])
            sum_over_allk += norm * np.exp(qsum)

        kl = (cur_liklihood * np.exp(sum_over_neighbors)) / sum_over_allk

        return kl

    def e_step(self, X, mu, sigma):
        q = initialize_variational_parameters(X.shape[0], X.shape[1], self.K)
        for m in range(self.n_vi_iter):
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    for k in range(self.K):
                        q[i, j, k] = self.KL(X[i, j], self.get_neighbors(X, i, j), mu, sigma, q, k)
        return q

    def m_step(self, X, mu, sigma):
        for k in range(self.K):



    def fit(self, *, X):
        mean, var = initialize_theta_parameters(self.K)
        # mean field estimate for this step
        q = self.e_step(X)
        print("yeet")


        """ Fit the model.
                Args:
                X: A matrix of floats with shape [num_rows_of_image, num_cols_of_image]
        """
        # TODO: Implement this!
        # Please use helper function 'initialize_theta_parameters' to initialize theta at the start of EM 
        #     Ex:  mu, sigma = initialize_theta_parameters(self.K)
        # Please use helper function 'initialize_variational_parameters' to initialize q at the start of each E step 
        #     Ex:  q = initialize_variational_parameters(X.shape[0], X.shape[1], self.K)
        raise Exception("You must implement this method!")

    def predict(self, X):
        """ Predict.
        Args:
                X: A matrix of floats with shape [num_rows_of_image, num_cols_of_image]

        Returns:
                A matrix of ints with shape [num_rows_of_image, num_cols_of_image].
                    - Each element of this matrix should be the most likely state according to the trained model for the pixel corresponding to that row and column
                    - States should be encoded as {0,..,K-1}
        """
        # TODO: Implement this!
        raise Exception("You must implement this method!")
