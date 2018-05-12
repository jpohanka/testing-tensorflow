import numpy as np
from scipy.stats import kendalltau
# import tensorflow as tf
from itertools import permutations


'''
Set Parameters and generate sample data
'''


class Config():

    sample_size = 2500
    lr = 0.001
    max_epochs = 1000
    batch_size = 100
    random_seed = 0
    dim = 5


config = Config()

sample_data = np.random.uniform(size=[config.sample_size, config.dim])


def isPSD(A, tol=1e-8):
    E, V = np.linalg.eigh(A)
    return np.all(E > -tol)


def copula_corr(data):
    _, dim = data.shape
    idx = list(permutations(range(dim), 2))
    tau = np.ones([dim, dim])

    for pairs in idx:
        tmp, _ = kendalltau(sample_data[:, pairs[0]], sample_data[:, pairs[1]])
        tau[pairs[0], pairs[1]] = tmp
    corr = np.sin(0.5*np.pi*tau)

    '''
    Algorithm 5.55 from McNeil, Frey, and Embrechts (2005) to make correlation
    matrix positive semidefinite.
    '''

    if isPSD(corr) is False:

        evals, E = np.linalg.eig(corr)
        evals = [1e-8 if e < 0 else e for e in evals]
        D = np.diag(evals)
        corr = np.matmul(np.matmul(E, D), np.transpose(E))

    return corr
