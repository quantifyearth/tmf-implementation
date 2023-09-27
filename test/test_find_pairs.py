import pytest
from methods.matching import find_pairs
import numpy as np
from scipy.spatial.distance import mahalanobis

# test that tests batch_mahalanobis matches the scipy implementation
def test_batch_mahalanovis():
    # set numpy random seed to 35
    np.random.seed(35)
    # create a random set of 3 rows of 5 columns
    rows = np.random.rand(3, 5)
    # create a random vector of 5 columns
    vector = np.random.rand(5)
    # create a random 5x5 inverse covariance matrix
    invcov = np.random.rand(5, 5)
    # calculate the scipy implementation
    scipy_dists = [mahalanobis(row, vector, invcov) for row in rows]
    # calculate the batch implementation
    batch_dists = np.sqrt(find_pairs.batch_mahalanobis_squared(rows, vector, invcov))
    # check that the results are the same
    assert np.allclose(scipy_dists, batch_dists)