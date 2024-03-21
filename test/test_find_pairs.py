import numpy as np
from scipy.spatial.distance import mahalanobis
import pandas as pd

from methods.matching import find_pairs

# test that tests batch_mahalanobis matches the scipy implementation
def test_batch_mahalanobis():
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

TEST_SUBSET_ROWS = 5
# test for make_s_set_mask
def test_make_s_set_mask():

    k_dist_hard = np.array([
        [1, 2],
        [7, 8],
        [1, 2],
        [7, 8],
    ], dtype=np.float32)

    k_dist_thresholded = np.array([
        [0.1, 0.2],
        [0.7, 0.8],
        [20.0, 21.0],
        [20.0, 21.0],
    ], dtype=np.float32)

    m_dist_hard = np.array([
        [1, 2],
        [3, 4],
        [12, 13],
        [100, 101],
        [150, 151],
    ], dtype=np.float32)

    m_dist_thresholded = np.array([
        [0.1, 0.2],
        [0.3, 0.4],
        [0.7, 0.8],
        [0.9, 0.1],
        [0.5, 0.6],
    ], dtype=np.float32)

    rng = np.random.default_rng(42)
    k_set = np.array([k_dist_hard, k_dist_thresholded])
    k_set = np.moveaxis(k_set, 0, 1).reshape(-1, k_dist_hard.shape[1] + k_dist_thresholded.shape[1])
    k_set = pd.DataFrame(k_set)

    m_set = np.array([m_dist_hard, m_dist_thresholded])
    m_set = np.moveaxis(m_set, 0, 1).reshape(-1, m_dist_hard.shape[1] + m_dist_thresholded.shape[1])
    m_set = pd.DataFrame(m_set)

    hard_match_columns = list(range(k_dist_hard.shape[1]))
    hard_match_categories = {k.tobytes(): k for k in k_dist_hard}

    # calculate using make_s_set_mask
    s_subset_mask, misses = find_pairs.make_s_set_mask(
        rng,
        k_set,
        m_set,
        m_dist_thresholded,
        k_dist_thresholded,
        hard_match_columns,
        2,
        hard_match_categories,
    )

    assert (s_subset_mask == [True, False, False, False, False]).all()
    assert (misses == [False, True, True, True]).all()

TEST_ROWS_ALL_TRUE = 35
# test for rows_all_true
def test_rows_all_true():
    # set numpy random seed to 35
    np.random.seed(35)
    # create a matrix with TEST_ROWS_ALL_TRUE rows and 1 column with a random boolean in it
    rows = np.random.randint(0, 2, size=(TEST_ROWS_ALL_TRUE, 1), dtype=np.bool_)
    # calculate using rows_all_true
    all_true = find_pairs.rows_all_true(rows)
    # check this matches numpy.all with axis=1
    assert np.all(all_true == np.all(rows, axis=1))