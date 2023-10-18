import numpy as np
from scipy.spatial.distance import mahalanobis

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

TEST_SUBSET_ROWS = 3
# test for make_s_subset_mask
def test_make_s_subset_mask():
    # set numpy random seed to 35
    np.random.seed(35)
    # create a random set of TEST_ROWS rows of 5 columns
    s_dist_thresholded = np.random.rand(TEST_SUBSET_ROWS, 5)
    # create a random set of TEST_ROWS rows of 5 columns
    k_dist_thresholded = np.random.rand(TEST_SUBSET_ROWS, 5)
    # create a random set of integers in TEST_ROWS rows of 5 columns
    s_dist_hard = np.random.randint(0, 2, size=(TEST_SUBSET_ROWS, 5))
    # create a random set of integers in TEST_ROWS rows of 5 columns
    k_dist_hard = np.random.randint(0, 2, size=(TEST_SUBSET_ROWS, 5))
    # calculate using make_s_subset_mask
    s_subset_mask = find_pairs.make_s_subset_mask(
        s_dist_thresholded,
        k_dist_thresholded,
        s_dist_hard,
        k_dist_hard,
        TEST_SUBSET_ROWS,
    )

    s_subset_hard = s_dist_hard[s_subset_mask]
    k_subset_hard = k_dist_hard

    print(f"s_subset_hard: {s_subset_hard}")
    print(f"k_subset_hard: {k_subset_hard}")

    s_subset_dist = s_dist_thresholded[s_subset_mask]
    k_subset_dist = k_dist_thresholded

    # check that for each row in s_subset_dist there is a row in k_subset_dist
    # that is less than the threshold for every column
    for i in range(s_subset_dist.shape[0]):
        one_below_threshold = False
        one_hard_match = False

        for k in range(k_subset_dist.shape[0]):
            if np.all(abs(k_subset_dist[k] - s_subset_dist[i]) < 1.0):
                one_below_threshold = True
            if np.allclose(k_subset_hard[k], s_subset_hard[i]):
                one_hard_match = True

        assert one_below_threshold, f"no corresponding row in k_subset_dist for {s_subset_dist[i]}"
        assert one_hard_match, f"no hard match for {s_subset_hard[i]}"

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