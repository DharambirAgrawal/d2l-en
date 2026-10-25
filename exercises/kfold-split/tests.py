import numpy as np

from submission import kfold_indices, log_rmse
from grader import expect_close, expect_equal, expect_true


def test_even():
    """10 examples, 5 folds: consecutive pairs"""
    folds = kfold_indices(10, 5)
    expect_equal(len(folds), 5)
    expect_equal([list(v) for _, v in folds], [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]])
    expect_equal(list(folds[1][0]), [0, 1, 4, 5, 6, 7, 8, 9])


def test_remainder():
    """11 examples, 3 folds: sizes 4, 4, 3, nothing left out"""
    folds = kfold_indices(11, 3)
    expect_equal([len(v) for _, v in folds], [4, 4, 3])
    allv = np.concatenate([v for _, v in folds])
    expect_equal(sorted(allv.tolist()), list(range(11)))


def test_disjoint():
    """Train and validation never overlap and together cover everything"""
    for n, k in ((1460, 5), (17, 4), (5, 5)):
        for tr, va in kfold_indices(n, k):
            expect_true(len(set(tr) & set(va)) == 0, 'train and validation overlap')
            expect_equal(len(tr) + len(va), n)
            expect_true(np.all(np.diff(tr) > 0), 'train indices should be increasing')


def test_log_rmse():
    """Log RMSE, with predictions clipped at 1"""
    expect_close(log_rmse(np.array([110000.0, 180000.0]), np.array([100000.0, 200000.0])),
                 np.sqrt((np.log(1.1) ** 2 + np.log(0.9) ** 2) / 2))
    expect_close(log_rmse(np.array([-5.0]), np.array([np.e])), 1.0)
    expect_close(log_rmse(np.array([3.0, 4.0]), np.array([3.0, 4.0])), 0.0)
