"""
    Unit tests for ReliefF.
"""

from ReliefF import ReliefF
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split

def test_init():
    """Make sure ReliefF instantiates correctly"""
    fs = ReliefF(n_neighbors=50, n_features_to_keep=100)
    assert fs.n_neighbors == 50
    assert fs.n_features_to_keep == 100

def test_fit():
    """Make sure ReliefF fits correctly"""
    data = pd.read_csv('data/GAMETES-test.csv.gz')
    X_train, X_test, y_train, y_test = train_test_split(data.drop('class', axis=1).values,
                                                        data['class'].values,
                                                        random_state=34895)

    fs = ReliefF(n_neighbors=100, n_features_to_keep=5)
    fs.fit(X_train, y_train)

    with np.load("data/test_arrays.npz") as arrays:
        correct_top_features = arrays['correct_top_features']
        correct_feature_scores = arrays['correct_feature_scores']

    assert np.all(np.equal(fs.top_features, correct_top_features))
    assert np.all(np.equal(fs.feature_scores, correct_feature_scores))

def test_transform():
    """Make sure ReliefF transforms correctly"""
    data = pd.read_csv('data/GAMETES-test.csv.gz')
    X_train, X_test, y_train, y_test = train_test_split(data.drop('class', axis=1).values,
                                                        data['class'].values,
                                                        random_state=34895)

    fs = ReliefF(n_neighbors=100, n_features_to_keep=5)
    fs.fit(X_train, y_train)
    X_test = fs.transform(X_test)

    assert np.all(np.equal(X_test[0], np.array([0, 1, 1, 1, 1])))
    assert np.all(np.equal(X_test[1], np.array([2, 1, 0, 1, 1])))
    assert np.all(np.equal(X_test[-2], np.array([1, 1, 0, 1, 0])))
    assert np.all(np.equal(X_test[-1], np.array([1, 0, 1, 0, 0])))

def test_fit_transform():
    """Make sure ReliefF fit_transforms correctly"""
    data = pd.read_csv('data/GAMETES-test.csv.gz')
    X_train, X_test, y_train, y_test = train_test_split(data.drop('class', axis=1).values,
                                                        data['class'].values,
                                                        random_state=34895)

    fs = ReliefF(n_neighbors=100, n_features_to_keep=5)
    X_train = fs.fit_transform(X_train, y_train)

    assert np.all(np.equal(X_train[0], np.array([1, 1, 0, 2, 1])))
    assert np.all(np.equal(X_train[1], np.array([0, 0, 0, 2, 0])))
    assert np.all(np.equal(X_train[-2], np.array([1, 1, 0, 1, 0])))
    assert np.all(np.equal(X_train[-1], np.array([0, 0, 0, 0, 0])))
