# -*- coding: utf-8 -*-

"""
Copyright (c) 2016 Randal S. Olson

Permission is hereby granted, free of charge, to any person obtaining a copy of this software
and associated documentation files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial
portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from __future__ import print_function
import numpy as np
from sklearn.neighbors import KDTree


class ReliefF(object):

    """Feature selection using data-mined expert knowledge.

    Based on the ReliefF algorithm as introduced in:

    Kononenko, Igor et al. Overcoming the myopia of inductive learning
    algorithms with RELIEFF (1997), Applied Intelligence, 7(1), p39-55

    """

    def __init__(self, n_neighbors=100, n_features_to_keep=None, alpha=0.1):
        """Sets up ReliefF to perform feature selection.

        Parameters
        ----------
        n_neighbors: int (default: 100)
            The number of neighbors to consider when assigning feature
            importance scores.
            More neighbors results in more accurate scores, but takes longer.
        n_features_to_keep: int (default: None)
            The number of features to retain when calling `transform()`
        alpha: float (default: 0.1)
            The amount of Type I error to tolerate when choosing a threshold
            `tau` to seperate relevant and irrelevant features.

        Returns
        -------
        None

        """

        self.feature_scores = None
        self.top_features = None
        self.tree = None
        self.n_neighbors = n_neighbors
        self.n_features_to_keep = n_features_to_keep
        self.alpha = alpha
        self.tau = 1 / np.sqrt(alpha * n_neighbors)

    def fit(self, X, y):
        """Computes the feature importance scores from the training data.

        Parameters
        ----------
        X: array-like {n_samples, n_features}
            Training instances to compute the feature importance scores from
        y: array-like {n_samples}
            Training labels

        Returns
        -------
        None

        """
        self.feature_scores = np.zeros(X.shape[1], dtype=np.float64)
        self.tree = KDTree(X)

        # Find nearest k neighbors of all points. The tree contains the query
        # points, so we discard the first match for all points (first column).
        indices = self.tree.query(X, k=self.n_neighbors+1,
                                  return_distance=False)[:, 1:]

        for (source, nn) in enumerate(indices):
            # Create a binary array that is 1 when the sample and neighbors
            # match and -1 everywhere else, for labels and features.
            labels_match = np.equal(y[source], y[nn]) * 2 - 1
            features_match = np.equal(X[source], X[nn]) * 2 - 1

            # The change in feature_scores is the dot product of these arrays
            self.feature_scores += np.dot(features_match.T, labels_match)

        # Normalize `feature_scores` between -1 and 1
        self.feature_scores = self.feature_scores / (self.n_neighbors * X.shape[0])
        # Compute indices of top features
        self.top_features = np.argsort(self.feature_scores)[::-1]

    def transform(self, X):
        """Reduces the feature set down to the top `n_features_to_keep` features.
        Or returns features above `tau` threshold if `alpha` was given.

        Parameters
        ----------
        X: array-like {n_samples, n_features}
            Feature matrix to perform feature selection on

        Returns
        -------
        X_reduced: array-like {n_samples, n_features_to_keep}
            Reduced feature matrix

        """
        if self.n_features_to_keep is not None:
            return X[:, self.top_features[:self.n_features_to_keep]]
        elif self.alpha is not None:
            return X[:, self.feature_scores > self.tau]
        else:
            raise NotImplementedError('No alternative methods implemented!')

    def fit_transform(self, X, y):
        """Computes the feature importance scores from the training data, then
        reduces the feature set down to the top `n_features_to_keep` features.

        Parameters
        ----------
        X: array-like {n_samples, n_features}
            Training instances to compute the feature importance scores from
        y: array-like {n_samples}
            Training labels

        Returns
        -------
        X_reduced: array-like {n_samples, n_features_to_keep}
            Reduced feature matrix

        """
        self.fit(X, y)
        return self.transform(X)
