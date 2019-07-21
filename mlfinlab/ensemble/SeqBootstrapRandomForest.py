"""
A edited class of RandomForest to implement Sequential Bootstrapping
"""

from warnings import catch_warnings, simplefilter, warn

import numpy as np
from numpy import float32
from numpy import float64
from scipy.sparse import issparse

from sklearn.utils._joblib import Parallel, delayed
from sklearn.utils import check_random_state, check_array, compute_sample_weight
from sklearn.exceptions import DataConversionWarning
from sklearn.utils.fixes import _joblib_parallel_args
from sklearn.ensemble import RandomForestClassifier

from mlfinlab.sampling.bootstrapping import get_ind_matrix
from mlfinlab.sampling.bootstrapping import seq_bootstrap
# ———————————————————————————————————————

DTYPE = float32
DOUBLE = float64
MAX_INT = np.iinfo(np.int32).max


def _generate_sample_indices(random_state, n_samples):
    """Private function used to _parallel_build_trees function."""
    random_instance = check_random_state(random_state)
    sample_indices = random_instance.randint(0, n_samples, n_samples)

    return sample_indices


def _generate_unsampled_indices(random_state, n_samples):
    """Private function used to forest._set_oob_score function."""
    sample_indices = _generate_sample_indices(random_state, n_samples)
    sample_counts = np.bincount(sample_indices, minlength=n_samples)
    unsampled_mask = sample_counts == 0
    indices_range = np.arange(n_samples)
    unsampled_indices = indices_range[unsampled_mask]

    return unsampled_indices


def custom_parallel_build_trees(tree, forest, features, target, sample_weight, tree_idx, n_trees,
                                verbose=0, class_weight=None):
    # Edited to include Sequential Bootstraping Case
    """Private function used to fit a single tree in parallel."""
    if verbose > 1:
        print("building tree %d of %d" % (tree_idx + 1, n_trees))

    if forest.triple_barrier_events is not None:
        n_samples = features.shape[0]
        if sample_weight is None:
            curr_sample_weight = np.ones((n_samples,), dtype=np.float64)
        else:
            curr_sample_weight = sample_weight.copy()
        # Create Index Matrix
        index_matrix = get_ind_matrix(forest.triple_barrier_events)
        # Select indices with Sequential Bootstrap
        indices = seq_bootstrap(index_matrix, compare=False, verbose=False, warmup_samples=None)
        sample_counts = np.bincount(indices, minlength=n_samples)
        curr_sample_weight *= sample_counts
        if class_weight == 'subsample':
            with catch_warnings():
                simplefilter('ignore', DeprecationWarning)
                curr_sample_weight *= compute_sample_weight('auto', target, indices)
        elif class_weight == 'balanced_subsample':
            curr_sample_weight *= compute_sample_weight('balanced', target, indices)

        tree.fit(features, target, sample_weight=curr_sample_weight, check_input=False)
    else:
        tree.fit(features, target, sample_weight=sample_weight, check_input=False)

    return tree


class SeqBootstrapRandomForest(RandomForestClassifier):
    """
    Random Forest Class using sequential bootstrapping
    Add seq_bootstrapping parameter to constructor function
    """
    def __init__(self,
                 n_estimators='warn',
                 criterion="gini",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features="auto",
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.,
                 min_impurity_split=None,
                 triple_barrier_events=None,
                 oob_score=False,
                 n_jobs=None,
                 random_state=None,
                 verbose=0,
                 warm_start=False,
                 class_weight=None):
        super().__init__(
                        n_estimators=n_estimators,
                        oob_score=oob_score,
                        n_jobs=n_jobs,
                        random_state=random_state,
                        verbose=verbose,
                        warm_start=warm_start,
                        class_weight=class_weight)

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split
        self.triple_barrier_events = triple_barrier_events

    def fit(self, features, target, sample_weight=None):
        # Edit original fit method to use custom _parallel_build_trees function (line: 279)

        """Build a forest of trees from the training set (features, target).
        Parameters
        ----------
        features : array-like or sparse matrix of shape = [n_samples, n_features]
            The training input samples. Internally, its dtype will be converted
            to ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csc_matrix``.
        target : array-like, shape = [n_samples] or [n_samples, n_outputs]
            The features values (class labels in classification, real numbers in
            regression).
        sample_weight : array-like, shape = [n_samples] or None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node. In the case of
            classification, splits are also ignored if they would result in any
            single class carrying a negative weight in either child node.
        Returns
        -------
        self : object
        """

        if self.n_estimators == 'warn':
            warn("The default value of n_estimators will change from "
                 "10 in version 0.20 to 100 in 0.22.", FutureWarning)
            self.n_estimators = 10

        # Validate or convert input data
        features = check_array(features, accept_sparse="csc", dtype=DTYPE)
        target = check_array(target, accept_sparse='csc', ensure_2d=False, dtype=None)
        if sample_weight is not None:
            sample_weight = check_array(sample_weight, ensure_2d=False)
        if issparse(features):
            # Pre-sort indices to avoid that each individual tree of the
            # ensemble sorts the indices.
            features.sort_indices()

        # Remap output
        self.n_features_ = features.shape[1]

        target = np.atleast_1d(target)
        if target.ndim == 2 and target.shape[1] == 1:
            warn("A column-vector target was passed when a 1d array was"
                 " expected. Please change the shape of target to "
                 "(n_samples,), for example using ravel().",
                 DataConversionWarning, stacklevel=2)

        if target.ndim == 1:
            # reshape is necessary to preserve the data contiguity against vs
            # [:, np.newaxis] that does not.
            target = np.reshape(target, (-1, 1))

        self.n_outputs_ = target.shape[1]

        target, expanded_class_weight = self._validate_y_class_weight(target)

        if getattr(target, "dtype", None) != DOUBLE or not target.flags.contiguous:
            target = np.ascontiguousarray(target, dtype=DOUBLE)

        if expanded_class_weight is not None:
            if sample_weight is not None:
                sample_weight = sample_weight * expanded_class_weight
            else:
                sample_weight = expanded_class_weight

        # Check parameters
        self._validate_estimator()

        if not self.bootstrap and self.oob_score:
            raise ValueError("Out of bag estimation only available"
                             " if bootstrap=True")

        random_state = check_random_state(self.random_state)

        if not self.warm_start or not hasattr(self, "estimators_"):
            # Free allocated memory, if any
            self.estimators_ = []

        n_more_estimators = self.n_estimators - len(self.estimators_)

        if n_more_estimators < 0:
            raise ValueError('n_estimators=%d must be larger or equal to '
                             'len(estimators_)=%d when warm_start==True'
                             % (self.n_estimators, len(self.estimators_)))
        elif n_more_estimators == 0:
            warn("Warm-start fitting without increasing n_estimators does not "
                 "fit new trees.")
        else:
            if self.warm_start and len(self.estimators_) > 0:
                # We draw from the random state to get the random state we
                # would have got if we hadn't used a warm_start.
                random_state.randint(MAX_INT, size=len(self.estimators_))

            trees = [self._make_estimator(append=False,
                                          random_state=random_state)
                     for i in range(n_more_estimators)]

            # Parallel loop: we prefer the threading backend as the Cython code
            # for fitting the trees is internally releasing the Python GIL
            # making threading more efficient than multiprocessing in
            # that case. However, for joblib 0.12+ we respect any
            # parallel_backend contexts set at a higher level,
            # since correctness does not rely on using threads.
            trees = Parallel(n_jobs=self.n_jobs,
                             verbose=self.verbose,
                             **_joblib_parallel_args(prefer='threads'))\
                (delayed(custom_parallel_build_trees)(t,
                                                      self,
                                                      features,
                                                      target,
                                                      sample_weight,
                                                      i,
                                                      len(trees),
                                                      verbose=self.verbose, class_weight=self.class_weight)
                 for i, t in enumerate(trees))

            # Collect newly grown trees
            self.estimators_.extend(trees)

        if self.oob_score:
            self._set_oob_score(features, target)

        # Decapsulate classes_ attributes
        if hasattr(self, "classes_") and self.n_outputs_ == 1:
            self.n_classes_ = self.n_classes_[0]
            self.classes_ = self.classes_[0]

        return self
