import numpy as np
import scipy
from sklearn.metrics import mutual_info_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVC
from collections import Counter

###############################################################################
#
# R4 and R4c scores (our contribution)
#
# These metrics quantify the extent to which every dimension of a ground-truth
# representation V can be mapped individually (via an invertible function) to
# dimensions of a learned representation Z. They accomplish this by considering
# the R^2 coefficient of determination in both directions and taking geometric
# means.
#
# The conditional version (R4c) also takes into account the hierarchy, scoping
# comparisons to cases where both learned and ground-truth factors are active,
# and not penalizing minor differences in the distribution of continuous dims.
#
###############################################################################

def activity_mask(v):
    # Slight kludge to detect activity; could pass a separate mask variable
    # instead
    return (np.abs(v) > 1e-10).astype(int)

def is_categorical(v, max_uniq=10):
    # Also kind of a kludge, but assume a variable is categorical if it's
    # integer-valued and there are few possible options. Could use the
    # hierarchy object instead.
    return len(np.unique(v)) <= max_uniq and np.allclose(v.astype(int), v)

def sample_R2_oneway(inputs, targets, reg=GradientBoostingRegressor, kls=GradientBoostingClassifier):
    if len(inputs) < 2:
        # Handle edge case of nearly empty input
        return 0

    x_train, x_test, y_train, y_test = train_test_split(inputs.reshape(-1,1), targets)
    n_uniq = min(len(np.unique(y_train)), len(np.unique(y_test)))

    if n_uniq == 1:
        # Handle edge case of only one target
        return 1 
    elif is_categorical(targets):
        # Use a classifier for categorical data
        y_train = y_train.astype(int)
        y_test = y_test.astype(int)
        model = kls()
    else:
        # Use a regressor otherwise
        model = reg()

    # Return the R^2 (or accuracy) score
    return model.fit(x_train, y_train).score(x_test, y_test)

def R2_oneway(inputs, targets, iters=5, **kw):
    # Repeatedly compute R^2 over random splits
    return np.mean([sample_R2_oneway(inputs, targets, **kw) for _ in range(iters)])

def R2_bothways(x, y):
    # Take the geometric mean of R^2 in both directions
    r1 = max(0, R2_oneway(x,y))
    r2 = max(0, R2_oneway(y,x))
    return np.sqrt(r1*r2)

def R4_scores(V, Z):
    # For each dimension, find the best R2_bothways
    scores = []

    for i in range(V.shape[1]):
        best = 0
        for j in range(Z.shape[1]):
            best = max(best, R2_bothways(V[:,i], Z[:,j]))
        scores.append(best)

    return scores

def R4c_scores(V, Z, Z_hier):
    # Assume that V and Z are vectors where categorical variables are now
    # represented as individual dimensions, which are [1,2,...] if active, and
    # 0 if inactive. Continuous variables are 0 if inactive.

    # Before we can compute R^4_c, we need to map particular nodes in the
    # hierarchy to dimensions in those flat vectors. Let's do this via these
    # slightly confusing recursive functions:

    def add_cont_indexes(hier, start=0):
        added = 0
        for node in hier:
            if node['type'] == 'continuous':
                node['index'] = start
                start += 1
                added += 1
            else:
                for child in node['options']:
                    extra = add_cont_indexes(child, start)
                    start += extra
                    added += extra
        return added

    def add_disc_indexes(hier, start=0):
        added = 0
        for node in hier:
            if node['type'] == 'categorical':
                node['index'] = start
                start += 1
                added += 1
                for child in node['options']:
                    extra = add_disc_indexes(child, start)
                    start += extra
                    added += extra
        return added

    def add_indexes(hier):
        add_cont_indexes(hier, add_disc_indexes(hier))

    add_indexes(Z_hier)

    # Now each variable dictionary in `Z_hier` contains a key-value pair giving
    # the index into `Z`.
    #
    # Next, we'll define a recursive function for computing the best
    # conditional R4 with respect to a group of variables, then apply it to the
    # root of the dimension hierarchy.

    def R4c_group(v, Z, group, mv=None):
        if mv is None:
            mv = activity_mask(v)

        on = np.argwhere(mv)[:,0]

        r2_max = 0

        for i, node in enumerate(group):
            z = Z[:,node['index']] # use the index we added

            # get the correspondence of this dimension
            r2_node = R2_bothways(v[on], z[on])

            if node['type'] == 'categorical':
                # get the best child correspondences down each branch, but
                # weight them by the probability that we actually go down that
                # branch with v active. 
                wgts = []
                vals = []
                for j, subgroup in enumerate(node['options']):
                    mvz = mv * (z == j+1) # 1 iff both v and child are active
                    wgts.append(np.sum(mvz)) # sum ~ prob(both are active)
                    vals.append(R4c_group(v, Z, subgroup, mvz)) # recurse
                wgts = np.array(wgts) / (np.sum(wgts) + 1e-10)
                vals = np.array(vals)

                # if the children have better correspondence than the
                # categorical dimension itself, use that correspondence instead
                r2_child = np.sum(vals * wgts)
                r2_node = max(r2_node, r2_child)

            # looking for the best correspondence across all possible paths
            r2_max = max(r2_max, r2_node)
            
        return r2_max

    # find the best R4c score for each dimension
    r4c_scores = []

    for i in range(V.shape[1]):
        r4c_scores.append(R4c_group(V[:,i], Z, Z_hier))

    return r4c_scores

###############################################################################
#
# Hierarchy / assignment correctness metrics for MIMOSA
#
###############################################################################

def purity_coverage(true_assigns, components):
    cassigns = [np.array([true_assigns[i] for i in c.index_list]) for c in components]
    counters = [Counter(ca) for ca in cassigns]
    maxes = [counter.most_common(1)[0][0] for counter in counters]
    purities  = [np.mean(ca == m) for ca, m in zip(cassigns, maxes)]
    misclasses = [np.sum(ca != m) for ca, m in zip(cassigns, maxes)]

    component_sizes = [len(c) for c in components]
    purity = 1 - (sum(misclasses) / sum(component_sizes))
    coverage = np.sum(component_sizes) / len(true_assigns)

    return {
        'component_sizes': component_sizes,
        'component_purities': purities,
        'component_misclasses': misclasses,
        'purity': purity,
        'coverage': coverage
    }

def H_error(hier1, hier2):
    from multiset import Multiset

    def min_dim(group):
        return min([
          sum([1 if dim['type'] == 'continuous' else min_dim(dim['options']) for dim in node])
          for node in group])
  
    def paths_in(hier, D=0, prefix=''):
        D2 = D + sum([n['type']=='continuous' for n in hier])
        children = []
        for n in hier:
            if n['type'] == 'categorical':
                children += n['options']

        if len(children):
            res = []
            for n in hier:
                if n['type'] == 'categorical':
                    opts = n['options']
                    for child in opts:
                        res += paths_in(child, D=D2, prefix=f"{prefix}c{D2+min_dim(opts)}->")
            return res
        else:
            return [prefix+f"{D2}D"]

    m1 = Multiset(paths_in(hier1))
    m2 = Multiset(paths_in(hier2))
    return len(m1 ^ m2)

###############################################################################
#
# Mutual Information Gap (MIG) Baseline
#
# Technically not defined for continuous targets, but we discretize with 20-bin
# histograms.
#
###############################################################################

def estimate_mutual_information(X, Y, bins=20):
  hist = np.histogram2d(X, Y, bins)[0] # approximate joint
  info = mutual_info_score(None, None, contingency=hist)
  return info / np.log(2) # bits

def estimate_entropy(X, **kw):
  return estimate_mutual_information(X, X, **kw)

def MIG(Z_true, Z_learned, **kw):
  K = Z_true.shape[1]
  gap = 0
  for k in range(K):
    H = estimate_entropy(Z_true[:,k], **kw)
    MIs = sorted([
      estimate_mutual_information(Z_learned[:,j], Z_true[:,k], **kw)
      for j in range(Z_learned.shape[1])
    ], reverse=True)
    gap += (MIs[0] - MIs[1]) / (H * K)
  return gap

###############################################################################
#
# SAP Score Baseline
#
###############################################################################

def SAP(V, Z):
    saps = []

    for i in range(V.shape[1]):
        v = V[:,i]
        
        if is_categorical(v):
            model = LinearSVC(C=0.01, class_weight="balanced")
            v = v.astype(int)
        else:
            model = LinearRegression()
        
        scores = []
        for j in range(Z.shape[1]):
            z = Z[:,j].reshape(-1,1)
            scores.append(model.fit(z,v).score(z,v))
        scores = list(sorted(scores))

        saps.append(scores[-1] - scores[-2])

    return np.mean(saps)

###############################################################################
#
# DCI (Disentanglement, Completeness, Informativeness) Baseline
#
# Code adapted from https://github.com/google-research/disentanglement_lib,
# original paper at https://openreview.net/forum?id=By-7dz-AZ.
#
###############################################################################

def DCI(gen_factors, latents):
  """Computes score based on both training and testing codes and factors."""
  mus_train, mus_test, ys_train, ys_test = train_test_split(gen_factors, latents, test_size=0.1)
  scores = {}
  importance_matrix, train_err, test_err = compute_importance_gbt(mus_train, ys_train, mus_test, ys_test)
  assert importance_matrix.shape[0] == mus_train.shape[1]
  assert importance_matrix.shape[1] == ys_train.shape[1]
  scores["informativeness_train"] = train_err
  scores["informativeness_test"] = test_err
  scores["disentanglement"] = disentanglement(importance_matrix)
  scores["completeness"] = completeness(importance_matrix)
  return scores

def compute_importance_gbt(x_train, y_train, x_test, y_test):
  """Compute importance based on gradient boosted trees."""
  num_factors = y_train.shape[1]
  num_codes = x_train.shape[1]
  importance_matrix = np.zeros(shape=[num_codes, num_factors],
                               dtype=np.float64)
  train_loss = []
  test_loss = []
  for i in range(num_factors):
    model = GradientBoostingRegressor()
    model.fit(x_train, y_train[:,i])
    importance_matrix[:, i] = np.abs(model.feature_importances_)
    train_loss.append(model.score(x_train, y_train[:,i]))
    test_loss.append(model.score(x_test, y_test[:,i]))
  return importance_matrix, np.mean(train_loss), np.mean(test_loss)


def disentanglement_per_code(importance_matrix):
  """Compute disentanglement score of each code."""
  # importance_matrix is of shape [num_codes, num_factors].
  return 1. - scipy.stats.entropy(importance_matrix.T + 1e-11,
                                  base=importance_matrix.shape[1])


def disentanglement(importance_matrix):
  """Compute the disentanglement score of the representation."""
  per_code = disentanglement_per_code(importance_matrix)
  if importance_matrix.sum() == 0.:
    importance_matrix = np.ones_like(importance_matrix)
  code_importance = importance_matrix.sum(axis=1) / importance_matrix.sum()

  return np.sum(per_code*code_importance)

def completeness_per_factor(importance_matrix):
  """Compute completeness of each factor."""
  # importance_matrix is of shape [num_codes, num_factors].
  return 1. - scipy.stats.entropy(importance_matrix + 1e-11,
                                  base=importance_matrix.shape[0])


def completeness(importance_matrix):
  """"Compute completeness of the representation."""
  per_factor = completeness_per_factor(importance_matrix)
  if importance_matrix.sum() == 0.:
    importance_matrix = np.ones_like(importance_matrix)
  factor_importance = importance_matrix.sum(axis=0) / importance_matrix.sum()
  return np.sum(per_factor*factor_importance)

###############################################################################
#
# FactorVAE Score Baseline
#
# Code adapted from https://github.com/google-research/disentanglement_lib,
# original paper at https://arxiv.org/abs/1802.05983
#
###############################################################################

def FactorVAE(ground_truth_X,
              ground_truth_Z,
              representation_function,
              random_state=np.random.RandomState(0),
              batch_size=64,
              num_train=4000,
              num_eval=2000,
              num_variance_estimate=1000):
  """Computes the FactorVAE disentanglement metric.

  Args:
    ground_truth_data: GroundTruthData to be sampled from.
    representation_function: Function that takes observations as input and
      outputs a dim_representation sized representation for each observation.
    random_state: Numpy random state used for randomness.
    batch_size: Number of points to be used to compute the training_sample.
    num_train: Number of points used for training.
    num_eval: Number of points used for evaluation.
    num_variance_estimate: Number of points used to estimate global variances.

  Returns:
    Dictionary with scores:
      train_accuracy: Accuracy on training set.
      eval_accuracy: Accuracy on evaluation set.
  """
  global_variances = _compute_variances(ground_truth_X,
                                        representation_function,
                                        num_variance_estimate, random_state)
  active_dims = _prune_dims(global_variances)
  scores_dict = {}

  if not active_dims.any():
    scores_dict["train_accuracy"] = 0.
    scores_dict["eval_accuracy"] = 0.
    scores_dict["num_active_dims"] = 0
    return scores_dict

  training_votes = _generate_training_batch(ground_truth_X, ground_truth_Z,
                                            representation_function, batch_size,
                                            num_train, random_state,
                                            global_variances, active_dims)
  classifier = np.argmax(training_votes, axis=0)
  other_index = np.arange(training_votes.shape[1])

  train_accuracy = np.sum(
      training_votes[classifier, other_index]) * 1. / np.sum(training_votes)

  eval_votes = _generate_training_batch(ground_truth_X, ground_truth_Z,
                                        representation_function, batch_size,
                                        num_eval, random_state,
                                        global_variances, active_dims)

  eval_accuracy = np.sum(eval_votes[classifier,
                                    other_index]) * 1. / np.sum(eval_votes)
  scores_dict["train_accuracy"] = train_accuracy
  scores_dict["eval_accuracy"] = eval_accuracy
  scores_dict["num_active_dims"] = len(active_dims)
  return scores_dict

def obtain_representation(observations, representation_function, batch_size):
  """"Obtain representations from observations.
  Args:
    observations: Observations for which we compute the representation.
    representation_function: Function that takes observation as input and
      outputs a representation.
    batch_size: Batch size to compute the representation.
  Returns:
    representations: Codes (num_codes, num_points)-Numpy array.
  """
  representations = None
  num_points = observations.shape[0]
  i = 0
  while i < num_points:
    num_points_iter = min(num_points - i, batch_size)
    current_observations = observations[i:i + num_points_iter]
    if i == 0:
      representations = representation_function(current_observations)
    else:
      representations = np.vstack((representations,
                                   representation_function(
                                       current_observations)))
    i += num_points_iter
  return np.transpose(representations)

def _prune_dims(variances, threshold=0.05):
  """Mask for dimensions collapsed to the prior."""
  scale_z = np.sqrt(variances)
  return scale_z >= threshold


def _compute_variances(ground_truth_X,
                       representation_function,
                       batch_size,
                       random_state,
                       eval_batch_size=64):
  """Computes the variance for each dimension of the representation.

  Args:
    ground_truth_data: GroundTruthData to be sampled from.
    representation_function: Function that takes observation as input and
      outputs a representation.
    batch_size: Number of points to be used to compute the variances.
    random_state: Numpy random state used for randomness.
    eval_batch_size: Batch size used to eval representation.

  Returns:
    Vector with the variance of each dimension.
  """
  observation_indexes = np.arange(len(ground_truth_X))
  np.random.shuffle(observation_indexes)
  observations = ground_truth_X[observation_indexes][:batch_size]
  representations = obtain_representation(observations,
                                                representation_function,
                                                eval_batch_size)
  representations = np.transpose(representations)
  assert representations.shape[0] == batch_size
  return np.var(representations, axis=0, ddof=1)

def _generate_training_sample(ground_truth_X, ground_truth_Z, representation_function,
                              batch_size, random_state, global_variances,
                              active_dims, tol=0.001):
  """Sample a single training sample based on a mini-batch of ground-truth data.

  Args:
    ground_truth_data: GroundTruthData to be sampled from.
    representation_function: Function that takes observation as input and
      outputs a representation.
    batch_size: Number of points to be used to compute the training_sample.
    random_state: Numpy random state used for randomness.
    global_variances: Numpy vector with variances for all dimensions of
      representation.
    active_dims: Indexes of active dimensions.

  Returns:
    factor_index: Index of factor coordinate to be used.
    argmin: Index of representation coordinate with the least variance.
  """
  # Select random coordinate to keep fixed.
  factor_index = random_state.randint(ground_truth_Z.shape[1])

  # Pick fixed factor value
  factor_value = np.random.choice(ground_truth_Z[:,factor_index])

  # Find indices of examples with closest values
  factor_diffs = np.abs(ground_truth_Z[:,factor_index]-factor_value)
  sorted_observation_indexes = factor_diffs.argsort()
  exact_observation_indexes = np.argwhere(factor_diffs == 0)[:,0]
  np.random.shuffle(exact_observation_indexes)
  if len(exact_observation_indexes) >= batch_size:
      # If there are enough which are exactly equal, shuffle
      observation_indexes = exact_observation_indexes[:batch_size]
  else:
      # If not, just pick all of the closest
      observation_indexes = sorted_observation_indexes[:batch_size]

  # Obtain the observations.
  observations = ground_truth_X[observation_indexes]

  representations = representation_function(observations)
  local_variances = np.var(representations, axis=0, ddof=1)
  argmin = np.argmin(local_variances[active_dims] /
                     global_variances[active_dims])
  return factor_index, argmin


def _generate_training_batch(ground_truth_X, ground_truth_Z, representation_function,
                             batch_size, num_points, random_state,
                             global_variances, active_dims):
  """Sample a set of training samples based on a batch of ground-truth data.

  Args:
    ground_truth_data: GroundTruthData to be sampled from.
    representation_function: Function that takes observations as input and
      outputs a dim_representation sized representation for each observation.
    batch_size: Number of points to be used to compute the training_sample.
    num_points: Number of points to be sampled for training set.
    random_state: Numpy random state used for randomness.
    global_variances: Numpy vector with variances for all dimensions of
      representation.
    active_dims: Indexes of active dimensions.

  Returns:
    (num_factors, dim_representation)-sized numpy array with votes.
  """
  votes = np.zeros((ground_truth_Z.shape[1], global_variances.shape[0]),
                   dtype=np.int64)
  for _ in range(num_points):
    factor_index, argmin = _generate_training_sample(ground_truth_X, ground_truth_Z,
                                                     representation_function,
                                                     batch_size, random_state,
                                                     global_variances,
                                                     active_dims)
    votes[factor_index, argmin] += 1
  return votes

