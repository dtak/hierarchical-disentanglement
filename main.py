import os
import sys
import re
import time
import json
import pickle
import argparse
import numpy as np
import tensorflow as tf
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import metrics
from mimosa import MIMOSA
from cofhae import HierarchicalAutoencoder
from helpers import property_cached, timer, batch_eval, NumpyEncoder
from helpers import plot_representation, plot_components

parser = argparse.ArgumentParser()

# Global config
parser.add_argument('--output_dir', type=str, default='')
parser.add_argument('--num_epochs', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--print_every', type=int, default=10)

# Dataset config
parser.add_argument('--dataset', type=str, default='')
parser.add_argument('--test_frac', type=float, default=0.1)

# MIMOSA config
# ...Initial smooth AE
parser.add_argument('--initial_dim', type=int, default=4)
# ...LocalSVD
parser.add_argument('--num_nearest_neighbors', default=40, type=int)
parser.add_argument('--ransac_frac', default=0.6667, type=float)
parser.add_argument('--contagion_num', default=5, type=int)
parser.add_argument('--eig_cumsum_thresh', default=0.95, type=float)
parser.add_argument('--eig_decay_thresh', default=4, type=int)
# ...BuildComponent
parser.add_argument('--cos_simil_thresh', default=0.99, type=float)
# ...MergeComponents
parser.add_argument('--min_size_init', default=20, type=int)
parser.add_argument('--min_size_merged', default=2000, type=int)
# ...ConstructHierarchy
parser.add_argument('--neighbor_lengthscale_mult', default=10.0, type=float)

# COFHAE config
parser.add_argument('--skip_cofhae', type=int, default=0)
parser.add_argument('--softmax_temperature', type=float, default=1.0)
parser.add_argument('--adversarial_penalty', type=float, default=1.0)
parser.add_argument('--assignment_penalty', type=float, default=1000.0)

FLAGS = parser.parse_args()

# Set up path to save model artifacts, possibly suffixed with an experiment ID
path = FLAGS.output_dir or f"/tmp/{int(time.time())}"
os.system('mkdir -p ' + path)

with open(os.path.join(path, 'flags.json'), 'w') as f:
    f.write(json.dumps(FLAGS.__dict__))

with timer("loading data"):
    if 'chopsticks' in FLAGS.dataset:
        from chopsticks import Chopsticks
        m = re.search(r'depth(\d)_([a-z]+)', FLAGS.dataset)
        depth = int(m.group(1))
        variant = m.group(2)
        noise = 0
        if 'noise' in FLAGS.dataset:
            noise = float(re.search(r'noise([0-9\.]+)', FLAGS.dataset).group(1))
        dataset = Chopsticks(depth, variant, noise)
    elif FLAGS.dataset == 'spaceshapes':
        from spaceshapes import Spaceshapes
        dataset = Spaceshapes()
    else:
        raise ValueError(f"Unrecognized dataset {FLAGS.dataset} -- should either be 'spaceshapes' or a Chopsticks variant string with a depth and slope/inter/either/both, e.g. 'chopsticks_depth3_both' or 'chopsticks_depth2_slope'.")

    true_hier = dataset.hierarchy
    data = dataset.data
    X = data.X
    A = data.A
    ground_truth_factors = data.AMZ

    order = np.arange(len(X))
    rs = np.random.RandomState(seed=0)
    rs.shuffle(order)
    n_test = int(FLAGS.test_frac * len(X))
    trn = order[n_test:]
    tst = order[:n_test]

    path_indices = {}
    true_assigns = np.empty(len(trn), dtype=int)
    for i, a in enumerate(A[trn]):
      p = tuple([aaa for aa in a for aaa in aa])
      if p not in path_indices:
        path_indices[p] = len(path_indices)
      true_assigns[i] = path_indices[p]

class Autoencoder(HierarchicalAutoencoder):
    @property_cached
    def X_in(self):
        return tf.placeholder("float", [None, X.shape[1]], name=self.name+'/X_in')

    @property
    def reconstruction_error(self):
        if 'spaceshapes' in FLAGS.dataset:
            return self.binary_xent
        else:
            return 100 * self.mean_sq_error

    @property
    def recons_transform(self):
        if 'spaceshapes' in FLAGS.dataset:
            return tf.nn.sigmoid
        else:
            return tf.identity

    @property
    def activation(self):
        return tf.nn.softplus

    def pre_encode(self, X_in):
        if 'chopsticks' in FLAGS.dataset:
            kw = dict(activation=self.activation, reuse=tf.AUTO_REUSE)
            L1_e = tf.layers.dense(X_in, 256, name=self.name+'/main/encoder/l1', **kw)
            L2_e = tf.layers.dense(L1_e, 256, name=self.name+'/main/encoder/l2', **kw)
            return L2_e
        elif 'spaceshapes' in FLAGS.dataset:
            from conv_helpers import create_recognition_network as conv_encoder
            return conv_encoder(X_in, pre=True, prefix=self.name+'/main', nonlinearity=self.activation)
        else:
            raise ValueError(f"Unrecognized dataset {FLAGS.dataset}")

    def decode(self, Z):
        if 'chopsticks' in FLAGS.dataset:
            kw = dict(activation=self.activation, reuse=tf.AUTO_REUSE)
            L1_d = tf.layers.dense(Z,    256, name=self.name+'/main/decoder/l1', **kw)
            L2_d = tf.layers.dense(L1_d, 256, name=self.name+'/main/decoder/l2', **kw)
            L3_d = tf.layers.dense(L2_d, X.shape[1], name=self.name+'/main/decoder/l3',
                                   activation=None, reuse=tf.AUTO_REUSE)
            return L3_d
        elif 'spaceshapes' in FLAGS.dataset:
            from conv_helpers import create_generator_network as conv_decoder
            return conv_decoder(Z, prefix=self.name+'/main')
        else:
            raise ValueError(f"Unrecognized dataset {FLAGS.dataset}")

sess = tf.Session()

if os.path.exists(f"{path}/assignments.pkl"):
    with open(f"{path}/hier.json", 'r') as f:
        hier = json.loads(f.read())
    with open(f"{path}/assignments.pkl", 'rb') as f:
        assignments = pickle.load(f)
else:
    train_kwargs = dict(
        print_every=FLAGS.print_every,
        num_epochs=FLAGS.num_epochs,
        batch_size=FLAGS.batch_size)

    with timer("Initial dimensionality reduction"):
        flat_dimensions = [{"type": "continuous"} for _ in range(FLAGS.initial_dim)]
        flat_ae = Autoencoder(flat_dimensions)
        flat_ae.train_normally(sess, X[trn], **train_kwargs)
        Z_init = batch_eval(flat_ae.learned_dimensions, sess, [(flat_ae.X_in, X[trn])])

    plot_representation(Z_init, true_assigns, f"{path}/initial_representation.png")

    mimosa_kwargs = dict(
        num_nearest_neighbors=FLAGS.num_nearest_neighbors,
        eig_cumsum_thresh=FLAGS.eig_cumsum_thresh,
        eig_decay_thresh=FLAGS.eig_decay_thresh,
        cos_simil_thresh=FLAGS.cos_simil_thresh,
        ransac_frac=FLAGS.ransac_frac,
        contagion_num=FLAGS.contagion_num,
        min_size_init=FLAGS.min_size_init,
        min_size_merged=FLAGS.min_size_merged,
        neighbor_lengthscale_mult=FLAGS.neighbor_lengthscale_mult)

    with timer("MIMOSA"):
        components1, components2, hier, assignments = MIMOSA(Z_init, **mimosa_kwargs)

    time_keys = ['local_ball_tree', 'local_neighborhoods', 'edge_mask']
    time_vals = defaultdict(float)
    for c in components1:
        for key in time_keys:
            if hasattr(c, f"_{key}_time"):
                time_vals[key] += getattr(c, f"_{key}_time")
    for key, val in time_vals.items():
        print(f"{key} took {val}s")

    with timer("saving MIMOSA outputs"):
        for comps, key in [(components1, 'initial_components'),
                           (components2, 'merged_components')]:
            with open(f"{path}/{key}.pkl", 'wb') as f:
                pickle.dump(comps, f)

            plot_components(comps, true_assigns, f"{path}/{key}.png")

        with open(f"{path}/mimosa_metrics.json", 'w') as f:
            mimosa_metrics = metrics.purity_coverage(true_assigns, components2)
            mimosa_metrics['H_error'] = metrics.H_error(hier, true_hier)
            f.write(json.dumps(mimosa_metrics, indent=4, cls=NumpyEncoder))

        with open(f"{path}/hier.json", 'w') as f:
            f.write(json.dumps(hier, indent=4))

        with open(f"{path}/assignments.pkl", 'wb') as f:
            pickle.dump(assignments, f)

if FLAGS.skip_cofhae:
    exit()

if os.path.exists(f"{path}/vals.pkl"):
    hier_ae = Autoencoder(hier)
    hier_ae.load(f"{path}/vals.pkl", sess)
else:
    cofhae_kwargs = dict(
        softmax_temperature=FLAGS.softmax_temperature,
        assignment_penalty=FLAGS.assignment_penalty,
        adversarial_penalty=FLAGS.adversarial_penalty,
        print_every=FLAGS.print_every,
        num_epochs=FLAGS.num_epochs,
        batch_size=FLAGS.batch_size)

    with timer("COFHAE"):
        hier_ae = Autoencoder(hier)
        hier_ae.train_with_COFHAE(sess, X[trn], assignments, **cofhae_kwargs)
    hier_ae.save(path, sess)

def compute_Z(x):
    return batch_eval(hier_ae.learned_dimensions, sess,
            [(hier_ae.X_in, x), (hier_ae.temper, 1e-10)])

def compute_mse(x):
    return batch_eval(hier_ae.mean_sq_error, sess,
            [(hier_ae.X_in, x), (hier_ae.temper, 1e-10)])

def compute_assign_loss(x, a):
    return batch_eval(hier_ae.assignment_loss, sess,
            [(hier_ae.X_in, x), (hier_ae.temper, 1e-10)] + list(zip(hier_ae.a_in, a)))

Z_true = ground_truth_factors[tst]
Z_test = compute_Z(X[tst])

np.save(path + '/Z_test.npy', Z_test)

with timer("Computing disentanglement metrics"):
    cofhae_metrics = {
        'R4': metrics.R4_scores(Z_true, Z_test),
        'R4c': metrics.R4c_scores(Z_true, Z_test, hier),
        'MIG': metrics.MIG(Z_true, Z_test),
        'SAP': metrics.SAP(Z_true, Z_test),
        'DCI': metrics.DCI(Z_true, Z_test),
        'FVAE': metrics.FactorVAE(X[tst], Z_test, compute_Z),
        'test_mse': compute_mse(X[tst]),
        'train_mse': compute_mse(X[trn]),
        'train_assign_err': compute_assign_loss(X[trn], assignments)
    }

    with open(f"{path}/cofhae_metrics.json", 'w') as f:
        f.write(json.dumps(cofhae_metrics, indent=4, cls=NumpyEncoder))

