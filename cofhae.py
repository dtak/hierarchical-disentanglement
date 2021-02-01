import os
import sys
import uuid
import pickle
import numpy as np
import tensorflow as tf
from collections import OrderedDict
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from helpers import minibatch, OpenStruct, property_cached, reinitialize_variables

class Discriminator(object):
    def __init__(self, name, real_inputs, fake_inputs, activation=tf.nn.leaky_relu, layer_width=1000, num_layers=3):
        self.activation = activation
        self.layer_width = layer_width
        self.num_layers = num_layers
        self.name = name
        self.real_inputs = real_inputs
        self.fake_inputs = fake_inputs
        self.fake_logits = self.forward(fake_inputs)
        self.real_logits = self.forward(real_inputs)
        self.fake_xent = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.fake_logits,
                                                    labels=tf.zeros_like(self.fake_logits)))
        self.real_xent = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.real_logits,
                                                    labels=tf.ones_like(self.real_logits)))
        self.discriminator_loss = 0.5 * (self.fake_xent + self.real_xent)
        self.generator_loss = tf.reduce_mean(self.real_logits)
        self.fool_rate = tf.reduce_mean(tf.cast(tf.less(self.real_logits, 0), tf.float32))

    def forward(self, x):
        for i in range(self.num_layers):
          x = tf.layers.dense(x, self.layer_width,
                              activation=self.activation, name=f"{self.name}/L{i}", reuse=tf.AUTO_REUSE)
        return tf.layers.dense(x, 1, activation=None, name=self.name+'/logits', reuse=tf.AUTO_REUSE)

# Recursively define Tensorflow operations to map an initial preactivation
# vector `Z_pre` to a set of categorical dimensions and associated soft masks
# for the continuous dimensions
def hierarchical_assignments(Z_pre, scope, hier, temper):
    M = [] # masks
    A = [] # assignments
    ones = tf.reshape(tf.ones(tf.shape(Z_pre)[0]), [-1,1]) # vector of 1s for initial mask

    for i, node in enumerate(hier):
        if node['type'] == 'continuous':
            M.append(ones)
        elif node['type'] == 'categorical':
            subscope = scope+'/categ{}'.format(i)

            # Add a linear layer predicting the value of this categorical dim.
            logits = tf.layers.dense(Z_pre, len(node['options']), activation=None, name=subscope+'_logits', reuse=tf.AUTO_REUSE)

            # Take a softmax with a configurable temperature
            assigns = tf.nn.softmax(logits / temper)

            A.append(assigns)

            # Recurse over children
            for j, subhier in enumerate(node['options']):
                if len(subhier):
                    M2, A2 = hierarchical_assignments(Z_pre, f"{subscope}_{j}", subhier, temper)

                    # Use the softmax value corresponding to this child as a mask
                    m = tf.reshape(assigns[:,j], [-1,1])

                    if M2 != []:
                        M.append(M2 * m)
                    if A2:
                        A = A + [a * m for a in A2]

    if M: M = tf.concat(M, axis=1) # stack 1D masks together

    return M, A

# Helper method to recursively compute the number of continuous variables in a
# hierarchy
def num_continuous_dims(hier):
    n = 0
    for i, node in enumerate(hier):
        if node['type'] == 'continuous':
            n += 1
        elif node['type'] == 'categorical':
            for j, subhier in enumerate(node['options']):
                if len(subhier):
                    n += num_continuous_dims(subhier)
    return n

# Helper method to permute the values of a matrix `Z` across its batch
# dimension, but only in places where a corresponding mask matrix `M` is
# nonzero.
def permute_active(Z,M):
    res = []
    for i in range(Z.shape[1]):
        zi = np.array(Z[:,i])
        mi = M[:,i]
        idx1 = np.argwhere(mi != 0)[:,0]
        idx2 = np.array(idx1)
        np.random.shuffle(idx2)
        zi[idx1] = zi[idx2]
        res.append(zi)
    return np.array(res).T

# Class to represent our hierarchical autoencoder
class HierarchicalAutoencoder(object):
    def __init__(self, hierarchy, name=None):
        self.hier = hierarchy
        self.name = name or str(uuid.uuid4())

        self.temper = tf.placeholder_with_default(
          tf.constant(1.0, dtype=tf.float32),
          shape=(),
          name=self.name+'/temperature')

        self.is_train = tf.placeholder_with_default(
          tf.constant(False, dtype=tf.bool),
          shape=(),
          name=self.name+'/is_train')

        self.graph = tf.get_default_graph()

    # Hierarchical autoencoders can use any initial encoder architecture, e.g.
    # convolutional or recurrent.
    def pre_encode(self, x):
        raise NotImplementedError("implement in subclass, scoped under /main")

    # Main method for hierarchical encoding; see `hierarchical_assignments` for
    # more of the logic.
    def encode(self, x):
        pre = self.pre_encode(x)
        z = tf.layers.dense(pre, num_continuous_dims(self.hier), activation=None,
                name=self.name+'/main/encoder/z', reuse=tf.AUTO_REUSE)
        m, a = hierarchical_assignments(pre,
                self.name+'/main/encoder/a', self.hier, self.temper)
        mz = m * z
        amz = tf.concat(a + [mz], axis=1)
        return OpenStruct(z=z, m=m, a=a, mz=mz, amz=amz) # returns a namedtuple

    # Hierarchical autoencoders can use any decoder architecture, e.g.
    # convolutional or recurrent.
    def decode(self, z):
        raise NotImplementedError("implement in subclass, scoped under /main")

    # COFHAE algorithm
    def train_with_COFHAE(self, sess, X, assignments,
            softmax_temperature=1, assignment_penalty=100, adversarial_penalty=10,
            num_epochs=50, print_every=10, lr=0.001, batch_size=256):
        # Discriminator loss = cross-entropy
        disc = Discriminator(self.name + '/disc/mz_permuted', self.mz, self.Z_perm)
        disc_loss = disc.discriminator_loss

        # Autoencoder loss = reconstruction error + λ1*||a-a'||^2 + λ2*adversarial loss
        main_loss = 0
        loss_terms = OrderedDict()
        loss_terms['recon_loss'] = self.reconstruction_error
        if adversarial_penalty > 0:
            loss_terms['adv_loss'] = adversarial_penalty * disc.generator_loss
        if assignment_penalty > 0:
            loss_terms['assign_loss'] = assignment_penalty * self.assignment_loss
        for k,v in loss_terms.items():
            main_loss += v

        # Use Adam for both objectives
        lr_var = tf.placeholder('float', shape=())
        main_optim = tf.group([
          tf.train.AdamOptimizer(learning_rate=lr_var).minimize(main_loss, var_list=self.main_vars),
          tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        ])
        disc_optim = tf.train.AdamOptimizer(learning_rate=lr_var).minimize(disc_loss, var_list=self.disc_vars)

        # Initialize Tensorflow variables
        sess.run(reinitialize_variables(sess))

        for i, epoch, idx in minibatch(len(X), num_epochs=num_epochs, batch_size=batch_size):
            feed = {}
            feed[self.X_in] = X[idx]
            feed[self.temper] = softmax_temperature
            feed[self.is_train] = True

            # To prepare for computing the adversarial loss, encode X -> Z and
            # permute its _active_ dimensions; the discriminator will attempt
            # to check if samples are coming from the original Z or the
            # conditially permuted Z.
            Z_init = sess.run(self.mz, feed_dict=feed)
            feed[self.temper] = 1e-10 # temporarily reduce temp. for masking
            hard_mask = (sess.run(self.encoded.m, feed_dict=feed) > 0.1)
            feed[self.temper] = softmax_temperature # restore orig temp.
            feed[self.Z_perm] = permute_active(Z_init, hard_mask)

            # Feed in the MIMOSA assignments
            for a_placeholder, a_values in zip(self.a_in, assignments):
                feed[a_placeholder] = a_values[idx] 

            feed[lr_var] = lr
            if epoch >= 0.50*num_epochs: feed[lr_var] = feed[lr_var] / 10.0
            if epoch >= 0.75*num_epochs: feed[lr_var] = feed[lr_var] / 10.0

            # Update the autoencoder parameters
            main_loss_vals = sess.run([main_optim] + list(loss_terms.values()), feed_dict=feed)[1:]
            main_loss_keys = list(loss_terms.keys())

            # Update the discriminator parameters
            _, adv_fool_rate = sess.run([disc_optim, disc.fool_rate], feed_dict=feed)

            # Progress update
            if i % print_every == 0:
                s = f"Epoch {epoch}, iter {i}"
                for k, v in zip(main_loss_keys, main_loss_vals):
                    s += f", {k}={v:.5f}"
                s += f", adv_fool_rate={adv_fool_rate:.5f}"
                print(s)

    # Train the autoencoder just based on reconstruction error
    def train_normally(self, sess, X, softmax_temperature=1,
            num_epochs=50, print_every=10, lr=0.001, batch_size=256):
        lr_var = tf.placeholder('float', shape=())

        optim = tf.group([
          tf.train.AdamOptimizer(learning_rate=lr_var).minimize(self.reconstruction_error, var_list=self.main_vars),
          tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        ])

        sess.run(reinitialize_variables(sess))

        for i, epoch, idx in minibatch(len(X), num_epochs=num_epochs, batch_size=batch_size):
            feed = {}
            feed[self.X_in] = X[idx]
            feed[self.temper] = softmax_temperature
            feed[self.is_train] = True

            feed[lr_var] = lr
            if epoch >= 0.50*num_epochs: feed[lr_var] = feed[lr_var] / 10.0
            if epoch >= 0.75*num_epochs: feed[lr_var] = feed[lr_var] / 10.0

            _, recon_err = sess.run([optim, self.reconstruction_error], feed_dict=feed)

            if i % print_every == 0:
                print(f"Epoch {epoch}, iter {i}, recon_err = {recon_err:.5f}")

    @property_cached
    def X_in(self):
        raise NotImplementedError("implement in subclass")

    @property_cached
    def encoded(self):
        return self.encode(self.X_in)

    @property
    def a(self):
        return self.encoded.a

    @property
    def mz(self):
        return self.encoded.mz

    @property
    def Dz(self): return int(self.Z_out.shape[1])

    @property
    def Dx(self): return int(self.X_in.shape[1])

    @property_cached
    def learned_dimensions(self):
        parts = []
        for a in self.encoded.a:
            parts.append(tf.reshape(tf.add_n([
                (i+1) * a[:,i] for i in range(a.shape[1])
            ]), [-1,1]))
        parts.append(self.encoded.mz)
        return tf.concat(parts, 1)

    @property_cached
    def true_factors_in(self):
        return tf.placeholder('float', [None, int(self.learned_dimensions.shape[1])])

    @property_cached
    def a_in(self):
        return [tf.placeholder('float', [None, a.shape[1]]) for a in self.encoded.a]

    @property_cached
    def assignment_loss(self):
        assignment_loss_terms = [
            tf.reduce_mean(
                tf.reduce_sum((a_true - a_pred)**2, axis=1) * tf.cast((a_true[:,0] >= 0), tf.float32)
            ) for a_true, a_pred in zip(self.a_in, self.a)
        ]
        if len(assignment_loss_terms) > 0:
            return tf.add_n(assignment_loss_terms)
        else:
            return 0

    @property_cached
    def Z_out(self):
        return tf.identity(self.encoded.amz, name=self.name+'/Z_out')

    @property_cached
    def Z_in(self):
        return tf.placeholder("float", [None, self.Dz], name=self.name+'/Z_in')

    @property_cached
    def Z_perm(self):
        K = int(self.encoded.mz.shape[1])
        return tf.placeholder("float", [None, K], name=self.name+'/Z_perm')

    @property_cached
    def X_out_raw(self):
        return self.decode(self.Z_in)

    @property_cached
    def X_rec_raw(self):
        return self.decode(self.Z_out)

    @property_cached
    def X_out(self):
        return self.recons_transform(self.X_out_raw, name=self.name+'/X_out')

    @property_cached
    def X_rec(self):
        return self.recons_transform(self.X_rec_raw, name=self.name+'/X_rec')

    @property
    def recons_transform(self):
        return tf.identity

    @property
    def all_vars(self):
        return self.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)

    def get_vars(self, scope):
        return self.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name+'/'+scope)

    @property
    def main_vars(self): return self.get_vars('main')

    @property
    def disc_vars(self): return self.get_vars('disc')

    @property_cached
    def mean_sq_errors(self):
        return tf.reduce_sum((self.X_in - self.X_rec)**2, axis=1)

    @property_cached
    def mean_sq_error(self):
        return tf.reduce_mean(self.mean_sq_errors)

    @property_cached
    def binary_xent(self):
        return tf.reduce_mean(tf.reduce_sum(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=self.X_in, logits=self.X_rec_raw), axis=1))

    @property
    def reconstruction_error(self):
        raise NotImplementedError("implement in subclass")

    def load(self, path, sess):
        try:
            with open(path, 'rb') as f:
                vals = pickle.load(f)
        except:
            with open(path, 'rb') as f:
                vals = pickle.load(f, encoding='latin1')

        self.X_rec; self.X_out
        sess.run(reinitialize_variables(sess))

        for var in self.all_vars:
            if var.name in vals:
                sess.run(var.assign(vals[var.name]))

        sess.run(reinitialize_variables(sess))

    def save(self, path, sess):
        os.system('mkdir -p {}'.format(path))

        vals = dict([(v.name, v.eval(session=sess)) for v in self.all_vars])
        with open(path+'/vals.pkl', 'wb') as f:
            pickle.dump(vals, f)
