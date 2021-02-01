import os
import json
import time
import matplotlib
if os.path.exists('/n'):
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from collections import namedtuple
from mpl_toolkits.mplot3d import Axes3D

# Helper class for defining @property properties but which are only run once,
# and which store the amount of time they took to run
class property_cached():
    def __init__(self, function):
        self.__doc__ = getattr(function, '__doc__')
        self.function = function

    def __get__(self, instance, klass):
        if instance is None: return self
        t1 = time.time()
        key = self.function.__name__
        value = instance.__dict__[key] = self.function(instance)
        t2 = time.time()
        setattr(instance, f"_{key}_time", t2-t1)
        return value

# Helper class for timing various operations.
class timer():
    def __init__(self, label):
        self.label = label

    def __enter__(self):
        self.t1 = time.time()

    def __exit__(self, _type, _value, _traceback):
        self.t2 = time.time()
        print(f"{self.label} took {self.t2 - self.t1}s")

# Helper method for converting dictionaries into namedtuples
def OpenStruct(**kwargs):
    return namedtuple('Struct', ' '.join(kwargs.keys()))(**kwargs)

# Helper method for minibatching
def minibatch(N, num_epochs=50, batch_size=256):
    indexes = np.arange(N)
    n = int(np.ceil(N / batch_size))
    for epoch in range(num_epochs):
        np.random.shuffle(indexes)
        for batch in range(n):
            i = epoch*n + batch
            sl = slice((i%n)*batch_size, ((i%n)+1)*batch_size)
            yield i, epoch, indexes[sl]

# Helper method for evaluating a Tensorflow variable over a batch
def batch_eval(tf_var, sess, unbatched_feed, batch_size=200):
    results = []
    i = 0
    for placeholder, value in unbatched_feed:
        if type(value) == np.ndarray:
            N = len(value)
    while i < N:
        feed = {}
        for placeholder, value in unbatched_feed:
            if type(value) == np.ndarray:
                feed[placeholder] = value[i:i+batch_size]
            else:
                feed[placeholder] = value
        result = sess.run(tf_var, feed_dict=feed)
        results.append(result)
        i += batch_size
    if type(result) == np.ndarray:
        if len(result.shape) >= 2:
            return np.vstack(results)
        else:
            return np.hstack(results)
    else:
        return np.mean(results)

# Helper method for initializing TF variables more intelligently.
def reinitialize_variables(sess):
    uninitialized_vars = []
    for var in tf.global_variables():
        try:
            sess.run(var)
        except tf.errors.FailedPreconditionError:
            uninitialized_vars.append(var)
    return tf.variables_initializer(uninitialized_vars) 

# Helper method for encoding numpy objects as JSON
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
            np.int16, np.int32, np.int64, np.uint8,
            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, 
            np.float64)):
            return float(obj)
        elif isinstance(obj,(np.ndarray,)): #### This is the fix
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

# Helper class for plotting grids of matplotlib figures
class figure_grid():
    def next_subplot(self, **kwargs):
        self.subplots += 1
        return self.fig.add_subplot(self.rows, self.cols, self.subplots, **kwargs)

    def each_subplot(self):
        for _ in range(self.rows * self.cols):
            yield self.next_subplot()

    def title(self, title, fontsize=16, y=1.0, **kwargs):
        self.fig.suptitle(title, y=y, fontsize=fontsize, va='bottom', **kwargs)

    def __init__(self, rows, cols, rowheight=3, rowwidth=12, filename=None):
        self.rows = rows
        self.cols = cols
        self.fig = plt.figure(figsize=(rowwidth, rowheight*self.rows))
        self.subplots = 0
        self.filename = filename

    def __enter__(self):
        return self

    def __exit__(self, _type, _value, _traceback):
        if self.filename:
            try:
                plt.tight_layout()
                plt.savefig(self.filename, bbox_inches='tight')
            except:
                print("ERROR SAVING FIGURE")
            plt.close(self.fig)
        else:
            plt.tight_layout()
            plt.show()

    next = next_subplot

# Helper method for plotting an initial AE representation and coloring by
# ground-truth value
def plot_representation(Z, true_assigns, filename):
    plt_kwargs = {}
    d = Z.shape[1]
    if d >= 3:
        plt_kwargs['projection'] = '3d'

    with figure_grid(1,1,rowwidth=10,rowheight=8,filename=filename) as g:
        ax = g.next(**plt_kwargs)
        ax.scatter(*Z[:,:3].T, c=true_assigns, alpha=0.1, edgecolors='black')

        ax.set_xlabel('Softplus AE dimension 1')
        ax.set_ylabel('Softplus AE dimension 2')
        if d >= 3: ax.set_zlabel('Softplus AE dimension 3')

        if d > 3:
            g.title(f"Dimensions 1-3 of initial {d}D softplus AE\nrepresentation, colored by true assignment")
        else:
            g.title(f"Initial {d}D softplus AE\nrepresentation, colored by true assignment")

# Helper method for plotting a set of learned components and coloring by
# ground-truth value
def plot_components(components, true_assigns, filename):
    plt_kwargs = {}
    d = components[0].points.shape[1]
    if d >= 3:
        plt_kwargs['projection'] = '3d'

    lims = []
    for i in range(d):
        lo = float('inf')
        hi = -float('inf')
        for c in components:
            lo = min(lo, c.points[:,i].min())
            hi = max(hi, c.points[:,i].max())
        lims.append((lo, hi))

    def plot_component(ax, i, c):
        lc = c.index_list
        comp_colors = np.array([true_assigns[i] for i in lc])
        ax.scatter(*c.points[:,:3].T, c=comp_colors, alpha=0.1, vmin=min(true_assigns), vmax=max(true_assigns), s=5)
        ax.set_xlim(lims[0])
        ax.set_ylim(lims[1])
        if d >= 3: ax.set_zlim(lims[2])
        ax.set_title(f"Component {i+1} ({c.dimensionality}D)")

    n_cols = min(6, len(components))
    n_rows = int(np.ceil(len(components)/n_cols))  

    with figure_grid(n_rows, n_cols, filename=filename) as g:
        g.title("Learned components colored by true assignment")
        for i, c in enumerate(components):
                plot_component(g.next(**plt_kwargs), i, c)
