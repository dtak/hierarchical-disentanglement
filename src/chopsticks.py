import os
import numpy as np
from collections import namedtuple

def Bern(p): return int(np.random.uniform() < p)
def Categ(p): return np.random.choice(len(p), p=p)
def Unif(a,b): return a + np.random.uniform() * (b-a)
def RandomSign(): return 2*Bern(0.5)-1

class Chopsticks():
  def __init__(self, depth, variant='inter', noise=0, min_offset=0, resolution=64):
    self.depth = depth
    self.noise = noise
    self.variant = variant
    self.resolution = resolution
    self.min_offset = min_offset

  def sample_slope(self, b=0.01):
      return RandomSign() * Unif(self.min_offset*b, b)

  def sample_inter(self, b=0.2):
      return RandomSign() * Unif(self.min_offset*b, b)

  def sample_node(self, a):
    v = self.variant
    if v == 'inter' or (v == 'either' and a == 1):
      return [self.sample_inter()]
    elif v == 'slope' or (v == 'either' and a == 2):
      return [self.sample_slope()]
    else:
      assert(v == 'both')
      return [self.sample_inter(), self.sample_slope()]

  @property
  def z_size(self):
    if self.variant == 'both':
      return 2
    else:
      return 1

  def probs(self, remaining_depth):
    if remaining_depth == self.depth:
      p = 1
    else:
      p = 1-np.power(2.0,-remaining_depth)

    if self.variant == 'either':
      return [1-p, p/2, p/2]
    else:
      return [1-p, p]

  def sample_az(self, depth=None):
    if depth is None: depth = self.depth
    a = []
    z = []
    probs = self.probs(depth)
    assignment = Categ(probs)

    if assignment == 0:
      a += [0 for d in range(depth)]
      z += [0 for d in range(depth) for _ in range(self.z_size)]
    else:
      a2, z2 = self.sample_az(depth-1)
      a += [assignment] + a2
      z += self.sample_node(assignment) + z2

    return a, z

  def generate(self, a, z):
    Dx = self.resolution
    x = np.zeros(Dx)
    v = self.variant

    for level in range(self.depth):
      if not a[level]: break

      start_frac = 1 - (1/2)**level
      start_idx = int(Dx * start_frac)

      if v == 'inter' or (v == 'either' and a[level] == 1):
        x[start_idx:] += z[level]
      elif v == 'slope' or (v == 'either' and a[level] == 2):
        x[start_idx:] += z[level] * np.arange(Dx - start_idx)
      else:
        assert(v == 'both')
        x[start_idx:] += z[2*level]
        x[start_idx:] += z[2*level+1] * np.arange(Dx - start_idx)

    if self.noise > 0:
        x += np.random.normal(size=len(x)) * self.noise

    return x

  def sample(self):
    a, z = self.sample_az()
    return a, z, self.generate(a, z)

  @property
  def option_string(self):
    s = self.variant
    if self.noise > 0:
       s += f"_noise{self.noise}"
    if self.min_offset > 0:
       s += f"_min_offset{self.min_offset}"
    return s

  # Generate a 100,000-sample dataset and cache it.
  @property
  def data(self):
    dirname = os.path.dirname(os.path.realpath(__file__))
    prefix = os.path.join(dirname, f"../data/chopsticks_depth{self.depth}_{self.option_string}")
    if not os.path.exists(f"{prefix}_X.npy"):
      A = []
      Z = []
      X = []

      for i in range(100000):
        a_,z_,x = self.sample()
        a,z = self.make_onehot(a_,z_)

        A.append(a)
        Z.append(z)
        X.append(x)

      np.save(f"{prefix}_A.npy", np.array(A))
      np.save(f"{prefix}_Z.npy", np.array(Z))
      np.save(f"{prefix}_X.npy", np.array(X))

    Dataset = namedtuple('Dataset', ['A', 'Z', 'X', 'AMZ'])

    A = np.load(f"{prefix}_A.npy", allow_pickle=True)
    Z = np.load(f"{prefix}_Z.npy")
    X = np.load(f"{prefix}_X.npy")

    parts = []
    for j in range(A.shape[1]):
        a = np.array(list(A[:,j]))
        parts.append(np.sum([
            (i+1) * a[:,i] for i in range(a.shape[1])
        ], axis=0))
    parts += [Z[:,i] for i in range(Z.shape[1])]
    AMZ = np.vstack(parts).T

    return Dataset(X=X, A=A, Z=Z, AMZ=AMZ)

  # Construct the ground-truth hierarchy for this variant of Chopsticks, in a
  # way that's compatible with the hierarchical autoencoder implementation.
  # Used for ablations and for comparing learned vs. ground-truth hierarchy.
  def construct_hier(self, depth=None):
    if depth is None: depth = self.depth

    a_config = []
    z_config = []

    if depth < self.depth:
      z_config += [dict(type="continuous") for _ in range(self.z_size)]

    probs = self.probs(depth)

    if depth >= 1:
      if self.variant == 'either':
        if probs[0] == 0:
          a_config.append(dict(
            type="categorical",
            probs=probs[1:],
            options=[self.construct_hier(depth-1), self.construct_hier(depth-1)]
          ))
        else:
          a_config.append(dict(
            type="categorical",
            probs=probs,
            options=[[], self.construct_hier(depth-1), self.construct_hier(depth-1)]
          ))
      else:
        if probs[0] == 0:
          a_config = self.construct_hier(depth-1)
        else:
          a_config.append(dict(
            type="categorical",
            probs=probs,
            options=[[], self.construct_hier(depth-1)]
          ))

    return z_config + a_config

  @property
  def hierarchy(self):
    return self.construct_hier()

  # Somewhat complicated method to convert data samples into flat vectors
  # compatible with hierarchical autoencoders; this is for comparing learned
  # and ground-truth represnetations.
  def make_onehot(self, a, z):
    def get_leaves(hier):
        res = []
        for i, node in enumerate(hier):
            if node['type'] == 'continuous':
                res.append(node)
            elif node['type'] == 'categorical':
                for j, subhier in enumerate(node['options']):
                    if len(subhier):
                        res += get_leaves(subhier)
        return res

    def get_categs(hier):
        res = []
        for i, node in enumerate(hier):
            if node['type'] == 'categorical':
                res.append(node)
                for j, subhier in enumerate(node['options']):
                    if len(subhier):
                        res += get_categs(subhier)
        return res

    if self.variant == 'either':
        hier = self.hierarchy
        node = hier[0]
        i = 0

        for j, assignment in enumerate(a):
            if j == 0:
              assignment -= 1
            node['value'] = assignment
            children = node['options'][assignment]

            conts = [c for c in children if c['type'] == 'continuous']
            discs = [c for c in children if c['type'] == 'categorical']
            for cont in conts:
                cont['value'] = z[i]
                i += 1

            assert(len(discs) <= 1)

            if len(discs) == 1:
                node = discs[0]

        res_a = []
        res_z = []
        for disc in get_categs(hier):
            el = [0 for _ in disc['options']]
            if 'value' in disc:
                el[disc['value']] = 1
            res_a.append(el)
        for cont in get_leaves(hier):
            res_z.append(cont.get('value', 0))

        return res_a, res_z

    else:
        base = [0,0]
        res = [list(base) for _ in a]
        for i in range(len(a)):
            res[i][a[i]] = 1
            if a[i] == 0:
                break
        assert(res[0][1])
        res = res[1:]
        return res, z
