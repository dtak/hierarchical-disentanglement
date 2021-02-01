import os
import sys
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import BallTree
from collections import defaultdict
from scipy.spatial import Delaunay
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from helpers import timer, property_cached

# Helper class for running and holding the result of a local singular value
# decomposition
class LocalSVD():
    def __init__(self, X, ransac_frac=0.6667, eig_cumsum_thresh=0.95, eig_decay_thresh=4, cos_simil_thresh=0.99):
        self.pca = PCA().fit(X)

        if ransac_frac < 1:
            error_vecs = np.array([ self.reconstruction_errors(X, k) for k in range(X.shape[1]) ]).T
            error_norms = np.linalg.norm(error_vecs, axis=1)
            inlier_idxs = np.argwhere(error_norms <= np.percentile(error_norms, ransac_frac * 100))[:,0]
            self.pca = PCA().fit(X[inlier_idxs])

        eigs = self.pca.explained_variance_ratio_

        explains_var = (np.cumsum(eigs)[:-1] >= eig_cumsum_thresh)
        decays_fast = ((eigs[1:] / eigs[:-1]) <= 1./eig_decay_thresh)
        meets_criteria = np.argwhere(decays_fast * explains_var)[:,0] + 1

        if len(meets_criteria):
            self.effective_dim = meets_criteria[0]
        else:
            self.effective_dim = X.shape[1]

        self.cos_simil_thresh = cos_simil_thresh

    @property
    def full_dim(self):
        return len(self.pca.explained_variance_ratio_)

    @property
    def active_components(self):
        return self.pca.components_[:self.effective_dim]

    def reconstruction_errors(self, X, k):
        V = self.pca.components_[:k]
        xform = np.dot(X - self.pca.mean_, V.T)
        recon = np.dot(xform, V) + self.pca.mean_
        return np.linalg.norm(X - recon, axis=1)

    def embed(self, X):
        V = self.active_components
        return np.dot(X - self.pca.mean_, V.T)

    def is_edge(self, point, neighbors):
        p = self.embed(point)
        Np = self.embed(neighbors)
        if self.effective_dim == 1:
            return p <= Np.min() or p >= Np.max()
        elif len(Np) <= self.effective_dim:
            return True
        else:
            try:
                return Delaunay(Np).find_simplex(p) < 0
            except:
                return False

    def tangent_plane_cos(self, other):
        U = self.active_components
        V = other.active_components
        if len(U) == len(V) == self.full_dim == other.full_dim:
            return 1
        elif len(U) != len(V):
            return 0
        else:
            return np.abs(np.linalg.det(np.dot(U, V.T)))

    def is_similar(self, other):
        return self.tangent_plane_cos(other) >= self.cos_simil_thresh

# Helper class for representing manifold components
class ManifoldComponent():
    def __init__(self, points, svds, index_list):
        self.index_list = index_list
        self.points = points
        self.svds = svds
        self.parent = None
        self.children = []

    def __len__(self):
        return len(self.points)

    @property
    def dimensionality(self):
        return self.svds[0].effective_dim

    @property_cached
    def local_ball_tree(self):
        return BallTree(self.points)

    @property_cached
    def local_neighborhoods(self):
        return self.local_ball_tree.query(self.points, k=min(40, len(self.points)))

    @property
    def local_neighbor_distances(self):
        return self.local_neighborhoods[0]

    @property
    def local_neighbors(self):
        return self.local_neighborhoods[1]

    @property
    def nearest_neighbor_lengthscale(self):
        return self.local_neighbor_distances[:,1].mean()

    def nearest_neighbor_distances(self, other):
        return self.local_ball_tree.query(other.points, k=1)[0][:,0]

    def enclosure_ratio(self, other):
        if other.dimensionality >= self.dimensionality:
            return float('inf')
        else:
            return np.mean(self.nearest_neighbor_distances(other)) / self.nearest_neighbor_lengthscale

    @property_cached
    def edge_mask(self):
        if len(self) <= 2:
            return np.ones(len(self))
        else:
            return np.array([svd.is_edge(p, self.points[Np[1:]])
                for svd, p, Np in zip(self.svds, self.points, self.local_neighbors)]).astype(int).flatten()

    @property
    def edges(self):
        return self.points[self.edge_indices]

    @property
    def edge_indices(self):
        return np.argwhere(self.edge_mask)[:,0]

    def edge_overlap(self, other, **kw):
        if len(self.edges) == 0:
            return 0
        else:
            return len(self.common_edges(other, **kw)) / len(self.edges)

    def closest_edge(self, x):
        return self.edge_indices[np.argmin(np.linalg.norm(self.edges - x, axis=1))]

    def common_edges(self, other):
        if len(other.edges) == 0 or self.dimensionality != other.dimensionality:
            return []
        else:
            edge_pairs = [(i, other.closest_edge(self.points[i])) for i in self.edge_indices]
            return [(i,j) for i,j in edge_pairs if self.svds[i].is_similar(other.svds[j])]

    def merge(self, other, **kw):
        new_points = np.vstack((self.points, other.points))
        new_svds = self.svds + other.svds
        new_idxs = self.index_list + other.index_list
        return ManifoldComponent(new_points, new_svds, new_idxs)

    @property
    def ancestors(self):
        p = self.parent
        if p is None:
            return []
        else:
            return [p] + p.ancestors

    @property
    def descendants(self):
        res = self.children
        for c in self.children:
          res += c.descendants
        return res

    @property
    def self_and_descendants(self):
        return [self] + self.descendants

# Run the MIMOSA hierarchy extraction method on Z, which can either be raw data
# or (preferably) a set of initial representations whose dimension has already
# been reduced with a smooth (non-ReLU) autoencoder.
def MIMOSA(Z,
        num_nearest_neighbors=40,
        eig_cumsum_thresh=0.95,
        eig_decay_thresh=4,
        cos_simil_thresh=0.99,
        ransac_frac=0.6667,
        contagion_num=5,
        min_size_init=20,
        min_size_merged=2000,
        neighbor_lengthscale_mult=10
    ):

    with timer("BallTree"):
        ball_tree = BallTree(Z)
        neighbors = ball_tree.query(Z, k=num_nearest_neighbors)[1]

    with timer("LocalSVD"):
        svd_kwargs = dict(eig_cumsum_thresh=eig_cumsum_thresh,
                          eig_decay_thresh=eig_decay_thresh,
                          cos_simil_thresh=cos_simil_thresh,
                          ransac_frac=ransac_frac)
        svds = [LocalSVD(Z[n], **svd_kwargs) for n in neighbors]

    covered = set()

    def BuildComponent(start):
        similar_neighbors = [n for n in neighbors[start] if n not in covered and svds[start].is_similar(svds[n])]
        component = set([start] + similar_neighbors)
        frontier = similar_neighbors
        visits = defaultdict(int)

        while len(frontier):
            i = frontier.pop()
            for j in neighbors[i]:
                if j in covered: continue
                if j in component: continue
                if svds[i].is_similar(svds[j]):
                    visits[j] += 1
                    if visits[j] >= contagion_num:
                        component.add(j)
                        frontier.append(j)

        idx = list(component)

        return ManifoldComponent(Z[idx], [svds[i] for i in idx], idx)

    with timer("BuildComponent"):
        components = []
        for i in range(len(Z)):
            if i not in covered:
                component = BuildComponent(i)
                components.append(component)
                for j in component.index_list:
                    covered.add(j)

    with timer("MergeComponents"):
        components2 = MergeComponents(components,
                min_size_init=min_size_init,
                min_size_merged=min_size_merged)

    with timer("ConstructHierarchy"):
        hierarchy, assignments = ConstructHierarchy(len(Z), components2,
                neighbor_lengthscale_mult=neighbor_lengthscale_mult)

    return components, components2, hierarchy, assignments


def MergeComponents(components, min_size_init=20, min_size_merged=2000):
    # Initial filtering step
    components = [c for c in components if len(c) >= min_size_init]

    # Compute edge overlap matrix
    M = np.array([[
        c1.edge_overlap(c2) if i != j else 0
            for i, c1 in enumerate(components)]
            for j, c2 in enumerate(components)])
    M = 0.5 * (M + M.T)

    # Use that matrix to determine which components should be merged
    min_common_edge_frac = lambda d: 0.5*(1/(2**d) + 1/(2**(d+1)))
    min_common_edge_fracs = [min_common_edge_frac(c.dimensionality) for c in components]
    merge_matrix = np.array([M[i] >= min_common_edge_fracs[i] for i in range(len(components))])
    merge_indices = lambda i: list(np.argwhere(merge_matrix[:,i])[:,0])

    def merge(comps):
        if len(comps) == 1:
            return comps[0]
        else:
            return comps[0].merge(merge(comps[1:]))

    # Iterate through the components and merge everything (requires doing
    # another breadth-first search)
    components2 = []
    already_merged = set()

    for i in range(len(components)):
        if i in already_merged: continue
        to_merge = set()
        frontier = [i]
        while frontier:
            j = frontier.pop()
            to_merge.add(j)
            for k in merge_indices(j):
                if k not in to_merge:
                    frontier.append(k)
        components2.append(merge([components[j] for j in to_merge]))
        already_merged = already_merged.union(to_merge)

    # Post-merge filtering step
    return [c for c in components2 if len(c) >= min_size_merged]


def ConstructHierarchy(N, components, neighbor_lengthscale_mult=10):
    # Create a component tree based on enclosure relationships
    for i, c1 in enumerate(sorted(components, key=lambda c: c.dimensionality)):
        # First find all smaller components "enclosed" by this one
        enclosed1 = []
        for c2 in components:
            ratio = c1.enclosure_ratio(c2)
            if ratio <= neighbor_lengthscale_mult:
                enclosed1.append((ratio, c2))

        # Next, exclude components enclosed within components we already enclose
        enclosed2 = []
        for j, (ratio, c2) in enumerate(enclosed1):
            if any(c2 in c3.ancestors for _, c3 in enclosed1):
                continue
            enclosed2.append((ratio, c2))

        # Finally, if we do enclose others, pick the one we enclose most tightly
        if len(enclosed2) > 0:
            best_ratio, c2 = min(enclosed2)
            c1.parent = c2
            c2.children.append(c1)

    # Convert the component tree into a dimension hierarchy
    def dimension_group(comp, minus=0):
        entry = []
        D = comp.dimensionality
        for _ in range(D - minus):
            entry.append({ 'type': 'continuous' })
        if len(comp.children):
            entry.append({ 'type': 'categorical',
                           'options': [[]] + [dimension_group(c, minus=D) for c in comp.children],
                           # Also add a list of lists of components that correspond to each branch down the tree;
                           # this isn't strictly part of the hierarchy but is helpful for generating the
                           # assignment vector.
                           '_flat_comp_idxs':
                               [[components.index(comp)]] +
                               [[components.index(cc) for cc in c.self_and_descendants] for c in comp.children]
                        })
        return entry

    roots = [c for c in components if c.parent is None]

    if len(roots) == 1:
      hierarchy = dimension_group(roots[0])
    else:
      hierarchy = [{ 'type': 'categorical',
                     'options': [dimension_group(c) for c in roots],
                     '_flat_comp_idxs': [[components.index(cc) for cc in c.self_and_descendants] for c in roots] }]


    # Create a vector of assignments to categorical dimension branches (for
    # points included in components)
    include = np.zeros(N)
    for c in components:
      include[c.index_list] = 1
    exclude = np.argwhere(1-include)[:,0]

    def generate_assignments(group):
        assignments = []
        for i, dim in enumerate(group):
            if dim['type'] == 'categorical':
                K = len(dim['options'])
                a = np.zeros((N,K))
                a[exclude, :] = -1
                for j, comp_idxs in enumerate(dim['_flat_comp_idxs']):
                  for c in comp_idxs:
                    a[components[c].index_list, j] = 1
                assignments.append(a)
                for j, subgroup in enumerate(dim['options']):
                    if len(subgroup):
                        assignments += generate_assignments(subgroup)
        return assignments

    return hierarchy, generate_assignments(hierarchy)
