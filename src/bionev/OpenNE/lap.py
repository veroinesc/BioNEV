# -*- coding: utf-8 -*-

import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigsh
import scipy.sparse as sp

class LaplacianEigenmaps(object):
    def __init__(self, graph, rep_size=128):
        self.g = graph
        self.node_size = self.g.number_of_nodes()
        self.rep_size = rep_size
        self.adj_mat = nx.to_scipy_sparse_matrix(self.g)
        self.vectors = {}
        self.embeddings = self.get_train()
        look_back = list(self.g.nodes())

        for i, embedding in enumerate(self.embeddings):
            self.vectors[look_back[i]] = embedding

    def getLap(self):
        G = self.g.to_undirected()
        print('begin norm_lap_mat')
        norm_lap_mat = nx.normalized_laplacian_matrix(G)
        print('finish norm_lap_mat')
        return norm_lap_mat

    def get_train(self):
        lap_mat = self.getLap()
        print('finish getLap...')
        w, vec = eigsh(lap_mat, k=self.rep_size)
        print('finish eigen decomposition...')
        return vec

    def save_embeddings(self, filename):
        fout = open(filename, 'w')
        node_num = len(self.vectors)
        fout.write("{} {}\n".format(node_num, self.rep_size))
        for node, vec in self.vectors.items():
            fout.write("{} {}\n".format(node, ' '.join([str(x) for x in vec])))
        fout.close()

# Ejemplo de uso:
# graph_small = nx.read_edgelist("small_graph.edgelist")
# graph_large = nx.read_edgelist("large_graph.edgelist")
# laplacian_small = LaplacianEigenmaps(graph_small)
# laplacian_large = LaplacianEigenmaps(graph_large)
# laplacian_small.save_embeddings("embeddings_small.txt")
# laplacian_large.save_embeddings("embeddings_large.txt")
