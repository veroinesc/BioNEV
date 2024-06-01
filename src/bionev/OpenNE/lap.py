# -*- coding: utf-8 -*-

import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigsh

class LaplacianEigenmaps(object):
    def __init__(self, graph, rep_size=128):
        self.g = graph
        self.node_size = self.g.number_of_nodes()  # Obtener el número de nodos del grafo
        self.rep_size = min(rep_size, self.node_size)  # Asegurar que rep_size no sea mayor que el número de nodos
        self.adj_mat = nx.to_numpy_array(self.g)
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
        # Calcular los valores y vectores propios para los primeros rep_size valores propios
        w, vec = eigsh(lap_mat, k=self.rep_size, which='SM')  # 'SM' para los valores propios más pequeños
        print('finish eigh(lap_mat)...')
        return vec

    def save_embeddings(self, filename):
        fout = open(filename, 'w')
        node_num = len(self.vectors)
        fout.write("{} {}\n".format(node_num, self.rep_size))
        for node, vec in self.vectors.items():
            fout.write("{} {}\n".format(node, ' '.join([str(x) for x in vec])))
        fout.close()
