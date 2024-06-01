import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigsh

class LaplacianEigenmaps(object):
    def __init__(self, graph, rep_size=128):
        self.g = graph
        self.node_size = self.g.number_of_nodes()
        self.rep_size = rep_size
        self.adj_mat = nx.to_numpy_array(self.g)
        self.vectors = {}
        self.embeddings = self.get_train()

        # Obtener los nodos del grafo
        nodes = list(self.g.nodes())

        for i, embedding in enumerate(self.embeddings):
            self.vectors[nodes[i]] = embedding

    def getAdj(self):
        look_up = self.g.look_up_dict if hasattr(self.g, 'look_up_dict') else None
        adj = np.zeros((self.node_size, self.node_size))
        for edge in self.g.edges():
            if look_up:
                adj[look_up[edge[0]]][look_up[edge[1]]] = self.g[edge[0]][edge[1]]['weight']
            else:
                adj[edge[0]][edge[1]] = self.g[edge[0]][edge[1]]['weight']
        return adj

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
        print('finish eigh(lap_mat)...')
        return vec

    def save_embeddings(self, filename):
        fout = open(filename, 'w')
        fout.write("{} {}\n".format(self.node_size, self.rep_size))
        for node, vec in self.vectors.items():
            fout.write("{} {}\n".format(node, ' '.join([str(x) for x in vec])))
        fout.close()
