import networkx as nx
import numpy as np
import scipy.sparse.linalg as lg

__author__ = "Alan WANG"
__email__ = "alan1995wang@outlook.com"

class HOPE(object):
    def __init__(self, graph, d):
        '''
          d: representation vector dimension
        '''
        self._d = d
        self._graph = graph.G
        self.g = graph
        self._node_num = graph.node_size
        self.learn_embedding()

    def learn_embedding(self):
        graph = self.g.G
        A = nx.to_numpy_array(graph)

        M_g = np.eye(graph.number_of_nodes())
        M_l = np.dot(A, A)

        S = np.dot(np.linalg.inv(M_g), M_l)

        # Ajustar el valor de k para que esté dentro del rango permitido
        k = min(self._d // 2, min(S.shape) - 1)
        if k <= 0:
            raise ValueError("El valor ajustado de 'k' no es válido. Verifique el tamaño del grafo y la dimensión deseada.")

        u, s, vt = lg.svds(S, k=k)
        sigma = np.diagflat(np.sqrt(s))
        X1 = np.dot(u, sigma)
        X2 = np.dot(vt.T, sigma)
        self._X = np.concatenate((X1, X2), axis=1)

    @property
    def vectors(self):
        vectors = {}
        look_back = self.g.look_back_list
        for i, embedding in enumerate(self._X):
            vectors[look_back[i]] = embedding
        return vectors

    def save_embeddings(self, filename):
        with open(filename, 'w') as fout:
            node_num = len(self.vectors.keys())
            fout.write("{} {}\n".format(node_num, self._d))
            for node, vec in self.vectors.items():
                fout.write("{} {}\n".format(node, ' '.join([str(x) for x in vec])))
