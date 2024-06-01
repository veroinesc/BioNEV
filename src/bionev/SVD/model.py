
import networkx as nx
import numpy as np
from scipy.sparse.linalg import svds

def SVD_embedding(G, output_filename, size=100):
    node_list = list(G.nodes())
    adjacency_matrix = nx.adjacency_matrix(G, node_list)
    adjacency_matrix = adjacency_matrix.astype(float)

    # Ajustar el valor de k para que esté dentro del rango permitido
    k = min(size, min(adjacency_matrix.shape) - 1)
    if k <= 0:
        raise ValueError("El valor ajustado de 'k' no es válido. Verifique el tamaño del grafo y la dimensión deseada.")

    U, Sigma, VT = svds(adjacency_matrix, k=k)
    Sigma = np.diag(Sigma)
    W = np.matmul(U, np.sqrt(Sigma))
    C = np.matmul(VT.T, np.sqrt(Sigma))
    embeddings = W + C
    vectors = {}
    for id, node in enumerate(node_list):
        vectors[node] = list(np.array(embeddings[id]))

    with open(output_filename, 'w') as fout:
        node_num = len(vectors.keys())
        fout.write("{} {}\n".format(node_num, size))
        for node, vec in vectors.items():
            fout.write("{} {}\n".format(node, ' '.join([str(x) for x in vec])))

    return
