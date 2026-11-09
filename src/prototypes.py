import numpy as np


def build_prototypes(marker_table, embedding, gene_index):
    prototypes = {}
    for cell_type, markers in marker_table.items():
        weights = np.zeros(embedding.shape[1])
        for gene in markers:
            if gene in gene_index:
                weights += embedding[:, gene_index[gene]]
        norm = np.linalg.norm(weights)
        prototypes[cell_type] = weights / norm if norm else weights
    return prototypes
