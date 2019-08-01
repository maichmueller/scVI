import numpy as np
import pandas as pd
from scvi.dataset.dataset import GeneExpressionDataset
import os
from pymc3.distributions.discrete import ZeroInflatedNegativeBinomial
from collections import defaultdict
import RNAscrutiny as scrutiny


class ArtificialData(GeneExpressionDataset):
    def __init__(self, save_path, gene_names_file, nr_cell_types=1000, nr_datapoints=5000,
                 cell_covariance_matrix=None):
        super().__init__()
        np.random.seed(0)
        gene_names = pd.read_csv(os.path.join(save_path, gene_names_file + ".csv"), header=0, index_col=0)
        nr_genes = len(gene_names)
        housekeeping_genes = np.random.choice(gene_names.index, int(0.1 * nr_genes))
        housekeeping_covariance = 0.5
        cell_types_generators = defaultdict(list)
        cell_types = "type#" + np.random.choice(1000, nr_cell_types).astype(str)
        for ct in enumerate(cell_types):
            for gene in range(nr_genes):
                psi = np.random.rand()
                mu = int(np.random.rand() * 100)
                alpha = int(np.random.rand() * 20)
                cell_types_generators[ct].append(ZeroInflatedNegativeBinomial(psi, mu, alpha))

        if cell_covariance_matrix is None:
            cm = np.random.rand




    def housekeeping_correlation(self, gene):
        gene_corr = self.gene_correlations[gene]

