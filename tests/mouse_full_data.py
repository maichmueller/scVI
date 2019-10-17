from scvi.dataset import Dataset10X, UnionDataset, MouseAtlas, EbiData, AnnDatasetFromAnnData
import scanpy as sc
import numpy as np
import os
import pandas as pd
from Eval_basis import *


if __name__ == '__main__':
    complete_mouse = UnionDataset("./data",
                                  gene_map_load_filename="gene_maps/ensembl_mouse_genes-proteincoding",
                                  data_load_filename="mouse_data_all.loom",
                                  low_memory=False)

    # union_dataset.join_datasets("loom", "memory", in_filename="test_loom_script.loom")
    n_epochs = 100
    colors = None

    print("Training VAE")

    trainer = train_vae(complete_mouse, "./data", f"../trained_models/mouse_data_full_model", n_epochs=n_epochs)
    # trainer_small = train_vae(data_small, "./data", f"small_{tissue}_data_portion", n_epochs=n_epochs)

    ebi_with_celltypes = EbiData("./data")

    dot_size = (mpl.rcParams['lines.markersize'] ** 2.0)

    posterior_big = plot_tsne(trainer, trainer.model, complete_mouse, f"./max_data_model",
                              colors=colors, s=dot_size, edgecolors='black')

    # posterior_ebi_annotated = trainer.create_posterior(model, ebi_with_celltypes, indices=np.arange(len(dataset)))
    posterior_ebi_annotated = plot_tsne(trainer_big, trainer_big.model, ebi_with_celltypes, f"./plots/testotesto",
                                       colors=colors, s=dot_size, edgecolors='black')

