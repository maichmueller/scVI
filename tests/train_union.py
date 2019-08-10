#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Tests for `scvi` package."""

import numpy as np

from scvi.dataset import union, GeneExpressionDataset, Dataset10X, UnionDataset, EbiData
from scvi.inference import UnsupervisedTrainer
from scvi.models import VAE, SCANVI, VAEC
import matplotlib.pyplot as plt
import matplotlib
import torch
import os


def train_vae(dataset, save_path, model_savename="vae", use_cuda=True, n_epochs=100, lr=0.01):
    use_batches = False
    use_cuda = use_cuda and torch.cuda.is_available()
    vae = VAE(dataset.nb_genes)
    trainer = UnsupervisedTrainer(
        vae,
        dataset,
        data_loader_kwargs={
            "batch_size": 256,
            "pin_memory": use_cuda
        },
        train_size=0.75,
        use_cuda=use_cuda,
        frequency=5,
    )
    matplotlib.use("TkAgg")
    if os.path.isfile(f'{save_path}/{model_savename}.pkl') and False:
        trainer.model.load_state_dict(torch.load('%s/vae.pkl' % save_path))
        trainer.model.eval()
    else:
        trainer.train(n_epochs=n_epochs, lr=lr)
        torch.save(trainer.model.state_dict(), f'{save_path}/{model_savename}.pkl')

    full = trainer.create_posterior(trainer.model, dataset, indices=np.arange(len(dataset)))
    latent, batch_indices, labels = full.sequential().get_latent()
    batch_indices = batch_indices.ravel()
    n_samples_tsne = 5000
    full.show_t_sne(n_samples=n_samples_tsne, color_by='labels', save_name=f"{model_savename}_tsne.png")
    plt.show()


if __name__ == '__main__':
    available_datasets = [
            # "fresh_68k_pbmc_donor_a",
            # "frozen_pbmc_donor_a",
            # "frozen_pbmc_donor_b",
            # "frozen_pbmc_donor_c",
            # "pbmc8k",
            # "pbmc4k",
            "t_3k",
            "t_4k",
            # "pbmc_1k_protein_v3",
            # "pbmc_10k_protein_v3",
            # "malt_10k_protein_v3",
            # "pbmc_1k_v2",
            # "pbmc_1k_v3",
            # "pbmc_10k_v3"
    ]
    # available_datasets = [(el, Dataset10X) for el in available_datasets]
    # available_datasets.extend(
    #     (elem for elem in zip([None]*3, (CortexDataset, PbmcDataset, CbmcDataset)))
    # )
    # data = UnionDataset("./data", map_save_fname="human_data_map", data_save_fname="human_data_union")
    # data = UnionDataset("./data",
    #                              map_fname="ensembl_human_genes_proteincoding",
    #                              data_fname="human_data_union",
    #                              low_memory=False)
    # data.build_mapping([elem[0] for elem in available_datasets], [elem[1] for elem in available_datasets])
    # data.concat_union_in_memory([elem[0] for elem in available_datasets], [elem[1] for elem in available_datasets])
    # data = UnionDataset("./data", map_fname="ensembl_human_genes_proteincoding", data_fname="human_data_union")
    union_data = UnionDataset("./data", map_fname="ensembl_mouse_genes-proteincoding", low_memory=False)
    data = EbiData("./data", experiment="E-ENAD-15")
    union_data.concat_union_from_memory([data])
    train_vae(union_data, save_path="./data", model_savename="ebi_data_vae_full", n_epochs=100)
