#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Tests for `scvi` package."""

import numpy as np

from scvi.dataset import union, GeneExpressionDataset, Dataset10X, UnionDataset
from scvi.inference import UnsupervisedTrainer
from scvi.models import VAE, SCANVI, VAEC

import torch


def train_vae(dataset, save_path, use_cuda=True, n_epochs=100, lr=0.01):
    use_batches = False
    use_cuda = use_cuda and torch.cuda.is_available()
    vae = VAE(dataset.nb_genes)
    trainer = UnsupervisedTrainer(
        vae,
        dataset,
        train_size=0.75,
        use_cuda=use_cuda,
        frequency=5,
    )

    # if os.path.isfile('%s/vae.pkl' % save_path):
    #     trainer.model.load_state_dict(torch.load('%s/vae.pkl' % save_path))
    #     trainer.model.eval()
    # else:
    trainer.train(n_epochs=n_epochs, lr=lr)
    torch.save(trainer.model.state_dict(), f'{save_path}/vae.pkl')


if __name__ == '__main__':
    union_dataset = UnionDataset("./data", map_fname="ensembl_human_genes_proteincoding", data_fname="human_data_union")
    train_vae(union_dataset, "./data")
