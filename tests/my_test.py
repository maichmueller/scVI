#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Tests for `scvi` package."""

import numpy as np

from scvi.benchmark import all_benchmarks, benchmark, benchmark_fish_scrna, ldvae_benchmark
from scvi.dataset import BrainLargeDataset, CortexDataset, RetinaDataset, BrainSmallDataset, HematoDataset, \
    LoomDataset, AnnDataset, CsvDataset, CiteSeqDataset, CbmcDataset, PbmcDataset, SyntheticDataset, \
    SeqfishDataset, SmfishDataset, BreastCancerDataset, MouseOBDataset, \
    GeneExpressionDataset, PurifiedPBMCDataset, SyntheticDatasetCorr, ZISyntheticDatasetCorr, \
    Dataset10X
from scvi.inference import JointSemiSupervisedTrainer, AlternateSemiSupervisedTrainer, ClassifierTrainer, \
    UnsupervisedTrainer, AdapterTrainer
from scvi.inference.annotation import compute_accuracy_rf, compute_accuracy_svc
from scvi.models import VAE, SCANVI, VAEC
from scvi.models.classifier import Classifier
import anndata
import os.path
import torch

use_cuda = False


def concat_datasets_10x(save_path):
    from scvi.dataset.dataset10X import available_datasets

    datasets_10x = []
    for idx, filen in enumerate(file for group in available_datasets.values() for file in group):
        data = Dataset10X(filen, save_path)
        if not data.dense and idx < 5:
            datasets_10x.append(data)

    datasets_10x_merged = GeneExpressionDataset.concat_datasets_union(*datasets_10x)
    return datasets_10x_merged


def train_vae(dataset, save_path, use_cuda=None, n_epochs=100, lr=0.01):
    use_batches = False
    use_cuda = torch.cuda.is_available() if use_cuda is None else use_cuda
    vae = VAE(dataset.nb_genes, n_batch=dataset.n_batches * use_batches)
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
    data = concat_datasets_10x("./data/")
    train_vae(data, "./data/")
