#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Tests for `scvi` package."""

import numpy as np

from scvi.dataset import union, GeneExpressionDataset, Dataset10X, UnionDataset, EbiData
from scvi.inference import UnsupervisedTrainer, Posterior
from scvi.models import VAE, SCANVI, VAEC, log_likelihood
import matplotlib.pyplot as plt
import matplotlib
import torch
import os

import matplotlib as mpl

import copy
import time


def time_dec(func):
    def wrapped(*args, **kwargs):
        t = time.perf_counter()
        ret = func(*args, **kwargs)
        print(func.__name__, ":", f"{time.perf_counter() - t:.2f}s")
        return ret
    return wrapped


def train_vae(dataset, save_path, model_savename="vae", use_cuda=True, n_epochs=100, lr=0.01, **kwargs):
    use_cuda = use_cuda and torch.cuda.is_available()
    vae = VAE(dataset.nb_genes)
    trainer = UnsupervisedTrainer(
        vae,
        dataset,
        data_loader_kwargs={
            "batch_size": 2048,
            "pin_memory": use_cuda
        },
        train_size=0.75,
        use_cuda=use_cuda,
        frequency=5,
    )
    matplotlib.use("TkAgg")
    if os.path.isfile(f'{save_path}/{model_savename}.pkl'):
        print("Loading model from file. No training.")
        trainer.model.load_state_dict(torch.load(f'{save_path}/{model_savename}.pkl'))
        trainer.model.eval()
    else:
        print("Initializing training.")
        trainer.train(n_epochs=n_epochs, lr=lr)
        torch.save(trainer.model.state_dict(), f'{save_path}/{model_savename}.pkl')
    return trainer


def plot_tsne(trainer, model, dataset, image_savename, n_samples_tsne=5000, colors=None, **kwargs):
    plt.style.use("ggplot")
    posterior = trainer.create_posterior(model, dataset, indices=np.arange(len(dataset)))
    posterior.show_t_sne(n_samples=n_samples_tsne,
                         labels=dataset.labels,
                         color_by='labels',
                         colormap=colors,
                         save_name=image_savename,
                         **kwargs)
    plt.show()
    return posterior


def compute_log_likelihood(dataset, save_path, model_savename, data_for_loglikelihood, n_samples_mc: int = 100,
                           n_epochs=100, lr=0.01, **kwargs):
    trainer = train_vae(dataset, save_path, model_savename=model_savename, use_cuda=True,
                        n_epochs=n_epochs, lr=lr, **kwargs)
    posterior = trainer.create_posterior(trainer.model, data_for_loglikelihood,
                                         indices=np.arange(len(data_for_loglikelihood)))
    llkl = log_likelihood.compute_marginal_log_likelihood(trainer.model, posterior, n_samples_mc)
    return llkl


if __name__ == '__main__':

    np.random.seed(1)
    data_full = EbiData("./data", experiment="E-ENAD-15")
    data_big = UnionDataset("./data", gene_map_load_filename="ensembl_mouse_genes-proteincoding", low_memory=False)
    data_big.join_datasets(data_source="memory", data_target="memory", gene_datasets=[data_full])
    data_big.filter_cell_types(np.array([ct for ct in data_big.cell_types if ct != "not available"]))

    nr_ct = len(data_big.cell_types)
    ct_pool = np.arange(nr_ct)
    cutout_ct = np.random.choice(nr_ct, int(0.1 * nr_ct))
    rem_ct = ct_pool[~np.isin(ct_pool, cutout_ct)]
    # print(data_big.cell_types[cutout_ct])
    rem_ct = data_big.cell_types[rem_ct]
    cutout_ct = data_big.cell_types[cutout_ct]

    data_small = copy.deepcopy(data_big)
    data_small.subsample_cells(0.7)
    data_small.filter_cell_types(np.concatenate((cutout_ct, np.random.choice(rem_ct, len(cutout_ct)))))

    data_big.filter_cell_types(rem_ct)

    # colors = {}
    # cmap_regular = plt.get_cmap("tab20b", len(rem_ct))
    # cmap_cutout = plt.get_cmap("tab20", len(cutout_ct))
    # colors.update({label: cmap_regular(idx)
    #                for idx, label in enumerate(rem_ct)})
    # colors.update({label: cmap_cutout(idx)
    #                for idx, label in enumerate(cutout_ct)})
    colors=None
    n_epochs = 100
    trainer_big = train_vae(data_big, "./data", "big_data_portion", n_epochs=n_epochs)
    trainer_small = train_vae(data_small, "./data", "small_data_portion", n_epochs=n_epochs)

    dot_size = (mpl.rcParams['lines.markersize'] ** 2.0)

    posterior_big = plot_tsne(trainer_big, trainer_big.model, data_big, "big_data_portion",
                              colors=colors, s=dot_size, edgecolors='black')
    posterior_small = plot_tsne(trainer_small, trainer_small.model, data_small, "small_data_portion",
                                colors=colors, s=dot_size, edgecolors='black')
    posterior_small_in_big = plot_tsne(trainer_big, trainer_big.model, data_small, "small_in_big",
                                       colors=colors, s=dot_size, edgecolors='black')
    plt.show()
