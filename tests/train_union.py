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
import copy
from Eval_basis import *

if __name__ == '__main__':
    dataset = Dataset10X("pbmc_1k_protein_v3")
    train_vae(dataset, "./data", f"../trained_models/test_stuff", n_epochs=100)

    data_small = copy.deepcopy(dataset)
    data_small.subsample_cells(0.1)

    x = compute_log_likelihood(dataset, "./data", "../trained_models/test_stuff", data_small, n_samples_mc=100)
    print(x)
