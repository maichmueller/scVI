import numpy as np
import pandas as pd
from datetime import datetime
import warnings
from scipy import sparse
from .dataset import GeneExpressionDataset
from .dataset_hdf import HDF5Dataset, convert_to_hdf5
import torch
from torch.utils import data


class IndepUnionDataset(data.dataset):

    def __init__(self, save_path, low_memory=True, load_map_fname=None,
                 save_mapping_fname=None,
                 hdf5_data_fname=None, hdf5_data_savename=None, **kwargs):

        self.gene_map = dict()
        self.index_map = []
        self.gene_names, self.gene_names_len = [], 0

        self.datasets_used = None

        self.save_path = save_path

        self.save_mapping_fname = save_mapping_fname
        if load_map_fname is not None:
            self.gene_map = pd.read_csv(self.save_path + load_map_fname + ".csv")

        self.hdf5_data_fname = hdf5_data_fname
        self.hdf5_data_savename = hdf5_data_savename

        if hdf5_data_fname is not None:
            self.hdf5_handler = HDF5Dataset(hdf5_data_fname, False, low_memory, **kwargs)
        else:
            self.hdf5_handler = None

    def build_mapping(self, dataset_names, dataset_classes, dataset_args):
        filtered_classes = dict()
        for ds_name, ds_class, ds_args in zip(dataset_names, dataset_classes, dataset_args):
            if ds_args:
                dataset = ds_class(ds_name, save_path=self.save_path, **ds_args)
            else:
                dataset = ds_class(ds_name, save_path=self.save_path)

            if not hasattr(dataset, "gene_names"):
                # without gene names we can't build a proper mapping
                warnings.warn(f"Dataset {ds_name} doesn't have gene_names as attribute. Skipping this dataset.")
                continue

            filtered_classes[str(ds_class)] = str(ds_name)
            self.index_map.append([(ds_name, ds_class) for _ in range(len(self.index_map),
                                                                      len(self.index_map) + len(dataset.X))])
            gns = getattr(dataset, "gene_names")
            for gn in gns:
                if gn not in self.gene_names:
                    self.gene_map[gn] = self.gene_names_len
                    self.gene_names.append(gn)
                    self.gene_names_len += 1

        self.datasets_used = filtered_classes

    @staticmethod
    def type_handler_dispatch(func):
        def wrapped(self, dataset, *args, **kwargs):
            ds_type = type(dataset)
            if ds_type == GeneExpressionDataset:
                if not hasattr(dataset, "gene_names"):
                    raise ValueError("Provided dataset doesn't have gene_names information.")

                gene_names = getattr(dataset, "gene_names")

                if not dataset.dense:
                    data = dataset.X.toarray()
                else:
                    data = dataset.X

            elif ds_type == tuple:
                data, gene_names = dataset[0:2]

            elif ds_type == pd.DataFrame:
                gene_names = dataset.columns
                data = dataset.values

            elif ds_type in [np.array, torch.tensor]:
                try:
                    gene_names = kwargs.pop("gene_names")
                except KeyError:
                    raise ValueError("No gene names provided to do the handling with.")

                data = dataset

            elif ds_type == sparse.csr_matrix:
                data = dataset.toarray()
                try:
                    gene_names = kwargs.pop("gene_names")
                except KeyError:
                    raise ValueError("No gene names provided to do the handling with.")

            else:
                raise ValueError(f"Provided data type '{type(dataset)}' currently not handled.")

            return func(self, data, gene_names, *args, **kwargs)

        return wrapped

    @type_handler_dispatch
    def map_data(self, data, gene_names, *args, **kwargs):

        data_out = np.zeros_like(len(data), self.gene_names_len)

        remapped_cols = [self.gene_map[gene] for gene in gene_names]
        data_out[:, remapped_cols] = data

        return torch.from_numpy(data_out)

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        return idx

    def collate_fn(self, indices):
        class_and_name = [self.index_map[ind] for ind in indices]

        d, gn, lm, lv, bi, l = self.hdf5_handler[idx]
        return self.map_data(d, gn), lm, lv, bi, l
