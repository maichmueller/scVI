import numpy as np
import pandas as pd
from datetime import datetime
import warnings
from scipy import sparse
from .dataset import GeneExpressionDataset
from .dataset_hdf import HDF5Dataset, convert_to_hdf5
import torch
from torch.utils import data
from collections import defaultdict


class IndepUnionDataset(data.dataset):

    def __init__(self, save_path, low_memory=True,
                 load_map_fname=None, save_mapping_fname=None,
                 data_fname=None, data_savename=None, col_width=None):
        self.data = None

        self.gene_map = dict()
        self.index_map = []
        self.gene_names, self.gene_names_len = [], 0

        self.datasets_used = None

        self.save_path = save_path

        self.load_map_fname = load_map_fname
        self.save_mapping_fname = save_mapping_fname
        if load_map_fname is not None:
            self.gene_map = pd.read_csv(self.save_path + load_map_fname + ".csv")

        self.data_fname = data_fname
        self.data_savename = data_savename

        self.low_memory = low_memory
        self.col_width = col_width

        # if hdf5_data_fname is not None:
        #     self.hdf5_handler = HDF5Dataset(hdf5_data_fname, False, low_memory, **kwargs)
        # else:
        #     self.hdf5_handler = None

    def build_mapping(self, dataset_names, dataset_classes, dataset_args):
        if self.gene_map:
            if self.data is None:
                self.concat_to_fwf(dataset_names, dataset_classes, dataset_args, save_path=self.save_path,
                                   out_fname=self.data_savename)
            else:
                print(f'Mapping and data already built/loaded (potentially from '
                      f'files {self.data_fname} and {self.load_map_fname}).')
            return

        filtered_classes = defaultdict(list)
        for ds_name, ds_class, ds_args in zip(dataset_names, dataset_classes, dataset_args):
            if ds_args:
                dataset = ds_class(ds_name, save_path=self.save_path, **ds_args)
            else:
                dataset = ds_class(ds_name, save_path=self.save_path)

            if not hasattr(dataset, "gene_names"):
                # without gene names we can't build a proper mapping
                warnings.warn(f"Dataset {ds_name} doesn't have gene_names as attribute. Skipping this dataset.")
                continue

            filtered_classes[str(ds_class)].append(str(ds_name))
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

        return data_out

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        return idx

    def collate_fn(self, indices):
        indices = sorted(indices)

        data_sample = []
        gene_sample = []
        local_means_sample = []
        local_vars_sample = []
        batch_indices_sample = []
        labels_sample = []

        with open(self.data_fname + '_data.fwf', 'rb') as d, \
             open(self.data_fname + '_genenames.fwf', 'rb') as gn, \
             open(self.data_fname + '_localmeans.fwf', 'rb') as lm, \
             open(self.data_fname + '_localvars.fwf', 'rb') as lv, \
             open(self.data_fname + '_batchindices.fwf', 'rb') as bi, \
             open(self.data_fname + '_labels.fwf', 'rb') as l:

            for i in indices:
                d.seek(i)
                gn.seek(i)
                lm.seek(i)
                lv.seek(i)
                bi.seek(i)
                l.seek(i)

                # Append the line to the sample sets
                data_sample.append(d.readline().strip())
                gene_sample.append(gn.readline().strip())
                local_means_sample.append(lm.readline().strip())
                local_vars_sample.append(lv.readline().strip())
                batch_indices_sample.append(bi.readline().strip())
                labels_sample.append(l.readline().strip())

        return (torch.FloatTensor(data_sample),
                torch.FloatTensor(local_means_sample),
                torch.FloatTensor(local_vars_sample),
                torch.LongTensor(batch_indices_sample),
                torch.LongTensor(labels_sample))

    def concat_to_fwf(self, dataset_names, dataset_classes, dataset_args, save_path, out_fname=None, col_width=16):
        if self.col_width is None:
            self.col_width = col_width
        else:
            if self.col_width != col_width:
                warnings.warn(f"Column width was already specified in the data handler object (width={self.col_width}),"
                              f" but was also passed as overwriting parameter of differing value (width={col_width})."
                              f" Remember to adapt the column width when loading the file later.")
        if out_fname is None:
            out_fname = self.data_savename

        used_datasets = dict()
        for dataset_fname, dataset_class, dataset_arg in zip(dataset_names, dataset_classes, dataset_args):
            if self.datasets_used:
                sets = self.datasets_used[dataset_class]
                if sets:
                    if dataset_fname not in sets:
                        continue

            if dataset_fname is None:
                if dataset_arg is not None:
                    dataset = dataset_class(dataset_arg, save_path=save_path)
                else:
                    dataset = dataset_class(save_path=save_path)
            else:
                if dataset_arg is not None:
                    dataset = dataset_class(dataset_fname, dataset_arg, save_path=save_path)
                else:
                    dataset = dataset_class(dataset_fname, save_path=save_path)

            if not hasattr(dataset, "gene_names"):
                continue

            used_datasets[str(dataset_class)] = dataset_fname

            # grab the necessary data parts:
            # aside from the data itself (X), the gene_names, local means, local_vars, batch_indices and labels
            # there are no guaranteed attributes of each dataset. Thus for now these will be the ones we use
            if not dataset.dense:
                dataset.X = dataset.X.toarray()
            data = self.map_to_genes(dataset.X)
            gene_names = dataset.gene_names.flatten()
            local_means = dataset.local_means.flatten()
            local_vars = dataset.local_vars.flatten()
            batch_indices = dataset.batch_indices.flatten()
            labels = dataset.labels.flatten()

            len_data = len(dataset)

            print(f"Writing dataset {dataset_class, dataset_fname} of length {len_data} to file.")

            # Build the group files for the dataset, under which the data is going to be stored
            # We will store the above mentioned data in the following scheme:
            # out_fname_data.fwf
            # out_fname_genenames.fwf
            # out_fname_localmeans.fwf
            # out_fname_localvars.fwf
            # out_fname_batchindices.fwf
            # out_fname_labels.fwf

            with open(self.save_path + '/' + out_fname + '_data.fwf', 'a') as d,\
                 open(self.save_path + '/' + out_fname + '_genemeans.fwf', 'a') as gn,\
                 open(self.save_path + '/' + out_fname + '_localmeans.fwf', 'a') as lm,\
                 open(self.save_path + '/' + out_fname + '_localvars.fwf', 'a') as lv,\
                 open(self.save_path + '/' + out_fname + '_batchindices.fwf', 'a') as bi,\
                 open(self.save_path + '/' + out_fname + '_labels.fwf', 'a') as l:

                for row in data:
                    d.write("".join([f"{entry: <{col_width}}" for entry in row]) + '\n')
                gn.write("".join([f"{entry: <{col_width}}" for entry in gene_names]) + '\n')
                lm.write("".join([f"{entry: <{col_width}}" for entry in local_means]) + '\n')
                lv.write("".join([f"{entry: <{col_width}}" for entry in local_vars]) + '\n')
                bi.write("".join([f"{entry: <{col_width}}" for entry in batch_indices]) + '\n')
                l.write("".join([f"{entry: <{col_width}}" for entry in labels]) + '\n')

        print(f"Conversion completed to file '{out_fname}_.fwf'")

