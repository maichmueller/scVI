import numpy as np
import pandas as pd
from datetime import datetime
import warnings
from scipy import sparse
from .dataset import *
from .dataset_hdf import HDF5Dataset, convert_to_hdf5
import torch
from torch.utils.data import Dataset
from collections import defaultdict
import sys


class IndepUnionDataset(Dataset):

    def __init__(self, save_path, low_memory=True,
                 load_map_fname=None, save_mapping_fname=None,
                 data_fname=None, data_savename=None, col_width=None):

        self.gene_map = dict()
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
        if low_memory:
            super().__init__([], [], [], [], [])
        else:
            self.concat_datasets_union()
            pass

    def build_mapping(self, dataset_names, dataset_classes, dataset_args=None, **kwargs):
        if self.gene_map:
            if self.low_memory and self.X is None:
                self.concat_to_fwf(dataset_names, dataset_classes, dataset_args,
                                   out_fname=self.data_savename, **kwargs)
            else:
                print(f'Mapping and data already built/loaded (potentially from '
                      f'files {self.data_fname} and {self.load_map_fname}).')
            return

        if dataset_args is None:
            dataset_args = [None] * len(dataset_names)

        filtered_classes = defaultdict(list)
        for ds_name, ds_class, ds_args in zip(dataset_names, dataset_classes, dataset_args):
            if ds_args is not None:
                dataset = ds_class(ds_name, save_path=self.save_path, **ds_args)
            else:
                dataset = ds_class(ds_name, save_path=self.save_path)

            if not hasattr(dataset, "gene_names"):
                # without gene names we can't build a proper mapping
                warnings.warn(f"Dataset {ds_name} doesn't have gene_names as attribute. Skipping this dataset.")
                continue

            filtered_classes[str(ds_class)].append(str(ds_name))

            print("Extending gene list...", end="")
            sys.stdout.flush()
            gns = getattr(dataset, "gene_names")
            for gn in gns:
                if gn not in self.gene_names:
                    self.gene_map[gn] = self.gene_names_len
                    self.gene_names.append(gn)
                    self.gene_names_len += 1
            print("done!")
            sys.stdout.flush()
        self.datasets_used = filtered_classes

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
        return len(self.gene_map)

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

    def concat_to_fwf(self, dataset_names, dataset_classes, dataset_args=None, out_fname=None, col_width=16,
                      **kwargs):
        if dataset_args is None:
            dataset_args = [dataset_args] * len(dataset_names)
        if self.col_width is None:
            self.col_width = col_width
        else:
            if self.col_width != col_width:
                warnings.warn(f"Column width was already specified in the data handler object (width={self.col_width}),"
                              f" but was also passed as overwriting parameter of differing value (width={col_width})."
                              f" Remember to adapt the column width when loading the file later.")
        if out_fname is None:
            out_fname = self.data_savename

        n_batch_offset = 0
        try:
            shared_batches = kwargs.pop("shared_batches")
        except KeyError:
            shared_batches = False
        try:
            shared_labels = kwargs.pop("shared_labels")
        except KeyError:
            shared_labels = False

        used_datasets = dict()
        for dataset_fname, dataset_class, dataset_arg in zip(dataset_names, dataset_classes, dataset_args):
            if self.datasets_used:
                sets = self.datasets_used[dataset_class]
                if sets:
                    if dataset_fname not in sets:
                        continue

            if dataset_fname is None:
                if dataset_arg is not None:
                    dataset = dataset_class(dataset_arg, save_path=self.save_path)
                else:
                    dataset = dataset_class(save_path=self.save_path)
            else:
                if dataset_arg is not None:
                    dataset = dataset_class(dataset_fname, dataset_arg, save_path=self.save_path)
                else:
                    dataset = dataset_class(dataset_fname, save_path=self.save_path)

            if not hasattr(dataset, "gene_names"):
                continue

            used_datasets[str(dataset_class)] = dataset_fname

            # grab the necessary data parts:
            # aside from the data itself (X), the gene_names, local means, local_vars, batch_indices and labels
            # there are no guaranteed attributes of each dataset. Thus for now these will be the ones we use
            if not dataset.dense:
                dataset.X = dataset.X.toarray()
            data = self.map_data(dataset.X)
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

            with open(self.save_path + '/' + out_fname + '_data.fwf', 'ab') as d,\
                 open(self.save_path + '/' + out_fname + '_genemeans.fwf', 'ab') as gn,\
                 open(self.save_path + '/' + out_fname + '_localmeans.fwf', 'ab') as lm,\
                 open(self.save_path + '/' + out_fname + '_localvars.fwf', 'ab') as lv,\
                 open(self.save_path + '/' + out_fname + '_batchindices.fwf', 'ab') as bi,\
                 open(self.save_path + '/' + out_fname + '_labels.fwf', 'ab') as l:

                for row in data:
                    d.write("".join([f"{entry: <{col_width}}" for entry in row]) + '\n')
                gn.write("".join([f"{entry: <{col_width}}" for entry in gene_names]) + '\n')
                lm.write("".join([f"{entry: <{col_width}}" for entry in local_means]) + '\n')
                lv.write("".join([f"{entry: <{col_width}}" for entry in local_vars]) + '\n')

                batch_indices = batch_indices + n_batch_offset
                n_batch_offset += dataset.n_batches if not shared_batches else 0
                bi.write("".join([f"{entry: <{col_width}}" for entry in batch_indices]) + '\n')

                if shared_labels:
                    if all([hasattr(gene_dataset, "cell_types") for gene_dataset in gene_datasets]):
                        cell_types = list(
                            set([cell_type for gene_dataset in gene_datasets for cell_type in gene_dataset.cell_types])
                        )
                        labels = []
                        for gene_dataset in gene_datasets:
                            mapping = [cell_types.index(cell_type) for cell_type in gene_dataset.cell_types]
                            labels += [arrange_categories(gene_dataset.labels, mapping_to=mapping)[0]]
                        labels = np.concatenate(labels)
                    else:
                        labels = np.concatenate([gene_dataset.labels for gene_dataset in gene_datasets])
                else:
                    labels = np.zeros((new_shape[0], 1))
                    n_labels_offset = 0
                    current_index = 0
                    for gene_dataset in gene_datasets:
                        next_index = current_index + len(gene_dataset)
                        labels[current_index:next_index] = gene_dataset.labels + n_labels_offset
                        n_labels_offset += gene_dataset.n_labels
                        current_index = next_index

                l.write("".join([f"{entry: <{col_width}}" for entry in labels]) + '\n')

        print(f"Conversion completed to file '{out_fname}_.fwf'")

    @staticmethod
    def _available_genes(dataset, on_ref, on="gene_names"):
        indices = []
        for gn in getattr(dataset, on):
            indices.append(on_ref.index(gn))
        return np.array(indices)

    @staticmethod
    def concat_datasets_union(*gene_datasets, on='gene_names', shared_labels=True, shared_batches=False):
        """
        Combines multiple unlabelled gene_datasets based on the union of gene names intersection.
        Datasets should all have gene_dataset.n_labels=0.
        Batch indices are generated in the same order as datasets are given.
        :param gene_datasets: a sequence of gene_datasets object
        :return: a GeneExpressionDataset instance of the concatenated datasets
        """
        assert all([hasattr(gene_dataset, on) for gene_dataset in gene_datasets])

        gene_names_ref = set()
        gene_names = []
        # add all gene names in the order they appear in the datasets
        for gene_dataset in gene_datasets:
            for name in list(set(getattr(gene_dataset, on))):
                if name not in gene_names_ref:
                    gene_names.append(name)
                gene_names_ref.add(name)

        print("All genes used %d" % len(gene_names))

        new_shape = (sum(len(dataset) for dataset in gene_datasets), len(gene_names))
        if sum((data.dense for data in gene_datasets)) > 0.5:
            # most datasets provided are dense
            X = np.zeros(new_shape, dtype=float)
        else:
            X = sp_sparse.csr_matrix(new_shape, dtype=float)
        start_row = 0
        # build a new dataset out of all datasets
        for dataset in gene_datasets:
            ds_len = len(dataset)
            subset_genes = IndepUnionDataset._available_genes(dataset, gene_names, on=on)
            X[start_row:start_row + ds_len, subset_genes] = dataset.X
            start_row += ds_len

        if not any([gene_dataset.dense for gene_dataset in gene_datasets]):
            X = sp_sparse.csr_matrix(X)

        batch_indices = np.zeros((new_shape[0], 1))
        n_batch_offset = 0
        current_index = 0
        for gene_dataset in gene_datasets:
            next_index = current_index + len(gene_dataset)
            batch_indices[current_index:next_index] = gene_dataset.batch_indices + n_batch_offset
            n_batch_offset += (gene_dataset.n_batches if not shared_batches else 0)
            current_index = next_index

        cell_types = None
        if shared_labels:
            if all([hasattr(gene_dataset, "cell_types") for gene_dataset in gene_datasets]):
                cell_types = list(
                    set([cell_type for gene_dataset in gene_datasets for cell_type in gene_dataset.cell_types])
                )
                labels = []
                for gene_dataset in gene_datasets:
                    mapping = [cell_types.index(cell_type) for cell_type in gene_dataset.cell_types]
                    labels += [arrange_categories(gene_dataset.labels, mapping_to=mapping)[0]]
                labels = np.concatenate(labels)
            else:
                labels = np.concatenate([gene_dataset.labels for gene_dataset in gene_datasets])
        else:
            labels = np.zeros((new_shape[0], 1))
            n_labels_offset = 0
            current_index = 0
            for gene_dataset in gene_datasets:
                next_index = current_index + len(gene_dataset)
                labels[current_index:next_index] = gene_dataset.labels + n_labels_offset
                n_labels_offset += gene_dataset.n_labels
                current_index = next_index

        local_means = np.concatenate([gene_dataset.local_means for gene_dataset in gene_datasets])
        local_vars = np.concatenate([gene_dataset.local_vars for gene_dataset in gene_datasets])
        result = GeneExpressionDataset(X, local_means, local_vars, batch_indices, labels,
                                       gene_names=gene_names, cell_types=cell_types)
        result.barcodes = [gene_dataset.barcodes if hasattr(gene_dataset, 'barcodes') else None
                           for gene_dataset in gene_datasets]
        return result

    type_handler_dispatch = staticmethod(type_handler_dispatch)
