import numpy as np
import pandas as pd
from datetime import datetime
import warnings
from scipy import sparse
from scvi.dataset import *
from scvi.dataset.dataset import *
import torch
from torch.utils.data import Dataset
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count, Pool, Lock, Process, Value
from functools import wraps
import sys
from tqdm import tqdm
import re
import h5py
import time

class_re_pattern = r"((?<=[.])\w+(?='>))|((?<=class ')\w+(?='>))"


class UnionDataset(GeneExpressionDataset):
    def __init__(self,
                 save_path,
                 low_memory=True,
                 map_fname=None,
                 map_save_fname=None,
                 data_fname=None,
                 data_save_fname=None
                 ):
        super().__init__()
        self.gene_map = None
        self.gene_names = []
        self.gene_names_converter = None
        self.gene_names_len = 0
        self.gene_names_processed = None

        self.save_path = save_path
        self.line_offsets = None

        self.index_map = []
        self.gene_names_map = dict()

        self.map_fname = map_fname
        self.map_save_fname = map_save_fname
        self.data_fname = data_fname
        self.data_save_fname = data_save_fname
        self.low_memory = low_memory

        self.hdf5_filepath = None

        if map_fname is not None:
            self.gene_map = pd.read_csv(
                os.path.join(self.save_path, map_fname + ".csv"),
                header=0,
                index_col=0
            ).sort_index()
            self.gene_map = pd.Series(range(len(self.gene_map)), index=self.gene_map.index)
            self.gene_names = self.gene_map.index
            self.gene_names_len = len(self.gene_names)
            self.gene_names_converter = pd.read_csv(
                os.path.join(save_path, "ensembl_human_conversion.csv"),
                header=0
            ).set_index("Gene stable ID")["Gene name"]

        if data_fname is not None:
            if not low_memory:
                X = np.array(self._read_nonuniform_csv(self.data_fname + '_X.nucsv'))
                gn = np.array(self._read_nonuniform_csv(self.data_fname + '_gene_names.nucsv'))
                lm = np.array(self._read_nonuniform_csv(self.data_fname + '_local_means.nucsv'))
                lv = np.array(self._read_nonuniform_csv(self.data_fname + '_local_vars.nucsv'))
                bi = np.array(self._read_nonuniform_csv(self.data_fname + '_batch_indices.nucsv'))
                l = np.array(self._read_nonuniform_csv(self.data_fname + '_labels.nucsv'))
                self.populate_from_data(X, gene_names=gn, batch_indices=bi, labels=l)
                self.local_means = lm
                self.local_vars = lv

                self.X_len = X.shape[0]
            else:
                self.set_filepaths(save_path, data_fname)
                self._fill_index_map()
                self._cache_processed_gene_names()

        # helper member for multiprocessing to avoid pickling big data files.
        # Don't access it outside of the functions
        self.dataset_holder = None

    def __len__(self):
        return self.X_len

    def __getitem__(self, idx):
        return idx

    @property
    def nb_genes(self) -> int:
        return self.gene_names_len

    def collate_fn_base(self,
                        attributes_and_types,
                        indices):
        indices = np.asarray(indices)
        indices.sort()

        self._cache_processed_gene_names()

        batch = defaultdict(list)
        h5_acc_dict = defaultdict(list)
        for idx in indices:
            ds_specifier, index = self.index_map[idx]
            h5_acc_dict[ds_specifier].append(index)
        with h5py.File(self.hdf5_filepath, "r") as h5_file:
            for ds_specifier, ds_indices in h5_acc_dict.items():
                group = h5_file[ds_specifier]
                for attr, _ in attributes_and_types.items():
                    if attr == "X":
                        col_indices, map_gene_ind = self.gene_names_processed[ds_specifier]
                        elem = self.map_data(data=group[attr][ds_indices],
                                             mappable_genes_indices=map_gene_ind,
                                             col_indices=col_indices
                                             )
                    elif attr in ["local_means", "local_vars"]:
                        elem = group[attr][ds_indices]
                    elif attr in ["batch_indices", "labels"]:
                        elem = group[attr]
                    else:
                        raise ValueError("Unknown attribute demanded.")
                    batch[attr].append(elem)

        batch_out = []
        for _, data in batch.items():
            batch_out.append(torch.from_numpy(np.vstack(data)))
        return tuple(batch_out)

    def set_filepaths(self, save_path, data_fname):
        self.hdf5_filepath = os.path.join(save_path, data_fname + ".hdf5")
        return self

    def build_mapping(self,
                      dataset_names,
                      dataset_classes,
                      dataset_args=None,
                      multiprocess=True,
                      **kwargs
                      ):

        if self.gene_map is not None:
            return

        if dataset_args is None:
            dataset_args = [None] * len(dataset_names)

        filtered_classes = defaultdict(list)

        if multiprocess:
            gene_map = self._build_mapping_mp(dataset_names, dataset_classes, filtered_classes,
                                              dataset_args=dataset_args, **kwargs)
        else:
            gene_map = self._build_mapping_serial(dataset_names, dataset_classes, filtered_classes,
                                                  dataset_args=dataset_args, **kwargs)

        self.gene_map = pd.Series(list(gene_map.values()), index=list(gene_map.keys()))
        self.gene_names_len = len(gene_map)
        self.datasets_used = filtered_classes
        if self.map_save_fname:
            self.gene_map.to_csv(
                os.path.join(self.save_path, self.map_save_fname + ".csv"),
                header=False
            )
            pd.Series(list(self.datasets_used.values()), index=list(self.datasets_used.keys())).to_csv(
                os.path.join(self.save_path, self.map_save_fname + "_used_datasets.csv"),
                header=False
            )

    def _load_dataset(self,
                      ds_name,
                      ds_class,
                      ds_args,
                      check_for_genenames=True
                      ):
        print(f"Loading {ds_class, ds_name}.")
        if ds_name is not None:
            if ds_args is not None:
                dataset = ds_class(ds_name, save_path=self.save_path, **ds_args)
            else:
                dataset = ds_class(ds_name, save_path=self.save_path)
        else:
            if ds_args is not None:
                dataset = ds_class(save_path=self.save_path, **ds_args)
            else:
                dataset = ds_class(save_path=self.save_path)

        if check_for_genenames and dataset.gene_names is None:
            # without gene names we can't build a proper mapping
            warnings.warn(
                f"Dataset {(ds_class, ds_name)} doesn't have gene_names as attribute. Skipping this dataset.")
            return None

        return dataset, ds_class, ds_name

    def _build_mapping_serial(self,
                              dataset_names,
                              dataset_classes,
                              filtered_classes,
                              dataset_args=None,
                              **kwargs
                              ):
        gene_map = dict()
        gene_names_len = 0
        for ds_name, ds_class, ds_args in zip(dataset_names, dataset_classes, dataset_args):
            res = self._load_dataset(ds_name, ds_class, ds_args)
            if res is None:
                continue
            dataset = res[0]
            filtered_classes[re.search(class_re_pattern, str(ds_class)).group()].append(str(ds_name))

            print("Extending gene list...", end="")
            sys.stdout.flush()
            gns = getattr(dataset, "gene_names")
            for gn in gns:
                if gn not in self.gene_names:
                    gene_map[gn] = gene_names_len
                    gene_names_len += 1
            print("done!")
            sys.stdout.flush()

        return gene_map

    def _build_mapping_mp(self,
                          dataset_names,
                          dataset_classes,
                          filtered_classes,
                          dataset_args=None,
                          **kwargs
                          ):
        total_genes = set()
        with ProcessPoolExecutor(max_workers=min(len(dataset_names), cpu_count() // 2)) as executor:
            futures = list(
                (executor.submit(self._load_dataset,
                                 ds_name,
                                 ds_class,
                                 ds_args)
                 for ds_name, ds_class, ds_args in zip(dataset_names, dataset_classes, dataset_args))
            )
            for future in as_completed(futures):
                res = future.result()
                if res is not None:
                    total_genes = total_genes.union(res[0].gene_names)
                    filtered_classes[re.search(class_re_pattern, str(res[1])).group()].append(res[2])

        gene_map = {gene: pos for (gene, pos) in zip(total_genes, range(len(total_genes)))}
        return gene_map

    def _type_dispatch(func):
        @wraps(func)
        def wrapped(self,
                    dataset,
                    *args,
                    **kwargs
                    ):
            ds_type = type(dataset)
            if ds_type == GeneExpressionDataset:
                if dataset.gene_names is not None:
                    raise ValueError("Provided dataset doesn't have gene_names information.")

                gene_names = dataset.gene_names

                if not dataset.dense:
                    data = dataset.X.toarray()
                else:
                    data = dataset.X

            elif ds_type == tuple:
                data, gene_names = dataset[0:2]

            elif ds_type == pd.DataFrame:
                gene_names = dataset.columns
                data = dataset.values

            elif ds_type in [np.ndarray, torch.tensor]:
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

    @_type_dispatch
    def map_data(self,
                 data,
                 gene_names=None,
                 **kwargs
                 ):
        # print("Grabbing batch from file...", end="")
        # s = time.perf_counter()
        data_out = np.zeros((len(data), self.gene_names_len), dtype=np.float32)
        try:
            mappable_genes_indices = kwargs["mappable_genes_indices"]
        except KeyError:
            mappable_genes_indices = np.isin(gene_names, self.gene_map.index)
        try:
            col_indices = kwargs["col_indices"]
        except KeyError:
            mappable_genes = gene_names[mappable_genes_indices]
            col_indices = self.gene_map[mappable_genes].values

        data_out[:, col_indices] = data[:, mappable_genes_indices.flatten()]
        # print(f"done ({time.perf_counter()-s:.2f}s).")
        return data_out

    @staticmethod
    def _toarray(data):
        return data.toarray()

    def _fill_index_map(self):
        if self.index_map:
            # map already filled
            return

        with h5py.File(self.hdf5_filepath, "r") as h5_file:
            # Walk through all groups, extracting datasets
            for group_name, group in h5_file.items():
                shape = group["data"].shape
                self.index_map.extend([(group_name, i) for i in range(shape[0])])

    def _cache_processed_gene_names(self, ):
        with h5py.File(self.hdf5_filepath, "r") as h5_file:
            # Walk through all groups, extracting datasets
            for group_name, group in h5_file.items():
                gene_names = group["gene_names"]
                mappable_genes_indices = np.isin(gene_names, self.gene_map.index)
                mappable_genes = gene_names[mappable_genes_indices]
                col_indices = self.gene_map[mappable_genes].values
                self.gene_names_processed[group_name] = (col_indices, mappable_genes_indices.flatten())

    def concat_to_hdf5(self,
                       dataset_names,
                       dataset_classes,
                       dataset_args=None,
                       out_fname=None):
        if out_fname is None:
            out_fname = self.data_save_fname
        with h5py.File(os.path.join(self.save_path, out_fname + ".hdf5"), 'w') as hdf:
            if dataset_args is None:
                dataset_args = [None]*len(dataset_names)
            for dataset_fname, dataset_class, dataset_arg in zip(dataset_names, dataset_classes, dataset_args):

                dataset_class_str = re.search(class_re_pattern, str(dataset_class)).group()

                dataset, _, _ = self._load_dataset(dataset_fname, dataset_class, dataset_arg, check_for_genenames=True)

                # grab the necessary data parts:
                # aside from the data itself (X), the gene_names, local means, local_vars, batch_indices and labels
                # there are no guaranteed attributes of each dataset. Thus for now these will be the ones we
                # work with
                gene_names = dataset.gene_names
                local_means = dataset.local_means
                local_vars = dataset.local_vars
                batch_indices = dataset.batch_indices
                labels = dataset.labels

                print(f"Writing dataset {dataset_class_str, dataset_fname} to hdf5 file.")

                # Build the group for the dataset, under which the data is going to be stored
                # We will store the above mentioned data in the following scheme (as this corresponds
                # to the hdf5 dataset class):
                # --1st DS NAME and CLASS
                # ------ X
                # ------ gene_names
                # ------ local_means
                # ------ local_vars
                # ------ batch_indices
                # ------ labels
                # --2nd DS NAME and CLASS
                # ------ ...
                # ...
                dataset_hdf5_g = hdf.create_group(f"{dataset_class_str}_{dataset_fname}")

                dataset_hdf5_g.create_dataset("X", data=dataset.X)
                dataset_hdf5_g.create_dataset("gene_names", data=gene_names)
                dataset_hdf5_g.create_dataset("local_means", data=local_means)
                dataset_hdf5_g.create_dataset("local_vars", data=local_vars)
                dataset_hdf5_g.create_dataset("batch_indices", data=batch_indices)
                dataset_hdf5_g.create_dataset("labels", data=labels)

                if hasattr(dataset, "cell_types"):
                    cell_types = dataset.cell_types
                    dataset_hdf5_g.create_dataset("cell_types", data=cell_types)

        print(f"conversion completed to file '{out_fname}.hdf5'")

    @staticmethod
    def concat_datasets_union(*gene_datasets,
                              on='gene_names',
                              shared_labels=True,
                              shared_batches=False
                              ):
        """
        Combines multiple unlabelled gene_datasets based on the union of gene names.
        Batch indices are generated in the same order as datasets are given.
        :param gene_datasets: a sequence of gene_datasets object
        :return: a GeneExpressionDataset instance of the concatenated datasets
        """
        assert all([getattr(gene_dataset, on) is None for gene_dataset in gene_datasets])

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
            indices = []
            for gn in dataset.gene_names:
                indices.append(gene_names.index(gn))
            subset_genes = np.array(indices)
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
            if all([gene_dataset.cell_types is not None for gene_dataset in gene_datasets]):
                cell_types = list(
                    set([cell_type for gene_dataset in gene_datasets for cell_type in gene_dataset.cell_types])
                )
                labels = []
                for gene_dataset in gene_datasets:
                    mapping = [cell_types.index(cell_type) for cell_type in gene_dataset.cell_types]
                    labels += [remap_categories(gene_dataset.labels, mapping_to=mapping)[0]]
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
        result = GeneExpressionDataset()
        result.populate_from_data(X, batch_indices=batch_indices, labels=labels,
                                  gene_names=gene_names, cell_types=cell_types)
        result.local_means = local_means
        result.local_vars = local_vars
        result.barcodes = [gene_dataset.barcodes if hasattr(gene_dataset, 'barcodes') else None
                           for gene_dataset in gene_datasets]
        return result

    _type_dispatch = staticmethod(_type_dispatch)
