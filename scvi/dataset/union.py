from __future__ import annotations
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
from typing import Dict, Tuple, Sequence

class_re_pattern = r"((?<=[.])[A-Za-z_0-9]+(?='>))|((?<=class ')\w+(?='>))"


class AttrLoaderHdf5:
    def __init__(self,
                 attr,
                 hdf5_filepath=None,
                 index_map=None):
        self.attr = attr
        self.hdf5_filepath = hdf5_filepath
        self.index_map = None
        self.set_index_map(index_map)

    @property
    def shape(self):
        return len(self), None

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        data = []
        gen = (self.index_map[i] for i in np.atleast_1d(idx))
        with h5py.File(self.hdf5_filepath, "r") as h5_file:
            for ds_specifier, index in gen:
                group = h5_file[ds_specifier]
                data.append(group[self.attr][index])
        return data

    def set_index_map(self, index_map):
        self.index_map = index_map


class MetaAttrLoaderHdf5(AttrLoaderHdf5):
    def __getitem__(self, idx):
        ds_specifier, index = self.index_map[idx]
        with h5py.File(self.hdf5_filepath, "r") as h5_file:
            group = h5_file[ds_specifier]
            return group[self.attr][:]

    def __len__(self):
        return None


class UnionDataset(GeneExpressionDataset):
    def __init__(self,
                 save_path,
                 low_memory=True,
                 map_fname=None,
                 map_save_fname=None,
                 data_fname=None,
                 data_save_fname=None,
                 gene_name_conversion_map_fname=None
                 ):
        super().__init__()
        self._len = None
        self.gene_map = None
        self.gene_names = []
        self.gene_names_converter = None
        self.gene_names_len = 0
        self.cached_gene_mapping = None

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
            self.gene_map.index = self.gene_map.index.astype(str).str.lower()
            self.gene_map = pd.Series(range(len(self.gene_map)), index=self.gene_map.index)
            self.gene_names = self.gene_map.index
            self.gene_names_len = len(self.gene_names)
            self.gene_names_converter = None
            if gene_name_conversion_map_fname is not None:
                gn_converter = pd.read_csv(
                    os.path.join(save_path, gene_name_conversion_map_fname),
                    header=0,
                    index_col=0
                )
                self.gene_names_converter.index = gn_converter.index.str.lower()

        if data_fname is not None:
            if low_memory:
                self._set_attributes(data_fname)
                self._cache_gene_mapping()
            else:
                self._union_read_hdf5_into_memory(data_fname)

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        return idx

    def convert_gene_names(self, gene_names):
        new_names = self.gene_names_converter.loc[gene_names].values
        return new_names

    def collate_fn_base(self,
                        attributes_and_types,
                        indices
                        ) -> Tuple[torch.Tensor]:
        indices = np.asarray(indices)
        indices.sort()
        if self.low_memory:
            self._cache_gene_mapping()

            batch = defaultdict(list)
            h5_acc_dict = defaultdict(list)
            for idx in indices:
                # sort the indices by the dataset they address for faster loading
                ds_specifier, index = self.index_map[idx]
                h5_acc_dict[ds_specifier].append(index)
            for ds_specifier, ds_indices in h5_acc_dict.items():
                for attr, dtype in attributes_and_types.items():
                    elems = getattr(self, attr)[ds_indices]
                    if attr == "X":
                        elems = np.vstack(elems).astype(dtype)
                        col_indices, map_gene_ind = self.cached_gene_mapping[ds_specifier]
                        elems = self.map_data(elems,
                                              mappable_genes_indices=map_gene_ind,
                                              col_indices=col_indices
                                              )
                        batch[attr].append(elems)
                    else:
                        batch[attr].append(np.asarray(elems).astype(dtype))

            batch_out = []
            for _, elems in batch.items():
                batch_out.append(torch.from_numpy(np.vstack(elems)))
            return tuple(batch_out)
        else:
            return super().collate_fn_base(attributes_and_types, indices)

    def _set_attributes(self, data_fname) -> UnionDataset:
        self.hdf5_filepath = os.path.join(self.save_path, data_fname + ".hdf5")
        # get the info for all shapes of the datasets
        self._fill_index_map()
        self._len = len(self.index_map)
        self.X = AttrLoaderHdf5("X",
                                hdf5_filepath=self.hdf5_filepath, index_map=self.index_map)
        self.gene_names = MetaAttrLoaderHdf5("gene_names",
                                             hdf5_filepath=self.hdf5_filepath, index_map=self.index_map)
        self.batch_indices = MetaAttrLoaderHdf5("batch_indices",
                                                hdf5_filepath=self.hdf5_filepath, index_map=self.index_map)
        self.labels = MetaAttrLoaderHdf5("labels",
                                         hdf5_filepath=self.hdf5_filepath, index_map=self.index_map)

        return self

    def _fill_index_map(self) -> None:
        if not self.index_map:
            with h5py.File(self.hdf5_filepath, "r") as h5_file:
                # Walk through all groups, extracting datasets
                for group_name, group in h5_file.items():
                    shape = group["X"].shape
                    self.index_map.extend([(group_name, i) for i in range(shape[0])])
        return

    def _cache_gene_mapping(self, ) -> None:
        if not self.cached_gene_mapping:
            self.cached_gene_mapping = dict()
            with h5py.File(self.hdf5_filepath, "r") as h5_file:
                # Walk through all groups, extracting datasets
                for group_name, group in h5_file.items():
                    gene_names = np.char.lower(group["gene_names"])
                    mappable_genes_indices = np.isin(gene_names, self.gene_map.index)
                    mappable_genes = gene_names[mappable_genes_indices]
                    col_indices = self.gene_map[mappable_genes].values
                    col_indices.sort()
                    self.cached_gene_mapping[group_name] = (col_indices, mappable_genes_indices.flatten())
        return

    def map_data(self,
                 data,
                 gene_names=None,
                 **kwargs
                 ) -> Union[np.ndarray, sp_sparse.lil_matrix]:
        if isinstance(data, np.ndarray):
            data_out = np.zeros((data.shape[0], self.gene_names_len), dtype=data.dtype)
        else:
            data_out = sp_sparse.lil_matrix(sp_sparse.eye(m=data.shape[0], n=self.gene_names_len, dtype=data.dtype)) * 0

        try:
            mappable_genes_indices = kwargs["mappable_genes_indices"]
        except KeyError:
            mappable_genes_indices = np.isin(np.char.lower(gene_names), self.gene_map.index)
        try:
            col_indices = kwargs["col_indices"]
        except KeyError:
            mappable_genes = gene_names[mappable_genes_indices]
            col_indices = self.gene_map[mappable_genes].values

        data_out[:, col_indices] = data[:, mappable_genes_indices.flatten()]
        return data_out

    def _load_dataset(self,
                      ds_name,
                      ds_class,
                      ds_args,
                      check_for_gene_annotation=True
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

        if check_for_gene_annotation and dataset.gene_names is None:
            # without gene names we can't build a proper mapping
            warnings.warn(
                f"Dataset {(ds_class, ds_name)} doesn't have gene_names as attribute. Skipping this dataset.")
            return None

        return dataset, ds_class, ds_name

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
        if self.map_save_fname:
            self.gene_map.to_csv(
                os.path.join(self.save_path, self.map_save_fname + ".csv"),
                header=False
            )
            pd.Series(list(filtered_classes.values()), index=list(filtered_classes.keys())).to_csv(
                os.path.join(self.save_path, self.map_save_fname + "_used_datasets.csv"),
                header=False
            )

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

    def join_datasets(self,
                      data_source: str,
                      data_target: str,
                      **kwargs):

        if data_source not in ["memory", "hdf5", "scvi"]:
            raise ValueError(f"Parameter 'data_source={data_source}' not supported.")
        if data_target not in ["memory", "hdf5"]:
            raise ValueError(f"Parameter 'data_target={data_target}' not supported.")

        if data_source == "memory":
            if data_target == "memory":
                self._union_combine_memory_into_memory(**kwargs)
            elif data_target == "hdf5":
                self._union_drop_memory_into_hdf5(**kwargs)
        elif data_source == "hdf5":
            if data_target == "memory":
                self._union_read_hdf5_into_memory(**kwargs)
            elif data_target == "hdf5":
                self._union_combine_hdf5_into_hdf5(**kwargs)
        elif data_source == "scvi":
            if data_target == "memory":
                self._union_read_dsets_into_memory(**kwargs)
            elif data_target == "hdf5":
                self._union_combine_dsets_into_hdf5(**kwargs)
        else:
            raise ValueError(f"Parameter 'data_source={data_source}' not supported.")
        return self

    def _union_combine_dsets_into_hdf5(self,
                                       dataset_names,
                                       dataset_classes,
                                       dataset_args=None,
                                       out_fname=None):
        """
        Combines multiple unlabelled gene_datasets based on a mapping of gene names. Stores the final
        dataset onto a Hdf5 file with out_fname filename.
        :param dataset_names: List, list of names complementing the dataset_classes (needed for some classes)
        :param dataset_classes: List, list of class-initializers of scvi GeneExpression datasets
        :param dataset_args: List, list of further positional arguments for when loading the datasets
        :param out_fname: str, name of the file to which to write.
        :return: self (with instantiated dataloaders for data access)
        """

        if out_fname is None:
            out_fname = self.data_save_fname
        string_dt = h5py.special_dtype(vlen=str)
        with h5py.File(os.path.join(self.save_path, out_fname + ".hdf5"), 'w') as hdf:
            if dataset_args is None:
                dataset_args = [None] * len(dataset_names)
            lock = Lock()
            with ThreadPoolExecutor() as executor:
                futures = list(
                    (executor.submit(self._load_dataset,
                                     ds_name,
                                     ds_class,
                                     ds_args,
                                     True)
                     for ds_name, ds_class, ds_args in zip(dataset_names, dataset_classes, dataset_args))
                )
                for future in as_completed(futures):
                    res = future.result()
                    if res is not None:
                        dataset, dataset_class, dataset_fname = res
                        if any(("ENS" not in gn for gn in dataset.gene_names[0:100])):
                            continue

                    dataset_class_str = re.search(class_re_pattern, str(dataset_class)).group()

                    # grab the necessary data parts:
                    # aside from the data itself (X), the gene_names, local means, local_vars, batch_indices and labels
                    # there are no guaranteed attributes of each dataset. Thus for now these will be the ones we
                    # work with
                    X = dataset.X
                    gene_names = np.char.lower(dataset.gene_names)
                    batch_indices = dataset.batch_indices
                    labels = dataset.labels

                    print(f"Writing dataset {dataset_class_str, dataset_fname} to hdf5 file.")

                    # Build the group for the dataset, under which the data is going to be stored
                    # We will store the above mentioned data in the following scheme (as this corresponds
                    # to the hdf5 dataset class):
                    # --1st DS CLASS and NAME
                    # ------ X
                    # ------ gene_names
                    # ------ batch_indices
                    # ------ labels
                    # ------ (cell_types)
                    # --2nd DS CLASS and NAME
                    # ------ ...
                    # ...
                    lock.acquire()
                    dataset_hdf5_g = hdf.create_group(f"{dataset_class_str}_{dataset_fname}")

                    if isinstance(X, sp_sparse.csr_matrix):
                        dset = dataset_hdf5_g.create_dataset("X",
                                                             shape=(X.shape[0], len(dataset.gene_names)),
                                                             dtype=np.int32)
                        nr_rows = 1000
                        for start in range(0, len(dataset), nr_rows):
                            end = start + nr_rows
                            dset[slice(start, end), :] = np.vstack(
                                [X.getrow(i).toarray() for i in range(start, min(end, X.shape[0]))]
                            ).astype(np.int32)
                    else:
                        dataset_hdf5_g.create_dataset("X", data=X)
                    dataset_hdf5_g.create_dataset("gene_names", data=gene_names.astype(np.dtype("S")), dtype=string_dt)
                    dataset_hdf5_g.create_dataset("batch_indices", data=batch_indices)
                    dataset_hdf5_g.create_dataset("labels", data=labels)

                    if hasattr(dataset, "cell_types"):
                        cell_types = dataset.cell_types
                        dataset_hdf5_g.create_dataset("cell_types",
                                                      data=cell_types.astype(np.dtype("S")), dtype=string_dt)
                    lock.release()
        print(f"conversion completed to file '{out_fname}.hdf5'")
        self._set_attributes(out_fname)
        return self

    def _union_combine_memory_into_memory(self,
                                          gene_datasets: List[GeneExpressionDataset],
                                          convert_gene_names=False
                                          ):
        """
        Combines multiple unlabelled gene_datasets based on a mapping of gene names. Loads the final
        dataset directly into memory.
        :param gene_datasets: List, the loaded data sets of (inherited) class GeneExpressionDataset to concatenate.
        :return: self (populated with data)
        """

        X = []
        batch_indices = []
        labels = []
        cell_types = []
        for gene_dataset in gene_datasets:
            gene_names = np.char.lower(gene_dataset.gene_names)
            if convert_gene_names:
                gene_names = self.convert_gene_names(gene_dataset.gene_names)
            X.append(sp_sparse.csr_matrix(self.map_data(gene_dataset.X, gene_names)))
            batch_indices.append(gene_dataset.batch_indices)
            labels.append(gene_dataset.labels)
            if gene_dataset.cell_types is not None:
                cell_types.append(gene_dataset.cell_types)
                labels[-1] = cell_types[-1][labels[-1]]
            else:
                cell_types.append(np.zeros((len(X[-1])), dtype=str))
                labels[-1] = labels[-1] - 1
        cell_types = np.sort(np.unique(np.concatenate(cell_types)))
        cell_types_map = pd.Series(range(len(cell_types)), index=cell_types)
        labels = np.concatenate(labels)

        for i, cell_type in np.ndenumerate(labels):
            if cell_type != -1:
                if cell_type != "undefined":
                    labels[i] = cell_types_map[cell_type]

        self.populate_from_data(X=sp_sparse.vstack(X),
                                batch_indices=np.concatenate(batch_indices),
                                labels=labels,
                                gene_names=self.gene_names,
                                cell_types=cell_types
                                )
        self._len = self.X.shape[0]

        logger.info(f"Joined {len(gene_datasets)} datasets to one of shape {self._len} x {self.gene_names_len}.")

        return self

    def _union_read_dsets_into_memory(self,
                                      dataset_names=None,
                                      dataset_classes=None,
                                      dataset_args=None,
                                      shared_batches=False
                                      ):
        """
        Loads scvi gene_datasets as specified and combines them based on a mapping of gene names.
        Loads the final dataset into memory.
        :param dataset_names: List, list of names complementing the dataset_classes (needed for some classes)
        :param dataset_classes: List, list of class-initializers of scvi GeneExpression datasets
        :param dataset_args: List, list of further positional arguments for when loading the datasets
        :param shared_batches: bool, whether the batch_indices are shared or not for the datasets
        :return: self (populated with data)
        """

        X = []
        batch_indices = []
        labels = []

        cell_types_map = dict()

        if dataset_args is None:
            dataset_args = [None] * len(dataset_names)

        _LOCK = Lock()
        cell_types_map_len = 0
        n_batch_offset = 0

        with ThreadPoolExecutor() as executor:
            futures = list(
                (executor.submit(self._load_dataset,
                                 ds_name,
                                 ds_class,
                                 ds_args,
                                 True)
                 for ds_name, ds_class, ds_args in zip(dataset_names, dataset_classes, dataset_args))
            )
            for future in as_completed(futures):
                res = future.result()
                if res is not None:
                    dataset, _, _ = res

                _LOCK.acquire()
                gene_names = self.convert_gene_names(dataset.gene_names)
                X.append(sp_sparse.csr_matrix(self.map_data(dataset.X, gene_names)))
                bis = dataset.batch_indices

                if bis.sum() == 0:
                    bis = bis + n_batch_offset
                n_batch_offset += (dataset.n_batches if not shared_batches else 0)
                batch_indices.append(bis)

                if dataset.cell_types[0] != "undefined":
                    for cell_type in dataset.cell_types:
                        if cell_type not in cell_types_map:
                            cell_types_map[cell_type] = cell_types_map_len
                            cell_types_map_len += 1
                    ls = np.array([cell_types_map[cell_type] for cell_type in dataset.cell_types], dtype=np.int16)
                else:
                    ls = np.zeros((len(dataset))) * np.nan
                labels.append(ls)
                _LOCK.release()

        if not cell_types_map:
            cell_types = None
            labels = None
        else:
            cell_types = np.unique(np.asarray((x[0] for x in sorted(cell_types_map.items(), key=lambda x: x[1]))))
            labels = np.concatenate(labels)

        for i, label in enumerate(labels):
            if label != 0:
                cell_type = cell_types[i]
                if cell_type != "undefined":
                    labels[i] = cell_types_map[cell_type]

        self.populate_from_data(X=sp_sparse.vstack(X),
                                batch_indices=np.concatenate(batch_indices),
                                labels=labels,
                                gene_names=self.gene_names,
                                cell_types=cell_types
                                )
        self._len = self.X.shape[0]
        return self

    def _union_read_hdf5_into_memory(self,
                                     fname=None):
        if fname is None:
            fname = self.data_fname

        X = []
        batch_indices = []
        labels = []
        cell_types = []

        with h5py.File(os.path.join(self.save_path, fname + ".hdf5"), 'r') as h5_file:
            for group_name, group in h5_file.items():
                X.append(sp_sparse.csr_matrix(self.map_data(group["X"][:], group["gene_names"][:])))
                batch_indices.append(group["batch_indices"][:])
                labels.append(group["labels"][:])
                if "cell_types" in group:
                    cell_types.append(group["cell_types"][:])
                else:
                    cell_types.append(np.zeros((len(X[-1])), dtype=str))
        cell_types = np.sort(np.unique(np.concatenate(cell_types)))
        cell_types_map = pd.Series(range(1, len(cell_types) + 1), index=cell_types)
        labels = np.concatenate(labels)

        for i, label in enumerate(labels):
            if label != 0:
                cell_type = cell_types[i]
                if cell_type != "undefined":
                    labels[i] = cell_types_map[cell_type]

        self.populate_from_data(X=sp_sparse.vstack(X),
                                batch_indices=np.concatenate(batch_indices),
                                labels=labels,
                                gene_names=self.gene_names,
                                cell_types=cell_types
                                )
        self._len = self.X.shape[0]
        return

    # @property
    # def X(self):
    #     return self._X

    # @X.setter
    # def X(self, X: Union[np.ndarray, sp_sparse.csr_matrix]):
    #     """Sets the data attribute ``X`` and (re)computes the library size."""
    #     n_dim = len(X.shape)
    #     if n_dim != 2:
    #         raise ValueError(
    #             "Gene expression data should be 2-dimensional not {}-dimensional.".format(
    #                 n_dim
    #             )
    #         )
    #     self._X = X
    #     if hasattr(X, "shape"):
    #         self._len = X.shape[0]

    @property
    def nb_genes(self) -> int:
        return self.gene_names_len
