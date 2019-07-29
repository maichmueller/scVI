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

class_re_pattern = r"((?<=[.])[A-Za-z_]+(?='>))|((?<=class ')\w+(?='>))"


class Hdf5AttributeLoader:
    def __init__(self, attr, is_metadata, hdf5_filepath=None, index_map=None):
        self.attr = attr
        self.hdf5_filepath = hdf5_filepath
        self.index_map = None
        self.set_index_map(index_map)

    def set_index_map(self, index_map):
        self.index_map = index_map

    def __getitem_data(self, idx):
        ds_specifier, index = self.index_map[idx]
        with h5py.File(self.hdf5_filepath, "r") as h5_file:
            group = h5_file[ds_specifier]
            return group[self.attr][index]

    def __getitem_metadata(self, idx):
        ds_specifier, index = self.index_map[idx]
        with h5py.File(self.hdf5_filepath, "r") as h5_file:
            group = h5_file[ds_specifier]
            return group[self.attr][:]


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
        self.X_len = None
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
            if low_memory:
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
        if self.low_memory:
            self._cache_processed_gene_names()

            batch = defaultdict(list)
            h5_acc_dict = defaultdict(list)
            for idx in indices:
                ds_specifier, index = self.index_map[idx]
                h5_acc_dict[ds_specifier].append(index)
            with h5py.File(self.hdf5_filepath, "r") as h5_file:
                for ds_specifier, ds_indices in h5_acc_dict.items():
                    group = h5_file[ds_specifier]
                    for attr, dtype in attributes_and_types.items():
                        if attr == "X":
                            col_indices, map_gene_ind = self.gene_names_processed[ds_specifier]
                            elem = self.map_data(group[attr][ds_indices],
                                                 mappable_genes_indices=map_gene_ind,
                                                 col_indices=col_indices
                                                 )
                        elif attr in ["local_means", "local_vars"]:
                            elem = group[attr][ds_indices]
                        elif attr in ["batch_indices", "labels"]:
                            elem = group[attr][:]
                        else:
                            raise ValueError("Unknown attribute demanded.")
                        batch[attr].append(elem.astype(dtype))

            batch_out = []
            for _, data in batch.items():
                batch_out.append(torch.from_numpy(np.vstack(data)))
            return tuple(batch_out)
        else:
            return super().collate_fn_base(attributes_and_types, indices)

    def set_filepaths(self, save_path, data_fname):
        self.hdf5_filepath = os.path.join(save_path, data_fname + ".hdf5")
        # get the info for all shapes of the datasets
        len = 0
        with h5py.File(self.hdf5_filepath, "r") as h5_file:
            # Walk through all groups, extracting data shape info
            for group_name, group in h5_file.items():
                len += group["X"].shape[0]
        self.X_len = len
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
            if isinstance(dataset, GeneExpressionDataset):
                if dataset.gene_names is None:
                    raise ValueError("Provided dataset doesn't have gene_names information.")

                gene_names = dataset.gene_names
                data = dataset.X

            elif isinstance(dataset, tuple):
                data, gene_names = dataset[0:2]

            elif isinstance(dataset, pd.DataFrame):
                gene_names = dataset.columns
                data = dataset.values

            elif any((isinstance(dataset, np.ndarray), isinstance(dataset, torch.Tensor))):
                try:
                    gene_names = kwargs.pop("gene_names")
                except KeyError:
                    gene_names = None

                data = dataset

            elif isinstance(dataset, sparse.csr_matrix):
                data = dataset.toarray()
                try:
                    gene_names = kwargs.pop("gene_names")
                except KeyError:
                    gene_names = None

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
        if isinstance(data, np.ndarray):
            data_out = np.zeros((data.shape[0], self.gene_names_len), dtype=np.int32)
        else:
            data_out = sp_sparse.lil_matrix(sp_sparse.eye(m=data.shape[0], n=self.gene_names_len, dtype=np.int32)) * 0
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
        return data_out

    @staticmethod
    def _toarray(data):
        return data.toarray()

    def _fill_index_map(self):
        if not self.index_map:
            with h5py.File(self.hdf5_filepath, "r") as h5_file:
                # Walk through all groups, extracting datasets
                for group_name, group in h5_file.items():
                    shape = group["X"].shape
                    self.index_map.extend([(group_name, i) for i in range(shape[0])])

    def _cache_processed_gene_names(self, ):
        if not self.gene_names_processed:
            self.gene_names_processed = dict()
            with h5py.File(self.hdf5_filepath, "r") as h5_file:
                # Walk through all groups, extracting datasets
                for group_name, group in h5_file.items():
                    gene_names = group["gene_names"]
                    mappable_genes_indices = np.isin(gene_names, self.gene_map.index)
                    mappable_genes = gene_names[mappable_genes_indices]
                    col_indices = self.gene_map[mappable_genes].values
                    col_indices.sort()
                    self.gene_names_processed[group_name] = (col_indices, mappable_genes_indices.flatten())
        return

    def concat_to_hdf5(self,
                       dataset_names,
                       dataset_classes,
                       dataset_args=None,
                       out_fname=None):
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
                    dataset_hdf5_g.create_dataset("local_means", data=local_means)
                    dataset_hdf5_g.create_dataset("local_vars", data=local_vars)
                    dataset_hdf5_g.create_dataset("batch_indices", data=batch_indices)
                    dataset_hdf5_g.create_dataset("labels", data=labels)

                    if hasattr(dataset, "cell_types"):
                        cell_types = dataset.cell_types
                        dataset_hdf5_g.create_dataset("cell_types",
                                                      data=cell_types.astype(np.dtype("S")), dtype=string_dt)
                    lock.release()
        print(f"conversion completed to file '{out_fname}.hdf5'")

    def concat_union_in_memory(self,
                               dataset_names,
                               dataset_classes,
                               dataset_args=None,
                               shared_batches=False
                               ):
        """
        Combines multiple unlabelled gene_datasets based on the union of gene names.
        Batch indices are generated in the same order as datasets are given.
        :param gene_datasets: a sequence of gene_datasets object
        :return: a GeneExpressionDataset instance of the concatenated datasets
        """
        if dataset_args is None:
            dataset_args = [None] * len(dataset_names)

        X = []
        local_means = []
        local_vars = []
        batch_indices = []
        labels = []

        n_batch_offset = 0

        cell_types_map = dict()
        cell_types_map_len = 0

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
                    dataset, _, _ = res
                    if any(("ENS" not in gn for gn in dataset.gene_names[0:100])):
                        continue
                lock.acquire()
                X.append(sp_sparse.csr_matrix(self.map_data(dataset)))
                local_means.append(dataset.local_means)
                local_vars.append(dataset.local_vars)
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
                lock.release()
        if not cell_types_map:
            cell_types_map = None
            labels = None
        else:
            cell_types_map = np.asarray((x[0] for x in sorted(cell_types_map.items(), key=lambda x: x[1])))
            labels = np.concatenate(labels)
        self.populate_from_data(X=sp_sparse.vstack(X),
                                batch_indices=np.concatenate(batch_indices),
                                labels=labels,
                                gene_names=self.gene_names,
                                cell_types=cell_types_map
                                )
        self.local_means = np.vstack(local_means)
        self.local_vars = np.vstack(local_vars)
        self.X_len = self.X.shape[0]
        return self

    _type_dispatch = staticmethod(_type_dispatch)
