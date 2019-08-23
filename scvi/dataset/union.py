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
import loompy
import time
from typing import Dict, Tuple, Sequence
from functools import singledispatch
import itertools

class_re_pattern = r"((?<=[.])[A-Za-z_0-9]+(?='>))|((?<=class ')\w+(?='>))"


class UnionDataset(GeneExpressionDataset):
    def __init__(self,
                 save_path,
                 low_memory=True,
                 ignore_batch_annotation=True,
                 gene_map_load_filename=None,
                 gene_map_save_filename=None,
                 data_load_filename=None,
                 data_save_filename=None
                 ):
        super().__init__()
        self.save_path = save_path

        self._len = None
        self.gene_map = None
        self.gene_names = []
        self.gene_names_len = 0

        self.gene_map_load_filename = gene_map_load_filename
        self.gene_map_save_filename = gene_map_save_filename
        self.data_load_filename = data_load_filename
        self.data_save_filename = data_save_filename

        self.dataset_to_genes_mapping_cached = None
        self.index_to_dataset_map = []
        self.dataset_to_index_map = dict()
        self.low_memory = low_memory

        self.ignore_batch_annotation = ignore_batch_annotation

        if gene_map_load_filename is not None:
            self.gene_map = pd.read_csv(
                os.path.join(self.save_path, gene_map_load_filename + ".csv"),
                header=0,
                index_col=0
            ).sort_index()
            self.gene_map.index = self.gene_map.index.astype(str).str.lower()
            self.gene_map = pd.Series(range(len(self.gene_map)), index=self.gene_map.index)
            self.gene_names = self.gene_map.index.values
            self.gene_names_len = len(self.gene_names)

        if data_load_filename is not None:
            if low_memory:
                self._set_attributes(data_load_filename)
                _, ext = os.path.splitext(data_load_filename)
                if ext == ".h5":
                    self._cache_gene_mapping()
            else:
                self._union_from_h5_to_memory(data_load_filename)

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        return idx

    def collate_fn_base(self,
                        attributes_and_types,
                        indices
                        ) -> Tuple[torch.Tensor]:
        return super().collate_fn_base(attributes_and_types, batch=indices)

    def collate_fn_base_h5(self,
                             attributes_and_types,
                             indices
                             ) -> Tuple[torch.Tensor]:
        indices = np.asarray(indices)
        indices.sort()
        self._cache_gene_mapping()

        batch = defaultdict(list)
        h5_acc_dict = defaultdict(list)
        for idx in indices:
            # sort the indices by the dataset they address for faster loading
            ds_specifier, index = self.index_to_dataset_map[idx]
            h5_acc_dict[ds_specifier].append(index)
        for ds_specifier, ds_indices in h5_acc_dict.items():
            for attr, dtype in attributes_and_types.items():
                if attr == "X":
                    elems = list(getattr(self, attr)[ds_indices])
                    elems = np.vstack(elems).astype(dtype)
                    col_indices, map_gene_ind = self.dataset_to_genes_mapping_cached[ds_specifier]
                    elems = self._map_data(elems,
                                           mappable_genes_indices=map_gene_ind,
                                           col_indices=col_indices
                                           )
                    batch[attr].append(elems)
                else:
                    elems = getattr(self, attr)[ds_indices]
                    batch[attr].append(np.asarray(elems).astype(dtype))

        batch_out = []
        for _, elems in batch.items():
            batch_out.append(torch.from_numpy(np.vstack(elems)))
        return tuple(batch_out)

    def collate_fn_base_loom(self,
                             attributes_and_types,
                             indices
                             ) -> Tuple[torch.Tensor]:
        indices = np.asarray(indices)
        indices.sort()

        batch = []

        for attr, dtype in attributes_and_types.items():
            elems = getattr(self, attr)[indices]
            batch.append(torch.from_numpy(elems.astype(dtype)))

        return tuple(batch)

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
        if self.gene_map_save_filename:
            self.gene_map.to_csv(
                os.path.join(self.save_path, self.gene_map_save_filename + ".csv"),
                header=False
            )
            pd.Series(list(filtered_classes.values()), index=list(filtered_classes.keys())).to_csv(
                os.path.join(self.save_path, self.gene_map_save_filename + "_used_datasets.csv"),
                header=False
            )

    def join_datasets(self,
                      data_source: str,
                      data_target: str,
                      **kwargs):

        if data_source not in ["memory", "h5", "loom", "self", "scvi"]:
            raise ValueError(f"Parameter 'data_source={data_source}' not supported.")
        if data_target not in ["memory", "h5", "loom"]:
            raise ValueError(f"Parameter 'data_target={data_target}' not supported.")

        eval(f"self._union_from_{data_source}_to_{data_target}(**kwargs)")

    def compute_library_size_batch(self):
        """Computes the library size per batch."""
        if not self.low_memory:
            self.local_means = np.zeros((self.nb_cells, 1))
            self.local_vars = np.zeros((self.nb_cells, 1))
            for i_batch in range(self.n_batches):
                idx_batch = np.squeeze(self.batch_indices == i_batch)
                self.local_means[idx_batch], \
                self.local_vars[idx_batch] = self._compute_library_size(self.X[idx_batch],
                                                                        batch_size=len(idx_batch))
            self.cell_attribute_names.update(["local_means", "local_vars"])

    def _compute_library_size(self, data, batch_size=None):
        sum_counts = []
        for cell_expression in tqdm(data, total=batch_size):
            sum_counts.append(cell_expression.sum())
        log_counts = np.log(sum_counts)
        m = np.mean(log_counts)
        v = np.var(log_counts)

        return np.array(m).astype(np.float32).reshape(-1, 1), \
               np.array(v).reshape(-1, 1).astype(np.float32)

    def change_memory_setting(self,
                              low_memory):
        self.low_memory = low_memory

    def set_data_filename(self, filename):
        self.data_load_filename = filename

    def _set_attributes(self, data_fname) -> UnionDataset:
        self.data_load_filename = data_fname
        filepath = os.path.join(self.save_path, data_fname)
        filename, ext = os.path.splitext(data_fname)
        if ext == ".h5":
            self.collate_fn_base = self.collate_fn_base_h5
            self._fill_index_map()
            self._cache_gene_mapping()
            # get the info for all shapes of the datasets
            self._len = len(self.index_to_dataset_map)
            self.batch_indices = MetaAttrLoaderh5("batch_indices",
                                                    h5_filepath=filepath,
                                                    attr_map=self.dataset_to_index_map)

            self.batch_indices = MetaAttrLoaderh5("batch_indices",
                                                    h5_filepath=filepath,
                                                    attr_map=self.dataset_to_index_map)

            self.labels = MetaAttrLoaderh5("labels",
                                             h5_filepath=filepath,
                                             attr_map=self.dataset_to_index_map)
            self.cell_types = self.labels.attr_values

            self.local_means = MetaAttrLoaderh5("local_means",
                                                  h5_filepath=filepath,
                                                  attr_map=self.dataset_to_index_map)
            self.local_vars = MetaAttrLoaderh5("local_varss",
                                                 h5_filepath=filepath,
                                                 attr_map=self.dataset_to_index_map)

            self.X = AttrLoaderh5("X",
                                    h5_filepath=filepath,
                                    attr_map=self.index_to_dataset_map)
            with h5py.File(filepath, "r") as h5_file:
                if not self.ignore_batch_annotation:
                    local_means = np.empty((0, 1))
                    local_vars = np.empty((0, 1))
                    for group_name, group in h5_file["Datasets"].items():
                        local_means = np.concatenate([local_means, group["local_means"][:]])
                        local_vars = np.concatenate([local_vars, group["local_vars"][:]])
                    self.local_means = local_means
                    self.local_vars = local_vars

                else:
                    self.local_means = np.repeat(h5_file["Metadata"]["local_mean_complete_dataset"][:].flatten(),
                                                 self._len).reshape(-1, 1).astype(np.float32)
                    self.local_vars = np.repeat(h5_file["Metadata"]["local_var_complete_dataset"][:].flatten(),
                                                self._len).reshape(-1, 1).astype(np.float32)

        elif ext == ".loom":
            self.collate_fn_base = self.collate_fn_base_loom
            with loompy.connect(filepath) as ds:
                gene_names = ds.ra["Gene"]
                if np.any(gene_names != self.gene_names):
                    raise ValueError("Chosen gene map and dataset genes are not equal.")
                self.X = AttrLoaderLoom(filepath)
                self._len = len(self.X)
                self.batch_indices = ds.ca["BatchID"]
                self.labels = ds.ca["ClusterID"]
                self.cell_types = ds.attrs["CellTypes"]
                if self.ignore_batch_annotation:
                    self.local_means = np.repeat(ds.attrs["LocalMeanCompleteDataset"].flatten(),
                                                 self._len).reshape(-1, 1).astype(np.float32)
                    self.local_vars = np.repeat(ds.attrs["LocalVarCompleteDataset"].flatten(),
                                                self._len).reshape(-1, 1).astype(np.float32)
                else:
                    self.local_means = ds.ca["LocalMeans"]
                    self.local_vars = ds.ca["LocalVars"]
                self.name = ds.attrs["DatasetName"]

        return self

    def _fill_index_map(self) -> None:
        if not self.index_to_dataset_map or not self.dataset_to_index_map:
            with h5py.File(self.data_load_filename, "r") as h5_file:
                # Walk through all groups, extracting datasets
                for group_name, group in h5_file.items():
                    shape = group["X"].shape
                    curr_index_len = len(self.index_to_dataset_map)
                    self.dataset_to_index_map[group_name] = [curr_index_len + i for i in range(shape[0])]
                    self.index_to_dataset_map.extend([(group_name, i) for i in range(shape[0])])
        return

    def _cache_gene_mapping(self, ) -> None:
        if not self.dataset_to_genes_mapping_cached:
            self.dataset_to_genes_mapping_cached = dict()
            with h5py.File(self.data_load_filename, "r") as h5_file:
                # Walk through all groups, extracting datasets
                for group_name, group in h5_file.items():
                    gene_names = np.char.lower(group["gene_names"][:].astype(str))
                    mappable_genes_indices = np.isin(gene_names, self.gene_map.index)
                    mappable_genes = gene_names[mappable_genes_indices]
                    col_indices = self.gene_map[mappable_genes].values
                    col_indices.sort()
                    self.dataset_to_genes_mapping_cached[group_name] = (col_indices, mappable_genes_indices.flatten())
        return

    def _map_data(self,
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

        return dataset, ds_class, dataset.name

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
            gns = dataset.gene_names
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

    def _write_dset_to_h5(self, out_filename, dataset, dataset_class, dataset_fname):
        string_dt = h5py.special_dtype(vlen=str)
        with h5py.File(os.path.join(self.save_path, out_filename), 'a') as h5file:
            data_group = h5file["Datasets"]
            dataset_class_str = re.search(class_re_pattern, str(dataset_class)).group()

            # grab the necessary data parts:
            # aside from the data itself (X), the gene_names, local means, local_vars, batch_indices and labels
            # there are no guaranteed attributes of each dataset. Thus for now these will be the ones we
            # work with
            X = dataset.X
            gene_names = np.char.lower(dataset.gene_names)
            local_means = dataset.local_means
            local_vars = dataset.local_vars
            batch_indices = dataset.batch_indices
            labels = dataset.labels

            print(f"Writing dataset {dataset_class_str, dataset_fname} to h5 file.")
            sys.stdout.flush()

            # Build the group for the dataset, under which the data is going to be stored
            # We will store the above mentioned data in the following scheme:
            # -- Datasets
            # ---- 1st Dataset CLASS and NAME
            # ------ X
            # ------ gene_names
            # ------ local_means
            # ------ local_vars
            # ------ batch_indices
            # ------ labels
            # ------ (cell_types)
            # ---- 2nd Dataset CLASS and NAME
            # ------ ...
            # -- Metadata
            # ---- metadata1
            # ---- metadata2
            # ---- ...
            dataset_h5_g = data_group.create_group(f"{dataset_class_str}_{dataset_fname}")

            if isinstance(X, (sp_sparse.csr_matrix, sp_sparse.csc_matrix)):
                dset = dataset_h5_g.create_dataset("X",
                                                     shape=(X.shape[0], len(dataset.gene_names)),
                                                     compression="lzf",
                                                     dtype=np.float32)
                nr_rows = 1000
                pbar = tqdm(range(0, len(dataset), nr_rows))
                pbar.set_description("Writing sparse matrix iteratively to file...")
                for start in pbar:
                    end = start + nr_rows
                    dset[slice(start, end), :] = np.vstack(
                        [X.getrow(i).toarray() for i in range(start, min(end, X.shape[0]))]
                    ).astype(np.float32)
            else:
                dataset_h5_g.create_dataset("X", data=X)
            dataset_h5_g.create_dataset("gene_names", data=gene_names.astype(np.dtype("S")), dtype=string_dt)
            dataset_h5_g.create_dataset("local_means", data=local_means)
            dataset_h5_g.create_dataset("local_vars", data=local_vars)
            dataset_h5_g.create_dataset("batch_indices", data=batch_indices)
            dataset_h5_g.create_dataset("labels", data=labels)

            if hasattr(dataset, "cell_types"):
                cell_types = dataset.cell_types
                dataset_h5_g.create_dataset("cell_types",
                                              data=cell_types.astype(np.dtype("S")), dtype=string_dt)

    def _write_dset_to_loom(self, dataset_ptr, dataset, dataset_class, dataset_fname):

        dataset_class_str = re.search(class_re_pattern, str(dataset_class)).group()
        dataset_ptr.attrs.DatasetName += f"_{dataset_class_str}"
        # grab the necessary data parts:
        # aside from the data itself (X), the gene_names, local means, local_vars, batch_indices and labels
        # there are no guaranteed attributes of each dataset.
        gene_names = np.char.lower(dataset.gene_names)
        X = dataset.X
        batch_indices = dataset.batch_indices
        labels = dataset.labels
        local_means = dataset.local_means
        local_vars = dataset.local_means

        if not all(dataset.cell_types == "undefined"):
            known_cts = [ct for ct in dataset_ptr.attrs.CellTypes]
            cts = dataset.cell_types
            labels = cts[labels]
            # append cell type only if unknown
            for ct in cts:
                if ct not in known_cts:
                    known_cts.append(ct)
            # remap from ["endothelial_cell", "B_cell", "B_cell", ...] to [3, 5, 3, ...]
            for cat_from, cat_to in zip(cts, [known_cts.index(ct) for ct in cts]):
                labels[labels == cat_from] = cat_to
            labels = labels.astype(np.uint16)
            dataset_ptr.attrs.CellTypes = known_cts
        if "BatchID" in dataset_ptr.col_attrs:
            max_existing_batch_idx = dataset_ptr.ca["BatchID"].max()
            batch_indices = batch_indices + max_existing_batch_idx

        if isinstance(X, sp_sparse.csc_matrix):
            X = X.tocsr()
        if isinstance(X, sp_sparse.csr_matrix):
            nr_rows = 5000

            mappable_genes_indices = np.isin(np.char.lower(gene_names), self.gene_map.index)
            mappable_genes = gene_names[mappable_genes_indices]
            col_indices = self.gene_map[mappable_genes].values

            pbar = tqdm(range(0, len(dataset), nr_rows))
            pbar.set_description("Writing sparse matrix iteratively to file...")
            for start in pbar:
                end = start + nr_rows
                X_batch = self._map_data(data=np.vstack(
                    [X.getrow(i).toarray() for i in range(start, min(end, X.shape[0]))]
                ),
                    col_indices=col_indices,
                    mappable_genes_indices=mappable_genes_indices) \
                    .astype(np.int32).transpose()
                selection = slice(start, min(end, X.shape[0]))
                dataset_ptr.add_columns(X_batch,
                                        col_attrs={"ClusterID": labels[selection],
                                                   "BatchID": batch_indices[selection],
                                                   "LocalMeans": local_means[selection],
                                                   "LocalVars": local_vars[selection]},
                                        row_attrs={"Gene": self.gene_names})
        else:
            dataset_ptr.add_columns(self._map_data(data=X, gene_names=gene_names).transpose(),
                                    col_attrs={"ClusterID": labels,
                                               "BatchID": batch_indices,
                                               "LocalMeans": local_means,
                                               "LocalVars": local_vars},
                                    row_attrs={"Gene": self.gene_names})

    def _union_from_memory_to_h5(self,
                                   out_filename,
                                   gene_datasets: List[GeneExpressionDataset],
                                   ):
        filename, ext = os.path.splitext(out_filename)
        if ext != ".h5":
            logger.warn(f"Chosen file type is 'hdf5', but provided ending is '{ext}' (expected '.h5').")
        with h5py.File(os.path.join(self.save_path, out_filename), 'w') as h5file:
            # just opening the file overwrite any existing content and enabling append mode for subfunction
            h5file.create_group("Datasets")
            h5file.create_group("Metadata")

        counts = []

        datasets_pbar = tqdm(gene_datasets)
        for dataset in datasets_pbar:
            datasets_pbar.set_description(f"Writing dataset '{type(dataset)} - {dataset.name}' to file")
            self._write_dset_to_h5(out_filename, dataset, type(dataset), dataset.name)

            counts.append(np.array(dataset.X.sum(axis=1)).flatten())
        log_counts = np.concatenate(np.log(counts))
        total_lm = np.mean(log_counts).reshape(-1, 1).astype(np.float32)
        total_lv = np.var(log_counts).reshape(-1, 1).astype(np.float32)
        with h5py.File(os.path.join(self.save_path, out_filename), 'a') as h5file:
            g = h5file["Metadata"]
            g.create_dataset("local_mean_complete_dataset", data=total_lm)
            g.create_dataset("local_var_complete_dataset", data=total_lv)

        self._set_attributes(out_filename)

    def _union_from_memory_to_loom(self,
                                   out_filename,
                                   gene_datasets: List[GeneExpressionDataset],
                                   ):
        filename, ext = os.path.splitext(out_filename)
        if ext != ".loom":
            logger.warn(f"Chosen file type is 'loom', but provided ending is '{ext}' (expected '.loom').")

        file = os.path.join(self.save_path, out_filename)

        dsout = loompy.new(file)
        dsout.attrs.CellTypes = ['undefined']
        dsout.attrs.DatasetName = ""

        counts = []

        datasets_pbar = tqdm(gene_datasets)
        for dataset in datasets_pbar:
            dataset_class_str = re.search(class_re_pattern, str(type(dataset))).group()
            datasets_pbar.set_description(f"Writing dataset '{dataset_class_str} - {dataset.name}' to file")

            self._write_dset_to_loom(dsout, dataset, type(dataset), dataset.name)

            counts.append(np.array(dataset.X.sum(axis=1)).flatten())
        log_counts = np.log(np.concatenate(counts))
        total_lm = np.mean(log_counts).reshape(-1, 1).astype(np.float32)
        total_lv = np.var(log_counts).reshape(-1, 1).astype(np.float32)
        dsout.attrs.LocalMeanCompleteDataset = total_lm
        dsout.attrs.LocalVarCompleteDataset = total_lv
        dsout.close()
        self._set_attributes(out_filename)

    def _union_from_scvi_to_h5(self,
                                 dataset_names,
                                 dataset_classes,
                                 dataset_args=None,
                                 out_filename=None):
        """
        Combines multiple unlabelled gene_datasets based on a mapping of gene names. Stores the final
        dataset onto a Hdf5 file with out_fname filename.
        :param dataset_names: List, list of names complementing the dataset_classes (needed for some classes)
        :param dataset_classes: List, list of class-initializers of scvi GeneExpression datasets
        :param dataset_args: List, list of further positional arguments for when loading the datasets
        :param out_filename: str, name of the file to which to write.
        :return: self (with instantiated dataloaders for data access)
        """

        if out_filename is None:
            out_filename = self.data_save_filename

        with h5py.File(os.path.join(self.save_path, out_filename + ".h5"), 'w') as _:
            # just opening the file overwrite any existing content and enabling append mode for subfunction
            pass

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

                lock.acquire()
                self._write_dset_to_h5(out_filename, dataset, dataset_class, dataset_fname)
                lock.release()
        print(f"conversion completed to file '{out_filename}.h5'")
        self._set_attributes(out_filename)
        return self

    def _union_from_memory_to_memory(self,
                                     gene_datasets: List[GeneExpressionDataset],
                                     ):
        """
        Combines multiple unlabelled gene_datasets based on a mapping of gene names. Loads the final
        dataset directly into memory.
        :param gene_datasets: List, the loaded data sets of (inherited) class GeneExpressionDataset to concatenate.
        :return: self (populated with data)
        """

        X = []
        local_means = []
        local_vars = []
        batch_indices = []
        labels = []
        cell_types = []
        pbar = tqdm(gene_datasets)
        pbar.set_description("Concatenating datasets")
        for gene_dataset in pbar:
            gene_names = np.char.lower(gene_dataset.gene_names)
            X.append(sp_sparse.csr_matrix(self._map_data(gene_dataset.X, gene_names)))
            local_means.append(gene_dataset.local_means)
            local_vars.append(gene_dataset.local_vars)
            batch_indices.append(gene_dataset.batch_indices)
            labels.append(gene_dataset.labels)
            cell_types.append(gene_dataset.cell_types)
            labels[-1] = cell_types[-1][labels[-1]]
        cell_types = np.sort(np.unique(np.concatenate(cell_types)))
        cell_types_map = pd.Series(range(len(cell_types)), index=cell_types)
        labels = np.concatenate(labels)

        for i, cell_type in np.ndenumerate(labels):
            labels[i] = cell_types_map[cell_type]

        X = sp_sparse.vstack(X)
        self._len = X.shape[0]
        self.populate_from_data(X=X,
                                batch_indices=np.concatenate(batch_indices),
                                labels=labels,
                                gene_names=self.gene_names,
                                cell_types=cell_types
                                )
        self.local_means = np.vstack(local_means)
        self.local_vars = np.vstack(local_vars)
        logger.info(f"Joined {len(gene_datasets)} datasets to one of shape {self._len} x {self.gene_names_len}.")

        return self

    def _union_from_self_to_h5(self, out_fname):
        self._union_from_memory_to_h5(out_filename=out_fname, gene_datasets=[self])

    def _union_from_self_to_loom(self, out_fname):
        self._union_from_memory_to_loom(out_filename=out_fname, gene_datasets=[self])

    def _union_from_scvi_to_memory(self,
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
        local_means = []
        local_vars = []
        labels = []

        cell_types_map = dict()

        if dataset_args is None:
            dataset_args = [None] * len(dataset_names)

        cell_types_map_len = 0
        n_batch_offset = 0

        _LOCK = Lock()

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

                ###################
                _LOCK.acquire()
                ###################

                gene_names = dataset.gene_names
                X.append(sp_sparse.csr_matrix(self._map_data(dataset.X, gene_names)))
                local_means.append(dataset.local_means)
                local_vars.append(dataset.local_vars)
                bis = dataset.batch_indices

                if bis.sum() == 0:
                    bis = bis + n_batch_offset
                n_batch_offset += (dataset.n_batches if not shared_batches else 0)
                batch_indices.append(bis)

                for cell_type in dataset.cell_types:
                    if cell_type not in cell_types_map:
                        cell_types_map[cell_type] = cell_types_map_len
                        cell_types_map_len += 1
                ls = np.array([cell_types_map[cell_type] for cell_type in dataset.cell_types], dtype=np.int16)

                labels.append(ls)

                ###################
                _LOCK.release()
                ###################

        if not cell_types_map:
            cell_types = None
            labels = None
        else:
            cell_types = np.unique(np.asarray((x[0] for x in sorted(cell_types_map.items(), key=lambda x: x[1]))))
            labels = np.concatenate(labels)

        for i, label in enumerate(labels):
            labels[i] = cell_types_map[cell_types[i]]
        X = sp_sparse.vstack(X)
        self._len = self.X.shape[0]
        self.populate_from_data(X=X,
                                batch_indices=np.concatenate(batch_indices),
                                labels=labels,
                                gene_names=self.gene_names,
                                cell_types=cell_types
                                )
        self.local_means = np.vstack(local_means)
        self.local_vars = np.vstack(local_vars)
        return self

    def _union_from_h5_to_memory(self,
                                   in_filename=None):
        if in_filename is None:
            in_filename = self.data_load_filename

        X = []
        batch_indices = []
        local_means = []
        local_vars = []
        labels = []
        cell_types = []

        with h5py.File(os.path.join(self.save_path, in_filename + ".h5"), 'r') as h5file:
            dataset_group = h5file["Datasets"]
            for group_name, group in dataset_group.items():
                X.append(sp_sparse.csr_matrix(self._map_data(group["X"][:], group["gene_names"][:])))
                local_means.append(group["local_means"][:])
                local_vars.append(group["local_vars"][:])
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
        X = sp_sparse.vstack(X)
        self.populate_from_data(X=X,
                                batch_indices=np.concatenate(batch_indices),
                                labels=labels,
                                gene_names=self.gene_names,
                                cell_types=cell_types,
                                local_means=local_means,
                                local_vars=local_vars
                                )
        return

    def _union_from_loom_to_memory(self,
                                   in_filename=None,
                                   gene_names_attribute_name="Gene",
                                   batch_indices_attribute_name="BatchID",
                                   local_means_attribute_name="LocalMeans",
                                   local_vars_attribute_name="LocalVars",
                                   labels_attribute_name="ClusterID",
                                   cell_types_attribute_name="CellTypes",
                                   total_local_mean_attribute_name="LocalMeanCompleteDataset",
                                   total_local_var_attribute_name="LocalVarCompleteDataset",
                                   dataset_name_attribute_name="DatasetName"):
        if in_filename is None:
            in_filename = self.data_load_filename

        X = None
        batch_indices = None
        labels = None
        cell_types = None
        gene_names = None
        local_means = None
        local_vars = None
        name = None

        with loompy.connect(os.path.join(self.save_path, in_filename)) as ds:

            for row_attribute_name in ds.ra:
                if row_attribute_name == gene_names_attribute_name:
                    gene_names = ds.ra[gene_names_attribute_name]
                else:
                    gene_attributes_dict = (
                        gene_attributes_dict if gene_attributes_dict is not None else {}
                    )
                    gene_attributes_dict[row_attribute_name] = ds.ra[row_attribute_name]

            for column_attribute_name in ds.ca:
                if column_attribute_name == batch_indices_attribute_name:
                    batch_indices = ds.ca[batch_indices_attribute_name][:]
                elif column_attribute_name == labels_attribute_name:
                    labels = ds.ca[labels_attribute_name][:]
                elif column_attribute_name == local_means_attribute_name:
                    if not self.ignore_batch_annotation:
                        local_means = ds.ca[local_means_attribute_name]
                elif column_attribute_name == local_vars_attribute_name:
                    if not self.ignore_batch_annotation:
                        local_vars = ds.ca[local_vars_attribute_name]
                else:
                    cell_attributes_dict = (
                        cell_attributes_dict if cell_attributes_dict is not None else {}
                    )
                    cell_attributes_dict[column_attribute_name] = ds.ca[column_attribute_name][:]

            global_attributes_dict = None
            for global_attribute_name in ds.attrs:
                if global_attribute_name == cell_types_attribute_name:
                    cell_types = ds.attrs[cell_types_attribute_name]
                elif global_attribute_name == total_local_mean_attribute_name:
                    if self.ignore_batch_annotation:
                        local_means = ds.attrs[total_local_mean_attribute_name]
                elif global_attribute_name == total_local_var_attribute_name:
                    if self.ignore_batch_annotation:
                        local_vars = ds.attrs[total_local_var_attribute_name]
                elif global_attribute_name == dataset_name_attribute_name:
                    name = ds.attrs[dataset_name_attribute_name]
                else:
                    global_attributes_dict = (
                        global_attributes_dict if global_attributes_dict is not None else {}
                    )
                    global_attributes_dict[global_attribute_name] = ds.attrs[global_attribute_name]

            if global_attributes_dict is not None:
                self.global_attributes_dict = global_attributes_dict
            X = ds[:, :].T

        self.populate_from_data(X=X,
                                gene_names=gene_names,
                                batch_indices=batch_indices,
                                cell_types=cell_types,
                                labels=labels,
                                local_means=local_means,
                                local_vars=local_vars,
                                name=name)

        return

    def populate_from_data(
        self,
        X: Union[np.ndarray, sp_sparse.csr_matrix],
        Ys: List[CellMeasurement] = None,
        batch_indices: Union[List[int], np.ndarray, sp_sparse.csr_matrix] = None,
        labels: Union[List[int], np.ndarray, sp_sparse.csr_matrix] = None,
        gene_names: Union[List[str], np.ndarray] = None,
        cell_types: Union[List[str], np.ndarray] = None,
        cell_attributes_dict: Dict[str, Union[List, np.ndarray]] = None,
        gene_attributes_dict: Dict[str, Union[List, np.ndarray]] = None,
        remap_attributes: bool = True,
        **kwargs
    ):
        self._len = X.shape[0]
        super().populate_from_data(X=X,
                                   Ys=Ys,
                                   batch_indices=np.concatenate(batch_indices),
                                   labels=labels,
                                   gene_names=self.gene_names,
                                   cell_types=cell_types,
                                   cell_attributes_dict=cell_attributes_dict,
                                   gene_attributes_dict=gene_attributes_dict,
                                   remap_attributes=remap_attributes
                                   )
        for kwarg, value in kwargs.items():
            setattr(self, kwarg, value)

    @property
    def nb_genes(self) -> int:
        return self.gene_names_len

    @nb_genes.setter
    def nb_genes(self, nb_genes: int):
        self.gene_names_len = nb_genes

    @property
    def batch_indices(self) -> np.ndarray:
        return self._batch_indices

    @batch_indices.setter
    def batch_indices(self, batch_indices):
        """Sets batch indices and the number of batches."""
        if not self.ignore_batch_annotation:
            batch_indices = np.asarray(batch_indices, dtype=np.uint16).reshape(-1, 1)
            self.n_batches = len(np.unique(batch_indices))
            self._batch_indices = batch_indices
        else:
            logger.info("Union dataset is set to ignore batch annotation.")
            self._batch_indices = np.zeros((len(self), 1), dtype=np.int64)
            self.n_batches = len(np.unique(batch_indices))

    @property
    def labels(self) -> np.ndarray:
        return self._labels

    @labels.setter
    def labels(self, labels: Union[List[int], np.ndarray, MetaAttrLoaderh5]):
        """Sets labels and the number of labels"""
        if not self.low_memory:
            labels = np.asarray(labels, dtype=np.uint16).reshape(-1, 1)
            self.n_labels = len(np.unique(labels))
            self._labels = labels
        else:
            self._labels = labels
            self.n_labels = len(self.labels)

    @property
    def X(self):
        return self._X

    @X.setter
    def X(self, X: Union[np.ndarray, sp_sparse.csr_matrix]):
        """Sets the data attribute ``X`` and (re)computes the library size."""
        n_dim = len(X.shape)
        if n_dim != 2:
            raise ValueError(
                "Gene expression data should be 2-dimensional not {}-dimensional.".format(
                    n_dim
                )
            )
        self._X = X


class AttrLoaderh5:
    def __init__(self,
                 attr,
                 h5_filepath=None,
                 attr_map=None):
        self.attr = attr
        self.h5_filepath = h5_filepath
        self.attr_map = None
        self.set_attr_map(attr_map)

    @property
    def shape(self):
        return len(self), None

    def __len__(self):
        return len(self.attr_map)

    def __getitem__(self, idx: Union[List, np.ndarray]):
        if np.array(idx).dtype == bool:
            idx = np.arange(len(self.attr_map))[idx]
        generator = (self.attr_map[i] for i in np.atleast_1d(idx))
        with h5py.File(self.h5_filepath, "r") as h5_file:
            for ds_specifier, index in generator:
                group = h5_file[ds_specifier]
                yield group[self.attr][index]

    def set_attr_map(self, attr_map):
        self.attr_map = attr_map


class MetaAttrLoaderh5(AttrLoaderh5):
    def __init__(self,
                 attr,
                 *args, **kwargs):
        super().__init__(attr, *args, **kwargs)
        nr_data_entries = 0
        with h5py.File(self.h5_filepath, "r") as h5file:
            for group_name, group in h5file["Datasets"].items():
                nr_data_entries += group["X"].shape[0]
        self.attr_access = np.zeros(nr_data_entries, dtype=np.int64)
        self.attr_values = None

        if attr == "batch_indices":
            with h5py.File(self.h5_filepath, "r") as h5file:
                dataset_group = h5file["Datasets"]
                curr_offset = 0
                for group_name, group in dataset_group.items():
                    bis = group[attr][:].flatten()
                    curr_offset += bis.max()
                    this_dset_indices = self.attr_map[group_name]
                    for k in range(len(bis)):
                        self.attr_access[this_dset_indices[k]] = bis[k] + curr_offset
            self.attr_values = np.unique(self.attr_access).astype(np.int64)

        elif attr in ["local_means", "local_vars"]:
            with h5py.File(self.h5_filepath, "r") as h5file:
                for group_name, group in h5file["Datasets"].items():
                    lms_or_lvs = group[attr][:]
                    this_dset_indices = np.array(self.attr_map[group_name])
                    self.attr_access[this_dset_indices] = lms_or_lvs.flatten()
            self.attr_values = np.unique(self.attr_access)

        elif attr == 'labels':
            known_cts = ["undefined"]  # known cell types
            with h5py.File(self.h5_filepath, "r") as h5file:
                for group_name, group in h5file["Datasets"].items():
                    labels = group[attr][:]
                    if labels.sum() == 0:
                        continue
                    cts = group["cell_types"][:]
                    labels = cts[labels]
                    # append cell type only if unknown
                    for ct in cts:
                        if ct not in known_cts:
                            known_cts.append(ct)
                    # remap from ["endothelial_cell", "B_cell", "B_cell", ...] to [3, 5, 3, ...]
                    for cat_from, cat_to in zip(cts, [known_cts.index(ct) for ct in cts]):
                        labels[labels == cat_from] = cat_to

                    this_dset_indices = np.array(self.attr_map[group_name])
                    self.attr_access[this_dset_indices] = labels.flatten()
            self.attr_values = np.array(known_cts)

    def __getitem__(self, idx):
        return self.attr_access[idx].reshape(-1, 1)

    def __len__(self):
        return len(self.attr_access)

    def __eq__(self, other):
        return self.attr_access == other


class AttrLoaderLoom:
    def __init__(self,
                 loom_filepath=None, ):
        self.loom_filepath = loom_filepath
        with loompy.connect(self.loom_filepath) as loom_file:
            self.shape = loom_file.shape
            self.dtype = loom_file[0, 0].dtype

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, idx: Union[List, np.ndarray]):
        with loompy.connect(self.loom_filepath) as loom_file:
            return loom_file[:, idx].transpose()
