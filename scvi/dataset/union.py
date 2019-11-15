from __future__ import annotations
import warnings
from scvi.dataset.dataset import GeneExpressionDataset, logger, remap_categories, CellMeasurement

import numpy as np
import pandas as pd
import scipy.sparse as sp_sparse
import os

import torch
from collections import defaultdict

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count, Lock

import re

from tqdm.auto import tqdm
import h5py
import loompy

from inspect import getfullargspec

from typing import Dict, Tuple, Callable, Union, Iterable, List

# this pattern will be used throughout the code to identify simply the scvi dataloader class
# without the need for the scvi.dataloader prefix when converted to a str.
# A bit unstable, especially if the str __rep__ of the dataloaders were to change, but that's a future
# github issue waiting to be solved then.
class_regex_pattern = r"((?<=[.])[A-Za-z_0-9]+(?='>))|((?<=class ')\w+(?='>))"


class UnionDataset(GeneExpressionDataset):
    """
    The UnionDataset class aims to provide a fully scVI compatible dataset concatenation API with large data support.

    Its 3 main features are:
        - Concatenating scVI datasets, preserving cell type labels, batch indices (if so wished), local means and vars,
          and mapping the datasets onto a common gene map.
        - Building of a common gene map either by loading it from a csv file of column structure
          (Genes, PositionalIndex) or by building it from: datasets to load, hdf5 datasets file, loom file.
        - Supporting out of memory data loading for hdf5 and loom files too big to be loaded into memory. With such it
          is possible to train on datasets worth multiple hundred gigabytes, albeit the speed of which doesn't convince
          (Private dataloading benchmark: loom ~400 times slower, hdf5 ~800 times slower than memory).
    """

    def __init__(
        self,
        save_path: str,
        low_memory: bool = True,
        ignore_batch_annotation: bool = True,
        gene_map_load_filename: str = None,
        gene_map_save_filename: str = None,
        data_load_filename: str = None,
        data_save_filename: str = None
    ):
        """
        Setting the most important features of the class on init. These settings can be overwritten after instantiation.
        :param save_path: str, the path, in which all of the data to load and save is stored and will be stored.
        :param low_memory: bool, if true the class will load metaloader for the data attribute ``X`` that load data
        on demand out of memory.
        :param ignore_batch_annotation: bool (optional), if true, all batch indices are reduced to 0.
        :param gene_map_load_filename: str (optional), the file, from which to load the gene map
        :param gene_map_save_filename: str (optional), the file, to which a potentially later built gene map would be
        saved.
        :param data_load_filename: str (optional), the file, from which data should be loaded (e.g. h5 file).
        :param data_save_filename: str (optional), the file, to which concatenated data should be saved. Can be easily
        changed later, on method call.
        """
        super().__init__()
        self.save_path = save_path

        self.gene_map = None
        self.gene_names = []
        self.gene_names_len = 0

        self.gene_map_load_filename = gene_map_load_filename
        self.gene_map_save_filename = gene_map_save_filename
        self.data_load_filename = data_load_filename
        self.data_save_filename = data_save_filename

        self.dataset_to_genemap_cache = None
        self.index_to_dataset_map = []
        self.dataset_to_index_map = dict()
        self.low_memory = low_memory

        self.ignore_batch_annotation = ignore_batch_annotation

        self.load_gene_map()

        if data_load_filename is not None:
            _, ext = os.path.splitext(data_load_filename)
            if low_memory:
                self._set_attributes(data_load_filename)
                if ext == ".h5":
                    self._cache_genemap()
            else:
                if ext == ".h5":
                    self._union_from_hdf5_to_memory(in_filename=data_load_filename)
                elif ext == ".loom":
                    self._union_from_loom_to_memory(in_filename=data_load_filename, as_sparse=True)

    #############################
    #                           #
    #     Public Interface      #
    #                           #
    #############################

    def build_genemap(
        self,
        data_source: str,
        force_build: bool = False,
        **kwargs
    ):
        """
        Concatenate datasets the way determined in ``data_source`` and ``data_target``. Any combination of the elements
        mentioned in ``data_source`` and ``data_target`` below are possible to be used.
        Kwargs needs to fit the combination precisely.

        For ``data_source``:
            - 'memory' will use datasets already loaded into memory.
              ``kwargs``:
                - ``gene_datasets``: list, a collection of scvi datasets, whose gene names are being included.
            - 'hdf5' will load gene names from the datasets of an hdf5 file.
              `kwargs``:
                - ``in_filename``: str, the hdf5 file containing the datasets to use.
            - 'loom' will load the gene names from a dataset in a loom file.
              `kwargs``:
                - ``in_filename``: str, the loom file containing the datasets to use.
            - 'scvi' will load datasets using the scvi dataloaders and then take their gene names.
              ``kwargs`` are:
                - ``dataset_classes``: list
                - ``dataset_names``: list, strings of the names of the datasets.
                - ``dataset_args``: list, list of lists of further keyword arguments to provide for each dataloader.
                - ``multiprocess``: bool, load the datasets in parallel or serial.
            - 'self' will simply map the gene names of the currently loaded dataset.

        :param data_source: str, one of "memory", "hdf5", "loom", "self", or "scvi".
        :param force_build: bool, if true, the gene map will be built regardless of an existing gene map load filename.
        :param kwargs: the arguments for the chosen combination method.
        :return: self
        """
        if self.gene_map_load_filename is not None:
            if not force_build:
                logger.log("Gene map filename found. Loading it instead.")
                self.load_gene_map()
                return
            else:
                logger.log("Gene map filename found, but 'force_build=True'. Continuing to build gene_map.")

        if data_source not in ["memory", "hdf5", "loom", "self", "scvi"]:
            raise ValueError(f"Parameter 'data_source={data_source}' not supported.")

        gene_map = eval(f"self._build_genemap_from_{data_source}(**kwargs)")
        self.gene_map = pd.Series(data=list(gene_map.values()), index=list(gene_map.keys()))
        self.gene_names = self.gene_map.index.values
        self.gene_names_len = len(self.gene_map)
        if self.gene_map_save_filename:
            self.gene_map.to_csv(
                os.path.join(self.save_path, self.gene_map_save_filename + ".csv"),
                header=False
            )
        return self

    def join_datasets(
        self,
        data_source: str,
        data_target: str,
        **kwargs
    ):
        """
        Concatenate datasets the way determined in ``data_source`` and ``data_target``. Any combination of the elements
        mentioned in ``data_source`` and ``data_target`` below are possible to be used.
        Kwargs needs to fit the combination precisely.

        For ``data_source``:
            - 'memory' will concatenate datasets already loaded into memory.
            - 'hdf5' will load datasets from a hdf5 file.
            - 'loom' will load the dataset from a loom file.
            - 'scvi' will load datasets using the scvi dataloaders.
            - 'self' will simply map the currently loaded dataset onto the gene map.
        For ``data_target``:
            - 'memory' will store the concatenated dataset into memory.
            - 'hdf5' will save the concatenated dataset into an hdf5 file.
            - 'loom' will save the concatenated dataset into a loom file.

        The ``kwargs`` for the various combinations are:
            -- ``memory`` to ``memory``:
                - ``gene_datasets``: list, a collection of scvi datasets to concatenate.
            -- ``memory`` to ``hdf5``:
                - ``gene_datasets``: list, a collection of scvi datasets to concatenate.
                - ``out_filename``: str, the filename of the hdf5 file to load. Warns if file extension is not '.h5'.
            -- ``memory`` to ``loom``:
                - ``gene_datasets``: list, a collection of scvi datasets to concatenate.
                - ``out_filename``: str, the filename of the loom file to load. Wanrs if file extension is not '.loom'.

            -- ``hdf5`` to ``memory``:
                - ``in_filename``: str, the filename of the hdf5 file to load. File extension should be '.h5'.
            -- ``hdf5`` to ``loom``:
                - ``in_filename``: str, the filename of the hdf5 file to load. File extension should be '.h5'.
                - ``out_filename``: str, the filename of the outgoing loom file. Warns if file extension is not '.loom'.

            -- ``loom`` to ``memory``:
                - ``in_filename``: str, the filename of the loom file to load. File extension should be '.loom'.
                - ``as_sparse``: bool, if true, will load the data into sparse matrix (default).
            -- ``loom`` to ``hdf5``:
                - ``in_filename``: str, the filename of the loom file to load. File extension should be '.loom'.
                - ``out_filename``: str, the filename of the outgoing hdf5 file. Warns if file extension is not '.h5'.

            -- ``scvi`` to ``memory``:
                - ``dataset_classes``: List, list of class-initializers of scvi GeneExpression datasets
                - ``dataset_names``: List, list of names complementing the dataset_classes (needed for some classes)
                - ``dataset_args``: List, list of further positional arguments for when loading the datasets
            -- ``scvi`` to ``hdf5``:
                - ``dataset_classes``: List, list of class-initializers of scvi GeneExpression datasets
                - ``dataset_names``: List, list of names complementing the dataset_classes (needed for some classes)
                - ``dataset_args``: List, list of further positional arguments for when loading the datasets
                - ``out_filename``: str, the filename of the outgoing hdf5 file. Warns if file extension is not '.h5'.
            -- ``scvi`` to ``loom``:
                - ``dataset_classes``: List, list of class-initializers of scvi GeneExpression datasets
                - ``dataset_names``: List, list of names complementing the dataset_classes (needed for some classes)
                - ``dataset_args``: List, list of further positional arguments for when loading the datasets
                - ``out_filename``: str, the filename of the outgoing loom file. Warns if file extension is not '.loom'.
            -- ``self`` to ``hdf5`` or ``loom``:
                - ``out_filename``: str, the filename of the outgoing loom or hdf5 file. Warnings for file extension.

        :param data_source: str, one of "memory", "hdf5", "loom", "self", or "scvi".
        :param data_target: str, one of "memory", "hdf5", or "loom"
        :param kwargs: the arguments for the chosen combination method.
        :return: self
        """
        if data_source not in ["memory", "hdf5", "loom", "self", "scvi"]:
            raise ValueError(f"Parameter 'data_source={data_source}' not supported.")
        if data_target not in ["memory", "hdf5", "loom"]:
            raise ValueError(f"Parameter 'data_target={data_target}' not supported.")

        eval(f"self._union_from_{data_source}_to_{data_target}(**kwargs)")
        return self

    def map_data(
        self,
        data: Union[np.ndarray, sp_sparse.csr_matrix],
        gene_names: np.ndarray = None,
        mappable_genes_indices: np.ndarray = None,
        col_indices: np.ndarray = None
    ) -> Union[np.ndarray, sp_sparse.lil_matrix]:
        """
        Maps single cell data gene wise onto a predefined gene map.
        Can take numpy arrays or scipy sparse matrices (assumes sparse matrix if not numpy).

        :param data: ndarray (#cells, #genes) or scipy sparse matrix, the data to map
        :param gene_names: ndarray (#genes,), gene codes to use for the mapping
        :param mappable_genes_indices: ndarray (optional), the column indices of the input array, which slices the data.
        A pre computed list of indices of the gene codes that can be found in the mapping (saves computational time if
        the same indices are mapped repeatedly and can therefore be reused).
        :param col_indices: ndarray (optional), the column indices of the output array, to which the sliced data
        is being mapped.
        :return: ndarray (#cells, #genes in mapping), the sliced data in the correct format of the gene map.
        """
        # check for already provided source data location indices
        if mappable_genes_indices is not None:
            mappable_genes_indices = mappable_genes_indices.flatten()
        else:
            mappable_genes_indices = np.isin(np.char.upper(gene_names), self.gene_map.index).flatten()
        # check for already provided target data location indices
        if col_indices is not None:
            col_indices = col_indices
        else:
            mappable_genes = gene_names[mappable_genes_indices]
            col_indices = self.gene_map[mappable_genes].values
        # actual data mapping now
        if isinstance(data, np.ndarray):
            data_out = np.zeros((data.shape[0], self.gene_names_len), dtype=data.dtype)
            try:
                data_out[:, col_indices] = data[:, mappable_genes_indices]
            except Exception as e:
                p = 3
        else:
            data_out = sp_sparse.csr_matrix(([], ([], [])), shape=(0, self.gene_names_len), dtype=data.dtype)
            nr_rows = 5000
            for i in range(0, data.shape[0], nr_rows):
                data_mapped = data[i:i + nr_rows, mappable_genes_indices].toarray()
                temp_mapped = np.zeros((data_mapped.shape[0], self.gene_names_len), dtype=data.dtype)
                temp_mapped[:, col_indices] = data_mapped
                data_out = sp_sparse.vstack([data_out, sp_sparse.csr_matrix(temp_mapped)])

        return data_out

    def set_gene_map_load_filename(
        self,
        filename: str = None
    ) -> UnionDataset:
        self.gene_map_load_filename = filename
        return self

    def load_gene_map(self):
        """
        Load the gene map from the file given in gene_map_load_filename.
        """
        if self.gene_map_load_filename is not None:
            self.gene_map = pd.read_csv(
                os.path.join(self.save_path, self.gene_map_load_filename + ".csv"),
                header=0,
                index_col=0
            ).sort_index()
            index = self.gene_map.index.astype(str).str.upper()
            self.gene_map = pd.Series(range(len(self.gene_map)), index=index)
            self.gene_names = self.gene_map.index.values
            self.gene_names_len = len(self.gene_names)
        return self

    def set_gene_map_save_filename(
        self,
        filename: bool
    ):
        self.gene_map_save_filename = filename
        return self

    def set_memory_setting(
        self,
        low_memory: bool
    ) -> UnionDataset:
        """
        Set the memory setting of the union object. As a side effect it also sets the appropriate collate method to the
        one of the datafile in question or to the standard (when the class handles data in memory).
        """
        if low_memory:
            self.low_memory = True
            filename, ext = os.path.splitext(self.data_load_filename)
            if ext == ".h5":
                self.collate_fn_base = self._collate_fn_base_h5
            elif ext == ".loom":
                self.collate_fn_base = self._collate_fn_base_loom
        else:
            self.low_memory = False
            self.collate_fn_base = super(UnionDataset, self).collate_fn_base
        return self

    def set_data_load_filename(
        self,
        filename: str
    ) -> UnionDataset:
        self.data_load_filename = filename
        return self

    def set_data_save_filename(
        self,
        filename: str
    ) -> UnionDataset:
        self.data_save_filename = filename
        return self

    def set_ignore_batch_annotation(
        self,
        ignore_batch_annotation: bool
    ):
        self.ignore_batch_annotation = ignore_batch_annotation
        return self

    #############################
    #                           #
    #       Internal Logic      #
    #                           #
    #############################

    def _collate_fn_base_h5(
        self,
        attributes_and_types: Dict,
        indices: Iterable
    ) -> Tuple[torch.Tensor]:
        """
        Collate method specialization for loading the data from an hdf5 file.
        Unlike for the loom collate method, the indices need to be first collected by associated dataset, so that h5py
        can load all the chosen indices from each dataset at once, instead of lazily iterating over every index and
        searching for the respective dataset anew every time.
        """
        indices = np.asarray(indices)
        indices.sort()
        self._cache_genemap()

        batch = defaultdict(list)
        for attr, dtype in attributes_and_types.items():
            elems = getattr(self, attr)[indices].astype(dtype)
            batch[attr].append(np.asarray(elems).astype(dtype))

        batch_out = []
        for _, elems in batch.items():
            batch_out.append(torch.from_numpy(np.vstack(elems)))
        return tuple(batch_out)

    def _collate_fn_base_loom(
        self,
        attributes_and_types,
        indices
    ) -> Tuple[torch.Tensor]:
        """
        Collate function specialization for loading the data from a loom file.
        """
        indices = np.asarray(indices)
        indices.sort()

        batch = []

        for attr, dtype in attributes_and_types.items():
            elems = getattr(self, attr)[indices]
            batch.append(torch.from_numpy(elems.astype(dtype)))

        return tuple(batch)

    def _set_attributes(
        self,
        data_filename: str = None
    ):
        """
        Set the attributes correctly after having concatenated datasets. This sets the data attribute ``X`` depending
        on the datafile extension, i.e. either of the metaloaders for loom or hdf5, if needed.
        """
        if data_filename is None:
            data_filename = self.data_load_filename
        else:
            self.data_load_filename = data_filename
        filepath = os.path.join(self.save_path, data_filename)
        filename, ext = os.path.splitext(data_filename)
        self.set_memory_setting(self.low_memory)
        if ext == ".h5":
            self._fill_index_map()
            self._cache_genemap()
            # get the info for all shapes of the datasets
            self.batch_indices, n_batches = self._load_batch_indices_from_hdf5(
                h5_filepath=filepath,
                attr_map=self.dataset_to_index_map
            )

            self.labels, self.cell_types = self._load_labels_from_hdf5(
                h5_filepath=filepath,
                attr_map=self.dataset_to_index_map
            )

            self.local_means, self.local_vars = self._load_local_means_vars_from_hdf5(
                h5_filepath=filepath,
                attr_map=self.dataset_to_index_map)

            self.X = DataLoaderh5(
                "X",
                parent=self,
                h5_filepath=filepath,
                attr_map=self.index_to_dataset_map
            )

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
                                                 self.nb_cells).reshape(-1, 1).astype(np.float32)
                    self.local_vars = np.repeat(h5_file["Metadata"]["local_var_complete_dataset"][:].flatten(),
                                                self.nb_cells).reshape(-1, 1).astype(np.float32)

        elif ext == ".loom":
            with loompy.connect(filepath) as ds:
                gene_names = ds.ra["Gene"]
                if np.any(gene_names != self.gene_names):
                    raise ValueError("Chosen gene map and dataset genes are not equal.")
                self.X = DataLoaderLoom(filepath)
                self.batch_indices = ds.ca["BatchID"]
                self.labels = ds.ca["ClusterID"]
                self.cell_types = ds.attrs["CellTypes"]
                if (self.labels > 0).all():
                    self.labels -= 1
                    self.cell_types = self.cell_types[self.cell_types != "undefined"]
                if self.ignore_batch_annotation:
                    self.local_means = np.repeat(ds.attrs["LocalMeanCompleteDataset"].flatten(),
                                                 self.nb_cells).reshape(-1, 1).astype(np.float32)
                    self.local_vars = np.repeat(ds.attrs["LocalVarCompleteDataset"].flatten(),
                                                self.nb_cells).reshape(-1, 1).astype(np.float32)
                else:
                    self.local_means = ds.ca["LocalMeans"]
                    self.local_vars = ds.ca["LocalVars"]
                self.name = ds.attrs["DatasetName"]

    def _fill_index_map(self) -> None:
        """
        If not already existing, create a mapping of each dataset to an associated index and its inverse.

        This is needed to speed up later access. Its necessity is given by the fact that one wants a simple index
        structure when accessing via indices, however an hdf5 file with multiple datasets would always first need to
        know the dataset and then the dataset specific index to access any data.

        In essence, we need a mapping of the kind
            (Dataset_name: 10x_mouse_10k,  index: 216)   ->  (index: 216)
            (Dataset_name: 10x_mouse_10k,  index: 429)   ->  (index: 429)
            (Dataset_name: smartseq_m_3k,  index: 216)   ->  (index: 10216)
            (Dataset_name: smartseq_m_3k,  index: 1659)  ->  (index: 11659)

        and its inverse.
        """
        if not self.index_to_dataset_map or not self.dataset_to_index_map:
            with h5py.File(os.path.join(self.save_path, self.data_load_filename), "r") as h5_file:
                # Walk through all groups, extracting datasets
                for group_name, group in h5_file["Datasets"].items():
                    shape = group["X"].shape
                    curr_index_len = len(self.index_to_dataset_map)
                    self.dataset_to_index_map[group_name] = [curr_index_len + i for i in range(shape[0])]
                    self.index_to_dataset_map.extend([(group_name, i) for i in range(shape[0])])

    def _cache_genemap(
        self,
        force_redo: bool = False
    ):
        """
        Compute the mapping of the gene names of each dataset inside the hdf5 file. Useful for faster data mapping when
        loading from file later in training.
        :param force_redo: bool, if true the cache is recomputed (e.g. for genemap) change.
        """
        ext = None
        if self.data_load_filename is not None:
            _, ext = os.path.splitext(self.data_load_filename)
        conditions = \
            self.low_memory \
            and ext in [".h5", ".loom"] \
            and (
                force_redo
                or not self.dataset_to_genemap_cache
            )
        if conditions:
            self.dataset_to_genemap_cache = dict()
            with h5py.File(os.path.join(self.save_path, self.data_load_filename), "r") as h5_file:
                # Walk through all groups, extracting datasets
                for group_name, group in h5_file["Datasets"].items():
                    gene_names = np.char.upper(group["gene_names"][:].astype(str))
                    mappable_genes_indices = np.isin(gene_names, self.gene_map.index)
                    mappable_genes = gene_names[mappable_genes_indices]
                    col_indices = self.gene_map[mappable_genes].values
                    col_indices.sort()
                    self.dataset_to_genemap_cache[group_name] = (col_indices, mappable_genes_indices.flatten())

    def _load_dataset(
        self,
        ds_class,
        ds_args,
    ):
        """
        Helper method to load datasets from the specified scvi class, name and further arguments.
        :param ds_class: object, the scvi dataloader provided as callable object.
        :param ds_args: list, further kwargs for the dataloader.
        :return: tuple, 1 - the dataset; 2 - the dataset class object; 3 - the dataset name
        """
        class_init_args = getfullargspec(ds_class.__init__).args
        if "save_path" in class_init_args:
            ds_args.update({"save_path": self.save_path})

        dataset = ds_class(**ds_args)
        return dataset, ds_class, dataset.name

    def _build_genemap_from_self(
        self
    ):
        return {gene: pos for (gene, pos) in zip(sorted(self.gene_names), range(len(self.gene_names)))}

    def _build_genemap_from_scvi(
        self,
        dataset_classes: List[GeneExpressionDataset],
        dataset_names: List[str],
        dataset_args: List[any] = None,
        multiprocess: bool = True,
    ):
        """
        Build the gene map by loading the datasets specified in the parameters through their respective dataloaders.

        :param dataset_classes: list
        :param dataset_names: list, strings of the names of the datasets.
        :param dataset_args: list, list of lists of further keyword arguments to provide for each specific dataloader.
        :param multiprocess: bool, load the datasets in parallel or serial.
        :return: dict, the gene map as dictionary with the genes as keys and their positional number as value.
        """
        if dataset_args is None:
            dataset_args = [{}] * len(dataset_names)

        total_genes = set()

        def append_genes(dset):
            nonlocal total_genes
            if dset.gene_names is None:
                # without gene names we can't build a proper mapping
                warnings.warn(
                    f"Dataset {(ds_class, ds_name)} does not have gene_names as attribute. Skipping this dataset.")
                return
            total_genes = total_genes.union(dataset.gene_names)

        if not multiprocess:
            for ds_name, ds_class, ds_args in zip(dataset_names, dataset_classes, dataset_args):
                dataset, _, _ = self._load_dataset(ds_name, ds_class, ds_args)
                append_genes(dataset)
        else:
            with ProcessPoolExecutor(max_workers=min(len(dataset_names), cpu_count() // 2)) as executor:
                futures = list(
                    (executor.submit(
                        self._load_dataset,
                        ds_name,
                        ds_class,
                        ds_args)
                        for ds_name, ds_class, ds_args in zip(dataset_names, dataset_classes, dataset_args))
                )
                for future in as_completed(futures):
                    dataset, ds_class, ds_name = future.result()
                    append_genes(dataset)

        return {gene: pos for (gene, pos) in zip(sorted(total_genes), range(len(total_genes)))}

    def _build_genemap_from_hdf5(
        self,
        in_filename: str = None,
        subselection_datasets: Union[List[str], np.ndarray] = None
    ):
        """
        Build the gene map by loading the gene names from a dataset inside an hdf5 file.
        :param in_filename: str, the name of the hdf5 file.
        :param subselection_datasets: list or ndarray, if provided, a list of dataset names that are to be considered.
        :return: dict, the gene map as dictionary with the genes as keys and their positional number as value.
        """
        if in_filename is None:
            if self.data_load_filename is not None:
                in_filename = self.data_load_filename
            else:
                raise ValueError("No filename to read from provided.")

        total_genes = set()
        with h5py.File(os.path.join(self.save_path, in_filename), 'r') as h5file:
            dataset_group = h5file["Datasets"]
            for dataset_name, dataset_acc in dataset_group.items():
                if subselection_datasets is not None and dataset_name in subselection_datasets:
                    total_genes = total_genes.union(dataset_acc["gene_names"][:].astype(str))
        return {gene: pos for (gene, pos) in zip(sorted(total_genes), range(len(total_genes)))}

    def _build_genemap_from_loom(
        self,
        in_filename: str = None,
        gene_names_attribute_name: str = "Gene"
    ):
        """
        Build the gene map by loading the gene names from a dataset inside a loom file.
        :param in_filename: str, the name of the loom file.
        :param gene_names_attribute_name: str, the accessor name of the attribute storing the gene names.
        :return: dict, the gene map as dictionary with the genes as keys and their positional number as value.
        """
        if in_filename is None and self.data_load_filename is not None:
            in_filename = self.data_load_filename
        else:
            raise ValueError("No filename to read from provided.")

        total_genes = set()
        with loompy.connect(os.path.join(self.save_path, in_filename)) as ds:
            for row_attribute_name in ds.ra:
                if row_attribute_name == gene_names_attribute_name:
                    gene_names = np.char.upper(ds.ra[gene_names_attribute_name].astype(str))
                    total_genes = total_genes.union(gene_names)

        return {gene: pos for (gene, pos) in zip(sorted(total_genes), range(len(total_genes)))}

    @staticmethod
    def _build_genemap_from_memory(
        gene_datasets: List[GeneExpressionDataset]
    ):
        """
        Build the gene map from datasets already loaded into memory.
        :param gene_datasets: list, all datasets, that are meant to be used for the gene map.
        :return: dict, the gene map as dictionary with the genes as keys and their positional number as value.
        """
        total_genes = set()
        for dataset in gene_datasets:
            gene_names = np.char.upper(dataset.gene_names.astype(str))
            total_genes = total_genes.union(gene_names)

        return {gene: pos for (gene, pos) in zip(sorted(total_genes), range(len(total_genes)))}

    @staticmethod
    def _dataset_class_str(dataset_class):
        return re.search(class_regex_pattern, str(dataset_class)).group()

    def _write_dataset_to_hdf5(
        self,
        out_filename,
        dataset,
        dataset_class,
        dataset_name
    ):
        """
        Method to append a dataset onto a previously created hdf5 file.

        :param out_filename: str, the filename of the hdf5 file.
        :param dataset: GeneExpressionDataset, the dataset to write into the hdf5 file.
        :param dataset_class: object, the dataloader class this dataset is from
        :param dataset_name: str, the filename that characterizes the data (e.g. the data name from 10x datasets).
        """
        string_dt = h5py.special_dtype(vlen=str)
        with h5py.File(os.path.join(self.save_path, out_filename), 'a') as h5file:
            data_group = h5file["Datasets"]
            dataset_class_str = self._dataset_class_str(dataset_class)

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
            group_name = f"{dataset_class_str}_{dataset_name}"

            if group_name in data_group:
                i = 0
                while True:
                    if f"{group_name}_{i}" not in data_group:
                        logger.warning(f"Dataset group name {group_name} already exists. "
                                       f"Appending suffix '_{i}'.")
                        group_name = f"{group_name}_{i}"
                        break
                    i += 1

            dataset_h5_g = data_group.create_group(group_name)

            if isinstance(X, (sp_sparse.csr_matrix, sp_sparse.csc_matrix)):
                dset = dataset_h5_g.create_dataset("X",
                                                   shape=(X.shape[0], len(dataset.gene_names)),
                                                   compression="lzf",
                                                   dtype=np.float32)
                nr_rows = 5000
                for start in tqdm(range(0, len(dataset), nr_rows),
                                  desc="Writing sparse matrix iteratively to file"):
                    sl = slice(start, min(start + nr_rows, X.shape[0]))
                    dset[sl, :] = X[sl, :].toarray().astype(np.float32)
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
        return

    def _write_dataset_to_loom(
        self,
        dataset_ptr,
        dataset,
        dataset_name=None,
        dataset_class=None
    ):
        """
        Method to write a dataset onto an opened loom file.
        :param dataset_ptr: loom file pointer, the reference to the loom file, onto which the dataset should be written.
        :param dataset: GeneExpressionDataset, the dataset to write.
        :param dataset_name: str, the name of the current dataset to append to the total name
        :param dataset_class: object, the class identifier, that signifies which dataloader this dataset came from.
        """
        if dataset_name is None:
            dataset_class_str = self._dataset_class_str(dataset_class)
            dataset_ptr.attrs.DatasetName += f"_{dataset_class_str}"
        else:
            dataset_ptr.attrs.DatasetName += f"_{dataset_name}"
        # grab the necessary data parts:
        # aside from the data itself (X), the gene_names, local means, local_vars, batch_indices and labels
        # there are no guaranteed attributes of each dataset.
        X, gene_names, batch_indices, labels, local_means, local_vars = (
            dataset.X,
            dataset.gene_names,
            dataset.batch_indices,
            dataset.labels,
            dataset.local_means,
            dataset.local_means
        )

        if not all(dataset.cell_types == "undefined"):
            known_cts = [ct for ct in dataset_ptr.attrs.CellTypes]
            cts = dataset.cell_types
            labels = cts[labels]
            # append cell type only if unknown
            for ct in cts:
                if ct not in known_cts:
                    known_cts.append(ct)
            # remap from ["endothelial_cell", "B_cell", "B_cell", ...] to [3, 5, 5, ...]
            for cat_from, cat_to in zip(cts, [known_cts.index(ct) for ct in cts]):
                labels[labels == cat_from] = cat_to
            labels = labels.astype(np.uint16)
            dataset_ptr.attrs.CellTypes = known_cts
        if "BatchID" in dataset_ptr.col_attrs:
            max_batch_idx = dataset_ptr.ca["BatchID"].max() + 1
            batch_indices = batch_indices + max_batch_idx

        if isinstance(X, sp_sparse.csc_matrix):
            X = X.tocsr()
        if isinstance(X, sp_sparse.csr_matrix):
            nr_rows = 5000

            mappable_genes_indices = np.isin(np.char.upper(gene_names), self.gene_map.index)
            mappable_genes = gene_names[mappable_genes_indices]
            col_indices = self.gene_map[mappable_genes].values

            for start in range(0, len(dataset), nr_rows):
                select = slice(start, min(start + nr_rows, X.shape[0]))
                X_batch = self.map_data(data=X[select, :].toarray(),
                                        col_indices=col_indices,
                                        mappable_genes_indices=mappable_genes_indices).astype(np.int32).transpose()

                dataset_ptr.add_columns(
                    X_batch,
                    col_attrs={"ClusterID": labels[select],
                               "BatchID": batch_indices[select],
                               "LocalMeans": local_means[select],
                               "LocalVars": local_vars[select]},
                    row_attrs={"Gene": self.gene_names}
                )
        else:
            dataset_ptr.add_columns(
                self.map_data(data=X, gene_names=gene_names).transpose(),
                col_attrs={"ClusterID": labels,
                           "BatchID": batch_indices,
                           "LocalMeans": local_means,
                           "LocalVars": local_vars},
                row_attrs={"Gene": self.gene_names}
            )
        return

    def _union_from_memory_to_hdf5(
        self,
        gene_datasets: List[GeneExpressionDataset],
        out_filename=None
    ):
        """
        Combines multiple unlabelled gene_datasets based on a mapping of gene names. Stores the final
        dataset onto a hdf5 file with filename ``out_filename``.
        :param out_filename: str, name of the file to which to write.
        :param gene_datasets:  List, list of already loaded datasets of class ``GeneExpressionDataset``.
        """
        if out_filename is None:
            if self.data_save_filename is not None:
                out_filename = self.data_save_filename
            else:
                raise ValueError("No filename to write to provided.")

        filename, ext = os.path.splitext(out_filename)

        if ext != ".h5":
            logger.warn(f"Chosen file type is 'hdf5'. Yet provided ending is: '{ext}' versus expected ending: '.h5'.")

        with h5py.File(os.path.join(self.save_path, out_filename), 'w') as h5file:
            # just opening the file to overwrite any existing content and enabling sub-function to simply append
            h5file.create_group("Datasets")
            h5file.create_group("Metadata")

        counts = []

        datasets_pbar = tqdm(gene_datasets)
        for dataset in datasets_pbar:
            dataset_class_str = self._dataset_class_str(type(dataset))
            datasets_pbar.set_description(f"Writing dataset {dataset_class_str, dataset.name} to h5 file")

            self._write_dataset_to_hdf5(out_filename, dataset, type(dataset), dataset.name)

            counts.append(np.array(dataset.X.sum(axis=1)).flatten())

        log_counts = np.concatenate(np.log(counts))
        total_lm = np.mean(log_counts).reshape(-1, 1).astype(np.float32)
        total_lv = np.var(log_counts).reshape(-1, 1).astype(np.float32)
        with h5py.File(os.path.join(self.save_path, out_filename), 'a') as h5file:
            g = h5file["Metadata"]
            g.create_dataset("local_mean_complete_dataset", data=total_lm)
            g.create_dataset("local_var_complete_dataset", data=total_lv)

        self._set_attributes(out_filename)
        return

    def _union_from_memory_to_loom(
        self,
        gene_datasets: List[GeneExpressionDataset],
        out_filename=None
    ):
        """
        Combines multiple unlabelled gene_datasets based on a mapping of gene names. Stores the final
        dataset onto a loom file with filename ``out_filename``.
        :param out_filename: str, name of the file to which to write.
        :param gene_datasets:  List, list of already loaded datasets of class ``GeneExpressionDataset``.
        """
        if out_filename is None:
            if self.data_save_filename is not None:
                out_filename = self.data_save_filename
            else:
                raise ValueError("No filename to write to provided.")

        filename, ext = os.path.splitext(out_filename)
        if ext != ".loom":
            logger.warn(f"Chosen file type is 'loom'. Yet provided ending is: '{ext}' versus expected ending: '.loom'.")

        file = os.path.join(self.save_path, out_filename)

        counts = []
        with loompy.new(file) as dsout:
            dsout.attrs.CellTypes = ['undefined']
            dsout.attrs.DatasetName = ""

            datasets_pbar = tqdm(gene_datasets)
            for dataset in datasets_pbar:
                dataset_class_str = self._dataset_class_str(type(dataset))
                datasets_pbar.set_description(f"Writing dataset '{dataset_class_str} - {dataset.name}' to loom file")
                self._write_dataset_to_loom(dsout, dataset, type(dataset))
                counts.append(np.array(dataset.X.sum(axis=1)).flatten())
            cts = dsout.attrs.CellTypes[:]
            labels = dsout.ca["ClusterID"][:]
            if (labels > 0).all():
                labels = cts[labels]
                # there are no undefined cell types in the data. Removing this label.
                cts = np.sort(cts[cts != "undefined"])
            else:
                labels = cts[labels]
                cts.sort()
            labels, _ = remap_categories(original_categories=labels, mapping_from=cts)
            dsout.attrs.CellTypes = cts
            dsout.ca["ClusterID"] = labels

            log_counts = np.log(np.concatenate(counts))
            total_lm = np.mean(log_counts).reshape(-1, 1).astype(np.float32)
            total_lv = np.var(log_counts).reshape(-1, 1).astype(np.float32)
            dsout.attrs.LocalMeanCompleteDataset = total_lm
            dsout.attrs.LocalVarCompleteDataset = total_lv

        self._set_attributes(out_filename)
        return

    def _union_from_memory_to_memory(
        self,
        gene_datasets: List[GeneExpressionDataset],
        shared_batches=False
    ):
        """
        Combines multiple unlabelled gene_datasets based on a mapping of gene names. Loads the final
        dataset directly into memory.
        :param gene_datasets: List, the loaded data sets of (inherited) class GeneExpressionDataset to concatenate.
        """
        containers = defaultdict(list)
        n_batch_offset = 0

        for dataset in tqdm(gene_datasets, desc="Concatenating datasets"):
            n_batch_offset = self._extract_content(dataset, containers, n_batch_offset, shared_batches)

        labels = np.concatenate(containers["labels"])
        containers["local_means"] = np.concatenate(containers["local_means"])
        containers["local_vars"] = np.concatenate(containers["local_vars"])
        containers["cell_types"] = np.sort(np.unique(labels))
        containers["labels"], _ = remap_categories(labels, containers["cell_types"])
        containers["X"] = sp_sparse.vstack(
            containers["X"]
        )  # done before the populate data file, in order to release unstacked memory of X
        self.populate_from_data(**containers,
                                gene_names=self.gene_names
                                )
        logger.info(f"Joined {len(gene_datasets)} datasets to one of shape {self.nb_cells} x {self.gene_names_len}.")
        return

    def _union_from_self_to_hdf5(
        self,
        out_filename
    ):
        """
        Simplifier for when one wants to write the loaded data to a hdf5 file.
        :param out_filename: str, the filename of the hdf5 file to write.
        """
        self._union_from_memory_to_hdf5(out_filename=out_filename, gene_datasets=[self])
        return

    def _union_from_self_to_loom(
        self,
        out_filename
    ):
        """
        Simplifier for when one wants to write the loaded data to a loom file.
        :param out_filename: str, the filename of the loom file to write.
        """
        self._union_from_memory_to_loom(out_filename=out_filename, gene_datasets=[self])
        return

    def _union_from_scvi_to_memory(
        self,
        dataset_classes=None,
        dataset_args=None,
        shared_batches=False
    ):
        """
        Loads scvi gene_datasets from the specified dataloaders and combines them based on a mapping of gene names.
        Loads the concatenated dataset into memory only.

        :param dataset_classes: List, list of class-initializers of scvi GeneExpression datasets
        :param dataset_args: List, list of further positional arguments for when loading the datasets
        :param shared_batches: bool, whether the batch_indices are shared or not for the datasets
        """
        containers = defaultdict(list)
        n_batch_offset = 0

        if dataset_args is None:
            dataset_args = [{}] * len(dataset_classes)

        _LOCK = Lock()

        with ThreadPoolExecutor() as executor:
            futures = list(
                (executor.submit(self._load_dataset,
                                 ds_class,
                                 ds_args
                                 )
                 for ds_class, ds_args in zip(dataset_classes, dataset_args))
            )
            for future in as_completed(futures):
                ###################
                _LOCK.acquire()

                dataset = future.result()[0]
                n_batch_offset = self._extract_content(dataset, containers, n_batch_offset, shared_batches)

                _LOCK.release()
                ###################

        labels = np.concatenate(containers["labels"])
        containers["local_means"] = np.concatenate(containers["local_means"])
        containers["local_vars"] = np.concatenate(containers["local_vars"])
        containers["cell_types"] = np.sort(np.unique(labels))
        containers["labels"], _ = remap_categories(labels, containers["cell_types"])
        containers["X"] = sp_sparse.vstack(
            containers["X"]
        )  # done before the populate data file, in order to release unstacked memory of X
        self.populate_from_data(**containers,
                                gene_names=self.gene_names
                                )

    def _union_from_scvi_to_hdf5(
        self,
        dataset_classes,
        dataset_args=None,
        out_filename=None
    ):
        """
        Combines multiple unlabelled gene_datasets based on a mapping of gene names. Stores the final
        dataset onto a Hdf5 file with filename ``out_filename``.
        :param dataset_classes: List, list of class-initializers of scvi GeneExpression datasets
        :param dataset_args: List, list of further positional arguments for when loading the datasets
        :param out_filename: str, name of the file to which to write.
        """
        if out_filename is None:
            if self.data_save_filename is not None:
                out_filename = self.data_save_filename
            else:
                raise ValueError("No filename to write to provided.")

        self._check_extension(out_filename, ".h5")

        with h5py.File(os.path.join(self.save_path, out_filename), 'w') as h5file:
            # just opening the file to overwrite any existing content and enabling sub-function to simply append
            h5file.create_group("Datasets")
            h5file.create_group("Metadata")
            pass

        if dataset_args is None:
            dataset_args = [{}] * len(dataset_classes)
        lock = Lock()
        with ThreadPoolExecutor() as executor:
            futures = list(
                (executor.submit(self._load_dataset,
                                 ds_class,
                                 ds_args
                                 )
                 for ds_class, ds_args in zip(dataset_classes, dataset_args))
            )
            for future in as_completed(futures):
                res = future.result()
                dataset, dataset_class, dataset_fname = res

                lock.acquire()
                self._write_dataset_to_hdf5(out_filename, dataset, dataset_class, dataset_fname)
                lock.release()
        print(f"conversion completed to file '{out_filename}'")
        self._set_attributes(out_filename)

    def _union_from_scvi_to_loom(
        self,
        dataset_classes,
        dataset_args=None,
        out_filename=None
    ):
        """
        Combines multiple unlabelled gene_datasets based on a mapping of gene names. Stores the final
        dataset onto a loom file with filename ``out_filename``.
        :param dataset_classes: List, list of class-initializers of scvi GeneExpression datasets
        :param dataset_args: List, list of further positional arguments for when loading the datasets
        :param out_filename: str, name of the file to which to write.
        """
        if out_filename is None:
            if self.data_save_filename is not None:
                out_filename = self.data_save_filename
            else:
                raise ValueError("No filename to write to provided.")

        self._check_extension(out_filename, ".loom")

        with loompy.new(os.path.join(self.save_path, out_filename)) as dsout:
            dsout.attrs.CellTypes = ['undefined']
            dsout.attrs.DatasetName = ""

            if dataset_args is None:
                dataset_args = [{}] * len(dataset_classes)

            counts = []

            lock = Lock()
            with ThreadPoolExecutor() as executor:
                futures = list(
                    (executor.submit(self._load_dataset,
                                     ds_class,
                                     ds_args
                                     )
                     for ds_class, ds_args in zip(dataset_classes, dataset_args))
                )
                futures_pbar = tqdm(as_completed(futures))
                for future in futures_pbar:
                    lock.acquire()

                    res = future.result()
                    dataset, dataset_class, dataset_fname = res
                    dataset_class_str = self._dataset_class_str(type(dataset))
                    futures_pbar.set_description(f"Writing dataset '{dataset_class_str} - {dataset.name}' to loom file")

                    self._write_dataset_to_loom(dsout, dataset, type(dataset))

                    counts.append(np.array(dataset.X.sum(axis=1)).flatten())

                    lock.release()

            log_counts = np.log(np.concatenate(counts))
            total_lm = np.mean(log_counts).reshape(-1, 1).astype(np.float32)
            total_lv = np.var(log_counts).reshape(-1, 1).astype(np.float32)
            dsout.attrs.LocalMeanCompleteDataset = total_lm
            dsout.attrs.LocalVarCompleteDataset = total_lv

            cts = dsout.attrs.CellTypes[:]
            labels = dsout.ca["ClusterID"][:]
            if (labels > 0).all():
                labels = cts[labels]
                # there are no undefined cell types in the data. Removing this label.
                cts = np.sort(cts[cts != "undefined"])
            else:
                labels = cts[labels]
                cts.sort()
            labels, _ = remap_categories(original_categories=labels, mapping_from=cts)
            dsout.attrs.CellTypes = cts
            dsout.ca["ClusterID"] = labels

        print(f"conversion completed to file '{out_filename}'")
        self._set_attributes(out_filename)

    def _union_from_hdf5_to_memory(
        self,
        in_filename=None,
        datasets_subselection: List[str] = None,
        shared_batches=False
    ):
        """
        Concatenate all datasets in an hdf5 file to memory.

        :param in_filename: str, the filename of the hdf5 file to read.
        """
        if in_filename is None:
            if self.data_load_filename is not None:
                in_filename = self.data_load_filename
            else:
                raise ValueError("No filename to read from provided.")

        subselection_check = lambda dset_name: dset_name in datasets_subselection
        if datasets_subselection is None:
            subselection_check = lambda dset_name: True

        containers = defaultdict(list)

        n_batch_offset = 0

        with h5py.File(os.path.join(self.save_path, in_filename), 'r') as h5file:
            for group_name, group in h5file["Datasets"].items():
                if subselection_check(group_name):
                    containers["X"].append(
                        sp_sparse.csr_matrix(
                            self.map_data(
                                group["X"][:],
                                np.char.upper(group["gene_names"][:].astype(str))
                            )
                        )
                    )
                    containers["local_means"].append(group["local_means"][:])
                    containers["local_vars"].append(group["local_vars"][:])
                    bis = group["batch_indices"][:]
                    if not shared_batches:
                        bis += n_batch_offset
                        n_batch_offset += bis.max() + 1
                    containers["batch_indices"].append(bis)
                    if "cell_types" in group:
                        containers["labels"].append(group["cell_types"][:][group["labels"][:]])
                    else:
                        containers["labels"].append(np.repeat("undefined", len(bis)))

        labels = np.concatenate(containers["labels"])
        containers["local_means"] = np.concatenate(containers["local_means"])
        containers["local_vars"] = np.concatenate(containers["local_vars"])
        containers["cell_types"] = np.sort(np.unique(labels))
        containers["labels"], _ = remap_categories(labels, containers["cell_types"])
        containers["X"] = sp_sparse.vstack(containers["X"])
        self.populate_from_data(
            **containers,
            gene_names=self.gene_names
        )
        return

    def _union_from_hdf5_to_loom(
        self,
        in_filename=None,
        out_filename=None,
        datasets_subselection: List[str] = None,
        shared_batches=False
    ):
        """
        Concatenate all datasets mentioned in `` datasets_subselection`` in an hdf5 file to a loom file.

        :param in_filename: str, the filename of the hdf5 file to read.
        :param out_filename: str, the filename of the loom file to write to.
        :param datasets_subselection: list, a list of dataset names which are to be considered for concatenation.
        :param shared_batches: bool, if true all batch annotation is considered to be from the same experiment.
        Otherwise each new dataset will see its annotation data consecutively increased to create distinction.
        """
        if in_filename is None:
            if self.data_load_filename is not None:
                in_filename = self.data_load_filename
            else:
                raise ValueError("No filename to read from provided.")
        if out_filename is None:
            if self.data_save_filename is not None:
                out_filename = self.data_save_filename
            else:
                raise ValueError("No filename to write to provided.")

        subselection_check: Callable[[str], bool] = lambda dset_name: dset_name in datasets_subselection
        if datasets_subselection is None:
            subselection_check = lambda dset_name: True

        labels = []
        cell_types = {'undefined'}

        with h5py.File(os.path.join(self.save_path, in_filename), 'r') as h5file, \
            loompy.new(os.path.join(self.save_path, out_filename)) as loomfile:

            loomfile.attrs.CellTypes = []
            loomfile.attrs.DatasetName = ""

            dataset_group = h5file["Datasets"]
            for group_name, group in dataset_group.items():
                if subselection_check(group_name):
                    if "cell_types" in group:
                        cts = group["cell_types"][:]
                        labels.append(cts[group["labels"][:]])
                        cell_types = cell_types.union(cts)
                    else:
                        labels.append(np.repeat("undefined", group["X"].shape[0]))

            labels = np.concatenate(labels)
            if (labels != 'undefined').all():
                cell_types.remove("undefined")

            cell_types = sorted(cell_types)
            labels, _ = remap_categories(labels, mapping_from=cell_types)

            start_idx = 0
            n_batch_offset = 0
            for group_name, group in dataset_group.items():
                if subselection_check(group_name):
                    X = sp_sparse.csr_matrix(
                        self.map_data(
                            group["X"][:],
                            np.char.upper(group["gene_names"][:].astype(str))
                        )
                    )
                    lbs = labels[start_idx:start_idx + X.shape[0]]
                    local_means = group["local_means"][:]
                    local_vars = group["local_vars"][:]
                    batch_indices = group["batch_indices"][:]
                    if not shared_batches:
                        batch_indices += n_batch_offset
                        n_batch_offset = batch_indices.max() + 1

                    start_idx = start_idx + X.shape[0]

                    self.populate_from_data(X=X,
                                            batch_indices=batch_indices,
                                            labels=lbs,
                                            gene_names=self.gene_names,
                                            cell_types=cell_types,
                                            local_means=local_means,
                                            local_vars=local_vars
                                            )
                    self._write_dataset_to_loom(loomfile, self, dataset_name=group_name)

        self._set_attributes(in_filename)
        return

    def _union_from_loom_to_memory(
        self,
        in_filename=None,
        as_sparse=True,
        gene_names_attribute_name="Gene",
        batch_indices_attribute_name="BatchID",
        local_means_attribute_name="LocalMeans",
        local_vars_attribute_name="LocalVars",
        labels_attribute_name="ClusterID",
        cell_types_attribute_name="CellTypes",
        total_local_mean_attribute_name="LocalMeanCompleteDataset",
        total_local_var_attribute_name="LocalVarCompleteDataset",
        dataset_name_attribute_name="DatasetName"
    ):
        """
        Loads a loom file completely into memory. If ``as_sparse`` is true, the data will be loaded into a scipy sparse
        matrix, otherwise a dense numpy array.

        The loading code has been taken mostly from the loom dataloader class in ``loom.py``.

        :param in_filename: str, filename of the loom file.
        :param as_sparse: bool, flag whether to use sparse matrices or numpy arrays.
        :param gene_names_attribute_name: str, the identifier for the gene names attribute within the loom file.
        :param batch_indices_attribute_name: str, the identifier for the batch indices attribute within the loom file.
        :param local_means_attribute_name: str, the identifier for the local means attribute within the loom file.
        :param local_vars_attribute_name: str, the identifier for the local vars attribute within the loom file.
        :param labels_attribute_name: str, the identifier for the label attribute within the loom file.
        :param cell_types_attribute_name: str, the identifier for the cell types attribute within the loom file.
        :param total_local_mean_attribute_name: str, the identifier for the attribute within the loom file. This is the
        attribute, which stores the local_mean of each data entry, if all of the data were to be seen as from a single
        batch. Needed for when batch annotation is ignored.
        :param total_local_var_attribute_name: str, the identifier for the local var attribute within the loom file, if
        the dataset ignores batch indices (see @total_local_mean).
        :param dataset_name_attribute_name: str, the identifier for the dataset name attribute within the loom file.
        """
        if in_filename is None:
            if self.data_load_filename is not None:
                in_filename = self.data_load_filename
            else:
                raise ValueError("No filename to read from provided.")

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
                    gene_names = np.char.upper(ds.ra[gene_names_attribute_name].astype(str))
                else:
                    gene_attributes_dict = (
                        gene_attributes_dict if gene_attributes_dict is not None else {}
                    )
                    gene_attributes_dict[row_attribute_name] = ds.ra[row_attribute_name]

            for column_attribute_name in ds.ca:
                if column_attribute_name == batch_indices_attribute_name:
                    batch_indices = ds.ca[batch_indices_attribute_name][:].astype(int)
                elif column_attribute_name == labels_attribute_name:
                    labels = ds.ca[labels_attribute_name][:].astype(int)
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
                    cell_types = ds.attrs[cell_types_attribute_name].astype(str)
                elif global_attribute_name == total_local_mean_attribute_name:
                    if self.ignore_batch_annotation:
                        local_means = ds.attrs[total_local_mean_attribute_name]
                elif global_attribute_name == total_local_var_attribute_name:
                    if self.ignore_batch_annotation:
                        local_vars = ds.attrs[total_local_var_attribute_name]
                elif global_attribute_name == dataset_name_attribute_name:
                    name = ds.attrs[dataset_name_attribute_name].astype(str)
                else:
                    global_attributes_dict = (
                        global_attributes_dict if global_attributes_dict is not None else {}
                    )
                    global_attributes_dict[global_attribute_name] = ds.attrs[global_attribute_name]

            if global_attributes_dict is not None:
                self.global_attributes_dict = global_attributes_dict

            if as_sparse:
                shape = ds.shape
                nr_rows = 5000
                X = sp_sparse.csr_matrix(([], ([], [])), shape=(0, shape[0]), dtype=np.float32)
                for i in tqdm(range(0, shape[1], nr_rows), desc="Loading from file to memory iteratively"):
                    X = sp_sparse.vstack([X, sp_sparse.csr_matrix(ds[:, i:i + nr_rows].T)])
            else:
                X = ds[:, :].T

        self.populate_from_data(X=X,
                                gene_names=gene_names,
                                batch_indices=batch_indices,
                                cell_types=cell_types,
                                labels=labels,
                                local_means=np.repeat(local_means, X.shape[0], axis=0),
                                local_vars=np.repeat(local_vars, X.shape[0], axis=0),
                                name=name)
        return

    def _union_from_loom_to_hdf5(
        self,
        in_filename=None,
        out_filename=None,
        as_sparse=True,
        gene_names_attribute_name="Gene",
        batch_indices_attribute_name="BatchID",
        local_means_attribute_name="LocalMeans",
        local_vars_attribute_name="LocalVars",
        labels_attribute_name="ClusterID",
        cell_types_attribute_name="CellTypes",
        total_local_mean_attribute_name="LocalMeanCompleteDataset",
        total_local_var_attribute_name="LocalVarCompleteDataset",
        dataset_name_attribute_name="DatasetName"
    ):
        """
        Loads a loom file completely into memory. If ``as_sparse`` is true, the data will be loaded into a scipy sparse
        matrix, otherwise a dense numpy array.

        The loading code has been taken mostly from the loom dataloader class in ``loom.py``.

        :param in_filename: str, filename of the loom file.
        :param as_sparse: bool, flag whether to use sparse matrices or numpy arrays.
        :param gene_names_attribute_name: str, the identifier for the gene names attribute within the loom file.
        :param batch_indices_attribute_name: str, the identifier for the batch indices attribute within the loom file.
        :param local_means_attribute_name: str, the identifier for the local means attribute within the loom file.
        :param local_vars_attribute_name: str, the identifier for the local vars attribute within the loom file.
        :param labels_attribute_name: str, the identifier for the label attribute within the loom file.
        :param cell_types_attribute_name: str, the identifier for the cell types attribute within the loom file.
        :param total_local_mean_attribute_name: str, the identifier for the attribute within the loom file. This is the
        attribute, which stores the local_mean of each data entry, if all of the data were to be seen as from a single
        batch. Needed for when batch annotation is ignored.
        :param total_local_var_attribute_name: str, the identifier for the local var attribute within the loom file, if
        the dataset ignores batch indices (see @total_local_mean).
        :param dataset_name_attribute_name: str, the identifier for the dataset name attribute within the loom file.
        """
        if in_filename is None:
            if self.data_load_filename is not None:
                in_filename = self.data_load_filename
            else:
                raise ValueError("No filename to read from provided.")
        if out_filename is None:
            if self.data_save_filename is not None:
                out_filename = self.data_save_filename
            else:
                raise ValueError("No filename to save to provided.")

        batch_indices = None
        labels = None
        cell_types = None
        gene_names = None
        local_means = None
        local_vars = None

        string_dt = h5py.special_dtype(vlen=str)

        with loompy.connect(os.path.join(self.save_path, in_filename)) as ds, \
            h5py.File(os.path.join(self.save_path, out_filename), 'w') as h5file:

            h5file.create_group("Datasets")
            h5file.create_group("Metadata")
            data_group = h5file["Datasets"]
            try:
                name = ds.attrs[dataset_name_attribute_name].astype(str)
            except KeyError as e:
                name = "UNKNOWN_NAME"
            dataset_group = data_group.create_group(name)
            ds_out = dataset_group.create_dataset("X",
                                                  shape=(ds.shape[1], ds.shape[0]),
                                                  compression="lzf",
                                                  dtype=np.float32)

            for row_attribute_name in ds.ra:
                if row_attribute_name == gene_names_attribute_name:
                    gene_names = np.char.upper(ds.ra[gene_names_attribute_name].astype(str))
                else:
                    gene_attributes_dict = (
                        gene_attributes_dict if gene_attributes_dict is not None else {}
                    )
                    gene_attributes_dict[row_attribute_name] = ds.ra[row_attribute_name]

            for column_attribute_name in ds.ca:
                if column_attribute_name == batch_indices_attribute_name:
                    batch_indices = ds.ca[batch_indices_attribute_name][:].astype(int)
                elif column_attribute_name == labels_attribute_name:
                    labels = ds.ca[labels_attribute_name][:].astype(int)
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
                    cell_types = ds.attrs[cell_types_attribute_name].astype(str)
                elif global_attribute_name == total_local_mean_attribute_name:
                    if self.ignore_batch_annotation:
                        local_means = ds.attrs[total_local_mean_attribute_name]
                elif global_attribute_name == total_local_var_attribute_name:
                    if self.ignore_batch_annotation:
                        local_vars = ds.attrs[total_local_var_attribute_name]
                else:
                    global_attributes_dict = (
                        global_attributes_dict if global_attributes_dict is not None else {}
                    )
                    global_attributes_dict[global_attribute_name] = ds.attrs[global_attribute_name]

            dataset_group.create_dataset("gene_names", data=gene_names.astype(np.dtype("S")), dtype=string_dt)
            dataset_group.create_dataset("local_means", data=local_means)
            dataset_group.create_dataset("local_vars", data=local_vars)
            dataset_group.create_dataset("batch_indices", data=batch_indices)
            dataset_group.create_dataset("labels", data=labels)
            dataset_group.create_dataset("cell_types",
                                         data=cell_types.astype(np.dtype("S")), dtype=string_dt)

            if global_attributes_dict is not None:
                self.global_attributes_dict = global_attributes_dict

            shape = ds.shape
            nr_rows = 5000
            for i in tqdm(range(0, shape[1], nr_rows), desc="Loading from loom and writing to hdf5 iteratively"):
                ds_out[i:i + nr_rows, :] = ds[:, i:i + nr_rows].astype(np.float32).T
        self._set_attributes(in_filename)
        return

    #############################
    #                           #
    #          utils            #
    #                           #
    #############################

    @staticmethod
    def _check_extension(out_filename, expected):
        _, ext = os.path.splitext(out_filename)
        if ext != expected:
            logger.warn(f"Provided file type is '{ext}', but expected: '{expected}'.")

    def _compute_library_size(
        self,
        data,
        batch_size=None
    ):
        if self.low_memory:
            logger.warn("Library size computation ignored in low memory mode. "
                        "Ensure you have loaded the local means and vars!")
        else:
            sum_counts = data.sum(axis=1)
            log_counts = np.log(sum_counts)
            m = np.mean(log_counts)
            v = np.var(log_counts)

            return np.array(m).astype(np.float32).reshape(-1, 1), \
                   np.array(v).reshape(-1, 1).astype(np.float32)

    def _extract_content(
        self,
        dataset,
        containers,
        n_batch_offset,
        shared_batches
    ) -> int:
        """
        Helper function to extract the main information contained in an scvi dataset to be put into an incoming
        storage container.
        This container is updated in-place.

        The extracted variables are:
            - X
            - gene_names
            - local_means
            - local_vars
            - batch_indices
            - labels

        However, the labels are not integers referring to a list of cell types, but rather the cell types themselves.

        :return: int, the updated batch_offset counter
        """
        containers["X"].append(
            sp_sparse.csr_matrix(
                self.map_data(
                    dataset.X,
                    np.char.upper(dataset.gene_names)
                )
            )
        )
        containers["local_means"].append(dataset.local_means)
        containers["local_vars"].append(dataset.local_vars)
        bis = dataset.batch_indices.copy()
        if not shared_batches:
            bis += n_batch_offset
            n_batch_offset += bis.max() + 1
        containers["batch_indices"].append(bis)
        if dataset.cell_types is not None:
            containers["labels"].append(dataset.cell_types[dataset.labels])
        else:
            containers["labels"].append(np.repeat("undefined", len(bis)))
        return n_batch_offset

    @staticmethod
    def _compute_nr_datapoints_in_hdf5(h5_filepath):
        nr_data_entries = 0
        with h5py.File(h5_filepath, "r") as h5file:
            for group_name, group in h5file["Datasets"].items():
                nr_data_entries += group["X"].shape[0]

        return nr_data_entries

    @staticmethod
    def _load_local_means_vars_from_hdf5(h5_filepath,
                                         attr_map,
                                         ):
        nr_datapoints = UnionDataset._compute_nr_datapoints_in_hdf5(h5_filepath)
        local_means = np.zeros(nr_datapoints, dtype=np.float32)
        local_vars = np.zeros(nr_datapoints, dtype=np.float32)

        with h5py.File(h5_filepath, "r") as h5file:
            for group_name, group in h5file["Datasets"].items():
                lms, lvs = group["local_means"][:], group["local_vars"][:]
                this_dset_indices = np.array(attr_map[group_name])
                local_means[this_dset_indices] = lms.flatten()
                local_vars[this_dset_indices] = lvs.flatten()

        return local_means, local_vars

    @staticmethod
    def _load_batch_indices_from_hdf5(h5_filepath,
                                      attr_map
                                      ):
        batch_indices = np.zeros(UnionDataset._compute_nr_datapoints_in_hdf5(h5_filepath), dtype=np.int64)
        with h5py.File(h5_filepath, "r") as h5file:
            curr_offset = 0
            for group_name, group in h5file["Datasets"].items():
                bis = group["batch_indices"][:].flatten()
                this_dset_indices = attr_map[group_name]
                batch_indices[this_dset_indices] = bis + curr_offset
                curr_offset += bis.max() + 1
        n_batches = len(np.unique(batch_indices).astype(np.int64))
        return batch_indices, n_batches

    @staticmethod
    def _load_labels_from_hdf5(h5_filepath,
                               attr_map
                               ):
        labels = np.zeros(UnionDataset._compute_nr_datapoints_in_hdf5(h5_filepath), dtype=np.int64)
        known_cts = ["undefined"]  # known cell types
        with h5py.File(h5_filepath, "r") as h5file:
            for group_name, group in h5file["Datasets"].items():
                lbs = group["labels"][:]
                if lbs.sum() == 0:
                    continue
                cts = group["cell_types"][:]
                lbs = cts[lbs]
                # append cell type only if unknown
                for ct in cts:
                    if ct not in known_cts:
                        known_cts.append(ct)
                # remap from ["endothelial_cell", "B_cell", "B_cell", ...] to [3, 5, 3, ...]
                for cat_from, cat_to in zip(cts, [known_cts.index(ct) for ct in cts]):
                    lbs[lbs == cat_from] = cat_to

                this_dataset_indices = np.array(attr_map[group_name])
                labels[this_dataset_indices] = lbs.flatten()
        if (labels > 0).all():
            # there are no undefined cell types
            known_cts.remove("undefined")
            labels -= 1  # reduce the label of all cells by 1
        labels = np.array(known_cts)[labels]
        known_cts = np.sort(known_cts)
        labels, _ = remap_categories(labels, mapping_from=known_cts)
        labels = labels.reshape(-1, 1)
        cell_types = np.array(known_cts)
        return labels, cell_types

    #############################
    #                           #
    #   Base methods override   #
    #                           #
    #############################

    def compute_library_size_batch(self):
        """
        Computes the library size per batch. Overrides base method with a method that practically avoids computing the
        library size for the low memmory setting, because computing library size for data stemming from out of memory is
        incredibly slow.

        More a hotfix than a smart solution.
        """
        if not self.low_memory:
            self.local_means = np.zeros((self.nb_cells, 1))
            self.local_vars = np.zeros((self.nb_cells, 1))
            for i_batch in range(self.n_batches):
                idx_batch = np.squeeze(self.batch_indices == i_batch)
                self.local_means[idx_batch], self.local_vars[idx_batch] = self._compute_library_size(
                    self.X[idx_batch],
                    batch_size=len(idx_batch)
                )
            self.cell_attribute_names.update(["local_means", "local_vars"])

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
        """
        Base class method override to allow the setting of extra features such as local_means via kwargs
        """
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
        return self

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
            self._batch_indices = np.zeros((len(batch_indices), 1), dtype=np.int64)
            self.n_batches = len(np.unique(batch_indices))

    @property
    def labels(self) -> np.ndarray:
        return self._labels

    @labels.setter
    def labels(
        self,
        labels: Union[List[int], np.ndarray]
    ):
        """Sets labels and the number of labels"""
        labels = np.asarray(labels, dtype=np.uint16).reshape(-1, 1)
        self.n_labels = len(np.unique(labels))
        self._labels = labels

    @property
    def X(self):
        return self._X

    @X.setter
    def X(
        self,
        X: Union[np.ndarray, sp_sparse.csr_matrix, DataLoaderh5, DataLoaderLoom]
    ):
        """Sets the data attribute ``X`` without recomputing the library size."""
        n_dim = len(X.shape)
        if n_dim != 2:
            raise ValueError(
                "Gene expression data should be 2-dimensional not {}-dimensional.".format(
                    n_dim
                )
            )
        self._X = X


class DataLoaderh5:
    def __init__(self,
                 attr,
                 parent,
                 h5_filepath=None,
                 attr_map=None):
        self.attr = attr
        self.parent = parent
        self.h5_filepath = h5_filepath
        self.attr_map = attr_map
        self.shape = 0
        with h5py.File(self.h5_filepath, "r") as h5file:
            for group_name, group in h5file["Datasets"].items():
                self.shape += group["X"].shape[0]
        self.shape = (self.shape, None)

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, idx: Union[List, np.ndarray]):
        if np.array(idx).dtype == bool:
            idx = np.arange(len(self.attr_map))[idx]
        data = self.load_data(idx)
        return np.vstack(data)

    def load_data(self, indices, map_data=True):
        datasets_to_indices = defaultdict(list)
        data = []
        for i in np.atleast_1d(indices):
            dset, i = self.attr_map[i]
            datasets_to_indices[dset].append(i)
        with h5py.File(self.h5_filepath, "r") as h5_file:
            datasets = h5_file["Datasets"]
            for ds_specifier, indices in datasets_to_indices.items():
                group = datasets[ds_specifier]
                loaded_data = group[self.attr][indices]
                col_indices, mappable_gene_ind = self.parent.dataset_to_genemap_cache[ds_specifier]
                if map_data:
                    loaded_data = self.parent.map_data(
                        loaded_data,
                        mappable_genes_indices=mappable_gene_ind,
                        col_indices=col_indices
                    )
                data.append(loaded_data)
        return np.vstack(data)


class DataLoaderLoom:
    def __init__(self,
                 loom_filepath=None, ):
        self.loom_filepath = loom_filepath
        with loompy.connect(self.loom_filepath) as loom_file:
            s = loom_file.shape
            self.shape = (s[1], s[0])
            self.dtype = loom_file[0, 0].dtype

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, idx: Union[List, np.ndarray]):
        with loompy.connect(self.loom_filepath) as loom_file:
            return loom_file[:, idx].transpose()
