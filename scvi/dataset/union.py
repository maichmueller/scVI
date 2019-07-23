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
import functools
import sys
from tqdm import tqdm
import re

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

        self.save_path = save_path
        self.line_offsets = None

        self.map_fname = map_fname
        self.map_save_fname = map_save_fname
        self.data_fname = data_fname
        self.data_save_fname = data_save_fname
        self.low_memory = low_memory
        self.datasets_used = None

        self.X_filepath = None
        self.gene_names_filepath = None
        self.local_means_filepath = None
        self.local_vars_filepath = None
        self.batch_indices_filepath = None
        self.labels_filepath = None
        self.X_metadata = None

        if map_fname is not None:
            self.gene_map = pd.read_csv(
                os.path.join(self.save_path, map_fname + ".csv"),
                header=None,
                index_col=0
            )
            self.gene_map = self.gene_map.loc[:, self.gene_map.columns[0]]
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

    def collate_fn_base(self, attributes_and_types, indices):
        indices = np.asarray(indices)
        indices.sort()

        data_and_files = []
        try:
            gene_names_file = self._read_nonuniform_csv(self.gene_names_filepath, with_class_sep=True)[0]
            gene_names_file.index = pd.MultiIndex.from_tuples(gene_names_file.index)
            for attr, dtype in attributes_and_types.items():
                try:
                    data_and_files.append((open(getattr(self, f"{attr}_filepath"), "rb"),
                                           [],
                                           attr,
                                           dtype))
                except OSError:
                    pass

            for i in indices:
                ds_class, ds_fname, X_off, lm_off, lv_off = self.X_metadata.loc[i]
                for file, sample_container, attr, dtype in data_and_files:

                    if attr == "X":
                        file.seek(X_off)
                        read_line = np.asarray(list(file.readline().decode().strip().split(",")))
                        read_line = self.map_data(read_line,
                                                  gene_names=gene_names_file.loc[(ds_class, ds_fname)]
                                                  )
                    elif attr == "local_mean":
                        file.seek(lm_off)
                        read_line = np.asarray(list(file.readline().decode().strip().split(",")))
                        read_line = read_line.astype(dtype).reshape(1, -1)

                    elif attr == "local_var":
                        file.seek(lv_off)
                        read_line = np.asarray(list(file.readline().decode().strip().split(",")))
                    elif attr in ["batch_indices", "labels"]:
                        (read_class, read_fname), read_line = list(map(lambda el: el.split(","),
                                                                       file.readline().decode().strip().split(":")))
                        read_line = np.asarray(read_line).astype(dtype).reshape(1, -1)

                    else:
                        raise ValueError(f"Attribute {attr} not supported by this dataset.")

                    sample_container.append(read_line)

            batch = []
            for file, container, _, dtype in data_and_files:
                batch.append(torch.from_numpy(np.vstack(container)))
                file.close()
            return tuple(batch)

        except Exception as e:
            if data_and_files:
                for content in data_and_files:
                    content[0].close()
            raise e

    @staticmethod
    def _read_nonuniform_csv(fname, with_class_sep=False, dtype=str):
        if with_class_sep:
            data = dict()
            with open(fname, 'rb') as file:
                for line in file:
                    linelist = list(map(lambda x: x.split(","), line.decode().split(":")))
                    (ds_class, ds_fname), line = linelist
                    data[ds_class, ds_fname] = np.asarray(line).astype(str).reshape(1, -1)
            data = pd.DataFrame.from_dict(data, orient="index")
        else:
            data = []
            with open(fname, 'rb') as file:
                for line in file:
                    line = np.asarray(line.decode().split(",")).astype(dtype).reshape(1, -1)
                    data.append(line)
            data = np.concatenate(data, axis=0)  # == np.vstack
        return data

    def set_filepaths(self, save_path, data_fname):
        self.X_filepath = os.path.join(save_path, data_fname + '_X.nucsv')
        self.gene_names_filepath = os.path.join(save_path, data_fname + '_gene_names.nucsv')
        self.local_means_filepath = os.path.join(save_path, data_fname + '_local_means.nucsv')
        self.local_vars_filepath = os.path.join(save_path, data_fname + '_local_vars.nucsv')
        self.batch_indices_filepath = os.path.join(save_path, data_fname + '_batch_indices.nucsv')
        self.labels_filepath = os.path.join(save_path, data_fname + '_labels.nucsv')

        self.X_metadata = pd.read_csv(os.path.join(save_path, data_fname + "_metadata.csv"),
                                      header=0, index_col=0)
        self.X_len = len(self.X_metadata)

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
                 gene_names,
                 *args,
                 **kwargs
                 ):

        data_out = np.zeros((len(data), self.gene_names_len), dtype=float)
        mappable_genes_indices = np.isin(gene_names, self.gene_map.index)
        try:
            col_indices = kwargs["col_indices"]
        except KeyError:
            mappable_genes = gene_names[mappable_genes_indices]
            col_indices = self.gene_map[mappable_genes]

        data_out[:, col_indices] = data[mappable_genes_indices]
        return data_out

    def _getrow(self, data, idx_start, idx_end):
        if isinstance(data, np.ndarray):
            d_out = data[idx_start:idx_end]
            return d_out.reshape(min(idx_end - idx_start, d_out.shape[0]), -1)
        else:
            return [data.getrow(i)
                    for i in range(idx_start, min(idx_end, data.shape[0]))]

    @staticmethod
    def _toarray(data):
        return data.toarray()

    def _mp_ds_to_file(self,
                       lock,
                       dataset_class, dataset_fname, dataset_arg,
                       out_fname, nr_rows,
                       line_count, X_offset, lm_offset, lv_offset,
                       n_batch_offset, n_labels_offset,
                       shared_batches):

        dataset_class_str = re.search(class_re_pattern, str(dataset_class)).group()

        dataset, _, _ = self._load_dataset(dataset_fname, dataset_class, dataset_arg, True)

        # grab the necessary data parts:
        # aside from the data itself (X), the gene_names, local means, local_vars, batch_indices and labels
        # there are no guaranteed attributes of each dataset. Thus for now these will be the ones we use
        data = dataset.X
        gene_names = dataset.gene_names.flatten()
        local_means = dataset.local_means.flatten()
        local_vars = dataset.local_vars.flatten()
        batch_indices = dataset.batch_indices.flatten()
        labels = dataset.labels.flatten()

        len_data = len(dataset)
        n_batches = dataset.n_batches
        n_labels = dataset.n_labels

        print(f"Writing dataset {dataset_class_str, dataset_fname} of length {len_data} to file.")
        sys.stdout.flush()

        proc_range = list(range(0, data.shape[0], nr_rows))
        with Pool(2) as pool:
            queue = tqdm(pool.starmap(self._getrow, [(data, row_start, row_start + nr_rows)
                                                     for row_start in proc_range]))

            lock.acquire()
            queue.set_description(f"Dataset {dataset_class_str, dataset_fname} in iterations of {nr_rows} rows")
            for idx, rows in enumerate(queue):
                if not isinstance(rows, np.ndarray):
                    rows = np.concatenate([row.toarray() for row in rows], axis=0)
                    args = (out_fname, rows, local_means, local_vars, gene_names,
                            batch_indices, labels, dataset_class_str, dataset_fname, line_count,
                            X_offset, lm_offset, lv_offset, n_batches, n_batch_offset, n_labels,
                            n_labels_offset, shared_batches)

                    Process(target=self._write_data, args=args).start()
        lock.release()

    def _write_data(self,
                    out_fname,
                    rows, local_means, local_vars, gene_names, batch_indices, labels,
                    dataset_class_str, dataset_fname,
                    line_count, X_offset, lm_offset, lv_offset,
                    n_batches, n_batch_offset, n_labels, n_labels_offset,
                    shared_batches=False):

        file_open_mode = 'ab'  # store as binary

        with open(self.save_path + '/' + out_fname + '_X.nucsv', file_open_mode) as d, \
            open(self.save_path + '/' + out_fname + '_metadata.csv', "a") as d_meta, \
            open(self.save_path + '/' + out_fname + '_gene_names.nucsv', file_open_mode) as gn, \
            open(self.save_path + '/' + out_fname + '_local_means.nucsv', file_open_mode) as lm, \
            open(self.save_path + '/' + out_fname + '_local_vars.nucsv', file_open_mode) as lv, \
            open(self.save_path + '/' + out_fname + '_batch_indices.nucsv', file_open_mode) as bi, \
            open(self.save_path + '/' + out_fname + '_labels.nucsv', file_open_mode) as l:

            for row, local_mean, local_var in zip(rows, local_means.flatten(), local_vars.flatten()):

                X_line = (",".join([str(entry) for entry in row]) + '\n').encode()
                lm_line = (str(local_mean) + '\n').encode()
                lv_line = (str(local_var) + '\n').encode()
                d.write(X_line)
                lm.write(lm_line)
                lv.write(lv_line)
                d_meta.write(f"{line_count.value},{dataset_class_str},{dataset_fname},"
                             f"{X_offset.value},{lm_offset.value},{lv_offset.value}\n")
                X_offset.value += len(X_line)
                lm_offset.value += len(lm_line)
                lv_offset.value += len(lv_line)
                line_count.value += 1

            gn.write(f"{dataset_class_str},{dataset_fname}:".encode())
            bi.write(f"{dataset_class_str},{dataset_fname}:".encode())
            l.write(f"{dataset_class_str},{dataset_fname}:".encode())

            gn.write((",".join([str(entry) for entry in gene_names]) + '\n').encode())

            batch_indices += n_batch_offset.value
            n_batch_offset.value += n_batches if not shared_batches else 0
            bi.write((",".join([str(entry) for entry in batch_indices]) + '\n').encode())

            labels += labels + n_labels_offset.value
            n_labels_offset.value += n_labels
            l.write((",".join([f"{entry}" for entry in labels]) + '\n').encode())

    def concat_to_nucsv(self,
                        dataset_names,
                        dataset_classes,
                        dataset_args=None,
                        out_fname=None,
                        nr_rows=100,
                        n_cpu=min(cpu_count() // 4, 1),
                        **kwargs
                        ):
        if self.X is not None:
            print(f'Data already built/loaded (potentially from file {self.map_fname}).')
            return
        if not self.low_memory:
            print(f"Low memory setting is '{self.low_memory}'. Exiting")
            return
        if dataset_args is None:
            dataset_args = [dataset_args] * len(dataset_names)

        if out_fname is None:
            out_fname = self.data_save_fname

        try:
            shared_batches = kwargs.pop("shared_batches")
        except KeyError:
            shared_batches = False

        lock = Lock()

        X_offset = Value("i", 0)
        lm_offset = Value("i", 0)
        lv_offset = Value("i", 0)
        line_count = Value("i", 0)
        n_batch_offset = Value("i", 0)
        n_labels_offset = Value("i", 0)

        # Build the group files for the dataset, under which the data is going to be stored
        # We will store the data in the following scheme:
        # out_fname_X.nucsv
        # out_fname_metadata.csv
        # out_fname_gene_names.nucsv
        # out_fname_local_means.nucsv
        # out_fname_local_vars.nucsv
        # out_fname_batch_indices.nucsv
        # out_fname_labels.nucsv

        with open(os.path.join(self.save_path, out_fname + '_X.nucsv'), "wb") as d, \
            open(os.path.join(self.save_path, out_fname + '_metadata.csv'), "w") as d_meta, \
            open(os.path.join(self.save_path, out_fname + '_gene_names.nucsv'), "wb") as gn, \
            open(os.path.join(self.save_path, out_fname + '_local_means.nucsv'), "wb") as lm, \
            open(os.path.join(self.save_path, out_fname + '_local_vars.nucsv'), "wb") as lv, \
            open(os.path.join(self.save_path, out_fname + '_batch_indices.nucsv'), "wb") as bi, \
            open(os.path.join(self.save_path, out_fname + '_labels.nucsv'), "wb") as l:

            d_meta.write("line, dataset_class, dataset_filename, X_offset, lm_offset, lv_offset\n")

        with ThreadPoolExecutor(2) as pool:
            ds_queue = list(pool.submit(self._mp_ds_to_file,
                                        lock,
                                        dataset_class, dataset_fname, dataset_arg,
                                        out_fname,
                                        nr_rows,
                                        line_count, X_offset, lm_offset, lv_offset,
                                        n_batch_offset, n_labels_offset, shared_batches)
                            for dataset_fname, dataset_class, dataset_arg in
                            zip(dataset_names, dataset_classes, dataset_args)
                            )
            for future in as_completed(ds_queue):
                future.result()

        print(f"Conversion completed to files: \n"
              f"'{out_fname}_data.nucsv'\n"
              f"'{out_fname}_metadata.csv'\n"
              f"'{out_fname}_genenames.nucsv'\n"
              f"'{out_fname}_localmeans.nucsv'\n"
              f"'{out_fname}_localvars.nucsv'\n"
              f"'{out_fname}_batchindices.nucsv'\n"
              f"'{out_fname}_labels.nucsv'\n")

        self.set_filepaths(self.save_path, out_fname)
        return

    # def concat_to_nucsv(self,
    #                     dataset_names,
    #                     dataset_classes,
    #                     dataset_args=None,
    #                     out_fname=None,
    #                     write_mode="w",
    #                     nr_rows=100,
    #                     n_cpu=min(cpu_count() // 4, 1),
    #                     **kwargs
    #                     ):
    #     if self.X is not None:
    #         print(f'Data already built/loaded (potentially from file {self.map_fname}).')
    #         return
    #     if not self.low_memory:
    #         print(f"Low memory setting is '{self.low_memory}'. Exiting")
    #         return
    #     if dataset_args is None:
    #         dataset_args = [dataset_args] * len(dataset_names)
    #
    #     if out_fname is None:
    #         out_fname = self.data_save_fname
    #
    #     n_batch_offset = 0
    #     n_labels_offset = 0
    #     try:
    #         shared_batches = kwargs.pop("shared_batches")
    #     except KeyError:
    #         shared_batches = False
    #
    #     lock = Lock()
    #
    #     X_offset = 0
    #     lm_offset = 0
    #     lv_offset = 0
    #     line_count = 0
    #
    #     # Build the group files for the dataset, under which the data is going to be stored
    #     # We will store the data in the following scheme:
    #     # out_fname_X.nucsv
    #     # out_fname_metadata.csv
    #     # out_fname_gene_names.nucsv
    #     # out_fname_local_means.nucsv
    #     # out_fname_local_vars.nucsv
    #     # out_fname_batch_indices.nucsv
    #     # out_fname_labels.nucsv
    #
    #     file_open_mode = write_mode + 'b'  # store as binary
    #
    #     with open(self.save_path + '/' + out_fname + '_X.nucsv', file_open_mode) as d, \
    #         open(self.save_path + '/' + out_fname + '_metadata.csv', write_mode) as d_meta, \
    #         open(self.save_path + '/' + out_fname + '_gene_names.nucsv', file_open_mode) as gn, \
    #         open(self.save_path + '/' + out_fname + '_local_means.nucsv', file_open_mode) as lm, \
    #         open(self.save_path + '/' + out_fname + '_local_vars.nucsv', file_open_mode) as lv, \
    #         open(self.save_path + '/' + out_fname + '_batch_indices.nucsv', file_open_mode) as bi, \
    #         open(self.save_path + '/' + out_fname + '_labels.nucsv', file_open_mode) as l:
    #
    #         d_meta.write("line, offset, dataset_class, dataset_filename\n")
    #         with ThreadPoolExecutor(2) as pool:
    #             ds_queue = list(pool.submit(self._load_dataset,
    #                                         dataset_fname,
    #                                         dataset_class,
    #                                         dataset_arg)
    #                             for dataset_fname, dataset_class, dataset_arg in
    #                             zip(dataset_names, dataset_classes, dataset_args)
    #                             )
    #             print(len(ds_queue))
    #             for future in as_completed(ds_queue):
    #             # for dataset_fname, dataset_class, dataset_arg in zip(dataset_names, dataset_classes, dataset_args):
    #                 dataset, dataset_class, dataset_fname = future.result()
    #                 dataset_class_str = re.search(class_re_pattern, str(dataset_class)).group()
    #
    #                 # if dataset_fname is None:
    #                 #     if dataset_arg is not None:
    #                 #         dataset = dataset_class(dataset_arg, save_path=self.save_path)
    #                 #     else:
    #                 #         dataset = dataset_class(save_path=self.save_path)
    #                 # else:
    #                 #     if dataset_arg is not None:
    #                 #         dataset = dataset_class(dataset_fname, dataset_arg, save_path=self.save_path)
    #                 #     else:
    #                 #         dataset = dataset_class(dataset_fname, save_path=self.save_path)
    #
    #                 if dataset.gene_names is None:
    #                     warnings.warn(f"Dataset {dataset_class_str}, {dataset_fname} doesn't have gene_names attribute."
    #                                   f"Skipping it.")
    #                     continue
    #
    #                 # grab the necessary data parts:
    #                 # aside from the data itself (X), the gene_names, local means, local_vars, batch_indices and labels
    #                 # there are no guaranteed attributes of each dataset. Thus for now these will be the ones we use
    #                 data = dataset.X
    #                 self.dataset_holder = data
    #                 gene_names = dataset.gene_names.flatten()
    #                 local_means = dataset.local_means.flatten()
    #                 local_vars = dataset.local_vars.flatten()
    #                 batch_indices = dataset.batch_indices.flatten()
    #                 labels = dataset.labels.flatten()
    #
    #                 len_data = len(dataset)
    #
    #                 print(f"Writing dataset {dataset_class_str, dataset_fname} of length {len_data} to file.")
    #                 sys.stdout.flush()
    #
    #                 # pbar = tqdm(range(0, data.shape[0], nr_rows))
    #                 # pbar.set_description(f"Iterations of {nr_rows} rows")
    #                 # for row_start in pbar:
    #                 #     rows = self._getrow(row_start, row_start + nr_rows)
    #                 #     if not isinstance(rows, np.ndarray):
    #                 #         rows = np.concatenate([row.toarray() for row in rows], axis=0)
    #                 #
    #                 #     rows = rows.astype(int)
    #                 #     for row in rows:
    #                 #         line = (",".join([str(entry) for entry in row]) + '\n').encode()
    #                 #         d.write(line)
    #                 #         d_meta.write(f"{line_count}, {offset}\n")
    #                 #         offset += len(line)
    #                 #         line_count += 1
    #
    #                 proc_range = list(range(0, data.shape[0], nr_rows))
    #                 with Pool(2) as pool:
    #                     queue = tqdm(pool.starmap(self._getrow, [(row_start, row_start + nr_rows)
    #                                                              for row_start in proc_range]))
    #                     queue.set_description(f"Iterations of {nr_rows} rows")
    #                     for idx, rows in enumerate(queue):
    #                         if not isinstance(rows, np.ndarray):
    #                             rows = np.concatenate([row.toarray() for row in rows], axis=0)
    #                         for row, local_mean, local_var in zip(rows, local_means.flatten(), local_vars.flatten()):
    #                             X_line = (",".join([str(entry) for entry in row]) + '\n').encode()
    #                             lm_line = (str(local_mean) + '\n').encode()
    #                             lv_line = (str(local_var) + '\n').encode()
    #                             d.write(X_line)
    #                             lm.write(lm_line)
    #                             lv.write(lv_line)
    #                             d_meta.write(f"{line_count},{dataset_class_str},{dataset_fname},"
    #                                          f"{X_offset},{lm_offset},{lv_offset}\n")
    #                             X_offset += len(X_line)
    #                             lm_offset += len(lm_line)
    #                             lv_offset += len(lv_line)
    #                             line_count += 1
    #
    #                 gn.write(f"{dataset_class_str},{dataset_fname}:".encode())
    #                 bi.write(f"{dataset_class_str},{dataset_fname}:".encode())
    #                 l.write(f"{dataset_class_str},{dataset_fname}:".encode())
    #
    #                 gn.write((",".join([str(entry) for entry in gene_names]) + '\n').encode())
    #
    #                 batch_indices += n_batch_offset
    #                 n_batch_offset += dataset.n_batches if not shared_batches else 0
    #                 bi.write((",".join([str(entry) for entry in batch_indices]) + '\n').encode())
    #
    #                 labels += labels + n_labels_offset
    #                 n_labels_offset += dataset.n_labels
    #                 l.write((",".join([f"{entry}" for entry in labels]) + '\n').encode())
    #
    #     print(f"Conversion completed to files: \n"
    #           f"'{out_fname}_data.nucsv'\n"
    #           f"'{out_fname}_metadata.csv'\n"
    #           f"'{out_fname}_genenames.nucsv'\n"
    #           f"'{out_fname}_localmeans.nucsv'\n"
    #           f"'{out_fname}_localvars.nucsv'\n"
    #           f"'{out_fname}_batchindices.nucsv'\n"
    #           f"'{out_fname}_labels.nucsv'\n")
    #
    #     self.set_filepaths(self.save_path, out_fname)
    #     return

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
