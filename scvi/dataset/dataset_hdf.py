import h5py
import numpy as np
from pathlib import Path
import torch
from torch.utils import data


class HDF5Dataset(data.Dataset):
    """Represents an abstract HDF5 dataset.

    Input params:
        file_path: Path to the folder containing the dataset (one or multiple HDF5 files).
        recursive: If True, searches for h5 files in subdirectories.
        load_data: If True, loads all the data immediately into RAM. Use this if
            the dataset fits into memory. Otherwise, leave this at false and
            the data will load lazily.
        data_cache_size: Number of HDF5 files that can be cached in the cache (default=3).
        transform: PyTorch transform to apply to every data instance (default=None).
    """

    def __init__(self, file_path, recursive, load_data, data_cache_size=3, transform=None):
        super().__init__()
        self.data_info = []
        self.data_cache = {}
        self.data_cache_size = data_cache_size
        self.transform = transform

        # Search for all h5 files
        p = Path(file_path)
        assert (p.is_dir())
        if recursive:
            files = sorted(p.glob('**/*.h5'))
        else:
            files = sorted(p.glob('*.h5'))
        if len(files) < 1:
            raise RuntimeError('No hdf5 datasets found')

        for h5dataset_fp in files:
            self._add_data_infos(str(h5dataset_fp.resolve()), load_data)

    def __getitem__(self, index):
        # get data
        d = self.get_data("data", index)

        gn = self.get_data("gene_names", index)
        gn = torch.from_numpy(gn)

        lm = self.get_data("local_means", index)
        lm = torch.from_numpy(lm)

        lv = self.get_data("local_vars", index)
        lv = torch.from_numpy(lv)

        bi = self.get_data("batch_indices", index)
        bi = torch.from_numpy(bi)

        l = self.get_data("labels", index)
        l = torch.from_numpy(l)

        if self.transform:
            d = self.transform(d, gn)
        else:
            d = torch.from_numpy(d)

        return d, gn, lm, lv, bi, l

    def __len__(self):
        return len(self.get_data_infos('data'))

    def _add_data_infos(self, file_path, load_data):
        with h5py.File(file_path) as h5_file:
            # Walk through all groups, extracting datasets
            for gname, group in h5_file.items():
                for dname, ds in group.items():
                    # if data is not loaded its cache index is -1
                    idx = -1
                    if load_data:
                        # add data to the data cache
                        idx = self._add_to_cache(ds.value, file_path)

                    # type is derived from the name of the dataset; we expect the dataset
                    # name to have a name such as 'data' or 'gene_names' to identify its type
                    # we also store the shape of the data in case we need it
                    self.data_info.append(
                        {'file_path': file_path, 'type': dname, 'shape': ds.value.shape, 'cache_idx': idx})

    def _load_data(self, file_path):
        """Load data to the cache given the file
        path and update the cache index in the
        data_info structure.
        """
        with h5py.File(file_path) as h5_file:
            for gname, group in h5_file.items():
                for dname, ds in group.items():
                    # add data to the data cache and retrieve
                    # the cache index
                    idx = self._add_to_cache(ds.value, file_path)

                    # find the beginning index of the hdf5 file we are looking for
                    file_idx = next(i for i, v in enumerate(self.data_info) if v['file_path'] == file_path)

                    # the data info should have the same index since we loaded it in the same way
                    self.data_info[file_idx + idx]['cache_idx'] = idx

        # remove an element from data cache if size was exceeded
        if len(self.data_cache) > self.data_cache_size:
            # remove one item from the cache at random
            removal_keys = list(self.data_cache)
            removal_keys.remove(file_path)
            self.data_cache.pop(removal_keys[0])
            # remove invalid cache_idx
            self.data_info = [
                {'file_path': di['file_path'], 'type': di['type'], 'shape': di['shape'], 'cache_idx': -1}
                if di['file_path'] == removal_keys[0] else di
                for di in self.data_info
            ]

    def _add_to_cache(self, data, file_path):
        """Adds data to the cache and returns its index. There is one cache
        list for every file_path, containing all datasets in that file.
        """
        if file_path not in self.data_cache:
            self.data_cache[file_path] = [data]
        else:
            self.data_cache[file_path].append(data)
        return len(self.data_cache[file_path]) - 1

    def get_data_infos(self, type):
        """Get data infos belonging to a certain type of data.
        """
        data_info_type = [di for di in self.data_info if di['type'] == type]
        return data_info_type

    def get_data(self, type, i):
        """Call this function anytime you want to access a chunk of data from the
            dataset. This will make sure that the data is loaded in case it is
            not part of the data cache.
        """
        fp = self.get_data_infos(type)[i]['file_path']
        if fp not in self.data_cache:
            self._load_data(fp)

        # get new cache_idx assigned by _load_data_info
        cache_idx = self.get_data_infos(type)[i]['cache_idx']
        return self.data_cache[fp][cache_idx]


def convert_to_hdf5(dataset_fnames_classes, save_path, out_fname):
    used_datasets = []
    with h5py.File(f'{save_path}/{out_fname}.hdf5', 'w') as hdf:
        for dataset_fname, dataset_class in dataset_fnames_classes:
            if dataset_fname is None:
                dataset = dataset_class(save_path=save_path)
            else:
                dataset = dataset_class(dataset_fname, save_path=save_path)

            if not hasattr(dataset, "gene_names"):
                continue

            # grab the necessary data parts:
            # aside from the data itself (X), the gene_names, local means, local_vars, batch_indices and labels
            # there are no guaranteed attributes of each dataset. Thus for now these will be the ones we
            # work with
            if not dataset.dense:
                dataset.X = dataset.X.toarray()
            gene_names = dataset.gene_names
            local_means = dataset.local_means
            local_vars = dataset.local_vars
            batch_indices = dataset.batch_indices
            labels = dataset.labels

            print(f"Writing dataset {dataset_class, dataset_fname} to hdf5 file.")

            # Build the group for the dataset, under which the data is going to be stored
            # We will store the above mentioned data in the following scheme (as this corresponds
            # to the hdf5 dataset class):
            # --1st DS NAME and CLASS
            # ------ data
            # ------ gene_names
            # ------ local_means
            # ------ local_vars
            # ------ batch_indices
            # ------ labels
            # --2nd DS NAME and CLASS
            # ------ ...
            # ...
            dataset_hdf5_g = hdf.create_group(f"{dataset_fname}_{dataset_class}")

            dataset_hdf5 = dataset_hdf5_g.create_dataset("data", dataset.X.shape,
                                                         dtype=dataset.X.dtype)
            dataset_hdf5[:] = dataset.X

            gene_names_hdf5 = dataset_hdf5_g.create_dataset("gene_names", gene_names.shape,
                                                            dtype=h5py.special_dtype(vlen=str))
            gene_names_hdf5[:] = gene_names

            local_means_hdf5 = dataset_hdf5_g.create_dataset("local_means", local_means.shape,
                                                             dtype=local_means.dtype)
            local_means_hdf5[:] = local_means

            local_vars_hdf5 = dataset_hdf5_g.create_dataset("local_vars", local_vars.shape,
                                                            dtype=local_vars.dtype)
            local_vars_hdf5[:] = local_vars

            batch_indices_hdf5 = dataset_hdf5_g.create_dataset("batch_indices", batch_indices.shape,
                                                               dtype=batch_indices.dtype)
            batch_indices_hdf5[:] = batch_indices

            labels_hdf5 = dataset_hdf5_g.create_dataset("labels", labels.shape,
                                                        dtype=labels.dtype)
            labels_hdf5[:] = labels

            if hasattr(dataset, "cell_types"):
                cell_types = dataset.cell_types
                cell_types_hdf5 = dataset_hdf5_g.create_dataset("cell_types", cell_types.shape,
                                                                dtype=h5py.special_dtype(vlen=str))
                cell_types_hdf5[:] = cell_types

    print(f"conversion completed to file '{out_fname}.hdf5'")





