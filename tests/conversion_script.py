from scvi.dataset.dataset10X import available_datasets, Dataset10X
from scvi.dataset.cortex import CortexDataset
from scvi.dataset.pbmc import PbmcDataset
from scvi.dataset.hemato import HematoDataset
from scvi.dataset.brain_large import BrainLargeDataset
from scvi.dataset.cite_seq import CbmcDataset
from scvi.dataset.dataset_hdf import convert_to_hdf5, HDF5Dataset


if __name__ == '__main__':
    avail_dsets = [(file, Dataset10X) for group in available_datasets.values() for file in group]
    # avail_dsets = []
    avail_dsets.extend(
        (elem for elem in zip([None]*5, (CortexDataset, PbmcDataset, HematoDataset, BrainLargeDataset, CbmcDataset)))
    )
    #convert_to_hdf5(avail_dsets, "/Users/b260-admin/Documents/GitHub/scVI/tests/data", "all")
    x = HDF5Dataset('"/Users/b260-admin/Documents/GitHub/scVI/tests/data/', False, False)
    x[2]
