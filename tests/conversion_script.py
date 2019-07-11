from scvi.dataset.dataset10X import available_datasets, Dataset10X
from scvi.dataset.cortex import CortexDataset
from scvi.dataset.pbmc import PbmcDataset
from scvi.dataset.hemato import HematoDataset
from scvi.dataset.brain_large import BrainLargeDataset
from scvi.dataset.cite_seq import CbmcDataset
from scvi.dataset.union import IndepUnionDataset


if __name__ == '__main__':
    avail_dsets = [(file, Dataset10X) for group in available_datasets.values() for file in group]
    # avail_dsets = []
    avail_dsets.extend(
        (elem for elem in zip([None]*5, (CortexDataset, PbmcDataset, HematoDataset, BrainLargeDataset, CbmcDataset)))
    )
    union_dataset = IndepUnionDataset("./data", save_mapping_fname="all_data", data_savename="complete_data_union")
    union_dataset.build_mapping([elem[0] for elem in avail_dsets], [elem[1] for elem in avail_dsets])
