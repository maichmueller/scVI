from scvi.dataset.dataset10X import available_datasets, Dataset10X
from scvi.dataset.cortex import CortexDataset
from scvi.dataset.pbmc import PbmcDataset
from scvi.dataset.brain_large import BrainLargeDataset
from scvi.dataset.cite_seq import CbmcDataset
from scvi.dataset.union import UnionDataset


if __name__ == '__main__':
    avail_dsets = [(file, Dataset10X) for group in available_datasets.values() for file in group]
    # avail_dsets = []
    avail_dsets.extend(
        (elem for elem in zip([None]*5, (CortexDataset, PbmcDataset, BrainLargeDataset, CbmcDataset)))
    )
    union_dataset = UnionDataset("./data", map_fname="complete_datasets_map", data_save_fname="complete_data_union")
    union_dataset.build_mapping([elem[0] for elem in avail_dsets], [elem[1] for elem in avail_dsets])
    union_dataset.concat_to_fwf([elem[0] for elem in avail_dsets], [elem[1] for elem in avail_dsets])
