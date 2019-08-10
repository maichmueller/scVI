from scvi.dataset.dataset10X import available_datasets, Dataset10X
from scvi.dataset.cortex import CortexDataset
from scvi.dataset.pbmc import PbmcDataset
from scvi.dataset.brain_large import BrainLargeDataset
from scvi.dataset.cite_seq import CbmcDataset
from scvi.dataset.union import UnionDataset


if __name__ == '__main__':
    available_datasets = [
            # "fresh_68k_pbmc_donor_a",
            # "frozen_pbmc_donor_a",
            # "frozen_pbmc_donor_b",
            # "frozen_pbmc_donor_c",
            # "pbmc8k",
            # "pbmc4k",
            "t_3k",
            "t_4k",
            # "pbmc_1k_protein_v3",
            # "pbmc_10k_protein_v3",
            # "malt_10k_protein_v3",
            # "pbmc_1k_v2",
            # "pbmc_1k_v3",
            # "pbmc_10k_v3"
    ]
    available_datasets = [(el, Dataset10X) for el in available_datasets]
    # available_datasets.extend(
    #     (elem for elem in zip([None]*3, (CortexDataset, PbmcDataset, CbmcDataset)))
    # )
    union_dataset = UnionDataset("./data", map_save_fname="human_data_map", data_save_fname="human_data_union")
    # union_dataset = UnionDataset("./tests/data", map_fname="ensembl_human_genes_proteincoding", data_save_fname="human_data_union")
    # union_dataset.build_mapping([elem[0] for elem in available_datasets], [elem[1] for elem in available_datasets])
    union_dataset.concat_union_into_hdf5([elem[0] for elem in available_datasets], [elem[1] for elem in available_datasets])
