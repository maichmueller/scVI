from scvi.dataset.dataset10X import available_datasets, Dataset10X
from scvi.dataset.cortex import CortexDataset
from scvi.dataset.pbmc import PbmcDataset
from scvi.dataset.brain_large import BrainLargeDataset
from scvi.dataset.cite_seq import CbmcDataset
from scvi.dataset.union import UnionDataset
from Eval_basis import *
import os

if __name__ == '__main__':
    print(os.getcwd())
    union_dataset = UnionDataset("./data",
                                 gene_map_load_filename="gene_maps/ensembl_mouse_genes-proteincoding",
                                 data_load_filename="/icgc/dkfzlsdf/analysis/B260/projects/vae/data/mouse_data_all.loom",
                                 low_memory=True)
    # union_dataset.join_datasets("loom", "memory",
    #                             in_filename="/icgc/dkfzlsdf/analysis/B260/projects/vae/data/mouse_data_all.loom")

    train_vae(union_dataset, save_path="./data", model_savename="mouse_data_full_model", n_epochs=100)
