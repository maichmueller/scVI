from scvi.dataset import Dataset10X, UnionDataset, MouseAtlas, EbiData, AnnDatasetFromAnnData
import scanpy as sc
import numpy as np
import os
import pandas as pd
from Eval_basis import *


def load_mouse_data():
    conv = pd.read_csv("./data/gene_maps/hugo_mouse_genes-proteincoding.csv", header=0, index_col=0)
    conv.index = conv.index.str.lower()

    data_path = os.path.join(("./data"))
    mouse_data_path = os.path.join(data_path, "mouse_data")
    dsets = []
    for file in os.listdir(f"{data_path}/mouse_data"):
        #     if "droplet" in file:
        dset = sc.read_h5ad(os.path.join(mouse_data_path, file))
        dset.obs.rename(columns={"cell_ontology_class": "cell_types"}, inplace=True)

        dset = AnnDatasetFromAnnData(dset)

        gns_conved = conv.reindex(np.char.upper(dset.gene_names))["ensembl"]
        if not isinstance(dset.X, np.ndarray):
            X = dset.X.toarray()
        else:
            X = dset.X
        mask = ~gns_conved.isnull()

        dset.gene_names = gns_conved[mask].values.astype(str)
        dset.X = X[:, mask]
        dset.cell_types = np.array([ct.replace("ï", "i") for ct in dset.cell_types])

        dsets.append(dset)
    return dsets


if __name__ == '__main__':
    # ebi1 = EbiData("./data")
    # ds2 = Dataset10X("neuron_1k_v3", save_path="./data")
    # # ebi2 = EbiData("./data", 'E-MTAB-7320', result_file='raw')
    #
    # union_dataset = UnionDataset("./data",
    #                              gene_map_load_filename="gene_maps/ensembl_mouse_genes-proteincoding")
    # union_dataset.join_datasets("memory", "loom", out_filename="test_loom_script.loom", gene_datasets=[ds2, ebi1])

    # #
    union_dataset = UnionDataset("./data",
                                 gene_map_load_filename="gene_maps/ensembl_mouse_genes-proteincoding",
                                 data_load_filename="test_loom_script.loom")

    dsets=load_mouse_data()
    mouse_muris_senis = UnionDataset("./data",
                                     gene_map_load_filename="gene_maps/ensembl_mouse_genes-proteincoding",
                                     low_memory=False)
    mouse_muris_senis.join_datasets(data_source="memory",
                                    data_target="memory",
                                    gene_datasets=dsets)
    mouse_muris_senis.name = "Tabula Muris Senis"
    # union_dataset.join_datasets("loom", "memory", in_filename="test_loom_script.loom")
    n_epochs = 100
    colors = None

    print("Training VAE")

    trainer = train_vae(union_dataset, "./data", f"test", n_epochs=n_epochs)
    # trainer_small = train_vae(data_small, "./data", f"small_{tissue}_data_portion", n_epochs=n_epochs)

