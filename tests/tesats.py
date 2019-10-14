from scvi.dataset import EbiData, MouseAtlas, UnionDataset, AnnDatasetFromAnnData
from Eval_basis import *
import scanpy as sc
import pandas as pd
import scipy.sparse as sparse
from tqdm import tqdm
import urllib


def do_procedure(dataset, ebi_data, model_filename, tsne_filename):
    n_epochs = 100
    colors = None

    print("Training VAE")

    trainer = train_vae(dataset, "./data", f"../trained_models/{model_filename}", n_epochs=n_epochs)

    dot_size = (mpl.rcParams['lines.markersize'] ** 2.0)

    _ = plot_tsne(trainer, trainer.model, dataset, f"./plots/{tsne_filename}", image_datatype="pdf",
                  colors=colors, s=dot_size, edgecolors='black')

    _ = plot_tsne(trainer, trainer.model, ebi_data,
                  f"./plots/ebi_annotated_data_in_{tsne_filename}",
                  image_datatype="pdf", colors=colors, s=dot_size, edgecolors='black')


if __name__ == '__main__':

    ebi = UnionDataset("./data",
                       gene_map_load_filename="gene_maps/ensembl_mouse_genes-proteincoding",
                       low_memory=False)
    ebi.join_datasets(data_source="memory",
                      data_target="memory",
                      gene_datasets=[EbiData("./data")])

    complete_mouse = UnionDataset("./data",
                                  gene_map_load_filename="gene_maps/ensembl_mouse_genes-proteincoding",
                                  data_load_filename="mouse_data_all.loom",
                                  low_memory=False)

    do_procedure(complete_mouse, ebi, "mouse_data_full", "mouse_data_full_tsne")

    del complete_mouse

    ####################################

    ebi_datasets_ids = ['E-MTAB-6946', 'E-MTAB-7320',
                        "E-MTAB-7417", "E-MTAB-6173", "E-GEOD-81682",
                        "E-ENAD-13", "E-GEOD-99235", "E-GEOD-71585",
                        "E-GEOD-87631", "E-GEOD-90848", "E-MTAB-4547",
                        "E-MTAB-6970", "E-MTAB-5802", "E-MTAB-5661",
                        "E-GEOD-99058", "E-GEOD-106973", "E-MTAB-7365",
                        "E-MTAB-5553", "E-MTAB-6925", "E-MTAB-6976"]

    ebi_data = []
    for e_id in tqdm(ebi_datasets_ids, desc="Loading EBI datasets"):
        try:
            data = EbiData("./data", e_id)
        except urllib.error.HTTPError:
            data = EbiData("./data", e_id, result_file='raw')
        data.X = sparse.csr_matrix(data.X)
        ebi_data.append(data)

    ebi_list = UnionDataset("./data",
                            gene_map_load_filename="gene_maps/ensembl_mouse_genes-proteincoding",
                            low_memory=False)
    ebi_list.join_datasets(data_source="memory",
                           data_target="memory",
                           gene_datasets=ebi_list)
    ebi_data = None

    do_procedure(ebi_list, ebi, "ebi_data_full", "ebi_data_full_tsne")

    del ebi_list

    ##########################################

    mouse_union = UnionDataset("./data",
                               gene_map_load_filename="gene_maps/ensembl_mouse_genes-proteincoding",
                               low_memory=False)
    mouse_union.join_datasets(data_source="memory",
                              data_target="memory",
                              gene_datasets=[MouseAtlas("./data/mouse_atlas",
                                                        {'data': "./data/mouse_atlas/cleaned_data_sparse.npz",
                                                         'cell': "./data/mouse_atlas/cell_annotation.csv",
                                                         'gene': "./data/mouse_atlas/gene_annotation.csv",
                                                         'pheno': "./data/mouse_atlas/phenotype_data.csv"},
                                                        True,
                                                        False)])
    mouse_union.name = "Mouse Atlas"

    do_procedure(mouse_union, ebi, "mouse_atlas", "mouse_atlas_tsne")

    del mouse_union

    ########################################

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

        gns_conved = conv.reindex(np.char.lower(dset.gene_names))["ensembl"]
        if not isinstance(dset.X, np.ndarray):
            X = dset.X.toarray()
        else:
            X = dset.X
        mask = ~gns_conved.isnull()

        dset.gene_names = gns_conved[mask].values.astype(str)
        dset.X = X[:, mask]
        dset.cell_types = np.array([ct.replace("ï", "i") for ct in dset.cell_types])

        dsets.append(dset)

    mouse_muris_senis = UnionDataset("./data",
                                     gene_map_load_filename="gene_maps/ensembl_mouse_genes-proteincoding",
                                     low_memory=False)
    mouse_muris_senis.join_datasets(data_source="memory",
                                    data_target="memory",
                                    gene_datasets=dsets)
    mouse_muris_senis.name = "Tabula Muris Senis"
    mouse_muris_senis.cell_types = np.array([ct.replace("ï", "i") for ct in mouse_muris_senis.cell_types])

    del dsets

    do_procedure(mouse_muris_senis, ebi, "mouse_muris_senis", "mouse_muris_senis_tsne")
