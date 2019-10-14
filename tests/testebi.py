from Eval_basis import *
import pandas as pd
from tqdm import tqdm
from scipy import sparse
from scvi.dataset import EbiData, MouseAtlas, UnionDataset, AnnDatasetFromAnnData
import scanpy as sc

np.random.seed(1)


data_full = EbiData("./data", experiment="E-ENAD-15")
data_big_mapped = UnionDataset("./data", gene_map_load_filename="gene_maps/ensembl_mouse_genes-proteincoding", low_memory=False)
data_big_mapped.join_datasets(data_source="memory", data_target="memory", gene_datasets= [data_full])
data_big_mapped.filter_cell_types(np.array([ct for ct in data_big_mapped.cell_types if ct != "not available"]))
agg = data_full.obs.groupby(["Sample Characteristic[organism part]", "cell_types"]).size()
with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(agg.sort_index())
def do_procedure(dataset, ebi_data, model_filename, tsne_filename):
    n_epochs = 100
    colors = None

    trainer = train_vae(dataset, "./data", f"../trained_models/{model_filename}", n_epochs=n_epochs)

    dot_size = (mpl.rcParams['lines.markersize'] ** 2.0)

    _ = plot_tsne(trainer, trainer.model, ebi_data,
                  f"./plots/tissue_wise/{tsne_filename}",
                  image_datatype="pdf", colors=colors, s=dot_size, edgecolors='black')
complete_mouse = UnionDataset("./data",
                              gene_map_load_filename="gene_maps/ensembl_mouse_genes-proteincoding",
                              data_load_filename="mouse_data_all.loom",
                              low_memory=True)
n_epochs = 100
colors = None
for tissue in np.unique(agg.index.get_level_values(0)):
    print("Training VAE for tissue ", tissue)
    cutout_cts = agg[tissue].index.values
    if "not available" in cutout_cts:
        cutout_cts = np.delete(cutout_cts, np.where(cutout_cts == "not available")[0])

    rem_ct = data_big_mapped.cell_types[~np.isin(data_big_mapped.cell_types, cutout_cts)]
    data_small = copy.deepcopy(data_big_mapped)
    data_big = copy.deepcopy(data_big_mapped)

    data_small.filter_cell_types(cutout_cts)
    data_big.filter_cell_types(rem_ct)

    do_procedure(complete_mouse, data_small, "mouse_data_full", f"{tissue}_in_mouse_data_full_tsne")

    do_procedure(complete_mouse, data_small, "ebi_data_full", f"{tissue}_in_ebi_data_full_tsne")

    do_procedure(complete_mouse, data_small, "mouse_atlas", f"{tissue}_in_mouse_atlas_tsne")

    do_procedure(complete_mouse, data_small, "mouse_muris_senis", f"{tissue}_in_mouse_muris_senis_tsne")
