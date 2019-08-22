from scvi.dataset import EbiData, MouseAtlas, UnionDataset
from Eval_basis import *

# ebi_1 = EbiData("./data", 'E-MTAB-6946', result_file='raw')
#
# ebi_2 = EbiData("./data", 'E-MTAB-7320', result_file='raw')
# #
# # ebi_3 = EbiData("./data")
#
# fpaths_and_fnames = {'data': "./data/mouse_atlas/cleaned_data_sparse.npz",
#                      'cell': "./data/mouse_atlas/cell_annotation.csv",
#                      'gene': "./data/mouse_atlas/gene_annotation.csv",
#                      'pheno': "./data/mouse_atlas/phenotype_data.csv"}
# mouse = MouseAtlas("./data/mouse_atlas",
#                    fpaths_and_fnames,
#                    True,
#                    False)
complete_mouse = UnionDataset("./data",
                              gene_map_load_filename="gene_maps/ensembl_mouse_genes-proteincoding",
                              data_load_filename="mouse_data_all",
                              low_memory=True)
# joined = UnionDataset("./data", gene_map_load_filename="gene_maps/ensembl_mouse_genes-proteincoding", low_memory=True)
#
# joined.join_datasets(data_source="memory", data_target="memory", gene_datasets=[mouse])

n_epochs = 100
colors=None

print("Training VAE")

trainer_big = train_vae(complete_mouse, "./data", f"max_data_model", n_epochs=n_epochs)
# trainer_small = train_vae(data_small, "./data", f"small_{tissue}_data_portion", n_epochs=n_epochs)
dot_size = (mpl.rcParams['lines.markersize'] ** 2.0)

posterior_big = plot_tsne(trainer_big, trainer_big.model, complete_mouse, f"./max_data_model_onlymouse",
                          colors=colors, s=dot_size, edgecolors='black')
# posterior_small = plot_tsne(trainer_small, trainer_small.model, data_small, f"./plots/small_{tissue}_data_portion",
#                             colors=colors, s=dot_size, edgecolors='black')
# posterior_small_in_big = plot_tsne(trainer_big, trainer_big.model, data_small, f"./plots/small_{tissue}_data_portion_in_big",
#                                    colors=colors, s=dot_size, edgecolors='black')


