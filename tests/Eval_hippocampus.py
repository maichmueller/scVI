from Eval_basis import *
import pandas as pd


if __name__ == '__main__':

    np.random.seed(1)

    data_full = EbiData("./data", experiment="E-ENAD-15")
    data_big_mapped = UnionDataset("./data", map_fname="ensembl_mouse_genes-proteincoding", low_memory=False)
    data_big_mapped.union_from_memory([data_full])
    data_big_mapped.filter_cell_types(np.array([ct for ct in data_big_mapped.cell_types if ct != "not available"]))

    agg = data_full.obs.groupby(["Sample Characteristic[organism part]", "cell_types"]).size()
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(agg.sort_index())

    n_epochs = 100
    colors = None
    tissue = 'hippocampus'
    print("Training VAE for tissue ", tissue)
    cutout_cts = agg[tissue].index.values
    if "not available" in cutout_cts:
        cutout_cts = np.delete(cutout_cts, np.where(cutout_cts == "not available")[0])
    rem_ct = data_big_mapped.cell_types[~np.isin(data_big_mapped.cell_types, cutout_cts)]
    data_small = copy.deepcopy(data_big_mapped)
    data_big = copy.deepcopy(data_big_mapped)

    data_small.filter_cell_types(cutout_cts)
    data_big.filter_cell_types(rem_ct)

    # trainer_big = train_vae(data_big, "./data", f"big_{tissue}_data_portion", n_epochs=n_epochs)
    trainer_small = train_vae(data_small, "./data", f"small_{tissue}_data_portion", n_epochs=n_epochs)
    dot_size = (mpl.rcParams['lines.markersize'] ** 2.0)

    # posterior_big = plot_tsne(trainer_big, trainer_big.model, data_big, f"./plots/big_{tissue}_data_portion",
    #                           colors=colors, s=dot_size, edgecolors='black')
    posterior_small = plot_tsne(trainer_small, trainer_small.model, data_small, f"./plots/small_{tissue}_data_portion",
                                colors=colors, s=dot_size, edgecolors='black')
    posterior_small_in_big = plot_tsne(trainer_big, trainer_big.model, data_small,
                                       f"./plots/small_{tissue}_data_portion_in_big",
                                       colors=colors, s=dot_size, edgecolors='black')
