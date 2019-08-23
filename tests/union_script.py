from scvi.dataset import Dataset10X, UnionDataset, MouseAtlas, EbiData


if __name__ == '__main__':
    # ebi1 = EbiData("./data")
    # ds2 = Dataset10X("pbmc_1k_v2", save_path="./data")
    # ebi2 = EbiData("./data", 'E-MTAB-7320', result_file='raw')
    union_dataset = UnionDataset("./data",
                                 gene_map_load_filename="gene_maps/ensembl_mouse_genes-proteincoding",
                                 data_load_filename="test_loom_script.loom")
    # union_dataset.join_datasets("memory", "loom", out_filename="test_loom_script.loom", gene_datasets=[ds2])
    # union_dataset.join_datasets("loom", "memory", in_filename="test_loom_script.loom")

