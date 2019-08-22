from scvi.dataset import Dataset10X, UnionDataset, MouseAtlas, EbiData


if __name__ == '__main__':
    ebi1 = EbiData("./data", 'E-MTAB-6946', result_file='raw')
    ebi2 = EbiData("./data", 'E-MTAB-7320', result_file='raw')
    union_dataset = UnionDataset("./data",
                                 gene_map_load_filename="ensembl_mouse_genes-proteincoding")
    union_dataset.join_datasets("memory", "loom", out_filename=)

