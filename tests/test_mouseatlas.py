from scvi.dataset import MouseAtlas

if __name__ == '__main__':
    # fpaths_and_fnames = {'data': "/icgc/dkfzlsdf/analysis/B260/projects/vae/data/cleaned_data_sparse.npz",
    #                      'cell': "/icgc/dkfzlsdf/analysis/B260/projects/vae/data/cell_annotation.csv",
    #                      'gene': "/icgc/dkfzlsdf/analysis/B260/projects/vae/data/gene_annotation.csv",
    #                      'pheno': "/icgc/dkfzlsdf/analysis/B260/projects/vae/data/phenotype_data.csv"}
    fpaths_and_fnames = {'data': "./data/mouse_atlas/cleaned_data_sparse.npz",
                         'cell': "./data/mouse_atlas/cell_annotation.csv",
                         'gene': "./data/mouse_atlas/gene_annotation.csv",
                         'pheno': "./data/mouse_atlas/phenotype_data.csv"}
    m = MouseAtlas("./data/mouse_atlas",
                   fpaths_and_fnames,
                   True,
                   False)
    x = 3
