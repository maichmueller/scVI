from scvi.dataset import MouseAtlas


if __name__ == '__main__':
    m = MouseAtlas("./data/",
                   {'data': "/icgc/dkfzlsdf/analysis/B260/projects/vae/data/cleaned_data_sparse.npz",
                    'cell': "/icgc/dkfzlsdf/analysis/B260/projects/vae/data/cell_annotation.csv",
                    'gene': "/icgc/dkfzlsdf/analysis/B260/projects/vae/data/gene_annotation.csv",
                    'pheno': "/icgc/dkfzlsdf/analysis/B260/projects/vae/data/phenotype_data.csv"},
                   True,
                   False)
    x=3
