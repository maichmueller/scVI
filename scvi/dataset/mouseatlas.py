from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import sparse
from scvi.dataset import *
from scvi.dataset.dataset import *
from tqdm import tqdm
import h5py
import time
from typing import Dict, Tuple, Sequence
# import rpy2.robjects as robjects


class MouseAtlas(DownloadableDataset):
    def __init__(self,
                 save_path,
                 fpaths_and_fnames=None,
                 use_ensembl_gene_names=True,
                 delayed_population=False
                 ):
        if fpaths_and_fnames is None:
            urls = [
                'https://shendure-web.gs.washington.edu/content/members/cao1025/public/mouse_embryo_atlas/'
                + file for file in ('cds_cleaned.RDS', 'cell_annotate.csv', 'gene_annotate.csv')]
            filenames = ["cleaned_data.rds",
                         "gene_annotation.csv",
                         "cell_annotation.csv"]
            super().__init__(save_path=save_path, urls=urls, filenames=filenames)
            filenames.append("phenotype_annotation.csv")

        else:
            super().__init__(save_path=save_path, urls=None, filenames=None, delayed_populating=True)
            keys = ["cell", "gene", "data", "pheno"]
            for ftype, fpath in fpaths_and_fnames.items():
                path = fpath.split("/")[:-1]
                if path != self.save_path:
                    if ftype == "cell":
                        file = os.path.join(self.save_path, "cell_annotation.csv")
                        if not os.path.exists(file):
                            os.symlink(fpath, file)
                    elif ftype == "gene":
                        file = os.path.join(self.save_path, "gene_annotation.csv")
                        if not os.path.exists(file):
                            os.symlink(fpath, file)
                    elif ftype == "data":
                        _, ext = os.path.splitext(fpath)
                        if ext.lower() == ".rds":
                            self._convert_source(data_filepath=fpath)
                        file = os.path.join(self.save_path, "cleaned_data_sparse.npz")
                        if not os.path.exists(file):
                            os.symlink(fpath, file)
                    elif ftype == "pheno":
                        file = os.path.join(self.save_path, "phenotype_annotation.csv")
                        if not os.path.exists(file):
                            os.symlink(fpath, file)
                    else:
                        raise ValueError(f"Key '{ftype}' not useful. Try one of {keys}.")

            filenames = ["cleaned_data_sparse.npz",
                         "gene_annotation.csv",
                         "cell_annotation.csv",
                         "phenotype_annotation.csv"]
        self.filenames = filenames
        self.use_ensembl = use_ensembl_gene_names
        if not delayed_population:
            if not os.path.exists(os.path.join(self.save_path, "cleaned_data_sparse.npz")):
                self._convert_source()
            self.populate()

    def populate(self):
        data = self._read_data_file(os.path.join(self.save_path, self.filenames[0]))
        gene_annotation = self._read_annotation_file(os.path.join(self.save_path, self.filenames[1]))
        if self.use_ensembl:
            gene_names = np.array([gn.split('.')[0] for gn in gene_annotation["gene_id"]])
        else:
            gene_names = gene_annotation["gene_short_name"]
        cell_annotation = self._read_annotation_file(os.path.join(self.save_path, self.filenames[2]))
        phenotype_annotation = self._read_annotation_file(os.path.join(self.save_path, self.filenames[3]))
        all_cell_types = self._map_celltypes_to_samples(cell_annotation, phenotype_annotation)
        cell_types = np.unique(all_cell_types)
        labels, n_labels = remap_categories(all_cell_types, mapping_from=cell_types)
        # remove temporary symlinks
        for fname in self.filenames:
            path = os.path.join(self.save_path, fname)
            if os.path.islink(path):
                os.unlink(path)

        super().populate_from_data(X=data,
                                   gene_names=gene_names,
                                   cell_types=cell_types,
                                   labels=labels,
                                   remap_attributes=False)

    @staticmethod
    def _read_data_file(fpath):
        _, ext = os.path.splitext(fpath)
        if ext == ".txt":
            with open(fpath, "r") as f:
                row, col, data = [], [], []
                # sample line describing the sample names, skipped
                f.readline()
                # shape line looks as follows: shape=(nr_rows, nr_cells)
                shape = tuple(map(int, eval(f.readline().split("=", 1)[-1])))
                # header line, skipped
                f.readline()
                for line in tqdm(f):
                    line = line.strip().split(",")[1:]
                    r, c, d = list(map(int, line))
                    row.append(r), col.append(c), data.append(d)

            sparse_matrix = sparse.csr_matrix((data, (row, col)), shape=shape, dtype=np.int64).transpose().tocsr()
        else:
            sparse_matrix = sparse.load_npz(fpath)
        return sparse_matrix

    @staticmethod
    def _read_annotation_file(fpath):
        annotation = pd.read_csv(fpath, header=0)
        return annotation

    @staticmethod
    def _map_celltypes_to_samples(cell_df, samples_df):
        cell_types_per_sample = cell_df[["Main_cell_type", 'sample']].set_index("sample")
        return cell_types_per_sample.loc[samples_df['sample']]

    def _convert_source(self, data_filepath=None):
        if data_filepath is None:
            data_filepath = os.path.join(self.save_path, "cleaned_data.RDS")
            path = self.save_path
        else:
            path = "/".join(data_filepath.split("/")[:-1])
        r = robjects.r
        _ = r["""
        monocle_installed <- require(monocle)
        if (!requireNamespace("BiocManager", quietly = TRUE))
            install.packages("BiocManager")
        if(monocle_installed == FALSE)
            BiocManager::install("monocle")

        write_dgCMatrix_csv <- function(mat,
                                filename,
                                col1_name = "gene",
                                chunk_size = 1000) {
        
        #library(Matrix)
        #library(data.table)
        
        # Transpose so retrieval of "rows" is much faster
        mat <- Matrix::t(mat)
        
        # Row names
        row_names <- colnames(mat)
        
        # gene names are now columns
        col_names <- rownames(mat)
        
        n_row <- length(row_names)
        n_col <- length(col_names)
        
        n_chunks <- floor(n_row/chunk_size)
        
        # Initial chunk
        chunk <- 1
        chunk_start <- 1 + chunk_size * (chunk - 1)
        chunk_end <- chunk_size * chunk
        print(paste0("Writing rows ",chunk_start," to ", chunk_end))
        chunk_mat <- t(as.matrix(mat[,chunk_start:chunk_end]))
        chunk_df <- cbind(data.frame(col1 = row_names[chunk_start:chunk_end]),as.data.frame(chunk_mat))
        names(chunk_df)[1] <- col1_name
        data.table::fwrite(chunk_df, file = filename, append = F)
        
        # chunkation over chunks
        for(chunk in 2:n_chunks) {
            chunk_start <- 1 + chunk_size * (chunk - 1)
            chunk_end <- chunk_size * chunk
            print(paste0("Writing rows ",chunk_start," to ", chunk_end))
            chunk_mat <- t(as.matrix(mat[,chunk_start:chunk_end]))
            chunk_df <- cbind(data.frame(col1 = row_names[chunk_start:chunk_end]),as.data.frame(chunk_mat))
            data.table::fwrite(chunk_df, file = filename, append = T)
        }
        
        # Remaining samples
        chunk_start <- (n_chunks*chunk_size + 1)
        chunk_end <- n_row
        print(paste0("Writing rows ",chunk_start," to ", chunk_end))
        chunk_mat <- t(as.matrix(mat[,chunk_start:chunk_end]))
        chunk_df <- cbind(data.frame(col1 = row_names[chunk_start:chunk_end]),as.data.frame(chunk_mat))
        data.table::fwrite(chunk_df, file = filename, append = T)
        
        }"""]
        _ = r[f'df <- readRDS({data_filepath}']
        _ = r[f"write.csv(df@phenoData@data, {os.path.join(path, 'phenotype_annotation.csv')}"]
        _ = r[f"write.csv(df@featureData@data, {os.path.join(path, 'gene_annotation.csv')}"]
        _ = r[f"write_dgCMatrix_csv(df@assayData$exprs, {os.path.join(path, 'cleaned_data_sparse.txt')},"
              f"col1_name = 'gene', chunk_size = 100)"]
        self._convert_data_to_sparse()

    def _convert_data_to_sparse(self, fpath=None, save_path=None):
        if save_path is None:
            save_path = self.save_path
        if fpath is None:
            fpath = os.path.join(save_path, self.filenames[0])

        rows, cols, data = [], [], []
        with open(fpath, "r") as f:
            # shape line looks as follows: shape=(nr_rows, nr_cells)
            shape = tuple(map(int, eval(f.readline().split("=", 1)[-1])))
            # sparse matrix init (will be concatenated)
            sparse_matrix = sparse.csr_matrix(([], ([], [])), shape=(0, shape[1]), dtype=np.int64)
            f.readline()  # skip header line
            total = 903942244 - 2  # this many lines should be left to iterate through
            r, last_iter_last_r = 0, -1
            seg_line_count = 0
            # controls the memory peaks which come from python's list containers with many entries
            granularity_of_segments = 100  # nr of segments the dataset read should be split into
            pbar = tqdm(f, total=total)
            pbar.set_description("NOT YET CONCATENATED")
            for i, line in enumerate(pbar):
                # print(line)
                r_prev = r
                line = line.strip().split(",")[1:]
                r, c, d = list(map(int, line))
                if seg_line_count > (total // granularity_of_segments) and r_prev != r:
                    seg_line_count = 0
                    row = [riga - (last_iter_last_r + 1) for riga in row]
                    pbar.set_description(
                        f"Row: {r_prev}| #data = {len(data)}"
                    )
                    new_sparse = sparse.csr_matrix((data, (row, col)),
                                                   shape=(row[-1] + 1, shape[1]),
                                                   dtype=np.int64)
                    sparse_matrix = sparse.vstack([sparse_matrix, new_sparse])
                    pbar.set_description(
                        f"Row: {r_prev}| #data = {len(data)}, sparse: {new_sparse.nnz}, total: {sparse_matrix.nnz}"
                    )
                    new_sparse = None
                    last_iter_last_r = row[-1]
                    row, col, data = [], [], []
                row.append(r), col.append(c), data.append(d)
                seg_line_count += 1
        if row:
            row = [ro - (last_iter_last_r + 1) for ro in row]
            sparse_matrix = sparse.vstack([sparse_matrix, sparse.csr_matrix((data, (row, col)),
                                                                            shape=(row[-1] + 1, shape[1]),
                                                                            dtype=np.int64)]).transpose().tocsr()
        sparse.save_npz(os.path.join(save_path, "cleaned_data_sparse.npz"), sparse_matrix)
