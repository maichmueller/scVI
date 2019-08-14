from __future__ import annotations
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
from scipy import sparse
from scvi.dataset import *
from scvi.dataset.dataset import *
import torch
from torch.utils.data import Dataset
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count, Pool, Lock, Process, Value
from functools import wraps
import sys
from tqdm import tqdm
import re
import h5py
import time
from typing import Dict, Tuple, Sequence
import rpy2.robjects as robjects


class MouseAtlas(DownloadableDataset):
    def __init__(self,
                 save_path,
                 low_memory=True,
                 delayed_population=False
                 ):
        urls = ['https://shendure-web.gs.washington.edu/content/members/cao1025/public/mouse_embryo_atlas/cds_cleaned.RDS',
                'https://shendure-web.gs.washington.edu/content/members/cao1025/public/mouse_embryo_atlas/cell_annotate.csv',
                'https://shendure-web.gs.washington.edu/content/members/cao1025/public/mouse_embryo_atlas/gene_annotate.csv']
        filenames = ["cleaned_data.rds",
                     "gene_annotation.csv",
                     "celltype_annotation.csv"]
        super().__init__(save_path=save_path, urls=urls, filenames=filenames)
        self.save_path = save_path
        if not delayed_population:
            if not os.path.exists(os.path.join(self.save_path, "cleaned_data_converted.txt")):
                self._convert_source()
            data = self._read_data_file()

    def _read_data_file(self):
        with open(os.path.join(self.save_path, "cleaned_data_sparse.txt"), "r") as f:
            row, col, data = [], [], []
            genes = set()
            samples = f.readline()
            shape = f.readline()
            header = f.readline()  # skip header file
            for line in f:
                line = line.strip().split(",")
                genes.add(line[0])
                r, c, d = list(map(int, line))
                row.append(r), col.append(c), data.append(d)

        sparse_matrix = sparse.csr_matrix((data, (row, col)), shape=shape, dtype=np.int64)
        return sparse_matrix

    def _read_annotation_file(self):
        pass

    def _convert_source(self):
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
        _ = r[f'df <- readRDS({os.path.join(self.save_path, "cleaned_data.rds")}']
        _ = r["""write.csv(df@phenoData@data, phenotype_data.csv
                 write.csv(df@featureData@data, gene_annotation.csv"
                 write_dgCMatrix_csv(df@assayData$exprs, cleaned_data_sparse.txt, col1_name = "gene", chunk_size = 100)
             """]

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        return idx
