import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy import sparse
from tqdm import tqdm as tqdm
import pickle
import os

if __name__ == '__main__':
    # print(os.getcwd())
    # genes = []
    # # fname = "/home/michael/GitHub/scVI_ma/tests/data/mouse_atlas/mouse_atlas_data.csv"
    # fname = "/icgc/dkfzlsdf/analysis/B260/projects/vae/data/mouse_atlas_data.csv"
    # with open(fname, "r") as in_f,\
    #      open("/icgc/dkfzlsdf/analysis/B260/projects/vae/data/cleaned_data_sparse.txt", "w") as out_f:
    #     header = in_f.readline()
    #     # out_f.write(f"{','.join(header[1:])}\n")
    #     out_f.write("shape=26183,1331984\n")
    #     out_f.write("gene row index,cell coloumn index,count\n")
    #     for row, line in enumerate(tqdm(in_f, total=26183)):
    #         line = line.strip().split(",")
    #         _, line = line[0], np.array(list(map(int, line[1:])))
    #         non_null = np.where(line != 0)[0]
    #         for col, d in zip(non_null, line[non_null]):
    #             out_f.write(f"{row},{col},{d}\n")

    # import mmap
    # with open(fname, "r+b") as f,\
    #      open("/icgc/dkfzlsdf/analysis/B260/projects/vae/data/cleaned_data_sparse.txt", "w") as out_f:
    #     map_file = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)
    #     out_f.write("shape=26183,1331984\n")
    #     out_f.write("gene ensembl id,gene row index,cell coloumn index,count\n")
    #     pbar = tqdm(f, total=26183)
    #     row = 0
    #     line = map_file.readline()
    #     while True:
    #         line = map_file.readline()
    #         if line == '': break
    #         line = line.strip().split(b",")
    #         _, line = line[0], np.array(list(map(int, line[1:])))
    #         non_null = np.where(line != 0)[0]
    #         for col, d in zip(non_null, line[non_null]):
    #             out_f.write(f"{row},{col},{d}\n")
    #         row += 1
    #         pbar.update()


    # print(os.getcwd())
    # genes = []
    # rows, cols, data = [], [], []
    # # fname = "/home/michael/GitHub/scVI_ma/tests/data/mouse_atlas/mouse_atlas_data.csv"
    # fname = "/icgc/dkfzlsdf/analysis/B260/projects/vae/data/mouse_atlas_data.csv"
    # with open(fname, "r") as in_f, \
    #     open("/icgc/dkfzlsdf/analysis/B260/projects/vae/data/mouse_atlas_data_sparse.csv", "w") as out_f:
    #     header = in_f.readline()
    #     for row, line in enumerate(tqdm(in_f, total=26183)):
    #         line = line.strip().split(",")
    #         gene, line = line[0], np.array(list(map(int, line[1:])))
    #         non_null = np.where(line != 0)[0]
    #         list(((rows.append(row), cols.append(col), data.append(d)) for (col, d) in zip(non_null, line[non_null])))
    # data = csr_matrix((data, (row, cols)), shape=(26183,1331984), dtype=np.int64)
    # sparse.save_npz("/icgc/dkfzlsdf/analysis/B260/projects/vae/data/cleaned_data_sparse.npz", data)

    # with open("/icgc/dkfzlsdf/analysis/B260/projects/vae/data/cleaned_data_sparse.txt", "r") as f,\
    # open("/icgc/dkfzlsdf/analysis/B260/projects/vae/data/cleaned_data_sparse2.txt", "w") as out:
    #     f.readline()
    #     for line in f:
    #         out.write(line)
    #
    row, col, data = [], [], []
    with open("/icgc/dkfzlsdf/analysis/B260/projects/vae/data/cleaned_data_sparse.txt", "r") as f:
        # sample line describing the sample names
        # shape line looks as follows: shape=(nr_rows, nr_cells)
        shape = tuple(map(int, eval(f.readline().split("=", 1)[-1])))
        sparse_matrix = sparse.csr_matrix(([], ([], [])), shape=(0, shape[1]), dtype=np.int64)
        header = f.readline()  # skip header file
        total = 903942242
        r, last_iter_last_r = 0, -1
        counter = 0
        pbar = tqdm(f, total=total)
        pbar.set_description("NOT YET CONCATENATED")
        for i, line in enumerate(pbar):
            # print(line)
            r_prev = r
            line = line.strip().split(",")[1:]
            r, c, d = list(map(int, line))
            if counter > (total//10) and r_prev != r:
                counter = 0
                temp = row[-1]
                row = [ro - (last_iter_last_r + 1) for ro in row]

                pbar.set_description(f"Row: {r_prev}| #data = {len(data)}")
                new_sparse = csr_matrix((data, (row, col)),
                                        shape=(row[-1] + 1, shape[1]),
                                        dtype=np.int64)
                sparse_matrix = sparse.vstack([sparse_matrix, new_sparse])
                pbar.set_description(f"Row: {r_prev}| {sparse_matrix.shape}")
                new_sparse = None
                last_iter_last_r = temp
                row, col, data = [], [], []
            row.append(r), col.append(c), data.append(d)
            counter += 1
    if row:
        row = [ro - (last_iter_last_r + 1) for ro in row]
        sparse_matrix = sparse.vstack([sparse_matrix, csr_matrix((data, (row, col)),
                                                                 shape=(row[-1]+1, shape[1]),
                                                                 dtype=np.int64)]).transpose()
    sparse.save_npz("/icgc/dkfzlsdf/analysis/B260/projects/vae/data/cleaned_data_sparse.npz", sparse_matrix)
