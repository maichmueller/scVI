import pandas as pd
import numpy as np
from scvi.dataset import UnionDataset, Dataset10X
from time import perf_counter


def timer(func):
    def wrapped(*args, **kwargs):
        t = perf_counter()
        ret = func(*args, **kwargs)
        e = perf_counter() - t
        print(f"{e:0.5f}s")
        return e
    return wrapped


def to_file(dataset, loom_or_h5):
    union = UnionDataset("./data",
                         gene_map_load_filename="gene_maps/ensembl_mouse_genes-proteincoding",
                         low_memory=True)
    union.join_datasets(data_source="memory", data_target=loom_or_h5,
                        gene_datasets=[dataset], out_filename="benchmark_data." + loom_or_h5)
    return union


def from_file(filename):
    union = UnionDataset("./data",
                         gene_map_load_filename="gene_maps/ensembl_mouse_genes-proteincoding",
                         data_load_filename=filename,
                         low_memory=True)
    return union


@timer
def load_rows(dataset, indices):
    return dataset.X[indices]


if __name__ == '__main__':
    data = Dataset10X("neuron_10k_v3")
    data_mapped = UnionDataset("./data",
                               gene_map_load_filename="gene_maps/ensembl_mouse_genes-proteincoding",
                               low_memory=True)
    data_mapped.join_datasets(data_source="memory", data_target="memory",
                              gene_datasets=[data])
    # data_loom = to_file(data, "loom")
    # data_h5 = to_file(data, "h5")
    data_loom = from_file("benchmark_data.loom")
    data_h5 = from_file("benchmark_data.h5")

    t1 = []
    t2 = []
    t3 = []
    for i in range(100):
        indices = np.sort(np.random.choice(np.arange(len(data)), 1000, replace=False))
        t1.append(load_rows(data_mapped, indices))
        t2.append(load_rows(data_h5, indices))
        t3.append(load_rows(data_loom, indices))
    print(np.mean(t1))
    print(np.mean(t2))
    print(np.mean(t3))
