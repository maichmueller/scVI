from scvi.dataset.dataset10X import available_datasets, Dataset10X
from scvi.dataset.cortex import CortexDataset
from scvi.dataset.pbmc import PbmcDataset
from scvi.dataset.brain_large import BrainLargeDataset
from scvi.dataset.cite_seq import CbmcDataset
from scvi.dataset.union import UnionDataset
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from collections import defaultdict
import pandas as pd
import sys
from matplotlib import pyplot as plt
import mygene


def _load_dataset(
    ds_name,
    ds_class,
    ds_args,
    save_path
):
    print(f"{ds_class, ds_name}...")
    if ds_name is not None:
        if ds_args is not None:
            dataset = ds_class(ds_name, save_path=save_path, **ds_args)
        else:
            dataset = ds_class(ds_name, save_path=save_path)
    else:
        if ds_args is not None:
            dataset = ds_class(save_path=save_path, **ds_args)
        else:
            dataset = ds_class(save_path=save_path)

    return dataset, ds_class, ds_name


if __name__ == '__main__':
    x = CbmcDataset().gene_names
    mg = mygene.MyGeneInfo()
    query_res = mg.querymany(x, scopes='symbol', fields='ensembl.gene', species='human')
    print(*query_res, sep="\n")
    # print(*(code["ensembl"] if "notfound" not in code else f"{code['query']}: NOTFOUND" for code in query_res), sep="\n")
    print(len(x))
    # available_datasets = [
    #         "fresh_68k_pbmc_donor_a",
    #         "frozen_pbmc_donor_a",
    #         "frozen_pbmc_donor_b",
    #         "frozen_pbmc_donor_c",
    #         "pbmc8k",
    #         "pbmc4k",
    #         "t_3k",
    #         "t_4k",
    #         "pbmc_1k_protein_v3",
    #         "pbmc_10k_protein_v3",
    #         "malt_10k_protein_v3",
    #         "pbmc_1k_v2",
    #         "pbmc_1k_v3",
    #         "pbmc_10k_v3"
    # ]
    # available_datasets = [(el, Dataset10X) for el in available_datasets]
    # # available_datasets = []
    # # available_datasets.extend(
    # #     (elem for elem in zip([None]*5, (CortexDataset, PbmcDataset, CbmcDataset)))
    # # )
    # print_msgs = []
    #
    # gene_names_counter = defaultdict(int)
    # total_gn = set()
    # with ProcessPoolExecutor(max_workers=cpu_count() // 2) as executor:
    #     futures = list(
    #         (executor.submit(_load_dataset,
    #                          ds_name,
    #                          ds_class,
    #                          None,
    #                          './data')
    #          for ds_name, ds_class in zip((el[0] for el in available_datasets), (el[1] for el in available_datasets)))
    #     )
    #     for future in as_completed(futures):
    #         dataset, ds_class, ds_name = future.result()
    #         for gn in dataset.gene_names:
    #             gene_names_counter[gn] += len(dataset)
    #         total_gn = total_gn.union(dataset.gene_names)
    # print("Building dataframe.")
    # sys.stdout.flush()
    # gene_names_counter = {gn: [count] for gn, count in gene_names_counter.items()}
    # gn_stats = pd.DataFrame.from_dict(gene_names_counter).iloc[0].sort_values(ascending=False)
    # gn_stats.to_csv("test.csv")
    # print(len(total_gn), len(gn_stats))
    # # print("Plotting genes top 100...")
    # # gn_stats.iloc[0:100].plot.bar()
    # # plt.show()
    # # print(*print_msgs, sep="\n")
