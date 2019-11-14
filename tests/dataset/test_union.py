from scvi.dataset import UnionDataset, GeneExpressionDataset, SyntheticDataset
import numpy as np
import pandas as pd
import os
import copy
import unittest
from unittest import TestCase


def create_datasets():
    np.random.seed(0)
    data_a = np.sort(np.random.normal(0, 10, 500)).astype(int).reshape(100, 5)
    gene_names_a = list("ABCDE")
    cell_types_a = ["alpha", "beta", "gamma", "delta"]
    labels_a = np.random.choice(np.arange(len(cell_types_a)), data_a.shape[0])
    batch_indices_a = np.random.choice(np.arange(5), size=data_a.shape[0])

    data_b = np.sort(np.random.normal(100, 10, 300)).astype(int).reshape(100, 3)
    gene_names_b = list("BFA")
    cell_types_b = ["alpha", "epsilon", "rho"]
    labels_b = np.random.choice(np.arange(len(cell_types_b)), data_b.shape[0])
    batch_indices_b = np.random.choice(np.arange(5), size=data_b.shape[0])

    dataset_a = GeneExpressionDataset()
    dataset_b = GeneExpressionDataset()
    dataset_a.populate_from_data(X=data_a,
                                 labels=labels_a,
                                 gene_names=gene_names_a,
                                 cell_types=cell_types_a,
                                 batch_indices=batch_indices_a)
    dataset_a.name = "test_a"

    dataset_b.populate_from_data(X=data_b,
                                 labels=labels_b,
                                 gene_names=gene_names_b,
                                 cell_types=cell_types_b,
                                 batch_indices=batch_indices_b)
    dataset_b.name = "test_b"
    return dataset_a, dataset_b


dataset1, dataset2 = create_datasets()
save_path = "../data"


class TestUnionDataset(TestCase):
    def build_datasets(self):
        self.dset1, self.dset2 = create_datasets()

    def test_build_gene_map_from_memory(self):
        dset1, dset2 = copy.deepcopy(dataset1), copy.deepcopy(dataset2)
        union = UnionDataset(save_path=save_path,
                             low_memory=True,
                             ignore_batch_annotation=False)
        union.build_genemap(data_source="memory",
                            gene_datasets=[dset1, dset2])

        expected_map = pd.Series(np.arange(len(list("ABCDEF"))), index=list("ABCDEF"))
        self.assertEqual(union.gene_names.tolist(), list("ABCDEF"))
        self.assertTrue((union.gene_map.index.values == expected_map.index.values).all())
        self.assertTrue((union.gene_map.values == expected_map.values).all())

    def test_concatenate_from_memory_to_memory(self):
        dset1, dset2 = copy.deepcopy(dataset1), copy.deepcopy(dataset2)
        union = UnionDataset(save_path=save_path,
                             low_memory=True,
                             ignore_batch_annotation=False)
        union.build_genemap(data_source="memory",
                            gene_datasets=[dset1, dset2])
        union.join_datasets(data_source="memory",
                            data_target="memory",
                            gene_datasets=[dset1, dset2])

        expected_gene_names = np.sort(np.unique(np.concatenate([dataset1.gene_names, dataset2.gene_names])))
        expected_cell_types = np.sort(np.unique(np.concatenate([dataset1.cell_types, dataset2.cell_types])))
        expected_batch_indices = np.concatenate([dataset1.batch_indices, dataset2.batch_indices + 5]).reshape(-1, 1)
        cell_types_1, cell_types_2 = dataset1.cell_types[dataset1.labels], dataset2.cell_types[dataset2.labels]
        expected_cell_types_rank = np.arange(len(expected_cell_types))
        expected_labels = np.concatenate([cell_types_1, cell_types_2])
        for rank, ct in zip(expected_cell_types_rank, expected_cell_types):
            expected_labels[expected_labels == ct] = rank
        expected_labels = expected_labels.astype(int)
        self.assertTrue((union.gene_names == expected_gene_names).all())
        self.assertTrue((union.cell_types == expected_cell_types).all())
        self.assertTrue((union.batch_indices == expected_batch_indices).all())
        self.assertTrue((union.labels == expected_labels).all())

    def test_concatenate_from_memory_to_hdf5(self):
        try:
            dset1, dset2 = copy.deepcopy(dataset1), copy.deepcopy(dataset2)

            union_mem = UnionDataset(save_path=save_path,
                                     low_memory=False,
                                     ignore_batch_annotation=False)
            union_mem.build_genemap(data_source="memory",
                                    gene_datasets=[dset1, dset2])
            union_mem.join_datasets(data_source="memory",
                                    data_target="memory",
                                    gene_datasets=[dset1, dset2])

            union = UnionDataset(save_path=save_path,
                                 low_memory=True,
                                 ignore_batch_annotation=False)
            union.build_genemap(data_source="memory", gene_datasets=[dset1, dset2])
            union.join_datasets(data_source='memory',
                                data_target='hdf5',
                                gene_datasets=[dset1, dset2],
                                out_filename="test_concat.h5")

            self.assertTrue(len(union) == len(union_mem))
            random_indices = np.sort(np.random.choice(np.arange(len(union)), size=int(len(union) / 5), replace=False))
            self.assertTrue((union.X[random_indices] == union_mem.X[random_indices]).all())

            self.assertTrue((union.gene_names == union_mem.gene_names).all())
            self.assertTrue((union.cell_types == union_mem.cell_types).all())
            self.assertTrue((union.batch_indices == union_mem.batch_indices).all())
            self.assertTrue((union.labels == union_mem.labels).all())

        except Exception as e:
            if os.path.exists(os.path.join(save_path, "test_concat.h5")):
                os.remove(os.path.join(save_path, "test_concat.h5"))
            raise e

    def test_concatenate_from_memory_to_loom(self):
        try:
            dset1, dset2 = copy.deepcopy(dataset1), copy.deepcopy(dataset2)

            union_mem = UnionDataset(save_path=save_path,
                                     low_memory=False,
                                     ignore_batch_annotation=False)
            union_mem.build_genemap(data_source="memory",
                                    gene_datasets=[dset1, dset2])
            union_mem.join_datasets(data_source="memory",
                                    data_target="memory",
                                    gene_datasets=[dset1, dset2])

            union = UnionDataset(save_path=save_path,
                                 low_memory=True,
                                 ignore_batch_annotation=False)
            union.build_genemap(data_source="memory", gene_datasets=[dset1, dset2])
            union.join_datasets(data_source='memory',
                                data_target='loom',
                                gene_datasets=[dset1, dset2],
                                out_filename="test_concat.loom")

            self.assertTrue(len(union) == len(union_mem))

            random_indices = np.sort(
                np.random.choice(
                    np.arange(len(union)),
                    size=int(len(union) / 5),
                    replace=False))

            self.assertTrue((union.X[random_indices] == union_mem.X[random_indices]).all())

            self.assertTrue((union.gene_names == union_mem.gene_names).all())
            self.assertTrue((union.cell_types == union_mem.cell_types).all())
            self.assertTrue((union.batch_indices == union_mem.batch_indices).all())
            self.assertTrue((union.labels == union_mem.labels).all())

        except Exception as e:
            if os.path.exists(os.path.join(save_path, "test_concat.loom")):
                os.remove(os.path.join(save_path, "test_concat.loom"))
            raise e

    def test_concatenate_from_hdf5_to_memory(self):
        try:
            dset1, dset2 = copy.deepcopy(dataset1), copy.deepcopy(dataset2)

            union_from_mem_to_mem = UnionDataset(save_path=save_path,
                                                 low_memory=True,
                                                 ignore_batch_annotation=False)
            union_from_mem_to_mem.build_genemap(data_source="memory", gene_datasets=[dset1, dset2])
            union_from_mem_to_mem.join_datasets(data_source='memory',
                                                data_target='memory',
                                                gene_datasets=[dset1, dset2])

            union_from_mem_to_h5 = UnionDataset(save_path=save_path,
                                                low_memory=True,
                                                ignore_batch_annotation=False)
            union_from_mem_to_h5.build_genemap(data_source="memory", gene_datasets=[dset1, dset2])
            union_from_mem_to_h5.join_datasets(data_source='memory',
                                               data_target='hdf5',
                                               gene_datasets=[dset1, dset2],
                                               out_filename="test_concat.h5")

            union_from_h5_to_mem = UnionDataset(save_path=save_path,
                                                low_memory=False,
                                                ignore_batch_annotation=False)
            union_from_h5_to_mem.build_genemap(data_source="memory",
                                               gene_datasets=[dset1, dset2])
            union_from_h5_to_mem.join_datasets(data_source="hdf5",
                                               data_target="memory",
                                               in_filename="test_concat.h5")

            self.assertTrue(len(union_from_h5_to_mem) == len(union_from_mem_to_mem))

            random_indices = np.sort(
                np.random.choice(
                    np.arange(len(union_from_mem_to_mem)),
                    size=int(len(union_from_mem_to_mem) / 5),
                    replace=False))

            self.assertTrue(
                (union_from_h5_to_mem.X[random_indices] != union_from_mem_to_mem.X[random_indices]).nnz == 0)

            self.assertTrue((union_from_h5_to_mem.gene_names == union_from_mem_to_mem.gene_names).all())
            self.assertTrue((union_from_h5_to_mem.cell_types == union_from_mem_to_mem.cell_types).all())
            self.assertTrue((union_from_h5_to_mem.batch_indices == union_from_mem_to_mem.batch_indices).all())
            self.assertTrue((union_from_h5_to_mem.labels == union_from_mem_to_mem.labels).all())

        except Exception as e:
            if os.path.exists(os.path.join(save_path, "test_concat.h5")):
                os.remove(os.path.join(save_path, "test_concat.h5"))
            raise e

    def test_concatenate_from_hdf5_to_loom(self):
        try:
            dset1, dset2 = copy.deepcopy(dataset1), copy.deepcopy(dataset2)

            union_from_mem_to_mem = UnionDataset(save_path=save_path,
                                                 low_memory=True,
                                                 ignore_batch_annotation=False)
            union_from_mem_to_mem.build_genemap(data_source="memory", gene_datasets=[dset1, dset2])
            union_from_mem_to_mem.join_datasets(data_source='memory',
                                                data_target='memory',
                                                gene_datasets=[dset1, dset2])

            union_from_mem_to_h5 = UnionDataset(save_path=save_path,
                                                low_memory=True,
                                                ignore_batch_annotation=False)
            union_from_mem_to_h5.build_genemap(data_source="memory", gene_datasets=[dset1, dset2])
            union_from_mem_to_h5.join_datasets(data_source='memory',
                                               data_target='hdf5',
                                               gene_datasets=[dset1, dset2],
                                               out_filename="test_concat.h5")

            union_from_h5_to_loom = UnionDataset(save_path=save_path,
                                                 low_memory=True,
                                                 ignore_batch_annotation=False)
            union_from_h5_to_loom.build_genemap(data_source="memory",
                                                gene_datasets=[dset1, dset2])
            union_from_h5_to_loom.join_datasets(data_source="hdf5",
                                                data_target="loom",
                                                in_filename="test_concat.h5",
                                                out_filename="test_concat.loom")

            self.assertTrue(len(union_from_h5_to_loom) == len(union_from_mem_to_mem))

            random_indices = np.sort(
                np.random.choice(
                    np.arange(len(union_from_mem_to_mem)),
                    size=int(len(union_from_mem_to_mem) / 5),
                    replace=False))

            self.assertTrue(
                (union_from_h5_to_loom.X[random_indices] == union_from_mem_to_mem.X[random_indices].toarray()).all()
            )

            self.assertTrue((union_from_h5_to_loom.gene_names == union_from_mem_to_mem.gene_names).all())
            self.assertTrue((union_from_h5_to_loom.cell_types == union_from_mem_to_mem.cell_types).all())
            self.assertTrue((union_from_h5_to_loom.batch_indices == union_from_mem_to_mem.batch_indices).all())
            self.assertTrue((union_from_h5_to_loom.labels == union_from_mem_to_mem.labels).all())

        except Exception as e:
            if os.path.exists(os.path.join(save_path, "test_concat.h5")):
                os.remove(os.path.join(save_path, "test_concat.h5"))
            if os.path.exists(os.path.join(save_path, "test_concat.loom")):
                os.remove(os.path.join(save_path, "test_concat.loom"))
            raise e

    def test_concatenate_from_loom_to_memory(self):
        try:
            dset1, dset2 = copy.deepcopy(dataset1), copy.deepcopy(dataset2)

            # Concatenate the datasets in memory first as reference
            union_from_mem_to_mem = UnionDataset(save_path=save_path,
                                                 low_memory=True,
                                                 ignore_batch_annotation=False)
            union_from_mem_to_mem.build_genemap(data_source="memory", gene_datasets=[dset1, dset2])
            union_from_mem_to_mem.join_datasets(data_source='memory',
                                                data_target='memory',
                                                gene_datasets=[dset1, dset2])

            # do the concatenation directly onto a loom file
            union_from_mem_to_loom = UnionDataset(save_path=save_path,
                                                  low_memory=True,
                                                  ignore_batch_annotation=False)
            union_from_mem_to_loom.build_genemap(data_source="memory", gene_datasets=[dset1, dset2])
            union_from_mem_to_loom.join_datasets(data_source='memory',
                                                 data_target='loom',
                                                 gene_datasets=[dset1, dset2],
                                                 out_filename="test_concat.loom")

            # load the loom file as out of memory dataset
            union_from_loom_to_memory = UnionDataset(save_path=save_path,
                                                     low_memory=True,
                                                     ignore_batch_annotation=False)
            union_from_loom_to_memory.build_genemap(data_source="memory",
                                                    gene_datasets=[dset1, dset2])
            union_from_loom_to_memory.join_datasets(data_source="loom",
                                                    data_target="memory",
                                                    in_filename="test_concat.loom")

            self.assertTrue(len(union_from_loom_to_memory) == len(union_from_mem_to_mem))

            random_indices = np.sort(
                np.random.choice(
                    np.arange(len(union_from_mem_to_mem)),
                    size=int(len(union_from_mem_to_mem) / 5),
                    replace=False))

            self.assertTrue(
                (union_from_loom_to_memory.X[random_indices] == union_from_mem_to_mem.X[random_indices].toarray()).all()
            )

            self.assertTrue((union_from_loom_to_memory.gene_names == union_from_mem_to_mem.gene_names).all())
            self.assertTrue((union_from_loom_to_memory.cell_types == union_from_mem_to_mem.cell_types).all())
            self.assertTrue((union_from_loom_to_memory.batch_indices == union_from_mem_to_mem.batch_indices).all())
            self.assertTrue((union_from_loom_to_memory.labels == union_from_mem_to_mem.labels).all())

        except Exception as e:
            if os.path.exists(os.path.join(save_path, "test_concat.loom")):
                os.remove(os.path.join(save_path, "test_concat.loom"))
            raise e

    def test_concatenate_from_loom_to_hdf5(self):
        try:
            dset1, dset2 = copy.deepcopy(dataset1), copy.deepcopy(dataset2)

            # Concatenate the datasets in memory first as reference
            union_from_mem_to_mem = UnionDataset(save_path=save_path,
                                                 low_memory=True,
                                                 ignore_batch_annotation=False)
            union_from_mem_to_mem.build_genemap(data_source="memory", gene_datasets=[dset1, dset2])
            union_from_mem_to_mem.join_datasets(data_source='memory',
                                                data_target='memory',
                                                gene_datasets=[dset1, dset2])

            # do the concatenation directly onto a loom file
            union_from_mem_to_loom = UnionDataset(save_path=save_path,
                                                  low_memory=True,
                                                  ignore_batch_annotation=False)
            union_from_mem_to_loom.build_genemap(data_source="memory", gene_datasets=[dset1, dset2])
            union_from_mem_to_loom.join_datasets(data_source='memory',
                                                 data_target='loom',
                                                 gene_datasets=[dset1, dset2],
                                                 out_filename="test_concat.loom")

            # convert the loom file to an hdf5 file
            union_from_loom_to_hdf5 = UnionDataset(save_path=save_path,
                                                   low_memory=True,
                                                   ignore_batch_annotation=False)
            union_from_loom_to_hdf5.build_genemap(data_source="memory",
                                                  gene_datasets=[dset1, dset2])
            union_from_loom_to_hdf5.join_datasets(data_source="loom",
                                                  data_target="hdf5",
                                                  in_filename="test_concat.loom",
                                                  out_filename="test_concat.h5")

            self.assertTrue(len(union_from_loom_to_hdf5) == len(union_from_mem_to_mem))

            random_indices = np.sort(
                np.random.choice(
                    np.arange(len(union_from_mem_to_mem)),
                    size=int(len(union_from_mem_to_mem) / 5),
                    replace=False))

            self.assertTrue(
                (union_from_loom_to_hdf5.X[random_indices] == union_from_mem_to_mem.X[random_indices].toarray()).all()
            )

            self.assertTrue((union_from_loom_to_hdf5.gene_names == union_from_mem_to_mem.gene_names).all())
            self.assertTrue((union_from_loom_to_hdf5.cell_types == union_from_mem_to_mem.cell_types).all())
            self.assertTrue((union_from_loom_to_hdf5.batch_indices == union_from_mem_to_mem.batch_indices).all())
            self.assertTrue((union_from_loom_to_hdf5.labels == union_from_mem_to_mem.labels).all())

        except Exception as e:
            if os.path.exists(os.path.join(save_path, "test_concat.loom")):
                os.remove(os.path.join(save_path, "test_concat.loom"))
            raise e

    def test_concatenate_from_scvi_to_memory(self):
        random_seed = 0

        dset1, dset2 = (SyntheticDataset(batch_size=10,
                                         nb_genes=4,
                                         n_proteins=4,
                                         n_batches=4,
                                         n_labels=3,
                                         seed=random_seed),
                        SyntheticDataset(batch_size=30,
                                         nb_genes=2,
                                         n_proteins=6,
                                         n_batches=2,
                                         n_labels=4,
                                         seed=random_seed))

        # Concatenate the datasets in memory first as reference
        union_from_mem_to_mem = UnionDataset(save_path=save_path,
                                             low_memory=True,
                                             ignore_batch_annotation=False)
        union_from_mem_to_mem.build_genemap(data_source="memory", gene_datasets=[dset1, dset2])
        union_from_mem_to_mem.join_datasets(data_source='memory',
                                            data_target='memory',
                                            gene_datasets=[dset1, dset2])

        # Load datasets from scvi and concatenate them in memory
        union_from_scvi_to_memory = UnionDataset(save_path=save_path,
                                                 low_memory=True,
                                                 ignore_batch_annotation=False)
        union_from_scvi_to_memory.build_genemap(data_source="memory",
                                                gene_datasets=[dset1, dset2])
        union_from_scvi_to_memory.join_datasets(data_source='scvi',
                                                data_target='memory',
                                                dataset_classes=[SyntheticDataset, SyntheticDataset],
                                                dataset_args=[{"batch_size": 10,
                                                               "nb_genes": 4,
                                                               "n_proteins": 4,
                                                               "n_batches": 4,
                                                               "n_labels": 3,
                                                               "seed": random_seed},
                                                              {"batch_size": 30,
                                                               "nb_genes": 2,
                                                               "n_proteins": 6,
                                                               "n_batches": 2,
                                                               "n_labels": 4,
                                                               "seed": random_seed}]
                                                )

        self.assertTrue(len(union_from_scvi_to_memory) == (len(dset1) + len(dset2)))

        random_indices = np.sort(
            np.random.choice(
                np.arange(len(union_from_scvi_to_memory)),
                size=int(len(union_from_scvi_to_memory) / 5),
                replace=False))

        self.assertTrue(
            (
                union_from_scvi_to_memory.X[random_indices].toarray() ==
                union_from_mem_to_mem.X[random_indices].toarray()
            ).all()
        )

        self.assertTrue((union_from_scvi_to_memory.gene_names == union_from_mem_to_mem.gene_names).all())
        self.assertTrue((union_from_scvi_to_memory.cell_types == union_from_mem_to_mem.cell_types).all())
        self.assertTrue((union_from_scvi_to_memory.batch_indices == union_from_mem_to_mem.batch_indices).all())
        self.assertTrue((union_from_scvi_to_memory.labels == union_from_mem_to_mem.labels).all())

    def test_concatenate_from_scvi_to_hdf5(self):
        try:
            dset1, dset2 = (SyntheticDataset(batch_size=10,
                                             nb_genes=4,
                                             n_proteins=4,
                                             n_batches=4,
                                             n_labels=3,
                                             seed=0),
                            SyntheticDataset(batch_size=30,
                                             nb_genes=2,
                                             n_proteins=6,
                                             n_batches=2,
                                             n_labels=4,
                                             seed=0))

            # Concatenate the datasets in memory first as reference
            union_from_mem_to_mem = UnionDataset(save_path=save_path,
                                                 low_memory=True,
                                                 ignore_batch_annotation=False)
            union_from_mem_to_mem.build_genemap(data_source="memory", gene_datasets=[dset1, dset2])
            union_from_mem_to_mem.join_datasets(data_source='memory',
                                                data_target='memory',
                                                gene_datasets=[dset1, dset2])

            # Load datasets from scvi and concatenate them in memory
            union_from_scvi_to_hdf5 = UnionDataset(save_path=save_path,
                                                   low_memory=True,
                                                   ignore_batch_annotation=False)
            union_from_scvi_to_hdf5.build_genemap(data_source="memory", gene_datasets=[dset1, dset2])
            union_from_scvi_to_hdf5.join_datasets(data_source='scvi',
                                                  data_target='hdf5',
                                                  dataset_classes=[SyntheticDataset, SyntheticDataset],
                                                  dataset_args=[{"batch_size": 10,
                                                                 "nb_genes": 4,
                                                                 "n_proteins": 4,
                                                                 "n_batches": 4,
                                                                 "n_labels": 3,
                                                                 "seed": 0},
                                                                {"batch_size": 30,
                                                                 "nb_genes": 2,
                                                                 "n_proteins": 6,
                                                                 "n_batches": 2,
                                                                 "n_labels": 4,
                                                                 "seed": 0}],
                                                  out_filename="test_concat.h5"
                                                  )

            self.assertTrue(len(union_from_scvi_to_hdf5) == (len(dset1) + len(dset2)))

            random_indices = np.sort(
                np.random.choice(
                    np.arange(len(union_from_scvi_to_hdf5)),
                    size=int(len(union_from_scvi_to_hdf5) / 5),
                    replace=False))

            self.assertTrue(
                (
                    union_from_scvi_to_hdf5.X[random_indices] ==
                    union_from_mem_to_mem.X[random_indices].toarray()
                ).all()
            )

            self.assertTrue((union_from_scvi_to_hdf5.gene_names == union_from_mem_to_mem.gene_names).all())
            self.assertTrue((union_from_scvi_to_hdf5.cell_types == union_from_mem_to_mem.cell_types).all())
            self.assertTrue((union_from_scvi_to_hdf5.batch_indices == union_from_mem_to_mem.batch_indices).all())
            self.assertTrue((union_from_scvi_to_hdf5.labels == union_from_mem_to_mem.labels).all())

        except Exception as e:
            if os.path.exists(os.path.join(save_path, "test_concat.h5")):
                os.remove(os.path.join(save_path, "test_concat.h5"))
            raise e


if __name__ == '__main__':
    unittest.main()
