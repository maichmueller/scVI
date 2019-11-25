from scipy.io import mmread
from scvi.dataset.anndataset import AnnDatasetFromAnnData
import scanpy
from scanpy._settings import ScanpyConfig
import os.path as path
import os
import anndata
from zipfile import ZipFile
import pandas as pd
import urllib
from urllib.request import urlretrieve, urlopen
from urllib.error import HTTPError
from zipfile import ZipFile
import numpy as np
from scipy import sparse

from tqdm import tqdm


class EbiData(AnnDatasetFromAnnData):
    def __init__(
        self,
        save_path,
        experiment="E-ENAD-15",
        result_file="filtered",
        filter_boring: bool = False,
        cell_types_column_name="Sample Characteristic[inferred cell type]",
        batch_indices_column_name="",
        labels_column_name="",
    ):
        settings = ScanpyConfig()
        settings.datasetdir = save_path
        if not path.isdir(save_path):
            os.mkdir(save_path)
        filepath = path.join(save_path, experiment)
        if not path.isdir(filepath):
            os.mkdir(filepath)

        if not path.isfile(
            path.join(filepath, "experimental_design.tsv")
        ) or not path.isfile(path.join(filepath, "expression_archive.zip")):

            experiment_dir = settings.datasetdir / experiment
            dataset_path = experiment_dir / "{}.h5ad".format(experiment)
            try:
                adata = anndata.read(dataset_path)
                if filter_boring:
                    adata.obs = _filter_boring(adata.obs)
            except OSError:
                # Dataset couldn't be read for whatever reason
                pass

            download_experiment(experiment, result_file=result_file)

            print("Downloaded {} to {}".format(experiment, experiment_dir.absolute()))

            with ZipFile(experiment_dir / "expression_archive.zip", "r") as f:
                adata = read_expression_from_archive(f)
            obs = pd.read_csv(
                experiment_dir / "experimental_design.tsv", sep="\t", index_col=0
            )

            adata.obs[obs.columns] = obs
            adata.write(dataset_path, compression="gzip")  # To be kind to disk space

            if filter_boring:
                adata.obs = _filter_boring(adata.obs)
        else:
            adata, obs = read_archive(filepath)
            adata.obs[obs.columns] = obs
            if filter_boring:
                adata.obs = _filter_boring(adata.obs)
        # for i, barcode in enumerate(pd.unique(adata.obs["batch_indices"])):
        #     adata.obs["batch_indices"][adata.obs["batch_indices"] == barcode] = i
        super().__init__(
            adata,
            cell_types_column_name=cell_types_column_name,
            batch_indices_column_name=batch_indices_column_name,
            labels_column_name=labels_column_name,
        )
        self.name = experiment


# incompatability of data reader in original scanpy with mouse dataset. Patching with hotfix


def read_archive(data_folderpath):
    with ZipFile(path.join(data_folderpath, "expression_archive.zip"), "r") as f:
        adata = read_expression_from_archive(f)
    obs = pd.read_csv(
        path.join(data_folderpath, "experimental_design.tsv"), sep="\t", index_col=0
    )
    return adata, obs


def read_expression_from_archive(archive: ZipFile):
    info = archive.infolist()
    assert len(info) == 3
    mtx_data_info = [i for i in info if i.filename.endswith(".mtx")][0]
    mtx_rows_info = [i for i in info if i.filename.endswith(".mtx_rows")][0]
    mtx_cols_info = [i for i in info if i.filename.endswith(".mtx_cols")][0]
    with archive.open(mtx_data_info, "r") as f:
        expr = read_mtx_from_stream(f)
    with archive.open(mtx_rows_info, "r") as f:
        varname = pd.read_csv(f, sep="\t", header=None)[1]
    arch_folder = "/".join(archive.fp.name.split("/")[0:-1])
    archive.extract(mtx_cols_info, path=arch_folder)
    col_filename = "/".join([arch_folder, mtx_cols_info.filename])
    try:
        obsname = pd.read_csv(col_filename, sep="\t", header=None)[1]
    except KeyError:
        obsname = pd.read_csv(col_filename, sep="\t", header=None)
        obsname = obsname[obsname.columns[0]]
    adata = anndata.AnnData(expr)
    adata.var_names = varname
    adata.obs_names = obsname
    return adata


def _filter_boring(dataframe):
    unique_vals = dataframe.apply(lambda x: len(x.unique()))
    is_boring = (unique_vals == 1) | (unique_vals == len(dataframe))
    return dataframe.loc[:, ~is_boring]


# Copied from tqdm examples
def tqdm_hook(t):
    """
    Wraps tqdm instance.

    Don't forget to close() or __exit__()
    the tqdm instance once you're done with it (easiest using `with` syntax).
    Example
    -------
    >>> with tqdm(...) as t:
    ...     reporthook = my_hook(t)
    ...     urllib.urlretrieve(..., reporthook=reporthook)
    """
    last_b = [0]

    def update_to(b=1, bsize=1, tsize=None):
        """
        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            t.total = tsize
        t.update((b - last_b[0]) * bsize)
        last_b[0] = b

    return update_to


def sniff_url(accession):
    # Note that data is downloaded from gxa/sc/experiment, not experiments
    base_url = "https://www.ebi.ac.uk/gxa/sc/experiments/{}/".format(accession)
    try:
        with urlopen(base_url) as req:  # Check if server up/ dataset exists
            pass
    except HTTPError as e:
        e.msg = e.msg + " ({})".format(base_url)  # Report failed url
        raise


def download_experiment(accession, result_file="filtered"):
    sniff_url(accession)

    base_url = "https://www.ebi.ac.uk/gxa/sc/experiment/{}/".format(accession)
    quantification_path = (
        f"download/zip?fileType=quantification-{result_file}&accessKey="
    )
    sampledata_path = "download?fileType=experiment-design&accessKey="

    experiment_dir = scanpy.settings.datasetdir / accession
    experiment_dir.mkdir(parents=True, exist_ok=True)

    with tqdm(
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        miniters=1,
        desc="experimental_design.tsv",
    ) as t:
        urlretrieve(
            base_url + sampledata_path,
            experiment_dir / "experimental_design.tsv",
            reporthook=tqdm_hook(t),
        )
    with tqdm(
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        miniters=1,
        desc="expression_archive.zip",
    ) as t:
        urlretrieve(
            base_url + quantification_path,
            experiment_dir / "expression_archive.zip",
            reporthook=tqdm_hook(t),
        )


def read_mtx_from_stream(stream):
    stream.readline()
    shape_line = stream.readline()
    shape_line = stream.readline() if shape_line == b"%\n" else shape_line
    n, m, _ = (int(x) for x in shape_line[:-1].split(b" "))
    data = pd.read_csv(
        stream,
        sep=r"\s+",
        header=None,
        dtype={0: np.integer, 1: np.integer, 2: np.float32},
    )
    mtx = sparse.csr_matrix((data[2], (data[1] - 1, data[0] - 1)), shape=(m, n))
    return mtx
