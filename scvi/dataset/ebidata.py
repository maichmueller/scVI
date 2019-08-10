from scipy.io import mmread
from scvi.dataset.anndataset import AnnDatasetFromAnnData
import scanpy
from scanpy._settings import ScanpyConfig
import os.path as path
import anndata
from zipfile import ZipFile
import pandas as pd


class EbiData(AnnDatasetFromAnnData):
    def __init__(self, save_path, experiment="E-ENAD-15", filter_boring: bool = False):
        settings = ScanpyConfig()
        settings.datasetdir = save_path
        filepath = path.join(save_path, experiment)
        if path.isdir(filepath):
            if not path.isfile(path.join(filepath, "experimental_design.tsv"))\
            or not path.isfile(path.join(filepath, "expression_archive.zip")):
                adata = scanpy.datasets.ebi_expression_atlas(experiment, filter_boring=filter_boring)
            else:
                adata, obs = read_archive(filepath)
                adata.obs[obs.columns] = obs
                if filter_boring:
                    adata.obs = scanpy.datasets._ebi_expression_atlas._filter_boring(adata.obs)
        else:
            adata = scanpy.datasets.ebi_expression_atlas(experiment, filter_boring=filter_boring)
        adata.obs = adata.obs.rename(columns={'Sample Characteristic[inferred cell type]': "cell_types"})
        # for i, barcode in enumerate(pd.unique(adata.obs["batch_indices"])):
        #     adata.obs["batch_indices"][adata.obs["batch_indices"] == barcode] = i
        super().__init__(adata)


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
        expr = scanpy.datasets._ebi_expression_atlas.read_mtx_from_stream(f)
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

scanpy.datasets._ebi_expression_atlas.read_expression_from_archive = read_expression_from_archive


