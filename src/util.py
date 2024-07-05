import logging
import pathlib
import urllib.request
from datetime import datetime
import numpy as np
from netCDF4 import Dataset

from src import MyExecError
from mce.util.io import write_nc

logger = logging.getLogger(__name__)

def read_url(url, decode=True, queries={}):
    """Read internet resource contents

    Parameters
    ----------
    url
        Base URL
    decode, optional
        Whether or not decoded, by default True
    queries, optional
        Query key-value pairs, by default {}

    Returns
    -------
        Decoded contents
    """
    if queries:
        url = '{}?{}'.format(
            url,
            '&'.join(['{}={}'.format(*item) for item in queries.items()]),
        )

    with urllib.request.urlopen(url) as f1:
        enc = f1.info().get_content_charset(failobj='utf-8')
        ret = f1.read()

    if decode:
        ret = ret.decode(enc)

    return ret


def retrieve_url(path, url):
    """Retrieve data from url

    Parameters
    ----------
    path
        Local file path
    url
        URL

    Returns
    -------
        Path object
    """
    path = pathlib.Path(path)

    if path.exists():
        logger.info('Use local file {} retrieved from {} on {}'.format(
            path.as_posix(), url,
            datetime.fromtimestamp(path.stat().st_mtime).strftime('%Y-%m-%d'),
        ))
    else:
        logger.info(f'Retrieve {url}')
        ret = read_url(url, decode=False)

        if not path.parent.is_dir():
            path.parent.mkdir(parents=True)
            logger.info('Directory {} created'.format(path.parent.as_posix()))

        with path.open('wb') as f1:
            f1.write(ret)

        logger.info('File {} created'.format(path.as_posix()))

    return path


class RetrieveGitHub:
    def __init__(self, owner, repo, path_local_dir):
        """Class for downloading files in a GitHub repository

        Parameters
        ----------
        owner
            owner part in the repository name
        repo
            repo part in the repository name
        path_local_dir
            local data directory
        """
        self.owner = owner
        self.repo = repo
        self.path_local_dir = pathlib.Path(path_local_dir)

    def retrieve(self, path):
        """Retrieve a given path

        Parameters
        ----------
        path
            file path in the repository

        Returns
        -------
            downloaded file path in the local data directory
        """
        owner = self.owner
        repo = self.repo
        path_local = self.path_local_dir.joinpath(owner, repo, path)

        # allow space character
        path = urllib.parse.quote(path)
        url = f'https://github.com/{owner}/{repo}/raw/main/{path}'

        return retrieve_url(path_local, url)


def write_nc(path, var_dict, gatts, dim_unlimited='time'):
    """Write variable data and attributes to netcdf file

    Parameters
    ----------
    path
        Output netcdf path
    var_dict
        Variable dictionary where dict values are defined
        as tuple of data, dimensions, and attributes
    gatts
        Global attribute dictionary
    dim_unlimited, optional
        Dimension treated as "unlimited", by default 'time'
    """
    # Inspect used dimensions
    # Size of time dimension is set to "unlimited"
    dim_dict = {}
    for data, dims, _ in var_dict.values():
        for name, size in zip(dims, data.shape):
            if name in dim_dict:
                continue
            dim_dict[name] = size if name != dim_unlimited else None

    ncf = netCDF4.Dataset(path, 'w')

    for name, size in dim_dict.items():
        ncf.createDimension(name, size)

    for name, (data, dims, atts) in var_dict.items():
        dtype = data.dtype
        if dtype == 'object':
            dtype = str

        ncv = ncf.createVariable(
            name, dtype, dims,
            fill_value=getattr(data, 'fill_value', None),
        )
        ncv.setncatts(atts)
        # Skip data writing for variables with unlimited dimension
        if dim_unlimited not in dims:
            ncv[:] = data

    ncf.setncatts(gatts)

    for name, (data, dims, _) in var_dict.items():
        # Writing data for variables with time dimension
        if dim_unlimited in dims:
            ncf.variables[name][:] = data

    ncf.close()


def df2nc(path, df, var_atts, gatts={}, new=False, with_reopen=None):
    """
    Save DataFrame structure to NetCDF file

    Parameters
    ----------
    path
        Output NetCDF path
    df
        Input DataFrame
        index should be pd.Index, treated as unlimited dimension
        columns can be pd.Index or pd.MultiIndex
        Elements of pd.Index or the first level of MultiIndex
        are treated as NetCDF variables
    var_atts
        Dictionary of variable attributes, such as long_name and units
    gatts, optional
        Global attributes, by default {}
    new, optional
        Existing NetCDF file is overridden when True, by default False
    with_reopen, optional
        Reopen the file with the specified mode, e.g., 'r' and 'r+',
        after the file is created, by default None (not reopened)

    Returns
    -------
        NetCDF4 object when a valid file open mode is given by with_reopen
    """
    path = pathlib.Path(path)
    if path.is_file():
        if new:
            logger.info('Old {} is deleted'.format(path.as_posix()))
            path.unlink()
        else:
            logger.error('{} already exists'.format(path.as_posix()))
            raise MyExecError()

    dname = df.index.name
    var_dict = {dname: (df.index, (dname,), {})}

    if df.columns.nlevels > 1:
        mi = df.columns.remove_unused_levels()
        dims = tuple([dname] + mi.names[1:])
        dshape = df.index.shape + mi.levshape[1:]
        var_dict.update({
            k: (np.array(v), (k,), {})
            for k, v in zip(mi.names[1:], mi.levels[1:])
        })
        var_dict.update({
            k: (
                np.ma.masked_all(dshape, dtype=df[k].dtypes[0]),
                dims, var_atts.get(k, {}),
            )
            for k in mi.levels[0]
        })
    else:
        dims = (dname,)
        dshape = df.index.shape
        var_dict.update({
            k: (
                np.ma.masked_all(dshape, dtype=df[k].dtype),
                dims, var_atts.get(k, {}),
            )
            for k in df
        })

    write_nc(path, var_dict, gatts, dim_unlimited=dname)
    logger.info('{} is created'.format(path.as_posix()))

    if with_reopen is not None:
        return Dataset(path, with_reopen)
