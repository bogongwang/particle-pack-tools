import xarray as xr


def load_nc(path):
    """Load a netCDF file or multiple files into an xarray Dataset.
    Usage: load_nc(blockxxxx.nc) -> load a single netCDF file
        or load_nc(block*.nc) -> load multiple netCDF files matching the pattern.

    :param path: path to the netCDF file or a list of paths to multiple files.
    :return: an xarray.Dataset containing the data from the netCDF file(s).
    """
    try:
        return xr.open_mfdataset(
            path,
            concat_dim="tomo_zdim",
            data_vars= "minimal",
            combine="nested",
            combine_attrs="drop_conflicts",
            coords="minimal",
            compat="override",
        )
    except:
        return xr.open_mfdataset(
            path,
            concat_dim="labels_zdim",
            data_vars= "minimal",
            combine="nested",
            combine_attrs="drop_conflicts",
            coords="minimal",
            compat="override",
        )

def load_nc_arr(path):
    """Load a netCDF file or multiple files into an xarray Dataset.
    Usage: load_nc(blockxxxx.nc) -> load a single netCDF file
        or load_nc(block*.nc) -> load multiple netCDF files matching the pattern.

    :param path: path to the netCDF file or a list of paths to multiple files.
    :return: a dask.array.Array containing the data from the 'tomo' or 'labels' variable.
    """
    try:
        return xr.open_mfdataset(
            path,
            concat_dim="tomo_zdim",
            data_vars= "minimal",
            combine="nested",
            combine_attrs="drop_conflicts",
            coords="minimal",
            compat="override",
        )['tomo'].data
    except:
        return xr.open_mfdataset(
            path,
            concat_dim="labels_zdim",
            data_vars= "minimal",
            combine="nested",
            combine_attrs="drop_conflicts",
            coords="minimal",
            compat="override",
        )['labels'].data