import xarray as xr
import dask.array as da

def load_nc(path: str) -> xr.Dataset:
    """
    Load one or more netCDF files into an xarray Dataset, attempting to interpret the data as tomograms first, then as labels if tomogram loading fails.

    This function loads tomograms or segmentation labels from netCDF files. 
    It supports loading a single file or multiple files using wildcards.

    Args:
        path (str or list): Path to a netCDF file or a list of file paths. Wildcards (e.g., block*.nc) are supported for batch loading.

    Returns:
        xarray.Dataset: Dataset containing the loaded data, with dimension 'tomo_zdim' or 'labels_zdim'.

    Example:
        ds = load_nc('block0001.nc')
        ds = load_nc('block*.nc')
    """
    print("func call")
    try:
        return xr.open_mfdataset(
            path,
            concat_dim="tomo_zdim",
            data_vars="minimal",
            combine="nested",
            combine_attrs="drop_conflicts",
            coords="minimal",
            compat="override",
        )
    except Exception as e_tomo:
        try:
            return xr.open_mfdataset(
                path,
                concat_dim="labels_zdim",
                data_vars="minimal",
                combine="nested",
                combine_attrs="drop_conflicts",
                coords="minimal",
                compat="override",
            )
        except Exception as e_labels:
            raise RuntimeError(f"Failed to load netCDF files as tomogram (error: {e_tomo}) and as labels (error: {e_labels}). Please check the file(s) and dimension names.")

def load_nc_arr(path: str) -> da.Array:
    """
    Load one or more netCDF files and return the data array for the tomogram or label variable as a Dask array.

    This function is useful for extracting the raw data array from volumetric tomogram or label datasets, enabling efficient out-of-core computation with Dask.

    Args:
        path (str or list): Path to a netCDF file or a list of file paths. Wildcards (e.g., block*.nc) are supported for batch loading.

    Returns:
        dask.array.Array: The data array from the 'tomo' variable (if present) or 'labels' variable (if tomogram loading fails).

    Example:
        arr = load_nc_arr('block0001.nc')
        arr = load_nc_arr('block*.nc')
    """
    try:
        ds = xr.open_mfdataset(
            path,
            concat_dim="tomo_zdim",
            data_vars="minimal",
            combine="nested",
            combine_attrs="drop_conflicts",
            coords="minimal",
            compat="override",
        )
        return ds["tomo"].data
    except Exception as e_tomo:
        try:
            ds = xr.open_mfdataset(
                path,
                concat_dim="labels_zdim",
                data_vars="minimal",
                combine="nested",
                combine_attrs="drop_conflicts",
                coords="minimal",
                compat="override",
            )
            return ds["labels"].data
        except Exception as e_labels:
            raise RuntimeError(f"Failed to load netCDF files as tomogram (error: {e_tomo}) and as labels (error: {e_labels}). Please check the file(s), variable names, and dimension names.")