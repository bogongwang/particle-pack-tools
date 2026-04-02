import os
from typing import Tuple, Optional
from pathlib import Path

import numpy as np
import xarray as xr
import dask
import dask.array as da
from netCDF4 import Dataset
import zarr
from zarr.codecs import BloscCodec


class ZarrWriter:
    """
    A writer class for creating and writing data to Zarr arrays with Blosc compression.
    """
    def __init__(
        self, 
        path, 
        shape, 
        dtype, 
        chunks=(128, 128, 128), 
        shards=(512, 512, 512),
        comp_level=0, 
        fill_value=0, 
        overwrite=False, 
        attributes=None
    ):
        """
        Initialize a ZarrWriter instance.

        Args:
            path (str): Path to the Zarr store.
            shape (tuple): Shape of the dataset.
            dtype (str or np.dtype): Data type of the array.
            chunks (tuple, optional): Chunk shape. Defaults to (128, 128, 128).
            comp_level (int, optional): Compression level for Zstandard compressor. Defaults to 0.
            fill_value (numeric, optional): Fill value for uninitialized regions. Defaults to 0.
            overwrite (bool, optional): Whether to overwrite if the path exists. Defaults to False.
            attributes (dict, optional): User-defined attributes to store. Defaults to None.
        """
        self.path = path
        self.zarr_array = zarr.create_array(
            store=path,
            shape=shape,
            dtype=dtype,
            chunks=chunks,
            shards=shards,
            compressors=[BloscCodec(
                cname="zstd",
                clevel=comp_level,
            )],
            fill_value=fill_value,
            overwrite=overwrite,
            attributes=attributes
        )
        self.shard_size = np.prod(self.zarr_array.shards) * self.zarr_array.dtype.itemsize
    
    def write(
        self,
        data: np.ndarray | da.Array,
        offset: tuple[int, int, int] = (0, 0, 0),
        write_positive_regions_only: bool = False,
        backend: str | None = None,
    ):
        """
        Write a block of data into the Zarr array at a specified offset.

        Args:
            data (np.ndarray | da.Array): The data block to write.
            offset (tuple, optional): The (z, y, x) starting index for the write. Defaults to (0, 0, 0).
            write_positive_regions_only (bool, optional): If True, only writes positive values. If False, writes all values. Defaults to False.
            backend (str, optional): The backend to use for writing. Defaults to "zarr".
        """
        if backend is None:
            if isinstance(data, da.Array):
                backend = "dask"
            elif isinstance(data, np.ndarray):
                backend = "zarr"
            else:
                raise TypeError("Unsupported data type. Please provide a NumPy array or a Dask array.")

        if backend == "zarr":
            self._zarr_write(data, offset, write_positive_regions_only)
        elif backend == "dask":
            self._dask_write(data, offset, write_positive_regions_only)
        else:
            raise ValueError(f"Unsupported backend: {backend}")
   
    def _zarr_write(
        self,
        data: np.ndarray,
        offset: tuple[int, int, int],
        write_positive_regions_only: bool,
    ):
        """
        Write a block of data into the Zarr array at a specified offset.

        Args:
            data (np.ndarray): The data block to write.
            offset (tuple, optional): The (z, y, x) starting index for the write. Defaults to (0, 0, 0).
            write_positive_regions_only (bool, optional): If True, only writes positive values. If False, writes all values. Defaults to False.
        """
        z_start, y_start, x_start = offset
        z_end = z_start + data.shape[0]
        y_end = y_start + data.shape[1]
        x_end = x_start + data.shape[2]
        sl = np.s_[z_start:z_end, y_start:y_end, x_start:x_end]
        # Write entire region
        if not write_positive_regions_only:
            data = data.astype(self.zarr_array.dtype)
            self.zarr_array[sl] = data
        else:
            data_mask = data > 0
            if np.any(data_mask):
                data = data.astype(self.zarr_array.dtype, copy=False)
                region = self.zarr_array[sl]
                region = np.where(data_mask, data, region)
                self.zarr_array[sl] = region
    
    def _dask_write(
        self,
        data: np.ndarray | da.Array,
        offset: tuple[int, int, int] = (0, 0, 0),
        write_positive_regions_only: bool = False,
    ):
        """
        Write a block of data into the Zarr array at a specified offset using Dask for parallelism.

        Args:
            data (np.ndarray | da.Array): The data block to write.
            offset (tuple, optional): The (z, y, x) starting index for the write. Defaults to (0, 0, 0).
            write_positive_regions_only (bool, optional): If True, only writes positive values. If False, writes all values. Defaults to False.
        """
        if isinstance(data, np.ndarray):
            data = da.from_array(data, chunks=self.zarr_array.shards)
        elif isinstance(data, da.Array):
            data = data.rechunk(self.zarr_array.shards)
        else:
            raise TypeError("Unsupported data type. Please provide a NumPy array or a Dask array.")

        z_start, y_start, x_start = offset
        z_end = z_start + data.shape[0]
        y_end = y_start + data.shape[1]
        x_end = x_start + data.shape[2]
        saved_data = data

        if write_positive_regions_only:
            target_region = da.from_array(
                self.zarr_array[z_start:z_end, y_start:y_end, x_start:x_end],
                chunks=self.zarr_array.shards,
            )
            saved_data = da.where(data > 0, data, target_region)
        
        with dask.config.set({"array.chunk-size": self.shard_size}):
            saved_data.to_zarr(
                self.zarr_array,
                region=(slice(z_start, z_end), slice(y_start, y_end), slice(x_start, x_end)),
                compute=True,
            )


class ZarrReader:
    """
    A reader class for lazily loading Zarr arrays using Dask.
    """
    def __init__(self, path, mode="r", persist_threshold=2e9):
        """
        Initialize a ZarrReader instance.

        Args:
            path (str): Path to the Zarr store.
            mode (str, optional): The mode to open the Zarr store. Defaults to "r".
            persist_threshold (float, optional): Maximum size in bytes below which data is loaded into memory (persisted). Defaults to 2e9 (2 GB).
        """
        self.zarr_array = zarr.open_array(path, mode=mode)
        self.dask_array = da.from_zarr(self.zarr_array)
        if self.dask_array.nbytes < persist_threshold:
            self.dask_array = self.dask_array.persist()
    
    def get_zarr(self) -> zarr.Array:
        """
        Get the underlying Zarr array.

        Returns:
            zarr.core.Array: The Zarr array object.
        """
        return self.zarr_array
    
    def get_writable(self) -> zarr.Array:
        """
        Get a writable reference to the Zarr array. (Alias to get_zarr)

        Returns:
            zarr.core.Array: The Zarr array object opened in write mode.
        """
        if self.zarr_array.mode == 'r':
            raise PermissionError("Zarr array is opened in read-only mode. Cannot get writable reference.")
        return self.get_zarr()
    
    def get_dask_array(self) -> da.Array:
        """
        Get the Dask array representation of the Zarr data.

        Returns:
            dask.array.Array: The Dask array for lazy loading and computation.
        """
        return self.dask_array
    
    def get_readable(self) -> da.Array:
        """
        Get a readable reference to the Dask array.

        Returns:
            dask.array.Array: The Dask array for lazy loading and computation.
        """
        return self.get_dask_array()
    

    def to_numpy(self):
        """
        Compute and return the loaded data as a NumPy array.

        Returns:
            np.ndarray: The complete data array in memory.
        """
        return self.zarr_array.compute()


def load_nc(
        path: str, 
        varname: Optional[str] = None,
        mask_and_scale: bool = False,
        decode_cf: bool = True
    ) -> xr.Dataset:
    """
    Load netCDF files into an xarray Dataset, attempting to interpret data based on a priority list 
    or a specific variable name.

    Args:
        path: Path to a netCDF file or list of paths. Wildcards supported.
        varname: Specific variable to load. If None, attempts to load ["tomo", "labels", "distance_map", "segmented"] in order.
                 The concat_dim is derived as f"{varname}_zdim".
        mask_and_scale: Whether to apply mask and scale when loading.
        decode_cf: Whether to decode CF conventions.
        refer to: https://docs.xarray.dev/en/stable/generated/xarray.open_dataset.html#xarray.open_dataset

    Returns:
        xarray.Dataset: The loaded dataset.
    """
    LOAD_ORDER = ["tomo", "labels", "distance_map", "segmented"]
    candidates = [varname] if varname else LOAD_ORDER
    errors = {}

    for var in candidates:
        try:
            return xr.open_mfdataset(
                path,
                concat_dim=f"{var}_zdim",
                data_vars="minimal",
                combine="nested",
                combine_attrs="drop_conflicts",
                coords="minimal",
                compat="override",
                mask_and_scale=mask_and_scale,
                decode_cf=decode_cf
            )
        except Exception as e:
            errors[f'varname={var}'] = str(e)

    raise RuntimeError(f"Failed to load netCDF. Tried: {candidates}. Errors: {errors}. Try specifying 'varname' explicitly.")

def load_nc_arr(
        path: str, 
        varname: Optional[str] = None, 
        mask_and_scale: bool = False, 
        decode_cf: bool = True
    ) -> da.Array:
    """
    Load netCDF files and return the specific data array as a Dask array.

    Args:
        path: Path to a netCDF file or list of paths.
        varname: Specific variable to load. If None, follows standard priority order.
        mask_and_scale: Whether to apply mask and scale when loading.
        decode_cf: Whether to decode CF conventions.

    Returns:
        dask.array.Array: The raw data array of the loaded variable.
    """
    LOAD_ORDER = ["tomo", "labels", "distance_map", "segmented"]
    candidates = [varname] if varname else LOAD_ORDER
    ds = load_nc(path, varname, mask_and_scale, decode_cf)

    # If auto-detected, find which variable from the priority list exists in the dataset.
    for var in candidates:
        if var in ds:
            return ds[var].data
            
    raise RuntimeError(f"Dataset loaded, but could not locate any of {candidates} in variables: {list(ds.data_vars)}")


class NCWriter:
    """
    NCWriter is a utility class for creating, writing to, and managing NetCDF files.

    This class provides methods to:
      - Create a new NetCDF file with specified label dimensions and attributes.
      - Open an existing NetCDF label file for modification.
      - Write 2D slices into a 3D label dataset efficiently.
      - Support context manager protocol for safe usage with 'with' statements.
      - Finalize and close the NetCDF file, ensuring data integrity.

    Usage:
        with NCWriter("output_labels.nc") as nw:
            nw.create_labels_nc(shape=(10, 128, 128), attrs={"description": "Segmentation labels"})
            for z in range(10):
                nw.write(z, label_slice[z])
        # File is automatically closed at the end of the with block.

    Key Features:
        - Efficient partial writing of large label volumes.
        - Optional attribute storage.
        - Automatic resource management via context manager.
    """
    def __init__(
        self,
        path: str,
    ):
        """
        Initialize a NCWriter instance for a given NetCDF file path.

        Args:
            path (str): Path to the NetCDF file to create or open. If the file exists, it is opened for writing; otherwise, a new file can be created with `create_labels_nc`.
        """
        self.path = path
        self._dataset= None
        self._nc_arr = None
        self._shape = None
        self._varname = None
    
    def __enter__(self):
        """
        Enter the context manager, returning this NCWriter instance.
        Enables use with 'with' statements for automatic cleanup.
        """
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        """
        Exit the context manager, automatically closing the NetCDF file and releasing resources.
        """
        self.close()
    
    def open(self, varname="labels"):
        """
        Open an existing NetCDF file for writing.

        Args:
            varname (str, optional): Name of the variable to write to (default: 'labels').

        Behaviour:
            Opens the specified NetCDF file in read/write mode and initializes internal arrays.
        """
        self._varname = varname
        self._open_nc()

    def create_nc(
        self,
        shape: Tuple[int, int, int],
        dtype: type = np.int32,
        fill_value: int = 0,
        varname: str = "labels",
        attrs: dict = None,
        complevel: int = 2,
        overwrite: bool = False,
    ):
        """
        Create a new NetCDF file with specified shape and attributes.

        Args:
            shape (tuple): A tuple of (z, y, x) specifying the dimensions of the label volume.
            dtype (type, optional): Data type for the NetCDF variable (default: np.int32).
            fill_value (int, optional): Fill value for uninitialized data (default: 0). Remember to change this if using unsigned types!
            attrs (dict, optional): Attributes to store in the NetCDF file's global attributes.
            complevel (int, optional): Compression level for zlib compression (default: 2).
            overwrite (bool, optional): If True, overwrite an existing file at the path; otherwise, raises an error if file exists.

        Behaviour:
            Creates a NetCDF file with dimensions ('varname_zdim', 'varname_ydim', 'varname_xdim') and a variable 'varname'.
            Applies optional compression and stores provided attributes.
            Opens the file for subsequent writing.
        """
        if self._dataset:
            if overwrite:
                self.close()
                os.remove(self.path)
                print(f"Existing nc file at {self.path} was overwritten.")
            else:
                raise FileExistsError(f"File {self.path} already exists (overwrite is disabled).")
        self._varname = varname
        dimensions = (f'{self._varname}_zdim', f'{self._varname}_ydim', f'{self._varname}_xdim')
        Path(self.path).parent.mkdir(exist_ok=True)
        ds = Dataset(self.path, 'w')
        ds.createDimension(dimensions[0], shape[0])
        ds.createDimension(dimensions[1], shape[1])
        ds.createDimension(dimensions[2], shape[2])
        ds.createVariable(
            varname=self._varname, 
            datatype=dtype,
            dimensions=dimensions,
            zlib=True,
            complevel=complevel,
            shuffle=False,
            chunksizes=(1, shape[1], shape[2]),
            fill_value=fill_value
        )
        if attrs:
            ds.setncatts(attrs)
        self._dataset = ds
        self._open_nc()

    def write(self, idx: int, data: np.ndarray, sync: bool = True):
        """
        Write a 2D data slice into the 3D label dataset at the specified index.

        Args:
            idx (int): Index along the z-dimension where the data slice will be written.
            data (np.ndarray): 2D numpy array of shape (y, x) to write into the dataset.
            sync (bool, optional): If True (default), flush changes to disk immediately. Always flushes on the last slice.

        Behaviour:
            Overwrites the specified z-slice in the NetCDF variable 'varname' with the provided data.
        """
        assert data.shape == self._shape[1:], f"Data shape {data.shape} does not match expected shape {self._shape[1:]}"
        self._nc_arr[idx] = data
        # Flush on request or on last slice
        if sync or (idx == self._shape[0] - 1):
            self.sync()
    
    def write_block(
        self, 
        data: np.ndarray, 
        offset: Tuple[int] = (0, 0, 0), 
        write_whitespace: bool = False,
        sync: bool = True
    ):
        """
        Write a 3D data block into the label dataset starting at the specified index.

        Args:
            data (np.ndarray): 3D numpy array of shape (z, y, x) to write into the dataset.
            offset (tuple): A tuple of (z_start, y_start, x_start) specifying where to write the data block.
            write_whitespace (bool, optional): If True, writes all values in the data block including zeros; if False, only non-zero values are written (default: False).
            sync (bool, optional): If True (default), flush changes to disk immediately after writing.

        Behaviour:
            Overwrites a sub-volume in the NetCDF variable 'varname' with the provided data block.
        """
        z_start, y_start, x_start = offset
        z_end = z_start + data.shape[0]
        y_end = y_start + data.shape[1]
        x_end = x_start + data.shape[2]
        if write_whitespace:
            self._nc_arr[z_start:z_end, y_start:y_end, x_start:x_end] = data
            if sync:
                self.sync()
        else:
            data_mask = data > 0
            if np.any(data_mask):
                write_target = self._nc_arr[z_start:z_end, y_start:y_end, x_start:x_end]
                # netCDF4 returns a copy for slices, so mutate then assign back explicitly
                write_target[data_mask] = data[data_mask]
                self._nc_arr[z_start:z_end, y_start:y_end, x_start:x_end] = write_target
            if sync:
                self.sync()
    
    def replace(self, idx: int, src: int, dst: int):
        """
        Replace all occurrences of a label value within a 2D slice of the dataset.

        Args:
            idx (int): Index along the z-dimension specifying which slice to modify.
            src (int): Source label value to be replaced.
            dst (int): Destination label value to replace the source with.

        Behaviour:
            Scans the specified z-slice in the NetCDF variable 'varname' and replaces
            all pixels with value `src` by `dst`. The modification is performed in-place.
        """
        self._nc_arr[idx] = xr.where(self._nc_arr[idx] == src, dst, self._nc_arr[idx])
    
    def sync(self):
        """
        Flush any pending changes in the dataset to disk.

        Behaviour:
            Ensures that all modifications made to the NetCDF variable 'varname'
            are written to disk, keeping the on-disk data consistent with memory.
            Useful after multiple write or correct operations.
        """
        self._dataset.sync()

    def close(self):
        """
        Finalize and close the NetCDF label file.

        Ensures all data is flushed to disk and resources are released. Safe to call multiple times.
        """
        if self._dataset:
            self._dataset.sync()
            self._dataset.close()
            self._dataset = None
            self._nc_arr = None

    def _open_nc(self):
        """
        Open an existing NetCDF file in read/write mode and initialize internal arrays.

        This method is called automatically when opening an existing file or after creating a new one,
        setting up the internal dataset and reference to the 'varname' variable for writing.
        """
        if not self._dataset:
            self._dataset = Dataset(self.path, 'r+')
        self._nc_arr = self._dataset[self._varname]
        self._shape = self._nc_arr.shape