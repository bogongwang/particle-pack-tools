import os
from typing import Tuple, Optional
import warnings

import numpy as np
import xarray as xr
import dask.array as da
from netCDF4 import Dataset

def load_nc(path: str, varname: Optional[str] = None) -> xr.Dataset:
    """
    Load netCDF files into an xarray Dataset, attempting to interpret data based on a priority list 
    or a specific variable name.

    Args:
        path: Path to a netCDF file or list of paths. Wildcards supported.
        varname: Specific variable to load. If None, attempts to load ["tomo", "labels", "distance_map", "segmented"] in order.
                 The concat_dim is derived as f"{varname}_zdim".

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
            )
        except Exception as e:
            errors[f'varname={var}'] = str(e)

    raise RuntimeError(f"Failed to load netCDF. Tried: {candidates}. Errors: {errors}. Try specifying 'varname' explicitly.")

def load_nc_arr(path: str, varname: Optional[str] = None) -> da.Array:
    """
    Load netCDF files and return the specific data array as a Dask array.

    Args:
        path: Path to a netCDF file or list of paths.
        varname: Specific variable to load. If None, follows standard priority order.

    Returns:
        dask.array.Array: The raw data array of the loaded variable.
    """
    LOAD_ORDER = ["tomo", "labels", "distance_map", "segmented"]
    candidates = [varname] if varname else LOAD_ORDER
    ds = load_nc(path, varname)
    
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
        varname: str = "labels",
        attrs: dict = None,
        complevel: int = 2,
        overwrite: bool = False,
    ):
        """
        Create a new NetCDF file with specified shape and attributes.

        Args:
            shape (tuple): A tuple of (z, y, x) specifying the dimensions of the label volume.
            attrs (dict, optional): Attributes to store in the NetCDF file's global attributes.
            complevel (int, optional): Compression level for zlib compression (default: 2).
            overwrite (bool, optional): If True, overwrite an existing file at the path; otherwise, raises an error if file exists.

        Behaviour:
            Creates a NetCDF file with dimensions ('labels_zdim', 'labels_ydim', 'labels_xdim') and a variable 'labels'.
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
        ds = Dataset(self.path, 'w')
        ds.createDimension(dimensions[0], shape[0])
        ds.createDimension(dimensions[1], shape[1])
        ds.createDimension(dimensions[2], shape[2])
        ds.createVariable(
            varname=self._varname, 
            datatype=np.int32,
            dimensions=dimensions,
            zlib=True,
            complevel=complevel,
            shuffle=False,
            chunksizes=(1, shape[1], shape[2]),
            fill_value=-1
        )
        if attrs:
            for key, val in attrs.items():
                setattr(ds, key, val)
        ds.close()
        self._open_nc()

    def write(self, idx: int, data: np.ndarray, sync: bool = True):
        """
        Write a 2D data slice into the 3D label dataset at the specified index.

        Args:
            idx (int): Index along the z-dimension where the data slice will be written.
            data (np.ndarray): 2D numpy array of shape (y, x) to write into the dataset.
            sync (bool, optional): If True (default), flush changes to disk immediately. Always flushes on the last slice.

        Behaviour:
            Overwrites the specified z-slice in the NetCDF variable 'labels' with the provided data.
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
        sync: bool = True
    ):
        """
        Write a 3D data block into the label dataset starting at the specified index.

        Args:
            data (np.ndarray): 3D numpy array of shape (z, y, x) to write into the dataset.
            offset (tuple): A tuple of (z_start, y_start, x_start) specifying where to write the data block.
            sync (bool, optional): If True (default), flush changes to disk immediately after writing.

        Behaviour:
            Overwrites a sub-volume in the NetCDF variable 'labels' with the provided data block.
        """
        z_start, y_start, x_start = offset
        z_end = z_start + data.shape[0]
        y_end = y_start + data.shape[1]
        x_end = x_start + data.shape[2]
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
            Scans the specified z-slice in the NetCDF variable 'labels' and replaces
            all pixels with value `src` by `dst`. The modification is performed in-place.
        """
        self._nc_arr[idx] = xr.where(self._nc_arr[idx] == src, dst, self._nc_arr[idx])
    
    def sync(self):
        """
        Flush any pending changes in the dataset to disk.

        Behaviour:
            Ensures that all modifications made to the NetCDF variable 'labels'
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
        setting up the internal dataset and reference to the 'labels' variable for writing.
        """
        if not self._dataset:
            self._dataset = Dataset(self.path, 'r+')
            self._nc_arr = self._dataset[self._varname]
            self._shape = self._nc_arr.shape

class MaskWriter:
    """
    MaskWriter is a utility class for creating, writing to, and managing NetCDF label (mask) files.

    This class provides methods to:
      - Create a new NetCDF file with specified label dimensions and attributes.
      - Open an existing NetCDF label file for modification.
      - Write 2D slices into a 3D label dataset efficiently.
      - Support context manager protocol for safe usage with 'with' statements.
      - Finalize and close the NetCDF file, ensuring data integrity.

    Usage:
        with MaskWriter("output_labels.nc") as mw:
            mw.create_labels_nc(shape=(10, 128, 128), attrs={"description": "Segmentation labels"})
            for z in range(10):
                mw.write(z, label_slice[z])
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
        Initialize a MaskWriter instance for a given NetCDF file path.

        Args:
            path (str): Path to the NetCDF file to create or open. If the file exists, it is opened for writing; otherwise, a new file can be created with `create_labels_nc`.
        """
        warnings.warn(
            "MaskWriter is deprecated and will be removed by 0.0.10 versions. Please use NCWriter instead.", 
            DeprecationWarning
        )
        self.path = path
        self._dataset= None
        self._labels_arr = None
        self._shape = None
        if os.path.exists(path):
            self._open_labels_nc()
    
    def __enter__(self):
        """
        Enter the context manager, returning this MaskWriter instance.
        Enables use with 'with' statements for automatic cleanup.
        """
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        """
        Exit the context manager, automatically closing the NetCDF file and releasing resources.
        """
        self.close()

    def create_labels_nc(
        self,
        shape: Tuple[int, int, int],
        attrs: dict = None,
        complevel: int = 2,
        overwrite: bool = False,
    ):
        """
        Create a new NetCDF file for labels with specified shape and attributes.

        Args:
            shape (tuple): A tuple of (z, y, x) specifying the dimensions of the label volume.
            attrs (dict, optional): Attributes to store in the NetCDF file's global attributes.
            complevel (int, optional): Compression level for zlib compression (default: 2).
            overwrite (bool, optional): If True, overwrite an existing file at the path; otherwise, raises an error if file exists.

        Behaviour:
            Creates a NetCDF file with dimensions ('labels_zdim', 'labels_ydim', 'labels_xdim') and a variable 'labels'.
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
        ds = Dataset(self.path, 'w')
        ds.createDimension('labels_zdim', shape[0])
        ds.createDimension('labels_ydim', shape[1])
        ds.createDimension('labels_xdim', shape[2])
        ds.createVariable(
            varname="labels", 
            datatype=np.int32,
            dimensions=('labels_zdim', 'labels_ydim', 'labels_xdim'),
            zlib=True,
            complevel=complevel,
            shuffle=False,
            chunksizes=(1, shape[1], shape[2]),
            fill_value=-1
        )
        if attrs:
            for key, val in attrs.items():
                setattr(ds, key, val)
        ds.close()
        self._open_labels_nc()

    def write(self, idx: int, data: np.ndarray, sync: bool = True):
        """
        Write a 2D data slice into the 3D label dataset at the specified index.

        Args:
            idx (int): Index along the z-dimension where the data slice will be written.
            data (np.ndarray): 2D numpy array of shape (y, x) to write into the dataset.
            sync (bool, optional): If True (default), flush changes to disk immediately. Always flushes on the last slice.

        Behaviour:
            Overwrites the specified z-slice in the NetCDF variable 'labels' with the provided data.
        """
        assert data.shape == self._shape[1:], f"Data shape {data.shape} does not match expected shape {self._shape[1:]}"
        self._labels_arr[idx] = data
        # Flush on request or on last slice
        if sync or (idx == self._shape[0] - 1):
            self.sync()
    
    def write_block(
        self, 
        data: np.ndarray, 
        offset: Tuple[int] = (0, 0, 0), 
        sync: bool = True
    ):
        """
        Write a 3D data block into the label dataset starting at the specified index.

        Args:
            start_idx (tuple): A tuple of (z_start, y_start, x_start) specifying where to write the data block.
            data (np.ndarray): 3D numpy array of shape (z, y, x) to write into the dataset.
            sync (bool, optional): If True (default), flush changes to disk immediately after writing.

        Behaviour:
            Overwrites a sub-volume in the NetCDF variable 'labels' with the provided data block.
        """
        z_start, y_start, x_start = offset
        z_end = z_start + data.shape[0]
        y_end = y_start + data.shape[1]
        x_end = x_start + data.shape[2]
        data_mask = data > 0
        if np.any(data_mask):
            write_target = self._labels_arr[z_start:z_end, y_start:y_end, x_start:x_end]
            # netCDF4 returns a copy for slices, so mutate then assign back explicitly
            write_target[data_mask] = data[data_mask]
            self._labels_arr[z_start:z_end, y_start:y_end, x_start:x_end] = write_target
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
            Scans the specified z-slice in the NetCDF variable 'labels' and replaces
            all pixels with value `src` by `dst`. The modification is performed in-place.
        """
        self._labels_arr[idx] = xr.where(self._labels_arr[idx] == src, dst, self._labels_arr[idx])
    
    def sync(self):
        """
        Flush any pending changes in the dataset to disk.

        Behaviour:
            Ensures that all modifications made to the NetCDF variable 'labels'
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
            self._labels_arr = None

    def _open_labels_nc(self):
        """
        Open an existing NetCDF label file in read/write mode and initialize internal arrays.

        This method is called automatically when opening an existing file or after creating a new one,
        setting up the internal dataset and reference to the 'labels' variable for writing.
        """
        if not self._dataset:
            self._dataset = Dataset(self.path, 'r+')
            self._labels_arr = self._dataset['labels']
            self._shape = self._labels_arr.shape
