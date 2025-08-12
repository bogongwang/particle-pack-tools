import numpy as np
import dask.array as da
from skimage.util import view_as_blocks
from numba import njit

@njit
def _get_bboxes_helper(_mask, _max_label, _offset):
    """
    Compute bounding boxes for labelled regions in a 3D mask array.

    :param ndarray _mask: 3D array where each voxel is labelled with an integer.
    :param int _max_label: Maximum number of labels.
    :param list _offset: Offset to adjust the bounding box coordinates.
    :return: Array of bounding boxes with shape (1, _max_label, 6), each as [x_min, y_min, z_min, x_max, y_max, z_max].
    :rtype: ndarray
    """
    bboxes = np.full((1, _max_label, 6), 32767, dtype=np.int16)
    bboxes[..., 3:] = -1
    for x in range(_mask.shape[0]):
        for y in range(_mask.shape[1]):
            for z in range(_mask.shape[2]):
                label = _mask[x, y, z]
                if label < _max_label:  # 0 is background
                    x_min, y_min, z_min, x_max, y_max, z_max = bboxes[0, label]
                    real_x = x + _offset[0]
                    real_y = y + _offset[1]
                    real_z = z + _offset[2]
                    bboxes[0, label, 0] = min(x_min, real_x)
                    bboxes[0, label, 1] = min(y_min, real_y)
                    bboxes[0, label, 2] = min(z_min, real_z)
                    bboxes[0, label, 3] = max(x_max, real_x)
                    bboxes[0, label, 4] = max(y_max, real_y)
                    bboxes[0, label, 5] = max(z_max, real_z)
    return bboxes

def get_bboxes_helper(mask, max_label, block_info=None):
    """
    Compute bounding boxes for a 3D mask using Dask block metadata.

    :param ndarray mask: 3D array where each voxel is labelled with an integer.
    :param int max_label: Maximum label number.
    :param dict block_info: Dask block information to compute offsets.
    :return: Array of bounding boxes for the provided mask block.
    :rtype: ndarray
    """
    offset = [loc[0] for loc in block_info[0]['array-location']]
    return _get_bboxes_helper(mask, _max_label=max_label, _offset=offset)

def get_bboxes(arr, max_label=10_000):
    """
    Aggregate bounding boxes from a Dask array across all blocks.

    :param ndarray arr: 3D array with labelled regions.
    :return: Aggregated array of bounding boxes, each as [x_min, y_min, z_min, x_max, y_max, z_max].
    :rtype: ndarray
    """
    blocked_res = da.map_blocks(get_bboxes_helper, arr, max_label=max_label, chunks=(1, max_label, 6), dtype=np.int16).compute()
    res_reshaped = view_as_blocks(blocked_res, block_shape=(1, max_label, 6)).reshape(-1, max_label, 6)
    res_arr = np.empty((max_label, 6), dtype=np.int16)
    res_arr[:, :3] = np.min(res_reshaped[:, :, :3], axis=0)
    res_arr[:, 3:] = np.max(res_reshaped[:, :, 3:], axis=0)
    return res_arr

def crop_3d(arr, loc, padding=0):
    """
    Crop a 3D array to the specified bounding box with optional padding.

    :param ndarray arr: 3D array to crop.
    :param tuple loc: Bounding box coordinates as (z_min, x_min, y_min, z_max, x_max, y_max).
    :param int padding: Number of voxels to pad around the bounding box.
    :return: Cropped 3D array.
    :rtype: ndarray
    """
    if padding == 0:
        return arr[loc[0]:loc[3] + 1, loc[1]:loc[4] + 1, loc[2]:loc[5] + 1]
    z_min = max(loc[0] - padding, 0)
    x_min = max(loc[1] - padding, 0)
    y_min = max(loc[2] - padding, 0)
    z_max = min(loc[3] + padding, arr.shape[0] - 1)
    x_max = min(loc[4] + padding, arr.shape[1] - 1)
    y_max = min(loc[5] + padding, arr.shape[2] - 1)
    return arr[z_min:z_max + 1, x_min:x_max + 1, y_min:y_max + 1]