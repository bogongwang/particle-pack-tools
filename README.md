# Particle Pack Tools

## Installation
To install the package, run:
```
pip install particle-pack-tools
```

## Usage

### Dataset I/O
Load particle pack datasets using:
```
from pptools.io import load_nc, load_nc_arr
```

### Dataset Preprocessing
Preprocess particle pack datasets with:
```
from pptools.preprocess import crop_3d
# ...additional preprocessing functions...
```

### Dataset Visualization
Visualize particle pack datasets using the `Visualizer` class:
```
from pptools.visualize import Visualizer

# Initialize the visualizer
vis = Visualizer()

# Plot a tomogram
vis.plot_tomo(tomo_img)
vis.show()

# Plot a multi-label mask
vis.plot_mask(mask_img)
vis.show()
```

---

Feel free to report any issues or share suggestions to help improve this toolkit.