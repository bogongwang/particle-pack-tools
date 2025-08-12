import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.colors import ListedColormap
import matplotlib.colors as mcolors
from matplotlib.widgets import Slider
from skimage.transform import resize

class Visualizer:
    @staticmethod
    def create_mask_cmap(n, cmap='gist_rainbow', seed: int = None):
        """
        Create a custom color map that plots n colors, while leaving the background black
        """
        original_cmap = colormaps[cmap]
        colors = np.linspace(0, 1, n - 1)
        if seed is not None:
            np.random.seed(seed)
            np.random.shuffle(colors)
        cmap_colors = original_cmap(colors)
        black = np.array([[0, 0, 0, 1]])
        cmap_colors = np.concatenate((black, cmap_colors))
        return ListedColormap(cmap_colors)

    def __init__(self, cmap='gist_rainbow', n_colors=32768, seed=42, latex=False):
        if latex:
            text_color = 'black'
            plt.style.use('default')
            plt.rcParams['text.color'] = text_color
            plt.rcParams['axes.labelcolor'] = text_color
            plt.rcParams['xtick.color'] = text_color
            plt.rcParams['ytick.color'] = text_color
            plt.rcParams['axes.edgecolor'] = text_color
            plt.rcParams['font.family'] = 'Computer Modern'
            plt.rcParams['mathtext.fontset'] = 'cm'
            plt.rcParams['text.usetex'] = True
            plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
        else:
            mpl.rcParams.update(mpl.rcParamsDefault)
        self.tomo_cmap = 'gray'
        self.mask_cmap = Visualizer.create_mask_cmap(n_colors, cmap=cmap, seed=seed)
        self.mask_cmap_norm = mcolors.Normalize(vmin=0, vmax=n_colors)
        self.sliders = []

    def plot_tomo(self, tomo, alpha=1, show_ticks=False, ax=None):
        """
        Plot a tomogram.

        Parameters:
        - tomo: The tomogram to be plotted.
        - alpha: The transparency level of the image.
        - show_ticks: Whether to show the xy ticks.
        - ax: The axis to plot on. If None, use the current axis.
        """
        if ax is None:
            if not show_ticks:
                plt.gca().set_axis_off()
            plt.imshow(tomo, alpha=alpha, cmap=self.tomo_cmap, interpolation='none')
        else:
            if not show_ticks:
                plt.gca().set_axis_off()
            return ax.imshow(tomo, alpha=alpha, cmap=self.tomo_cmap, interpolation='none', rasterized=True)

    def plot_mask(self, mask, max_mask_val=None, alpha=1, show_ticks=False, ax=None):
        """
        Plot a mask image.

        Parameters:
        - mask: The mask data to be plotted.
        - alpha: The transparency level of the image.
        - show_ticks: Whether to show the xy ticks.
        - ax: The axis to plot on. If None, use the current axis.
        """
        if max_mask_val is not None:
            mask = mask.copy()
            mask[mask > max_mask_val] = 0 
        if ax is None:
            if not show_ticks:
                plt.gca().set_axis_off()
            plt.imshow(mask, alpha=alpha, cmap=self.mask_cmap, norm=self.mask_cmap_norm, interpolation='none', rasterized=True)
        else:
            if not show_ticks:
                plt.gca().set_axis_off()
            return ax.imshow(mask, alpha=alpha, cmap=self.mask_cmap, norm=self.mask_cmap_norm, interpolation='none', rasterized=True)


    def compare_tomo_mask(self, tomo, mask, figsize=(10, 5), show_ticks=False, stacked=True):
        """
        Compare a tomogram with a mask.
        The mask will be plotted on top of the tomogram.

        Parameters:
        - tomo: The tomography data to be plotted.
        - mask: The mask data to be plotted.
        - figsize: The size of the figure.
        - show_ticks: Whether to show the xy ticks.
        - stacked: Whether to stack the images on top of each other or displayed in a row.
        """
        if stacked:
            fig, ax = plt.subplots(figsize=figsize)
            self.plot_tomo(tomo, show_ticks=show_ticks, ax=ax)
            top_im = self.plot_mask(mask, show_ticks=show_ticks, ax=ax)
            slider_ax = plt.axes([0.2, 0, 0.6, 0.04])
            slider = Slider(slider_ax, '', 0, 1, valinit=0.5)
            def _update(val):
                top_im.set_alpha(slider.val)
                fig.canvas.draw_idle()
            slider.on_changed(_update)
        else:
            fig, ax = plt.subplots(1, 2, figsize=figsize)
            self.plot_tomo(tomo, ax=ax[0], show_ticks=show_ticks)
            self.plot_mask(mask, ax=ax[1], show_ticks=show_ticks)
        self.show()

    def compare_masks(self, *masks, figsize=(10, 5), show_ticks=False, stacked=True):
        """
        Compare multiple masks.
        The masks can be stacked on top of each other with sliders to adjust transparency, or displayed side by side.

        Parameters:
        - masks: The masks to be compared. At least two masks are required.
        - figsize: The size of the figure.
        - show_ticks: Whether to show the xy ticks.
        - stacked: Whether to stack the images on top of each other or display them side by side.
        """
        if len(masks) < 2:
            raise ValueError("At least two masks are required.")
        if stacked:
            n_sliders = len(masks) - 1
            slider_height = 0.04
            gap = 0.01
            total_slider_height = n_sliders * (slider_height + gap)
            fig, ax = plt.subplots(1, 1, figsize=figsize)
            fig.subplots_adjust(bottom=total_slider_height + 0.05)
            # Plot the base mask.
            self.plot_mask(masks[0], ax=ax, show_ticks=show_ticks)
            for i, mask in enumerate(masks[1:]):
                im = self.plot_mask(mask, ax=ax, show_ticks=show_ticks)
                slider_ax = fig.add_axes([0.2, 0.05 + (n_sliders - i - 1)*(slider_height + gap), 0.6, slider_height])
                slider = Slider(slider_ax, f'mask {i+1}', 0, 1, valinit=0.5)
                def update(val, im=im):
                    im.set_alpha(val)
                    fig.canvas.draw_idle()
                slider.on_changed(update)
                self.sliders.append(slider)
            self.show()
        else:
            n_axes = len(masks)
            fig, axes = plt.subplots(1, n_axes, figsize=figsize)
            for ax, mask in zip(axes, masks):
                self.plot_mask(mask, ax=ax, show_ticks=show_ticks)
            self.show()

    def compare_tomo_mask_pred(self, tomo, mask, pred):
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        self.plot_tomo(tomo, ax=ax[0])
        self.plot_mask(mask, ax=ax[1])
        self.plot_mask(pred, ax=ax[2])
        plt.show()

    def plot_tomo_stack(self, tomo_stack, figsize=(10, 10), downscale=False):
        """
        Visualise a tomogram stack with a slider.

        Note: Using Dask array directly as input can significantly slow down 
        drawing performance. It is recommended to load the Dask array in advance 
        before using this function.

        Parameters:
        - tomo_stack: 3D numpy array of slices.
        - figsize   : Figure size.
        - downscale : If True, each slice is downscaled to 300*300.
        """
        n_slices = tomo_stack.shape[0]
        fig, ax = plt.subplots(figsize=figsize)
        plt.subplots_adjust(bottom=0.25)
        if downscale:
            slice0 = resize(np.array(tomo_stack[0]), (300, 300), mode='reflect', anti_aliasing=True)
        else:
            slice0 = np.array(tomo_stack[0])
        im = self.plot_tomo(slice0, show_ticks=False, ax=ax)
        ax.set_title("Slice 0")
        slider_ax = fig.add_axes([0.2, 0, 0.6, 0.03])
        slider = Slider(slider_ax, 'Slice', 0, n_slices - 1, valinit=0, valfmt='%d', valstep=1)
        def update(val):
            idx = int(slider.val)
            if downscale:
                slice_img = resize(np.array(tomo_stack[idx]), (300, 300), mode='reflect', anti_aliasing=True)
            else:
                slice_img = np.array(tomo_stack[idx])
            im.set_data(slice_img)
            ax.set_title(f"Slice {idx}")
            fig.canvas.draw_idle()
            del slice_img
        slider.on_changed(update)
        self.sliders.append(slider)
        self.show()
    
    def plot_mask_stack(self, mask_stack, figsize=(10, 10), downscale=False):
        """
        Visualise a mask stack with a slider.

        Note: Using Dask array directly as input can significantly slow down 
        drawing performance. It is recommended to load the Dask array in advance 
        before using this function.
        
        Parameters:
        - tomo_stack: 3D numpy array of slices.
        - figsize   : Figure size.
        - downscale : If True, each slice is downscaled to 300*300.
        """
        n_slices = mask_stack.shape[0]
        fig, ax = plt.subplots(figsize=figsize)
        plt.subplots_adjust(bottom=0.25)
        if downscale:
            slice0 = np.array(mask_stack[0]).astype(np.float32)
            slice0 = resize(slice0, (300, 300), mode='reflect', anti_aliasing=False)
        else:
            slice0 = mask_stack[0]
        im = self.plot_mask(slice0, show_ticks=False, ax=ax)
        ax.set_title("Slice 0")
        slider_ax = fig.add_axes([0.2, 0, 0.6, 0.03])
        slider = Slider(slider_ax, 'Slice', 0, n_slices - 1, valinit=0, valfmt='%d', valstep=1)
        def update(val):
            idx = int(slider.val)
            if downscale:
                slice_img = np.array(mask_stack[idx]).astype(np.float32)
                slice_img = resize(slice_img, (300, 300), mode='reflect', anti_aliasing=False)
            else:
                slice_img = mask_stack[idx]
            im.set_data(slice_img)
            ax.set_title(f"Slice {idx}")
            fig.canvas.draw_idle()
        slider.on_changed(update)
        self.sliders.append(slider)
        self.show()
    
    def figsize(self, figsize=(10, 10)):
        plt.figure(figsize=figsize)

    def show(self):
        plt.show()
