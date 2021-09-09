# -*- coding: utf-8 -*-
"""
Functions and classes for analyzing multiplex imaging data

@author: C Heiser
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

sns.set_style("white")
plt.rcParams["font.family"] = "monospace"

from math import ceil
from skimage import exposure
from matplotlib.lines import Line2D


def checktype(obj):
    return bool(obj) and all(isinstance(elem, str) for elem in obj)


def plot_hist(ax, data, title=None):
    """
    Helper function for plotting histograms
    """
    ax.hist(data.ravel(), bins=256)
    # ax.ticklabel_format(ax='y', style='scientific', scilimits=(0, 0))
    if title:
        ax.set_title(title)
    return None


def scale_rgb(img, channels=None):
    """
    Scale to [0.0, 1.0] for RGB image

    Parameters
    ----------
    img : np.ndarray
        The image
    channels : tuple of int or None, optional (default=`None`)
        Channels to scale on img.shape[2]. If None, scale values in all channels.

    Returns
    -------
    img_cp : np.ndarray
        Image with scaled values
    """
    img_cp = img.copy()
    if channels is None or img.ndim == 2:
        img_cp = img_cp - img_cp.min()
        img_cp = img_cp / img_cp.max()
    else:
        for z in channels:
            plane = img_cp[:, :, z].copy()
            plane = plane - plane.min()
            img_cp[:, :, z] = plane / plane.max()
    return img_cp


def clip_values(img, channels=None):
    """
    Clip outlier values from specified channels of an image

    Parameters
    ----------
    img : np.ndarray
        The image
    channels : tuple of int or None, optional (default=`None`)
        Channels to clip on img.shape[2]. If None, clip values in all channels.

    Returns
    -------
    img_cp : np.ndarray
        Image with clipped values
    """
    img_cp = img.copy()
    if channels is None or img.ndim == 2:
        vmin, vmax = np.nanpercentile(img_cp[img_cp != -99999], q=(0.5, 99.5))
        plane_clip = exposure.rescale_intensity(
            img_cp,
            in_range=(vmin, vmax),
            out_range=np.float32,
        )
        img_cp = plane_clip
    else:
        for z in channels:
            plane = img_cp[:, :, z].copy()
            vmin, vmax = np.nanpercentile(plane, q=(0.5, 99.5))
            plane_clip = exposure.rescale_intensity(
                plane,
                in_range=(vmin, vmax),
                out_range=np.float32,
            )
            img_cp[:, :, z] = plane_clip
    return img_cp


class img:
    def __init__(self, img_arr, channels=None):
        """
        Initialize img class

        Parameters
        ----------
        img_arr : np.ndarray
            The image as a numpy array
        channels : tuple of str or None, optional (default=`None`)
            List of channel names corresponding to img.shape[2]. i.e. `("DAPI","GFAP",
            "NeuH")`. If `None`, channels are named "ch_0", "ch_1", etc.
        """
        assert (
            img_arr.ndim > 1
        ), "Image does not have enough dimensions: {} given".format(img_arr.ndim)
        self.img = img_arr  # save image array to .img attribute
        if img_arr.ndim > 2:
            self.n_ch = img_arr.shape[2]  # save number of channels to attribute
        else:
            self.n_ch = 1
        if channels is None:
            # if channel names not specified, name them numerically
            self.ch = ("ch_{}".format(x) for x in range(self.n_ch))
        else:
            assert (
                len(channels) == self.n_ch
            ), "Number of channels must match img_arr.shape[2]"
            self.ch = channels

    def scale(self, **kwargs):
        """
        Scales intensities to [0.0, 1.0]
        """
        self.img = scale_rgb(self.img, **kwargs)

    def clip(self, **kwargs):
        """
        Clips outlier values
        """
        self.img = clip_values(self.img, **kwargs)

    def show(
        self,
        channels=None,
        RGB=False,
        cbar=False,
        ncols=4,
        figsize=(7, 7),
        save_to=None,
        **kwargs,
    ):
        """
        Plot image using imshow

        Parameters
        ----------
        channels : tuple of int or None, optional (default=`None`)
            List of channels by index or name to show
        RGB : bool
            Treat 3- or 4-dimensional array as RGB image. If `False`, plot channels
            individually.
        cbar : bool
            Show colorbar for scale of image intensities if plotting individual
            channels.
        ncols : int
            Number of columns for gridspec if plotting individual channels.
        figsize : tuple of float
            Size in inches of output figure.
        save_to : str or None
            Path to image file to save results. If `None`, show figure.
        **kwargs
            Arguments to pass to `plt.imshow()` function.

        Returns
        -------
        Matplotlib object (if plotting one feature or RGB) or gridspec object (for
        multiple features). Saves plot to file if `save_to` is not `None`.
        """
        # if only one feature (2D), plot it quickly
        if self.img.ndim == 2:
            fig = plt.figure(figsize=figsize)
            plt.imshow(self.img, **kwargs)
            plt.tick_params(labelbottom=False, labelleft=False)
            sns.despine(bottom=True, left=True)
            if cbar:
                plt.colorbar(shrink=0.8)
            plt.tight_layout()
            if save_to:
                plt.savefig(
                    fname=save_to, transparent=True, bbox_inches="tight", dpi=800
                )
            return fig
        # if image has multiple channels, plot them in gridspec
        if isinstance(channels, int):  # force channels into list if single integer
            channels = [channels]
        if isinstance(channels, str):  # force channels into int if single string
            channels = [self.ch.index(channels)]
        if checktype(channels):  # force channels into list of int if list of strings
            channels = [self.ch.index(x) for x in channels]
        if channels is None:  # if no channels are given, use all of them
            channels = [x for x in range(self.n_ch)]
        assert (
            len(channels) <= self.n_ch
        ), "Too many channels given: image has {}, expected {}".format(
            self.n_ch, len(channels)
        )
        if RGB:
            # if third dim has 3 or 4 features, treat as RGB and plot it quickly
            assert (self.img.ndim == 3) & (
                len(channels) == 3
            ), "Need 3 dimensions and 3 given channels for an RGB image; shape = {}; channels given = {}".format(
                self.img.shape, len(channels)
            )
            fig = plt.figure(figsize=figsize)
            # rearrange channels to specified order
            im_tmp = np.dstack(
                [
                    self.img[:, :, channels[0]],
                    self.img[:, :, channels[1]],
                    self.img[:, :, channels[2]],
                ]
            )
            plt.imshow(im_tmp, **kwargs)
            # add legend for channel IDs
            custom_lines = [
                Line2D([0], [0], color=(1, 0, 0), lw=5),
                Line2D([0], [0], color=(0, 1, 0), lw=5),
                Line2D([0], [0], color=(0, 0, 1), lw=5),
            ]
            plt.legend(custom_lines, [self.ch[x] for x in channels], fontsize="medium")
            plt.tick_params(labelbottom=False, labelleft=False)
            sns.despine(bottom=True, left=True)
            plt.tight_layout()
            if save_to:
                plt.savefig(
                    fname=save_to, transparent=True, bbox_inches="tight", dpi=800
                )
            return fig
        # calculate gridspec dimensions
        if len(channels) <= ncols:
            n_rows, n_cols = 1, len(channels)
        else:
            n_rows, n_cols = ceil(len(channels) / ncols), ncols
        fig = plt.figure(figsize=(ncols * n_cols, ncols * n_rows))
        # arrange axes as subplots
        gs = gridspec.GridSpec(n_rows, n_cols, figure=fig)
        # add plots to axes
        i = 0
        for channel in channels:
            ax = plt.subplot(gs[i])
            im = ax.imshow(self.img[:, :, channel], **kwargs)
            ax.tick_params(labelbottom=False, labelleft=False)
            sns.despine(bottom=True, left=True)
            ax.set_title(
                label=self.ch[channel],
                loc="left",
                fontweight="bold",
                fontsize=16,
            )
            if cbar:
                _ = plt.colorbar(im, shrink=0.8)
            i = i + 1
        fig.tight_layout()
        if save_to:
            plt.savefig(fname=save_to, transparent=True, bbox_inches="tight", dpi=800)
        return fig
