# -*- coding: utf-8 -*-
"""
Functions and classes for manipulating 10X Visium spatial transcriptomic (ST) and 
histological imaging data
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import scanpy as sc

sc.set_figure_params(dpi=100, dpi_save=400)
sns.set_style("white")
plt.rcParams["font.family"] = "monospace"

from math import ceil
from matplotlib.lines import Line2D
from scipy.spatial import cKDTree
from scipy.interpolate import interpnd, griddata
from sklearn.metrics.pairwise import euclidean_distances


def bin_threshold(mat, threshmin=None, threshmax=0.5):
    """
    Generate binary segmentation from probabilities

    Parameters
    ----------
    mat : np.array
        The data
    threshmin : float or None
        Minimum value on [0,1] to assign binary IDs from probabilities.
    thresmax : float
        Maximum value on [0,1] to assign binary IDs from probabilities. Values higher
        than threshmax -> 1. Values lower than thresmax -> 0.

    Returns
    -------
    a : np.array
        Thresholded matrix
    """
    a = np.ma.array(mat, copy=True)
    mask = np.zeros(a.shape, dtype=bool)
    if threshmin:
        mask |= (a < threshmin).filled(False)

    if threshmax:
        mask |= (a > threshmax).filled(False)

    a[mask] = 1
    a[~mask] = 0
    return a


def map_pixels(adata, filter_label="in_tissue", img_key="hires", library_id=None):
    """
    Map spot IDs to 'pixel space' by assigning spot ID values to evenly spaced grid

    Parameters
    ----------
    adata : AnnData.anndata
        The data
    filter_label : str or None
        adata.obs column key that contains binary labels for filtering barcodes. If
        None, do not filter.
    img_key : str
        adata.uns key containing the image to use for mapping

    Returns
    -------
    adata : AnnData.anndata
        with the following attributes:
        adata.uns["pixel_map_df"] : pd.DataFrame
            Long-form dataframe of Visium spot barcode IDs, pixel coordinates, and
            .obs metadata
        adata.uns["pixel_map"] : np.array
            Pixel space array of Visium spot barcode IDs
    """
    adata.uns["pixel_map_params"] = {
        "img_key": img_key
    }  # create params dict for future use
    # add library_id key to params
    if library_id is None:
        library_id = adata.uns["pixel_map_params"]["library_id"] = list(
            adata.uns["spatial"].keys()
        )[0]
    else:
        adata.uns["pixel_map_params"]["library_id"] = library_id
    # first get center-to-face pixel distance of hexagonal Visium spots
    dist = euclidean_distances(adata.obsm["spatial"])
    adata.uns["pixel_map_params"]["ctr_to_face"] = (
        np.unique(dist)[np.unique(dist) != 0].min() / 2
    )
    # also save center-to-vertex pixel distance as vadata attribute
    adata.uns["pixel_map_params"]["ctr_to_vert"] = adata.uns["pixel_map_params"][
        "ctr_to_face"
    ] / np.cos(30 * (np.pi / 180))
    # get the spot radius from adata.uns["spatial"] as well
    adata.uns["pixel_map_params"]["radius"] = (
        adata.uns["spatial"][library_id]["scalefactors"]["spot_diameter_fullres"] / 2
    )
    # get scale factor from adata.uns["spatial"]
    adata.uns["pixel_map_params"]["scalef"] = adata.uns["spatial"][library_id][
        "scalefactors"
    ][f"tissue_{img_key}_scalef"]

    # determine pixel bounds from spot coords, adding center-to-face distance
    adata.uns["pixel_map_params"]["xmin_px"] = int(
        np.floor(
            adata.uns["pixel_map_params"]["scalef"]
            * (
                adata.obsm["spatial"][:, 0].min()
                - adata.uns["pixel_map_params"]["radius"]
            )
        )
    )
    adata.uns["pixel_map_params"]["xmax_px"] = int(
        np.ceil(
            adata.uns["pixel_map_params"]["scalef"]
            * (
                adata.obsm["spatial"][:, 0].max()
                + adata.uns["pixel_map_params"]["radius"]
            )
        )
    )
    adata.uns["pixel_map_params"]["ymin_px"] = int(
        np.floor(
            adata.uns["pixel_map_params"]["scalef"]
            * (
                adata.obsm["spatial"][:, 1].min()
                - adata.uns["pixel_map_params"]["radius"]
            )
        )
    )
    adata.uns["pixel_map_params"]["ymax_px"] = int(
        np.ceil(
            adata.uns["pixel_map_params"]["scalef"]
            * (
                adata.obsm["spatial"][:, 1].max()
                + adata.uns["pixel_map_params"]["radius"]
            )
        )
    )

    print("Creating pixel grid and mapping to nearest barcode coordinates")
    # define grid for pixel space
    grid_y, grid_x = np.mgrid[
        adata.uns["pixel_map_params"]["ymin_px"] : adata.uns["pixel_map_params"][
            "ymax_px"
        ],
        adata.uns["pixel_map_params"]["xmin_px"] : adata.uns["pixel_map_params"][
            "xmax_px"
        ],
    ]
    # map barcodes to pixel coordinates
    pixel_coords = np.column_stack((grid_x.ravel(order="C"), grid_y.ravel(order="C")))
    barcode_list = griddata(
        np.multiply(adata.obsm["spatial"], adata.uns["pixel_map_params"]["scalef"]),
        adata.obs_names,
        (pixel_coords[:, 0], pixel_coords[:, 1]),
        method="nearest",
    )
    # save grid_x and grid_y to adata.uns
    adata.uns["grid_x"], adata.uns["grid_y"] = grid_x, grid_y

    # put results into DataFrame for filtering and reindexing
    print("Saving barcode mapping to adata.uns['pixel_map_df'] and adding metadata")
    adata.uns["pixel_map_df"] = pd.DataFrame(pixel_coords, columns=["x", "y"])
    # add barcodes to long-form dataframe
    adata.uns["pixel_map_df"]["barcode"] = barcode_list
    # merge master df with self.adata.obs for metadata
    adata.uns["pixel_map_df"] = adata.uns["pixel_map_df"].merge(
        adata.obs, how="outer", left_on="barcode", right_index=True
    )
    # filter using label from adata.obs if desired (i.e. "in_tissue")
    if filter_label is not None:
        print(
            "Filtering barcodes using labels in self.adata.obs['{}']".format(
                filter_label
            )
        )
        # set empty pixels (no Visium spot) to "none"
        adata.uns["pixel_map_df"].loc[
            adata.uns["pixel_map_df"][filter_label] == 0,
            "barcode",
        ] = "none"
        # subset the entire anndata object using filter_label
        adata = adata[adata.obs[filter_label] == 1, :].copy()
        print("New size: {} spots x {} genes".format(adata.n_obs, adata.n_vars))

    print("Done!")
    return adata


def trim_image(
    adata, distance_trim=False, threshold=None, channels=None, plot_out=True, **kwargs
):
    """
    Trim pixels in image using pixel map output from Visium barcodes

    Parameters
    ----------
    adata : AnnData.anndata
        The data
    distance_trim : bool
        Manually trim pixels by distance to nearest Visium spot center
    threshold : int or None
        Number of pixels from nearest Visium spot center to call barcode ID. Ignored
        if `distance_trim==False`.
    channels : list of str or None
        Names of image channels in axis order. If None, channels are named "ch_0",
        "ch_1", etc.
    plot_out : bool
        Plot final trimmed image
    **kwargs
        Arguments to pass to `show_pita()` function if `plot_out==True`

    Returns
    -------
    adata.uns["pixel_map_trim"] : np.array
        Contains image with unused pixels set to `np.nan`
    adata.obsm["spatial_trim"] : np.array
        Contains spatial coords with adjusted pixel values after image cropping
    """
    assert (
        adata.uns["pixel_map_params"] is not None
    ), "Pixel map not yet created. Run map_pixels() first."

    print(
        "Cropping image to pixel dimensions and adding values to adata.uns['pixel_map_df']"
    )
    cropped = adata.uns["spatial"][adata.uns["pixel_map_params"]["library_id"]][
        "images"
    ][adata.uns["pixel_map_params"]["img_key"]].transpose(1, 0, 2)[
        int(adata.uns["pixel_map_params"]["xmin_px"]) : int(
            (adata.uns["pixel_map_params"]["xmax_px"])
        ),
        int(adata.uns["pixel_map_params"]["ymin_px"]) : int(
            (adata.uns["pixel_map_params"]["ymax_px"])
        ),
    ]
    # crop x,y coords and save to .obsm as well
    print("Cropping Visium spot coordinates and saving to adata.obsm['spatial_trim']")
    adata.obsm["spatial_trim"] = adata.obsm["spatial"] - np.repeat(
        [
            [
                adata.uns["pixel_map_params"]["xmin_px"],
                adata.uns["pixel_map_params"]["ymin_px"],
            ]
        ],
        adata.obsm["spatial"].shape[0],
        axis=0,
    )

    # manual trimming of pixels by distance if desired
    if distance_trim:
        print("Calculating pixel distances from spot centers for thresholding")
        tree = cKDTree(adata.obsm["spatial"])
        xi = interpnd._ndim_coords_from_arrays(
            (adata.uns["grid_x"], adata.uns["grid_y"]),
            ndim=adata.obsm["spatial"].shape[1],
        )
        dists, _ = tree.query(xi)

        # determine distance threshold
        if threshold is None:
            threshold = int(adata.uns["pixel_map_params"]["ctr_to_vert"] + 1)
            print(
                "Using distance threshold of {} pixels from adata.uns['pixel_map_params']['ctr_to_vert']".format(
                    threshold
                )
            )

        dist_mask = bin_threshold(dists, threshmax=threshold)
        if plot_out:
            # plot pixel distances from spot centers on image
            show_pita(pita=dists, figsize=(4, 4))
            # plot binary thresholded image
            show_pita(pita=dist_mask, figsize=(4, 4))

        print(
            "Trimming pixels by spot distance and adjusting labels in adata.uns['pixel_map_df']"
        )
        mask_df = pd.DataFrame(dist_mask.T.ravel(order="F"), columns=["manual_trim"])
        adata.uns["pixel_map_df"] = adata.uns["pixel_map_df"].merge(
            mask_df, left_index=True, right_index=True
        )
        adata.uns["pixel_map_df"].loc[
            adata.uns["pixel_map_df"]["manual_trim"] == 1, ["barcode"]
        ] = "none"  # set empty pixels to empty barcode
        adata.uns["pixel_map_df"].drop(
            columns="manual_trim", inplace=True
        )  # remove unneeded label

    if channels is None:
        # if channel names not specified, name them numerically
        channels = ["ch_{}".format(x) for x in range(cropped.shape[2])]
    # cast image intensity values to long-form and add to adata.uns["pixel_map_df"]
    rgb = pd.DataFrame(
        np.column_stack(
            [cropped[:, :, x].ravel(order="F") for x in range(cropped.shape[2])]
        ),
        columns=channels,
    )
    adata.uns["pixel_map_df"] = adata.uns["pixel_map_df"].merge(
        rgb, left_index=True, right_index=True
    )
    adata.uns["pixel_map_df"].loc[
        adata.uns["pixel_map_df"]["barcode"] == "none", channels
    ] = np.nan  # set empty pixels to invalid image intensity value

    # calculate mean image values for each channel and create .obsm key
    adata.obsm["image_means"] = (
        adata.uns["pixel_map_df"]
        .loc[adata.uns["pixel_map_df"]["barcode"] != "none", ["barcode"] + channels]
        .groupby("barcode")
        .mean()
        .values
    )

    print(
        "Saving cropped and trimmed image to adata.uns['spatial']['{}']['images']['{}_trim']".format(
            adata.uns["pixel_map_params"]["library_id"],
            adata.uns["pixel_map_params"]["img_key"],
        )
    )
    adata.uns["spatial"][adata.uns["pixel_map_params"]["library_id"]]["images"][
        "{}_trim".format(adata.uns["pixel_map_params"]["img_key"])
    ] = np.dstack(
        [
            adata.uns["pixel_map_df"]
            .pivot(index="y", columns="x", values=[channels[x]])
            .values
            for x in range(len(channels))
        ]
    )
    # save scale factor as well
    adata.uns["spatial"][adata.uns["pixel_map_params"]["library_id"]]["scalefactors"][
        "tissue_{}_trim_scalef".format(adata.uns["pixel_map_params"]["img_key"])
    ] = adata.uns["spatial"][adata.uns["pixel_map_params"]["library_id"]][
        "scalefactors"
    ][
        "tissue_{}_scalef".format(adata.uns["pixel_map_params"]["img_key"])
    ]
    # plot results if desired
    if plot_out:
        if len(channels) == 3:
            show_pita(
                pita=adata.uns["spatial"][adata.uns["pixel_map_params"]["library_id"]][
                    "images"
                ]["{}_trim".format(adata.uns["pixel_map_params"]["img_key"])],
                RGB=True,
                label=channels,
                **kwargs,
            )
        else:
            show_pita(
                pita=adata.uns["spatial"][adata.uns["pixel_map_params"]["library_id"]][
                    "images"
                ]["{}_trim".format(adata.uns["pixel_map_params"]["img_key"])],
                RGB=False,
                label=channels,
                **kwargs,
            )
    print("Done!")


def assemble_pita(
    adata, features=None, use_rep=None, layer=None, plot_out=True, histo=None, **kwargs
):
    """
    Cast feature into pixel space to construct gene expression image ("pita")

    Parameters
    ----------
    adata : AnnData.anndata
        the data
    features : list of int or str
        Names or indices of features to cast onto spot image. If `None`, cast all
        features. If `plot_out`, first feature in list will be plotted. If not
        specified and `plot_out`, first feature (index 0) will be plotted.
    use_rep : str
        Key from `adata.obsm` to use for plotting. If `None`, use `adata.X`.
    layer :str
        Key from `adata.layers` to use for plotting. Ignored if `use_rep` is not `None`
    plot_out : bool
        Show resulting image?
    histo : str or `None`, optional (default=`None`)
        Histology image to show along with pita in gridspec (i.e. "hires",
        "hires_trim", "lowres"). If `None` or if `plot_out`==`False`, ignore.
    **kwargs
        Arguments to pass to `show_pita()` function

    Returns
    -------
    assembled : np.array
        Image of desired expression in pixel space
    """
    assert (
        adata.uns["pixel_map_params"] is not None
    ), "Pixel map not yet created. Run map_pixels() first."

    # coerce features to list if only single string
    if features and not isinstance(features, list):
        features = [features]

    if use_rep is None:
        # use all genes if no gene features specified
        if not features:
            features = adata.var_names  # [adata.var.highly_variable == 1].tolist()
        if layer is None:
            print("Assembling pita with {} features from adata.X".format(len(features)))
            mapper = pd.DataFrame(
                adata.X[:, [adata.var_names.get_loc(x) for x in features]],
                index=adata.obs_names,
            )
        else:
            print(
                "Assembling pita with {} features from adata.layers['{}']".format(
                    len(features), layer
                )
            )
            mapper = pd.DataFrame(
                adata.layers[layer][:, [adata.var_names.get_loc(x) for x in features]],
                index=adata.obs_names,
            )
    elif use_rep in [".obs", "obs"]:
        assert features is not None, "Must provide feature(s) from adata.obs"
        print("Assembling pita with {} features from adata.obs".format(len(features)))
        if all(isinstance(x, int) for x in features):
            mapper = adata.obs.iloc[:, features].copy()
        else:
            mapper = adata.obs[features].copy()
    else:
        if not features:
            print(
                "Assembling pita with {} features from adata.obsm['{}']".format(
                    adata.obsm[use_rep].shape[1], use_rep
                )
            )
            mapper = pd.DataFrame(adata.obsm[use_rep], index=adata.obs_names)
        else:
            assert all(
                isinstance(x, int) for x in features
            ), "Features must be integer indices if using rep from adata.obsm"
            print(
                "Assembling pita with {} features from adata.obsm['{}']".format(
                    len(features), use_rep
                )
            )
            mapper = pd.DataFrame(
                adata.obsm[use_rep][:, features], index=adata.obs_names
            )

    # cast barcodes into pixel dimensions for reindexing
    print("Casting barcodes to pixel dimensions and saving to adata.uns['pixel_map']")
    pixel_map = (
        adata.uns["pixel_map_df"].pivot(index="y", columns="x", values="barcode").values
    )

    assembled = np.array(
        [mapper.reindex(index=pixel_map[x], copy=True) for x in range(len(pixel_map))]
    ).squeeze()

    if plot_out:
        # determine where the histo image is in anndata
        if histo is not None:
            assert (
                histo
                in adata.uns["spatial"][list(adata.uns["spatial"].keys())[0]][
                    "images"
                ].keys()
            ), "Must provide one of {} for histo".format(
                adata.uns["spatial"][list(adata.uns["spatial"].keys())[0]][
                    "images"
                ].keys()
            )
            histo = adata.uns["spatial"][list(adata.uns["spatial"].keys())[0]][
                "images"
            ][histo]
        show_pita(pita=assembled, features=None, histo=histo, **kwargs)
    print("Done!")
    return assembled


def show_pita(
    pita,
    features=None,
    RGB=False,
    histo=None,
    label="feature",
    ncols=4,
    figsize=(7, 7),
    save_to=None,
    **kwargs,
):
    """
    Plot assembled pita using `plt.imshow()`

    Parameters
    ----------
    pita : np.array
        Image of desired expression in pixel space from `.assemble_pita()`
    features : list of int, optional (default=`None`)
        List of features by index to show in plot. If `None`, use all features.
    RGB : bool, optional (default=`False`)
        Treat 3-dimensional array as RGB image
    histo : np.array or `None`, optional (default=`None`)
        Histology image to show along with pita in gridspec. If `None`, ignore.
    label : str
        What to title each panel of the gridspec (i.e. "PC" or "usage") or each
        channel in RGB image. Can also pass list of names e.g. ["NeuN","GFAP",
        "DAPI"] corresponding to channels.
    ncols : int
        Number of columns for gridspec
    figsize : tuple of float
        Size in inches of output figure
    save_to : str or None
        Path to image file to save results. if `None`, show figure.
    **kwargs
        Arguments to pass to `plt.imshow()` function

    Returns
    -------
    Matplotlib object (if plotting one feature or RGB) or gridspec object (for
    multiple features). Saves plot to file if `save_to` is not `None`.
    """
    assert pita.ndim > 1, "Pita does not have enough dimensions: {} given".format(
        pita.ndim
    )
    assert pita.ndim < 4, "Pita has too many dimensions: {} given".format(pita.ndim)
    # if only one feature (2D), plot it quickly
    if (pita.ndim == 2) and histo is None:
        fig = plt.figure(figsize=figsize)
        plt.imshow(pita, **kwargs)
        plt.tick_params(labelbottom=False, labelleft=False)
        sns.despine(bottom=True, left=True)
        plt.colorbar(shrink=0.8)
        plt.tight_layout()
        if save_to:
            plt.savefig(fname=save_to, transparent=True, bbox_inches="tight", dpi=800)
        return fig
    if (pita.ndim == 2) and histo is not None:
        n_rows, n_cols = 1, 2  # two images here, histo and RGB
        fig = plt.figure(figsize=(ncols * n_cols, ncols * n_rows))
        # arrange axes as subplots
        gs = gridspec.GridSpec(n_rows, n_cols, figure=fig)
        # add plots to axes
        ax = plt.subplot(gs[0])
        im = ax.imshow(histo, **kwargs)
        ax.tick_params(labelbottom=False, labelleft=False)
        sns.despine(bottom=True, left=True)
        ax.set_title(
            label="Histology",
            loc="left",
            fontweight="bold",
            fontsize=16,
        )
        ax = plt.subplot(gs[1])
        im = ax.imshow(pita, **kwargs)
        ax.tick_params(labelbottom=False, labelleft=False)
        sns.despine(bottom=True, left=True)
        cbar = plt.colorbar(im, shrink=0.8)
        fig.tight_layout()
        if save_to:
            plt.savefig(fname=save_to, transparent=True, bbox_inches="tight", dpi=800)
        return fig
    if RGB:
        # if third dim has 3 features, treat as RGB and plot it quickly
        assert (pita.ndim == 3) & (
            pita.shape[2] == 3
        ), "Need 3 dimensions and 3 given features for an RGB image; shape = {}; features given = {}".format(
            pita.shape, len(features)
        )
        print("Plotting pita as RGB image")
        if isinstance(label, str):
            # if label is single string, name channels numerically
            channels = ["{}_{}".format(label, x) for x in range(pita.shape[2])]
        else:
            assert (
                len(label) == 3
            ), "Please pass 3 channel names for RGB plot; {} labels given: {}".format(
                len(label), label
            )
            channels = label
        if histo is not None:
            n_rows, n_cols = 1, 2  # two images here, histo and RGB
            fig = plt.figure(figsize=(ncols * n_cols, ncols * n_rows))
            # arrange axes as subplots
            gs = gridspec.GridSpec(n_rows, n_cols, figure=fig)
            # add plots to axes
            ax = plt.subplot(gs[0])
            im = ax.imshow(histo, **kwargs)
            ax.tick_params(labelbottom=False, labelleft=False)
            sns.despine(bottom=True, left=True)
            ax.set_title(
                label="Histology",
                loc="left",
                fontweight="bold",
                fontsize=16,
            )
            ax = plt.subplot(gs[1])
            im = ax.imshow(pita, **kwargs)
            # add legend for channel IDs
            custom_lines = [
                Line2D([0], [0], color=(1, 0, 0), lw=5),
                Line2D([0], [0], color=(0, 1, 0), lw=5),
                Line2D([0], [0], color=(0, 0, 1), lw=5),
            ]
            plt.legend(custom_lines, channels, fontsize="medium")
            ax.tick_params(labelbottom=False, labelleft=False)
            sns.despine(bottom=True, left=True)
            fig.tight_layout()
            if save_to:
                plt.savefig(
                    fname=save_to, transparent=True, bbox_inches="tight", dpi=800
                )
            return fig
        else:
            fig = plt.figure(figsize=figsize)
            plt.imshow(pita, **kwargs)
            # add legend for channel IDs
            custom_lines = [
                Line2D([0], [0], color=(1, 0, 0), lw=5),
                Line2D([0], [0], color=(0, 1, 0), lw=5),
                Line2D([0], [0], color=(0, 0, 1), lw=5),
            ]
            plt.legend(custom_lines, channels, fontsize="medium")
            plt.tick_params(labelbottom=False, labelleft=False)
            sns.despine(bottom=True, left=True)
            plt.tight_layout()
            if save_to:
                plt.savefig(
                    fname=save_to, transparent=True, bbox_inches="tight", dpi=800
                )
            return fig
    # if pita has multiple features, plot them in gridspec
    if isinstance(features, int):  # force features into list if single integer
        features = [features]
    # if no features are given, use all of them
    if features is None:
        features = [x + 1 for x in range(pita.shape[2])]
    else:
        assert (
            pita.ndim > 2
        ), "Not enough features in pita: shape {}, expecting 3rd dim with length {}".format(
            pita.shape, len(features)
        )
        assert (
            len(features) <= pita.shape[2]
        ), "Too many features given: pita has {}, expected {}".format(
            pita.shape[2], len(features)
        )
    if isinstance(label, str):
        # if label is single string, name channels numerically
        labels = ["{}_{}".format(label, x) for x in features]
    else:
        assert len(label) == len(
            features
        ), "Please provide the same number of labels as features; {} labels given, {} features given.".format(
            len(label), len(features)
        )
        labels = label
    # calculate gridspec dimensions
    if histo is not None:
        labels = ["Histology"] + labels  # append histo to front of labels
        if len(features) + 1 <= ncols:
            n_rows, n_cols = 1, len(features) + 1
        else:
            n_rows, n_cols = ceil((len(features) + 1) / ncols), ncols
    else:
        if len(features) <= ncols:
            n_rows, n_cols = 1, len(features)
        else:
            n_rows, n_cols = ceil(len(features) / ncols), ncols
    fig = plt.figure(figsize=(ncols * n_cols, ncols * n_rows))
    # arrange axes as subplots
    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig)
    # add plots to axes
    i = 0
    if histo is not None:
        # add histology plot to first axes
        ax = plt.subplot(gs[i])
        im = ax.imshow(histo, **kwargs)
        ax.tick_params(labelbottom=False, labelleft=False)
        sns.despine(bottom=True, left=True)
        ax.set_title(
            label=labels[i],
            loc="left",
            fontweight="bold",
            fontsize=16,
        )
        i = i + 1
    for feature in features:
        ax = plt.subplot(gs[i])
        im = ax.imshow(pita[:, :, feature - 1], **kwargs)
        ax.tick_params(labelbottom=False, labelleft=False)
        sns.despine(bottom=True, left=True)
        ax.set_title(
            label=labels[i],
            loc="left",
            fontweight="bold",
            fontsize=16,
        )
        cbar = plt.colorbar(im, shrink=0.8)
        i = i + 1
    fig.tight_layout()
    if save_to:
        plt.savefig(fname=save_to, transparent=True, bbox_inches="tight", dpi=800)
    return fig
