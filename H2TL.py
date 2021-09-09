# -*- coding: utf-8 -*-
"""
Classes for assigning tissue region IDs to multiplex immunofluorescence (MxIF) or 10X 
Visium spatial transcriptomic (ST) and histological imaging data

@author: C Heiser
"""
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans


class tissue_labeler:
    """
    Master tissue region labeling class
    """

    def __init__(self):
        """
        Initialize tissue labeler parent class
        """
        self.cluster_data = None  # start out with no data to cluster on

    def find_tissue_regions(self, k, random_state=18):
        """
        Perform tissue-level clustering and label pixels in the corresponding
        `anndata` objects.

        Parameters
        ----------
        k : int
            Number of tissue regions to define
        random_state : int, optional (default=18)
            Seed for k-means clustering model.

        Returns
        -------
        Does not return anything. `self.kmeans` contains trained `sklearn` clustering
        model. Parameters are also captured as attributes for posterity.
        """
        if self.cluster_data is None:
            print("No cluster data found. Run prep_cluster_data() first.")
            pass
        # save the hyperparams as object attributes
        self.k = k
        self.random_state = random_state
        print("Performing k-means clustering with {} target clusters".format(self.k))
        self.kmeans = KMeans(n_clusters=k, random_state=random_state).fit(
            self.cluster_data
        )


class st_labeler(tissue_labeler):
    """
    Tissue region labeling class for spatial transcriptomics (ST) data
    """

    def __init__(self, adatas):
        """
        Initialize ST tissue labeler class

        Parameters
        ----------
        adatas : list of anndata.AnnData
            Single anndata object or list of objects to label consensus tissue regions

        Returns
        -------
        Does not return anything. `self.adatas` attribute is updated,
        `self.cluster_data` attribute is initiated as `None`.
        """
        tissue_labeler.__init__(self)  # initialize parent class
        if not isinstance(adatas, list):  # force single anndata object to list
            adatas = [adatas]
        print("Initiating ST labeler with {} anndata objects".format(len(adatas)))
        self.adatas = adatas

    def prep_cluster_data(
        self, use_rep, features, blur_pix=2, histo=False, fluor_channels=None
    ):
        """
        Prepare master dataframe for tissue-level clustering

        Parameters
        ----------
        use_rep : str
            Representation from `adata.obsm` to use as clustering data (e.g. "X_pca")
        features : list of int
            List of features (1-indexed) to use from `adata.obsm[use_rep]` (e.g.
            [1,2,3,4,5] to use first 5 principal components when `use_rep`="X_pca)
        blur_pix : int, optional (default=2)
            Radius of nearest spatial transcriptomics spots to blur features by for
            capturing regional information. Assumes hexagonal spot grid (10X Genomics
            Visium platform).
        histo : bool, optional (default `False`)
            Use histology data from Visium anndata object (R,G,B brightfield features)
            in addition to `adata.obsm[use_rep]`? If fluorescent imaging data rather
            than brightfield, use `fluor_channels` argument instead.
        fluor_channels : list of int or None, optional (default `None`)
            Channels from fluorescent image to use for model training (e.g. [1,3] for
            channels 1 and 3 of Visium fluorescent imaging data). If `None`, do not
            use imaging data for training.

        Returns
        -------
        Does not return anything. `self.adatas` are updated, adding "blur_*" features
        to `.obs`. `self.cluster_data` becomes master `np.array` for cluster training.
        Parameters are also captured as attributes for posterity.
        """
        if self.cluster_data is not None:
            print("WARNING: overwriting existing cluster data")
            self.cluster_data = None
        # save the hyperparams as object attributes
        self.rep = use_rep
        self.features = features
        self.histo = histo
        self.blur_pix = blur_pix
        indices = [
            i - 1 for i in self.features
        ]  # go back to zero indexing from 1-indexed feature list
        for adata_i, adata in enumerate(self.adatas):
            print(
                "Collecting {} features from .obsm[{}] for adata #{}".format(
                    len(self.features), self.rep, adata_i
                )
            )
            tmp = adata.obs[["array_row", "array_col"]].copy()
            tmp[[use_rep + "_{}".format(x) for x in self.features]] = adata.obsm[
                use_rep
            ][:, indices]
            if histo:
                assert (
                    fluor_channels is None
                ), "If histo is True, fluor_channels must be None. \
                    Histology specifies brightfield H&E with three (3) features."
                print(
                    "Adding mean RGB histology features for adata #{}".format(adata_i)
                )
                tmp[["R_mean", "G_mean", "B_mean"]] = adata.obsm["image_means"]
            if fluor_channels:
                assert (
                    histo is False
                ), "If fluorescence channels are given, histo must be False. \
                    Histology specifies brightfield H&E with three (3) features."
                print(
                    "Adding mean fluorescent channels {} for adata #{}".format(
                        fluor_channels, adata_i
                    )
                )
                tmp[["ch_{}_mean".format(x) for x in fluor_channels]] = adata.obsm[
                    "image_means"
                ][:, fluor_channels]
            tmp2 = (
                tmp.copy()
            )  # copy of temporary dataframe for dropping blurred features into
            cols = tmp.columns[
                ~tmp.columns.str.startswith("array_")
            ]  # get names of training features to blur
            # perform blurring by nearest spot neighbors
            print("Blurring training features for adata #{}".format(adata_i))
            for y in range(tmp.array_row.min(), tmp.array_row.max() + 1):
                for x in range(tmp.array_col.min(), tmp.array_col.max() + 1):
                    vals = tmp.loc[
                        tmp.array_row.isin(
                            [i for i in range(y - blur_pix, y + blur_pix + 1)]
                        )
                        & tmp.array_col.isin(
                            [i for i in range(x - 2 * blur_pix, x + 2 * blur_pix + 1)]
                        ),
                        :,
                    ]
                    vals = vals.loc[:, cols].mean()
                    tmp2.loc[
                        tmp2.array_row.isin([y]) & tmp2.array_col.isin([x]), cols
                    ] = vals.values
            # add blurred features to anndata object
            adata.obs[["blur_" + x for x in cols]] = tmp2.loc[:, cols].values
            # append blurred features to cluster_data df for cluster training
            if self.cluster_data is None:
                self.cluster_data = tmp2.loc[:, cols].copy()
            else:
                self.cluster_data = pd.concat([self.cluster_data, tmp2.loc[:, cols]])
        self.cluster_data = self.cluster_data.values
        print("Collected clustering data of shape: {}".format(self.cluster_data.shape))

    def label_tissue_regions(self, k, random_state=18):
        """
        Perform tissue-level clustering and label pixels in the corresponding
        `anndata` objects.

        Parameters
        ----------
        k : int
            Number of tissue regions to define
        random_state : int, optional (default=18)
            Seed for k-means clustering model.

        Returns
        -------
        Does not return anything. `self.adatas` are updated, adding "tissue_ID" field
        to `.obs`. `self.kmeans` contains trained `sklearn` clustering model.
        Parameters are also captured as attributes for posterity.
        """
        # call k-means model from parent class
        self.find_tissue_regions(k=k, random_state=random_state)
        # loop through anndata object and add tissue labels to adata.obs dataframe
        start = 0
        print("Adding tissue_ID label to anndata objects")
        for i in range(len(self.adatas)):
            IDs = self.kmeans.labels_
            self.adatas[i].obs["tissue_ID"] = IDs[start : start + self.adatas[i].n_obs]
            self.adatas[i].obs["tissue_ID"] = (
                self.adatas[i].obs["tissue_ID"].astype("category")
            )
            self.adatas[i].obs["tissue_ID"] = (
                self.adatas[i].obs["tissue_ID"].cat.set_categories(np.unique(IDs))
            )
            start += self.adatas[i].n_obs


class mxif_labeler(tissue_labeler):
    """
    Tissue region labeling class for multiplex immunofluorescence (MxIF) data
    """

    def __init__(self, images):
        """
        Initialize MxIF tissue labeler class

        Parameters
        ----------
        images : list of [img class?]
            Single img object or list of objects to label consensus tissue regions

        Returns
        -------
        Does not return anything. `self.images` attribute is updated,
        `self.cluster_data` attribute is initiated as `None`.
        """
        tissue_labeler.__init__(self)  # initialize parent class
        if not isinstance(images, list):  # force single image object to list
            images = [images]
        print("Initiating MxIF labeler with {} images".format(len(images)))
        self.images = images

    def prep_cluster_data(self, features, downsample_factor=8, blur_pix=50):
        """
        Prepare master dataframe for tissue-level clustering

        Parameters
        ----------
        features : list of int or str
            Indices or names of MxIF channels to use for tissue labeling
        downsample_factor : int
            Factor by which to downsample images from their original resolution
        blur_pix : int, optional (default=50)
            Radius of nearest image pixels to blur features by for capturing regional
            information.

        Returns
        -------
        Does not return anything. `self.images` are downsampled and blurred according
        to user parameters. `self.cluster_data` becomes master `np.array` for cluster
        training. Parameters are also captured as attributes for posterity.
        """
        if self.cluster_data is not None:
            print("WARNING: overwriting existing cluster data")
            self.cluster_data = None
        # save the hyperparams as object attributes
        self.features = features
        self.downsample_factor = downsample_factor
        self.blur_pix = blur_pix
        # perform image downsampling, blurring, subsampling, and compile cluster_data

    def label_tissue_regions(self, k, random_state=18):
        """
        Perform tissue-level clustering and label pixels in the corresponding
        images.

        Parameters
        ----------
        k : int
            Number of tissue regions to define
        random_state : int, optional (default=18)
            Seed for k-means clustering model.

        Returns
        -------
        Does not return anything. `self.tissue_ID` is added, containing image with
        final tissue region IDs. `self.kmeans` contains trained `sklearn` clustering
        model. Parameters are also captured as attributes for posterity.
        """
        # call k-means model from parent class
        self.find_tissue_regions(k=k, random_state=random_state)
        # loop through image objects and create tissue label images
