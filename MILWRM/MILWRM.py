# -*- coding: utf-8 -*-
"""
Classes for assigning tissue domain IDs to multiplex immunofluorescence (MxIF) or 10X 
Visium spatial transcriptomic (ST) and histological imaging data
"""
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from math import ceil
from joblib import Parallel, delayed
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from skimage.filters import gaussian

from .MxIF import checktype


def kMeansRes(scaled_data, k, alpha_k=0.02, random_state=18):
    """
    Calculates inertia value for a given k value by fitting k-means model to scaled data

    Adapted from
    https://towardsdatascience.com/an-approach-for-choosing-number-of-clusters-for-k-means-c28e614ecb2c

    Parameters
    ----------
    scaled_data: np.array
        Scaled data. Rows are samples and columns are features for clustering
    k: int
        Current k for applying KMeans
    alpha_k: float
        Manually tuned factor that gives penalty to the number of clusters

    Returns
    -------
    scaled_inertia: float
        Scaled inertia value for current k
    """
    inertia_o = np.square((scaled_data - scaled_data.mean(axis=0))).sum()
    # fit k-means
    kmeans = KMeans(n_clusters=k, random_state=random_state).fit(scaled_data)
    scaled_inertia = kmeans.inertia_ / inertia_o + alpha_k * k
    return scaled_inertia


def chooseBestKforKMeansParallel(scaled_data, k_range, n_jobs=-1, **kwargs):
    """
    Determines optimal k value by fitting k-means models to scaled data and minimizing
    scaled inertia

    Adapted from
    https://towardsdatascience.com/an-approach-for-choosing-number-of-clusters-for-k-means-c28e614ecb2c

    Parameters
    ----------
    scaled_data: np.array
        Scaled data. Rows are samples and columns are features for clustering.
    k_range: list of int
        k range for applying KMeans
    n_jobs : int
        Number of cores to parallelize k-choosing across
    **kwargs
        Arguments to pass to `kMeansRes()` (i.e. `alpha_k`, `random_state`)

    Returns
    -------
    best_k: int
        Chosen value of k out of the given k range. Chosen k is k with the minimum
        scaled inertia value.
    results: pd.DataFrame
        Adjusted inertia value for each k in k_range
    """
    ans = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(kMeansRes)(scaled_data, k, **kwargs) for k in k_range
    )
    ans = list(zip(k_range, ans))
    results = pd.DataFrame(ans, columns=["k", "Scaled Inertia"]).set_index("k")
    best_k = results.idxmin()[0]
    return best_k, results


def prep_data_single_sample_st(
    adata, adata_i, use_rep, features, blur_pix, histo, fluor_channels
):
    """
    Prepare dataframe for tissue-level clustering from a single AnnData sample

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object containing Visium data
    adata_i : int
        Index of AnnData object for identification within `st_labeler` object
    use_rep : str
        Representation from `adata.obsm` to use as clustering data (e.g. "X_pca")
    features : list of int or None, optional (default=`None`)
        List of features to use from `adata.obsm[use_rep]` (e.g. [0,1,2,3,4] to
        use first 5 principal components when `use_rep`="X_pca"). If `None`, use
        all features from `adata.obsm[use_rep]`
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
    pd.DataFrame
        Clustering data from `adata.obsm[use_rep]`
    """
    tmp = adata.obs[["array_row", "array_col"]].copy()
    tmp[[use_rep + "_{}".format(x) for x in features]] = adata.obsm[use_rep][
        :, features
    ]
    if histo:
        assert (
            fluor_channels is None
        ), "If histo is True, fluor_channels must be None. \
            Histology specifies brightfield H&E with three (3) features."
        print("Adding mean RGB histology features for adata #{}".format(adata_i))
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
    tmp2 = tmp.copy()  # copy of temporary dataframe for dropping blurred features into
    cols = tmp.columns[
        ~tmp.columns.str.startswith("array_")
    ]  # get names of training features to blur
    # perform blurring by nearest spot neighbors
    for y in range(tmp.array_row.min(), tmp.array_row.max() + 1):
        for x in range(tmp.array_col.min(), tmp.array_col.max() + 1):
            vals = tmp.loc[
                tmp.array_row.isin([i for i in range(y - blur_pix, y + blur_pix + 1)])
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
    return tmp2.loc[:, cols]


def prep_data_single_sample_mxif(image, features, downsample_factor, sigma, fract):
    """
    Prepare dataframe for tissue-level clustering from a single MxIF sample

    Parameters
    ----------
    image : MILWRM.MxIF.img
        Image object containing MxIF data
    features : list of int or str
        Indices or names of MxIF channels to use for tissue labeling
    downsample_factor : int
        Factor by which to downsample images from their original resolution
    sigma : float, optional (default=2)
        Standard deviation of Gaussian kernel for blurring
    fract : float, optional (default=0.2)
        Fraction of cluster data from each image to randomly select for model
        building

    Returns
    -------
    image : MILWRM.MxIF.img
        Image object containing MxIF data downsampled and blurred
    tmp : np.array
        Clustering data from `image`
    """
    # downsample image
    image.downsample(fact=downsample_factor, func=np.mean)
    # normalize and log-transform image
    image.log_normalize(pseudoval=1, mask=True)
    # blur downsampled image
    image.img = gaussian(image.img, sigma=sigma, multichannel=True)
    # get list of int for features
    if isinstance(features, int):  # force features into list if single integer
        features = [features]
    if isinstance(features, str):  # force features into int if single string
        features = [image.ch.index(features)]
    if checktype(features):  # force features into list of int if list of strings
        features = [image.ch.index(x) for x in features]
    if features is None:  # if no features are given, use all of them
        features = [x for x in range(image.n_ch)]
    # get cluster data for image_i
    tmp = []
    for i in range(image.img.shape[2]):
        tmp.append(image.img[:, :, i][image.mask != 0])
    tmp = np.column_stack(tmp)
    # select cluster data
    i = np.random.choice(tmp.shape[0], int(tmp.shape[0] * fract))
    tmp = tmp[np.ix_(i, features)]
    return image, tmp


def add_tissue_ID_single_sample_mxif(image, features, kmeans):
    """
    Label pixels in a single MxIF sample with kmeans results

    Parameters
    ----------
    image : MILWRM.MxIF.img
        AnnData object containing Visium data
    features : list of int or str
        Indices or names of MxIF channels to use for tissue labeling
    kmeans : sklearn.kmeans
        Trained k-means model

    Returns
    -------
    tID : np.array
        Image where pixel values are kmeans cluster IDs
    """
    if isinstance(features, int):  # force features into list if single integer
        features = [features]
    if isinstance(features, str):  # force features into int if single string
        features = [image.ch.index(features)]
    if checktype(features):  # force features into list of int if list of strings
        features = [image.ch.index(x) for x in features]
    if features is None:  # if no features are given, use all of them
        features = [x for x in range(image.n_ch)]
    # subset to features used in prep_cluster_data
    tmp = image.img[:, :, features]
    tID = np.zeros(tmp.shape[:2])
    for x in range(tmp.shape[0]):
        for y in range(tmp.shape[1]):
            tID[x, y] = kmeans.predict(tmp[x, y, :].reshape(1, -1))
    tID[image.mask == 0] = np.nan  # set masked-out pixels to NaN
    return tID


class tissue_labeler:
    """
    Master tissue domain labeling class
    """

    def __init__(self):
        """
        Initialize tissue labeler parent class
        """
        self.cluster_data = None  # start out with no data to cluster on
        self.k = None  # start out with no k value

    def find_optimal_k(self, plot_out=False, alpha=0.05, random_state=18, n_jobs=-1):
        """
        Uses scaled inertia to decide on k clusters for clustering in the
        corresponding `anndata` objects

        Parameters
        ----------
        plot_out : boolean, optional (default=FALSE)
            Determines if scaled inertia graph should be output
        alpha: float
            Manually tuned factor on [0.0, 1.0] that penalizes the number of clusters
        random_state : int, optional (default=18)
            Seed for k-means clustering models
        n_jobs : int
            Number of cores to parallelize k-choosing across

        Returns
        -------
        Does not return anything. `self.k` contains integer value for number of
        clusters. Parameters are also captured as attributes for posterity.
        """
        if self.cluster_data is None:
            raise Exception("No cluster data found. Run prep_cluster_data() first.")
        self.random_state = random_state

        k_range = range(2, 21)  # choose k range
        # compute scaled inertia
        best_k, results = chooseBestKforKMeansParallel(
            self.cluster_data,
            k_range,
            n_jobs=n_jobs,
            random_state=random_state,
            alpha_k=alpha,
        )
        if plot_out:
            # plot the results
            plt.figure(figsize=(7, 4))
            plt.plot(results, "o")
            plt.title("Adjusted Inertia for each K")
            plt.xlabel("K")
            plt.ylabel("Adjusted Inertia")
            plt.xticks(range(2, 21, 1))
            plt.show()
        # save optimal k to object
        print("The optimal number of clusters is {}".format(best_k))
        self.k = best_k

    def find_tissue_regions(self, k=None, random_state=18):
        """
        Perform tissue-level clustering and label pixels in the corresponding
        `anndata` objects.

        Parameters
        ----------
        k : int, optional (default=None)
            Number of tissue domains to define
        random_state : int, optional (default=18)
            Seed for k-means clustering model.

        Returns
        -------
        Does not return anything. `self.kmeans` contains trained `sklearn` clustering
        model. Parameters are also captured as attributes for posterity.
        """
        if self.cluster_data is None:
            raise Exception("No cluster data found. Run prep_cluster_data() first.")
        if k is None and self.k is None:
            raise Exception(
                "No k found or provided. Run find_optimal_k() first or pass a k value."
            )
        if k is not None:
            print("Overriding optimal k value with k={}.".format(k))
            self.k = k
        # save the hyperparams as object attributes
        self.random_state = random_state
        print("Performing k-means clustering with {} target clusters".format(self.k))
        self.kmeans = KMeans(n_clusters=self.k, random_state=random_state).fit(
            self.cluster_data
        )

    def plot_feature_proportions(self, figsize=(10, 7), save_to=None):
        """
        Plots contributions of each training feature to k-means cluster centers as
        percentages of total

        Parameters
        ----------
        figsize : tuple of float, optional (default=(10,7))
            Size of matplotlib figure
        save_to : str, optional (default=`None`)
            Path to image file to save plot

        Returns
        -------
        `plt.figure` if `save_to` is `None`, else saves plot to file
        """
        assert (
            self.kmeans is not None
        ), "No cluster results found. Run \
        label_tissue_regions() first."
        if "st_labeler" in str(self.__class__):
            labels = [self.rep + "_" + str(x) for x in self.features]
            if self.histo:
                labels = labels + ["R", "G", "B"]
            if self.flour_channels is not None:
                labels = labels + ["ch_" + str(x) for x in self.flour_channels]
        elif "mxif_labeler" in str(self.__class__):
            labels = self.model_features
        # create pandas df and calculate percentages of total
        ctr_df = pd.DataFrame(self.kmeans.cluster_centers_, columns=labels)
        totals = ctr_df.sum(axis=1)
        ctr_df_prop = ctr_df.div(totals, axis=0).multiply(100)
        # make plot
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ctr_df_prop.plot.bar(stacked=True, ax=ax, width=0.85)
        for p in ax.patches:
            ax.annotate(
                "{} %".format(str(np.round(p.get_height(), 2))),
                (p.get_x() + 0.05, p.get_y() + (p.get_height() * 0.4)),
                fontsize=10,
            )
        plt.ylim([0, 100])
        plt.xlabel("tissue_ID")
        plt.xticks(rotation=0)
        plt.ylabel("% K-Means Loading")
        plt.legend(bbox_to_anchor=(1, 1), loc="upper left", title="Feature")
        plt.tight_layout()
        if save_to is not None:
            print("Saving feature proportions to {}".format(save_to))
            plt.savefig(save_to)
        else:
            return fig

    def plot_feature_loadings(
        self, ncols=None, nfeatures=None, figsize=(5, 5), save_to=None
    ):
        """
        Plots contributions of each training feature to k-means cluster centers

        Parameters
        ----------
        ncols : int, optional (default=`None`)
            Number of columns for gridspec. If `None`, uses number of tissue domains k.
        nfeatures : int, optional (default=`None`)
            Number of top-loaded features to show for each tissue domain
        figsize : tuple of float, optional (default=(5,5))
            Size of matplotlib figure
        save_to : str, optional (default=`None`)
            Path to image file to save plot

        Returns
        -------
        `gridspec.GridSpec` if `save_to` is `None`, else saves plot to file
        """
        assert (
            self.kmeans is not None
        ), "No cluster results found. Run \
        label_tissue_regions() first."
        if "st_labeler" in str(self.__class__):
            labels = [self.rep + "_" + str(x) for x in self.features]
            if self.histo:
                labels = labels + ["R", "G", "B"]
            if self.flour_channels is not None:
                labels = labels + ["ch_" + str(x) for x in self.flour_channels]
        elif "mxif_labeler" in str(self.__class__):
            labels = self.model_features
        titles = [
            "tissue_ID " + str(x) for x in range(self.kmeans.cluster_centers_.shape[0])
        ]
        if nfeatures is None:
            nfeatures = len(labels)
        scores = self.kmeans.cluster_centers_.copy()
        n_panels = len(titles)
        if ncols is None:
            ncols = len(titles)
        if n_panels <= ncols:
            n_rows, n_cols = 1, n_panels
        else:
            n_rows, n_cols = ceil(n_panels / ncols), ncols
        fig = plt.figure(figsize=(n_cols * figsize[0], n_rows * figsize[1]))
        left, bottom = 0.1 / n_cols, 0.1 / n_rows
        gs = gridspec.GridSpec(
            nrows=n_rows,
            ncols=n_cols,
            wspace=0.1,
            left=left,
            bottom=bottom,
            right=1 - (n_cols - 1) * left - 0.01 / n_cols,
            top=1 - (n_rows - 1) * bottom - 0.1 / n_rows,
        )
        for iscore, score in enumerate(scores):
            plt.subplot(gs[iscore])
            indices = np.argsort(score)[::-1][: nfeatures + 1]
            for ig, g in enumerate(indices[::-1]):
                plt.text(
                    x=score[g],
                    y=ig,
                    s=labels[g],
                    color="black",
                    # rotation="vertical",
                    verticalalignment="center",
                    horizontalalignment="right",
                    fontsize="medium",
                    fontstyle="italic",
                )
            plt.title(titles[iscore], fontsize="x-large")
            plt.ylim(-0.9, ig + 0.9)
            score_min, score_max = np.min(score[indices]), np.max(score[indices])
            plt.xlim(
                (0.95 if score_min > 0 else 1.05) * score_min,
                (1.05 if score_max > 0 else 0.95) * score_max,
            )
            plt.xticks(rotation=45)
            plt.tick_params(labelsize="medium")
            plt.tick_params(
                axis="y",  # changes apply to the y-axis
                which="both",  # both major and minor ticks are affected
                left=False,
                right=False,
                labelleft=False,
            )
            plt.grid(False)
        gs.tight_layout(fig)
        if save_to is not None:
            print("Saving feature loadings to {}".format(save_to))
            plt.savefig(save_to)
        else:
            return gs


class st_labeler(tissue_labeler):
    """
    Tissue domain labeling class for spatial transcriptomics (ST) data
    """

    def __init__(self, adatas):
        """
        Initialize ST tissue labeler class

        Parameters
        ----------
        adatas : list of anndata.AnnData
            Single anndata object or list of objects to label consensus tissue domains

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
        self,
        use_rep,
        features=None,
        blur_pix=2,
        histo=False,
        fluor_channels=None,
        n_jobs=-1,
    ):
        """
        Prepare master dataframe for tissue-level clustering

        Parameters
        ----------
        use_rep : str
            Representation from `adata.obsm` to use as clustering data (e.g. "X_pca")
        features : list of int or None, optional (default=`None`)
            List of features to use from `adata.obsm[use_rep]` (e.g. [0,1,2,3,4] to
            use first 5 principal components when `use_rep`="X_pca"). If `None`, use
            all features from `adata.obsm[use_rep]`
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
        n_jobs : int, optional (default=-1)
            Number of cores to parallelize over. Default all available cores.

        Returns
        -------
        Does not return anything. `self.adatas` are updated, adding "blur_*" features
        to `.obs`. `self.cluster_data` becomes master `np.array` for cluster training.
        Parameters are also captured as attributes for posterity.
        """
        if self.cluster_data is not None:
            print("WARNING: overwriting existing cluster data")
            self.cluster_data = None
        if features is None:
            self.features = [x for x in range(self.adatas[0].obsm[use_rep].shape[1])]
        else:
            self.features = features
        # save the hyperparams as object attributes
        self.rep = use_rep
        self.histo = histo
        self.flour_channels = fluor_channels
        self.blur_pix = blur_pix
        # collect clustering data from self.adatas in parallel
        print(
            "Collecting and blurring {} features from .obsm[{}]...".format(
                len(self.features),
                use_rep,
            )
        )
        cluster_data = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(prep_data_single_sample_st)(
                adata, adata_i, use_rep, self.features, blur_pix, histo, fluor_channels
            )
            for adata_i, adata in enumerate(self.adatas)
        )
        # concatenate blurred features into cluster_data df for cluster training
        self.cluster_data = pd.concat(cluster_data)
        # perform min-max scaling on final cluster data
        mms = MinMaxScaler()
        unscaled_data = self.cluster_data.values
        self.cluster_data = mms.fit_transform(unscaled_data)
        print("Collected clustering data of shape: {}".format(self.cluster_data.shape))

    def label_tissue_regions(
        self, k=None, alpha=0.05, plot_out=True, random_state=18, n_jobs=-1
    ):
        """
        Perform tissue-level clustering and label pixels in the corresponding
        `anndata` objects.

        Parameters
        ----------
        k : int, optional (default=None)
            Number of tissue regions to define
        alpha: float
            Manually tuned factor on [0.0, 1.0] that penalizes the number of clusters
        plot_out : boolean, optional (default=True)
            Determines if scaled inertia plot should be output
        random_state : int, optional (default=18)
            Seed for k-means clustering model.
        n_jobs : int
            Number of cores to parallelize k-choosing across

        Returns
        -------
        Does not return anything. `self.adatas` are updated, adding "tissue_ID" field
        to `.obs`. `self.kmeans` contains trained `sklearn` clustering model.
        Parameters are also captured as attributes for posterity.
        """
        # find optimal k with parent class
        if k is None:
            print("Determining optimal cluster number k via scaled inertia")
            self.find_optimal_k(
                plot_out=plot_out, alpha=alpha, random_state=random_state, n_jobs=n_jobs
            )
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
    Tissue domain labeling class for multiplex immunofluorescence (MxIF) data
    """

    def __init__(self, images):
        """
        Initialize MxIF tissue labeler class

        Parameters
        ----------
        images : list of MILWRM.MxIF.img
            Single MILWRM.MxIF.img object or list of objects to label consensus
            tissue domains

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

    def prep_cluster_data(
        self, features, downsample_factor=8, sigma=2, fract=0.2, n_jobs=-1
    ):
        """
        Prepare master dataframe for tissue-level clustering

        Parameters
        ----------
        features : list of int or str
            Indices or names of MxIF channels to use for tissue labeling
        downsample_factor : int
            Factor by which to downsample images from their original resolution
        sigma : float, optional (default=2)
            Standard deviation of Gaussian kernel for blurring
        fract : float, optional (default=0.2)
            Fraction of cluster data from each image to randomly select for model
            building
        n_jobs : int, optional (default=-1)
            Number of cores to parallelize over. Default all available cores.

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
        self.model_features = features
        self.downsample_factor = downsample_factor
        self.sigma = sigma
        # downsampling, blurring, subsampling, and compiling cluster_data in parallel
        print(
            "Downsampling, log-normalizing, and blurring {} features from {} images...".format(
                len(features),
                len(self.images),
            )
        )
        out = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(prep_data_single_sample_mxif)(
                image, features, downsample_factor, sigma, fract
            )
            for image in self.images
        )
        # unpack results from parallel process
        self.images = [x[0] for x in out]
        cluster_data = [x[1] for x in out]
        # concatenate blurred features into cluster_data df for cluster training
        self.cluster_data = np.row_stack(cluster_data)
        # perform min-max scaling on final cluster data
        mms = MinMaxScaler()
        unscaled_data = self.cluster_data
        self.cluster_data = mms.fit_transform(unscaled_data)
        print("Collected clustering data of shape: {}".format(self.cluster_data.shape))

    def label_tissue_regions(
        self, k=None, alpha=0.05, plot_out=True, random_state=18, n_jobs=-1
    ):
        """
        Perform tissue-level clustering and label pixels in the corresponding
        images.

        Parameters
        ----------
        k : int, optional (default=None)
            Number of tissue regions to define
        alpha: float
            Manually tuned factor on [0.0, 1.0] that penalizes the number of clusters
        plot_out : boolean, optional (default=True)
            Determines if scaled inertia plot should be output
        random_state : int, optional (default=18)
            Seed for k-means clustering model
        n_jobs : int
            Number of cores to parallelize k-choosing and tissue ID assignment across.
            Default all available cores.

        Returns
        -------
        Does not return anything. `self.tissue_ID` is added, containing image with
        final tissue region IDs. `self.kmeans` contains trained `sklearn` clustering
        model. Parameters are also captured as attributes for posterity.
        """
        # find optimal k with parent class
        if k is None:
            print("Determining optimal cluster number k via scaled inertia")
            self.find_optimal_k(
                alpha=alpha, plot_out=plot_out, random_state=random_state, n_jobs=n_jobs
            )
        # call k-means model from parent class
        self.find_tissue_regions(random_state=random_state)
        # loop through image objects and create tissue label images
        print("Creating tissue_ID images for image objects...")
        self.tissue_IDs = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(add_tissue_ID_single_sample_mxif)(
                image, self.model_features, self.kmeans
            )
            for image in self.images
        )
