# -*- coding: utf-8 -*-
"""
Classes for assigning tissue domain IDs to multiplex immunofluorescence (MxIF) or 10X 
Visium spatial transcriptomic (ST) and histological imaging data
"""
import os
from tkinter import E
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools
import seaborn as sns

sns.set_style("white")

from math import ceil
from joblib import Parallel, delayed
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from .MxIF import checktype, img
from .ST import assemble_pita


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
    tmp2 = tmp.copy()  # copy of temporary dataframe for dropping blurred features int
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


def prep_data_single_sample_mxif(image, use_path, mean, filter_name, sigma, features, fract, path_save):
    """
    Perform log normalization, blurring and minmax scaling on the given image data

    Parameters
    ----------
    image : MILWRM.MxIF.img or str
        np.array containing MxIF data or path to the compressed npz file
    use_path : Boolean
        True if image is given as a path to the compressed npz file, False if image is given
        as MILWRM.MxIF.img object
    mean : numpy array
        Containing mean for each channel for that batch
    filter_name : str
        Name of the filter to use - gaussian, median or bilateral
    sigma : float, optional
        Standard deviation of Gaussian kernel for blurring
    features : list of int or str
        Indices or names of MxIF channels to use for tissue labeling
    fract : float, optional
        Fraction of cluster data from each image to randomly select for model
        building
    path_save : str
        Path to save final preprocessed files 
    Returns
    -------
    Subsampled_data : np.array
        np.array containing randomly sampled pixels for that image
    file_save : str
        path to save preprocessed img object
    """
    if use_path == True: # if images are given as path to the compressed npz file
        if path_save == None: # check if path to save final processed file is given
            raise Exception("Path to save final preprocessed npz files is requird when given path to image files")
        image_path = image
        image = img.from_npz(image_path + ".npz")
    # batch correction
    image.log_normalize(mean = mean)
    # apply the desired filter
    image.blurring(filter_name=filter_name, sigma = sigma)
    # min max scaling of each channel
    for i in range(image.img.shape[2]):
        img_ar = image.img[:, :, i][image.mask != 0]
        img_ar_max = img_ar.max()
        img_ar_min = img_ar.min()
        # print(img_ar_max, img_ar_min)
        image_ar_scaled = (image.img[:,:,i] - img_ar_min)/(img_ar_max - img_ar_min)
        image.img[:,:,i] = image_ar_scaled
    # subsample pixels to build the kmeans model
    subsampled_data = image.subsample_pixels(features,fract)
    if use_path == True:
        new_image_path = os.path.join(path_save,"final_preprocessed_images")
        if not os.path.exists(new_image_path):
            os.mkdir(new_image_path)
        file_name = image_path.split('/')[-1] + "final_preprocessed"
        file_save = os.path.join(new_image_path,file_name)
        image.to_npz(file_save)
        return subsampled_data, file_save
    return subsampled_data   
        

def add_tissue_ID_single_sample_mxif(image, use_path, features, kmeans, scaler):
    """
    Label pixels in a single MxIF sample with kmeans results

    Parameters
    ----------
    image : MILWRM.MxIF.img or str
        np.array containing MxIF data or path to the compressed npz file
    use_path : Boolean
        True if image is given as a path to the compressed npz file, False if image is given
        as MILWRM.MxIF.img object
    features : list of int or str
        Indices or names of MxIF channels to use for tissue labeling
    kmeans : sklearn.kmeans
        Trained k-means model

    Returns
    -------
    tID : np.array
        Image where pixel values are kmeans cluster IDs
    """
    if use_path == True:
        image_path = image + ".npz"
        image = img.from_npz(image_path)
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
    w, h, d = tuple(tmp.shape)
    image_array = tmp.reshape((w * h, d))
    scaled_image_array = scaler.transform(image_array)
    tID = kmeans.predict(scaled_image_array).reshape(w, h)
    tID = tID.astype(float)  # TODO: Figure out dtypes
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

    def plot_feature_proportions(self, labels=None, figsize=(10, 7), save_to=None):
        """
        Plots contributions of each training feature to k-means cluster centers as
        percentages of total

        Parameters
        ----------
        labels : list of str, optional (default=`None`)
            Labels corresponding to each MILWRM training feature. If `None`, features
            will be numbered 0 through p.
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
            if labels is not None:
                assert len(labels) == len(
                    self.features
                ), "'labels' must be the same length as self.features."
            else:
                labels = [self.rep + "_" + str(x) for x in self.features]
                if self.histo:
                    labels = labels + ["R", "G", "B"]
                if self.fluor_channels is not None:
                    labels = labels + ["ch_" + str(x) for x in self.fluor_channels]
        elif "mxif_labeler" in str(self.__class__):
            if labels is not None:
                assert len(labels) == len(
                    self.features
                ), "'labels' must be the same length as self.features."
            else:
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
        self,
        ncols=None,
        nfeatures=None,
        labels=None,
        titles=None,
        figsize=(5, 5),
        save_to=None,
    ):
        """
        Plots contributions of each training feature to k-means cluster centers

        Parameters
        ----------
        ncols : int, optional (default=`None`)
            Number of columns for gridspec. If `None`, uses number of tissue domains k.
        nfeatures : int, optional (default=`None`)
            Number of top-loaded features to show for each tissue domain
        labels : list of str, optional (default=`None`)
            Labels corresponding to each MILWRM training feature. If `None`, features
            will be numbered 0 through p.
        titles : list of str, optional (default=`None`)
            Titles of plots corresponding to each MILWRM domain. If `None`, titles
            will be numbers 0 through k.
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
            if labels is not None:
                assert len(labels) == len(
                    self.features
                ), "'labels' must be the same length as self.features."
            else:
                labels = [self.rep + "_" + str(x) for x in self.features]
                if self.histo:
                    labels = labels + ["R", "G", "B"]
                if self.fluor_channels is not None:
                    labels = labels + ["ch_" + str(x) for x in self.fluor_channels]
        elif "mxif_labeler" in str(self.__class__):
            if labels is not None:
                assert len(labels) == len(
                    self.features
                ), "'labels' must be the same length as self.features."
            else:
                labels = self.model_features
        if titles is None:
            titles = [
                "tissue_ID " + str(x)
                for x in range(self.kmeans.cluster_centers_.shape[0])
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
        self.raw = adatas.copy()

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
        self.fluor_channels = fluor_channels
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
        batch_labels = [[x]*len(cluster_data[x]) for x in range(len(cluster_data))] # batch labels for umap
        self.merged_batch_labels = list(itertools.chain(*batch_labels))
        # concatenate blurred features into cluster_data df for cluster training
        self.cluster_data = pd.concat(cluster_data)
        # perform min-max scaling on final cluster data
        # scaler = StandardScaler()
        scaler = MinMaxScaler()
        self.scaler = scaler.fit(cluster_data)
        scaled_data = scaler.transform(cluster_data)
        self.cluster_data = scaled_data 
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

    def show_feature_overlay(
        self,
        adata_index,
        pita,
        features=None,
        histo=None,
        cmap="plasma",
        label="feature",
        ncols=4,
        save_to=None,
        **kwargs,
    ):
        """
        Plot tissue_ID with individual pita features as alpha values to distinguish
        expression in identified tissue domains

        Parameters
        ----------
        adata_index : int
            Index of adata from `self.adatas` to plot overlays for (e.g. 0 for first
            adata object)
        pita : np.array
            Image of desired expression in pixel space from `.assemble_pita()`
        features : list of int, optional (default=`None`)
            List of features by index to show in plot. If `None`, use all features.
        histo : np.array or `None`, optional (default=`None`)
            Histology image to show along with pita in gridspec. If `None`, ignore.
        cmap : str, optional (default="plasma")
            Matplotlib colormap to use for plotting tissue IDs
        label : str
            What to title each panel of the gridspec (i.e. "PC" or "usage") or each
            channel in RGB image. Can also pass list of names e.g. ["NeuN","GFAP",
            "DAPI"] corresponding to channels.
        ncols : int
            Number of columns for gridspec
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
        # create tissue_ID pita for plotting
        tIDs = assemble_pita(
            self.adatas[adata_index],
            features="tissue_ID",
            use_rep="obs",
            plot_out=False,
            verbose=False,
        )
        # if pita has multiple features, plot them in gridspec
        if isinstance(features, int):  # force features into list if single integer
            features = [features]
        # if no features are given, use all of them
        elif features is None:
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
        # min-max scale each feature in pita to convert to interpretable alpha values
        mms = MinMaxScaler()
        if pita.ndim == 3:
            pita_tmp = mms.fit_transform(
                pita.reshape((pita.shape[0] * pita.shape[1], pita.shape[2]))
            )
        elif pita.ndim == 2:
            pita_tmp = mms.fit_transform(
                pita.reshape((pita.shape[0] * pita.shape[1], 1))
            )
        # reshape back to original
        pita = pita_tmp.reshape(pita.shape)
        # figure out labels for gridspec plots
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
            # determine where the histo image is in anndata
            assert (
                histo
                in self.adatas[adata_index]
                .uns["spatial"][
                    list(self.adatas[adata_index].uns["spatial"].keys())[0]
                ]["images"]
                .keys()
            ), "Must provide one of {} for histo".format(
                self.adatas[adata_index]
                .uns["spatial"][
                    list(self.adatas[adata_index].uns["spatial"].keys())[0]
                ]["images"]
                .keys()
            )
            histo = self.adatas[adata_index].uns["spatial"][
                list(self.adatas[adata_index].uns["spatial"].keys())[0]
            ]["images"][histo]
            if len(features) + 2 <= ncols:
                n_rows, n_cols = 1, len(features) + 2
            else:
                n_rows, n_cols = ceil((len(features) + 2) / ncols), ncols
            labels = ["Histology", "tissue_ID"] + labels  # append to front of labels
        else:
            if len(features) + 1 <= ncols:
                n_rows, n_cols = 1, len(features) + 1
            else:
                n_rows, n_cols = ceil(len(features) + 1 / ncols), ncols
            labels = ["tissue_ID"] + labels  # append to front of labels
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
        # plot tissue_ID first with colorbar
        ax = plt.subplot(gs[i])
        im = ax.imshow(tIDs, cmap=cmap, **kwargs)
        ax.tick_params(labelbottom=False, labelleft=False)
        sns.despine(bottom=True, left=True)
        ax.set_title(
            label=labels[i],
            loc="left",
            fontweight="bold",
            fontsize=16,
        )
        # colorbar scale for tissue_IDs
        _ = plt.colorbar(im, shrink=0.7)
        i = i + 1
        for feature in features:
            ax = plt.subplot(gs[i])
            im = ax.imshow(tIDs, alpha=pita[:, :, feature - 1], cmap=cmap, **kwargs)
            ax.tick_params(labelbottom=False, labelleft=False)
            sns.despine(bottom=True, left=True)
            ax.set_title(
                label=labels[i],
                loc="left",
                fontweight="bold",
                fontsize=16,
            )
            i = i + 1
        fig.tight_layout()
        if save_to:
            plt.savefig(fname=save_to, transparent=True, bbox_inches="tight", dpi=300)
        return fig


class mxif_labeler(tissue_labeler):
    """
    Tissue domain labeling class for multiplex immunofluorescence (MxIF) data
    """

    def __init__(self, image_df):
        """
        Initialize MxIF tissue labeler class

        Parameters
        ----------
        image_df : pd.DataFrame object
            Containing MILWRM.MxIF.img objects or str path to compressed npz files, batch names, 
            mean estimator and pixel count for each image in the following column order
            ['Img', 'batch_names', 'mean estimators', 'pixels']

        Returns
        -------

        Does not return anything. `self.images` attribute is updated,
        `self.cluster_data` attribute is initiated as `None`.
        """
        tissue_labeler.__init__(self)  # initialize parent class
        # validate the format of the image_df dataframe
        if np.all(image_df.columns == ['Img', 'batch_names', 'mean estimators', 'pixels']):
            self.image_df = image_df
        else:
            raise Exception("Image_df must be given with these columns in this format ['Img', 'batch_names', 'mean estimators', 'pixels']")
        if self.image_df['Img'].apply(isinstance, args = [img]).all():
            self.use_paths = False
        elif self.image_df['Img'].apply(isinstance, args = [str]).all():
            self.use_paths = True
        else:
            raise Exception("Img column in the dataframe should be either str for paths to the files or mxif.img object")

    def prep_cluster_data(self, features, filter_name = 'gaussian', sigma = 2, fract = 0.2, path_save = None):
        """
        Prepare master array for tissue level clustering

        Parameters
        ----------
        features : list of int or str
            Indices or names of MxIF channels to use for tissue labeling
        filter_name : str
            Name of the filter to use - gaussian, median or bilateral
        sigma : float, optional (default=2)
            Standard deviation of Gaussian kernel for blurring
        fract : float, optional (default=0.2)
            Fraction of cluster data from each image to randomly select for model
            building
        path_save : str (default = None)
            Path to save final preprocessed files, if self.use_path is True
            default path_save will raise Exception

        Returns
        -------
        Does not return anything. `self.images` are normalized, blurred and scaled according
        to user parameters. `self.cluster_data` becomes master `np.array` for cluster
        training. Parameters are also captured as attributes for posterity.

        """
        if self.cluster_data is not None:
            print("WARNING: overwriting existing cluster data")
            self.cluster_data = None
        # save the hyperparams as object attributes
        self.model_features = features
        use_path = self.use_paths
        # calculate the batch wise means
        mean_for_each_batch = {}
        for batch in self.image_df['batch_names'].unique():
            list_mean_estimators = list(self.image_df[self.image_df['batch_names'] == batch]['mean estimators'])
            mean_estimator_batch = sum(map(np.array, list_mean_estimators))
            pixels = sum(self.image_df[self.image_df['batch_names'] == batch]['pixels'])
            mean_for_each_batch[batch] = mean_estimator_batch/pixels
        # log_normalize, apply blurring filter, minmax scale each channel and subsample data
        subsampled_data = []
        path_to_blurred_npz = []
        for image,batch in zip(self.image_df['Img'],self.image_df['batch_names']):
            tmp = prep_data_single_sample_mxif(
                image,use_path=use_path, mean = mean_for_each_batch[batch], filter_name= filter_name,
                sigma = sigma, features = self.model_features, fract = fract, path_save= path_save
            )
            if self.use_paths == True:
                subsampled_data.append(tmp[0])
                path_to_blurred_npz.append(tmp[1])
            else:
                subsampled_data.append(tmp)
        batch_labels = [[x]*len(subsampled_data[x]) for x in range(len(subsampled_data))] # batch labels for umap
        self.merged_batch_labels = list(itertools.chain(*batch_labels))
        if self.use_paths == True:
            self.image_df['Img'] = path_to_blurred_npz
        cluster_data = np.row_stack(subsampled_data)
        # perform z-score normalization on cluster_Data
        scaler = StandardScaler()
        self.scaler = scaler.fit(cluster_data)
        scaled_data = scaler.transform(cluster_data)
        self.cluster_data = scaled_data 

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
        # save the hyperparams as object attributes
        use_path = self.use_paths
        # find optimal k with parent class
        if k is None:
            print("Determining optimal cluster number k via scaled inertia")
            self.find_optimal_k(
                alpha=alpha, plot_out=plot_out, random_state=random_state, n_jobs=n_jobs
            )
        # call k-means model from parent class
        self.find_tissue_regions(k=k, random_state=random_state)
        # loop through image objects and create tissue label images
        print("Creating tissue_ID images for image objects...")
        #TODO: Make this compatible with npz paths
        self.tissue_IDs = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(add_tissue_ID_single_sample_mxif)(
                image, use_path, self.model_features, self.kmeans, self.scaler
            )
            for image in self.image_df['Img']
        )

    def show_marker_overlay(
        self,
        image_index,
        channels=None,
        cmap="Set1",
        mask_out=True,
        ncols=4,
        save_to=None,
        **kwargs,
    ):
        """
        Plot tissue_ID with individual markers as alpha values to distinguish
        expression in identified tissue domains

        Parameters
        ----------
        image_index : int
            Index of image from `self.images` to plot overlays for (e.g. 0 for first
            image)
        channels : tuple of int or None, optional (default=`None`)
            List of channels by index or name to show
        cmap : str, optional (default="plasma")
            Matplotlib colormap to use for plotting tissue IDs
        mask_out : bool, optional (default=`True`)
            Mask out non-tissue pixels prior to showing
        ncols : int
            Number of columns for gridspec if plotting individual channels.
        save_to : str or None
            Path to image file to save results. If `None`, show figure.
        **kwargs
            Arguments to pass to `plt.imshow()` function.

        Returns
        -------
        Matplotlib object (if plotting one feature or RGB) or gridspec object (for
        multiple features). Saves plot to file if `save_to` is not `None`.
        """
        # if image has multiple channels, plot them in gridspec
        if isinstance(channels, int):  # force channels into list if single integer
            channels = [channels]
        if isinstance(channels, str):  # force channels into int if single string
            channels = [self[image_index].ch.index(channels)]
        if checktype(channels):  # force channels into list of int if list of strings
            channels = [self[image_index].ch.index(x) for x in channels]
        if channels is None:  # if no channels are given, use all of them
            channels = [x for x in range(self[image_index].n_ch)]
        assert (
            len(channels) <= self[image_index].n_ch
        ), "Too many channels given: image has {}, expected {}".format(
            self[image_index].n_ch, len(channels)
        )
        # creating a copy of the image
        image_cp = self[image_index].copy()
        # re-scaling to set pixel value range between 0 to 1
        image_cp.scale()
        # defining cmap for discrete color bar
        cmap = plt.cm.get_cmap(cmap, self.k)
        # calculate gridspec dimensions
        if len(channels) + 1 <= ncols:
            n_rows, n_cols = 1, len(channels) + 1
        else:
            n_rows, n_cols = ceil(len(channels) + 1 / ncols), ncols
        fig = plt.figure(figsize=(ncols * n_cols, ncols * n_rows))
        # arrange axes as subplots
        gs = gridspec.GridSpec(n_rows, n_cols, figure=fig)
        # plot tissue_ID first with colorbar
        ax = plt.subplot(gs[0])
        im = ax.imshow(self.tissue_IDs[image_index], cmap=cmap, **kwargs)
        ax.set_title(
            label="tissue_ID",
            loc="left",
            fontweight="bold",
            fontsize=16,
        )
        ax.tick_params(labelbottom=False, labelleft=False)
        sns.despine(bottom=True, left=True)
        # colorbar scale for tissue_IDs
        _ = plt.colorbar(im, ticks = range(self.k), shrink=0.7)
        # add plots to axes
        i = 1
        for channel in channels:
            ax = plt.subplot(gs[i])
            # make copy for alpha
            im_tmp = image_cp.img[:, :, channel].copy()
            if self[image_index].mask is not None and mask_out:
                # area outside mask NaN
                self.tissue_IDs[image_index][self[image_index].mask == 0] = np.nan
                im = ax.imshow(
                    self.tissue_IDs[image_index], cmap=cmap, alpha=im_tmp, **kwargs
                )
            else:
                ax.imshow(self.tissue_IDs[image_index], alpha=im_tmp, **kwargs)
            ax.tick_params(labelbottom=False, labelleft=False)
            sns.despine(bottom=True, left=True)
            ax.set_title(
                label=self[image_index].ch[channel],
                loc="left",
                fontweight="bold",
                fontsize=16,
            )
            i = i + 1
        fig.tight_layout()
        if save_to:
            plt.savefig(fname=save_to, transparent=True, bbox_inches="tight", dpi=300)
        return fig