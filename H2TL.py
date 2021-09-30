# -*- coding: utf-8 -*-
"""
Classes for assigning tissue region IDs to multiplex immunofluorescence (MxIF) or 10X 
Visium spatial transcriptomic (ST) and histological imaging data

@author: C Heiser
"""
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from skimage.filters import gaussian
from img_utils import checktype
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


class tissue_labeler:
    """
    Master tissue region labeling class
    """

    def __init__(self):
        """
        Initialize tissue labeler parent class
        """
        self.cluster_data = None  # start out with no data to cluster on
        self.k = None # start out with no k value
    
    def find_optimal_k(self,plot_out=False,random_state=18):
        """
        Uses silhouette analysis to decide on k clusters for clustering in the 
        corresponding `anndata` objects.

        Parameters
        ----------
        plot_out : boolean, optional (default=FALSE)
            Determines if silhouette plots should be output
        random_state : int, optional (default=18)
            Seed for k-means clustering models.

        Returns
        -------
        Does not return anything. `self.k` contains integer value for number of clusters.
        Parameters are also captured as attributes for posterity.
        """
        if self.cluster_data is None:
            print("No cluster data found. Run prep_cluster_data() first.")
            pass
        self.random_state = random_state
        all_silhouette_scores = np.array([])
        range_clusters = np.array(range(3,10))
        
        print("Running silhouette analysis for optimal k on cluster_data.")
        
        ## loop over possibilities
        for n_clusters in range_clusters:
            ## setup KMeans algorithm
            clusterer = KMeans(n_clusters=n_clusters,random_state=random_state)
    
            ## fit to cluster data
            cluster_labels = clusterer.fit_predict(self.cluster_data)
    
            ## calculate avg silhouette score
            silhouette_avg = silhouette_score(self.cluster_data, cluster_labels)
    
            ## collect all scores
            all_silhouette_scores = np.append(all_silhouette_scores,silhouette_avg)
    
    
            sample_values = silhouette_samples(self.cluster_data, cluster_labels)
            ## if plot_out, setup plt for each set of silhouette scores
            ## adapted from: https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html#sphx-glr-auto-examples-cluster-plot-kmeans-silhouette-analysis-py
            if(plot_out):
                # Create a subplot with 1 row and 2 columns
                fig, (ax1, ax2) = plt.subplots(1, 2)
                fig.set_size_inches(18, 7)
        
                # The 1st subplot is the silhouette plot
                ax1.set_xlim([-.25, 1])
                # The (n_clusters+1)*10 is for inserting blank space between silhouette
                # plots of individual clusters, to demarcate them clearly.
                ax1.set_ylim([0, len(self.cluster_data) + (n_clusters + 1) * 10])
    
                y_lower = 10
                for i in range(n_clusters):
                    # Aggregate the silhouette scores for samples belonging to
                    # cluster i, and sort them
                    ith_cluster_silhouette_values = \
                        sample_values[cluster_labels == i]

                    ith_cluster_silhouette_values.sort()

                    size_cluster_i = ith_cluster_silhouette_values.shape[0]
                    y_upper = y_lower + size_cluster_i

                    color = cm.nipy_spectral(float(i) / n_clusters)
                    ax1.fill_betweenx(np.arange(y_lower, y_upper),
                                      0, ith_cluster_silhouette_values,
                                      facecolor=color, edgecolor=color, alpha=0.7)

                    # Label the silhouette plots with their cluster numbers at the middle
                    ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

                    # Compute the new y_lower for next plot
                    y_lower = y_upper + 10  # 10 for the 0 samples

                    ax1.set_title("The silhouette plot for the various clusters.")
                    ax1.set_xlabel("The silhouette coefficient values")
                    ax1.set_ylabel("Cluster label")

                    # The vertical line for average silhouette score of all the values
                    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

                    ax1.set_yticks([])  # Clear the yaxis labels / ticks
                    ax1.set_xticks([-.25, 0.0, 0.25,0.5,0.75, 1])

                    # 2nd Plot showing the actual clusters formed
                    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
                    ax2.scatter(self.cluster_data[:, 0], self.cluster_data[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                                c=colors, edgecolor='k')

                    # Labeling the clusters
                    centers = clusterer.cluster_centers_
                    # Draw white circles at cluster centers
                    ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                                c="white", alpha=1, s=200, edgecolor='k')

                    for i, c in enumerate(centers):
                        ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                                    s=50, edgecolor='k')

                    ax2.set_title("The visualization of the clustered data.")
                    ax2.set_xlabel("Feature space for the 1st feature")
                    ax2.set_ylabel("Feature space for the 2nd feature")

                    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                                  "with n_clusters = %d" % n_clusters),
                                 fontsize=14, fontweight='bold')


        ## determine optimal cluster by silhouette score
        print("The optimal number of clusters is {}, with a mean silhouette score of {}.".format(range_clusters[all_silhouette_scores.argmax()],all_silhouette_scores.max()))
        if(plot_out): plt.show()
        self.k = range_clusters[all_silhouette_scores.argmax()]

    def find_tissue_regions(self, k=None, random_state=18):
        """
        Perform tissue-level clustering and label pixels in the corresponding
        `anndata` objects.

        Parameters
        ----------
        k : int, optional (default=None)
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
        if k is None and self.k is None:
            print("No k found or provided. Run find_optimal_k() first or pass a k value.")
            pass
        if k is not None:
            print("Overriding optimal k value with k={}.".format(k))
            self.k = k
        # save the hyperparams as object attributes
        self.random_state = random_state
        print("Performing k-means clustering with {} target clusters".format(self.k))
        self.kmeans = KMeans(n_clusters=self.k, random_state=random_state).fit(
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

    def label_tissue_regions(self, k=None, plot_out = False, random_state=18):
        """
        Perform tissue-level clustering and label pixels in the corresponding
        `anndata` objects.

        Parameters
        ----------
        k : int, optional (default=None)
            Number of tissue regions to define
        plot_out : boolean, optional (default=FALSE)
            Determines if silhouette plots should be output
        random_state : int, optional (default=18)
            Seed for k-means clustering model.

        Returns
        -------
        Does not return anything. `self.adatas` are updated, adding "tissue_ID" field
        to `.obs`. `self.kmeans` contains trained `sklearn` clustering model.
        Parameters are also captured as attributes for posterity.
        """
        # find optimal k with parent class
        if k is None:
            self.find_optimal_k(plot_out=plot_out,random_state=random_state)
        # call k-means model from parent class
        self.find_tissue_regions(k=k,random_state=random_state)
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

    def prep_cluster_data(self, features, downsample_factor=8, sigma=2, fract=0.2):
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
        # perform image downsampling, blurring, subsampling, and compile cluster_data
        for image_i, image in enumerate(self.images):
            print(
                "Downsampling, log-normalizing, and blurring image #{}".format(image_i)
            )
            # downsample image
            image.downsample(fact=downsample_factor, func=np.mean)
            # normalize and log-transform image
            image.log_normalize(pseudoval=1, mask=True)
            # blur downsampled image
            image.img = gaussian(image.img, sigma=sigma, multichannel=True)
            print("Collecting {} features for image #{}".format(len(features), image_i))
            # get list of int for features
            self.features = features
            if isinstance(
                self.features, int
            ):  # force features into list if single integer
                self.features = [self.features]
            if isinstance(
                self.features, str
            ):  # force features into int if single string
                self.features = [image.ch.index(self.features)]
            if checktype(
                self.features
            ):  # force features into list of int if list of strings
                self.features = [image.ch.index(x) for x in self.features]
            if self.features is None:  # if no features are given, use all of them
                self.features = [x for x in range(image.n_ch)]
            # get cluster data for image_i
            tmp = []
            for i in range(image.img.shape[2]):
                tmp.append(image.img[:, :, i][image.mask != 0])
            tmp = np.column_stack(tmp)
            # select cluster data
            i = np.random.choice(tmp.shape[0], int(tmp.shape[0] * fract))
            tmp = tmp[np.ix_(i, self.features)]
            # append blurred features to cluster_data df for cluster training
            if self.cluster_data is None:
                self.cluster_data = tmp.copy()
            else:
                self.cluster_data = np.row_stack([self.cluster_data, tmp])
        print("Collected clustering data of shape: {}".format(self.cluster_data.shape))

    def label_tissue_regions(self, k=None, plot_out=False, random_state=18):
        """
        Perform tissue-level clustering and label pixels in the corresponding
        images.

        Parameters
        ----------
        ----------
        k : int, optional (default=None)
            Number of tissue regions to define
        plot_out : boolean, optional (default=FALSE)
            Determines if silhouette plots should be output
        random_state : int, optional (default=18)
            Seed for k-means clustering model.

        Returns
        -------
        Does not return anything. `self.tissue_ID` is added, containing image with
        final tissue region IDs. `self.kmeans` contains trained `sklearn` clustering
        model. Parameters are also captured as attributes for posterity.
        """
        # find optimal k with parent class
        if k is None:
            self.find_optimal_k(plot_out=plot_out,random_state=random_state)
        # call k-means model from parent class
        self.find_tissue_regions(random_state=random_state)
        # loop through image objects and create tissue label images
        print("Creating tissue_ID images for image objects:")
        self.tissue_IDs = []
        for i in range(len(self.images)):
            print("\tImage #{}".format(i))
            # get list of int for features
            self.features = self.model_features
            if isinstance(
                self.features, int
            ):  # force features into list if single integer
                self.features = [self.features]
            if isinstance(
                self.features, str
            ):  # force features into int if single string
                self.features = [self.images[i].ch.index(self.features)]
            if checktype(
                self.features
            ):  # force features into list of int if list of strings
                self.features = [self.images[i].ch.index(x) for x in self.features]
            if self.features is None:  # if no features are given, use all of them
                self.features = [x for x in range(self.images[i].n_ch)]
            # subset to features used in prep_cluster_data
            tmp = self.images[i].img[:, :, self.features]
            tID = np.zeros(tmp.shape[:2])
            for x in range(tmp.shape[0]):
                for y in range(tmp.shape[1]):
                    tID[x, y] = self.kmeans.predict(tmp[x, y, :].reshape(1, -1))
            tID[self.images[i].mask == 0] = np.nan  # set masked-out pixels to NaN
            self.tissue_IDs.append(tID)  # append tID to list of cluster images
