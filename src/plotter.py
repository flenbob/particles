from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker
import matplotlib.ticker as ticker

from .content_uniformity import COVPredictor, Stange

@dataclass
class Plotter:
    def plot_Z_distribution(self, particle_types: np.ndarray, particle_contacts: np.ndarray) -> None:
        """Plots coordination number distributions for each particle type. Input data from last frame (-1).

        Args:
            particle_types (np.ndarray): 1D array of particle type for each particle.
            particle_contacts (np.ndarray): 1D array number of contacts for each particle.
        """
        #Read contact data

        #Componentwise contact lists
        contact_list = [particle_contacts[particle_types == i] for i in range(1, int(max(particle_types))+1)]
        
        #Densities
        for i, contacts in enumerate(contact_list):
            Z_vals = np.arange(min(contacts), max(contacts))
            density = [contacts[contacts == Z_val].shape[0]/contacts.shape[0] for Z_val in Z_vals]
            plt.plot(Z_vals, density, label=i+1)
        
        #Plot with logscale on x-axis
        plt.ylabel('Density')
        plt.xlabel('Coordination number')
        plt.xscale('log')
        plt.legend()
        plt.show()

    def plot_Z_matrix(self, particle_types: np.ndarray, contact_pairs: np.ndarray) -> None:
        """Plots coordination number matrix between particle types. Recommended to input data from last frame (-1).

        Args:
            particle_types (np.ndarray): 1D array of particle type for each particle.
            contact_pairs (np.ndarray): 2D array of particle ids between two contacting particles.
        """
        #Helper functions
        def fmt(x, pos):
            a, b = '{:.1e}'.format(x).split('e')
            b = int(b)
            return '{:.1f}'.format(10**float(x))

        def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel="", **kwargs):
            """
            Create a heatmap from a numpy array and two lists of labels.

            Parameters
            ----------
            data
                A 2D numpy array of shape (M, N).
            row_labels
                A list or array of length M with the labels for the rows.
            col_labels
                A list or array of length N with the labels for the columns.
            ax
                A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
                not provided, use current axes or create a new one.  Optional.
            cbar_kw
                A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
            cbarlabel
                The label for the colorbar.  Optional.
            **kwargs
                All other arguments are forwarded to `imshow`.
            """

            if ax is None:
                ax = plt.gca()

            if cbar_kw is None:
                cbar_kw = {}

            # Plot the heatmap
            im = ax.imshow(data, **kwargs)

            # Create colorbar
            cbar = ax.figure.colorbar(ax.imshow(np.log10(data), **kwargs), ax=ax, format=ticker.FuncFormatter(fmt), **cbar_kw)
            cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

            # Show all ticks and label them with the respective list entries.
            ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
            ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

            # Let the horizontal axes labeling appear on top.
            ax.tick_params(top=True, bottom=False,
                        labeltop=True, labelbottom=False)

            # Rotate the tick labels and set their alignment.
            plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
                    rotation_mode="anchor")

            # Turn spines off and create white grid.
            ax.spines[:].set_visible(False)

            ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
            ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
            ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
            ax.tick_params(which="minor", bottom=False, left=False)
            return im, cbar

        def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                            textcolors=("black", "white"),
                            threshold=None, **textkw):
            """
            A function to annotate a heatmap.

            Parameters
            ----------
            im
                The AxesImage to be labeled.
            data
                Data used to annotate.  If None, the image's data is used.  Optional.
            valfmt
                The format of the annotations inside the heatmap.  This should either
                use the string format method, e.g. "$ {x:.2f}", or be a
                `matplotlib.ticker.Formatter`.  Optional.
            textcolors
                A pair of colors.  The first is used for values below a threshold,
                the second for those above.  Optional.
            threshold
                Value in data units according to which the colors from textcolors are
                applied.  If None (the default) uses the middle of the colormap as
                separation.  Optional.
            **kwargs
                All other arguments are forwarded to each call to `text` used to create
                the text labels.
            """

            if not isinstance(data, (list, np.ndarray)):
                data = im.get_array()

            # Normalize the threshold to the images color range.
            if threshold is not None:
                threshold = im.norm(threshold)
            else:
                threshold = im.norm(data.max())/2.

            # Set default alignment to center, but allow it to be
            # overwritten by textkw.
            kw = dict(horizontalalignment="center",
                    verticalalignment="center")
            kw.update(textkw)

            # Get the formatter in case a string is supplied
            if isinstance(valfmt, str):
                valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

            # Loop over the data and create a `Text` for each "pixel".
            # Change the text's color depending on the data.
            texts = []
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
                    text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
                    texts.append(text)
            return texts

        #Componentwise counts
        _, types_cnt = np.unique(particle_types, return_counts=True)
        
        #Componentwise contact occurences
        N_types = int(max(particle_types))
        contact_matrix = np.zeros((N_types, N_types))
        for contact_pair in contact_pairs[0]:
            I, J = int(particle_types[contact_pair[0]-1]), int(particle_types[contact_pair[1]-1])
            contact_matrix[I-1, J-1] += 1
            contact_matrix[J-1, I-1] += 1

        #Component-wise coordination number ()
        component_Z_matrix = contact_matrix/types_cnt

        #Add color scale to matrix
        fig, ax = plt.subplots()
        im, cbar = heatmap(component_Z_matrix, np.arange(N_types)+1, np.arange(N_types)+1, ax=ax,
                        cmap="rainbow", cbarlabel="Coordination number")
        texts = annotate_heatmap(im, valfmt="{x:.1f}")

        fig.tight_layout()
        plt.show()

        #Pairwise coordination number()
        cnt_matrix = types_cnt[np.newaxis, :] + types_cnt[:, np.newaxis]
        pair_Z_matrix = contact_matrix/cnt_matrix
        pair_Z_matrix[np.diag_indices_from(pair_Z_matrix)]*=2
        
        fig, ax = plt.subplots()
        im, cbar = heatmap(pair_Z_matrix, np.arange(N_types)+1, np.arange(N_types)+1, ax=ax,
                        cmap="rainbow", cbarlabel="Coordination number")
        texts = annotate_heatmap(im, valfmt="{x:.1f}")

        fig.tight_layout()
        plt.show()

    def plot_content_uniformity(self, cv_pred: COVPredictor, stange: Stange, xlab  : str):
        # Cov predictor is already fitted to data
        # Stange has been set to either mass or volume

        # Get fitting and prediction range
        X0, Xp = cv_pred.x_pts, cv_pred.x_pts_contd

        # True data
        Y0, sY0 = cv_pred.cov_mean, cv_pred.cov_std

        # Prediction of fitting range and continuation
        Y0p, sY0p, Yp, sYp = cv_pred.cov_mean_pred, cv_pred.cov_std_pred, cv_pred.cov_mean_pred_contd, cv_pred.cov_std_pred_contd

        # Get Stange
        Y0_stange = stange.calc_CU(X0)
        Yp_stange = stange.calc_CU(Xp)

        Ncomps = np.shape(Y0)[0]
        for i in range(Ncomps):
            # Plot in a single window
            axs1 = plt.subplot(2, 2, 1)
            axs2 = plt.subplot(2, 2, 2)
            axs3 = plt.subplot(2, 1, 2)

            # Axs 1: 
            axs1.set_title("COV of CSSM data and curvefit")

            # Plot original data
            # Cut such that Stange < 0.1
            ind = Y0_stange[i, :]<0.1
            axs1.plot(X0[ind], Y0[i][ind], color = "k", label = "Raw data")
            axs1.plot(X0[ind], Y0[i][ind] + 1.96*sY0[i][ind], color = 'k', linestyle = 'dashed')
            axs1.plot(X0[ind], Y0[i][ind] - 1.96*sY0[i][ind], color = 'k', linestyle = 'dashed')

            # Plot fitting and Stange on fitting range
            axs1.plot(X0[ind], Y0p[i][ind], color = "b", label  = "Fit")
            axs1.plot(X0[ind], Y0p[i][ind] + 1.96*sY0p[i][ind], color = 'b', linestyle = 'dashed')
            axs1.plot(X0[ind], Y0p[i][ind] - 1.96*sY0p[i][ind], color = 'b', linestyle = 'dashed')
            axs1.plot(X0[ind], Y0_stange[i, ind], color = "r", label  = "Stange")

            # Axs 2: 
            axs2.set_title("STD of COV of CSSM data and curvefit")

            # Plot the cov std data and its fit
            axs2.plot(X0, sY0[i], color = "k", label = "Std data")
            axs2.plot(X0, sY0p[i], color = "b", label = "Std fit")

            # Axs 3: 
            axs3.set_title("COV of CSSM data and curvefit over entire range")

            # Plot fit prediction and Stange on continuation range
            axs3.plot(Xp, Yp[i], color = "b", label = "Prediction")
            axs3.plot(Xp, Yp[i] + 1.96*sYp[i], color = 'b', linestyle = 'dashed')
            axs3.plot(Xp, Yp[i] - 1.96*sYp[i], color = 'b', linestyle = 'dashed')
            axs3.plot(Xp, Yp_stange[i, :], color = "r", label = "Stange")

            # Misc.
            axs1.set_ylabel("Cv")
            axs1.set_ylabel("Std")
            axs3.set_ylabel("Cv")

            axs1.set_xlabel(xlab)
            axs1.set_xlabel(xlab)
            axs3.set_xlabel(xlab)

            axs1.legend()
            axs2.legend()
            axs3.legend()

            axs1.set_yscale('log')
            axs2.set_yscale('log')
            axs3.set_yscale('log')

            axs1.set_ylabel('COV')
            axs2.set_ylabel('COV')
            axs3.set_ylabel('COV')
            plt.suptitle(f"Component {i+1}, CV/STD wrt. mass[{xlab}]")
            plt.show()

    def plot_particle_size_distribution(self, particle_types: np.ndarray, particle_diameters: np.ndarray):
        pass

    def print_summary_statistics(self):
        pass
        
