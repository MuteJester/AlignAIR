from collections import defaultdict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
from multiprocessing import Pool, cpu_count

sns.set_context('poster')


def process_gene_likelihood(args):
    """Helper function to process likelihoods for a single gene using multiprocessing."""
    gene, iterator, ground_truth_column, index_to_allele, likelihood_round_to = args
    likelihood_hit_function = defaultdict(int)
    likelihood_hit_total = defaultdict(int)

    for likelihoods_row, groundtruth_call in zip(iterator, ground_truth_column):
        groundtruth_decomposed = groundtruth_call.split(',')
        for index, likelihood in enumerate(likelihoods_row):
            likelihood_hit_function[likelihood] += index_to_allele[gene][index] in groundtruth_decomposed
            likelihood_hit_total[likelihood] += 1

    return gene, likelihood_hit_function, likelihood_hit_total


class LikelihoodCalibrationPlot:
    def __init__(self, likelihood_round_to=2):

        self.fig = None
        self.axd = None
        self.likelihood_bins = np.linspace(0, 1, 25)
        self.mounted_models = {}
        self.likelihood_round_to = likelihood_round_to
        self.index_to_allele = {}

    def mount_index_to_allele_map(self, allele_map):
        self.index_to_allele = allele_map

    def index_to_allele_map_from_dataconfig(self, dataconfig_heavy_or_kappa, dataconfig_lambda=None):
        if dataconfig_lambda is not None:
            v_kappa_dict = {j.name: j.ungapped_seq.upper() for i in dataconfig_heavy_or_kappa.v_alleles for j in
                            dataconfig_heavy_or_kappa.v_alleles[i]}
            j_kappa_dict = {j.name: j.ungapped_seq.upper() for i in dataconfig_heavy_or_kappa.j_alleles for j in
                            dataconfig_heavy_or_kappa.j_alleles[i]}

            v_lambda_dict = {j.name: j.ungapped_seq.upper() for i in dataconfig_lambda.v_alleles for j in
                             dataconfig_lambda.v_alleles[i]}
            j_lambda_dict = {j.name: j.ungapped_seq.upper() for i in dataconfig_lambda.j_alleles for j in
                             dataconfig_lambda.j_alleles[i]}

            v_alleles = sorted(list(v_kappa_dict)) + sorted(list(v_lambda_dict))
            j_alleles = sorted(list(j_kappa_dict)) + sorted(list(j_lambda_dict))

            v_allele_count = len(v_alleles)
            j_allele_count = len(j_alleles)

            v_allele_call_ohe = {i: f for i, f in enumerate(v_alleles)}
            j_allele_call_ohe = {i: f for i, f in enumerate(j_alleles)}

            self.index_to_allele = {
                "v": v_allele_call_ohe,
                "j": v_allele_call_ohe,
            }
        else:
            v_heavy_dict = {j.name: j.ungapped_seq.upper() for i in dataconfig_heavy_or_kappa.v_alleles for j in
                            dataconfig_heavy_or_kappa.v_alleles[i]}
            d_heavy_dict = {j.name: j.ungapped_seq.upper() for i in dataconfig_heavy_or_kappa.d_alleles for j in
                            dataconfig_heavy_or_kappa.d_alleles[i]}
            j_heavy_dict = {j.name: j.ungapped_seq.upper() for i in dataconfig_heavy_or_kappa.j_alleles for j in
                            dataconfig_heavy_or_kappa.j_alleles[i]}

            v_alleles = sorted(list(v_heavy_dict))
            d_alleles = sorted(list(d_heavy_dict))
            d_alleles = d_alleles + ['Short-D']
            j_alleles = sorted(list(j_heavy_dict))

            v_allele_count = len(v_alleles)
            d_allele_count = len(d_alleles)
            j_allele_count = len(j_alleles)

            v_allele_call_ohe = {i: f for i, f in enumerate(v_alleles)}
            d_allele_call_ohe = {i: f for i, f in enumerate(d_alleles)}
            j_allele_call_ohe = {i: f for i, f in enumerate(j_alleles)}

            self.index_to_allele = {
                "v": v_allele_call_ohe,
                "d": d_allele_call_ohe,
                "j": j_allele_call_ohe,
            }

    def categorize_value(self, value):
        bin_category = pd.cut([value], bins=self.likelihood_bins, include_lowest=False)
        return bin_category[0]

    def mount_likelihood_functions(self, predictions: dict, ground_truth: pd.DataFrame, model_name: str):
        likelihood_hit_function = dict()
        likelihood_hit_total = dict()

        for gene in tqdm(['v', 'd', 'j']):
            if f'{gene}_allele' in list(predictions):
                iterator = predictions[f'{gene}_allele'].round(self.likelihood_round_to)
                likelihood_hit_function[gene] = defaultdict(int)
                likelihood_hit_total[gene] = defaultdict(int)
                for likelihoods_row, groundtruth_call in tqdm(zip(iterator, ground_truth[f'{gene}_call']),
                                                              total=len(iterator)):
                    groundtruth_decomposed = groundtruth_call.split(',')
                    for index, likelihood in enumerate(likelihoods_row):
                        likelihood_hit_function[gene][likelihood] += self.index_to_allele[gene][
                                                                         index] in groundtruth_decomposed
                        likelihood_hit_total[gene][likelihood] += 1

        for gene in likelihood_hit_function:
            likelihood_hit_function[gene] = pd.Series(likelihood_hit_function[gene],
                                                      name='hits').sort_values().to_frame().reset_index().rename(
                columns={'index': 'likelihood'})
            likelihood_hit_total[gene] = pd.Series(likelihood_hit_total[gene],
                                                   name='hits').sort_values().to_frame().reset_index().rename(
                columns={'index': 'likelihood'})
            likelihood_hit_function[gene] = likelihood_hit_function[gene].set_index('likelihood')
            likelihood_hit_function[gene] = pd.concat([likelihood_hit_function[gene],
                                                       likelihood_hit_total[gene].set_index('likelihood').rename(
                                                           columns={'hits': 'counts'})], axis=1)
            likelihood_hit_function[gene] = likelihood_hit_function[gene].reset_index()
            # likelihood_hit_function[gene]['hits']/=len(ground_truth)
            likelihood_hit_function[gene]['bins'] = pd.cut(likelihood_hit_function[gene]['likelihood'], 25,
                                                           include_lowest=False)
            likelihood_hit_function[gene] = likelihood_hit_function[gene].groupby('bins').agg(
                {'hits': 'sum', 'likelihood': 'mean', 'counts': 'sum'})
            likelihood_hit_function[gene]['hits'] = likelihood_hit_function[gene].apply(
                lambda x: np.round(x['hits'] / x['counts'], 5), axis=1)

        self.mounted_models[model_name] = likelihood_hit_function

    def mount_likelihood_functions_mp(self, predictions: dict, ground_truth: pd.DataFrame, model_name: str):
        likelihood_hit_function = dict()
        likelihood_hit_total = dict()

        # Prepare arguments for multiprocessing
        args = [
            (
                gene,
                predictions[f'{gene}_allele'].round(self.likelihood_round_to),
                ground_truth[f'{gene}_call'],
                self.index_to_allele[gene],
                self.likelihood_round_to,
            )
            for gene in ['v', 'd', 'j'] if f'{gene}_allele' in predictions
        ]

        # Use multiprocessing to process each gene in parallel
        with Pool(processes=min(cpu_count(), len(args))) as pool:
            results = pool.map(process_gene_likelihood, args)

        # Combine results
        for gene, hits, totals in results:
            likelihood_hit_function[gene] = pd.Series(hits, name='hits').sort_values().to_frame().reset_index().rename(
                columns={'index': 'likelihood'})
            likelihood_hit_total[gene] = pd.Series(totals, name='hits').sort_values().to_frame().reset_index().rename(
                columns={'index': 'likelihood'})
            likelihood_hit_function[gene] = likelihood_hit_function[gene].set_index('likelihood')
            likelihood_hit_function[gene] = pd.concat([likelihood_hit_function[gene],
                                                       likelihood_hit_total[gene].set_index('likelihood').rename(
                                                           columns={'hits': 'counts'})], axis=1)
            likelihood_hit_function[gene] = likelihood_hit_function[gene].reset_index()
            likelihood_hit_function[gene]['bins'] = pd.cut(likelihood_hit_function[gene]['likelihood'], 25,
                                                           include_lowest=False)
            likelihood_hit_function[gene] = likelihood_hit_function[gene].groupby('bins').agg(
                {'hits': 'sum', 'likelihood': 'mean', 'counts': 'sum'})
            likelihood_hit_function[gene]['hits'] = likelihood_hit_function[gene].apply(
                lambda x: np.round(x['hits'] / x['counts'], 5), axis=1)

        self.mounted_models[model_name] = likelihood_hit_function


    def init_figure(self, figsize=(20, 11)):
        mosaic = """
                    A
                """
        self.fig = plt.figure(layout="tight", figsize=figsize)
        self.axd = self.fig.subplot_mosaic(mosaic)

    @staticmethod
    def plot_point_and_lines(ax, x, y, point_label=None):
        """
        Plots a point and extends lines from the point to the x and y axes.

        Args:
        ax (AxesSubplot): The axes on which to plot.
        x (float): The x-coordinate of the point.
        y (float): The y-coordinate of the point.
        point_label (str): Label for the point.
        """
        # Plot the point
        ax.plot(x, y, 'ro')  # 'ro' for red circle

        if point_label:
            ax.annotate(point_label, (x, y), textcoords="offset points", xytext=(0, 10), ha='center')

        # Draw lines extending to the axes
        ax.axhline(y=y, color='gray', linestyle='--', xmax=x / ax.get_xlim()[1])
        ax.axvline(x=x, color='gray', linestyle='--', ymax=y / ax.get_ylim()[1])

    def plot(self):
        self.init_figure()

        for name in self.mounted_models:
            # AlignAIR LIKELIHOOD FUCNTION
            likelihood_hit_function = self.mounted_models[name]
            sns.lineplot(x=likelihood_hit_function['v'].likelihood, y=likelihood_hit_function['v'].hits,
                         label=f'V_{name}', ax=self.axd['A'])
            sns.lineplot(x=likelihood_hit_function['d'].likelihood, y=likelihood_hit_function['d'].hits,
                         label=f'D_{name}', ax=self.axd['A'])
            sns.lineplot(x=likelihood_hit_function['j'].likelihood, y=likelihood_hit_function['j'].hits,
                         label=f'J_{name}', ax=self.axd['A'])
            self.plot_point_and_lines(self.axd['B'], 0.7, 0.7)

            self.axd['A'].grid(lw=1, ls=':')
            self.axd['A'].set_ylabel('Average Agreement')
            self.axd['A'].set_xlabel('Average AlingAIR Normalized Likelihood')
            self.axd['A'].locator_params(axis='x', nbins=20)
            self.axd['A'].locator_params(axis='y', nbins=20)
            self.axd['A'].text(-0.1, 1.24, 'A', transform=self.axd['A'].transAxes, fontsize=24, fontweight='bold',
                               va='top',
                               ha='right')

        plt.savefig('likelihood_calibration_plot.png',dpi=300)

        plt.show()