from collections import defaultdict
from matplotlib.ticker import FixedLocator
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm.auto import tqdm
sns.set_context('poster')
from AlignAIR.PostProcessing.AlleleSelector import CappedDynamicConfidenceThreshold
from AlignAIR.Step.Step import Step

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
        ax.annotate(point_label, (x, y), textcoords="offset points", xytext=(0,10), ha='center')

    # Draw lines extending to the axes
    ax.axhline(y=y, color='gray', linestyle='--', xmax=x/ax.get_xlim()[1])
    ax.axvline(x=x, color='gray', linestyle='--', ymax=y/ax.get_ylim()[1])
class ModelLikelihoodSummaryPlotStep(Step):

    def categorize_value(self,value, likelihood_bins=None):
        bin_category = pd.cut([value], bins=likelihood_bins, include_lowest=False)
        return bin_category[0]



    def derive_likelihood_data(self,predict_object):
        likelihood_bins = np.linspace(0, 1, 25)
        self.extractor = CappedDynamicConfidenceThreshold(predict_object.data_config['heavy'])

        alignair_likelihood_hit_function = dict()
        alignair_likelihood_hit_total = dict()

        for gene in tqdm(['v', 'd', 'j']):
            iterator = predict_object.results['cleaned_data'][f'{gene}_allele']
            alignair_likelihood_hit_function[gene] = defaultdict(int)
            alignair_likelihood_hit_total[gene] = defaultdict(int)
            for likelihoods_row, groundtruth_call in tqdm(
                    zip(iterator, predict_object.groundtruth_table[f'{gene}_call']),
                    total=len(iterator)):
                groundtruth_decomposed = groundtruth_call.split(',')
                for index, likelihood in enumerate(np.round(likelihoods_row / np.sum(likelihoods_row), 4)):
                    alignair_likelihood_hit_function[gene][likelihood] += self.extractor[gene][
                                                                              index] in groundtruth_decomposed
                    alignair_likelihood_hit_total[gene][likelihood] += 1

        for gene in alignair_likelihood_hit_function:
            alignair_likelihood_hit_function[gene] = pd.Series(alignair_likelihood_hit_function[gene],
                                                               name='hits').sort_values().to_frame().reset_index().rename(
                columns={'index': 'likelihood'})
            alignair_likelihood_hit_total[gene] = pd.Series(alignair_likelihood_hit_total[gene],
                                                            name='hits').sort_values().to_frame().reset_index().rename(
                columns={'index': 'likelihood'})
            alignair_likelihood_hit_function[gene] = alignair_likelihood_hit_function[gene].set_index('likelihood')
            alignair_likelihood_hit_function[gene] = pd.concat([alignair_likelihood_hit_function[gene],
                                                                alignair_likelihood_hit_total[gene].set_index(
                                                                    'likelihood').rename(columns={'hits': 'counts'})],
                                                               axis=1)
            alignair_likelihood_hit_function[gene] = alignair_likelihood_hit_function[gene].reset_index()
            # alignair_likelihood_hit_function[gene]['hits']/=len(s5f_clean_groundtruth)
            alignair_likelihood_hit_function[gene]['bins'] = pd.cut(
                alignair_likelihood_hit_function[gene]['likelihood'], 25, include_lowest=False)
            alignair_likelihood_hit_function[gene] = alignair_likelihood_hit_function[gene].groupby('bins').agg(
                {'hits': 'sum', 'likelihood': 'mean', 'counts': 'sum'})
            alignair_likelihood_hit_function[gene]['hits'] = alignair_likelihood_hit_function[gene].apply(
                lambda x: np.round(x['hits'] / x['counts'], 5), axis=1)

            self.alignair_likelihood_hit_function = alignair_likelihood_hit_function

    def derive_agg_data(self,predict_object):
        targets = ['v','d','j']
        self.hit_mrate_comparison_data = dict()
        for gene in tqdm(targets):
            self.hit_mrate_comparison_data[gene] = dict()
            ar_top_k_hits = {}

            groundtruth_calls = predict_object.groundtruth_table[f'{gene}_call'].apply(lambda x: set(x.split(',')))

            sorted_alignair = []
            for row in predict_object.results['cleaned_data'][f'{gene}_allele']:
                # if gene == 'd':
                #     # ignore short d
                #     sorted_likelihoods = np.argsort(row[:-1])[::-1]
                # else:
                sorted_likelihoods = np.argsort(row)[::-1]
                sorted_alignair.append(([self.extractor[gene][i] for i in sorted_likelihoods[:3]]))

            for i in tqdm(range(1, 4)):
                filtered = [set(j[:i]) for j in sorted_alignair]
                hits = []
                for ar, gt in zip(filtered, groundtruth_calls):
                    hits.append(len(ar & gt) > 0)
                ar_top_k_hits[i] = hits

            self.thdf = pd.DataFrame(ar_top_k_hits)

            def get_agreement(predicted_labels, ground_truth):
                return [len(set(i) & set(j)) > 0 for i, j in zip(predicted_labels, ground_truth)]


            cfs = {'v': predict_object.script_arguments.v_allele_threshold,
                   'd': predict_object.script_arguments.d_allele_threshold,
                   'j': predict_object.script_arguments.j_allele_threshold}
            caps = {'v': predict_object.script_arguments.v_cap,
                   'd': predict_object.script_arguments.d_cap,
                   'j': predict_object.script_arguments.j_cap}

            predicted_labels = self.extractor.get_alleles(predict_object.results['cleaned_data'][f'{gene}_allele'],
                                                     confidence=cfs[gene], allele=gene, cap=caps[gene])
            predicted_labels = [i[0] for i in predicted_labels]
            # Evaluate the predictions
            hits = get_agreement(predicted_labels, predict_object.groundtruth_table[f'{gene}_call'].apply(lambda x: x.split(',')))

            self.thdf['mutation_rate'] = predict_object.groundtruth_table[f'mutation_rate']

            self.thdf['dt_hits'] = hits

            self.thdf['bins'] = pd.cut(self.thdf.mutation_rate, bins=np.linspace(0, 0.20, 5))

            thdf_agg = self.thdf.groupby('bins').mean()
            self.hit_mrate_comparison_data[gene] = thdf_agg
    def execute(self, predict_object):
        self.log("Generating Model Likelihood Summary Figure")

        self.derive_likelihood_data(predict_object)
        self.derive_agg_data(predict_object)


        mosaic = """
                BBBBBB
                VVDDJJ
                """
        fig = plt.figure(layout="tight", figsize=(20, 15))
        axd = fig.subplot_mosaic(mosaic)

        # AlignAIR LIKELIHOOD FUCNTION
        sns.lineplot(x=self.alignair_likelihood_hit_function['v'].likelihood, y=self.alignair_likelihood_hit_function['v'].hits,
                     label='V', ax=axd['B'])
        sns.lineplot(x=self.alignair_likelihood_hit_function['d'].likelihood, y=self.alignair_likelihood_hit_function['d'].hits,
                     label='D', ax=axd['B'])
        sns.lineplot(x=self.alignair_likelihood_hit_function['j'].likelihood, y=self.alignair_likelihood_hit_function['j'].hits,
                     label='J', ax=axd['B'])
        # axd['B'].axvline(0.7,ymin=0,ymax=0.99)
        # axd['B'].axhline(0.99,xmin=0,xmax=0.7)
        plot_point_and_lines(axd['B'], 0.7, 0.99)

        # sns.lineplot(x=np.linspace(0,1,25),y=np.linspace(0,1,25),color='black',ls='--',label='Identity',ax=axd['B'])
        axd['B'].grid(lw=1, ls=':')
        axd['B'].set_ylabel('Average Agreement')
        axd['B'].set_xlabel('Average AlingAIR Normalized Likelihood')
        axd['B'].locator_params(axis='x', nbins=20)
        axd['B'].locator_params(axis='y', nbins=20)
        axd['B'].text(-0.1, 1.24, 'B', transform=axd['B'].transAxes, fontsize=24, fontweight='bold', va='top',
                      ha='right')



        # HIT COMPARISON
        for ax_name, panel in zip(['D', 'E', 'F'], ['V', 'D', 'J']):
            g = panel.lower()
            thdf_agg = self.hit_mrate_comparison_data[g]

            sns.lineplot(x=thdf_agg.mutation_rate, y=thdf_agg[1], label='AlignAIR: Top 1', color='tab:blue',
                         ax=axd[panel], marker='X')
            sns.lineplot(x=thdf_agg.mutation_rate,y=thdf_agg[2],label='AlignAIR: Top 2',color='tab:green',ax=axd[panel])
            sns.lineplot(x=thdf_agg.mutation_rate, y=thdf_agg['dt_hits'], label='AlignAIR: Dynamic\nThreshold',
                         color='black', ax=axd[panel], marker='X')

            sns.lineplot(x=thdf_agg.mutation_rate, y=thdf_agg[3], label='AlignAIR: Top 3', color='tab:red',
                         ax=axd[panel], marker='X')

            axd[panel].grid(lw=1, ls=':')
            axd[panel].set_title(f'{panel} Allele')
            axd[panel].set_ylabel('Agreement')
            axd[panel].set_xlabel(f'Average {panel} Mutation Rate')
            axd[panel].locator_params(axis='y', nbins=6)
            axd[panel].locator_params(axis='x', nbins=10)
            axd[panel].set_xticklabels(axd[panel].get_xticklabels(), rotation=90)

            axd[panel].legend(loc='best', fontsize=14)
            axd[panel].text(-0.16, 1.1, f'{ax_name}', transform=axd[panel].transAxes, fontsize=24, fontweight='bold',
                            va='top', ha='right')

        plt.savefig(predict_object.script_arguments.save_path+'AlignAIR_Likelihoods_Figure.pdf', dpi=200, bbox_inches='tight')

        return predict_object
