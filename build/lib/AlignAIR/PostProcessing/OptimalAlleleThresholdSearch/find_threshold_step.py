import pickle
import random

import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt

from AlignAIR.PostProcessing.AlleleSelector import CappedDynamicConfidenceThreshold
from AlignAIR.Step.Step import Step
from tqdm.auto import tqdm
sns.set_context('poster')
def get_agreement(predicted_labels, ground_truth):
    return [len(set(i) & set(j)) > 0 for i, j in zip(predicted_labels, ground_truth)]




def find_optimal_hyperparameters(likelihoods, ground_truth, extractor, get_agreement, cap_range, confidence_range,
                                 allele='v'):
    results = []
    NS = len(ground_truth)
    for cap in tqdm(cap_range):
        for confidence in (confidence_range):
            # Get predictions using the current cap and confidence
            predicted_labels = extractor.get_alleles(likelihoods, confidence=confidence, allele=allele, cap=cap)
            predicted_labels = [i[0] for i in predicted_labels]
            # Evaluate the predictions
            hits = get_agreement(predicted_labels, ground_truth)
            num_hits = sum(hits) / NS
            avg_labels_returned = np.mean([len(pred) for pred in predicted_labels])
            results.append({
                'cap': cap,
                'confidence': confidence,
                'num_hits': num_hits,
                'avg_labels_returned': avg_labels_returned
            })

    # Convert results to a DataFrame for easy analysis
    results_df = pd.DataFrame(results)
    return results_df




class OptimalAlleleThresholdSearchStep(Step):

    def execute(self, predict_object):
        # Example usage:
        cap_range = range(1, 4)  # Example range for cap
        confidence_range = np.arange(0.1, 1.0, 0.05)  # Example range for confidence
        max_samples = 300_000

        # Function to get random or all indexes
        def get_indexes(data_length):
            if data_length > max_samples:
                return random.sample(range(data_length), max_samples)
            else:
                return list(range(data_length))

        # Function to sample data
        def sample_data(cleaned_data, ground_truth):
            indexes = get_indexes(len(cleaned_data))
            sampled_cleaned_data = [cleaned_data[i] for i in indexes]
            sampled_ground_truth = [ground_truth[i] for i in indexes]
            return sampled_cleaned_data, sampled_ground_truth

        # Assuming likelihoods and ground_truth are already defined
        self.log("Starting to Search for Optimal Cap and Threshold for V Alleles")

        v_cleaned_data, v_ground_truth = sample_data(
            predict_object.results['cleaned_data']['v_allele'],
            predict_object.groundtruth_table.v_call.apply(lambda x: x.split(',')).tolist()
        )

        extractor = CappedDynamicConfidenceThreshold(predict_object.data_config['heavy'])
        v_results_df = find_optimal_hyperparameters(
            v_cleaned_data, v_ground_truth, extractor, get_agreement, cap_range, confidence_range
        )

        self.log("Starting to Search for Optimal Cap and Threshold for D Alleles")

        d_cleaned_data, d_ground_truth = sample_data(
            predict_object.results['cleaned_data']['d_allele'],
            predict_object.groundtruth_table.d_call.apply(lambda x: x.split(',')).tolist()
        )

        d_results_df = find_optimal_hyperparameters(
            d_cleaned_data, d_ground_truth, extractor, get_agreement, cap_range, confidence_range, allele='d'
        )

        self.log("Starting to Search for Optimal Cap and Threshold for J Alleles")

        j_cleaned_data, j_ground_truth = sample_data(
            predict_object.results['cleaned_data']['j_allele'],
            predict_object.groundtruth_table.j_call.apply(lambda x: x.split(',')).tolist()
        )

        j_results_df = find_optimal_hyperparameters(
            j_cleaned_data, j_ground_truth, extractor, get_agreement, cap_range, confidence_range, allele='j'
        )

        with open(predict_object.script_arguments.save_path + 'cap_confidence_selection_chart_data.pkl', 'wb') as h:
            pickle.dump({'v': v_results_df, 'd': d_results_df, 'j': j_results_df}, h)

        def select_best_hyperparameters(results_df, weight_hits=0.7, weight_labels=0.3):
            # Normalize the num_hits and avg_labels_returned
            results_df['normalized_hits'] = (results_df['num_hits'] - results_df['num_hits'].min()) / (
                        results_df['num_hits'].max() - results_df['num_hits'].min())
            results_df['normalized_labels'] = (results_df['avg_labels_returned'] - results_df[
                'avg_labels_returned'].min()) / (results_df['avg_labels_returned'].max() - results_df[
                'avg_labels_returned'].min())

            # Invert the normalized_labels to treat lower values as better
            results_df['normalized_labels'] = 1 - results_df['normalized_labels']

            # Compute the weighted score
            results_df['score'] = (weight_hits * results_df['normalized_hits']) + (
                        weight_labels * results_df['normalized_labels'])

            # Select the combination with the highest score
            best_combination = results_df.loc[results_df['score'].idxmax()]

            return best_combination

        best_caps = {
            'V': 0.9,
            'D': 0.3,
            'J': 0.8,
        }
        best_ths = {
            'V': 0.9,
            'D': 0.3,
            'J': 0.8,
        }

        for gene,df in zip(['v','d','j'],[v_results_df,d_results_df,j_results_df]):
            best_hyperparameters = select_best_hyperparameters(df)

            rstring = f"""Results for: {gene.upper()} Allele\n\nBest Cap: {best_hyperparameters['cap']}\nBest Confidence: {best_hyperparameters['confidence']}\nNumber of Hits: {best_hyperparameters['num_hits']}\nAverage Labels Returned: {best_hyperparameters['avg_labels_returned']}\n<*>\n"""

            print(rstring)
            with open(predict_object.script_arguments.save_path+'Threshold_Search_Results_Textual.txt','a') as h:
                h.write(rstring)

            best_caps[gene.upper()] = best_hyperparameters['cap']
            best_ths[gene.upper()] = best_hyperparameters['confidence']


        plt.figure(figsize=(14, 20))
        flabel = {0: 'A', 1: 'B', 2: 'C'}
        for en, (_allele, _dt) in enumerate(zip(['V', 'D', 'J'], [v_results_df, d_results_df, j_results_df])):
            plt.subplot(3, 1, en + 1)
            ax = sns.lineplot(data=_dt, x='confidence', y='num_hits', hue='cap', palette='tab10')
            plt.ylabel(f'{_allele} Agreement')
            ax.text(-0.1, 1.1, flabel[en], transform=ax.transAxes, fontsize=24, fontweight='bold', va='top', ha='right')

            ax.locator_params(axis='y', nbins=10)
            ax.locator_params(axis='x', nbins=10)
            plt.grid(lw=2, ls=':')
            plt.xlabel('Confidence Threshold')

            ax2 = ax.twinx()
            sns.lineplot(data=_dt, x='confidence', y='avg_labels_returned', hue='cap', palette='tab10', ax=ax2, ls='--')
            plt.ylabel(f'{_allele} Avg Calls')

            ax2.axvline(best_ths[_allele], label='Used\nConfidence', color='black', ls='--')
            ax2.legend(ncols=2, loc='best' if _allele != 'D' else 'lower right')

            ax2.locator_params(axis='y', nbins=10)
            ax2.locator_params(axis='x', nbins=10)

            if ax != 0:
                # ax2.legend().remove()
                ax.legend().remove()

        plt.tight_layout()
        plt.savefig(predict_object.script_arguments.save_path+'threshold_selection_function_iteration.pdf', dpi=150, facecolor='white', bbox_inches='tight')

        # predict_object.caps_and_thresholds['caps'] = {key.lower():value for key,value in best_caps.items()}
        # predict_object.caps_and_thresholds['thresholds'] = {key.lower():value for key,value in best_ths.items()}

        predict_object.script_arguments.v_allele_threshold = best_ths['V']
        predict_object.script_arguments.d_allele_threshold = best_ths['D']
        predict_object.script_arguments.j_allele_threshold = best_ths['J']
        predict_object.script_arguments.v_cap = best_caps['V']
        predict_object.script_arguments.d_cap = best_caps['D']
        predict_object.script_arguments.j_cap = best_caps['J']

        self.log('Optimal Caps and Thresholds Derived and Saved!')

        return predict_object
