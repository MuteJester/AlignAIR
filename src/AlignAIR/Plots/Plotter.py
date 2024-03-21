from matplotlib.ticker import FuncFormatter
from sklearn.metrics import multilabel_confusion_matrix, precision_recall_curve, hamming_loss, accuracy_score, f1_score, \
    roc_curve
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm

def get_top_i(calls, i=1):
    if len(calls) < i:
        return calls
    else:
        return calls[:i]


class ModelAnalysisPlotter:
    def __init__(self, groundtruth_df, alignairr_dict, allele='v'):
        self.ground_truth = groundtruth_df
        self.alignairr = alignairr_dict
        self.allele = allele

    def derive_metadata(self):
        self.ar_hits = list(map(lambda x: len(set(x[0].split(',')) & set(x[1])) > 0,
                                zip(self.ground_truth[f'{self.allele}_allele'], self.alignairr[f'{self.allele}_call'])))
        self.ar_calls = list(map(len, self.alignairr[f'{self.allele}_call']))
        self.mr_bins = pd.cut(self.ground_truth['mutation_rate'], 25, precision=1)
        self.ar_hits_t1 = list(map(lambda x: len(set(x[0].split(',')) & set(get_top_i(x[1], 1))) > 0,
                                   zip(self.ground_truth[f'{self.allele}_allele'],
                                       self.alignairr[f'{self.allele}_call'])))
        self.ar_hits_t2 = list(map(lambda x: len(set(x[0].split(',')) & set(get_top_i(x[1], 2))) > 0,
                                   zip(self.ground_truth[f'{self.allele}_allele'],
                                       self.alignairr[f'{self.allele}_call'])))
        self.ar_hits_t3 = list(map(lambda x: len(set(x[0].split(',')) & set(get_top_i(x[1], 3))) > 0,
                                   zip(self.ground_truth[f'{self.allele}_allele'],
                                       self.alignairr[f'{self.allele}_call'])))

    def derive_aggregated_dataframe(self):
        self.combined_hits = pd.DataFrame({'ar_hits': self.ar_hits, 'bins': self.mr_bins,
                                           'ar_calls': self.ar_calls,
                                           'ar_hits_i1': self.ar_hits_t1,
                                           'ar_hits_i2': self.ar_hits_t2,
                                           'ar_hits_i3': self.ar_hits_t3, })
        self.combined_hits_group = self.combined_hits.groupby('bins').mean()

    def calls_and_agreement_plot(self):
        xx = np.arange(0, len(self.combined_hits_group))

        plt.xticks(xx, labels=self.combined_hits_group.index, rotation=90)
        ax = sns.lineplot(x=xx, y=self.combined_hits_group['ar_hits'], label='ALignAIRR', marker='x')
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
        plt.ylabel('Agreement with Ground Truth')
        plt.grid(lw=2, ls=':')
        plt.xlabel('Mutation Rate Range')

        ax2 = ax.twinx()
        sns.lineplot(x=xx, y=self.combined_hits_group['ar_calls'], marker='x', ax=ax2, ls='--')
        ax2.set_ylabel('Number of Calls')

    def top_three_alleles_plot(self):
        xx = np.arange(0, len(self.combined_hits_group))

        ax = sns.lineplot(x=xx, y=self.combined_hits_group['ar_hits_i1'], label='Top 1', marker='o', ms=10)
        sns.lineplot(x=xx, y=self.combined_hits_group['ar_hits_i2'], label='Top 2', marker='o', ms=10)
        sns.lineplot(x=xx, y=self.combined_hits_group['ar_hits_i3'], label='Top 3', marker='o', ms=10, color='black')
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
        plt.xticks(xx, labels=self.combined_hits_group.index, rotation=90)
        plt.grid(lw=2, ls=':')
        plt.ylabel('Agreement with Groundtruth')

    def derive_allele_wise_metric_dataframe(self):
        true_labels = self.ground_truth[f'{self.allele}_allele'].apply(lambda x: x.split(',')).to_list()
        predicted_labels = self.alignairr[f'{self.allele}_call']

        all_labels = set().union(*true_labels, *predicted_labels)
        label_to_index = {label: i for i, label in enumerate(sorted(all_labels))}

        def binarize(labels):
            binary_labels = np.zeros(len(all_labels))
            for label in labels:
                binary_labels[label_to_index[label]] = 1
            return binary_labels

        y_true = np.array([binarize(labels) for labels in true_labels])
        y_pred = np.array([binarize(labels) for labels in predicted_labels])
        # Calculating metrics for each label
        label_metrics = {
            'Label': [],
            'Precision': [],
            'Recall': [],
            'F1-Score': []
        }

        for i, label in tqdm(enumerate(sorted(all_labels)), total=len(all_labels)):
            label_metrics['Label'].append(label)
            label_metrics['Precision'].append(precision_score(y_true[:, i], y_pred[:, i], zero_division=0))
            label_metrics['Recall'].append(recall_score(y_true[:, i], y_pred[:, i], zero_division=0))
            label_metrics['F1-Score'].append(f1_score(y_true[:, i], y_pred[:, i], zero_division=0))

        self.metrics_df = pd.DataFrame(label_metrics).set_index('Label')


