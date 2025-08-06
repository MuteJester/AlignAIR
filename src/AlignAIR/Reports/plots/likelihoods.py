import numpy as np
import pandas as pd
from GenAIRR.dataconfig import DataConfig
from tqdm.auto import tqdm
from collections import defaultdict
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from AlignAIR.Data import MultiDataConfigContainer


class LikelihoodCalibrationPlot:
    """
    Encapsulates logic for analyzing the likelihood of V/D/J gene predictions
    relative to a ground-truth table (mutation rates, hits, top-k hits, etc.).
    """

    def __init__(self, predict_object, groundtruth_table):
        """
        Initializes the LikelihoodAnalysisFigureData object.

        :param predict_object: An object holding prediction data (e.g.,
                               processed predictions, selected allele calls).
        :param groundtruth_table: A DataFrame containing groundtruth fields
                                  (mutation rates, V/D/J calls, etc.).
        """
        self.predict_object = predict_object
        self.groundtruth_table = groundtruth_table

        # Centralize the check for D-gene data.
        if isinstance(predict_object.dataconfig,DataConfig):
            self.has_d_gene = predict_object.dataconfig.metadata.has_d
        elif isinstance(predict_object.dataconfig,MultiDataConfigContainer):
            self.has_d_gene = predict_object.dataconfig.has_at_least_one_d()
        else:
            # Fallback for unexpected dataconfig types
            self.has_d_gene = False


        # Create a dynamic list of genes to process.
        self.gene_types = ['v', 'd', 'j'] if self.has_d_gene else ['v', 'j']

        # Bins for discretizing likelihood values
        self.likelihood_bins = np.linspace(0, 1, 25)

        # Populate groundtruth_table with v/d/j mutation rates
        self.update_ground_truth_mutation_rates()

        # Compute all additional metadata (likelihood hits, top-k hits, etc.)
        self.derive_metadata()

    def categorize_value(self, value):
        """
        Categorize a single value into the pre-defined likelihood bins using pandas.cut.
        """
        bin_category = pd.cut([value], bins=self.likelihood_bins, include_lowest=False)
        return bin_category[0]

    def update_ground_truth_mutation_rates(self):
        """
        Computes mutation rates and adds them to the 'groundtruth_table'.
        This version is more robust and handles the absence of D-gene data cleanly.
        """
        v_rates, d_rates, j_rates = [], [], []

        for _, row in tqdm(self.groundtruth_table.iterrows(),
                           total=len(self.groundtruth_table),
                           desc='Calculating mutation rates'):

            mutation_positions = list(eval(row['mutations']).keys())

            # V-gene processing
            v_len = row['v_sequence_end'] - row['v_sequence_start']
            v_len = v_len if v_len > 0 else 1
            v_count = sum(1 for pos in mutation_positions if row['v_sequence_start'] <= pos <= row['v_sequence_end'])
            v_rates.append(v_count / v_len)

            # J-gene processing
            j_len = row['j_sequence_end'] - row['j_sequence_start']
            j_len = j_len if j_len > 0 else 1
            j_count = sum(1 for pos in mutation_positions if row['j_sequence_start'] <= pos <= row['j_sequence_end'])
            j_rates.append(j_count / j_len)

            # Conditional D-gene processing using the instance flag
            if self.has_d_gene and pd.notna(row.get('d_call')):
                d_len = row['d_sequence_end'] - row['d_sequence_start']
                d_len = d_len if d_len > 0 else 1
                d_count = sum(
                    1 for pos in mutation_positions if row['d_sequence_start'] <= pos <= row['d_sequence_end'])
                d_rates.append(d_count / d_len)
            else:
                # Append NaN to keep list lengths consistent if D is missing for a row
                d_rates.append(np.nan)

        self.groundtruth_table['v_mutation_rate'] = v_rates
        self.groundtruth_table['j_mutation_rate'] = j_rates
        # Only add the D mutation column if D-gene is present
        if self.has_d_gene:
            self.groundtruth_table['d_mutation_rate'] = d_rates

    def derive_metadata(self):
        """
        Performs two major analyses:
        1. AlignAIR likelihood hits.
        2. Top-k hits for each available gene.
        """
        self.alignair_likelihood_hit_function = self._compute_alignair_likelihood_hits()
        self._compute_top_k_hits()

    def _compute_alignair_likelihood_hits(self):
        """
        Computes how often the 'predicted' allele calls (likelihoods) match
        the groundtruth calls. Groups them into bins and aggregates hits.
        Returns a dictionary keyed by gene type.
        """
        alignair_likelihood_hit_function = {}
        alignair_likelihood_hit_total = {}

        # Use the dynamic self.gene_types list
        for gene in tqdm(self.gene_types, desc='AlignAIR Likelihood Hits'):
            if f'{gene}_allele' not in self.predict_object.processed_predictions:
                continue

            iterator = self.predict_object.processed_predictions[f'{gene}_allele'].round(2)
            alignair_likelihood_hit_function[gene] = defaultdict(int)
            alignair_likelihood_hit_total[gene] = defaultdict(int)
            groundtruth_calls = self.groundtruth_table[f'{gene}_call']
            reverse_mapping = self.predict_object.threshold_extractor_instances[gene].properties_map[gene.upper()][
                'reverse_mapping']

            for likelihoods_row, groundtruth_call in tqdm(zip(iterator, groundtruth_calls),
                                                          total=len(iterator), desc=f'Gene: {gene}'):
                groundtruth_decomposed = set(groundtruth_call.split(','))
                for idx, likelihood in enumerate(likelihoods_row):
                    is_hit = (reverse_mapping[idx] in groundtruth_decomposed)
                    alignair_likelihood_hit_function[gene][likelihood] += int(is_hit)
                    alignair_likelihood_hit_total[gene][likelihood] += 1

        for gene, hits_dict in alignair_likelihood_hit_function.items():
            hits_df = pd.Series(hits_dict, name='hits').sort_index().to_frame()
            total_df = pd.Series(alignair_likelihood_hit_total[gene], name='counts').sort_index().to_frame()
            merged_df = hits_df.join(total_df, how='left').reset_index().rename(columns={'index': 'likelihood'})
            merged_df['bins'] = pd.cut(merged_df['likelihood'], 25, include_lowest=False)
            grouped = merged_df.groupby('bins').agg({'hits': 'sum', 'counts': 'sum', 'likelihood': 'mean'})
            grouped['hits'] = (grouped['hits'] / grouped['counts']).round(5)
            alignair_likelihood_hit_function[gene] = grouped

        return alignair_likelihood_hit_function

    @staticmethod
    def get_agreement(predicted_labels, ground_truth):
        return [len(set(i) & set(j)) > 0 for i, j in zip(predicted_labels, ground_truth)]

    def _compute_top_k_hits(self):
        """
        Computes top-k hits for each available gene type.
        """
        self.thdfs = {'v': {}, 'j': {}}
        if self.has_d_gene:
            self.thdfs['d'] = {}

        # Use the dynamic self.gene_types list
        for gene in self.thdfs:
            ar_top_k_hits = {}
            gt_sets = self.groundtruth_table[f'{gene}_call'].apply(lambda x: set(x.split(',')))

            for k in tqdm(range(1, 4), desc=f'Top-k Hits for {gene.upper()}'):
                predicted_top_k = (set(alleles[:k]) for alleles in self.predict_object.selected_allele_calls[gene])
                hits = self.get_agreement(predicted_top_k, gt_sets)
                ar_top_k_hits[k] = hits

            thdf = pd.DataFrame(ar_top_k_hits)
            thdf['dt_hits'] = self.get_agreement(self.predict_object.selected_allele_calls[gene], gt_sets)
            thdf['mutation_rate'] = self.groundtruth_table[f'{gene}_mutation_rate']
            rate_bins = np.linspace(0, 0.25, 25)
            thdf['bins'] = pd.cut(thdf['mutation_rate'], bins=rate_bins)
            thdf_agg = thdf.groupby('bins').mean()

            self.thdfs[gene]['thdf'] = thdf
            self.thdfs[gene]['thdf_agg'] = thdf_agg

    def get_enhanced_panel(self):
        """
        Enhanced version with improved visual clarity, better colors, and confidence intervals.
        """
        color_palette = {
            'v': '#1f77b4', 'd': '#ff7f0e', 'j': '#2ca02c',
            'identity': '#d62728', 'confidence_band': 'rgba(128, 128, 128, 0.2)'
        }

        # Create subplot titles dynamically based on D-gene presence.
        subplot_titles = [
            'Enhanced Likelihood Calibration Analysis',
            'V Gene Performance vs Mutation Rate',
            'D Gene Performance vs Mutation Rate' if self.has_d_gene else '',  # Empty title if no D
            'J Gene Performance vs Mutation Rate'
        ]

        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=subplot_titles,
            row_heights=[0.6, 0.4],
            column_widths=[1.0, 0.33, 0.33],
            specs=[
                [{"type": "xy", 'colspan': 3}, None, None],
                [{"type": "xy"}, {"type": "xy"}, {"type": "xy"}],
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.08
        )

        # Loop over the dynamic gene list for the first plot
        for gene in self.gene_types:
            if gene not in self.alignair_likelihood_hit_function:
                continue
            data = self.alignair_likelihood_hit_function[gene]

            hits_proportion = data['hits']
            n_samples = data['counts'].sum()  # Use sum of counts in bins for accuracy
            std_error = np.sqrt(hits_proportion * (1 - hits_proportion) / n_samples)
            ci_lower = np.maximum(0, hits_proportion - 1.96 * std_error)
            ci_upper = np.minimum(1, hits_proportion + 1.96 * std_error)

            fig.add_trace(go.Scatter(
                x=data['likelihood'], y=hits_proportion, mode='lines+markers',
                name=f"{gene.upper()} Gene Calibration",
                line=dict(color=color_palette[gene], width=3),
                marker=dict(size=8, symbol='circle', line=dict(width=1, color='white')),
                hovertemplate=f"{gene.upper()} Gene<br>Likelihood: %{{x:.3f}}<br>Agreement: %{{y:.3f}}<br><extra></extra>",
                showlegend=True
            ), row=1, col=1)

            fig.add_trace(go.Scatter(
                x=np.concatenate([data['likelihood'], data['likelihood'][::-1]]),
                y=np.concatenate([ci_upper, ci_lower[::-1]]),
                fill='toself', fillcolor=color_palette['confidence_band'],
                line=dict(color='rgba(255,255,255,0)'),
                name=f"{gene.upper()} 95% CI", showlegend=False, hoverinfo='skip'
            ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode='lines', name='Perfect Calibration (y=x)',
            line=dict(color=color_palette['identity'], width=2, dash='dash'),
            hovertemplate="Perfect Calibration<br>y = x<extra></extra>", showlegend=True
        ), row=1, col=1)

        fig.update_xaxes(title_text="Average AlignAIR Likelihood", row=1, col=1, range=[0, 1], tickformat='.2f',
                         gridcolor='lightgray', showgrid=True)
        fig.update_yaxes(title_text="Proportion of Correct Predictions", row=1, col=1, range=[0, 1], tickformat='.2f',
                         gridcolor='lightgray', showgrid=True)

        # Performance vs mutation rate subplots
        genes_for_subplot = ['v', 'd', 'j']
        subplot_positions = [(2, 1), (2, 2), (2, 3)]
        colors_top_k = ['#1f77b4', '#ff7f0e', '#2ca02c']

        for i, gene in enumerate(genes_for_subplot):
            row, col = subplot_positions[i]

            # Explicitly skip the D-gene plot if no data is available
            if gene == 'd' and not self.has_d_gene:
                fig.add_annotation(text="No D-gene data", x=0.5, y=0.5, xref=f"x{col}", yref=f"y{row}", showarrow=False,
                                   font=dict(size=14, color="gray"))
                fig.update_xaxes(title_text="", row=row, col=col, showticklabels=False)
                fig.update_yaxes(title_text="", row=row, col=col, showticklabels=False)
                continue

            if gene not in self.thdfs or not self.thdfs[gene]:
                fig.add_annotation(text=f"No {gene.upper()}-gene data", x=0.5, y=0.5, xref=f"x{col}", yref=f"y{row}",
                                   showarrow=False, font=dict(size=14, color="gray"))
                continue

            data = self.thdfs[gene]['thdf_agg']
            x_vals = data['mutation_rate']
            line_configs = [
                {'y': data[1], 'name': 'Top-1 Hit Rate', 'color': colors_top_k[0], 'dash': None},
                {'y': data[3], 'name': 'Top-3 Hit Rate', 'color': colors_top_k[1], 'dash': None},
                {'y': data['dt_hits'], 'name': 'Dynamic Threshold', 'color': colors_top_k[2], 'dash': 'dot'}
            ]
            for line_config in line_configs:
                fig.add_trace(go.Scatter(
                    x=x_vals, y=line_config['y'], mode='lines+markers', name=line_config['name'],
                    line=dict(color=line_config['color'], width=3, dash=line_config['dash']),
                    marker=dict(size=6), showlegend=(i == 0),
                    legendgroup=line_config['name'].lower().replace(' ', '_'),
                    hovertemplate=f"{gene.upper()} {line_config['name']}<br>Mutation Rate: %{{x:.4f}}<br>Hit Rate: %{{y:.3f}}<extra></extra>"
                ), row=row, col=col)

            fig.update_xaxes(title_text="Mutation Rate", row=row, col=col, tickformat='.3f', gridcolor='lightgray',
                             gridwidth=0.5, showgrid=True)
            fig.update_yaxes(title_text="Hit Rate", row=row, col=col, range=[0, 1], tickformat='.2f',
                             gridcolor='lightgray', gridwidth=0.5, showgrid=True)

        fig.update_layout(
            height=1200, title=dict(text="Enhanced AlignAIR Likelihood Calibration Report", x=0.5, font=dict(size=20)),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            plot_bgcolor='white', paper_bgcolor='white', font=dict(family="Arial, sans-serif", size=12),
            hovermode="closest", autosize=True
        )

        return fig

    def get_original_panel(self):
        """Original version of the panel for backward compatibility."""
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=('Likelihood Calibration', 'V-Gene Hits', "D-Gene Hits", 'J-Gene Hits'),
            row_heights=[0.5, 0.5],
            column_widths=[0.3, 0.3, 0.3],
            specs=[
                [{"type": "xy", 'colspan': 3}, None, None],
                [{"type": "xy"}, {"type": "xy"}, {"type": "xy"}],
            ]
        )

        # Use the dynamic self.gene_types list for the calibration plot
        colors = {'v': 'blue', 'd': 'red', 'j': 'green'}
        for allele in self.gene_types:
            if allele in self.alignair_likelihood_hit_function:
                fig.add_trace(go.Scatter(
                    x=self.alignair_likelihood_hit_function[allele]['likelihood'],
                    y=self.alignair_likelihood_hit_function[allele]['hits'],
                    mode='lines', name=f"{allele.upper()} Allele Calibration",
                    line=dict(color=colors[allele])
                ), row=1, col=1)

        fig.add_trace(
            go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Identity Line', line=dict(color='black', dash='dash')),
            row=1, col=1)
        fig.update_xaxes(title_text="Average AlignAIR Likelihood", row=1, col=1)
        fig.update_yaxes(title_text="Average Agreement", row=1, col=1)
        fig.update_layout(hovermode="x unified")

        # Hit Comparison
        panels = ['V', 'D', 'J']
        for i, panel in enumerate(panels):
            g = panel.lower()

            # Skip D-panel if no D-gene data
            if g == 'd' and not self.has_d_gene:
                if fig.layout.annotations[i + 1]:
                    fig.layout.annotations[i + 1].text = 'No D-Gene Data'
                continue

            if g in self.thdfs and self.thdfs[g]:
                fig.add_trace(go.Scatter(x=self.thdfs[g]['thdf_agg']['mutation_rate'], y=self.thdfs[g]['thdf_agg'][1],
                                         mode='lines+markers', name='AlignAIR: Top 1', line=dict(color='blue'),
                                         showlegend=(i == 0)), row=2, col=i + 1)
                fig.add_trace(go.Scatter(x=self.thdfs[g]['thdf_agg']['mutation_rate'], y=self.thdfs[g]['thdf_agg'][3],
                                         mode='lines+markers', name='AlignAIR: Top 3', line=dict(color='red'),
                                         showlegend=(i == 0)), row=2, col=i + 1)
                fig.add_trace(
                    go.Scatter(x=self.thdfs[g]['thdf_agg']['mutation_rate'], y=self.thdfs[g]['thdf_agg']['dt_hits'],
                               mode='lines+markers', name='AlignAIR: Dynamic Threshold',
                               line=dict(color='green', dash='dot'), showlegend=(i == 0)), row=2, col=i + 1)

        fig.update_layout(height=1000, width=1200, showlegend=True, title_text="AlignAIR Likelihood Report")

        return fig

    def get_panel(self, enhanced=True):
        """
        Get the likelihood calibration panel.

        Args:
            enhanced (bool): If True, returns the enhanced version with better visuals.
                           If False, returns the original version.
        """
        if enhanced:
            return self.get_enhanced_panel()
        else:
            return self.get_original_panel()