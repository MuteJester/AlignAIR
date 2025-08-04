import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from collections import defaultdict
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px


class EnhancedLikelihoodCalibrationPlot:
    """
    Enhanced version of LikelihoodCalibrationPlot with improved visual clarity,
    better color schemes, enhanced interactivity, and clearer annotations.
    """

    def __init__(self, predict_object, groundtruth_table):
        """
        Initializes the enhanced likelihood calibration plot.

        :param predict_object: An object holding prediction data
        :param groundtruth_table: A DataFrame containing groundtruth fields
        """
        self.predict_object = predict_object
        self.groundtruth_table = groundtruth_table

        # Enhanced binning with more granular control
        self.likelihood_bins = np.linspace(0, 1, 25)
        
        # Color palette for better visual distinction
        self.color_palette = {
            'v': '#1f77b4',  # Blue
            'd': '#ff7f0e',  # Orange  
            'j': '#2ca02c',  # Green
            'identity': '#d62728',  # Red for identity line
            'confidence_band': 'rgba(128, 128, 128, 0.2)'  # Gray with transparency
        }
        
        # Enhanced styling
        self.line_styles = {
            'v': dict(width=3, dash=None),
            'd': dict(width=3, dash=None),
            'j': dict(width=3, dash=None),
            'identity': dict(width=2, dash='dash'),
            'confidence': dict(width=1, dash='dot')
        }

        # Populate groundtruth_table with mutation rates
        self.update_ground_truth_mutation_rates()

        # Compute all metadata
        self.derive_metadata()

    def categorize_value(self, value):
        """Categorize a single value into pre-defined likelihood bins."""
        bin_category = pd.cut([value], bins=self.likelihood_bins, include_lowest=False)
        return bin_category[0]

    def update_ground_truth_mutation_rates(self):
        """
        Computes mutation rates for V, D, and J segments with improved error handling.
        """
        mutation_rates = {'v': [], 'd': [], 'j': []}

        for _, row in tqdm(self.groundtruth_table.iterrows(),
                          total=len(self.groundtruth_table),
                          desc='Calculating mutation rates'):

            # Extract segment lengths with better error handling
            segments = {
                'v': (row.get('v_sequence_start', 0), row.get('v_sequence_end', 1)),
                'j': (row.get('j_sequence_start', 0), row.get('j_sequence_end', 1))
            }
            
            # Add D segment if present
            if 'd_call' in row and pd.notna(row.get('d_sequence_start')):
                segments['d'] = (row.get('d_sequence_start', 0), row.get('d_sequence_end', 1))

            # Parse mutations safely
            try:
                mutation_positions = list(eval(row['mutations']).keys()) if row['mutations'] else []
            except:
                mutation_positions = []

            # Calculate mutation rates for each segment
            for gene, (start, end) in segments.items():
                length = max(end - start, 1)  # Avoid division by zero
                mutation_count = sum(1 for pos in mutation_positions if start <= pos <= end)
                mutation_rates[gene].append(mutation_count / length)

        # Update groundtruth table
        for gene, rates in mutation_rates.items():
            if rates:  # Only add if we have data
                self.groundtruth_table[f'{gene}_mutation_rate'] = rates
            else:
                self.groundtruth_table[f'{gene}_mutation_rate'] = np.nan

    def derive_metadata(self):
        """Performs likelihood analysis with enhanced error handling."""
        self.alignair_likelihood_hit_function = self._compute_alignair_likelihood_hits()
        self._compute_top_k_hits()

    def _compute_alignair_likelihood_hits(self):
        """
        Enhanced likelihood hits computation with better statistical measures.
        """
        alignair_likelihood_hit_function = {}
        alignair_likelihood_hit_total = {}
        confidence_intervals = {}

        genes_to_process = ['v', 'd', 'j']
        available_genes = [gene for gene in genes_to_process 
                          if f'{gene}_allele' in self.predict_object.processed_predictions]

        for gene in tqdm(available_genes, desc='Computing likelihood hits'):
            iterator = self.predict_object.processed_predictions[f'{gene}_allele'].round(2)
            
            alignair_likelihood_hit_function[gene] = defaultdict(int)
            alignair_likelihood_hit_total[gene] = defaultdict(int)

            groundtruth_calls = self.groundtruth_table[f'{gene}_call']
            reverse_mapping = self.predict_object.threshold_extractor_instances[gene].properties_map[gene.upper()]['reverse_mapping']

            for likelihoods_row, groundtruth_call in tqdm(
                    zip(iterator, groundtruth_calls),
                    total=len(iterator),
                    desc=f'Processing {gene.upper()} gene'):

                groundtruth_decomposed = set(str(groundtruth_call).split(','))
                
                for idx, likelihood in enumerate(likelihoods_row):
                    is_hit = reverse_mapping[idx] in groundtruth_decomposed
                    alignair_likelihood_hit_function[gene][likelihood] += int(is_hit)
                    alignair_likelihood_hit_total[gene][likelihood] += 1

        # Enhanced binning with confidence intervals
        for gene in alignair_likelihood_hit_function:
            hits_df = pd.Series(alignair_likelihood_hit_function[gene], name='hits').sort_index().to_frame()
            total_df = pd.Series(alignair_likelihood_hit_total[gene], name='counts').sort_index().to_frame()

            merged_df = hits_df.join(total_df, how='left').reset_index().rename(columns={'index': 'likelihood'})
            merged_df['bins'] = pd.cut(merged_df['likelihood'], 25, include_lowest=False)

            grouped = merged_df.groupby('bins').agg({
                'hits': 'sum',
                'counts': 'sum',
                'likelihood': 'mean'
            })
            
            # Calculate proportion and confidence intervals
            grouped['proportion'] = grouped['hits'] / grouped['counts']
            grouped['std_error'] = np.sqrt(grouped['proportion'] * (1 - grouped['proportion']) / grouped['counts'])
            grouped['ci_lower'] = np.maximum(0, grouped['proportion'] - 1.96 * grouped['std_error'])
            grouped['ci_upper'] = np.minimum(1, grouped['proportion'] + 1.96 * grouped['std_error'])

            alignair_likelihood_hit_function[gene] = grouped

        return alignair_likelihood_hit_function

    @staticmethod
    def get_agreement(predicted_labels, ground_truth):
        """Enhanced agreement calculation with better error handling."""
        agreements = []
        for pred, truth in zip(predicted_labels, ground_truth):
            try:
                pred_set = set(pred) if hasattr(pred, '__iter__') else {pred}
                truth_set = set(str(truth).split(','))
                agreements.append(len(pred_set & truth_set) > 0)
            except:
                agreements.append(False)
        return agreements

    def _compute_top_k_hits(self):
        """Enhanced top-k hits computation."""
        self.thdfs = {'v': {'thdf': None, 'thdf_agg': None},
                      'd': {'thdf': None, 'thdf_agg': None},
                      'j': {'thdf': None, 'thdf_agg': None}}

        available_genes = ['v', 'j']
        if 'd_call' in self.groundtruth_table.columns:
            available_genes.append('d')

        for gene in available_genes:
            if gene not in self.predict_object.selected_allele_calls:
                continue
                
            ar_top_k_hits = {}
            gt_calls = self.groundtruth_table[f'{gene}_call'].apply(
                lambda x: set(str(x).split(',')) if pd.notna(x) else set()
            )

            for k in tqdm(range(1, 4), desc=f'Computing top-k hits for {gene.upper()}'):
                predicted_top_k = [set(alleles[:k]) if len(alleles) >= k else set(alleles) 
                                 for alleles in self.predict_object.selected_allele_calls[gene]]
                hits = self.get_agreement(predicted_top_k, gt_calls)
                ar_top_k_hits[k] = hits

            # Create enhanced DataFrame
            thdf = pd.DataFrame(ar_top_k_hits)
            thdf['dt_hits'] = self.get_agreement(self.predict_object.selected_allele_calls[gene], gt_calls)
            thdf['mutation_rate'] = self.groundtruth_table[f'{gene}_mutation_rate']

            # Enhanced binning
            rate_bins = np.linspace(0, min(0.25, thdf['mutation_rate'].max()), 25)
            thdf['bins'] = pd.cut(thdf['mutation_rate'], bins=rate_bins)

            # Enhanced aggregation with confidence intervals
            thdf_agg = thdf.groupby('bins').agg({
                1: ['mean', 'std', 'count'],
                2: ['mean', 'std', 'count'],
                3: ['mean', 'std', 'count'],
                'dt_hits': ['mean', 'std', 'count'],
                'mutation_rate': 'mean'
            })

            self.thdfs[gene]['thdf'] = thdf
            self.thdfs[gene]['thdf_agg'] = thdf_agg

    def get_enhanced_panel(self):
        """
        Create an enhanced interactive panel with improved visual clarity.
        """
        # Create enhanced subplot layout
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=(
                'Likelihood Calibration Analysis', 
                'V Gene Performance vs Mutation Rate',
                'D Gene Performance vs Mutation Rate', 
                'J Gene Performance vs Mutation Rate'
            ),
            row_heights=[0.6, 0.4],
            column_widths=[1.0, 0.33, 0.33],
            specs=[
                [{"type": "xy", 'colspan': 3}, None, None],
                [{"type": "xy"}, {"type": "xy"}, {"type": "xy"}],
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.08
        )

        # Enhanced likelihood calibration plot
        available_genes = [gene for gene in ['v', 'd', 'j'] 
                          if gene in self.alignair_likelihood_hit_function]

        for gene in available_genes:
            data = self.alignair_likelihood_hit_function[gene]
            
            # Main calibration line
            fig.add_trace(go.Scatter(
                x=data['likelihood'],
                y=data['proportion'],
                mode='lines+markers',
                name=f"{gene.upper()} Gene Calibration",
                line=dict(color=self.color_palette[gene], **self.line_styles[gene]),
                marker=dict(size=6, symbol='circle'),
                hovertemplate=f"{gene.upper()} Gene<br>" +
                            "Likelihood: %{x:.3f}<br>" +
                            "Agreement: %{y:.3f}<br>" +
                            "<extra></extra>",
                showlegend=True
            ), row=1, col=1)
            
            # Confidence interval bands
            fig.add_trace(go.Scatter(
                x=np.concatenate([data['likelihood'], data['likelihood'][::-1]]),
                y=np.concatenate([data['ci_upper'], data['ci_lower'][::-1]]),
                fill='toself',
                fillcolor=self.color_palette['confidence_band'],
                line=dict(color='rgba(255,255,255,0)'),
                name=f"{gene.upper()} 95% CI",
                showlegend=False,
                hoverinfo='skip'
            ), row=1, col=1)

        # Perfect calibration line (y=x)
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Perfect Calibration',
            line=dict(color=self.color_palette['identity'], **self.line_styles['identity']),
            hovertemplate="Perfect Calibration<br>x = y<extra></extra>",
            showlegend=True
        ), row=1, col=1)

        # Update main plot styling
        fig.update_xaxes(
            title_text="Average AlignAIR Likelihood", 
            row=1, col=1,
            range=[0, 1],
            tickformat='.2f',
            gridcolor='lightgray',
            gridwidth=1
        )
        fig.update_yaxes(
            title_text="Proportion of Correct Predictions", 
            row=1, col=1,
            range=[0, 1],
            tickformat='.2f',
            gridcolor='lightgray',
            gridwidth=1
        )

        # Enhanced performance vs mutation rate subplots
        genes_for_subplot = ['v', 'd', 'j']
        subplot_positions = [(2, 1), (2, 2), (2, 3)]

        for i, gene in enumerate(genes_for_subplot):
            row, col = subplot_positions[i]
            
            if gene not in self.thdfs or self.thdfs[gene]['thdf_agg'] is None:
                # Add placeholder if no data
                fig.add_annotation(
                    text=f"No {gene.upper()} gene data available",
                    x=0.5, y=0.5,
                    xref=f"x{i+2}", yref=f"y{i+2}",
                    showarrow=False,
                    font=dict(size=14, color="gray")
                )
                continue

            data = self.thdfs[gene]['thdf_agg']
            x_vals = data['mutation_rate']

            # Top-1 performance
            fig.add_trace(go.Scatter(
                x=x_vals, 
                y=data[1]['mean'],
                mode='lines+markers',
                name=f'Top-1 Hit Rate',
                line=dict(color='#1f77b4', width=2),
                marker=dict(size=5),
                showlegend=(i == 0),  # Only show legend for first subplot
                legendgroup="top1",
                hovertemplate=f"{gene.upper()} Top-1<br>" +
                            "Mutation Rate: %{x:.4f}<br>" +
                            "Hit Rate: %{y:.3f}<br>" +
                            "<extra></extra>"
            ), row=row, col=col)

            # Top-3 performance
            fig.add_trace(go.Scatter(
                x=x_vals,
                y=data[3]['mean'],
                mode='lines+markers',
                name='Top-3 Hit Rate',
                line=dict(color='#ff7f0e', width=2),
                marker=dict(size=5),
                showlegend=(i == 0),
                legendgroup="top3",
                hovertemplate=f"{gene.upper()} Top-3<br>" +
                            "Mutation Rate: %{x:.4f}<br>" +
                            "Hit Rate: %{y:.3f}<br>" +
                            "<extra></extra>"
            ), row=row, col=col)

            # Dynamic threshold performance
            fig.add_trace(go.Scatter(
                x=x_vals,
                y=data['dt_hits']['mean'],
                mode='lines+markers',
                name='Dynamic Threshold',
                line=dict(color='#2ca02c', width=2, dash='dot'),
                marker=dict(size=5),
                showlegend=(i == 0),
                legendgroup="dynamic",
                hovertemplate=f"{gene.upper()} Dynamic<br>" +
                            "Mutation Rate: %{x:.4f}<br>" +
                            "Hit Rate: %{y:.3f}<br>" +
                            "<extra></extra>"
            ), row=row, col=col)

            # Update subplot axes
            fig.update_xaxes(
                title_text="Mutation Rate",
                row=row, col=col,
                tickformat='.3f',
                gridcolor='lightgray',
                gridwidth=0.5
            )
            fig.update_yaxes(
                title_text="Hit Rate",
                row=row, col=col,
                range=[0, 1],
                tickformat='.2f',
                gridcolor='lightgray',
                gridwidth=0.5
            )

        # Enhanced layout
        fig.update_layout(
            height=1000, 
            width=1400,
            title=dict(
                text="Enhanced AlignAIR Likelihood Calibration Report",
                x=0.5,
                font=dict(size=20, family="Arial, sans-serif")
            ),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(size=12)
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family="Arial, sans-serif", size=12),
            hovermode="closest"
        )

        # Add annotations for better understanding
        fig.add_annotation(
            text="Points closer to the diagonal line indicate better calibration",
            x=0.5, y=-0.15,
            xref="x", yref="paper",
            showarrow=False,
            font=dict(size=11, color="gray", style="italic")
        )

        return fig

    def get_panel(self):
        """Backward compatibility - returns the enhanced panel."""
        return self.get_enhanced_panel()
