import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from collections import defaultdict










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

        # Bins for discretizing likelihood values
        self.likelihood_bins = np.linspace(0, 1, 25)

        # Populate groundtruth_table with v/d/j mutation rates
        self.update_ground_truth_mutation_rates()

        # Compute all additional metadata (likelihood hits, top-k hits, etc.)
        self.derive_metadata()

    # -------------------------------------------------------------------------
    # Public Methods
    # -------------------------------------------------------------------------
    def categorize_value(self, value):
        """
        Categorize a single value into the pre-defined likelihood bins using pandas.cut.
        """
        bin_category = pd.cut([value], bins=self.likelihood_bins, include_lowest=False)
        return bin_category[0]

    def update_ground_truth_mutation_rates(self):
        """
        Computes mutation rates for V, D, and J segments and adds them
        to the 'groundtruth_table' as new columns: 'v_mutation_rate',
        'd_mutation_rate', and 'j_mutation_rate'.
        """
        v_mutation_rate = []
        d_mutation_rate = []
        j_mutation_rate = []

        # Calculate mutation rates per row
        for _, row in tqdm(self.groundtruth_table.iterrows(),
                           total=len(self.groundtruth_table),
                           desc='Calculating mutation rates'):

            # Extract segment lengths
            v_length = row['v_sequence_end'] - row['v_sequence_start']
            d_length = None
            if 'd_call' in row:
                d_length = row['d_sequence_end'] - row['d_sequence_start']
            j_length = row['j_sequence_end'] - row['j_sequence_start']

            # Avoid division by zero
            v_length = v_length if v_length > 0 else 1
            if d_length is not None:
                d_length = d_length if d_length > 0 else 1
            j_length = j_length if j_length > 0 else 1

            # Gather positions for mutations
            mutation_positions = list(eval(row['mutations']).keys())

            # Count how many fall into each segment
            v_count = sum(1 for pos in mutation_positions
                          if row['v_sequence_start'] <= pos <= row['v_sequence_end'])
            if d_length is not None:
                d_count = sum(1 for pos in mutation_positions
                              if row['d_sequence_start'] <= pos <= row['d_sequence_end'])
            j_count = sum(1 for pos in mutation_positions
                          if row['j_sequence_start'] <= pos <= row['j_sequence_end'])

            v_mutation_rate.append(v_count / v_length)
            if d_length is not None:
                d_mutation_rate.append(d_count / d_length)
            j_mutation_rate.append(j_count / j_length)

        # Save to the groundtruth table
        self.groundtruth_table['v_mutation_rate'] = v_mutation_rate
        # If the table always has 'd_call', fill it; otherwise handle length mismatch
        if len(d_mutation_rate) == len(self.groundtruth_table):
            self.groundtruth_table['d_mutation_rate'] = d_mutation_rate
        else:
            # Fallback if 'd_call' not always present
            self.groundtruth_table['d_mutation_rate'] = np.nan

        self.groundtruth_table['j_mutation_rate'] = j_mutation_rate

    def derive_metadata(self):
        """
        Performs two major analyses:

        1. AlignAIR likelihood hits (storing in `self.alignair_likelihood_hit_function`).
        2. Top-k hits for the 'v' call (storing in `self.thdf` & `self.thdf_agg`).
        """
        # Part A: AlignAIR likelihood hits
        self.alignair_likelihood_hit_function = self._compute_alignair_likelihood_hits()

        # Part B: Top-k hits for the 'v' gene
        self._compute_top_k_hits()

    # -------------------------------------------------------------------------
    # Private Helper Methods
    # -------------------------------------------------------------------------
    def _compute_alignair_likelihood_hits(self):
        """
        Computes how often the 'predicted' allele calls (likelihoods) match
        the groundtruth calls. Groups them into bins and aggregates hits.
        Returns a dictionary keyed by gene ('v', 'd', 'j').
        """
        alignair_likelihood_hit_function = {}
        alignair_likelihood_hit_total = {}

        # Check each gene: v, d, j
        for gene in tqdm(['v', 'd', 'j'], desc='AlignAIR Likelihood Hits'):
            # Only process if gene_allele predictions exist
            if f'{gene}_allele' not in self.predict_object.processed_predictions:
                continue

            # Round to 2 decimals
            iterator = self.predict_object.processed_predictions[f'{gene}_allele'].round(2)

            # Initialize counters
            alignair_likelihood_hit_function[gene] = defaultdict(int)
            alignair_likelihood_hit_total[gene] = defaultdict(int)

            # Groundtruth calls for this gene
            groundtruth_calls = self.groundtruth_table[f'{gene}_call']
            reverse_mapping = self.predict_object.threshold_extractor_instances[gene].properties_map[gene.upper()]['reverse_mapping']

            # Loop over each row
            for likelihoods_row, groundtruth_call in tqdm(
                    zip(iterator, groundtruth_calls),
                    total=len(iterator),
                    desc=f'Gene: {gene}'):

                # Convert groundtruth call to a set
                groundtruth_decomposed = set(groundtruth_call.split(','))
                # For each index in likelihoods_row
                for idx, likelihood in enumerate(likelihoods_row):
                    # True if predicted allele is in the groundtruth set

                    is_hit = (reverse_mapping[idx]
                              in groundtruth_decomposed)

                    alignair_likelihood_hit_function[gene][likelihood] += int(is_hit)
                    alignair_likelihood_hit_total[gene][likelihood] += 1

        # Transform raw data into bins
        for gene, hits_dict in alignair_likelihood_hit_function.items():
            # Convert hits dict -> Series -> DataFrame
            hits_df = pd.Series(hits_dict, name='hits').sort_index().to_frame()
            total_df = pd.Series(alignair_likelihood_hit_total[gene], name='counts').sort_index().to_frame()

            # Merge them
            merged_df = hits_df.join(total_df, how='left')
            merged_df = merged_df.reset_index().rename(columns={'index': 'likelihood'})

            # Bin by 'likelihood'
            merged_df['bins'] = pd.cut(merged_df['likelihood'], 25, include_lowest=False)

            # Group by bins
            grouped = merged_df.groupby('bins').agg({
                'hits': 'sum',
                'counts': 'sum',
                'likelihood': 'mean'
            })
            # Compute fraction
            grouped['hits'] = (grouped['hits'] / grouped['counts']).round(5)

            alignair_likelihood_hit_function[gene] = grouped

        return alignair_likelihood_hit_function

    @staticmethod
    def get_agreement(predicted_labels, ground_truth):
        return [len(set(i) & set(j)) > 0 for i, j in zip(predicted_labels, ground_truth)]

    def _compute_top_k_hits(self):
        """
        Computes top-k hits for the 'v' gene. For k in [1..3],
        determines if at least one of the predicted alleles appears
        in the groundtruth set. Then bins by mutation rate.
        Returns (thdf, thdf_agg).
        """

        self.thdfs = {'v': {'thdf': None, 'thdf_agg': None},
                      'd': {'thdf': None, 'thdf_agg': None},
                      'j': {'thdf': None, 'thdf_agg': None}
                      }

        # For k=1..3
        allele_set = ['v', 'j']
        if 'd_call' in self.groundtruth_table.columns:
            allele_set.append('d')

        for gene in allele_set:
            ar_top_k_hits = {}
            v_gt = self.groundtruth_table[f'{gene}_call'].apply(lambda x: set(x.split(',')))

            for k in tqdm(range(1, 4), desc='Top-k Hits'):
                # Take top-k predicted alleles for each row
                predicted_top_k = (set(alleles[:k]) for alleles in self.predict_object.selected_allele_calls[gene])
                # Compare with groundtruth
                hits = self.get_agreement(predicted_top_k, v_gt)
                ar_top_k_hits[k] = hits

            # Convert to DataFrame
            thdf = pd.DataFrame(ar_top_k_hits)
            thdf['dt_hits'] = self.get_agreement(self.predict_object.selected_allele_calls[gene], v_gt)

            # Add groundtruth mutation rate
            thdf['mutation_rate'] = self.groundtruth_table[f'{gene}_mutation_rate']

            # Bin by mutation rate (0..0.25 in 25 steps)
            rate_bins = np.linspace(0, 0.25, 25)
            thdf['bins'] = pd.cut(thdf['mutation_rate'], bins=rate_bins)

            # Aggregate: average success rate by bin
            thdf_agg = thdf.groupby('bins').mean()

            self.thdfs[gene]['thdf'] = thdf
            self.thdfs[gene]['thdf_agg'] = thdf_agg

    def get_enhanced_panel(self):
        """
        Enhanced version with improved visual clarity, better colors, and confidence intervals.
        """
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import pandas as pd
        import numpy as np

        # Enhanced color palette
        color_palette = {
            'v': '#1f77b4',  # Blue
            'd': '#ff7f0e',  # Orange  
            'j': '#2ca02c',  # Green
            'identity': '#d62728',  # Red for identity line
            'confidence_band': 'rgba(128, 128, 128, 0.2)'  # Gray with transparency
        }

        # Create enhanced subplot layout
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=(
                'Enhanced Likelihood Calibration Analysis', 
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

        # Enhanced likelihood calibration plot with confidence intervals
        available_genes = [gene for gene in ['v', 'd', 'j'] 
                          if gene in self.alignair_likelihood_hit_function]

        for gene in available_genes:
            data = self.alignair_likelihood_hit_function[gene]
            
            # Calculate confidence intervals for better uncertainty visualization
            hits_proportion = data['hits']
            n_samples = len(self.groundtruth_table)
            std_error = np.sqrt(hits_proportion * (1 - hits_proportion) / n_samples)
            ci_lower = np.maximum(0, hits_proportion - 1.96 * std_error)
            ci_upper = np.minimum(1, hits_proportion + 1.96 * std_error)
            
            # Main calibration line with enhanced styling
            fig.add_trace(go.Scatter(
                x=data['likelihood'],
                y=hits_proportion,
                mode='lines+markers',
                name=f"{gene.upper()} Gene Calibration",
                line=dict(color=color_palette[gene], width=3),
                marker=dict(size=8, symbol='circle', line=dict(width=1, color='white')),
                hovertemplate=f"{gene.upper()} Gene<br>" +
                            "Likelihood: %{x:.3f}<br>" +
                            "Agreement: %{y:.3f}<br>" +
                            "Samples: " + str(n_samples) + "<br>" +
                            "<extra></extra>",
                showlegend=True
            ), row=1, col=1)
            
            # Confidence interval bands
            fig.add_trace(go.Scatter(
                x=np.concatenate([data['likelihood'], data['likelihood'][::-1]]),
                y=np.concatenate([ci_upper, ci_lower[::-1]]),
                fill='toself',
                fillcolor=color_palette['confidence_band'],
                line=dict(color='rgba(255,255,255,0)'),
                name=f"{gene.upper()} 95% CI",
                showlegend=False,
                hoverinfo='skip'
            ), row=1, col=1)

        # Perfect calibration line (y=x) with enhanced styling
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Perfect Calibration (y=x)',
            line=dict(color=color_palette['identity'], width=2, dash='dash'),
            hovertemplate="Perfect Calibration<br>y = x<extra></extra>",
            showlegend=True
        ), row=1, col=1)

        # Update main plot styling with grid and better formatting
        fig.update_xaxes(
            title_text="Average AlignAIR Likelihood", 
            row=1, col=1,
            range=[0, 1],
            tickformat='.2f',
            gridcolor='lightgray',
            gridwidth=1,
            showgrid=True
        )
        fig.update_yaxes(
            title_text="Proportion of Correct Predictions", 
            row=1, col=1,
            range=[0, 1],
            tickformat='.2f',
            gridcolor='lightgray',
            gridwidth=1,
            showgrid=True
        )

        # Enhanced performance vs mutation rate subplots
        genes_for_subplot = ['v', 'd', 'j']
        subplot_positions = [(2, 1), (2, 2), (2, 3)]
        colors_top_k = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green

        for i, gene in enumerate(genes_for_subplot):
            row, col = subplot_positions[i]
            
            if gene not in self.thdfs or self.thdfs[gene]['thdf_agg'] is None:
                # Add informative placeholder if no data
                fig.add_annotation(
                    text=f"No {gene.upper()} gene data available",
                    x=0.5, y=0.5,
                    xref=f"x{i+2 if i > 0 else 2}", yref=f"y{i+2 if i > 0 else 2}",
                    showarrow=False,
                    font=dict(size=14, color="gray")
                )
                continue

            data = self.thdfs[gene]['thdf_agg']
            x_vals = data['mutation_rate']

            # Enhanced styling for each line
            line_configs = [
                {'y': data[1], 'name': 'Top-1 Hit Rate', 'color': colors_top_k[0], 'dash': None},
                {'y': data[3], 'name': 'Top-3 Hit Rate', 'color': colors_top_k[1], 'dash': None},
                {'y': data['dt_hits'], 'name': 'Dynamic Threshold', 'color': colors_top_k[2], 'dash': 'dot'}
            ]

            for line_config in line_configs:
                fig.add_trace(go.Scatter(
                    x=x_vals,
                    y=line_config['y'],
                    mode='lines+markers',
                    name=line_config['name'],
                    line=dict(color=line_config['color'], width=3, dash=line_config['dash']),
                    marker=dict(size=6),
                    showlegend=(i == 0),  # Only show legend for first subplot
                    legendgroup=line_config['name'].lower().replace(' ', '_').replace('-', '_'),
                    hovertemplate=f"{gene.upper()} {line_config['name']}<br>" +
                                "Mutation Rate: %{x:.4f}<br>" +
                                "Hit Rate: %{y:.3f}<br>" +
                                "<extra></extra>"
                ), row=row, col=col)

            # Update subplot axes with better formatting
            fig.update_xaxes(
                title_text="Mutation Rate",
                row=row, col=col,
                tickformat='.3f',
                gridcolor='lightgray',
                gridwidth=0.5,
                showgrid=True
            )
            fig.update_yaxes(
                title_text="Hit Rate",
                row=row, col=col,
                range=[0, 1],
                tickformat='.2f',
                gridcolor='lightgray',
                gridwidth=0.5,
                showgrid=True
            )

        # Enhanced layout with better typography and spacing
        fig.update_layout(
            height=1200,  # Increased height
            width=None,  # Let it be responsive
            title=dict(
                text="Enhanced AlignAIR Likelihood Calibration Report",
                x=0.5,
                font=dict(size=20, family="Arial, sans-serif", color="black")
            ),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(size=12),
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="lightgray",
                borderwidth=1
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family="Arial, sans-serif", size=12, color="black"),
            hovermode="closest",
            autosize=True
        )

        # Add informative annotations
        fig.add_annotation(
            text="Points closer to the diagonal line indicate better calibration. Confidence bands show uncertainty.",
            x=0.5, y=-0.15,
            xref="x", yref="paper",
            showarrow=False,
            font=dict(size=11, color="gray"),
            align="center"
        )

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

    def get_original_panel(self):
        """Original version of the panel for backward compatibility."""
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import pandas as pd
        import numpy as np

        # Create subplots grid
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=('Likelihood Calibration', 'A', "B", 'C'),
            row_heights=[0.5, 0.5],
            column_widths=[0.3, 0.3, 0.3],
            specs=[
                [{"type": "xy", 'colspan': 3}, None, None],
                [{"type": "xy"}, {"type": "xy"}, {"type": "xy"}],
            ]
        )

        # Add AlignAIR Likelihood Function
        for allele, color in zip(['v', 'd', 'j'], ['blue', 'red', 'green']):
            if allele in self.alignair_likelihood_hit_function:
                fig.add_trace(go.Scatter(
                    x=self.alignair_likelihood_hit_function[allele]['likelihood'],
                    y=self.alignair_likelihood_hit_function[allele]['hits'],
                    mode='lines',
                    name=f"{allele.upper()} Allele Calibration",
                    line=dict(color=color)
                ), row=1, col=1)

        # draw y=x line
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Identity Line',
            line=dict(color='black', dash='dash')
        ), row=1, col=1)
        
        # xlabel
        fig.update_xaxes(title_text="Average AlignAIR Likelihood", row=1, col=1)
        fig.update_yaxes(title_text="Average Agreement", row=1, col=1)
        # cursor type xy
        fig.update_layout(hovermode="x unified")

        # Hit Comparison
        panels = ['V', 'D', 'J']
        for i, panel in enumerate(panels):
            g = panel.lower()
            if g in self.thdfs and self.thdfs[g]['thdf_agg'] is not None:
                fig.add_trace(go.Scatter(
                    x=self.thdfs[g]['thdf_agg']['mutation_rate'], 
                    y=self.thdfs[g]['thdf_agg'][1],
                    mode='lines+markers', name='AlignAIR: Top 1',
                    line=dict(color='blue')
                ), row=2, col=i + 1)
                fig.add_trace(go.Scatter(
                    x=self.thdfs[g]['thdf_agg']['mutation_rate'], 
                    y=self.thdfs[g]['thdf_agg'][3],
                    mode='lines+markers', name='AlignAIR: Top 3',
                    line=dict(color='red')
                ), row=2, col=i + 1)
                fig.add_trace(go.Scatter(
                    x=self.thdfs[g]['thdf_agg']['mutation_rate'],
                    y=self.thdfs[g]['thdf_agg']['dt_hits'],
                    mode='lines+markers', name='AlignAIR: Dynamic Threshold',
                    line=dict(color='green', dash='dot')
                ), row=2, col=i + 1)

        # Update layout
        fig.update_layout(
            height=1000, width=1200, showlegend=True,
            title_text="AlignAIR Likelihood Report"
        )

        return fig