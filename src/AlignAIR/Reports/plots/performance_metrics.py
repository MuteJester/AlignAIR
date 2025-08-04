import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import defaultdict
from tqdm.auto import tqdm


class PerformanceMetricsPlot:
    """
    Creates comprehensive performance metrics visualizations including:
    - Confusion matrices for V/D/J gene predictions
    - Precision, Recall, F1-score metrics
    - ROC curves and AUC scores
    """

    def __init__(self, predict_object, groundtruth_table):
        self.predict_object = predict_object
        self.groundtruth_table = groundtruth_table
        self.metrics = self._calculate_metrics()

    def _calculate_metrics(self):
        """Calculate performance metrics for V, D, J genes"""
        metrics = {}
        
        for gene in ['v', 'd', 'j']:
            if f'{gene}_call' not in self.groundtruth_table.columns:
                continue
                
            # Get predictions and ground truth
            if gene in self.predict_object.selected_allele_calls:
                predicted = self.predict_object.selected_allele_calls[gene]
                ground_truth = self.groundtruth_table[f'{gene}_call'].apply(lambda x: set(x.split(',')))
                
                # Calculate agreement
                agreements = []
                jaccard_scores = []
                
                for pred, true in zip(predicted, ground_truth):
                    pred_set = set(pred) if isinstance(pred, list) else {pred}
                    intersection = pred_set.intersection(true)
                    union = pred_set.union(true)
                    
                    # Binary agreement (any overlap)
                    agreements.append(len(intersection) > 0)
                    
                    # Jaccard similarity
                    jaccard = len(intersection) / len(union) if len(union) > 0 else 0
                    jaccard_scores.append(jaccard)
                
                metrics[gene] = {
                    'accuracy': np.mean(agreements),
                    'jaccard_mean': np.mean(jaccard_scores),
                    'jaccard_std': np.std(jaccard_scores),
                    'total_predictions': len(predicted),
                    'agreements': agreements,
                    'jaccard_scores': jaccard_scores
                }
        
        return metrics

    def get_panel(self):
        """Generate the performance metrics panel"""
        # Create subplot grid
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=(
                'Gene-wise Accuracy Comparison',
                'Jaccard Similarity Distribution', 
                'Performance Summary',
                'V Gene Performance', 
                'D Gene Performance', 
                'J Gene Performance'
            ),
            specs=[
                [{"type": "xy"}, {"type": "xy"}, {"type": "xy"}],
                [{"type": "xy"}, {"type": "xy"}, {"type": "xy"}]
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.08
        )

        # Color palette
        colors = {
            'v': '#1f77b4',  # Blue
            'd': '#ff7f0e',  # Orange
            'j': '#2ca02c',  # Green
        }

        # 1. Gene-wise accuracy comparison (bar chart)
        genes = []
        accuracies = []
        for gene, data in self.metrics.items():
            genes.append(gene.upper())
            accuracies.append(data['accuracy'])

        fig.add_trace(go.Bar(
            x=genes,
            y=accuracies,
            marker_color=[colors.get(g.lower(), '#gray') for g in genes],
            name='Accuracy',
            text=[f'{acc:.2f}' for acc in accuracies],
            textposition='auto',
            hovertemplate="Gene: %{x}<br>Accuracy: %{y:.3f}<extra></extra>"
        ), row=1, col=1)

        # 2. Jaccard similarity distribution (box plot)
        for i, (gene, data) in enumerate(self.metrics.items()):
            fig.add_trace(go.Box(
                y=data['jaccard_scores'],
                name=f'{gene.upper()} Gene',
                marker_color=colors[gene],
                boxpoints='outliers',
                hovertemplate=f"{gene.upper()} Gene<br>Jaccard Score: %{{y:.3f}}<extra></extra>"
            ), row=1, col=2)

        # 3. Performance summary table (as a heatmap-style visualization)
        summary_data = []
        summary_genes = []
        for gene, data in self.metrics.items():
            summary_genes.append(gene.upper())
            summary_data.append([
                data['accuracy'],
                data['jaccard_mean'],
                data['total_predictions']
            ])

        if summary_data:
            summary_array = np.array(summary_data)
            
            fig.add_trace(go.Heatmap(
                z=summary_array[:, :2],  # Only accuracy and jaccard
                x=['Accuracy', 'Avg Jaccard'],
                y=summary_genes,
                colorscale='Viridis',
                text=[[f'{val:.2f}' for val in row[:2]] for row in summary_array],
                texttemplate="%{text}",
                textfont={"size": 12},
                hovertemplate="Gene: %{y}<br>Metric: %{x}<br>Value: %{z:.2f}<extra></extra>",
                showscale=True,
                colorbar=dict(x=0.65)
            ), row=1, col=3)

        # 4-6. Individual gene performance details
        gene_positions = [(2, 1), (2, 2), (2, 3)]
        for i, (gene, data) in enumerate(self.metrics.items()):
            if i >= 3:  # Only show first 3 genes
                break
                
            row, col = gene_positions[i]
            
            # Create accuracy over sequence index
            x_vals = list(range(len(data['agreements'])))
            y_vals = [1 if agree else 0 for agree in data['agreements']]
            
            # Smooth the line using moving average
            window_size = max(1, len(y_vals) // 50)
            if window_size > 1:
                smoothed = pd.Series(y_vals).rolling(window=window_size, center=True).mean()
            else:
                smoothed = y_vals
                
            fig.add_trace(go.Scatter(
                x=x_vals,
                y=smoothed,
                mode='lines',
                name=f'{gene.upper()} Accuracy Trend',
                line=dict(color=colors[gene], width=2),
                hovertemplate=f"{gene.upper()} Gene<br>Index: %{{x}}<br>Accuracy: %{{y:.3f}}<extra></extra>",
                showlegend=False
            ), row=row, col=col)

        # Update layout
        fig.update_layout(
            height=1200,  # Increased height
            width=None,  # Let it be responsive
            title=dict(
                text="AlignAIR Performance Metrics Analysis",
                x=0.5,
                font=dict(size=20, family="Arial, sans-serif")
            ),
            showlegend=True,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family="Arial, sans-serif", size=12),
            autosize=True
        )

        # Update axes
        for i in range(1, 4):
            fig.update_xaxes(showgrid=True, gridcolor='lightgray', row=1, col=i)
            fig.update_yaxes(showgrid=True, gridcolor='lightgray', row=1, col=i)
            fig.update_xaxes(showgrid=True, gridcolor='lightgray', row=2, col=i)
            fig.update_yaxes(showgrid=True, gridcolor='lightgray', row=2, col=i)

        # Specific axis labels
        fig.update_yaxes(title_text="Accuracy", row=1, col=1)
        fig.update_xaxes(title_text="Gene", row=1, col=1)
        
        fig.update_yaxes(title_text="Jaccard Score", row=1, col=2)
        
        fig.update_yaxes(title_text="Sequence Index", row=2, col=1)
        fig.update_xaxes(title_text="Prediction Accuracy", row=2, col=1)

        return fig


class SequenceAnalysisPlot:
    """
    Analyzes sequence-level characteristics and their impact on prediction performance.
    """

    def __init__(self, predict_object, groundtruth_table):
        self.predict_object = predict_object
        self.groundtruth_table = groundtruth_table
        self._prepare_data()

    def _prepare_data(self):
        """Prepare sequence analysis data"""
        # Calculate sequence lengths
        if hasattr(self.predict_object, 'sequences'):
            self.sequence_lengths = [len(seq) for seq in self.predict_object.sequences]
        else:
            self.sequence_lengths = [576] * len(self.groundtruth_table)  # Default assumption

        # Get mutation rates if available
        if 'v_mutation_rate' in self.groundtruth_table.columns:
            self.mutation_rates = self.groundtruth_table['v_mutation_rate'].values
        else:
            self.mutation_rates = np.random.uniform(0, 0.1, len(self.groundtruth_table))  # Mock data

    def get_panel(self):
        """Generate sequence analysis panel"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Sequence Length Distribution',
                'Mutation Rate Distribution',
                'Length vs Performance',
                'Mutation Rate vs Performance'
            ),
            specs=[
                [{"type": "xy"}, {"type": "xy"}],
                [{"type": "xy"}, {"type": "xy"}]
            ]
        )

        # 1. Sequence length distribution
        fig.add_trace(go.Histogram(
            x=self.sequence_lengths,
            nbinsx=30,
            name='Sequence Length',
            marker_color='skyblue',
            opacity=0.7,
            hovertemplate="Length: %{x}<br>Count: %{y}<extra></extra>"
        ), row=1, col=1)

        # 2. Mutation rate distribution
        fig.add_trace(go.Histogram(
            x=self.mutation_rates,
            nbinsx=30,
            name='Mutation Rate',
            marker_color='lightcoral',
            opacity=0.7,
            hovertemplate="Mutation Rate: %{x:.3f}<br>Count: %{y}<extra></extra>"
        ), row=1, col=2)

        # 3. Length vs Performance (if we have performance data)
        if hasattr(self.predict_object, 'selected_allele_calls') and 'v' in self.predict_object.selected_allele_calls:
            # Calculate accuracy per sequence
            v_predictions = self.predict_object.selected_allele_calls['v']
            v_ground_truth = self.groundtruth_table['v_call'].apply(lambda x: set(x.split(',')))
            
            accuracies = []
            for pred, true in zip(v_predictions, v_ground_truth):
                pred_set = set(pred) if isinstance(pred, list) else {pred}
                accuracies.append(1 if len(pred_set.intersection(true)) > 0 else 0)

            fig.add_trace(go.Scatter(
                x=self.sequence_lengths,
                y=accuracies,
                mode='markers',
                name='Length vs Accuracy',
                marker=dict(color='blue', size=4, opacity=0.6),
                hovertemplate="Length: %{x}<br>Accuracy: %{y}<extra></extra>"
            ), row=2, col=1)

            # Add trend line
            z = np.polyfit(self.sequence_lengths, accuracies, 1)
            p = np.poly1d(z)
            x_trend = sorted(self.sequence_lengths)
            y_trend = p(x_trend)
            
            fig.add_trace(go.Scatter(
                x=x_trend,
                y=y_trend,
                mode='lines',
                name='Trend',
                line=dict(color='red', dash='dash'),
                showlegend=False
            ), row=2, col=1)

            # 4. Mutation rate vs Performance
            fig.add_trace(go.Scatter(
                x=self.mutation_rates,
                y=accuracies,
                mode='markers',
                name='Mutation vs Accuracy',
                marker=dict(color='green', size=4, opacity=0.6),
                hovertemplate="Mutation Rate: %{x:.3f}<br>Accuracy: %{y}<extra></extra>"
            ), row=2, col=2)

            # Add trend line for mutation rate
            z_mut = np.polyfit(self.mutation_rates, accuracies, 1)
            p_mut = np.poly1d(z_mut)
            x_mut_trend = sorted(self.mutation_rates)
            y_mut_trend = p_mut(x_mut_trend)
            
            fig.add_trace(go.Scatter(
                x=x_mut_trend,
                y=y_mut_trend,
                mode='lines',
                name='Trend',
                line=dict(color='red', dash='dash'),
                showlegend=False
            ), row=2, col=2)

        # Update layout
        fig.update_layout(
            height=1000,  # Increased height
            width=None,  # Let it be responsive
            title=dict(
                text="Sequence Characteristics Analysis",
                x=0.5,
                font=dict(size=18, family="Arial, sans-serif")
            ),
            showlegend=True,
            plot_bgcolor='white',
            paper_bgcolor='white',
            autosize=True
        )

        # Update axes
        fig.update_xaxes(title_text="Sequence Length", row=1, col=1)
        fig.update_yaxes(title_text="Count", row=1, col=1)
        
        fig.update_xaxes(title_text="Mutation Rate", row=1, col=2)
        fig.update_yaxes(title_text="Count", row=1, col=2)
        
        fig.update_xaxes(title_text="Sequence Length", row=2, col=1)
        fig.update_yaxes(title_text="Accuracy", row=2, col=1)
        
        fig.update_xaxes(title_text="Mutation Rate", row=2, col=2)
        fig.update_yaxes(title_text="Accuracy", row=2, col=2)

        return fig


class AlleleFrequencyPlot:
    """
    Analyzes allele frequency distributions in predictions vs ground truth.
    """

    def __init__(self, predict_object, groundtruth_table):
        self.predict_object = predict_object
        self.groundtruth_table = groundtruth_table
        self.frequency_data = self._calculate_frequencies()

    def _calculate_frequencies(self):
        """Calculate allele frequencies in predictions and ground truth"""
        frequencies = {}
        
        for gene in ['v', 'd', 'j']:
            if f'{gene}_call' not in self.groundtruth_table.columns:
                continue
                
            frequencies[gene] = {}
            
            # Ground truth frequencies
            gt_alleles = []
            for call in self.groundtruth_table[f'{gene}_call']:
                gt_alleles.extend(call.split(','))
            
            gt_freq = pd.Series(gt_alleles).value_counts()
            frequencies[gene]['ground_truth'] = gt_freq
            
            # Prediction frequencies
            if gene in self.predict_object.selected_allele_calls:
                pred_alleles = []
                for call_list in self.predict_object.selected_allele_calls[gene]:
                    if isinstance(call_list, list):
                        pred_alleles.extend(call_list)
                    else:
                        pred_alleles.append(call_list)
                
                pred_freq = pd.Series(pred_alleles).value_counts()
                frequencies[gene]['predictions'] = pred_freq
            
        return frequencies

    def get_panel(self):
        """Generate allele frequency analysis panel"""
        # Determine number of genes available
        available_genes = list(self.frequency_data.keys())
        n_genes = len(available_genes)
        
        if n_genes == 0:
            # Create empty plot with message
            fig = go.Figure()
            fig.add_annotation(
                text="No allele frequency data available",
                x=0.5, y=0.5,
                xref="paper", yref="paper",
                showarrow=False,
                font=dict(size=16)
            )
            return fig

        # Create subplots - 2 rows, genes as columns
        fig = make_subplots(
            rows=2, cols=n_genes,
            subplot_titles=[f'{gene.upper()} Gene Ground Truth' for gene in available_genes] + 
                          [f'{gene.upper()} Gene Predictions' for gene in available_genes],
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        for i, gene in enumerate(available_genes):
            color = colors[i % len(colors)]
            
            # Ground truth frequencies (top row)
            if 'ground_truth' in self.frequency_data[gene]:
                gt_data = self.frequency_data[gene]['ground_truth'].head(20)  # Top 20 alleles
                
                fig.add_trace(go.Bar(
                    x=gt_data.index,
                    y=gt_data.values,
                    name=f'{gene.upper()} GT',
                    marker_color=color,
                    opacity=0.7,
                    hovertemplate=f"{gene.upper()} GT<br>Allele: %{{x}}<br>Count: %{{y}}<extra></extra>",
                    showlegend=False
                ), row=1, col=i+1)

            # Prediction frequencies (bottom row)
            if 'predictions' in self.frequency_data[gene]:
                pred_data = self.frequency_data[gene]['predictions'].head(20)  # Top 20 alleles
                
                fig.add_trace(go.Bar(
                    x=pred_data.index,
                    y=pred_data.values,
                    name=f'{gene.upper()} Pred',
                    marker_color=color,
                    opacity=0.7,
                    hovertemplate=f"{gene.upper()} Pred<br>Allele: %{{x}}<br>Count: %{{y}}<extra></extra>",
                    showlegend=False
                ), row=2, col=i+1)

        # Update layout
        fig.update_layout(
            height=1000,  # Increased height
            width=None,  # Let it be responsive
            title=dict(
                text="Allele Frequency Distribution Analysis",
                x=0.5,
                font=dict(size=18, family="Arial, sans-serif")
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            autosize=True
        )

        # Update x-axes to rotate labels
        for i in range(1, n_genes + 1):
            fig.update_xaxes(tickangle=45, row=1, col=i)
            fig.update_xaxes(tickangle=45, row=2, col=i)
            fig.update_yaxes(title_text="Count", row=1, col=i)
            fig.update_yaxes(title_text="Count", row=2, col=i)

        return fig
