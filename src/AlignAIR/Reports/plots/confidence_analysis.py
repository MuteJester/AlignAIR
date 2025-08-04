import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import defaultdict


class ConfidenceAnalysisPlot:
    """
    Analyzes confidence scores and threshold effects on predictions.
    """

    def __init__(self, predict_object, groundtruth_table):
        self.predict_object = predict_object
        self.groundtruth_table = groundtruth_table
        self._prepare_confidence_data()

    def _prepare_confidence_data(self):
        """Prepare confidence and likelihood data for analysis"""
        self.confidence_data = {}
        
        # Extract likelihood data if available
        if hasattr(self.predict_object, 'processed_predictions'):
            for gene in ['v', 'd', 'j']:
                if f'{gene}_allele' in self.predict_object.processed_predictions:
                    likelihoods = self.predict_object.processed_predictions[f'{gene}_allele']
                    
                    # Calculate max likelihood (confidence) for each prediction
                    max_likelihoods = np.max(likelihoods, axis=1)
                    mean_likelihoods = np.mean(likelihoods, axis=1)
                    std_likelihoods = np.std(likelihoods, axis=1)
                    
                    # Calculate entropy (uncertainty measure)
                    # Add small epsilon to avoid log(0)
                    epsilon = 1e-10
                    normalized_likelihoods = likelihoods + epsilon
                    normalized_likelihoods = normalized_likelihoods / np.sum(normalized_likelihoods, axis=1, keepdims=True)
                    entropy = -np.sum(normalized_likelihoods * np.log(normalized_likelihoods), axis=1)
                    
                    self.confidence_data[gene] = {
                        'max_likelihood': max_likelihoods,
                        'mean_likelihood': mean_likelihoods,
                        'std_likelihood': std_likelihoods,
                        'entropy': entropy,
                        'raw_likelihoods': likelihoods
                    }

    def get_panel(self):
        """Generate confidence analysis panel"""
        available_genes = list(self.confidence_data.keys())
        
        if not available_genes:
            fig = go.Figure()
            fig.add_annotation(
                text="No confidence data available",
                x=0.5, y=0.5,
                xref="paper", yref="paper",
                showarrow=False,
                font=dict(size=16)
            )
            return fig

        # Create subplot grid
        fig = make_subplots(
            rows=3, cols=len(available_genes),
            subplot_titles=[f'{gene.upper()} Max Likelihood' for gene in available_genes] +
                          [f'{gene.upper()} Entropy Distribution' for gene in available_genes] +
                          [f'{gene.upper()} Likelihood vs Entropy' for gene in available_genes],
            vertical_spacing=0.12,
            horizontal_spacing=0.08
        )

        colors = {
            'v': '#1f77b4',  # Blue
            'd': '#ff7f0e',  # Orange
            'j': '#2ca02c',  # Green
        }

        for i, gene in enumerate(available_genes):
            col = i + 1
            color = colors.get(gene, '#gray')
            data = self.confidence_data[gene]

            # Row 1: Max likelihood distribution
            fig.add_trace(go.Histogram(
                x=data['max_likelihood'],
                nbinsx=30,
                name=f'{gene.upper()} Max Likelihood',
                marker_color=color,
                opacity=0.7,
                hovertemplate=f"{gene.upper()}<br>Max Likelihood: %{{x:.3f}}<br>Count: %{{y}}<extra></extra>",
                showlegend=False
            ), row=1, col=col)

            # Row 2: Entropy distribution
            fig.add_trace(go.Histogram(
                x=data['entropy'],
                nbinsx=30,
                name=f'{gene.upper()} Entropy',
                marker_color=color,
                opacity=0.7,
                hovertemplate=f"{gene.upper()}<br>Entropy: %{{x:.3f}}<br>Count: %{{y}}<extra></extra>",
                showlegend=False
            ), row=2, col=col)

            # Row 3: Scatter plot of max likelihood vs entropy
            fig.add_trace(go.Scatter(
                x=data['max_likelihood'],
                y=data['entropy'],
                mode='markers',
                name=f'{gene.upper()} Confidence vs Uncertainty',
                marker=dict(
                    color=data['std_likelihood'],
                    colorscale='Viridis',
                    size=4,
                    opacity=0.6,
                    colorbar=dict(
                        title=f"{gene.upper()} Std Dev",
                        x=0.95 if col == len(available_genes) else None,
                        y=0.25
                    ) if col == len(available_genes) else None,
                    showscale=col == len(available_genes)
                ),
                hovertemplate=f"{gene.upper()}<br>Max Likelihood: %{{x:.3f}}<br>Entropy: %{{y:.3f}}<br>Std Dev: %{{marker.color:.3f}}<extra></extra>",
                showlegend=False
            ), row=3, col=col)

        # Update layout
        fig.update_layout(
            height=1400,  # Increased height for 3-row layout
            width=None,  # Let it be responsive
            title=dict(
                text="Confidence and Uncertainty Analysis",
                x=0.5,
                font=dict(size=20, family="Arial, sans-serif")
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family="Arial, sans-serif", size=12),
            autosize=True
        )

        # Update axes labels
        for i in range(1, len(available_genes) + 1):
            fig.update_xaxes(title_text="Max Likelihood", row=1, col=i)
            fig.update_yaxes(title_text="Count", row=1, col=i)
            
            fig.update_xaxes(title_text="Entropy", row=2, col=i)
            fig.update_yaxes(title_text="Count", row=2, col=i)
            
            fig.update_xaxes(title_text="Max Likelihood", row=3, col=i)
            fig.update_yaxes(title_text="Entropy", row=3, col=i)

        return fig


class ThresholdEffectsPlot:
    """
    Analyzes the effects of different threshold settings on prediction performance.
    """

    def __init__(self, predict_object, groundtruth_table):
        self.predict_object = predict_object
        self.groundtruth_table = groundtruth_table
        self._analyze_threshold_effects()

    def _analyze_threshold_effects(self):
        """Analyze how different thresholds would affect predictions"""
        self.threshold_analysis = {}
        
        if not hasattr(self.predict_object, 'processed_predictions'):
            return

        for gene in ['v', 'd', 'j']:
            if f'{gene}_allele' not in self.predict_object.processed_predictions:
                continue
                
            likelihoods = self.predict_object.processed_predictions[f'{gene}_allele']
            
            # Test different threshold values
            thresholds = np.arange(0.1, 1.0, 0.05)
            metrics = []
            
            # Get ground truth for this gene
            if f'{gene}_call' in self.groundtruth_table.columns:
                ground_truth = self.groundtruth_table[f'{gene}_call'].apply(lambda x: set(x.split(',')))
                
                # Get reverse mapping from threshold extractor if available
                reverse_mapping = None
                if hasattr(self.predict_object, 'threshold_extractor_instances') and \
                   gene in self.predict_object.threshold_extractor_instances:
                    extractor = self.predict_object.threshold_extractor_instances[gene]
                    if hasattr(extractor, 'properties_map') and gene.upper() in extractor.properties_map:
                        reverse_mapping = extractor.properties_map[gene.upper()].get('reverse_mapping', {})
                
                for threshold in thresholds:
                    # Apply threshold to get predictions
                    predicted_indices = []
                    avg_calls_per_sequence = []
                    
                    for likelihood_row in likelihoods:
                        # Get indices above threshold
                        above_threshold = np.where(likelihood_row >= threshold)[0]
                        predicted_indices.append(above_threshold)
                        avg_calls_per_sequence.append(len(above_threshold))
                    
                    # Calculate accuracy if we have reverse mapping
                    accuracy = 0
                    if reverse_mapping:
                        correct_predictions = 0
                        for pred_indices, true_alleles in zip(predicted_indices, ground_truth):
                            pred_alleles = set()
                            for idx in pred_indices:
                                if idx in reverse_mapping:
                                    pred_alleles.add(reverse_mapping[idx])
                            
                            if len(pred_alleles.intersection(true_alleles)) > 0:
                                correct_predictions += 1
                        
                        accuracy = correct_predictions / len(predicted_indices) if predicted_indices else 0
                    
                    metrics.append({
                        'threshold': threshold,
                        'accuracy': accuracy,
                        'avg_calls': np.mean(avg_calls_per_sequence),
                        'std_calls': np.std(avg_calls_per_sequence)
                    })
                
                self.threshold_analysis[gene] = pd.DataFrame(metrics)

    def get_panel(self):
        """Generate threshold effects analysis panel"""
        available_genes = list(self.threshold_analysis.keys())
        
        if not available_genes:
            fig = go.Figure()
            fig.add_annotation(
                text="No threshold analysis data available",
                x=0.5, y=0.5,
                xref="paper", yref="paper",
                showarrow=False,
                font=dict(size=16)
            )
            return fig

        # Create subplot grid
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Threshold vs Accuracy',
                'Threshold vs Average Calls',
                'Accuracy vs Average Calls Trade-off',
                'Threshold Effects Summary'
            ),
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}]
            ]
        )

        colors = {
            'v': '#1f77b4',  # Blue
            'd': '#ff7f0e',  # Orange
            'j': '#2ca02c',  # Green
        }

        # Plot 1: Threshold vs Accuracy
        for gene in available_genes:
            data = self.threshold_analysis[gene]
            fig.add_trace(go.Scatter(
                x=data['threshold'],
                y=data['accuracy'],
                mode='lines+markers',
                name=f'{gene.upper()} Accuracy',
                line=dict(color=colors.get(gene, '#gray'), width=3),
                marker=dict(size=6),
                hovertemplate=f"{gene.upper()}<br>Threshold: %{{x:.3f}}<br>Accuracy: %{{y:.3f}}<extra></extra>"
            ), row=1, col=1)

        # Plot 2: Threshold vs Average Calls
        for gene in available_genes:
            data = self.threshold_analysis[gene]
            fig.add_trace(go.Scatter(
                x=data['threshold'],
                y=data['avg_calls'],
                mode='lines+markers',
                name=f'{gene.upper()} Avg Calls',
                line=dict(color=colors.get(gene, '#gray'), width=3, dash='dash'),
                marker=dict(size=6),
                hovertemplate=f"{gene.upper()}<br>Threshold: %{{x:.3f}}<br>Avg Calls: %{{y:.2f}}<extra></extra>",
                showlegend=False
            ), row=1, col=2)

        # Plot 3: Accuracy vs Average Calls Trade-off
        for gene in available_genes:
            data = self.threshold_analysis[gene]
            fig.add_trace(go.Scatter(
                x=data['avg_calls'],
                y=data['accuracy'],
                mode='markers+lines',
                name=f'{gene.upper()} Trade-off',
                line=dict(color=colors.get(gene, '#gray'), width=2),
                marker=dict(
                    size=8,
                    color=data['threshold'],
                    colorscale='Viridis',
                    showscale=gene == available_genes[0],
                    colorbar=dict(title="Threshold", x=0.48, y=0.25) if gene == available_genes[0] else None
                ),
                hovertemplate=f"{gene.upper()}<br>Avg Calls: %{{x:.2f}}<br>Accuracy: %{{y:.3f}}<br>Threshold: %{{marker.color:.3f}}<extra></extra>",
                showlegend=False
            ), row=2, col=1)

        # Plot 4: Summary heatmap
        if available_genes:
            # Create summary data
            summary_thresholds = [0.2, 0.4, 0.6, 0.8]
            summary_data = []
            gene_labels = []
            
            for gene in available_genes:
                data = self.threshold_analysis[gene]
                gene_row = []
                for thresh in summary_thresholds:
                    # Find closest threshold
                    closest_idx = np.argmin(np.abs(data['threshold'] - thresh))
                    gene_row.append(data.iloc[closest_idx]['accuracy'])
                summary_data.append(gene_row)
                gene_labels.append(gene.upper())
            
            fig.add_trace(go.Heatmap(
                z=summary_data,
                x=[f'T={t}' for t in summary_thresholds],
                y=gene_labels,
                colorscale='RdYlBu_r',
                text=[[f'{val:.2f}' for val in row] for row in summary_data],
                texttemplate="%{text}",
                textfont={"size": 12},
                hovertemplate="Gene: %{y}<br>Threshold: %{x}<br>Accuracy: %{z:.2f}<extra></extra>",
                showscale=True,
                colorbar=dict(title="Accuracy", x=1.02, y=0.25)
            ), row=2, col=2)

        # Update layout
        fig.update_layout(
            height=1200,  # Increased height
            width=None,  # Let it be responsive
            title=dict(
                text="Threshold Effects on Prediction Performance",
                x=0.5,
                font=dict(size=20, family="Arial, sans-serif")
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family="Arial, sans-serif", size=12),
            autosize=True
        )

        # Update axes
        fig.update_xaxes(title_text="Threshold", row=1, col=1)
        fig.update_yaxes(title_text="Accuracy", row=1, col=1)
        
        fig.update_xaxes(title_text="Threshold", row=1, col=2)
        fig.update_yaxes(title_text="Average Calls per Sequence", row=1, col=2)
        
        fig.update_xaxes(title_text="Average Calls per Sequence", row=2, col=1)
        fig.update_yaxes(title_text="Accuracy", row=2, col=1)

        return fig
