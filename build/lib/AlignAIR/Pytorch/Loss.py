import torch
import torch.nn.functional as F

class AlignAIRHeavyChainLoss:
    def __init__(self, model):
        """
        Initializes the hierarchical loss class.

        Args:
        - model: The HeavyChainAlignAIRR_PT model instance, containing the `RegularizedConstrainedLogVar` parameters.
        """
        self.model = model
        self.loss_components = {}
        self.regularization_loss = 0.0

    def compute_task_loss(self, y_true, y_pred, task, loss_fn, log_var_name):
        """
        Computes the weighted loss for a single task, including regularization.

        Args:
        - y_true: Ground truth for the task.
        - y_pred: Model prediction for the task.
        - task: Task name (used for tracking losses).
        - loss_fn: Loss function to compute the base loss.
        - log_var_name: Name of the `RegularizedConstrainedLogVar` instance for the task.

        Returns:
        - Weighted loss for the task.
        """
        base_loss = loss_fn(y_pred, y_true, reduction='mean')
        precision, reg_loss = getattr(self.model, log_var_name).forward()
        weighted_loss = precision * base_loss
        self.loss_components[f"{task}_loss"] = weighted_loss
        self.regularization_loss += reg_loss
        return weighted_loss

    def compute_segmentation_loss(self, y_true, y_pred):
        """
        Computes the segmentation loss across all tasks.
        """
        segmentation_tasks = ['v_start', 'v_end', 'd_start', 'd_end', 'j_start', 'j_end']
        for task in segmentation_tasks:
            self.compute_task_loss(
                y_true[task], y_pred[task],
                task=task,
                loss_fn=F.l1_loss,
                log_var_name=f"log_var_{task}"
            )

    def compute_classification_loss(self, y_true, y_pred):
        """
        Computes the classification loss across all tasks.
        """
        classification_tasks = ['v_allele', 'd_allele', 'j_allele']
        for task in classification_tasks:
            self.compute_task_loss(
                y_true[task], y_pred[task],
                task=task,
                loss_fn=F.binary_cross_entropy,
                log_var_name=f"log_var_{task}_classification"
            )

    def compute_additional_losses(self, y_true, y_pred):
        """
        Computes additional losses for mutation rate, indel count, and productivity.
        """
        # Mutation Rate Loss
        self.compute_task_loss(
            y_true['mutation_rate'], y_pred['mutation_rate'],
            task="mutation_rate",
            loss_fn=F.l1_loss,
            log_var_name="log_var_mutation"
        )

        # Indel Count Loss
        self.compute_task_loss(
            y_true['indel_count'], y_pred['indel_count'],
            task="indel_count",
            loss_fn=F.l1_loss,
            log_var_name="log_var_indel"
        )

        # Productivity Loss
        self.compute_task_loss(
            y_true['productive'], y_pred['productive'],
            task="productive",
            loss_fn=F.binary_cross_entropy,
            log_var_name="log_var_productivity"
        )

    def __call__(self, y_true, y_pred):
        """
        Computes the total hierarchical loss.

        Args:
        - y_true: Ground truth dictionary.
        - y_pred: Predictions dictionary.

        Returns:
        - total_loss: Weighted sum of all losses, including regularization.
        - loss_components: Dictionary with individual losses for monitoring.
        """
        # Reset losses
        self.loss_components = {}
        self.regularization_loss = 0.0

        # Compute segmentation, classification, and additional losses
        self.compute_segmentation_loss(y_true, y_pred)
        self.compute_classification_loss(y_true, y_pred)
        self.compute_additional_losses(y_true, y_pred)

        # Total loss includes all task losses and the regularization loss
        total_loss = sum(self.loss_components.values()) + self.regularization_loss
        return total_loss, self.loss_components
