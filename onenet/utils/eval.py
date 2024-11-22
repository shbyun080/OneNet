from typing import Any

import torch
from torch import Tensor
from torchmetrics.functional import fbeta_score
from torchmetrics.functional.classification import accuracy
from torchmetrics.functional.segmentation import generalized_dice_score, mean_iou


class Evaluator:
    def __init__(self, preds: Tensor, targets: Tensor, num_classes: int = 2, iou_threshold: float = 0.5) -> None:
        """
        Initialize the Evaluator with ground truth and prediction tensors, dynamically assigning the device.
        This evaluator uses torchmetrics for metric calculations and supports both binary and multiclass data.

        Attributes:
            preds (torch.Tensor): Predicted mask with class labels (integers for multiclass or binary).
            targets (torch.Tensor): Ground truth mask with class labels (integers for multiclass or binary).
            num_classes (int): Number of classes (2 for binary classification).
            iou_threshold (float): Intersection over Union threshold for positive detection.
        """
        assert preds.shape[0] == targets.shape[0], "Shape of ground truth and prediction must match"

        # Dynamically select device based on GPU availability
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Assign class attributes
        self.num_classes = num_classes
        self.task = "multiclass"
        self.input_format = "one-hot"
        self.iou_threshold = iou_threshold

        # Convert one-hot encoded predictions to class labels
        self.y_pred = preds.argmax(dim=1).to(self.device)
        self.y_true = targets.to(self.device)

    def mean_iou(self, **kwargs) -> Tensor:
        """
        Calculate the Mean Intersection over Union (mIOU) using torchmetrics.

        Returns:
            torch.Tensor: The mean Intersection over Union (mIOU) score.
        """
        return mean_iou(
            self.y_pred, self.y_true, num_classes=self.num_classes, input_format=self.input_format, **kwargs
        )

    def f1_score(self, beta: float = 1.0, **kwargs) -> Tensor:
        """
        Calculate the F-score using torchmetrics, the weighted harmonic mean of precision and recall.

        Args:
            beta (float): Weighting factor for precision and recall; beta < 1 prioritizes precision,
            while beta > 1 prioritizes recall.

        Returns:
            torch.Tensor: The F-score with the specified beta.
        """
        return fbeta_score(
            self.y_pred, self.y_true, task=self.task, num_classes=self.num_classes, beta=beta, **kwargs
        ).item()

    def accuracy(self, **kwargs) -> Tensor:
        """
        Calculate the accuracy using torchmetrics, the ratio of correct predictions to total predictions.

        Returns:
            torch.Tensor: The accuracy score.
        """
        return accuracy(self.y_pred, self.y_true, task=self.task, num_classes=self.num_classes, **kwargs).item()

    def dice_similarity(self, **kwargs) -> Tensor:
        """
        Calculate the Dice Similarity Coefficient (DSC) using torchmetrics, a measure of overlap between two samples.

        Returns:
            torch.Tensor: The Dice Similarity Coefficient.
        """
        return generalized_dice_score(
            self.y_pred, self.y_true, num_classes=self.num_classes, input_format=self.input_format, **kwargs
        )

    def map_dataset(self, iou_threshold: float = 0.1) -> float:
        """
        Calculate the mean average precision at a given IoU threshold for the entire dataset.

        Args:
            iou_threshold (float): Intersection over Union threshold for positive detection.

        Returns:
            float: The mean average precision at the specified IoU threshold.
        """
        average_precision = []

        # Loop over each class
        for cls in range(self.num_classes):
            # Get predictions and ground truths for this class
            y_pred_class = (self.y_pred == cls).float()
            y_true_class = (self.y_true == cls).float()

            # Calculate intersection and union
            intersection = (y_pred_class * y_true_class).sum()
            union = y_pred_class.sum() + y_true_class.sum() - intersection

            # Calculate IoU
            iou = intersection / union if union > 0 else torch.tensor(0.0, device=self.device)

            # Calculate true positives, false positives, and false negatives
            true_positives = (iou >= iou_threshold).float() * y_true_class.sum()
            false_positives = y_pred_class.sum() - true_positives
            # false_negatives = y_true_class.sum() - true_positives

            # Calculate precision and recall
            precision = (
                true_positives / (true_positives + false_positives)
                if true_positives + false_positives > 0
                else torch.tensor(0.0, device=self.device)
            )
            # recall = (
            #     true_positives / (true_positives + false_negatives)
            #     if true_positives + false_negatives > 0
            #     else torch.tensor(0.0, device=self.device)
            # )

            # Add to average precision list
            average_precision.append(precision)

        # Mean of the average precisions for each class
        mean_avg_precision = sum(average_precision) / len(average_precision)
        return mean_avg_precision.item()

    def map_per_image(self, iou_threshold: float = 0.5) -> float:
        """
        Calculate the mean average precision at a given IoU threshold, on a per-image basis.

        Args:
            iou_threshold (float): Intersection over Union threshold for positive detection.

        Returns:
            float: The mean of per-image average precision at the specified IoU threshold.
        """
        per_image_ap = []

        # Iterate through each image in the batch
        for i in range(self.y_pred.shape[0]):
            average_precision = []

            # Calculate mAP for each class for this specific image
            for cls in range(self.num_classes):
                y_pred_class = (self.y_pred[i] == cls).float()
                y_true_class = (self.y_true[i] == cls).float()

                intersection = (y_pred_class * y_true_class).sum()
                union = y_pred_class.sum() + y_true_class.sum() - intersection

                iou = intersection / union if union > 0 else torch.tensor(0.0, device=self.device)

                true_positives = (iou >= iou_threshold).float() * y_true_class.sum()
                false_positives = y_pred_class.sum() - true_positives
                # false_negatives = y_true_class.sum() - true_positives

                precision = (
                    true_positives / (true_positives + false_positives)
                    if true_positives + false_positives > 0
                    else torch.tensor(0.0, device=self.device)
                )

                average_precision.append(precision)

            # Append the mean precision for this image
            per_image_ap.append(sum(average_precision) / len(average_precision))

        # Mean of the average precisions across all images
        mean_avg_precision = sum(per_image_ap) / len(per_image_ap)
        return mean_avg_precision.item()

    def evaluate_all(self) -> dict[str, Any]:
        """
        Calculate all metrics (mIOU, F-score, accuracy, and Dice Similarity) and return as a dictionary.

        Returns:
            dict[str, float]: A dictionary containing all evaluation metrics with their respective scores.
        """
        return {
            "mean_IoU": self.mean_iou(),
            "f1_score": self.f1_score(),
            "accuracy": self.accuracy(),
            "dice_similarity": self.dice_similarity(),
            "mAP_dataset": self.map_dataset(iou_threshold=self.iou_threshold),
            "mAP_per_image": self.map_per_image(iou_threshold=self.iou_threshold),
        }


# Example usage
if __name__ == "__main__":
    from pprint import pprint

    from torch import randint

    torch.random.manual_seed(756)

    # Example binary ground truth and prediction tensors
    N = 5  # Number of samples
    C = 3  # Number of classes
    H, W = 64, 64  # Spatial dimensions: H*W resolution

    # Predicted one-hot encoded tensor (N, C, H, W)
    preds = randint(0, 2, (N, C, H, W))
    # Ground truth label tensor (N, H, W) with values from 0 to C-1
    target = randint(0, C, (N, H, W))

    evaluator = Evaluator(preds, target, num_classes=C, iou_threshold=0.1)
    results = evaluator.evaluate_all()

    pprint(results)
