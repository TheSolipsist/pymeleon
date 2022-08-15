"""
Classification metrics module
"""
import torch
from torchmetrics import AUROC
import warnings

class Metrics:
    """
    Metrics class for classification
    """
    def __init__(self, loss_func=torch.nn.BCELoss(), auc_class=AUROC, num_classes: int = 1) -> None:
        # AUROC initialization throws a warning that is irrelevant to the user, so it is suppressed
        warnings.filterwarnings("ignore", category=UserWarning, message="Metric `AUROC` will save all targets and predictions in buffer. For large datasets this may lead to large memory footprint.")
        # _auc_func = auc_class(num_classes=num_classes)
        self.metric_funcs = {
            "loss": loss_func,
            # "AUC": lambda y_hat, y: _auc_func(y_hat, y.to(torch.uint8)),
        }


def trapezoid_rule(points_y: torch.Tensor, points_x: torch.Tensor) -> torch.Tensor:
    """
    Returns the area defined by a set of sorted points that are connected by a line

    Args:
        points_y (torch.Tensor): Tensor containing the y coordinates of the points
        points_x (torch.Tensor): Tensor containing the x coordinates of the points

    Returns:
        torch.Tensor: Tensor of shape [1] containing the area defined by the lines
    """
    area = torch.zeros(size=(1,), device=points_y.device)
    for i in range(1, points_y.shape[0]):
        area += (points_y[i] + points_y[i - 1]) * (points_x[i] - points_x[i - 1]) / 2
    return area
    
def auc_roc(y: torch.Tensor, y_prob: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Returns the AUC based on a Tensor of predictions and its corresponding labels

    Args:
        labels (torch.Tensor): The actual labels of the source data
        y_prob (torch.Tensor): The predicted probability that the source data belongs to the positive class
    """
    y_prob, indices = y_prob.sort(dim=0, descending=True)
    y = y[indices.squeeze()]
    tpr = torch.empty(size=(y.shape[0], 1), dtype=torch.float32, device=device)
    fpr = torch.empty_like(tpr)
    tpr[0], fpr[0] = 0, 0
    matr = torch.ones(size=(y.shape[0], 1), device=device).T
    for i in range(y.shape[0]):
        matr[0][i] = 0
        torch.mm(matr, y)
    return trapezoid_rule(tpr, fpr)
