from dataclasses import dataclass
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix

@dataclass
class Metrics:
    auc: float
    f1: float
    sensitivity: float
    specificity: float

def compute_metrics(y_true, y_prob, thr: float = 0.5) -> Metrics:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= thr).astype(int)

    # AUC needs both classes
    auc = float("nan")
    if len(np.unique(y_true)) == 2:
        auc = float(roc_auc_score(y_true, y_prob))

    f1 = float(f1_score(y_true, y_pred))

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = float(tp / (tp + fn + 1e-12))
    specificity = float(tn / (tn + fp + 1e-12))

    return Metrics(auc=auc, f1=f1, sensitivity=sensitivity, specificity=specificity)