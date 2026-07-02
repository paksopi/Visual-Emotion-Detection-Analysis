"""Track A closed-set classification metrics: accuracy, macro-F1, confusion matrix."""
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


def classification_metrics(y_true: list[str], y_pred: list[str], labels: list[str]) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "labels": labels,
        "n": len(y_true),
    }
