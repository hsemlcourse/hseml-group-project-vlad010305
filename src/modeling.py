import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def get_predictions_and_scores(model, X):
    """Get binary predictions and probability scores from classifier."""
    y_pred = model.predict(X)

    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X)
    else:
        y_score = y_pred

    return y_pred, y_score


def evaluate_classifier(model, X, y_true) -> dict:
    """Evaluate classifier using metrics for imbalanced classification."""
    y_pred, y_score = get_predictions_and_scores(model, X)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        "PR-AUC": average_precision_score(y_true, y_score),
        "ROC-AUC": roc_auc_score(y_true, y_score),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
    }


def make_results_table(results: list[dict]) -> pd.DataFrame:
    """Convert list of experiment dictionaries to sorted dataframe."""
    return pd.DataFrame(results).sort_values(
        by="PR-AUC",
        ascending=False,
    )