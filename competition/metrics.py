from sklearn.metrics import f1_score, roc_auc_score

def macro_f1(y_true, y_pred):
    return float(f1_score(y_true, y_pred, average="macro"))

def auroc(y_true, y_pred):
    try:
        return float(roc_auc_score(y_true, y_pred))
    except ValueError:
        return 0.0  # If only one class present
