from sklearn.metrics import f1_score

def macro_f1(y_true, y_pred):
    return float(f1_score(y_true, y_pred, average="macro"))
