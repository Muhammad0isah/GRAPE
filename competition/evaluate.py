import pandas as pd
import sys
from metrics import macro_f1, auroc

def main(pred_path, label_path):
    preds = pd.read_csv(pred_path).sort_values("graph_id")
    labels = pd.read_csv(label_path).sort_values("graph_id")
    merged = labels.merge(preds, on="graph_id", how="inner", suffixes=('_true','_pred'))
    if len(merged) != len(labels):
        raise ValueError("ID mismatch")
    f1 = macro_f1(merged["label_true"], merged["label_pred"])
    auc = auroc(merged["label_true"], merged["label_pred"])
    print(f"MACRO_F1={f1:.6f}")
    print(f"AUROC={auc:.6f}")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
