import pandas as pd
import sys

def main(pred_path, test_path):
    preds = pd.read_csv(pred_path)
    test = pd.read_csv(test_path)
    
    if "graph_id" not in preds.columns or "label" not in preds.columns:
        raise ValueError("Must have graph_id and label columns")
    if preds["graph_id"].duplicated().any():
        raise ValueError("Duplicate graph_id found")
    if preds["label"].isna().any():
        raise ValueError("NaN predictions found")
    if not preds["label"].isin([0,1]).all():
        raise ValueError("Predictions must be 0 or 1")
    if set(preds["graph_id"]) != set(test["graph_id"].unique()):
        raise ValueError("graph_id mismatch with test set")
    print("VALID SUBMISSION")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
