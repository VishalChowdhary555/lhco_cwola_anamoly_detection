import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score


def predict_scores(model, X, batch_size=4096):
    return model.predict(X, batch_size=batch_size).ravel()


def evaluate_pseudo_labels(y_true, scores):
    auc = roc_auc_score(y_true, scores)
    pred = (scores >= 0.5).astype(int)
    acc = accuracy_score(y_true, pred)
    fpr, tpr, thresholds = roc_curve(y_true, scores)

    return {
        "auc": auc,
        "accuracy": acc,
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
    }


def evaluate_truth_labels(y_true, scores):
    auc = roc_auc_score(y_true, scores)
    fpr, tpr, thresholds = roc_curve(y_true, scores)

    return {
        "auc": auc,
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
    }


def build_enrichment_table(mjj_test, truth_test, scores):
    test_results = pd.DataFrame({
        "mjj": mjj_test,
        "truth_label": truth_test,
        "cwola_score": scores
    })

    rows = []
    for q in [0.00, 0.90, 0.95, 0.99]:
        if q == 0.00:
            selected = test_results.copy()
            name = "All events"
        else:
            thr = np.quantile(scores, q)
            selected = test_results[test_results["cwola_score"] >= thr]
            name = f"Top {int((1-q)*100)}%"

        rows.append({
            "Selection Region": name,
            "Events Selected": len(selected),
            "Signal Fraction": selected["truth_label"].mean()
        })

    return pd.DataFrame(rows), test_results


def build_top_candidates(X_test, feature_cols, mjj_test, truth_test, scores, top_k=20):
    X_test_df = pd.DataFrame(X_test, columns=feature_cols).reset_index(drop=True)

    candidates = X_test_df.copy()
    candidates["mjj"] = mjj_test
    candidates["truth_label"] = truth_test
    candidates["cwola_score"] = scores

    top_candidates = candidates.sort_values("cwola_score", ascending=False).head(top_k).copy()
    top_candidates.insert(0, "rank", np.arange(1, len(top_candidates) + 1))

    return top_candidates
