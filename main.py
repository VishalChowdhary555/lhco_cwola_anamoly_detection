from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from src.config import (
    RAW_DIR,
    PROCESSED_DIR,
    FIG_DIR,
    MODEL_DIR,
    TABLE_DIR,
    TARGET_FILENAME,
    HDF_KEY,
    FEATURE_COLS,
    SR_LOW,
    SR_HIGH,
    SB_LEFT_LOW,
    SB_LEFT_HIGH,
    SB_RIGHT_LOW,
    SB_RIGHT_HIGH,
    RANDOM_STATE,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
)
from src.utils import ensure_dir
from src.download_data import download_lhco_dataset
from src.data_loader import load_full_dataframe
from src.features import compute_physics_features
from src.regions import apply_cwola_regions
from src.preprocess import build_cwola_dataframe, split_dataset, scale_features
from src.model import build_cwola_model
from src.train import train_model
from src.evaluate import (
    predict_scores,
    evaluate_pseudo_labels,
    evaluate_truth_labels,
    build_enrichment_table,
    build_top_candidates,
)


def save_training_plots(history_df: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(history_df["loss"], label="Train Loss")
    plt.plot(history_df["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("CWoLa Training Loss")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "cwola_training_loss.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(history_df["auc"], label="Train AUC")
    plt.plot(history_df["val_auc"], label="Validation AUC")
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    plt.title("CWoLa Training AUC")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "cwola_training_auc.png", dpi=200)
    plt.close()


def save_roc_plot(fpr, tpr, auc, title, filename):
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], "--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / filename, dpi=200)
    plt.close()


def main():
    # -------------------------------------------------
    # 1. Ensure folders exist
    # -------------------------------------------------
    for folder in [RAW_DIR, PROCESSED_DIR, FIG_DIR, MODEL_DIR, TABLE_DIR]:
        ensure_dir(folder)

    # -------------------------------------------------
    # 2. Download dataset if needed
    # -------------------------------------------------
    data_path = RAW_DIR / TARGET_FILENAME
    if not data_path.exists():
        print("Downloading LHCO dataset...")
        download_lhco_dataset(data_path)
    else:
        print(f"Dataset already present: {data_path}")

    # -------------------------------------------------
    # 3. Load dataset
    # -------------------------------------------------
    print("Loading dataframe...")
    df = load_full_dataframe(data_path, key=HDF_KEY)
    print("Loaded dataframe shape:", df.shape)

    # -------------------------------------------------
    # 4. Feature engineering
    # -------------------------------------------------
    print("Computing physics features...")
    df_feat = compute_physics_features(df)
    df_feat.to_parquet(PROCESSED_DIR / "lhco_engineered_features.parquet", index=False)

    # -------------------------------------------------
    # 5. Define CWoLa regions
    # -------------------------------------------------
    print("Applying CWoLa region selection...")
    df_sr, df_sb = apply_cwola_regions(
        df_feat,
        sr_low=SR_LOW,
        sr_high=SR_HIGH,
        sb_left_low=SB_LEFT_LOW,
        sb_left_high=SB_LEFT_HIGH,
        sb_right_low=SB_RIGHT_LOW,
        sb_right_high=SB_RIGHT_HIGH,
    )

    print("Signal region events:", len(df_sr))
    print("Sideband events:", len(df_sb))

    # -------------------------------------------------
    # 6. Build dataset
    # -------------------------------------------------
    cwola_df = build_cwola_dataframe(df_sr, df_sb)

    X = cwola_df[FEATURE_COLS].copy()
    y_cwola = cwola_df["cwola_target"].values
    y_truth = cwola_df["label"].values
    mjj_vals = cwola_df["mjj"].values

    (
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        truth_train, truth_val, truth_test,
        mjj_train, mjj_val, mjj_test
    ) = split_dataset(
        X, y_cwola, y_truth, mjj_vals, random_state=RANDOM_STATE
    )

    scaler, X_train_scaled, X_val_scaled, X_test_scaled = scale_features(
        X_train, X_val, X_test
    )

    # -------------------------------------------------
    # 7. Build and train model
    # -------------------------------------------------
    model = build_cwola_model(
        input_dim=X_train_scaled.shape[1],
        learning_rate=LEARNING_RATE
    )

    print("Training model...")
    model, history_df = train_model(
        model,
        X_train_scaled,
        y_train,
        X_val_scaled,
        y_val,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS
    )

    history_df.to_csv(TABLE_DIR / "training_history.csv", index=False)
    model.save(MODEL_DIR / "cwola_classifier.keras")

    save_training_plots(history_df)

    # -------------------------------------------------
    # 8. Predictions
    # -------------------------------------------------
    print("Generating predictions...")
    test_scores = predict_scores(model, X_test_scaled)

    # -------------------------------------------------
    # 9. Evaluation
    # -------------------------------------------------
    pseudo_eval = evaluate_pseudo_labels(y_test, test_scores)
    truth_eval = evaluate_truth_labels(truth_test, test_scores)

    pd.DataFrame({
        "fpr": pseudo_eval["fpr"],
        "tpr": pseudo_eval["tpr"]
    }).to_csv(TABLE_DIR / "pseudo_label_roc.csv", index=False)

    pd.DataFrame({
        "fpr": truth_eval["fpr"],
        "tpr": truth_eval["tpr"]
    }).to_csv(TABLE_DIR / "truth_label_roc.csv", index=False)

    save_roc_plot(
        pseudo_eval["fpr"],
        pseudo_eval["tpr"],
        pseudo_eval["auc"],
        "Pseudo-label ROC Curve",
        "pseudo_label_roc.png"
    )

    save_roc_plot(
        truth_eval["fpr"],
        truth_eval["tpr"],
        truth_eval["auc"],
        "Truth-label ROC Curve",
        "truth_label_roc.png"
    )

    # -------------------------------------------------
    # 10. Score distributions
    # -------------------------------------------------
    score_df = pd.DataFrame({
        "cwola_score": test_scores,
        "truth_label": truth_test
    })
    score_df.to_csv(TABLE_DIR / "score_distribution_data.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.hist(test_scores[truth_test == 0], bins=80, alpha=0.6, density=True, label="Background")
    plt.hist(test_scores[truth_test == 1], bins=80, alpha=0.6, density=True, label="Signal")
    plt.xlabel("CWoLa Score")
    plt.ylabel("Density")
    plt.title("CWoLa Score Distribution")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "cwola_score_distribution.png", dpi=200)
    plt.close()

    # -------------------------------------------------
    # 11. Score vs mjj
    # -------------------------------------------------
    score_vs_mjj_df = pd.DataFrame({
        "mjj": mjj_test,
        "cwola_score": test_scores,
        "truth_label": truth_test
    })
    score_vs_mjj_df.to_csv(TABLE_DIR / "score_vs_mjj_data.csv", index=False)

    plt.figure(figsize=(8, 5))
    bg_mask = truth_test == 0
    sig_mask = truth_test == 1
    plt.scatter(mjj_test[bg_mask], test_scores[bg_mask], s=4, alpha=0.12, label="Background")
    plt.scatter(mjj_test[sig_mask], test_scores[sig_mask], s=6, alpha=0.25, label="Signal")
    plt.xlabel("Dijet Invariant Mass $m_{jj}$ [GeV]")
    plt.ylabel("CWoLa Score")
    plt.title("CWoLa Score vs Dijet Mass")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "cwola_score_vs_mjj.png", dpi=200)
    plt.close()

    # -------------------------------------------------
    # 12. Enrichment table and post-selection mass
    # -------------------------------------------------
    enrichment_df, test_results = build_enrichment_table(mjj_test, truth_test, test_scores)
    enrichment_df.to_csv(TABLE_DIR / "signal_enrichment_table.csv", index=False)

    thr_95 = test_results["cwola_score"].quantile(0.95)
    selected_95 = test_results[test_results["cwola_score"] >= thr_95]
    selected_95.to_csv(TABLE_DIR / "postselection_mjj_data.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.hist(test_results["mjj"], bins=90, alpha=0.35, density=True, label="All test events")
    plt.hist(selected_95["mjj"], bins=60, alpha=0.60, density=True, label="Top 5% score selection")
    plt.xlabel("Dijet Invariant Mass $m_{jj}$ [GeV]")
    plt.ylabel("Density")
    plt.title("Mass Distribution After High-score Selection")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "postselection_mjj.png", dpi=200)
    plt.close()

    # -------------------------------------------------
    # 13. Top candidate events
    # -------------------------------------------------
    top_candidates = build_top_candidates(
        X_test=X_test,
        feature_cols=FEATURE_COLS,
        mjj_test=mjj_test,
        truth_test=truth_test,
        scores=test_scores,
        top_k=20
    )
    top_candidates.to_csv(TABLE_DIR / "top_candidate_events.csv", index=False)

    # -------------------------------------------------
    # 14. Summary table
    # -------------------------------------------------
    best_epoch = int(history_df["val_auc"].idxmax() + 1)
    summary_table = pd.DataFrame({
        "Metric": [
            "Best Epoch",
            "Training Loss",
            "Validation Loss",
            "Training AUC",
            "Validation AUC",
            "Pseudo-label Test AUC",
            "Pseudo-label Test Accuracy",
            "Truth-label Test AUC"
        ],
        "Value": [
            best_epoch,
            float(history_df.loc[best_epoch - 1, "loss"]),
            float(history_df.loc[best_epoch - 1, "val_loss"]),
            float(history_df.loc[best_epoch - 1, "auc"]),
            float(history_df.loc[best_epoch - 1, "val_auc"]),
            float(pseudo_eval["auc"]),
            float(pseudo_eval["accuracy"]),
            float(truth_eval["auc"])
        ]
    })
    summary_table.to_csv(TABLE_DIR / "results_summary_table.csv", index=False)

    print("\nPipeline complete.")
    print("Saved outputs to:")
    print(f"  Figures: {FIG_DIR}")
    print(f"  Tables : {TABLE_DIR}")
    print(f"  Models : {MODEL_DIR}")


if __name__ == "__main__":
    main()
