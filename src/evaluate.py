import os
import joblib
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error

from data_preprocessing import engineer_features

# Set aesthetic styling
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.titlesize": 15
})

def generate_visualizations(models_dir: str = None, output_dir: str = None, csv_path: str = None):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))

    if models_dir is None:
        models_dir = os.path.join(project_root, "models")
    if output_dir is None:
        output_dir = os.path.join(project_root, "artifacts", "visualizations")
    if csv_path is None:
        csv_path = os.path.join(project_root, "data", "raw", "student_data.csv")

    os.makedirs(output_dir, exist_ok=True)

    # Load model, preprocessor, and metadata
    model = joblib.load(os.path.join(models_dir, "student_performance_model.joblib"))
    preprocessor = joblib.load(os.path.join(models_dir, "preprocessor.joblib"))
    with open(os.path.join(models_dir, "model_metadata.json"), "r") as f:
        metadata = json.load(f)

    feature_names = metadata["feature_names"]
    df = pd.read_csv(csv_path)

    print("[EVAL] Generating Visual Analytics Suite...")

    # -------------------------------------------------------------
    # 1. Correlation Heatmap
    # -------------------------------------------------------------
    numeric_df = df.select_dtypes(include=[np.number])
    plt.figure(figsize=(10, 8), dpi=200)
    corr = numeric_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        vmax=1.0,
        vmin=-0.5,
        center=0,
        annot=True,
        fmt=".2f",
        square=True,
        linewidths=0.7,
        cbar_kws={"shrink": 0.8}
    )
    plt.title("Correlation Matrix of Student Academic Factors", pad=16, fontweight="bold")
    plt.tight_layout()
    corr_path = os.path.join(output_dir, "correlation_heatmap.png")
    plt.savefig(corr_path, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] {corr_path}")

    # -------------------------------------------------------------
    # 2. Actual vs. Predicted Performance
    # -------------------------------------------------------------
    engineered_df = engineer_features(df)
    features_to_drop = [c for c in ["student_id", "performance_tier", "final_score"] if c in engineered_df.columns]
    X_raw = engineered_df.drop(columns=features_to_drop)
    y_true = engineered_df["final_score"].values

    X_trans = preprocessor.transform(X_raw)
    y_pred = model.predict(X_trans)

    r2 = r2_score(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    plt.figure(figsize=(8, 7), dpi=200)
    plt.scatter(y_true, y_pred, alpha=0.45, color="#2563eb", edgecolors="none", s=28)
    
    # 45-degree perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], color="#ef4444", linestyle="--", linewidth=2, label="Ideal Fit (y = x)")

    plt.xlabel("Actual Final Score", fontweight="semibold")
    plt.ylabel("Predicted Final Score", fontweight="semibold")
    plt.title(f"Actual vs. Predicted Final Exam Score\n(R² = {r2:.4f}, RMSE = {rmse:.2f}, MAE = {mae:.2f})", pad=14, fontweight="bold")
    plt.legend(frameon=True, facecolor="white", edgecolor="#e2e8f0")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    act_pred_path = os.path.join(output_dir, "actual_vs_predicted.png")
    plt.savefig(act_pred_path, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] {act_pred_path}")

    # -------------------------------------------------------------
    # 3. Feature Importance (Tree-based model)
    # -------------------------------------------------------------
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        sorted_features = [feature_names[i] for i in indices]
        sorted_importances = [importances[i] for i in indices]

        plt.figure(figsize=(9, 6), dpi=200)
        bars = plt.barh(range(len(sorted_features)), sorted_importances[::-1], color="#3b82f6", edgecolor="#1d4ed8", alpha=0.85)
        plt.yticks(range(len(sorted_features)), sorted_features[::-1])
        plt.xlabel("Relative Importance Score (Gini / MDI)", fontweight="semibold")
        plt.title(f"Key Feature Importances ({metadata['best_model_name']})", pad=14, fontweight="bold")
        
        # Add value labels to bars
        for bar in bars:
            width = bar.get_width()
            plt.text(width + 0.005, bar.get_y() + bar.get_height() / 2, f"{width:.3f}", va="center", fontsize=9, color="#334155")
            
        plt.tight_layout()
        fi_path = os.path.join(output_dir, "feature_importance.png")
        plt.savefig(fi_path, bbox_inches="tight")
        plt.close()
        print(f"  [SAVED] {fi_path}")

    # -------------------------------------------------------------
    # 4. Residuals Distribution
    # -------------------------------------------------------------
    residuals = y_true - y_pred
    plt.figure(figsize=(8, 5), dpi=200)
    sns.histplot(residuals, kde=True, color="#059669", bins=35, edgecolor="white", alpha=0.7)
    plt.axvline(0, color="#dc2626", linestyle="--", linewidth=1.5, label="Zero Error Mean")
    plt.xlabel("Residual Error (Actual - Predicted)", fontweight="semibold")
    plt.ylabel("Frequency", fontweight="semibold")
    plt.title(f"Model Residuals Distribution (Mean: {residuals.mean():.2f}, Std: {residuals.std():.2f})", pad=14, fontweight="bold")
    plt.legend(frameon=True)
    plt.tight_layout()
    res_path = os.path.join(output_dir, "residuals_distribution.png")
    plt.savefig(res_path, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] {res_path}")

    # -------------------------------------------------------------
    # 5. Performance Tiers vs Study Hours & Attendance
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=200)
    tier_order = ["At-Risk", "Pass", "Merit", "Distinction"]
    palette = ["#ef4444", "#f59e0b", "#3b82f6", "#10b981"]

    # Boxplot: Hours studied by tier
    sns.boxplot(
        data=df,
        x="performance_tier",
        y="hours_studied",
        order=tier_order,
        palette=palette,
        ax=axes[0],
        boxprops=dict(alpha=0.85)
    )
    axes[0].set_title("Weekly Study Hours by Performance Tier", fontweight="bold")
    axes[0].set_xlabel("Academic Tier", fontweight="semibold")
    axes[0].set_ylabel("Hours Studied / Week", fontweight="semibold")

    # Boxplot: Attendance by tier
    sns.boxplot(
        data=df,
        x="performance_tier",
        y="attendance_rate",
        order=tier_order,
        palette=palette,
        ax=axes[1],
        boxprops=dict(alpha=0.85)
    )
    axes[1].set_title("Attendance Rate by Performance Tier", fontweight="bold")
    axes[1].set_xlabel("Academic Tier", fontweight="semibold")
    axes[1].set_ylabel("Attendance Rate (%)", fontweight="semibold")

    plt.tight_layout()
    tiers_path = os.path.join(output_dir, "academic_tiers_distribution.png")
    plt.savefig(tiers_path, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] {tiers_path}")

    print("[DONE] All visualization artifacts generated successfully!")

if __name__ == "__main__":
    generate_visualizations()
