from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "model_comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# ACTUAL TEST RESULTS
# ============================================================

# Values are stored as percentages.
# These are the test-set results from your trained models.

EYE_MODELS = {
    "EfficientNet-B0": {
        "Accuracy": 75.00,
        "Precision": 81.30,
        "Recall": 81.27,
        "F1 Score": 81.23,
    },

    "DenseNet-121": {
        "Accuracy": 74.73,
        "Precision": 79.65,
        "Recall": 80.11,
        "F1 Score": 79.74,
    },

    "CSF-Net": {
        "Accuracy": 73.20,
        "Precision": 79.33,
        "Recall": 78.01,
        "F1 Score": 78.41,
    },

    "MCFA-Net V2": {
        "Accuracy": 74.91,
        "Precision": 80.79,
        "Recall": 81.56,
        "F1 Score": 80.69,
    },
}


SKIN_MODELS = {
    "EfficientNet-B0": {
        "Accuracy": 84.66,
        "Precision": 74.21,
        "Recall": 77.91,
        "F1 Score": 73.63,
    },

    "EfficientNetV2-S": {
        "Accuracy": 85.92,
        "Precision": 72.95,
        "Recall": 73.98,
        "F1 Score": 71.53,
    },

    "ConvNeXt-Tiny": {
        "Accuracy": 85.74,
        "Precision": 72.63,
        "Recall": 73.39,
        "F1 Score": 71.94,
    },

    "CSTF-Net": {
        "Accuracy": 85.20,
        "Precision": 75.12,
        "Recall": 78.53,
        "F1 Score": 73.84,
    },
}


# ============================================================
# GRAPH SETTINGS
# ============================================================

METRICS = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score",
]

Y_LABEL = "Scores"
X_LABEL = "Model Variants"

FIG_SIZE = (10, 5.2)

# Total width occupied by each model group
GROUP_WIDTH = 0.82

# Your PPT-style graph
SHOW_TITLE = False
SHOW_VALUE_LABELS = False

# Keep the y-axis exactly 0–1 in the generated graph,
# because the values are converted to decimals below.
Y_MIN = 0.0
Y_MAX = 1.0


# ============================================================
# PLOT FUNCTION
# ============================================================

def plot_model_comparison(
    models,
    output_path,
    caption_name,
):
    model_names = list(models.keys())

    n_models = len(model_names)
    n_metrics = len(METRICS)

    x = np.arange(n_models)

    width = GROUP_WIDTH / n_metrics

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    # --------------------------------------------------------
    # Draw grouped bars
    # --------------------------------------------------------

    for metric_index, metric in enumerate(METRICS):

        values = [
            models[model][metric] / 100.0
            for model in model_names
        ]

        offset = (
            metric_index
            - (n_metrics - 1) / 2
        ) * width

        bars = ax.bar(
            x + offset,
            values,
            width,
            label=metric,
        )

        # ----------------------------------------------------
        # Optional value labels
        # ----------------------------------------------------

        if SHOW_VALUE_LABELS:

            for bar in bars:

                height = bar.get_height()

                ax.annotate(
                    f"{height:.2f}",
                    xy=(
                        bar.get_x()
                        + bar.get_width() / 2,
                        height,
                    ),
                    xytext=(0, 2),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

    # ========================================================
    # X AXIS
    # ========================================================

    ax.set_xticks(x)

    ax.set_xticklabels(
        model_names,
        fontsize=9,
    )

    ax.set_xlabel(
        X_LABEL,
        fontsize=9,
    )

    # ========================================================
    # Y AXIS
    # ========================================================

    ax.set_ylabel(
        Y_LABEL,
        fontsize=9,
    )

    ax.set_ylim(
        Y_MIN,
        Y_MAX,
    )

    ax.set_yticks(
        np.arange(
            0,
            1.01,
            0.2,
        )
    )

    ax.tick_params(
        axis="y",
        labelsize=8,
    )

    # ========================================================
    # GRID
    # ========================================================

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.3,
    )

    ax.set_axisbelow(True)

    # ========================================================
    # LEGEND
    # ========================================================

    ax.legend(
        loc="upper left",
        fontsize=9,
    )

    # ========================================================
    # OPTIONAL TITLE
    # ========================================================

    if SHOW_TITLE:

        ax.set_title(
            caption_name,
            fontsize=13,
            fontweight="bold",
        )

    # ========================================================
    # SAVE
    # ========================================================

    fig.tight_layout()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Saved: {output_path}"
    )


# ============================================================
# PRINT RESULTS
# ============================================================

def print_results(title, models):

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)

    print(
        f"{'Model':<22}"
        f"{'Accuracy':>12}"
        f"{'Precision':>12}"
        f"{'Recall':>12}"
        f"{'F1':>12}"
    )

    print("-" * 70)

    for model, values in models.items():

        print(
            f"{model:<22}"
            f"{values['Accuracy']:>11.2f}%"
            f"{values['Precision']:>11.2f}%"
            f"{values['Recall']:>11.2f}%"
            f"{values['F1 Score']:>11.2f}%"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("PET HEALTH CV - MODEL COMPARISON GRAPH GENERATOR")
    print("=" * 70)

    # --------------------------------------------------------
    # Eye models
    # --------------------------------------------------------

    print_results(
        "EYE MODEL RESULTS",
        EYE_MODELS,
    )

    eye_output = (
        OUTPUT_DIR
        / "eye_model_comparison.png"
    )

    plot_model_comparison(
        models=EYE_MODELS,
        output_path=eye_output,
        caption_name="Eye Model Comparison",
    )

    # --------------------------------------------------------
    # Skin models
    # --------------------------------------------------------

    print_results(
        "SKIN MODEL RESULTS",
        SKIN_MODELS,
    )

    skin_output = (
        OUTPUT_DIR
        / "skin_model_comparison.png"
    )

    plot_model_comparison(
        models=SKIN_MODELS,
        output_path=skin_output,
        caption_name="Skin Model Comparison",
    )

    # --------------------------------------------------------
    # Done
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)

    print()
    print("Eye graph:")
    print(eye_output)

    print()
    print("Skin graph:")
    print(skin_output)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()