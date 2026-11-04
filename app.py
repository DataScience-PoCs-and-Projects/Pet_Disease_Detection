from pathlib import Path
import sys
import tempfile
import traceback

import gradio as gr
from PIL import Image


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

# Allow Python to import the project's prediction module.
sys.path.insert(0, str(SRC_DIR))

import predict_proposed as predictor


# ============================================================
# IMAGE ANALYSIS
# ============================================================

def analyze_image(image):
    """
    Analyze an uploaded pet image.

    Pipeline:
        1. Validate input.
        2. Predict modality: eye or skin.
        3. Run the appropriate specialist model.
        4. Generate screening report.
        5. Display prediction confidence and LayerCAM.

    VLM inference is intentionally disabled.
    """

    if image is None:
        raise gr.Error("Please upload an image first.")

    image = image.convert("RGB")

    # Temporary directory for generated LayerCAM images.
    output_dir = Path(tempfile.mkdtemp(prefix="pep_cv_"))

    try:
        # ----------------------------------------------------
        # 1. Automatically classify the image modality
        # ----------------------------------------------------

        modality, routing_confidence = predictor.predict_modality(
            image
        )

        # ----------------------------------------------------
        # 2. Run the appropriate specialist model
        # ----------------------------------------------------

        if modality == "eye":
            result = predictor.run_eye_mcfa(
                image,
                output_dir,
            )

        elif modality == "skin":
            result = predictor.run_skin_cstf(
                image,
                output_dir,
            )

        else:
            raise ValueError(
                f"Unsupported modality returned: {modality}"
            )

        # ----------------------------------------------------
        # 3. Extract predictions
        # ----------------------------------------------------

        prediction = result["prediction"]
        top_k = result["top_k"]

        confidence_scores = {
            item["class"]: float(item["confidence"])
            for item in top_k
        }

        # ----------------------------------------------------
        # 4. Validate LayerCAM output
        # ----------------------------------------------------

        heatmap_path = Path(result["layercam_path"])

        if not heatmap_path.exists():
            raise FileNotFoundError(
                "LayerCAM output was not created: "
                f"{heatmap_path}"
            )

        # ----------------------------------------------------
        # 5. Build the screening report
        # ----------------------------------------------------

        report = f"""
# PEP-CV Screening Results

### Image Analysis

**Detected category:** {modality.title()}

**Routing confidence:** {routing_confidence:.2%}

### Specialist Model

**Model:** {result['model']}

**Predicted class:** {prediction['class']}

**Prediction confidence:** {prediction['confidence']:.2%}

### Interpretation

The model predicted **{prediction['class']}** for the uploaded
{modality} image.

The confidence score represents the model's prediction confidence,
not the probability that the animal has a confirmed disease.

### LayerCAM Explanation

The heatmap highlights image regions that influenced the model's
prediction. These highlighted regions do not establish a diagnosis.

### Important Disclaimer

This is an AI-generated research screening result, not a veterinary
diagnosis. Please consult a qualified veterinarian before making
treatment decisions.
"""

        # ----------------------------------------------------
        # 6. Return results to the Gradio interface
        # ----------------------------------------------------

        return (
            report,
            confidence_scores,
            str(heatmap_path),
        )

    except Exception as exc:
        traceback.print_exc()

        raise gr.Error(
            f"Image analysis failed: {exc}. "
            "Check the VS Code terminal for details."
        ) from exc


# ============================================================
# INTERFACE STYLING
# ============================================================

custom_css = """
.gradio-container {
    max-width: 1150px !important;
    margin: auto;
}

#app-title {
    text-align: center;
}

#disclaimer {
    font-size: 14px;
}
"""


# ============================================================
# GRADIO INTERFACE
# ============================================================

with gr.Blocks(
    title="PEP-CV | Pet Health Screening",
    theme=gr.themes.Soft(),
    css=custom_css,
) as demo:

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    gr.Markdown(
        """
        # PEP-CV: Explainable Pet Health Screening

        Upload an image of a dog's or cat's eye or skin.

        The application automatically selects the appropriate
        specialist model, predicts a category, and generates a
        LayerCAM visualization.

        **Research prototype — not a substitute for veterinary diagnosis.**
        """,
        elem_id="app-title",
    )

    gr.Markdown("---")

    # --------------------------------------------------------
    # Input and screening results
    # --------------------------------------------------------

    with gr.Row():

        # Image input
        with gr.Column(scale=1):

            input_image = gr.Image(
                type="pil",
                label="Upload Pet Image",
            )

            analyze_button = gr.Button(
                "Analyze Image",
                variant="primary",
            )

            gr.Markdown(
                """
                **Supported inputs**

                Upload an image of a dog's or cat's eye or skin.
                The routing model will automatically select the
                corresponding specialist.
                """
            )

        # Prediction outputs
        with gr.Column(scale=1):

            report_output = gr.Markdown(
                label="Screening Report",
            )

            confidence_output = gr.Label(
                label="Top Predictions",
                num_top_classes=3,
            )

    # --------------------------------------------------------
    # LayerCAM visualization
    # --------------------------------------------------------

    gr.Markdown("---")

    gr.Markdown(
        """
        ## Visual Explanation

        LayerCAM highlights image regions that contribute to the
        specialist model's prediction. It should be interpreted as
        a model explanation, not as proof of disease.
        """
    )

    heatmap_output = gr.Image(
        label="LayerCAM Visualization",
        type="filepath",
    )

    # --------------------------------------------------------
    # Event handler
    # --------------------------------------------------------

    analyze_button.click(
        fn=analyze_image,
        inputs=[input_image],
        outputs=[
            report_output,
            confidence_output,
            heatmap_output,
        ],
    )

    # --------------------------------------------------------
    # Footer
    # --------------------------------------------------------

    gr.Markdown(
        """
        ---

        **Disclaimer:** This application is intended for research
        and educational screening. Predictions and heatmaps do not
        replace veterinary examination, diagnostic testing, or
        professional medical advice.
        """,
        elem_id="disclaimer",
    )


# ============================================================
# LAUNCH APPLICATION
# ============================================================

if __name__ == "__main__":
    demo.launch()