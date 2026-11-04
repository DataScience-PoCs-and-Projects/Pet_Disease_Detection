const API_BASE_URL = "https://pet-disease-cv.onrender.com";

const imageInput = document.getElementById("imageInput");
const analyzeBtn = document.getElementById("analyzeBtn");
const statusText = document.getElementById("status");
const previewWrap = document.getElementById("previewWrap");
const preview = document.getElementById("preview");
const results = document.getElementById("results");

const predictionTitle = document.getElementById("predictionTitle");
const modalityBadge = document.getElementById("modalityBadge");
const modelName = document.getElementById("modelName");
const confidenceText = document.getElementById("confidenceText");
const topK = document.getElementById("topK");
const heatmap = document.getElementById("heatmap");
const notice = document.getElementById("notice");

const apiUrl = document.getElementById("apiUrl");
if (apiUrl) {
    apiUrl.value = API_BASE_URL;
    apiUrl.hidden = true;
}

let selectedFile = null;

imageInput.addEventListener("change", () => {
    selectedFile = imageInput.files[0] || null;
    results.classList.add("hidden");
    statusText.textContent = "";

    if (!selectedFile) {
        previewWrap.classList.add("hidden");
        return;
    }

    preview.src = URL.createObjectURL(selectedFile);
    previewWrap.classList.remove("hidden");
});

analyzeBtn.addEventListener("click", async () => {
    if (!selectedFile) {
        statusText.textContent = "Please choose an image first.";
        return;
    }

    analyzeBtn.disabled = true;
    statusText.textContent = "Analyzing image. Please wait...";
    results.classList.add("hidden");

    try {
        const formData = new FormData();
        formData.append("file", selectedFile);

        const response = await fetch(`${API_BASE_URL}/analyze`, {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || `HTTP ${response.status}`);
        }

        const prediction = data.prediction || {};

        predictionTitle.textContent =
            prediction.class || prediction.label || "Prediction unavailable";

        modalityBadge.textContent = data.modality || "Unknown";
        modelName.textContent = `Model: ${data.model || "Not specified"}`;

        const confidence = Number(prediction.confidence);
        confidenceText.textContent = Number.isFinite(confidence)
            ? `Confidence: ${(confidence * 100).toFixed(2)}%`
            : "Confidence unavailable.";

        topK.replaceChildren();

        (Array.isArray(data.top_k) ? data.top_k : []).forEach(item => {
            const row = document.createElement("p");
            const score = Number(item.confidence);

            row.textContent =
                `${item.class || item.label || "Unknown"} — ` +
                (Number.isFinite(score)
                    ? `${(score * 100).toFixed(2)}%`
                    : "N/A");

            topK.appendChild(row);
        });

        if (data.layercam &&
            data.layercam.startsWith("data:image/")) {
            heatmap.src = data.layercam;
            heatmap.classList.remove("hidden");
        } else {
            heatmap.removeAttribute("src");
            heatmap.classList.add("hidden");
        }

        notice.textContent =
            "Research screening only. This is not a veterinary diagnosis.";

        results.classList.remove("hidden");
        statusText.textContent = "Analysis completed.";

    } catch (error) {
        statusText.textContent =
            `Analysis failed: ${error.message}.`;
    } finally {
        analyzeBtn.disabled = false;
    }
});